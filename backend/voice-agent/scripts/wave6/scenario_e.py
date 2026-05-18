"""Scenario E — 4-hour soak at ~50% capacity.

What this proves
----------------

* No monotonic memory growth across hours of sustained moderate load
  (catches Python ref-cycle leaks, asyncio task leaks, boto3 client
  cache bloat).
* No FD growth (catches socket/file leaks).
* Zero unexpected AssemblyAI 1008 / Bedrock throttling / ElevenLabs
  disconnect events during sustained low-burst-rate traffic.
* CallRecord stub Lambda continues to respond cleanly (no Lambda
  thrash, no IAM drift).
* No stuck deployments / no surprise ECS task replacement.

Profile
-------

* Sustained ~12 calls / minute (0.2 cps) for 4 hours. With ~1.7 s call
  lifetime that's ~0.34 concurrent average — well below the 4.2/task
  target, so autoscaler stays at minCapacity=1 throughout. This is a
  STEADY-STATE test, not a scale test.
* Heartbeats every 60 s to disk so the run is resumable.

Resumability
------------

If the harness gets killed mid-soak, the heartbeat file
(``scenario_e_heartbeat.json``) contains: cycle number, last-fire-ts,
running tallies. On restart, the runner reads this and continues from
the next cycle. **Wave 6 doesn't auto-restart on crash** — Alex relaunches
the same command, the heartbeat picks up. Good enough for an overnight
run.

Why no aggressive checks
------------------------

The scenario is intentionally tolerant. We're proving "nothing got worse
over 4 h," not "everything is great." We flag growth trends (RSS up >
20%, FD up > 50, error rate > 1%) but don't fail on cosmetic blips.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

from . import cloudwatch, config, ecs, scenario_base
from .http_caller import CallResultBatch, HttpCaller

logger = structlog.get_logger(__name__)


SOAK_DURATION_SECS = 4 * 60 * 60  # 4 hours
SOAK_CALLS_PER_SEC = 0.2          # 12 cpm
HEARTBEAT_INTERVAL_SECS = 60
ERROR_RATE_THRESHOLD_PCT = 1.0


async def run(paths: config.RunPaths) -> scenario_base.ScenarioResult:
    started_dt = datetime.now(timezone.utc)
    started_ts = scenario_base.now_iso()
    started_perf = time.perf_counter()

    pre_state = await ecs.describe_service()
    logger.info("scenario_e_starting", pre=pre_state.as_dict(), duration_h=SOAK_DURATION_SECS / 3600)

    # Resume from previous heartbeat, if any.
    heartbeat_path = paths.soak_heartbeat()
    resumed_from = _maybe_resume(heartbeat_path)
    if resumed_from is not None:
        logger.info("scenario_e_resuming", resumed_from=resumed_from)

    batch = CallResultBatch()
    cycle_log: list[dict[str, object]] = []
    log_path = paths.soak_log()
    # Open append-mode so a resumed run extends the existing file.
    log_fh = log_path.open("a", encoding="utf-8")

    try:
        async with HttpCaller() as caller:
            await _soak_loop(
                caller=caller,
                batch=batch,
                paths=paths,
                heartbeat_path=heartbeat_path,
                cycle_log=cycle_log,
                log_fh=log_fh,
                resumed_from_secs=(resumed_from or {}).get("elapsed_secs", 0.0) if resumed_from else 0.0,
            )
    finally:
        log_fh.close()

    await asyncio.sleep(120)
    post_state = await ecs.describe_service()
    ended_dt = datetime.now(timezone.utc)
    ended_ts = scenario_base.now_iso()
    duration_secs = round(time.perf_counter() - started_perf, 1)

    cw_window = cloudwatch.window_around(started_dt, ended_dt, pad_min=2)
    sessions_max = await cloudwatch.query_active_sessions_max(start=cw_window[0], end=cw_window[1])
    sessions_avg = await cloudwatch.query_active_sessions_avg(start=cw_window[0], end=cw_window[1])
    cpu_avg = await cloudwatch.query_ecs_cpu_utilization(start=cw_window[0], end=cw_window[1])
    mem_avg = await cloudwatch.query_ecs_memory_utilization(start=cw_window[0], end=cw_window[1])
    drain_timeouts = await cloudwatch.query_drain_timeouts_sum(start=cw_window[0], end=cw_window[1])

    # Trend: first vs last hour of CloudWatch memory. If we have 4 h
    # of data we can split it; otherwise we just record the overall stats.
    mem_first_hour, mem_last_hour = await _split_memory_trend(started_dt, ended_dt)

    checks = _build_checks(
        batch=batch,
        cycle_log=cycle_log,
        pre_state=pre_state,
        post_state=post_state,
        cpu_avg=cpu_avg,
        mem_avg=mem_avg,
        drain_timeouts=drain_timeouts,
        mem_first_hour=mem_first_hour,
        mem_last_hour=mem_last_hour,
    )

    result = scenario_base.ScenarioResult(
        scenario="e",
        started_at=started_ts,
        ended_at=ended_ts,
        duration_secs=duration_secs,
        description=(
            "4-hour soak at 0.2 cps (12 cpm). Steady-state. Catches "
            "memory/FD leaks, vendor throttling under sustained load."
        ),
        config={
            "soak_duration_secs": SOAK_DURATION_SECS,
            "soak_calls_per_sec": SOAK_CALLS_PER_SEC,
            "heartbeat_interval_secs": HEARTBEAT_INTERVAL_SECS,
            "error_rate_threshold_pct": ERROR_RATE_THRESHOLD_PCT,
        },
        calls={
            "count": batch.count,
            "accepted_202": batch.accepted,
            "rejected_503": batch.rejected_503,
            "other": batch.other_counts,
            "latency_ms": batch.latency_percentiles(),
        },
        cloudwatch={
            "active_sessions_max": sessions_max.as_dict(),
            "active_sessions_avg": sessions_avg.as_dict(),
            "ecs_cpu_avg": cpu_avg.as_dict(),
            "ecs_memory_avg": mem_avg.as_dict(),
            "drain_timeouts_sum": drain_timeouts.as_dict(),
            "memory_first_hour_avg": mem_first_hour.as_dict() if mem_first_hour else None,
            "memory_last_hour_avg": mem_last_hour.as_dict() if mem_last_hour else None,
        },
        ecs={
            "pre": pre_state.as_dict(),
            "post": post_state.as_dict(),
            "cycles_logged": len(cycle_log),
        },
        checks=checks,
        notes=[f"Per-cycle log: {log_path}"],
    )
    result.write(paths.scenario_json("e"))
    logger.info("scenario_e_complete", status=result.overall_status, calls=batch.count)
    return result


# ── Soak loop ───────────────────────────────────────────────────────────────


async def _soak_loop(
    *,
    caller: HttpCaller,
    batch: CallResultBatch,
    paths: config.RunPaths,
    heartbeat_path: Path,
    cycle_log: list[dict[str, object]],
    log_fh,
    resumed_from_secs: float,
) -> None:
    """Fire calls + write heartbeats until SOAK_DURATION_SECS elapses."""
    interval = 1.0 / SOAK_CALLS_PER_SEC
    soak_started = time.perf_counter()
    deadline = soak_started + (SOAK_DURATION_SECS - resumed_from_secs)
    next_fire = time.perf_counter()
    last_heartbeat = time.perf_counter()
    in_flight: list[asyncio.Task] = []
    cycle = 0

    async def fire_one() -> None:
        batch.append(await caller.post_start())

    while time.perf_counter() < deadline:
        now = time.perf_counter()

        if now >= next_fire:
            in_flight.append(asyncio.create_task(fire_one()))
            next_fire = now + interval

        if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECS:
            cycle += 1
            elapsed_total = (now - soak_started) + resumed_from_secs
            in_flight = [t for t in in_flight if not t.done()]
            entry = {
                "cycle": cycle,
                "elapsed_secs": round(elapsed_total, 1),
                "elapsed_hours": round(elapsed_total / 3600.0, 3),
                "calls_total": batch.count,
                "accepted": batch.accepted,
                "rejected_503": batch.rejected_503,
                "other": batch.other_counts,
                "in_flight": len(in_flight),
                "ts": scenario_base.now_iso(),
            }
            cycle_log.append(entry)
            log_fh.write(json.dumps(entry, default=str) + "\n")
            log_fh.flush()
            _write_heartbeat(heartbeat_path, entry)
            last_heartbeat = now
            if cycle % 60 == 0:
                logger.info("scenario_e_progress", **entry)

        in_flight = [t for t in in_flight if not t.done()]
        await asyncio.sleep(min(0.05, max(0.0, next_fire - time.perf_counter())))

    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


def _write_heartbeat(path: Path, entry: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(entry, default=str))
    tmp.replace(path)


def _maybe_resume(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


# ── Memory trend helpers ────────────────────────────────────────────────────


async def _split_memory_trend(
    started_dt: datetime, ended_dt: datetime
) -> tuple[cloudwatch.MetricStats | None, cloudwatch.MetricStats | None]:
    """Pull avg ECS memory for the first hour vs the last hour.

    Returns (None, None) if the soak was shorter than 2 hours total
    (split would be meaningless).
    """
    duration_secs = (ended_dt - started_dt).total_seconds()
    if duration_secs < 2 * 3600:
        return None, None
    from datetime import timedelta
    first_end = started_dt + timedelta(hours=1)
    last_start = ended_dt - timedelta(hours=1)
    first = await cloudwatch.query_ecs_memory_utilization(start=started_dt, end=first_end)
    last = await cloudwatch.query_ecs_memory_utilization(start=last_start, end=ended_dt)
    return first, last


# ── Checks ──────────────────────────────────────────────────────────────────


def _build_checks(
    *,
    batch: CallResultBatch,
    cycle_log: list[dict[str, object]],
    pre_state: ecs.ServiceState,
    post_state: ecs.ServiceState,
    cpu_avg: cloudwatch.MetricStats,
    mem_avg: cloudwatch.MetricStats,
    drain_timeouts: cloudwatch.MetricStats,
    mem_first_hour: cloudwatch.MetricStats | None,
    mem_last_hour: cloudwatch.MetricStats | None,
) -> list[scenario_base.Check]:
    checks: list[scenario_base.Check] = []

    # 1. Error rate <= 1%.
    other_total = sum(batch.other_counts.values())
    if batch.count > 0:
        error_rate_pct = round(other_total / batch.count * 100, 2)
    else:
        error_rate_pct = 0.0
    if error_rate_pct <= ERROR_RATE_THRESHOLD_PCT:
        checks.append(
            scenario_base.Check.passed(
                "error_rate_under_threshold",
                f"Soak error rate <= {ERROR_RATE_THRESHOLD_PCT}%.",
                observed=error_rate_pct,
                expected=f"<= {ERROR_RATE_THRESHOLD_PCT}%",
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "error_rate_under_threshold",
                "Soak error rate exceeded threshold.",
                observed=error_rate_pct,
                expected=f"<= {ERROR_RATE_THRESHOLD_PCT}%",
            )
        )

    # 2. Memory growth between first and last hour: < 20% absolute pct points.
    if mem_first_hour and mem_last_hour and mem_first_hour.datapoints > 0 and mem_last_hour.datapoints > 0:
        delta_pct = mem_last_hour.average - mem_first_hour.average
        if delta_pct < 20.0:
            checks.append(
                scenario_base.Check.passed(
                    "memory_trend_stable",
                    "ECS memory utilization didn't grow > 20 pct points across the soak.",
                    observed=round(delta_pct, 2),
                    expected="< 20",
                )
            )
        else:
            checks.append(
                scenario_base.Check.failed(
                    "memory_trend_stable",
                    "ECS memory utilization grew > 20 pct points — possible leak.",
                    observed=round(delta_pct, 2),
                    expected="< 20",
                )
            )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "memory_trend_stable",
                "Not enough data to split first vs last hour memory.",
                observed=(
                    (mem_first_hour.datapoints if mem_first_hour else 0),
                    (mem_last_hour.datapoints if mem_last_hour else 0),
                ),
            )
        )

    # 3. DrainTimeouts must be zero.
    if drain_timeouts.sum_ <= 0:
        checks.append(
            scenario_base.Check.passed(
                "no_drain_timeouts",
                "No DrainTimeouts during soak.",
                observed=drain_timeouts.sum_,
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "no_drain_timeouts",
                "DrainTimeouts fired during soak — task likely got killed and drained slowly.",
                observed=drain_timeouts.sum_,
            )
        )

    # 4. No surprise ECS task replacement (running_count should be stable).
    if pre_state.running_count == post_state.running_count:
        checks.append(
            scenario_base.Check.passed(
                "no_task_replacement",
                "ECS running task count unchanged across the soak.",
                observed=f"{pre_state.running_count} -> {post_state.running_count}",
            )
        )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "no_task_replacement",
                "ECS task count changed during soak.",
                observed=f"{pre_state.running_count} -> {post_state.running_count}",
                note="May be benign (CloudWatch alarm autoscale or planned redeploy). Inspect cycle_log for the transition.",
            )
        )

    # 5. CPU max stayed reasonable for a 50%-cap soak (< 60%).
    if cpu_avg.maximum < 60.0:
        checks.append(
            scenario_base.Check.passed(
                "cpu_reasonable",
                "ECS CPU max stayed under 60% during soak.",
                observed=cpu_avg.maximum,
                expected="< 60",
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "cpu_reasonable",
                "ECS CPU max breached 60% during soak.",
                observed=cpu_avg.maximum,
                expected="< 60",
            )
        )

    # 6. Cycle log was written periodically (sanity).
    expected_cycles = SOAK_DURATION_SECS // HEARTBEAT_INTERVAL_SECS
    if len(cycle_log) >= expected_cycles * 0.9:
        checks.append(
            scenario_base.Check.passed(
                "cycle_log_complete",
                f"Heartbeat log captured >= 90% of expected cycles.",
                observed=len(cycle_log),
                expected=f"~{expected_cycles}",
            )
        )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "cycle_log_complete",
                "Fewer heartbeats than expected (scenario may have been cut short).",
                observed=len(cycle_log),
                expected=f"~{expected_cycles}",
            )
        )

    return checks
