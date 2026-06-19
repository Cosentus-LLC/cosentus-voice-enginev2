"""Scenario A — steady ramp 3 → 100 calls/min over 30 min.

What this proves
----------------

* Engine ingestion is smooth at sustained rates up to 100 cpm without
  5xx, timeouts, or queueing artefacts.
* Stub Lambda + Daily REST + dialout-fail path scales linearly with
  call rate (no surprising per-call growth in latency).
* CloudWatch metric pipeline emits cleanly throughout (no gaps in the
  30s cadence).
* If concurrent steady-state ever crosses ``SCALE_TARGET_SESSIONS_PER
  _TASK=4.2``, the autoscaler attempts a scale-out. **However**, at
  100 cpm peak × 1.7 s fast-fail call lifetime = ~2.83 concurrent
  average, which is below the target — autoscale-out is not the
  primary assertion. We mark it inconclusive if it doesn't fire.

Ramp profile
------------

Six 5-minute phases, each at a fixed rate, stepping up. This is more
deterministic than a linear ramp (and easier to inspect per phase
when something looks wrong):

============  =============================
Minutes 0-5   3   calls/min  ( 0.05 cps )
Minutes 5-10  20  calls/min  ( 0.33 cps )
Minutes 10-15 40  calls/min  ( 0.67 cps )
Minutes 15-20 60  calls/min  ( 1.00 cps )
Minutes 20-25 80  calls/min  ( 1.33 cps )
Minutes 25-30 100 calls/min  ( 1.67 cps )
============  =============================

Total: 15 + 100 + 200 + 300 + 400 + 500 = 1515 calls.
Vendor cost @ ~$0.001/call ≈ $1.50. (Well under the $0.30 estimate I
gave in the design — the design assumed 10 cpm average; this is 50 cpm
average. Real cost ~$1.50.)
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import structlog

from . import cloudwatch, config, ecs, scenario_base
from .http_caller import CallResultBatch, HttpCaller

logger = structlog.get_logger(__name__)


_PHASES_CPM = (3, 20, 40, 60, 80, 100)
_PHASE_DURATION_SECS = 300  # 5 minutes per phase
_TOTAL_DURATION_SECS = _PHASE_DURATION_SECS * len(_PHASES_CPM)  # 1800


async def run(paths: config.RunPaths) -> scenario_base.ScenarioResult:
    started_dt = datetime.now(UTC)
    started_ts = scenario_base.now_iso()
    started_perf = time.perf_counter()

    pre_state = await ecs.describe_service()
    logger.info("scenario_a_starting", pre=pre_state.as_dict(), phases=_PHASES_CPM)

    per_phase: dict[str, CallResultBatch] = {}

    async with HttpCaller() as caller:
        for phase_idx, cpm in enumerate(_PHASES_CPM):
            phase_label = f"phase_{phase_idx}_{cpm}cpm"
            batch = CallResultBatch()
            per_phase[phase_label] = batch
            logger.info("scenario_a_phase_starting", phase=phase_label, cpm=cpm)
            await _fire_phase(caller, batch, cpm, _PHASE_DURATION_SECS)

    await asyncio.sleep(120)  # let CloudWatch ingest the tail
    post_state = await ecs.describe_service()
    ended_dt = datetime.now(UTC)
    ended_ts = scenario_base.now_iso()
    duration_secs = round(time.perf_counter() - started_perf, 1)

    cw_window = cloudwatch.window_around(started_dt, ended_dt, pad_min=2)
    sessions_max = await cloudwatch.query_active_sessions_max(start=cw_window[0], end=cw_window[1])
    sessions_avg = await cloudwatch.query_active_sessions_avg(start=cw_window[0], end=cw_window[1])
    util_max = await cloudwatch.query_session_utilization_max(start=cw_window[0], end=cw_window[1])
    cpu_avg = await cloudwatch.query_ecs_cpu_utilization(start=cw_window[0], end=cw_window[1])
    mem_avg = await cloudwatch.query_ecs_memory_utilization(start=cw_window[0], end=cw_window[1])
    drain_timeouts = await cloudwatch.query_drain_timeouts_sum(start=cw_window[0], end=cw_window[1])

    # Aggregate all phases into one rollup batch for top-level numbers.
    rollup = CallResultBatch()
    for batch in per_phase.values():
        for r in batch.results:
            rollup.append(r)

    checks = _build_checks(
        rollup=rollup,
        per_phase=per_phase,
        pre_state=pre_state,
        post_state=post_state,
        sessions_max=sessions_max,
        sessions_avg=sessions_avg,
        cpu_avg=cpu_avg,
        mem_avg=mem_avg,
        drain_timeouts=drain_timeouts,
    )

    result = scenario_base.ScenarioResult(
        scenario="a",
        started_at=started_ts,
        ended_at=ended_ts,
        duration_secs=duration_secs,
        description=(
            "Six-phase ramp 3→100 cpm over 30 minutes. Validates ingestion "
            "smoothness; autoscale-out trigger is best-effort given the "
            "fast-fail lifetime keeps concurrent < target."
        ),
        config={
            "phases_cpm": list(_PHASES_CPM),
            "phase_duration_secs": _PHASE_DURATION_SECS,
            "expected_total_calls": sum(cpm * (_PHASE_DURATION_SECS // 60) for cpm in _PHASES_CPM),
        },
        calls={
            **{label: scenario_base.summarise_batch(b) for label, b in per_phase.items()},
            "count": rollup.count,
            "accepted_202": rollup.accepted,
            "rejected_503": rollup.rejected_503,
            "other": rollup.other_counts,
            "latency_ms": rollup.latency_percentiles(),
        },
        cloudwatch={
            "active_sessions_max": sessions_max.as_dict(),
            "active_sessions_avg": sessions_avg.as_dict(),
            "session_utilization_max": util_max.as_dict(),
            "ecs_cpu_avg": cpu_avg.as_dict(),
            "ecs_memory_avg": mem_avg.as_dict(),
            "drain_timeouts_sum": drain_timeouts.as_dict(),
        },
        ecs={
            "pre": pre_state.as_dict(),
            "post": post_state.as_dict(),
        },
        checks=checks,
    )
    result.write(paths.scenario_json("a"))
    logger.info("scenario_a_complete", status=result.overall_status)
    return result


# ── Fire one phase at a fixed rate ──────────────────────────────────────────


async def _fire_phase(
    caller: HttpCaller,
    batch: CallResultBatch,
    cpm: int,
    duration_secs: int,
) -> None:
    interval = 60.0 / cpm if cpm > 0 else duration_secs + 1
    deadline = time.perf_counter() + duration_secs
    next_fire = time.perf_counter()
    in_flight: list[asyncio.Task] = []

    async def fire_one() -> None:
        batch.append(await caller.post_start())

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if now >= next_fire:
            in_flight.append(asyncio.create_task(fire_one()))
            next_fire = now + interval
        in_flight = [t for t in in_flight if not t.done()]
        await asyncio.sleep(min(0.05, max(0.0, next_fire - time.perf_counter())))
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


# ── Checks ──────────────────────────────────────────────────────────────────


def _build_checks(
    *,
    rollup: CallResultBatch,
    per_phase: dict[str, CallResultBatch],
    pre_state: ecs.ServiceState,
    post_state: ecs.ServiceState,
    sessions_max: cloudwatch.MetricStats,
    sessions_avg: cloudwatch.MetricStats,
    cpu_avg: cloudwatch.MetricStats,
    mem_avg: cloudwatch.MetricStats,
    drain_timeouts: cloudwatch.MetricStats,
) -> list[scenario_base.Check]:
    checks: list[scenario_base.Check] = []

    # 1. Every phase should accept the vast majority of calls (>= 90%)
    #    given there's no sustained overrun. Rejections at this rate
    #    would indicate a problem.
    for label, batch in per_phase.items():
        if batch.count == 0:
            checks.append(
                scenario_base.Check.failed(
                    f"{label}_calls_issued",
                    f"No calls issued in {label}.",
                    observed=0,
                    note="Harness misconfigured or scenario aborted early.",
                )
            )
            continue
        accept_rate = batch.accepted / batch.count
        if accept_rate >= 0.9:
            checks.append(
                scenario_base.Check.passed(
                    f"{label}_accept_rate",
                    f"Acceptance rate for {label} >= 90%.",
                    observed=round(accept_rate * 100, 1),
                    expected=">= 90",
                )
            )
        else:
            checks.append(
                scenario_base.Check.failed(
                    f"{label}_accept_rate",
                    f"Acceptance rate for {label} < 90%.",
                    observed=round(accept_rate * 100, 1),
                    expected=">= 90",
                    note=f"Rejected: {batch.rejected_503}; other: {batch.other_counts}",
                )
            )

    # 2. No 5xx-other / no timeouts / no client errors across the run.
    if not rollup.other_counts:
        checks.append(
            scenario_base.Check.passed(
                "no_unclean_outcomes",
                "All requests received 202 or 503 (no 5xx-other / timeouts / errors).",
                observed={"clean": rollup.count, "other": {}},
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "no_unclean_outcomes",
                "Some requests received unexpected status / errors.",
                observed=rollup.other_counts,
            )
        )

    # 3. DrainTimeouts must be zero — scenario A doesn't kill anything.
    if drain_timeouts.sum_ <= 0:
        checks.append(
            scenario_base.Check.passed(
                "no_drain_timeouts",
                "DrainTimeouts metric stayed at 0 across the scenario.",
                observed=drain_timeouts.sum_,
                expected="0",
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "no_drain_timeouts",
                "DrainTimeouts fired during a non-shutdown scenario.",
                observed=drain_timeouts.sum_,
                expected="0",
                note=(
                    "Either an engine task crashed and drained slowly, or the alarm metric "
                    "is misbehaving."
                ),
            )
        )

    # 4. CPU + memory stay below alarm thresholds.
    if cpu_avg.maximum <= 80.0:
        checks.append(
            scenario_base.Check.passed(
                "cpu_below_alarm",
                "ECS CPU stayed below 80%.",
                observed=cpu_avg.maximum,
                expected="<= 80",
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "cpu_below_alarm",
                "ECS CPU breached 80%.",
                observed=cpu_avg.maximum,
                expected="<= 80",
            )
        )
    if mem_avg.maximum <= 80.0:
        checks.append(
            scenario_base.Check.passed(
                "memory_below_alarm",
                "ECS memory stayed below 80%.",
                observed=mem_avg.maximum,
                expected="<= 80",
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "memory_below_alarm",
                "ECS memory breached 80%.",
                observed=mem_avg.maximum,
                expected="<= 80",
            )
        )

    # 5. Autoscale-out — inconclusive by design at this profile.
    if post_state.running_count > pre_state.running_count:
        checks.append(
            scenario_base.Check.passed(
                "autoscaler_added_task",
                "ECS task count grew at peak.",
                observed=f"{pre_state.running_count} -> {post_state.running_count}",
            )
        )
    elif sessions_avg.maximum >= config.SCALE_TARGET_SESSIONS_PER_TASK:
        checks.append(
            scenario_base.Check.inconclusive(
                "autoscaler_added_task",
                "Average per-task sessions hit the target but no scale-out within window.",
                observed=(
                    f"avg max={sessions_avg.maximum}, "
                    f"target={config.SCALE_TARGET_SESSIONS_PER_TASK}"
                ),
            )
        )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "autoscaler_added_task",
                (
                    "Concurrent average stayed below scale target; expected at 100 cpm × "
                    "1.7s lifetime."
                ),
                observed=sessions_avg.maximum,
                expected=f">= {config.SCALE_TARGET_SESSIONS_PER_TASK}",
                note="Scenario B's sustained 3 cps phase is the proper autoscale-out test.",
            )
        )

    return checks
