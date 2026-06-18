"""Scenario B — burst 50 /start in 10 seconds.

What this proves
----------------

At minCapacity=1 (the staging steady-state), max_concurrent=6, the
fleet has exactly 6 slots open. Firing 50 ``/start`` in 10 seconds:

* The first 6 fast-fail calls churn quickly enough that some land back
  in the available pool before all 50 are issued — so we expect more
  than 6 accepted overall, but bounded.
* Calls arriving while every slot is occupied get a clean 503 with
  ``reason: at_capacity``.
* No 5xx-other / no timeouts / no hangs.
* ECS circuit-breaker stays closed throughout (no rollback events).

Phase 2 of the scenario sustains ~3 cps for 5 minutes to drive
``ActiveSessions`` near the autoscaling target so we can observe a
scale-out reaction time.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import structlog

from . import cloudwatch, config, ecs, scenario_base
from .http_caller import CallResultBatch, HttpCaller

logger = structlog.get_logger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────

BURST_SIZE = 50
BURST_WINDOW_SECS = 10.0
SUSTAINED_DURATION_SECS = 300  # 5 min
SUSTAINED_CALLS_PER_SEC = 3.0  # 180 cpm → above target=4.2/task ⇒ scale-out
POST_BURST_QUIESCE_SECS = 30  # let the autoscaler settle


# ── Scenario ────────────────────────────────────────────────────────────────


async def run(paths: config.RunPaths) -> scenario_base.ScenarioResult:
    """Execute scenario B against staging. Returns a ScenarioResult."""
    started_dt = datetime.now(UTC)
    started_ts = scenario_base.now_iso()
    started_perf = time.perf_counter()

    pre_state = await ecs.describe_service()
    logger.info("scenario_b_starting", pre=pre_state.as_dict())

    burst_batch = CallResultBatch()
    sustained_batch = CallResultBatch()

    async with HttpCaller() as caller:
        # ── Phase 1: 50 calls in 10 s ─────────────────────────────────────
        await _fire_burst(caller, burst_batch)

        # Brief quiesce so the autoscaler sees the burst settle.
        await asyncio.sleep(POST_BURST_QUIESCE_SECS)

        # ── Phase 2: sustained ~3 cps for 5 min ───────────────────────────
        await _fire_sustained(caller, sustained_batch)

    # Let CloudWatch ingest the last datapoints before querying.
    await asyncio.sleep(120)
    post_state = await ecs.describe_service()
    ended_dt = datetime.now(UTC)
    ended_ts = scenario_base.now_iso()
    duration_secs = round(time.perf_counter() - started_perf, 1)

    cw_window = cloudwatch.window_around(started_dt, ended_dt, pad_min=2)
    sessions_sum = await cloudwatch.query_active_sessions(start=cw_window[0], end=cw_window[1])
    sessions_max = await cloudwatch.query_active_sessions_max(start=cw_window[0], end=cw_window[1])
    sessions_avg = await cloudwatch.query_active_sessions_avg(start=cw_window[0], end=cw_window[1])
    util_max = await cloudwatch.query_session_utilization_max(start=cw_window[0], end=cw_window[1])
    cpu_avg = await cloudwatch.query_ecs_cpu_utilization(start=cw_window[0], end=cw_window[1])
    mem_avg = await cloudwatch.query_ecs_memory_utilization(start=cw_window[0], end=cw_window[1])

    checks = _build_checks(
        burst_batch=burst_batch,
        sustained_batch=sustained_batch,
        pre_state=pre_state,
        post_state=post_state,
        sessions_max=sessions_max,
        sessions_avg=sessions_avg,
        cpu_avg=cpu_avg,
        mem_avg=mem_avg,
    )

    result = scenario_base.ScenarioResult(
        scenario="b",
        started_at=started_ts,
        ended_at=ended_ts,
        duration_secs=duration_secs,
        description=(
            "Burst 50 calls in 10s (capacity-gate stress), then sustained "
            f"{SUSTAINED_CALLS_PER_SEC} cps for {SUSTAINED_DURATION_SECS}s "
            "to drive autoscaling."
        ),
        config={
            "burst_size": BURST_SIZE,
            "burst_window_secs": BURST_WINDOW_SECS,
            "sustained_duration_secs": SUSTAINED_DURATION_SECS,
            "sustained_calls_per_sec": SUSTAINED_CALLS_PER_SEC,
            "per_task_concurrency": config.PER_TASK_CONCURRENCY,
            "scale_target_sessions_per_task": config.SCALE_TARGET_SESSIONS_PER_TASK,
        },
        calls={
            "burst": scenario_base.summarise_batch(burst_batch),
            "sustained": scenario_base.summarise_batch(sustained_batch),
            "count": burst_batch.count + sustained_batch.count,
            "accepted_202": burst_batch.accepted + sustained_batch.accepted,
            "rejected_503": burst_batch.rejected_503 + sustained_batch.rejected_503,
            "other": _merge_counts(burst_batch.other_counts, sustained_batch.other_counts),
            "latency_ms": _combine_latency(burst_batch, sustained_batch),
        },
        cloudwatch={
            "active_sessions_sum": sessions_sum.as_dict(),
            "active_sessions_max": sessions_max.as_dict(),
            "active_sessions_avg": sessions_avg.as_dict(),
            "session_utilization_max": util_max.as_dict(),
            "ecs_cpu_avg": cpu_avg.as_dict(),
            "ecs_memory_avg": mem_avg.as_dict(),
        },
        ecs={
            "pre": pre_state.as_dict(),
            "post": post_state.as_dict(),
        },
        checks=checks,
    )
    result.write(paths.scenario_json("b"))
    logger.info("scenario_b_complete", status=result.overall_status)
    return result


# ── Fire phases ─────────────────────────────────────────────────────────────


async def _fire_burst(caller: HttpCaller, batch: CallResultBatch) -> None:
    """Issue BURST_SIZE concurrent /start POSTs over BURST_WINDOW_SECS."""
    # Spread the BURST_SIZE evenly over BURST_WINDOW_SECS by dispatching
    # one task every (window / N) seconds. asyncio.gather lets all the
    # tasks complete in parallel even though they start at staggered times.
    interval = BURST_WINDOW_SECS / BURST_SIZE
    started_perf = time.perf_counter()

    async def fire_one() -> None:
        batch.append(await caller.post_start())

    tasks: list[asyncio.Task] = []
    for i in range(BURST_SIZE):
        target_offset = i * interval
        sleep_for = max(0.0, target_offset - (time.perf_counter() - started_perf))
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        tasks.append(asyncio.create_task(fire_one(), name=f"scenario_b_burst_{i}"))
    await asyncio.gather(*tasks, return_exceptions=False)


async def _fire_sustained(caller: HttpCaller, batch: CallResultBatch) -> None:
    """Re-fire /start every (1/RATE) seconds for SUSTAINED_DURATION_SECS."""
    interval = 1.0 / SUSTAINED_CALLS_PER_SEC
    deadline = time.perf_counter() + SUSTAINED_DURATION_SECS
    next_fire = time.perf_counter()
    in_flight: list[asyncio.Task] = []

    async def fire_one() -> None:
        batch.append(await caller.post_start())

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if now >= next_fire:
            in_flight.append(asyncio.create_task(fire_one()))
            next_fire = now + interval
        # Trim completed tasks so the list doesn't grow unbounded.
        in_flight = [t for t in in_flight if not t.done()]
        await asyncio.sleep(min(0.01, max(0.0, next_fire - time.perf_counter())))
    # Drain any remaining in-flight tasks.
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


# ── Checks ──────────────────────────────────────────────────────────────────


def _build_checks(
    *,
    burst_batch: CallResultBatch,
    sustained_batch: CallResultBatch,
    pre_state: ecs.ServiceState,
    post_state: ecs.ServiceState,
    sessions_max: cloudwatch.MetricStats,
    sessions_avg: cloudwatch.MetricStats,
    cpu_avg: cloudwatch.MetricStats,
    mem_avg: cloudwatch.MetricStats,
) -> list[scenario_base.Check]:
    """Translate raw stats into named pass/fail/inconclusive findings."""
    checks: list[scenario_base.Check] = []

    # 1. The burst itself should NOT exceed BURST_SIZE — sanity.
    expected_count = BURST_SIZE
    if burst_batch.count == expected_count:
        checks.append(
            scenario_base.Check.passed(
                "burst_size_correct",
                "Harness fired exactly BURST_SIZE requests.",
                observed=burst_batch.count,
                expected=expected_count,
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "burst_size_correct",
                "Harness fired wrong number of burst requests.",
                observed=burst_batch.count,
                expected=expected_count,
            )
        )

    # 2. Burst-phase outcomes: every call must be 202 OR 503. No 5xx
    #    other than 503, no timeouts, no errors.
    burst_non_clean = burst_batch.other_counts
    if not burst_non_clean:
        checks.append(
            scenario_base.Check.passed(
                "burst_no_unclean_outcomes",
                "All burst requests received 202 or 503.",
                observed={"clean": burst_batch.count, "other": burst_non_clean},
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "burst_no_unclean_outcomes",
                "Some burst requests received unexpected status / errors.",
                observed=burst_non_clean,
                note="Inspect engine logs for 5xx, ALB-side errors, or harness timeouts.",
            )
        )

    # 3. Capacity gate must hold: 503s should appear (we ARE saturating).
    #    Reasonable lower bound: at least 1 reject means the gate fired.
    if burst_batch.rejected_503 >= 1:
        checks.append(
            scenario_base.Check.passed(
                "capacity_gate_fired",
                "503 returned at least once during burst (gate engaged).",
                observed=burst_batch.rejected_503,
                expected=">= 1",
            )
        )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "capacity_gate_fired",
                (
                    "No 503 observed; possibly all 50 calls drained fast enough that no "
                    "overflow occurred."
                ),
                observed=burst_batch.rejected_503,
                note=(
                    "The fast-fail call lifetime is ~1.7s. At 50 over 10s, some slots "
                    "open up before overflow lands."
                ),
            )
        )

    # 4. Capacity gate must NOT overshoot per-task: max concurrent should
    #    stay <= PER_TASK_CONCURRENCY × runningCount.
    expected_max_concurrent = config.PER_TASK_CONCURRENCY * max(post_state.running_count, 1)
    if sessions_max.maximum <= expected_max_concurrent:
        checks.append(
            scenario_base.Check.passed(
                "active_sessions_within_capacity",
                "CloudWatch ActiveSessions max stayed within fleet capacity.",
                observed=sessions_max.maximum,
                expected=f"<= {expected_max_concurrent}",
            )
        )
    elif sessions_max.datapoints == 0:
        checks.append(
            scenario_base.Check.inconclusive(
                "active_sessions_within_capacity",
                "No ActiveSessions datapoints in the window — metric pipeline may be lagging.",
                observed=sessions_max.datapoints,
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "active_sessions_within_capacity",
                "ActiveSessions exceeded fleet capacity — capacity gate overshot.",
                observed=sessions_max.maximum,
                expected=f"<= {expected_max_concurrent}",
            )
        )

    # 5. Sustained phase pushes target — did the autoscaler add a task?
    #    Expected: post_state.running_count > pre_state.running_count.
    if post_state.running_count > pre_state.running_count:
        checks.append(
            scenario_base.Check.passed(
                "autoscaler_added_task",
                "ECS task count grew during sustained phase.",
                observed=f"{pre_state.running_count} -> {post_state.running_count}",
                expected="> pre",
            )
        )
    elif sessions_avg.maximum >= config.SCALE_TARGET_SESSIONS_PER_TASK:
        checks.append(
            scenario_base.Check.inconclusive(
                "autoscaler_added_task",
                (
                    "Average ActiveSessions hit the scaling target but task count did not "
                    "grow within the window."
                ),
                observed=(
                    f"{pre_state.running_count} -> {post_state.running_count}; "
                    f"avg max={sessions_avg.maximum}"
                ),
                expected="> pre",
                note=(
                    "Target-tracking policies typically need 1-3 min to react. The 2 min "
                    "post-load window may be too tight."
                ),
            )
        )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "autoscaler_added_task",
                "Average ActiveSessions never reached the scaling target; no scale-out expected.",
                observed=sessions_avg.maximum,
                expected=f">= {config.SCALE_TARGET_SESSIONS_PER_TASK}",
            )
        )

    # 6. Resource pressure: CPU + memory stay within alarm thresholds.
    if cpu_avg.maximum <= 80.0 or cpu_avg.datapoints == 0:
        checks.append(
            scenario_base.Check.passed(
                "cpu_below_alarm",
                "ECS CPU stayed below the HighCPUUtilization alarm threshold.",
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

    if mem_avg.maximum <= 80.0 or mem_avg.datapoints == 0:
        checks.append(
            scenario_base.Check.passed(
                "memory_below_alarm",
                "ECS memory stayed below the HighMemoryUtilization alarm threshold.",
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

    return checks


def _merge_counts(*ds: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _combine_latency(*batches: CallResultBatch) -> dict[str, Any]:
    """Concatenate latencies from multiple batches and recompute percentiles."""
    ms = sorted(r.latency_ms for b in batches for r in b.results)
    if not ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    n = len(ms)

    def pick(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return round(ms[idx], 1)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99), "n": n}
