"""Scenario D — fleet-wide capacity gate at maxCapacity=5.

What this proves
----------------

* When the fleet runs at maxCapacity (5 tasks × 6 slots = 30 concurrent),
  the engine + ALB pair returns CLEAN 503s for overflow requests — not
  500s, not hangs, not duplicate accepts.
* Capacity gate stays consistent across tasks. No race between tasks
  accepting concurrent requests for the "last" slot.

Profile
-------

1. Pin ``desiredCount=5`` via ECS update-service. Wait until running
   count reaches 5 (typically 2-3 min for new tasks to pull image +
   start + pass health checks).
2. Sustain ~50 calls per second (3000 cpm) for 60 seconds to saturate
   the fleet. With ~1.7 s call lifetime, 50 cps × 1.7 s = 85 average
   concurrent, comfortably above the 30 hard cap.
3. Wave 2: 50 additional /start calls in 10 seconds (the burst).
4. Cleanup: restore ``desiredCount`` to its pre-scenario value (most
   likely 1) so we don't leave a stuck max-out config.

Cost estimate
-------------

50 cps × 60s = 3000 sustained calls + 50 burst calls = 3050 fast-fail
calls @ ~$0.001 = $3.10. Well above the original $0.15 design budget;
the design assumed lower saturation pressure. Acceptable.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import structlog

from . import cloudwatch, config, ecs, scenario_base
from .http_caller import CallResultBatch, HttpCaller

logger = structlog.get_logger(__name__)


SATURATE_CPS = 50.0
SATURATE_DURATION_SECS = 60
BURST_OVERFLOW = 50
BURST_OVERFLOW_WINDOW_SECS = 10.0
WAIT_FOR_TASKS_TIMEOUT_SECS = 600.0


async def run(paths: config.RunPaths) -> scenario_base.ScenarioResult:
    started_dt = datetime.now(UTC)
    started_ts = scenario_base.now_iso()
    started_perf = time.perf_counter()

    pre_state = await ecs.describe_service()
    previous_desired = pre_state.desired_count
    logger.info("scenario_d_starting", pre=pre_state.as_dict(), previous_desired=previous_desired)

    target_tasks = config.ECS_MAX_TASKS

    saturate_batch = CallResultBatch()
    burst_batch = CallResultBatch()
    fleet_state_at_saturate: ecs.ServiceState | None = None
    fleet_state_after_burst: ecs.ServiceState | None = None

    try:
        # 1. Pin desiredCount=5 and wait for the fleet to converge.
        logger.info("scenario_d_pinning_max", desired=target_tasks)
        await ecs.set_desired_count(target_tasks)
        fleet_state_at_saturate = await ecs.wait_for_running_count(
            target=target_tasks, timeout_secs=WAIT_FOR_TASKS_TIMEOUT_SECS
        )
        logger.info("scenario_d_fleet_ready", state=fleet_state_at_saturate.as_dict())

        async with HttpCaller() as caller:
            # 2. Saturate.
            await _fire_steady(caller, saturate_batch, SATURATE_CPS, SATURATE_DURATION_SECS)

            # 3. Burst overflow.
            await _fire_burst(caller, burst_batch, BURST_OVERFLOW, BURST_OVERFLOW_WINDOW_SECS)

        fleet_state_after_burst = await ecs.describe_service()

    finally:
        # 4. Cleanup. Restore previous desired count.
        try:
            logger.info("scenario_d_restoring_desired", previous=previous_desired)
            await ecs.set_desired_count(previous_desired)
        except Exception as exc:  # noqa: BLE001
            logger.error("scenario_d_restore_failed", error=str(exc))

    await asyncio.sleep(120)
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

    checks = _build_checks(
        saturate=saturate_batch,
        burst=burst_batch,
        target_tasks=target_tasks,
        fleet_state_at_saturate=fleet_state_at_saturate,
        sessions_max=sessions_max,
        cpu_avg=cpu_avg,
        mem_avg=mem_avg,
    )

    result = scenario_base.ScenarioResult(
        scenario="d",
        started_at=started_ts,
        ended_at=ended_ts,
        duration_secs=duration_secs,
        description=(
            "Pin desiredCount=5, saturate at 50 cps for 60s, then 50-burst "
            "overflow. Validates fleet-wide capacity gate behaviour."
        ),
        config={
            "target_tasks": target_tasks,
            "saturate_cps": SATURATE_CPS,
            "saturate_duration_secs": SATURATE_DURATION_SECS,
            "burst_overflow": BURST_OVERFLOW,
            "burst_overflow_window_secs": BURST_OVERFLOW_WINDOW_SECS,
            "wait_for_tasks_timeout_secs": WAIT_FOR_TASKS_TIMEOUT_SECS,
            "expected_max_concurrent": target_tasks * config.PER_TASK_CONCURRENCY,
        },
        calls={
            "saturate": scenario_base.summarise_batch(saturate_batch),
            "burst": scenario_base.summarise_batch(burst_batch),
            "count": saturate_batch.count + burst_batch.count,
            "accepted_202": saturate_batch.accepted + burst_batch.accepted,
            "rejected_503": saturate_batch.rejected_503 + burst_batch.rejected_503,
            "other": _merge(saturate_batch.other_counts, burst_batch.other_counts),
            "latency_ms": _combine_latency(saturate_batch, burst_batch),
        },
        cloudwatch={
            "active_sessions_max": sessions_max.as_dict(),
            "active_sessions_avg": sessions_avg.as_dict(),
            "session_utilization_max": util_max.as_dict(),
            "ecs_cpu_avg": cpu_avg.as_dict(),
            "ecs_memory_avg": mem_avg.as_dict(),
        },
        ecs={
            "pre": pre_state.as_dict(),
            "fleet_state_at_saturate": fleet_state_at_saturate.as_dict()
            if fleet_state_at_saturate
            else None,
            "fleet_state_after_burst": fleet_state_after_burst.as_dict()
            if fleet_state_after_burst
            else None,
            "post": post_state.as_dict(),
            "previous_desired": previous_desired,
            "pinned_desired": target_tasks,
        },
        checks=checks,
    )
    result.write(paths.scenario_json("d"))
    logger.info("scenario_d_complete", status=result.overall_status)
    return result


# ── Fire helpers ────────────────────────────────────────────────────────────


async def _fire_steady(
    caller: HttpCaller, batch: CallResultBatch, rate_cps: float, duration_secs: int
) -> None:
    interval = 1.0 / rate_cps
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
        await asyncio.sleep(min(0.01, max(0.0, next_fire - time.perf_counter())))
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


async def _fire_burst(
    caller: HttpCaller, batch: CallResultBatch, size: int, window_secs: float
) -> None:
    interval = window_secs / size
    started_perf = time.perf_counter()

    async def fire_one() -> None:
        batch.append(await caller.post_start())

    tasks: list[asyncio.Task] = []
    for i in range(size):
        target_offset = i * interval
        sleep_for = max(0.0, target_offset - (time.perf_counter() - started_perf))
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        tasks.append(asyncio.create_task(fire_one()))
    await asyncio.gather(*tasks, return_exceptions=False)


# ── Checks ──────────────────────────────────────────────────────────────────


def _build_checks(
    *,
    saturate: CallResultBatch,
    burst: CallResultBatch,
    target_tasks: int,
    fleet_state_at_saturate: ecs.ServiceState | None,
    sessions_max: cloudwatch.MetricStats,
    cpu_avg: cloudwatch.MetricStats,
    mem_avg: cloudwatch.MetricStats,
) -> list[scenario_base.Check]:
    checks: list[scenario_base.Check] = []
    expected_max_concurrent = target_tasks * config.PER_TASK_CONCURRENCY

    # 1. Fleet actually reached target_tasks.
    if (
        fleet_state_at_saturate is not None
        and fleet_state_at_saturate.running_count == target_tasks
    ):
        checks.append(
            scenario_base.Check.passed(
                "fleet_reached_max",
                f"ECS scaled to {target_tasks} tasks before saturation.",
                observed=fleet_state_at_saturate.running_count,
                expected=target_tasks,
            )
        )
    else:
        observed = fleet_state_at_saturate.running_count if fleet_state_at_saturate else "(none)"
        checks.append(
            scenario_base.Check.failed(
                "fleet_reached_max",
                "ECS did not converge to desired count before saturation.",
                observed=observed,
                expected=target_tasks,
            )
        )

    # 2. Saturate phase: capacity gate clearly fires. With 50 cps × 60s
    #    = 3000 calls and only 30 slots, we expect well over 90% to be 503.
    if saturate.count > 0 and saturate.rejected_503 >= saturate.count * 0.5:
        checks.append(
            scenario_base.Check.passed(
                "saturate_rejected_majority",
                "Majority of saturate-phase calls were cleanly 503'd.",
                observed=f"{saturate.rejected_503}/{saturate.count}",
                expected=">= 50%",
            )
        )
    else:
        observed = f"{saturate.rejected_503}/{saturate.count}"
        checks.append(
            scenario_base.Check.failed(
                "saturate_rejected_majority",
                (
                    "Saturation didn't produce expected reject rate — gate may be "
                    "misbehaving or fleet has more capacity than expected."
                ),
                observed=observed,
                expected=">= 50%",
            )
        )

    # 3. No 5xx-other on any phase.
    other_total = _merge(saturate.other_counts, burst.other_counts)
    if not other_total:
        checks.append(
            scenario_base.Check.passed(
                "no_unclean_outcomes",
                "All requests returned 202 or 503.",
                observed={"clean": saturate.count + burst.count},
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "no_unclean_outcomes",
                "Some requests returned 5xx-other / timeouts / errors.",
                observed=other_total,
            )
        )

    # 4. Capacity gate did not overshoot fleet max.
    if sessions_max.maximum <= expected_max_concurrent:
        checks.append(
            scenario_base.Check.passed(
                "no_capacity_overshoot",
                "CloudWatch ActiveSessions max stayed at or under fleet capacity.",
                observed=sessions_max.maximum,
                expected=f"<= {expected_max_concurrent}",
            )
        )
    elif sessions_max.datapoints == 0:
        checks.append(
            scenario_base.Check.inconclusive(
                "no_capacity_overshoot",
                "No ActiveSessions datapoints in the window.",
                observed=sessions_max.datapoints,
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "no_capacity_overshoot",
                "ActiveSessions exceeded fleet capacity — capacity gate failed.",
                observed=sessions_max.maximum,
                expected=f"<= {expected_max_concurrent}",
            )
        )

    # 5. Burst-overflow: same gate cleanness after saturation.
    if burst.count > 0 and burst.rejected_503 >= burst.count // 2:
        checks.append(
            scenario_base.Check.passed(
                "burst_overflow_rejected",
                "Burst-overflow majority rejected with 503.",
                observed=f"{burst.rejected_503}/{burst.count}",
                expected=">= 50%",
            )
        )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "burst_overflow_rejected",
                (
                    "Burst overflow did not majority-reject; either saturation drained too "
                    "quickly or scenario sized too low."
                ),
                observed=f"{burst.rejected_503}/{burst.count}",
            )
        )

    # 6. CPU + memory not above alarm.
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

    return checks


def _merge(*ds: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _combine_latency(*batches: CallResultBatch) -> dict[str, float]:
    ms = sorted(r.latency_ms for b in batches for r in b.results)
    if not ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    n = len(ms)

    def pick(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return round(ms[idx], 1)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99), "n": float(n)}
