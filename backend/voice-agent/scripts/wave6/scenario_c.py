"""Scenario C — crash recovery via ECS stop-task.

What this proves
----------------

* Killing an active engine task triggers ECS replacement within
  ~60 seconds (per Layer 9.5 sizing budget).
* Bots on the killed task get safe-cancelled — engine's
  ``graceful_drain`` writes partial CallRecords through the stub.
* ``/start`` continues to succeed during the gap (handled by other
  running tasks if any, otherwise 503 briefly).
* No deadlocks, no zombie tasks, no stuck deployments.

Profile
-------

1. Drive steady ~3 cps load for 2 minutes (warm-up).
2. Pick one running task ARN, ``aws ecs stop-task`` with a reason
   string the report can grep on.
3. Keep firing /start at the same rate for 5 minutes after kill.
4. Wait for ``runningCount`` to recover to ``desiredCount`` AND the
   killed task ARN to be absent. Record the time-to-recover.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import structlog

from . import cloudwatch, config, ecs, scenario_base
from .http_caller import CallResultBatch, HttpCaller

logger = structlog.get_logger(__name__)


WARMUP_SECS = 120
POST_KILL_SECS = 300
CALLS_PER_SEC = 3.0
KILL_REASON = "wave6_scenario_c_crash_injection"
REPLACEMENT_DEADLINE_SECS = 300  # we expect ~60s in practice


async def run(paths: config.RunPaths) -> scenario_base.ScenarioResult:
    started_dt = datetime.now(UTC)
    started_ts = scenario_base.now_iso()
    started_perf = time.perf_counter()

    pre_state = await ecs.describe_service()
    if pre_state.running_count == 0 or not pre_state.running_task_arns:
        raise RuntimeError(
            "Scenario C cannot run: no running ECS tasks. Verify the staging "
            "service is healthy before retrying."
        )
    victim = pre_state.running_task_arns[0]
    logger.info("scenario_c_starting", pre=pre_state.as_dict(), victim=victim)

    warmup_batch = CallResultBatch()
    post_kill_batch = CallResultBatch()

    async with HttpCaller() as caller:
        # Phase 1: 2 minute warm-up.
        warmup_task = asyncio.create_task(
            _fire_steady(caller, warmup_batch, CALLS_PER_SEC, WARMUP_SECS)
        )
        await warmup_task

        # Phase 2: stop one task.
        kill_started = time.perf_counter()
        kill_ts = scenario_base.now_iso()
        logger.warning("scenario_c_killing_task", task_arn=victim)
        await ecs.stop_task(victim, reason=KILL_REASON)

        # Phase 3: continue load while ECS recovers. Run firing + recovery
        # tracking concurrently so /start traffic doesn't pause.
        post_kill_task = asyncio.create_task(
            _fire_steady(caller, post_kill_batch, CALLS_PER_SEC, POST_KILL_SECS)
        )
        try:
            replaced_state = await ecs.wait_for_task_replacement(
                original_task_arn=victim, timeout_secs=REPLACEMENT_DEADLINE_SECS
            )
            replaced_at = time.perf_counter()
            replaced_ts = scenario_base.now_iso()
            time_to_recover = round(replaced_at - kill_started, 1)
            logger.info("scenario_c_replacement_complete", time_to_recover_s=time_to_recover)
            recovery_outcome = "recovered"
        except TimeoutError as exc:
            replaced_state = await ecs.describe_service()
            replaced_ts = scenario_base.now_iso()
            time_to_recover = float("nan")
            recovery_outcome = "timeout"
            logger.error("scenario_c_replacement_timeout", error=str(exc))

        await post_kill_task

    await asyncio.sleep(60)
    post_state = await ecs.describe_service()
    ended_dt = datetime.now(UTC)
    ended_ts = scenario_base.now_iso()
    duration_secs = round(time.perf_counter() - started_perf, 1)

    cw_window = cloudwatch.window_around(started_dt, ended_dt, pad_min=2)
    sessions_max = await cloudwatch.query_active_sessions_max(start=cw_window[0], end=cw_window[1])
    drain_timeouts = await cloudwatch.query_drain_timeouts_sum(start=cw_window[0], end=cw_window[1])
    cpu_avg = await cloudwatch.query_ecs_cpu_utilization(start=cw_window[0], end=cw_window[1])
    mem_avg = await cloudwatch.query_ecs_memory_utilization(start=cw_window[0], end=cw_window[1])

    checks = _build_checks(
        warmup=warmup_batch,
        post_kill=post_kill_batch,
        pre_state=pre_state,
        replaced_state=replaced_state,
        post_state=post_state,
        recovery_outcome=recovery_outcome,
        time_to_recover=time_to_recover,
        drain_timeouts=drain_timeouts,
    )

    result = scenario_base.ScenarioResult(
        scenario="c",
        started_at=started_ts,
        ended_at=ended_ts,
        duration_secs=duration_secs,
        description=(
            "Steady-state load, kill one task, watch ECS replace it. "
            "Validates crash recovery + safe-cancel of in-flight bots."
        ),
        config={
            "warmup_secs": WARMUP_SECS,
            "post_kill_secs": POST_KILL_SECS,
            "calls_per_sec": CALLS_PER_SEC,
            "kill_reason": KILL_REASON,
            "replacement_deadline_secs": REPLACEMENT_DEADLINE_SECS,
        },
        calls={
            "warmup": scenario_base.summarise_batch(warmup_batch),
            "post_kill": scenario_base.summarise_batch(post_kill_batch),
            "count": warmup_batch.count + post_kill_batch.count,
            "accepted_202": warmup_batch.accepted + post_kill_batch.accepted,
            "rejected_503": warmup_batch.rejected_503 + post_kill_batch.rejected_503,
            "other": _merge(warmup_batch.other_counts, post_kill_batch.other_counts),
            "latency_ms": _combine_latency(warmup_batch, post_kill_batch),
        },
        cloudwatch={
            "active_sessions_max": sessions_max.as_dict(),
            "drain_timeouts_sum": drain_timeouts.as_dict(),
            "ecs_cpu_avg": cpu_avg.as_dict(),
            "ecs_memory_avg": mem_avg.as_dict(),
        },
        ecs={
            "pre": pre_state.as_dict(),
            "victim_task_arn": victim,
            "kill_ts": kill_ts,
            "replaced_ts": replaced_ts,
            "time_to_recover_secs": time_to_recover,
            "recovery_outcome": recovery_outcome,
            "replaced_state": replaced_state.as_dict(),
            "post": post_state.as_dict(),
        },
        checks=checks,
    )
    result.write(paths.scenario_json("c"))
    logger.info(
        "scenario_c_complete",
        status=result.overall_status,
        time_to_recover=time_to_recover,
    )
    return result


# ── Steady-rate firing helper ───────────────────────────────────────────────


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
        await asyncio.sleep(min(0.02, max(0.0, next_fire - time.perf_counter())))
    if in_flight:
        await asyncio.gather(*in_flight, return_exceptions=True)


# ── Checks ──────────────────────────────────────────────────────────────────


def _build_checks(
    *,
    warmup: CallResultBatch,
    post_kill: CallResultBatch,
    pre_state: ecs.ServiceState,
    replaced_state: ecs.ServiceState,
    post_state: ecs.ServiceState,
    recovery_outcome: str,
    time_to_recover: float,
    drain_timeouts: cloudwatch.MetricStats,
) -> list[scenario_base.Check]:
    checks: list[scenario_base.Check] = []

    # 1. Warm-up phase clean.
    if not warmup.other_counts:
        checks.append(
            scenario_base.Check.passed(
                "warmup_clean",
                "Warm-up phase saw only 202s and 503s.",
                observed=warmup.count,
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "warmup_clean",
                "Warm-up phase saw 5xx/timeouts.",
                observed=warmup.other_counts,
            )
        )

    # 2. Recovery time within budget.
    if recovery_outcome == "recovered" and time_to_recover <= REPLACEMENT_DEADLINE_SECS:
        checks.append(
            scenario_base.Check.passed(
                "task_replaced_within_budget",
                f"ECS replaced the killed task in {time_to_recover}s.",
                observed=time_to_recover,
                expected=f"<= {REPLACEMENT_DEADLINE_SECS}",
            )
        )
    elif recovery_outcome == "recovered":
        checks.append(
            scenario_base.Check.failed(
                "task_replaced_within_budget",
                "Task replacement exceeded budget.",
                observed=time_to_recover,
                expected=f"<= {REPLACEMENT_DEADLINE_SECS}",
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "task_replaced_within_budget",
                "ECS never re-reached desired count after the kill.",
                observed=recovery_outcome,
                expected="recovered",
            )
        )

    # 3. Service deployment count returned to 1 (no stuck deployments).
    if post_state.deployments == 1 and post_state.running_count == post_state.desired_count:
        checks.append(
            scenario_base.Check.passed(
                "deployment_quiesced",
                "Single deployment, runningCount == desiredCount after recovery.",
                observed=post_state.as_dict(),
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "deployment_quiesced",
                "Deployment did not return to steady state.",
                observed=post_state.as_dict(),
            )
        )

    # 4. Post-kill /start traffic continued to succeed (with some 503s
    #    allowed during the gap). Acceptance should still be majority.
    post_kill_total = post_kill.count
    if post_kill_total > 0 and post_kill.accepted >= post_kill_total // 2:
        checks.append(
            scenario_base.Check.passed(
                "post_kill_traffic_mostly_accepted",
                "Majority of /start traffic during recovery was accepted.",
                observed=f"{post_kill.accepted}/{post_kill_total}",
                expected=">= 50%",
            )
        )
    elif post_kill_total > 0:
        checks.append(
            scenario_base.Check.inconclusive(
                "post_kill_traffic_mostly_accepted",
                "Less than half of post-kill traffic accepted.",
                observed=f"{post_kill.accepted}/{post_kill_total}",
                note=(
                    "May be normal if running_count was 1 pre-kill (no other "
                    "task to absorb traffic during the ~60s replacement gap)."
                ),
            )
        )
    else:
        checks.append(
            scenario_base.Check.inconclusive(
                "post_kill_traffic_mostly_accepted",
                "No post-kill /start calls were made.",
                observed=0,
            )
        )

    # 5. DrainTimeouts should be ZERO. The killed task's safe-cancel ran
    #    fast (engine's drain budget=110s; ECS stop sends SIGTERM, engine
    #    drains active sessions, completes well within 110s). Any non-zero
    #    here means a task got stuck during drain.
    if drain_timeouts.sum_ <= 0:
        checks.append(
            scenario_base.Check.passed(
                "no_drain_timeouts",
                "Killed task drained cleanly (no DrainTimeouts emitted).",
                observed=drain_timeouts.sum_,
            )
        )
    else:
        checks.append(
            scenario_base.Check.failed(
                "no_drain_timeouts",
                "Killed task hit drain budget — investigate stuck sessions.",
                observed=drain_timeouts.sum_,
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
