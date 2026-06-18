"""ECS API helpers — task count, kill, capacity controls.

Wave 6 scenarios need three primitives the ALB doesn't expose:

* **observe** — current desired/running/pending counts, plus the list of
  running task ARNs. Used by every scenario for sanity-checking task
  state before/after the work phase.
* **stop_task** — kill one running task to trigger ECS replacement.
  Scenario c's whole point.
* **set_desired_count** — short-circuit the autoscaler by pinning a
  desired count. Scenario d uses this to deterministically reach
  maxCapacity=5 without waiting for the autoscaler to ramp.

All boto3 work is sync, wrapped in :func:`asyncio.to_thread` so the
async harness doesn't block.

Operational safety
------------------

``stop_task`` and ``set_desired_count`` modify live infrastructure. The
harness restricts both to the staging service (hardcoded via
``config.ECS_CLUSTER`` / ``config.ECS_SERVICE``). The scenario-level
finally-blocks restore desired-count to the previous value after the
work phase to avoid leaking a stuck max-out config into the next run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import boto3

from . import config

_ECS_CLIENT: Any = None


def _get_ecs_client() -> Any:
    global _ECS_CLIENT
    if _ECS_CLIENT is None:
        _ECS_CLIENT = boto3.session.Session().client("ecs", region_name=config.AWS_REGION)
    return _ECS_CLIENT


# ── Observe ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ServiceState:
    """Snapshot of the ECS service at one moment."""

    desired_count: int
    running_count: int
    pending_count: int
    deployments: int
    task_definition_arn: str
    primary_deployment_status: str
    running_task_arns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "desired_count": self.desired_count,
            "running_count": self.running_count,
            "pending_count": self.pending_count,
            "deployments": self.deployments,
            "primary_deployment_status": self.primary_deployment_status,
            "task_definition_arn": self.task_definition_arn,
            "running_task_count": len(self.running_task_arns),
        }


async def describe_service() -> ServiceState:
    """Pull the current state of the staging ECS service."""
    client = _get_ecs_client()
    response = await asyncio.to_thread(
        client.describe_services,
        cluster=config.ECS_CLUSTER,
        services=[config.ECS_SERVICE],
    )
    services = response.get("services") or []
    if not services:
        raise RuntimeError(
            f"ECS service {config.ECS_SERVICE} not found in cluster "
            f"{config.ECS_CLUSTER}. Is staging deployed?"
        )
    svc = services[0]
    deployments = svc.get("deployments") or []
    primary_status = ""
    for d in deployments:
        if d.get("status") == "PRIMARY":
            primary_status = d.get("rolloutState") or d.get("status") or ""
            break

    # Pull running task ARNs in parallel — needed for stop_task to pick
    # a victim, and for sanity-checking the running_count number.
    tasks_resp = await asyncio.to_thread(
        client.list_tasks,
        cluster=config.ECS_CLUSTER,
        serviceName=config.ECS_SERVICE,
        desiredStatus="RUNNING",
    )
    task_arns = tuple(tasks_resp.get("taskArns") or [])

    return ServiceState(
        desired_count=int(svc.get("desiredCount", 0)),
        running_count=int(svc.get("runningCount", 0)),
        pending_count=int(svc.get("pendingCount", 0)),
        deployments=len(deployments),
        task_definition_arn=str(svc.get("taskDefinition") or ""),
        primary_deployment_status=primary_status,
        running_task_arns=task_arns,
    )


# ── Mutate ──────────────────────────────────────────────────────────────────


async def stop_task(task_arn: str, *, reason: str) -> None:
    """Terminate one running task. ECS replaces it to satisfy desired count.

    Used by scenario c. The reason string surfaces in CloudWatch / Console
    for traceability ("wave6_scenario_c_crash_injection").
    """
    client = _get_ecs_client()
    await asyncio.to_thread(
        client.stop_task,
        cluster=config.ECS_CLUSTER,
        task=task_arn,
        reason=reason,
    )


async def set_desired_count(desired: int) -> None:
    """Update the staging service's desiredCount, bounded by maxCapacity.

    Scenario d uses this to deterministically reach maxCapacity=5 — the
    autoscaler will respect the manual override (target-tracking
    autoscaling adjusts desiredCount but never below the explicit value
    we set; in practice it'll re-evaluate once the work phase ends).

    The autoscaler's ScalableTarget min/max bounds still apply, so values
    outside [minCapacity, maxCapacity] silently clamp.
    """
    if desired < 0 or desired > config.ECS_MAX_TASKS * 2:
        raise ValueError(
            f"set_desired_count={desired} outside [0, {config.ECS_MAX_TASKS * 2}] — "
            "refusing to modify staging service with a clearly-bogus value."
        )
    client = _get_ecs_client()
    await asyncio.to_thread(
        client.update_service,
        cluster=config.ECS_CLUSTER,
        service=config.ECS_SERVICE,
        desiredCount=desired,
    )


async def wait_for_running_count(
    target: int, *, timeout_secs: float = 600.0, poll_interval_secs: float = 10.0
) -> ServiceState:
    """Poll until ``runningCount == target`` (or timeout).

    Used after ``set_desired_count`` so scenarios can assume the cluster
    is steady before starting the work phase.
    """
    started = asyncio.get_event_loop().time()
    while True:
        state = await describe_service()
        if state.running_count == target and state.pending_count == 0:
            return state
        if asyncio.get_event_loop().time() - started > timeout_secs:
            raise TimeoutError(
                f"ECS service did not reach runningCount={target} "
                f"in {timeout_secs}s. Last state: {state.as_dict()}"
            )
        await asyncio.sleep(poll_interval_secs)


async def wait_for_task_replacement(
    *, original_task_arn: str, timeout_secs: float = 300.0
) -> ServiceState:
    """Poll until the killed task is replaced.

    Considered replaced when ``runningCount`` returns to its pre-kill
    value AND the killed task ARN is no longer in the running list.
    """
    started = asyncio.get_event_loop().time()
    while True:
        state = await describe_service()
        if (
            original_task_arn not in state.running_task_arns
            and state.running_count == state.desired_count
            and state.pending_count == 0
        ):
            return state
        if asyncio.get_event_loop().time() - started > timeout_secs:
            raise TimeoutError(
                f"Task {original_task_arn} not replaced within {timeout_secs}s. "
                f"Last state: {state.as_dict()}"
            )
        await asyncio.sleep(5.0)
