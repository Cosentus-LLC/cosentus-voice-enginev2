"""CloudWatch + Container Insights metric reads for Wave 6 scenarios.

All boto3 work is sync — wrapped in :func:`asyncio.to_thread` so the
async harness doesn't block. The scenarios call these helpers once per
scenario (typically at end-of-window) to pull aggregate stats; they
don't poll mid-scenario (too expensive, too noisy).

Two metric families
-------------------

* **VoiceAgent/Pipeline** — emitted by the engine task. Carries the
  ``Environment`` dimension (``staging`` for Wave 6).
  - ``ActiveSessions`` (Count, 30 s cadence) — drives the autoscaling
    policy + the ActiveSessionsApproachingMax alarm.
  - ``SessionUtilization`` (Percent, 30 s) — diagnostic.
  - ``DrainTimeouts`` (Count, on-demand) — alarm-only; Wave 6 will
    look for this near zero.

* **AWS/ECS + ECS/ContainerInsights** — emitted by ECS itself.
  - ``CPUUtilization`` / ``MemoryUtilization`` per service.
  - Container Insights ``RunningTaskCount`` for ECS scale tracking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

from . import config


# ── Boto3 clients (lazy + cached) ───────────────────────────────────────────

_CW_CLIENT: Any = None


def _get_cw_client() -> Any:
    global _CW_CLIENT
    if _CW_CLIENT is None:
        _CW_CLIENT = boto3.session.Session().client(
            "cloudwatch", region_name=config.AWS_REGION
        )
    return _CW_CLIENT


# ── Query primitive ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricStats:
    """Aggregated result of one CloudWatch query window."""

    metric_name: str
    namespace: str
    statistic: str
    period_secs: int
    datapoints: int
    minimum: float
    maximum: float
    average: float
    sum_: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "namespace": self.namespace,
            "statistic": self.statistic,
            "period_secs": self.period_secs,
            "datapoints": self.datapoints,
            "min": round(self.minimum, 2),
            "max": round(self.maximum, 2),
            "avg": round(self.average, 2),
            "sum": round(self.sum_, 2),
        }


async def query_metric(
    *,
    namespace: str,
    metric_name: str,
    dimensions: list[dict[str, str]],
    statistic: str,
    period_secs: int,
    start: datetime,
    end: datetime,
) -> MetricStats:
    """Pull stats for one metric over a window.

    The ``statistic`` parameter must be one of CloudWatch's supported
    values (Sum, Average, Maximum, Minimum, SampleCount). We compute
    min/max/avg/sum from the returned datapoints so callers don't
    have to.
    """
    client = _get_cw_client()
    response = await asyncio.to_thread(
        client.get_metric_statistics,
        Namespace=namespace,
        MetricName=metric_name,
        Dimensions=dimensions,
        StartTime=start,
        EndTime=end,
        Period=period_secs,
        Statistics=[statistic],
    )
    pts = response.get("Datapoints", []) or []
    values = [pt[statistic] for pt in pts]
    if not values:
        return MetricStats(
            metric_name=metric_name,
            namespace=namespace,
            statistic=statistic,
            period_secs=period_secs,
            datapoints=0,
            minimum=0.0,
            maximum=0.0,
            average=0.0,
            sum_=0.0,
        )
    return MetricStats(
        metric_name=metric_name,
        namespace=namespace,
        statistic=statistic,
        period_secs=period_secs,
        datapoints=len(values),
        minimum=min(values),
        maximum=max(values),
        average=sum(values) / len(values),
        sum_=sum(values),
    )


# ── Convenience: engine custom metrics ──────────────────────────────────────


_ENV_DIMENSION = [
    {"Name": "Environment", "Value": config.CW_ENVIRONMENT_DIMENSION}
]


async def query_active_sessions(
    *, start: datetime, end: datetime, period_secs: int = 60
) -> MetricStats:
    """Fleet-wide ActiveSessions for the window. Sum + Maximum together
    paint the autoscaling story.
    """
    return await query_metric(
        namespace=config.CW_NAMESPACE,
        metric_name="ActiveSessions",
        dimensions=_ENV_DIMENSION,
        statistic="Sum",
        period_secs=period_secs,
        start=start,
        end=end,
    )


async def query_active_sessions_max(
    *, start: datetime, end: datetime, period_secs: int = 60
) -> MetricStats:
    return await query_metric(
        namespace=config.CW_NAMESPACE,
        metric_name="ActiveSessions",
        dimensions=_ENV_DIMENSION,
        statistic="Maximum",
        period_secs=period_secs,
        start=start,
        end=end,
    )


async def query_active_sessions_avg(
    *, start: datetime, end: datetime, period_secs: int = 60
) -> MetricStats:
    """Average ActiveSessions per task across the window — what the
    target-tracking scaling policy actually evaluates against.
    """
    return await query_metric(
        namespace=config.CW_NAMESPACE,
        metric_name="ActiveSessions",
        dimensions=_ENV_DIMENSION,
        statistic="Average",
        period_secs=period_secs,
        start=start,
        end=end,
    )


async def query_session_utilization_max(
    *, start: datetime, end: datetime, period_secs: int = 60
) -> MetricStats:
    return await query_metric(
        namespace=config.CW_NAMESPACE,
        metric_name="SessionUtilization",
        dimensions=_ENV_DIMENSION,
        statistic="Maximum",
        period_secs=period_secs,
        start=start,
        end=end,
    )


async def query_drain_timeouts_sum(
    *, start: datetime, end: datetime, period_secs: int = 3600
) -> MetricStats:
    return await query_metric(
        namespace=config.CW_NAMESPACE,
        metric_name="DrainTimeouts",
        dimensions=_ENV_DIMENSION,
        statistic="Sum",
        period_secs=period_secs,
        start=start,
        end=end,
    )


# ── ECS service metrics (AWS/ECS namespace) ─────────────────────────────────


_ECS_DIMENSIONS = [
    {"Name": "ServiceName", "Value": config.ECS_SERVICE},
    {"Name": "ClusterName", "Value": config.ECS_CLUSTER},
]


async def query_ecs_cpu_utilization(
    *, start: datetime, end: datetime, period_secs: int = 60
) -> MetricStats:
    """ECS CPUUtilization Average — alarm threshold is 80 %."""
    return await query_metric(
        namespace="AWS/ECS",
        metric_name="CPUUtilization",
        dimensions=_ECS_DIMENSIONS,
        statistic="Average",
        period_secs=period_secs,
        start=start,
        end=end,
    )


async def query_ecs_memory_utilization(
    *, start: datetime, end: datetime, period_secs: int = 60
) -> MetricStats:
    return await query_metric(
        namespace="AWS/ECS",
        metric_name="MemoryUtilization",
        dimensions=_ECS_DIMENSIONS,
        statistic="Average",
        period_secs=period_secs,
        start=start,
        end=end,
    )


# ── Convenience: scenario windows ───────────────────────────────────────────


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def window_around(start: datetime, end: datetime, *, pad_min: int = 1) -> tuple[datetime, datetime]:
    """Pad the window so CloudWatch's ~1 min reporting lag is covered."""
    return (
        start - timedelta(minutes=pad_min),
        end + timedelta(minutes=pad_min),
    )
