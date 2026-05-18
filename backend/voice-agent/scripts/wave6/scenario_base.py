"""Shared scenario base — result envelope and structured pass/fail.

Each scenario builds a :class:`ScenarioResult` it serialises to
``scenario_<name>.json``. The :func:`report.aggregate` step reads those
back to render ``report.md``.

The pass/fail model is intentionally minimal:

* ``"pass"``  — every assertion in the scenario's checks list was true.
* ``"fail"``  — at least one assertion was false.
* ``"inconclusive"`` — the check couldn't be evaluated (e.g., expected
  data didn't arrive in the window). Distinguished from fail because
  we don't want flaky reports to mask real regressions.

Scenarios attach a list of named ``Check`` records to the result so the
final markdown can show exactly which checks passed/failed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Check:
    """One named assertion within a scenario."""

    name: str
    """Short identifier the report can group on."""
    description: str
    """One-liner explaining what the check verifies."""
    status: str
    """``pass`` / ``fail`` / ``inconclusive``."""
    observed: Any = None
    """Whatever observed value the check evaluated (number, str, dict)."""
    expected: Any = None
    """The threshold / shape the check compared against."""
    note: str = ""
    """Free-form context for the markdown report."""

    @classmethod
    def passed(
        cls,
        name: str,
        description: str,
        *,
        observed: Any = None,
        expected: Any = None,
        note: str = "",
    ) -> "Check":
        return cls(
            name=name,
            description=description,
            status="pass",
            observed=observed,
            expected=expected,
            note=note,
        )

    @classmethod
    def failed(
        cls,
        name: str,
        description: str,
        *,
        observed: Any = None,
        expected: Any = None,
        note: str = "",
    ) -> "Check":
        return cls(
            name=name,
            description=description,
            status="fail",
            observed=observed,
            expected=expected,
            note=note,
        )

    @classmethod
    def inconclusive(
        cls,
        name: str,
        description: str,
        *,
        observed: Any = None,
        expected: Any = None,
        note: str = "",
    ) -> "Check":
        return cls(
            name=name,
            description=description,
            status="inconclusive",
            observed=observed,
            expected=expected,
            note=note,
        )


@dataclass
class ScenarioResult:
    """The thing each scenario serialises."""

    scenario: str
    started_at: str
    ended_at: str
    duration_secs: float
    description: str
    config: dict[str, Any]
    """Per-scenario knobs (target rps, burst size, soak duration, etc.)."""
    calls: dict[str, Any]
    """Counts + latency stats from CallResultBatch.as_summary()."""
    cloudwatch: dict[str, Any]
    """Stats pulled from VoiceAgent/Pipeline + AWS/ECS."""
    ecs: dict[str, Any]
    """Pre/post service state, deployment events."""
    checks: list[Check]
    notes: list[str] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        """Roll-up: fail beats inconclusive beats pass."""
        if any(c.status == "fail" for c in self.checks):
            return "fail"
        if any(c.status == "inconclusive" for c in self.checks):
            return "inconclusive"
        if not self.checks:
            return "inconclusive"
        return "pass"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overall_status"] = self.overall_status
        return d

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2, default=str))


# ── Utilities scenarios use ─────────────────────────────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarise_batch(batch: Any) -> dict[str, Any]:
    """Map a CallResultBatch into the dict we put in ScenarioResult.calls."""
    return {
        "count": batch.count,
        "accepted_202": batch.accepted,
        "rejected_503": batch.rejected_503,
        "other": batch.other_counts,
        "latency_ms": batch.latency_percentiles(),
    }
