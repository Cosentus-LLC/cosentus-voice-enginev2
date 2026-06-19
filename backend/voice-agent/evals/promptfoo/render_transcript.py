"""Render synthetic transcript fixtures into PromptFoo replay prompts."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# PromptFoo's Python provider loads this module via importlib's ``exec_module``
# WITHOUT registering it in ``sys.modules``. Combined with
# ``from __future__ import annotations`` (every annotation is a string), the
# ``@dataclass`` below makes the stdlib resolve annotations via
# ``sys.modules.get(cls.__module__).__dict__`` — which is ``None`` under that
# loader and raises ``AttributeError: 'NoneType' object has no attribute
# '__dict__'``. Register a stand-in so the lookup succeeds. When imported
# normally (e.g. pytest) ``__name__`` is already present and this is a no-op.
if __name__ not in sys.modules:  # pragma: no cover - only under PromptFoo's loader
    sys.modules[__name__] = types.ModuleType(__name__)

_CASES_DIR = Path(__file__).resolve().parent / "cases"
_REQUIRED_SCORING_DIMENSIONS = (
    "task_completion",
    "data_capture",
    "safety_compliance",
    "interruption_handling",
    "call_closure",
)
_REQUIRED_SCORING_DIMENSION_SET = frozenset(_REQUIRED_SCORING_DIMENSIONS)
_REQUIRED_FIELDS = {
    "case_type",
    "case_id",
    "tags",
    "title",
    "agent_name",
    "success_criteria",
    "scoring_dimensions",
    "transcript",
    "expected_min_score",
}
_VALID_CASE_TYPES = frozenset({"calibration", "regression"})
_VALID_SPEAKERS = frozenset({"user", "assistant", "tool"})


@dataclass(frozen=True)
class ScoringDimension:
    """One named dimension the judge scores for a transcript case."""

    name: str
    threshold: float
    blocking: bool
    criteria: list[str]


class TranscriptEvalCase:
    """One de-identified transcript replay case."""

    __slots__ = (
        "agent_name",
        "case_type",
        "case_id",
        "expected_min_score",
        "notes",
        "scoring_dimensions",
        "success_criteria",
        "tags",
        "title",
        "transcript",
    )

    def __init__(
        self,
        *,
        case_id: str,
        title: str,
        agent_name: str,
        success_criteria: list[str],
        transcript: list[dict[str, Any]],
        expected_min_score: float,
        case_type: str,
        tags: list[str],
        scoring_dimensions: dict[str, ScoringDimension],
        notes: str = "",
    ) -> None:
        self.case_id = case_id
        self.case_type = case_type
        self.title = title
        self.agent_name = agent_name
        self.tags = tags
        self.success_criteria = success_criteria
        self.scoring_dimensions = scoring_dimensions
        self.transcript = transcript
        self.expected_min_score = expected_min_score
        self.notes = notes


def load_case(case_path: str | Path) -> TranscriptEvalCase:
    """Load and validate a transcript eval fixture."""

    resolved = _resolve_case_path(str(case_path))
    with resolved.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{resolved.name} must contain a JSON object")

    missing = sorted(_REQUIRED_FIELDS - set(data))
    if missing:
        raise ValueError(f"{resolved.name} missing required fields: {', '.join(missing)}")

    case_id = _non_empty_string(data["case_id"], "case_id", resolved)
    case_type = _case_type(data["case_type"], resolved)
    title = _non_empty_string(data["title"], "title", resolved)
    agent_name = _non_empty_string(data["agent_name"], "agent_name", resolved)
    tags = _string_list(data["tags"], "tags", resolved)
    success_criteria = _string_list(data["success_criteria"], "success_criteria", resolved)
    scoring_dimensions = _scoring_dimensions(data["scoring_dimensions"], resolved)
    transcript = _validate_transcript(data["transcript"], resolved)
    expected_min_score = _score(data["expected_min_score"], "expected_min_score", resolved)
    notes = data.get("notes", "")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{resolved.name} field notes must be a string")

    return TranscriptEvalCase(
        case_id=case_id,
        title=title,
        agent_name=agent_name,
        success_criteria=success_criteria,
        transcript=transcript,
        expected_min_score=expected_min_score,
        case_type=case_type,
        tags=tags,
        scoring_dimensions=scoring_dimensions,
        notes=notes or "",
    )


def render_transcript_prompt(context: dict[str, Any]) -> str:
    """PromptFoo entry point that renders a fixture selected by ``case_file``."""

    vars_ = context.get("vars") or {}
    case_file = vars_.get("case_file")
    if not isinstance(case_file, str) or not case_file.strip():
        raise ValueError("PromptFoo vars.case_file is required")

    case = load_case(case_file)
    criteria = "\n".join(f"- {criterion}" for criterion in case.success_criteria)
    dimensions = "\n".join(
        _render_scoring_dimension(case.scoring_dimensions[name])
        for name in _REQUIRED_SCORING_DIMENSIONS
    )
    turns = "\n".join(_render_turn(turn) for turn in case.transcript)
    notes = f"\nNotes: {case.notes}\n" if case.notes else "\n"

    return "\n".join(
        [
            "Voice call quality evaluation replay",
            "",
            f"Case id: {case.case_id}",
            f"Case type: {case.case_type}",
            f"Title: {case.title}",
            f"Agent: {case.agent_name}",
            f"Tags: {', '.join(case.tags)}",
            f"Expected minimum score: {case.expected_min_score:.2f}",
            notes.rstrip(),
            "",
            "Success criteria:",
            criteria,
            "",
            "Scoring dimensions:",
            dimensions,
            "",
            "Transcript:",
            turns,
            "",
            "Judge this completed call against the success criteria above.",
            "The transcript is synthetic and de-identified. Do not infer facts not present.",
        ]
    )


def _render_scoring_dimension(dimension: ScoringDimension) -> str:
    criteria = "\n".join(f"  - {criterion}" for criterion in dimension.criteria)
    blocking = "blocking" if dimension.blocking else "advisory"
    return "\n".join(
        [
            f"- {dimension.name} (threshold {dimension.threshold:.2f}, {blocking})",
            criteria,
        ]
    )


def _render_turn(turn: dict[str, Any]) -> str:
    """Render one engine-shaped transcript turn for the judge."""

    turn_number = turn.get("turn_number", "?")
    speaker = turn.get("speaker", "unknown")
    content = str(turn.get("content", "")).strip()
    interrupted = " [interrupted]" if speaker == "assistant" and turn.get("interrupted") else ""
    return f"[{turn_number}] {speaker}{interrupted}: {content}"


def _resolve_case_path(case_file: str) -> Path:
    """Resolve a fixture path while confining reads to the cases directory."""

    raw = Path(case_file)
    candidate = raw if raw.is_absolute() else _CASES_DIR / raw
    resolved = candidate.resolve()
    cases_dir = _CASES_DIR.resolve()
    try:
        resolved.relative_to(cases_dir)
    except ValueError as exc:
        raise ValueError(f"case_file must resolve under {cases_dir}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Eval fixture not found: {resolved}")
    return resolved


def _non_empty_string(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path.name} field {field} must be a non-empty string")
    return value


def _case_type(value: Any, path: Path) -> str:
    case_type = _non_empty_string(value, "case_type", path)
    if case_type not in _VALID_CASE_TYPES:
        raise ValueError(f"{path.name} field case_type must be one of {sorted(_VALID_CASE_TYPES)}")
    return case_type


def _string_list(value: Any, field: str, path: Path) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path.name} field {field} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{path.name} field {field} must contain only non-empty strings")
    return list(value)


def _score(value: Any, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path.name} field {field} must be a number")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{path.name} field {field} must be between 0.0 and 1.0")
    return score


def _scoring_dimensions(value: Any, path: Path) -> dict[str, ScoringDimension]:
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} field scoring_dimensions must be an object")

    keys = set(value)
    missing = sorted(_REQUIRED_SCORING_DIMENSION_SET - keys)
    extra = sorted(keys - _REQUIRED_SCORING_DIMENSION_SET)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"extra: {', '.join(extra)}")
        raise ValueError(
            f"{path.name} field scoring_dimensions must contain exactly "
            f"{', '.join(_REQUIRED_SCORING_DIMENSIONS)} ({'; '.join(details)})"
        )

    dimensions: dict[str, ScoringDimension] = {}
    for name in _REQUIRED_SCORING_DIMENSIONS:
        raw = value[name]
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} scoring dimension {name} must be an object")
        threshold = _score(raw.get("threshold"), f"scoring_dimensions.{name}.threshold", path)
        blocking = raw.get("blocking")
        if not isinstance(blocking, bool):
            raise ValueError(
                f"{path.name} field scoring_dimensions.{name}.blocking must be boolean"
            )
        criteria = _string_list(raw.get("criteria"), f"scoring_dimensions.{name}.criteria", path)
        dimensions[name] = ScoringDimension(
            name=name,
            threshold=threshold,
            blocking=blocking,
            criteria=criteria,
        )
    return dimensions


def _validate_transcript(value: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path.name} field transcript must be a non-empty list")

    turns: list[dict[str, Any]] = []
    for index, turn in enumerate(value, start=1):
        if not isinstance(turn, dict):
            raise ValueError(f"{path.name} transcript turn {index} must be an object")
        if not isinstance(turn.get("turn_number"), int) or turn["turn_number"] < 1:
            raise ValueError(f"{path.name} transcript turn {index} needs a positive turn_number")
        speaker = turn.get("speaker")
        if speaker not in _VALID_SPEAKERS:
            raise ValueError(
                f"{path.name} transcript turn {index} speaker must be one of "
                f"{sorted(_VALID_SPEAKERS)}"
            )
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError(f"{path.name} transcript turn {index} needs non-empty content")
        if "interrupted" in turn and not isinstance(turn["interrupted"], bool):
            raise ValueError(f"{path.name} transcript turn {index} interrupted must be boolean")
        turns.append(dict(turn))
    return turns
