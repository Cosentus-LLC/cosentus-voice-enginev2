"""Tests for the PromptFoo transcript replay eval assets."""

from __future__ import annotations

import json
import re
from pathlib import Path

from evals.promptfoo.render_transcript import (
    _render_turn,
    load_case,
    render_transcript_prompt,
)

ROOT = Path(__file__).resolve().parents[2]
PROMPTFOO_DIR = ROOT / "evals" / "promptfoo"
CASES_DIR = PROMPTFOO_DIR / "cases"
CONFIG_PATH = PROMPTFOO_DIR / "promptfooconfig.yaml"

SCORING_DIMENSIONS = {
    "task_completion",
    "data_capture",
    "safety_compliance",
    "interruption_handling",
    "call_closure",
}

REQUIRED_REGRESSION_TAGS = {
    "paid",
    "denied",
    "pending",
    "no-claim-found",
    "transferred",
    "disconnected",
    "payer-refuses-info",
    "ivr",
    "hold",
    "transfer-heavy",
}

EXPECTED_CASE_IDS = {
    "claim-status-denied",
    "claim-status-disconnected-retry-needed",
    "claim-status-missing-reference",
    "claim-status-no-claim-found",
    "claim-status-paid",
    "claim-status-payer-refuses-info",
    "claim-status-pending",
    "claim-status-transferred-to-claims",
    "ivr-digits-pending",
    "ivr-hold-paid",
    "transfer-heavy-denied",
}

PHI_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "dob_label": re.compile(r"\b(?:dob|date of birth|birth date)\b", re.IGNORECASE),
    "mrn_label": re.compile(r"\b(?:mrn|medical record)\b", re.IGNORECASE),
    "placeholder_name": re.compile(r"\b(?:john doe|jane doe|patient name)\b", re.IGNORECASE),
}


def _case_files() -> list[Path]:
    return sorted(CASES_DIR.glob("*.json"))


def _load_cases():
    return [load_case(path) for path in _case_files()]


def _string_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            if key == "timestamp":
                continue
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def test_eval_cases_load_successfully():
    cases = _load_cases()

    assert {case.case_id for case in cases} == EXPECTED_CASE_IDS
    assert all(case.transcript for case in cases)
    assert all(0.0 <= case.expected_min_score <= 1.0 for case in cases)
    assert all(set(case.scoring_dimensions) == SCORING_DIMENSIONS for case in cases)


def test_render_transcript_prompt_includes_success_criteria_dimensions_and_turns():
    prompt = render_transcript_prompt({"vars": {"case_file": "claim_status_paid.json"}})

    assert "Case id: claim-status-paid" in prompt
    assert "Case type: regression" in prompt
    assert "Success criteria:" in prompt
    assert "Scoring dimensions:" in prompt
    for dimension in SCORING_DIMENSIONS:
        assert dimension in prompt
    assert "CLAIM-EXAMPLE-PAID" in prompt
    assert "[4] user: That claim was paid." in prompt
    assert "REF-PAID-ALPHA" in prompt


def test_render_transcript_prompt_marks_interrupted_assistant_turns():
    rendered = _render_turn(
        {
            "turn_number": 9,
            "speaker": "assistant",
            "content": "I was cut off.",
            "interrupted": True,
        }
    )

    assert rendered == "[9] assistant [interrupted]: I was cut off."


def test_eval_fixtures_have_no_phi_like_strings():
    for path in _case_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        for value in _string_values(data):
            for name, pattern in PHI_PATTERNS.items():
                assert not pattern.search(value), f"{path.name} matched {name}: {value!r}"


def test_promptfoo_config_uses_echo_provider_and_seed_cases():
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert "id: echo" in config
    assert "file://render_transcript.py:render_transcript_prompt" in config
    assert "llm-rubric" in config
    for path in _case_files():
        assert path.name in config

    live_call_paths = (
        "app/bot",
        "app/services",
        "app/runner",
        "Daily",
        "ElevenLabs",
        "AssemblyAI",
    )
    assert all(path not in config for path in live_call_paths)


def test_regression_suite_has_at_least_ten_cases():
    regression_cases = [case for case in _load_cases() if case.case_type == "regression"]

    assert len(regression_cases) >= 10


def test_regression_suite_covers_required_call_paths():
    regression_tags = {
        tag for case in _load_cases() if case.case_type == "regression" for tag in case.tags
    }

    assert REQUIRED_REGRESSION_TAGS <= regression_tags


def test_each_regression_case_has_blocking_core_dimensions():
    for case in _load_cases():
        if case.case_type != "regression":
            continue

        assert case.scoring_dimensions["task_completion"].blocking is True
        assert case.scoring_dimensions["data_capture"].blocking is True
        assert case.scoring_dimensions["safety_compliance"].blocking is True
        assert "interruption_handling" in case.scoring_dimensions
        assert "call_closure" in case.scoring_dimensions


def test_calibration_cases_are_not_counted_as_regression_cases():
    calibration_cases = [case for case in _load_cases() if case.case_type == "calibration"]

    assert [case.case_id for case in calibration_cases] == ["claim-status-missing-reference"]
    for case in calibration_cases:
        assert all(not dimension.blocking for dimension in case.scoring_dimensions.values())


def test_case_ids_are_unique_and_match_files():
    cases = _load_cases()
    case_ids = [case.case_id for case in cases]

    assert len(case_ids) == len(set(case_ids))
    for path, case in zip(_case_files(), cases, strict=True):
        assert case.case_id.replace("-", "_") == path.stem


def test_promptfoo_config_exposes_named_dimension_metrics():
    config = CONFIG_PATH.read_text(encoding="utf-8")

    for dimension in SCORING_DIMENSIONS:
        assert f"metric: {dimension}" in config
