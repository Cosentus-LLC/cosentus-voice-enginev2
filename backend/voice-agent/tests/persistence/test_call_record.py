"""Tests for ``app/persistence/call_record.py``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from app.persistence.call_record import CallRecord

# ── Test fixture ───────────────────────────────────────────────────────────


def _full_record(**overrides) -> CallRecord:
    """Build a CallRecord with sensible defaults; override per-test."""
    base = {
        "id": "11111111-1111-1111-1111-111111111111",
        "agent_name": "test-agent",
        "agent_display_name": "Test Agent",
        "from_number": "+19494360836",
        "target_number": "+12098075018",
        "direction": "inbound",
        "status": "completed",
        "started_at": datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC),
        "ended_at": datetime(2026, 5, 4, 12, 5, 30, tzinfo=UTC),
        "duration_secs": 330,
        "case_data": {"Service_Date": "2026-04-01"},
        "transcript": [{"turn_number": 1, "speaker": "user", "content": "hi"}],
        "session_id": "daily-room-abc",
    }
    base.update(overrides)
    return CallRecord(**base)


# ── to_lambda_body field mapping ───────────────────────────────────────────


def test_to_lambda_body_contains_every_voice_calls_column():
    """Confirm wire shape covers every ``voice_calls.*`` field the engine writes."""
    body = _full_record().to_lambda_body()
    expected_keys = {
        "id",
        "agent_name",
        "agent_display_name",
        "from_number",
        "target_number",
        "direction",
        "status",
        "started_at",
        "ended_at",
        "duration_secs",
        "case_data",
        "transcript",
        "recording_path",
        "post_call_analyses",
        "error",
        "batch_id",
        "batch_row_index",
        "session_id",
        "updated_at",
    }
    assert set(body.keys()) == expected_keys


def test_to_lambda_body_passthrough_values():
    body = _full_record().to_lambda_body()
    assert body["id"] == "11111111-1111-1111-1111-111111111111"
    assert body["agent_name"] == "test-agent"
    assert body["from_number"] == "+19494360836"
    assert body["direction"] == "inbound"
    assert body["status"] == "completed"
    assert body["session_id"] == "daily-room-abc"
    assert body["case_data"] == {"Service_Date": "2026-04-01"}


def test_recording_path_always_passed_through():
    """Engine never sets ``recording_path``; it defaults to None and stays."""
    rec = _full_record()
    assert rec.recording_path is None
    assert rec.to_lambda_body()["recording_path"] is None


def test_post_call_analyses_default_is_empty_dict():
    """No analyses configured → empty dict on wire."""
    rec = _full_record()
    assert rec.to_lambda_body()["post_call_analyses"] == {}


def test_post_call_analyses_preserved_when_set():
    rec = _full_record(post_call_analyses={"summary": "ok"})
    assert rec.to_lambda_body()["post_call_analyses"] == {"summary": "ok"}


# ── Status validation ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status",
    [
        "pending",
        "in_progress",
        "completed",
        "failed",
        "cancelled",
        "no_answer",
        "busy",
        "skipped",
    ],
)
def test_valid_statuses_accepted(status):
    """Every status v1 ever writes should be accepted."""
    rec = _full_record(status=status)
    assert rec.status == status


def test_invalid_status_raises():
    with pytest.raises(ValueError, match="Invalid status"):
        _full_record(status="not_a_real_status")


def test_invalid_status_error_lists_valid_choices():
    """Operator running a typo wants the full list — surface it."""
    try:
        _full_record(status="completd")
    except ValueError as exc:
        # Validate at least one canonical value appears in the error.
        assert "completed" in str(exc)
        assert "pending" in str(exc)


# ── Timestamp serialization ────────────────────────────────────────────────


def test_iso8601_timestamp_serialization():
    body = _full_record().to_lambda_body()
    started = datetime.fromisoformat(body["started_at"])
    assert started.tzinfo is not None
    assert body["started_at"] == "2026-05-04T12:00:00+00:00"
    assert body["ended_at"] == "2026-05-04T12:05:30+00:00"


def test_naive_datetime_coerced_to_utc():
    """A producer that forgets ``tzinfo`` still gets UTC on the wire."""
    naive = datetime(2026, 5, 4, 12, 0, 0)  # no tzinfo
    rec = _full_record(started_at=naive)
    body = rec.to_lambda_body()
    assert body["started_at"].endswith("+00:00")


def test_ended_at_nullable():
    """A record built before pipeline tears down has ``ended_at=None``."""
    rec = _full_record(ended_at=None)
    body = rec.to_lambda_body()
    assert body["ended_at"] is None


def test_updated_at_added_at_serialize_time():
    """updated_at is fresh on each serialize, not stored on the dataclass."""
    rec = _full_record()
    assert not hasattr(rec, "updated_at")
    body = rec.to_lambda_body()
    assert "updated_at" in body
    # Round-trips as ISO-8601.
    datetime.fromisoformat(body["updated_at"])


# ── duration_secs handling ────────────────────────────────────────────────


def test_duration_none_serializes_as_zero():
    rec = _full_record(duration_secs=None)
    assert rec.to_lambda_body()["duration_secs"] == 0


def test_duration_int_passes_through():
    rec = _full_record(duration_secs=42)
    assert rec.to_lambda_body()["duration_secs"] == 42


def test_duration_zero_serializes_as_zero():
    """Edge case — instant-failure call legitimately has duration_secs=0."""
    rec = _full_record(duration_secs=0)
    assert rec.to_lambda_body()["duration_secs"] == 0


# ── Error truncation ──────────────────────────────────────────────────────


def test_error_none_passes_through():
    rec = _full_record(error=None)
    assert rec.to_lambda_body()["error"] is None


def test_error_truncated_to_1000_chars():
    long_error = "x" * 5000
    rec = _full_record(status="failed", error=long_error)
    assert len(rec.to_lambda_body()["error"]) == 1000


def test_error_short_passes_through_unchanged():
    rec = _full_record(status="failed", error="boom")
    assert rec.to_lambda_body()["error"] == "boom"


# ── Batch correlation ────────────────────────────────────────────────────


def test_batch_fields_default_none():
    rec = _full_record()
    body = rec.to_lambda_body()
    assert body["batch_id"] is None
    assert body["batch_row_index"] is None


def test_batch_fields_passed_through():
    rec = _full_record(
        batch_id="22222222-2222-2222-2222-222222222222",
        batch_row_index=5,
    )
    body = rec.to_lambda_body()
    assert body["batch_id"] == "22222222-2222-2222-2222-222222222222"
    assert body["batch_row_index"] == 5
