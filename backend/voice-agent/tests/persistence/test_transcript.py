"""Tests for ``app/persistence/transcript.py``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from app.persistence.transcript import TranscriptAccumulator, TranscriptTurn

# ── Empty / boundary cases ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_accumulator_returns_empty_list():
    """A fresh accumulator has no turns."""
    accum = TranscriptAccumulator()
    assert accum.to_list() == []
    assert accum.turn_count() == 0


@pytest.mark.asyncio
async def test_empty_arguments_dict_renders_clean():
    """``end_call({})`` renders as ``end_call() → success``."""
    accum = TranscriptAccumulator()
    await accum.append_tool_turn(
        tool_name="end_call",
        arguments={},
        status="success",
    )
    turn = accum.to_list()[0]
    assert turn["content"] == "end_call() → success"


# ── Single-speaker appends ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_turn_basic_fields():
    accum = TranscriptAccumulator()
    await accum.append_user_turn("Hi, I need help with a claim.")
    turns = accum.to_list()
    assert len(turns) == 1
    assert turns[0]["speaker"] == "user"
    assert turns[0]["turn_number"] == 1
    assert turns[0]["content"] == "Hi, I need help with a claim."
    # Timestamp is ISO-8601 and parses back.
    parsed = datetime.fromisoformat(turns[0]["timestamp"])
    assert parsed.tzinfo is not None


@pytest.mark.asyncio
async def test_assistant_turn_basic_fields():
    accum = TranscriptAccumulator()
    await accum.append_assistant_turn("Sure, I can help with that.")
    turns = accum.to_list()
    assert len(turns) == 1
    assert turns[0]["speaker"] == "assistant"


@pytest.mark.asyncio
async def test_explicit_timestamp_preserved():
    """When the producer passes ``timestamp``, the accumulator preserves it."""
    accum = TranscriptAccumulator()
    fixed = datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC)
    await accum.append_user_turn("Hello", timestamp=fixed)
    assert accum.to_list()[0]["timestamp"] == fixed.isoformat()


# ── Tool-turn formatting ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_turn_success_format():
    """Format: ``tool_name(key='val') → success``."""
    accum = TranscriptAccumulator()
    await accum.append_tool_turn(
        tool_name="transfer_call",
        arguments={"target": "billing"},
        status="success",
    )
    assert accum.to_list()[0]["content"] == "transfer_call(target='billing') → success"


@pytest.mark.asyncio
async def test_tool_turn_error_format():
    """Format: ``tool_name(key='val') → error: <msg>``."""
    accum = TranscriptAccumulator()
    await accum.append_tool_turn(
        tool_name="press_digit",
        arguments={"digits": "1234"},
        status="error",
        error="No SIP session available for DTMF",
    )
    assert (
        accum.to_list()[0]["content"]
        == "press_digit(digits='1234') → error: No SIP session available for DTMF"
    )


@pytest.mark.asyncio
async def test_tool_turn_arguments_sorted_for_determinism():
    """Args render alphabetically so two identical calls render identically."""
    accum = TranscriptAccumulator()
    await accum.append_tool_turn(
        tool_name="hypothetical",
        arguments={"zebra": 1, "alpha": 2, "mango": 3},
        status="success",
    )
    content = accum.to_list()[0]["content"]
    # Alpha first, mango second, zebra last.
    alpha_idx = content.index("alpha")
    mango_idx = content.index("mango")
    zebra_idx = content.index("zebra")
    assert alpha_idx < mango_idx < zebra_idx


@pytest.mark.asyncio
async def test_tool_turn_error_omitted_on_success():
    """Even if ``error`` is provided, success status doesn't show it."""
    accum = TranscriptAccumulator()
    await accum.append_tool_turn(
        tool_name="ok_tool",
        arguments={},
        status="success",
        error="this should not appear",
    )
    assert "this should not appear" not in accum.to_list()[0]["content"]


@pytest.mark.asyncio
async def test_tool_turn_no_error_message_omits_suffix():
    """``status="error"`` without an ``error`` arg drops the colon suffix."""
    accum = TranscriptAccumulator()
    await accum.append_tool_turn(
        tool_name="t",
        arguments={"x": "y"},
        status="error",
    )
    assert accum.to_list()[0]["content"] == "t(x='y') → error"


# ── Turn-number sequencing ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_numbers_monotonic_across_speakers():
    accum = TranscriptAccumulator()
    await accum.append_user_turn("hello")
    await accum.append_assistant_turn("hi there")
    await accum.append_tool_turn("end_call", {}, "success")
    nums = [t["turn_number"] for t in accum.to_list()]
    assert nums == [1, 2, 3]
    speakers = [t["speaker"] for t in accum.to_list()]
    assert speakers == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_turn_count_matches_appends():
    accum = TranscriptAccumulator()
    for i in range(5):
        await accum.append_user_turn(f"turn {i}")
    assert accum.turn_count() == 5


# ── Concurrent appends preserve ordering ───────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_appends_preserve_unique_turn_numbers():
    """Ten coroutines hammer the accumulator in parallel; turn numbers must
    remain a contiguous 1..N sequence with no duplicates."""
    accum = TranscriptAccumulator()
    await asyncio.gather(*(accum.append_user_turn(f"t{i}") for i in range(10)))
    nums = sorted(t["turn_number"] for t in accum.to_list())
    assert nums == list(range(1, 11))


# ── Serialization shape ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_to_list_returns_serializable_dicts():
    """to_list yields plain dicts with iso timestamps."""
    accum = TranscriptAccumulator()
    await accum.append_user_turn("hello")
    turn = accum.to_list()[0]
    assert isinstance(turn, dict)
    assert set(turn.keys()) == {"turn_number", "speaker", "content", "timestamp"}
    # ISO-8601 round-trips.
    datetime.fromisoformat(turn["timestamp"])


@pytest.mark.asyncio
async def test_to_list_returns_fresh_list_on_each_call():
    """Mutating the returned list doesn't affect the accumulator."""
    accum = TranscriptAccumulator()
    await accum.append_user_turn("a")
    snapshot = accum.to_list()
    snapshot.append({"oops": True})
    # The internal state is untouched.
    assert accum.turn_count() == 1


# ── TranscriptTurn dataclass ───────────────────────────────────────────────


def test_transcript_turn_to_dict():
    """Direct dataclass-to-dict round-trip."""
    turn = TranscriptTurn(
        turn_number=3,
        speaker="user",
        content="hello",
        timestamp=datetime(2026, 5, 4, 12, 0, tzinfo=UTC),
    )
    assert turn.to_dict() == {
        "turn_number": 3,
        "speaker": "user",
        "content": "hello",
        "timestamp": "2026-05-04T12:00:00+00:00",
    }


def test_transcript_turn_is_frozen():
    """The dataclass should be immutable so callers can't mutate accumulator state."""
    turn = TranscriptTurn(
        turn_number=1,
        speaker="user",
        content="x",
        timestamp=datetime.now(UTC),
    )
    with pytest.raises((AttributeError, TypeError)):
        turn.content = "y"  # type: ignore[misc]


# ── Producer-bug guards ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_speaker_raises_value_error():
    """Hand-rolled append with an unknown speaker is a producer bug; fail loud."""
    accum = TranscriptAccumulator()
    with pytest.raises(ValueError, match="Invalid speaker"):
        await accum._append(speaker="bot", content="x", timestamp=None)
