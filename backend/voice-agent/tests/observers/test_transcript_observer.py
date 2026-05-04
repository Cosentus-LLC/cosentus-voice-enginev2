"""Tests for ``app/observers/transcript_observer.py``.

Tests build real Pipecat frames (via the public dataclass
constructors) and a real :class:`TranscriptAccumulator`. Frames
auto-receive unique ``id``s via ``Frame.__post_init__``, so the
dedup tests rely on the actual id-allocator.

``FramePushed`` events are constructed manually with ``MagicMock``
processors as source/destination — observer code never inspects
those fields, so MagicMock is sufficient.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from app.observers import transcript_observer as transcript_observer_module
from app.observers.transcript_observer import TranscriptObserver
from app.persistence.transcript import TranscriptAccumulator
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

# ── Helpers ───────────────────────────────────────────────────────────────


def _push(frame) -> FramePushed:
    """Wrap a frame in a ``FramePushed`` with placeholder source/dest."""
    return FramePushed(
        source=MagicMock(),
        destination=MagicMock(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def _user_frame(text: str) -> TranscriptionFrame:
    """Build a final ``TranscriptionFrame`` for user speech."""
    return TranscriptionFrame(
        text=text,
        user_id="caller-1",
        timestamp="2026-05-04T12:00:00+00:00",
    )


# ── User turns ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transcription_frame_appends_user_turn():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(_user_frame("Hi, I need help.")))

    turns = accum.to_list()
    assert len(turns) == 1
    assert turns[0]["speaker"] == "user"
    assert turns[0]["content"] == "Hi, I need help."


@pytest.mark.asyncio
async def test_empty_transcription_text_is_skipped():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(_user_frame("")))
    await observer.on_push_frame(_push(_user_frame("   ")))

    assert accum.turn_count() == 0


@pytest.mark.asyncio
async def test_transcription_text_is_stripped():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(_user_frame("  hello world  ")))

    assert accum.to_list()[0]["content"] == "hello world"


@pytest.mark.asyncio
async def test_iso_timestamp_parsed_into_datetime():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    frame = TranscriptionFrame(
        text="hi",
        user_id="u",
        timestamp="2026-05-04T12:00:00+00:00",
    )
    await observer.on_push_frame(_push(frame))

    captured_ts = accum.to_list()[0]["timestamp"]
    parsed = datetime.fromisoformat(captured_ts)
    assert parsed.year == 2026 and parsed.month == 5 and parsed.day == 4


@pytest.mark.asyncio
async def test_z_suffix_timestamp_normalized():
    """Some STT providers emit ``Z`` instead of ``+00:00``. Both should parse."""
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    frame = TranscriptionFrame(text="hi", user_id="u", timestamp="2026-05-04T12:00:00Z")
    await observer.on_push_frame(_push(frame))

    parsed = datetime.fromisoformat(accum.to_list()[0]["timestamp"])
    assert parsed.tzinfo is not None


# ── Assistant turns ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_response_lifecycle_appends_assistant_turn():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="Hello")))
    await observer.on_push_frame(_push(LLMTextFrame(text=" there")))
    await observer.on_push_frame(_push(LLMTextFrame(text=", how can I help?")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    turns = accum.to_list()
    assert len(turns) == 1
    assert turns[0]["speaker"] == "assistant"
    assert turns[0]["content"] == "Hello there, how can I help?"


@pytest.mark.asyncio
async def test_start_frame_resets_buffer():
    """A second LLMFullResponseStart while still 'responding' must clear stale chunks."""
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="stale")))
    # Reset before End — buffer should drop the stale chunk.
    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="fresh")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    turns = accum.to_list()
    assert len(turns) == 1
    assert turns[0]["content"] == "fresh"


@pytest.mark.asyncio
async def test_llm_text_frame_outside_response_is_ignored():
    """LLMTextFrame without a preceding Start frame should be dropped."""
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMTextFrame(text="orphan")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    assert accum.turn_count() == 0


@pytest.mark.asyncio
async def test_empty_assistant_turn_not_appended():
    """End frame with no chunks (or all-whitespace chunks) doesn't append."""
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="   ")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    assert accum.turn_count() == 0


@pytest.mark.asyncio
async def test_whitespace_before_punctuation_cleaned():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="Hello ")))
    await observer.on_push_frame(_push(LLMTextFrame(text=" , how can I help ")))
    await observer.on_push_frame(_push(LLMTextFrame(text=" .")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    assert accum.to_list()[0]["content"] == "Hello, how can I help."


@pytest.mark.asyncio
async def test_multiple_spaces_collapsed():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="foo")))
    await observer.on_push_frame(_push(LLMTextFrame(text="    ")))
    await observer.on_push_frame(_push(LLMTextFrame(text="bar")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    assert accum.to_list()[0]["content"] == "foo bar"


@pytest.mark.asyncio
async def test_two_full_responses_produce_two_turns():
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="first")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(LLMTextFrame(text="second")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    turns = accum.to_list()
    assert len(turns) == 2
    assert turns[0]["content"] == "first"
    assert turns[1]["content"] == "second"


@pytest.mark.asyncio
async def test_plain_text_frame_is_not_captured_as_assistant():
    """Layer 7 uses ``LLMTextFrame``, not bare ``TextFrame``.

    A plain ``TextFrame`` (e.g. from a TTS service) inside the LLM
    response window must NOT contribute to the assistant turn.
    """
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_push(TextFrame(text="this is a TTS frame")))
    await observer.on_push_frame(_push(LLMTextFrame(text="this is the LLM frame")))
    await observer.on_push_frame(_push(LLMFullResponseEndFrame()))

    assert accum.to_list()[0]["content"] == "this is the LLM frame"


# ── Frame deduplication ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_same_frame_processed_only_once():
    """Pipecat fires ``on_push_frame`` per processor hop. Dedup by frame.id."""
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    frame = _user_frame("hello")
    # Same frame instance, three different (source, destination) pairs
    # — simulating three processor hops.
    await observer.on_push_frame(_push(frame))
    await observer.on_push_frame(_push(frame))
    await observer.on_push_frame(_push(frame))

    assert accum.turn_count() == 1


@pytest.mark.asyncio
async def test_distinct_frames_with_same_text_each_processed():
    """Two separate user utterances with identical text are still distinct turns."""
    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(_user_frame("yes")))
    await observer.on_push_frame(_push(_user_frame("yes")))

    assert accum.turn_count() == 2


@pytest.mark.asyncio
async def test_dedup_set_is_bounded(monkeypatch):
    """The seen set caps at _MAX_SEEN_FRAMES; eviction keeps it bounded."""
    # Lower the cap so the test is fast.
    monkeypatch.setattr(transcript_observer_module, "_MAX_SEEN_FRAMES", 100)
    monkeypatch.setattr(transcript_observer_module, "_EVICT_BATCH", 10)

    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    # Push 250 distinct user frames. The seen set must stay bounded.
    for i in range(250):
        await observer.on_push_frame(_push(_user_frame(f"turn {i}")))

    assert len(observer._seen_frame_ids) <= 100
    # All 250 were distinct frames so all 250 should have been recorded.
    assert accum.turn_count() == 250


# ── Other frame types are no-ops ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_unrelated_frame_type_is_ignored():
    """A frame the observer doesn't care about doesn't crash or append."""

    @_make_frame_dataclass
    class _UnrelatedFrame:
        pass

    accum = TranscriptAccumulator()
    observer = TranscriptObserver(accum)

    await observer.on_push_frame(_push(_UnrelatedFrame()))

    assert accum.turn_count() == 0


def _make_frame_dataclass(cls):
    """Decorator to create a Pipecat-compatible ``Frame`` subclass on-the-fly."""
    from dataclasses import dataclass

    from pipecat.frames.frames import Frame

    return dataclass(type(cls.__name__, (Frame,), dict(cls.__dict__)))
