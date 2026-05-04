"""Tests for ``app/observers/error_observer.py`` + ``error_state.py``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from app.observers import ErrorObserver, ErrorState
from pipecat.frames.frames import (
    ErrorFrame,
    FatalErrorFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

# ── Helpers ───────────────────────────────────────────────────────────────


def _push(frame) -> FramePushed:
    return FramePushed(
        source=MagicMock(),
        destination=MagicMock(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


# ── ErrorState dataclass ───────────────────────────────────────────────────


def test_default_error_state_has_no_error():
    state = ErrorState()
    assert state.last_error is None
    assert state.last_error_type is None
    assert state.last_error_fatal is False
    assert state.has_error() is False


def test_record_populates_fields():
    state = ErrorState()
    exc = ValueError("oops")
    state.record(error="something broke", exception=exc, fatal=False)

    assert state.last_error == "something broke"
    assert state.last_error_type == "ValueError"
    assert state.last_error_fatal is False
    assert state.has_error() is True


def test_record_truncates_long_error_strings():
    state = ErrorState()
    state.record(error="x" * 5000)
    assert len(state.last_error) == 1000


def test_record_handles_no_exception():
    """``ErrorFrame`` permits ``error="..."`` without an exception."""
    state = ErrorState()
    state.record(error="no exception here")
    assert state.last_error_type is None


def test_last_write_wins():
    """Multiple errors during one call: most recent overwrites."""
    state = ErrorState()
    state.record(error="first")
    state.record(error="second")
    state.record(error="third")
    assert state.last_error == "third"


# ── ErrorObserver routing ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_error_frame_recorded_into_state():
    state = ErrorState()
    observer = ErrorObserver(state)

    frame = ErrorFrame(error="bedrock validation")
    await observer.on_push_frame(_push(frame))

    assert state.last_error == "bedrock validation"
    assert state.last_error_fatal is False


@pytest.mark.asyncio
async def test_error_frame_with_exception_records_type():
    state = ErrorState()
    observer = ErrorObserver(state)

    frame = ErrorFrame(error="thing exploded")
    frame.exception = RuntimeError("backing exception")
    await observer.on_push_frame(_push(frame))

    assert state.last_error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_fatal_error_frame_sets_fatal_flag():
    state = ErrorState()
    observer = ErrorObserver(state)

    frame = FatalErrorFrame(error="unrecoverable")
    await observer.on_push_frame(_push(frame))

    assert state.last_error_fatal is True
    assert state.last_error == "unrecoverable"


@pytest.mark.asyncio
async def test_error_frame_with_fatal_kwarg_sets_fatal_flag():
    """A plain ErrorFrame with ``fatal=True`` is treated the same as FatalErrorFrame."""
    state = ErrorState()
    observer = ErrorObserver(state)

    frame = ErrorFrame(error="mostly recoverable", fatal=True)
    await observer.on_push_frame(_push(frame))

    assert state.last_error_fatal is True


@pytest.mark.asyncio
async def test_multiple_errors_last_write_wins():
    state = ErrorState()
    observer = ErrorObserver(state)

    await observer.on_push_frame(_push(ErrorFrame(error="first")))
    await observer.on_push_frame(_push(ErrorFrame(error="second")))
    await observer.on_push_frame(_push(ErrorFrame(error="third")))

    assert state.last_error == "third"


# ── Frame deduplication ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_same_error_frame_processed_only_once():
    """Pipecat fires ``on_push_frame`` per processor hop. Dedup by frame.id."""
    state = ErrorState()
    observer = ErrorObserver(state)

    frame = ErrorFrame(error="unique")
    # Same frame instance, three pushes — emulate three processor hops.
    await observer.on_push_frame(_push(frame))
    await observer.on_push_frame(_push(frame))
    await observer.on_push_frame(_push(frame))

    # Recorded only once — but since record() is idempotent the
    # state's value is the same regardless. The observable signal is
    # in the seen-id set.
    assert len(observer._seen_frame_ids) == 1


# ── Non-error frames ignored ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_error_frame_ignored():
    """TranscriptionFrames and other non-Error frames must not record into ErrorState."""
    state = ErrorState()
    observer = ErrorObserver(state)

    frame = TranscriptionFrame(text="hi", user_id="u", timestamp="t")
    await observer.on_push_frame(_push(frame))

    assert state.has_error() is False


# ── Observer never raises / never pushes ───────────────────────────────────


@pytest.mark.asyncio
async def test_observer_does_not_push_frames():
    """``BaseObserver`` doesn't expose push_frame; verify by inspection."""
    state = ErrorState()
    observer = ErrorObserver(state)

    # No public push_frame method on BaseObserver — the surface is
    # observation-only. Just verify on_push_frame returns None
    # (no return value implies no side-effect frames).
    frame = ErrorFrame(error="x")
    result = await observer.on_push_frame(_push(frame))
    assert result is None
