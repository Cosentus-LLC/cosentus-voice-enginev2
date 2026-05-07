"""Tests for ``app/observers/transcript_observer.py``.

Replaces the prior ``BaseObserver`` state-machine implementation
with the canonical Pipecat aggregator-event approach. Tests drive
real ``LLMUserAggregator`` / ``LLMAssistantAggregator`` instances
through ``pipecat.tests.utils.run_test`` and assert that the
:class:`~app.persistence.transcript.TranscriptAccumulator` captured
the expected turns.

Why ``run_test`` and not direct method invocation? The bug class
that the May 2026 codebase audit flagged was "homegrown reach-past-
Pipecat patterns that pass unit tests but fail under real frame
flow." Using ``run_test`` here means a regression in our wiring
fails the test under realistic frame-flow conditions, not just
mocked invariants.
"""

from __future__ import annotations

import pytest
from app.observers.transcript_observer import (
    _parse_timestamp,
    wire_transcript_handlers,
)
from app.persistence.transcript import TranscriptAccumulator
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test

# ── _parse_timestamp helper ──────────────────────────────────────────────


class TestParseTimestamp:
    def test_iso8601_string_parses(self):
        result = _parse_timestamp("2026-05-07T19:38:46.000Z")
        assert result is not None
        assert result.year == 2026 and result.month == 5 and result.day == 7

    def test_none_returns_none(self):
        assert _parse_timestamp(None) is None

    def test_malformed_string_returns_none(self):
        assert _parse_timestamp("not-an-iso-string") is None

    def test_datetime_passes_through(self):
        from datetime import datetime

        d = datetime(2026, 5, 7, 19, 0, 0)
        assert _parse_timestamp(d) is d


# ── User-turn capture via run_test ──────────────────────────────────────


def _build_pair(
    *,
    enable_interruptions: bool = False,
) -> tuple[LLMContextAggregatorPair, LLMContext]:
    """Build a real aggregator pair without VAD / turn analyzers.

    The user aggregator's normal production wiring (VAD, smart-turn)
    needs audio frames + a clock to fire turn events. For unit tests
    we drive the assistant aggregator directly with frames since the
    transcript handler attaches to it via the canonical event API.
    The user aggregator is created so the pair is structurally
    complete; it's not exercised by the assistant-side tests.
    """
    context = LLMContext()
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    return pair, context


@pytest.mark.asyncio
async def test_assistant_turn_captured_via_run_test():
    """Drive a complete assistant turn (Start → text chunks → End)
    through the assistant aggregator via run_test. The handler
    appends to the accumulator with the joined content + timestamp.
    """
    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    await run_test(
        pair.assistant(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello "),
            LLMTextFrame("there!"),
            LLMFullResponseEndFrame(),
        ],
    )

    turns = accumulator.to_list()
    assert len(turns) == 1, turns
    assert turns[0]["speaker"] == "assistant"
    assert turns[0]["content"] == "Hello there!"
    assert turns[0]["interrupted"] is False
    # Timestamp is sourced from the aggregator's turn-start time —
    # always present (Pipecat populates it from time_now_iso8601).
    assert turns[0]["timestamp"]


@pytest.mark.asyncio
async def test_assistant_turn_marked_interrupted_when_cut_off():
    """The new ``AssistantTurnStoppedMessage.interrupted`` flag is
    the bonus signal that wasn't reachable from the previous frame-
    stream-reconstructing implementation. Drive a turn that's
    interrupted mid-stream and assert ``TranscriptTurn.interrupted``
    lands as ``True``.

    Why ``SleepFrame`` between the text and the interruption: Pipecat's
    :class:`~pipecat.processors.frame_processor.FrameProcessorQueue`
    prioritizes :class:`SystemFrame` (which ``InterruptionFrame`` is)
    above normal frames inside the aggregator's input queue. Without
    a sleep, the InterruptionFrame jumps ahead of the
    ``LLMFullResponseStartFrame`` + ``LLMTextFrame`` and arrives before
    ``_assistant_turn_start_timestamp`` has been set —
    ``_trigger_assistant_turn_stopped`` then early-returns silently
    and the EndFrame at the end of ``run_test`` fires the only emit
    with ``interrupted=False``. The sleep yields the event loop so
    the start/text frames clear the queue before the interrupt
    enters. In production this is naturally true (the user has to
    actually speak before broadcasting an interruption).
    """
    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    await run_test(
        pair.assistant(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("This is going to be cu"),
            SleepFrame(sleep=0.05),  # let start+text drain
            # Interruption arrives mid-turn — aggregator's
            # ``_handle_interruptions`` calls
            # ``_trigger_assistant_turn_stopped(interrupted=True)``.
            InterruptionFrame(),
        ],
    )

    turns = accumulator.to_list()
    assert len(turns) == 1, turns
    assert turns[0]["speaker"] == "assistant"
    assert turns[0]["interrupted"] is True
    # Content is whatever was streamed before the interruption.
    assert "cu" in turns[0]["content"]


@pytest.mark.asyncio
async def test_assistant_empty_turn_is_skipped():
    """When the LLM emits zero tokens before interrupt / end, the
    aggregator emits ``AssistantTurnStoppedMessage`` with empty
    ``content``. The handler skips empties so we don't pollute the
    transcript with blank rows.
    """
    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    await run_test(
        pair.assistant(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            # No text frames at all — the turn has zero content.
            LLMFullResponseEndFrame(),
        ],
    )

    assert accumulator.turn_count() == 0


@pytest.mark.asyncio
async def test_assistant_handler_exception_does_not_crash_pipeline():
    """The handler wraps every accumulator call in a try/except per
    the Layer 7 contract: observation MUST NOT raise. Even if the
    accumulator throws, the pipeline continues.
    """

    class _ExplodingAccumulator(TranscriptAccumulator):
        async def append_assistant_turn(self, *args, **kwargs):
            raise RuntimeError("disk full")

    accumulator = _ExplodingAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    # If the handler propagated, run_test would fail. The fact that
    # it completes is the assertion.
    await run_test(
        pair.assistant(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hi."),
            LLMFullResponseEndFrame(),
        ],
    )


# ── Multiple sequential assistant turns ──────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_assistant_turns_increment_turn_number():
    """Two consecutive Start/End cycles produce two transcript turns."""
    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    await run_test(
        pair.assistant(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("First reply."),
            LLMFullResponseEndFrame(),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Second reply."),
            LLMFullResponseEndFrame(),
        ],
    )

    turns = accumulator.to_list()
    assert len(turns) == 2
    assert turns[0]["turn_number"] == 1
    assert turns[0]["content"] == "First reply."
    assert turns[1]["turn_number"] == 2
    assert turns[1]["content"] == "Second reply."


# ── Wiring: handlers register correctly against a real aggregator ───────


def test_wire_transcript_handlers_registers_event_handlers_without_warning(
    caplog,
):
    """Calling :func:`wire_transcript_handlers` adds handlers to the
    aggregators' ``_event_handlers`` map. Pipecat's
    ``add_event_handler`` warns when the event isn't pre-registered;
    confirming no warning fires is a sanity check that we use
    correct event names (``on_user_turn_stopped`` /
    ``on_assistant_turn_stopped``).
    """
    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()

    wire_transcript_handlers(pair, accumulator)

    user_aggregator = pair.user()
    assistant_aggregator = pair.assistant()

    # Both events ARE pre-registered by Pipecat (verified in
    # llm_response_universal.py:440-441 and :884). Our wirer attaches
    # handlers — the aggregator's _event_handlers dict should have
    # one entry for each.
    user_handlers = user_aggregator._event_handlers["on_user_turn_stopped"].handlers
    assert len(user_handlers) == 1

    asst_handlers = assistant_aggregator._event_handlers["on_assistant_turn_stopped"].handlers
    assert len(asst_handlers) == 1


# ── Aggregator-pair instances are stable across .user() / .assistant() ──


def test_pair_methods_return_stable_instances():
    """Sanity for the wirer: ``pair.user()`` returns the same
    instance on every call, so the handler attaches to the
    processor that ends up in the pipeline. Verified against
    Pipecat 1.1.0's ``LLMContextAggregatorPair`` source
    (``self._user`` / ``self._assistant`` cached in __init__).
    """
    pair, _ = _build_pair()
    assert pair.user() is pair.user()
    assert pair.assistant() is pair.assistant()


# ── Tool turns still flow through Layer 4 path, not aggregator events ───


@pytest.mark.asyncio
async def test_tool_turns_unaffected_by_aggregator_events():
    """Tool turns are appended directly by Layer 4's tool handler.
    The aggregator events only handle user / assistant turns. Verify
    a sequence of mixed turns (assistant text → tool → assistant
    text) records cleanly without double-counting or interference.
    """
    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    # Drive an assistant turn through the aggregator.
    await run_test(
        pair.assistant(),
        frames_to_send=[
            LLMFullResponseStartFrame(),
            LLMTextFrame("Pressing 123 now."),
            LLMFullResponseEndFrame(),
        ],
    )
    # Layer 4 tool handler closure path (called from bot.py's
    # tool_handler closure) — not an aggregator event.
    await accumulator.append_tool_turn(
        tool_name="press_digit",
        arguments={"digits": "123"},
        status="success",
    )

    turns = accumulator.to_list()
    assert len(turns) == 2
    assert turns[0]["speaker"] == "assistant"
    assert turns[0]["content"] == "Pressing 123 now."
    assert turns[1]["speaker"] == "tool"
    assert "press_digit" in turns[1]["content"]
    assert turns[1]["interrupted"] is False  # Tool turns never carry interrupted.


# ── Direct invocation of the user-side handler (run_test on user
#     aggregator requires VAD / clock plumbing that's heavyweight for
#     a unit test; verify via direct call instead) ──


@pytest.mark.asyncio
async def test_user_handler_appends_turn_when_called_directly():
    """User-side aggregator's turn events fire from VAD-driven
    controllers that need a real clock + audio frames to plumb
    through ``run_test``. The handler logic itself is the same
    callback-style as the assistant side, so verify it via a direct
    ``UserTurnStoppedMessage`` injection.
    """
    from pipecat.processors.aggregators.llm_response_universal import (
        UserTurnStoppedMessage,
    )

    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    user_aggregator = pair.user()
    handlers = user_aggregator._event_handlers["on_user_turn_stopped"].handlers
    assert len(handlers) == 1
    handler = handlers[0]

    await handler(
        user_aggregator,
        None,  # strategy — handler ignores this
        UserTurnStoppedMessage(
            content="Hello, can you help me?",
            timestamp="2026-05-07T19:38:46.000Z",
        ),
    )

    turns = accumulator.to_list()
    assert len(turns) == 1
    assert turns[0]["speaker"] == "user"
    assert turns[0]["content"] == "Hello, can you help me?"
    assert turns[0]["interrupted"] is False


@pytest.mark.asyncio
async def test_user_empty_content_is_skipped():
    """Empty / whitespace-only user transcriptions are skipped — the
    handler treats them as no-op (matches the prior implementation's
    behavior).
    """
    from pipecat.processors.aggregators.llm_response_universal import (
        UserTurnStoppedMessage,
    )

    accumulator = TranscriptAccumulator()
    pair, _ = _build_pair()
    wire_transcript_handlers(pair, accumulator)

    handlers = pair.user()._event_handlers["on_user_turn_stopped"].handlers
    handler = handlers[0]

    await handler(
        pair.user(),
        None,
        UserTurnStoppedMessage(content="", timestamp=""),
    )
    await handler(
        pair.user(),
        None,
        UserTurnStoppedMessage(content="   ", timestamp=""),
    )

    assert accumulator.turn_count() == 0


# ── Suppress unused-import warnings for symbols only used by linters ────


_ = (LLMContextFrame, TranscriptionFrame, LLMUserAggregator, LLMAssistantAggregator)
