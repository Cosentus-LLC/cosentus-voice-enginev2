"""Layer 7 user/assistant turn capture — Pipecat aggregator event handlers.

Pipecat 1.1.0's :class:`~pipecat.processors.aggregators.llm_response_universal.LLMUserAggregator`
and :class:`~pipecat.processors.aggregators.llm_response_universal.LLMAssistantAggregator`
expose ``on_user_turn_stopped`` and ``on_assistant_turn_stopped``
event handlers that fire **exactly once per turn** with the fully-
aggregated text plus a turn-start timestamp (and, on the assistant
side, an ``interrupted`` flag). This module wires those handlers to
a :class:`~app.persistence.transcript.TranscriptAccumulator`.

History
-------

The earlier v2 implementation subclassed
:class:`~pipecat.observers.base_observer.BaseObserver` and
reconstructed turns by hand from the frame stream:
``LLMFullResponseStartFrame`` → N×``LLMTextFrame`` →
``LLMFullResponseEndFrame``, plus a bounded ``frame.id`` dedup set
because ``on_push_frame`` fires per processor hop, plus regex-based
whitespace cleanup. That worked but was ~120 lines of state machine
duplicating Pipecat's own aggregation logic.

The May 2026 codebase audit found that Pipecat ships canonical
event handlers covering this exact use case and recommended
switching. Benefits of the rewrite:

* One callback per turn instead of N callbacks per processor hop —
  no dedup set, no eviction logic.
* No state machine — the aggregator owns the assistant-turn buffer.
* No regex whitespace cleanup — Pipecat applies inter-frame space
  handling internally during aggregation.
* New :attr:`AssistantTurnStoppedMessage.interrupted` signal lands
  in :class:`~app.persistence.transcript.TranscriptTurn.interrupted`
  for free — couldn't be derived from the frame stream alone.

Tool turns
----------

Tool turns are NOT covered by the aggregator events. Layer 4's tool
handler closure (built in :func:`~app.bot.bot.run_bot`) calls
:meth:`~app.persistence.transcript.TranscriptAccumulator.append_tool_turn`
directly after the executor completes. That path is unchanged.

Static opener
-------------

The static-opener path (``speak_first=True`` with a non-empty
``first_message``) bypasses the LLM entirely via ``TTSSpeakFrame`` —
no ``LLMFullResponseStart/End`` frames fire and the aggregator does
not see the utterance. Layer 8 calls
:meth:`TranscriptAccumulator.append_assistant_turn` directly in
that path, the same as the prior implementation. Dynamic openers
DO go through the aggregator and are captured by these handlers.
"""

from __future__ import annotations

from datetime import datetime

import structlog

# Forward-reference imports keep this module light: Pipecat types
# are needed only at type-check time inside handler signatures, but
# we import the messages at runtime because their attribute access
# (``message.content``, ``message.interrupted``) happens here.
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
)

from app.persistence.transcript import TranscriptAccumulator

logger = structlog.get_logger(__name__)


def wire_transcript_handlers(
    aggregator_pair: LLMContextAggregatorPair,
    accumulator: TranscriptAccumulator,
) -> None:
    """Attach turn-capture handlers to the aggregator pair.

    Call once per pipeline construction, after
    :class:`LLMContextAggregatorPair` is built and before the
    pipeline starts. The handlers persist for the life of the
    aggregator instances (one per call in v2's per-call construction
    pattern).

    The handlers are best-effort: a malformed timestamp falls back
    to ``datetime.now(UTC)`` inside the accumulator; an exception in
    one handler logs but doesn't propagate (preserves the existing
    Layer 7 contract — observation MUST NOT raise into the
    pipeline).

    Args:
        aggregator_pair: Pipecat's user+assistant aggregator pair.
            Built by :func:`~app.bot.bot.run_bot` before pipeline
            assembly.
        accumulator: Layer 6 transcript accumulator that will hold
            user and assistant turns. Tool turns are appended
            directly by Layer 4's tool handler closure (see module
            docstring).
    """
    user_aggregator = aggregator_pair.user()
    assistant_aggregator = aggregator_pair.assistant()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def _on_user_turn_stopped(
        _aggregator,
        _strategy,
        message: UserTurnStoppedMessage,
    ) -> None:
        try:
            await _capture_user_turn(accumulator, message)
        except Exception as exc:  # noqa: BLE001 — observation must never raise
            logger.exception(
                "transcript_user_handler_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def _on_assistant_turn_stopped(
        _aggregator,
        message: AssistantTurnStoppedMessage,
    ) -> None:
        try:
            await _capture_assistant_turn(accumulator, message)
        except Exception as exc:  # noqa: BLE001 — observation must never raise
            logger.exception(
                "transcript_assistant_handler_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )


async def _capture_user_turn(
    accumulator: TranscriptAccumulator,
    message: UserTurnStoppedMessage,
) -> None:
    """Persist the user-side turn if non-empty."""
    text = (message.content or "").strip()
    if not text:
        return

    timestamp = _parse_timestamp(message.timestamp)
    await accumulator.append_user_turn(content=text, timestamp=timestamp)
    logger.debug(
        "transcript_user_turn_captured",
        text_chars=len(text),
    )


async def _capture_assistant_turn(
    accumulator: TranscriptAccumulator,
    message: AssistantTurnStoppedMessage,
) -> None:
    """Persist the assistant-side turn if non-empty.

    Pipecat may emit ``AssistantTurnStoppedMessage`` with empty
    ``content`` when the turn was interrupted before any tokens
    streamed. Skip those — there's nothing meaningful to record and
    a blank turn would clutter the operator-facing transcript.
    """
    text = (message.content or "").strip()
    if not text:
        return

    timestamp = _parse_timestamp(message.timestamp)
    await accumulator.append_assistant_turn(
        content=text,
        timestamp=timestamp,
        interrupted=bool(message.interrupted),
    )
    logger.debug(
        "transcript_assistant_turn_captured",
        text_chars=len(text),
        interrupted=bool(message.interrupted),
    )


def _parse_timestamp(timestamp: object) -> datetime | None:
    """Parse Pipecat's ISO-8601 timestamp string.

    Both :class:`UserTurnStoppedMessage` and
    :class:`AssistantTurnStoppedMessage` declare ``timestamp`` as
    ``str`` (ISO-8601 from :func:`time_now_iso8601`). Defensive: also
    accept :class:`datetime` and ``None`` so we can survive a future
    Pipecat type change without a regression.
    """
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
    return None
