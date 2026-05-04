"""Observe Pipecat frame stream and route turns to the transcript accumulator.

Mirrors v1's ``ConversationObserver`` pattern (minus barge-in
tracking and metric emission, which v2 drops). The observer is
read-only — it never pushes frames, never cancels the pipeline,
and never raises.

Frame routing
-------------

* :class:`~pipecat.frames.frames.TranscriptionFrame` (final user
  transcription) → :meth:`~app.persistence.transcript.TranscriptAccumulator.append_user_turn`
* :class:`~pipecat.frames.frames.LLMFullResponseStartFrame` → reset
  the bot-text buffer
* :class:`~pipecat.frames.frames.LLMTextFrame` between Start/End →
  accumulate
* :class:`~pipecat.frames.frames.LLMFullResponseEndFrame` → join the
  buffer (with whitespace cleanup) and append as an assistant turn

Tool turns are NOT captured here — they're appended directly by
Layer 4's tool handler closure (which Layer 8 builds).

Static-opener turns (``speak_first=True``, ``first_message`` non-
empty) are NOT captured here either — those bypass the LLM entirely
via ``TTSSpeakFrame``, so ``LLMFullResponseStart/End`` never fire.
Layer 8's ``on_client_connected`` handler appends the static opener
to the accumulator directly before queueing the TTS frame.

Frame deduplication
-------------------

Pipecat's ``BaseObserver.on_push_frame`` fires once per processor
hop, so a single ``TranscriptionFrame`` triggers N invocations
across the pipeline (one per processor that touches the frame). We
dedup via ``frame.id`` per observer instance.

We considered Pipecat 1.1.0's ``broadcast_sibling_id`` field but
verified it is only set when a processor explicitly calls
``broadcast_frame()`` / ``broadcast_frame_instance()``. Inspection
of ``site-packages/pipecat`` confirms none of the frame types this
observer cares about (``TranscriptionFrame``, ``LLMTextFrame``,
``LLMFullResponseStart/EndFrame``) are ever broadcast — they all
travel via normal :meth:`~pipecat.processors.frame_processor.FrameProcessor.push_frame`.
For our frames ``broadcast_sibling_id`` is always ``None``, so the
``frame.id`` set is the only thing that matters in practice.

The dedup set is bounded — long calls (Cosentus IVR sessions can run
15+ minutes with hundreds of frames) shouldn't grow the set
unbounded. v1's helper had no cap; v2 caps at
:data:`_MAX_SEEN_FRAMES` and evicts the oldest 10% when full.
"""

from __future__ import annotations

import re
from datetime import datetime

import structlog
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from app.persistence.transcript import TranscriptAccumulator

logger = structlog.get_logger(__name__)


# Whitespace cleanup matching v1's ``ConversationObserver._flush_bot_response``.
# LLM token streams often emit fragments like ``"Hello "`` followed by
# ``"."`` on a new chunk; naive concatenation produces ``"Hello ."``
# which the operator-facing transcript should clean up.
_PUNCT_SPACE_RE = re.compile(r"\s+([.,!?;:'])")
_MULTI_SPACE_RE = re.compile(r"\s+")


# Bound the per-observer ``_seen`` set. Picked an order of magnitude
# above what a worst-case 30-minute call would produce — typical
# Cosentus calls land at 200–600 frames per minute through the
# observer, so 10k covers a long edge case and the eviction logic
# protects against pathological loops.
_MAX_SEEN_FRAMES = 10_000

# When the seen set reaches the cap, drop this many old IDs to make
# room. 10% lets us amortize the eviction cost over many subsequent
# appends rather than thrashing on every single new frame.
_EVICT_BATCH = _MAX_SEEN_FRAMES // 10


class TranscriptObserver(BaseObserver):
    """Pipecat observer that captures user + assistant turns into Layer 6.

    State machine:

    * ``_llm_responding`` — bool flag, set on
      ``LLMFullResponseStartFrame``, cleared on
      ``LLMFullResponseEndFrame``.
    * ``_current_bot_chunks`` — ``list[str]`` accumulating
      ``LLMTextFrame.text`` while ``_llm_responding`` is true.
    * ``_seen_frame_ids`` — bounded ``set[int]`` for dedup.

    All state is per-observer-instance, so per-call when Layer 8
    constructs one observer per pipeline.
    """

    def __init__(self, accumulator: TranscriptAccumulator) -> None:
        super().__init__()
        self._accumulator = accumulator
        self._llm_responding: bool = False
        self._current_bot_chunks: list[str] = []
        self._seen_frame_ids: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        """Route every push to the appropriate handler.

        ``BaseObserver`` contract: this method receives every frame
        that flows through the pipeline. We dedup, type-dispatch, and
        return — never raise, never push.
        """
        frame = data.frame

        if not self._is_new_frame(frame):
            return

        if isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_responding = True
            self._current_bot_chunks = []
            return

        if isinstance(frame, LLMTextFrame) and self._llm_responding:
            if frame.text:
                self._current_bot_chunks.append(frame.text)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._llm_responding:
                await self._flush_bot_response()
            self._llm_responding = False
            return

    async def _handle_transcription(self, frame: TranscriptionFrame) -> None:
        text = frame.text.strip() if frame.text else ""
        if not text:
            return

        timestamp = _parse_timestamp(frame.timestamp)
        await self._accumulator.append_user_turn(content=text, timestamp=timestamp)
        logger.debug(
            "transcript_user_turn_captured",
            text_chars=len(text),
        )

    async def _flush_bot_response(self) -> None:
        """Join the buffer, clean whitespace, append as one assistant turn.

        v1 joined chunks with spaces; v2 joins with empty string
        because Pipecat's LLM services emit ``LLMTextFrame`` per
        token and tokens already contain their own boundary
        whitespace (Anthropic / Bedrock tokens are subword units
        like ``"Hello"`` then ``" there"``). The post-join
        whitespace cleanup handles any pathological "stuck punctuation"
        cases the same way v1 did.
        """
        if not self._current_bot_chunks:
            return

        text = "".join(self._current_bot_chunks).strip()
        if not text:
            self._current_bot_chunks = []
            return

        # ``"Hello , how can I help ."`` → ``"Hello, how can I help."``
        text = _PUNCT_SPACE_RE.sub(r"\1", text)
        # ``"foo  bar"`` → ``"foo bar"`` — collapse any run of whitespace
        # that may have leaked through token boundaries.
        text = _MULTI_SPACE_RE.sub(" ", text)

        await self._accumulator.append_assistant_turn(content=text)
        logger.debug(
            "transcript_assistant_turn_captured",
            text_chars=len(text),
            chunk_count=len(self._current_bot_chunks),
        )
        self._current_bot_chunks = []

    def _is_new_frame(self, frame: Frame) -> bool:
        """Dedup by ``frame.id``. Returns ``True`` if this is the first sighting.

        ``broadcast_sibling_id`` would be the more elegant key for
        broadcast pairs, but verification of Pipecat 1.1.0 confirms
        none of the frame types this observer cares about ever pass
        through ``broadcast_frame()`` — so the field is always
        ``None`` for us. ``frame.id`` is the right key.

        Bounded set: when the cap is hit, evict the first
        :data:`_EVICT_BATCH` IDs (insertion order in Python 3.7+ sets
        is not guaranteed, but for our use case — sequential frame
        IDs that monotonically increase — any eviction strategy is
        safe because evicted IDs won't reappear).
        """
        if frame.id in self._seen_frame_ids:
            return False

        self._seen_frame_ids.add(frame.id)

        if len(self._seen_frame_ids) > _MAX_SEEN_FRAMES:
            # Evict oldest IDs (per the comment above, "oldest" means
            # "lowest ID" given Pipecat's monotonic counter, but any
            # subset is safe). ``sorted()`` is O(n log n) but only
            # runs once every _MAX_SEEN_FRAMES additions, so amortized
            # cost is negligible.
            keep = sorted(self._seen_frame_ids)[_EVICT_BATCH:]
            self._seen_frame_ids = set(keep)

        return True


def _parse_timestamp(timestamp: object) -> datetime | None:
    """Parse :class:`~pipecat.frames.frames.TranscriptionFrame.timestamp`.

    Pipecat declares the field as ``str`` but in practice STT
    services may emit ISO-8601 strings, ``datetime`` instances, or
    ``None`` depending on the provider. We accept all three and let
    a malformed value fall through to ``None`` — the accumulator
    will fill in ``datetime.now(UTC)`` as a default.
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
