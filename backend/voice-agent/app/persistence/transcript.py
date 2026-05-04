"""Per-call transcript accumulator — feeds end-of-call persistence.

The transcript is built in-memory while the call runs. Three
producers append turns:

* Layer 7 user-turn observer → :meth:`TranscriptAccumulator.append_user_turn`
* Layer 7 assistant-turn observer → :meth:`TranscriptAccumulator.append_assistant_turn`
* Layer 4 tool executor → :meth:`TranscriptAccumulator.append_tool_turn`

At end of call, Layer 6's call writer reads the accumulated turns via
:meth:`TranscriptAccumulator.to_list` and ships them to the lambda as
the ``transcript`` field on the ``voice_calls`` upsert.

Tool events live INLINE in the same array (``speaker="tool"``) rather
than in a separate Aurora table — this keeps the schema flat and the
frontend renderer trivially supports a third speaker without a join.
v1 dropped tool events on the floor (only logged to CloudWatch); v2
preserves them in the call record.

Concurrency: an :class:`asyncio.Lock` guards every append so multiple
producer coroutines (the user-turn observer firing while a tool
handler is mid-flight, for example) can't race the turn-number
sequence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Speakers are constrained at the producer side — there's no
# enforcement at the wire boundary because the lambda's ``transcript``
# column is plain JSONB. Frontend currently renders ``user`` and
# ``assistant`` distinctly; ``tool`` is new in v2 and gets its own
# styling once the frontend is rebuilt.
_VALID_SPEAKERS = frozenset({"user", "assistant", "tool"})


@dataclass(frozen=True)
class TranscriptTurn:
    """One turn in a call transcript.

    Attributes:
        turn_number: 1-indexed monotonic sequence within the call.
        speaker: One of ``"user"``, ``"assistant"``, ``"tool"``.
        content: Human-readable text. For ``tool`` turns this is a
            formatted ``tool_name(args) → status`` line — see
            :meth:`TranscriptAccumulator.append_tool_turn`.
        timestamp: When the turn was finalized. UTC.
    """

    turn_number: int
    speaker: str
    content: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the ``voice_calls.transcript`` JSONB shape."""
        return {
            "turn_number": self.turn_number,
            "speaker": self.speaker,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


class TranscriptAccumulator:
    """Per-call in-memory transcript holder.

    No singleton — Layer 8 constructs one accumulator per call at
    pipeline-build time and passes references to Layer 7 observers
    and Layer 4 tool handlers. Reading via :meth:`to_list` is
    lock-free (returns a snapshot copy) so it's safe to call from
    the end-of-call writer concurrently with late-arriving appends
    in the unlikely event a tool finishes during shutdown.
    """

    def __init__(self) -> None:
        self._turns: list[TranscriptTurn] = []
        # Producers run in the same event loop as the pipeline; the
        # lock matters only when two producers race within a single
        # tick (e.g. tool-end and final user transcription landing
        # simultaneously). Cheap given the low contention.
        self._lock = asyncio.Lock()

    async def append_user_turn(
        self,
        content: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a finalized user-side transcription as a turn.

        ``content`` is the post-aggregation user utterance — Layer 7's
        observer is responsible for collecting any interim STT
        results into the final string and only calling here when the
        turn is finalized.
        """
        await self._append(speaker="user", content=content, timestamp=timestamp)

    async def append_assistant_turn(
        self,
        content: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a finalized bot response as a turn.

        ``content`` is the assistant's full text for the turn; Layer 7
        accumulates ``TextFrame`` chunks between
        ``LLMFullResponseStartFrame`` and ``LLMFullResponseEndFrame``
        and calls here once with the joined string.
        """
        await self._append(speaker="assistant", content=content, timestamp=timestamp)

    async def append_tool_turn(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        status: str,
        error: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a tool invocation as a transcript turn.

        Format: ``tool_name(key='val', ...) → status[: error]``.

        Examples::

            transfer_call(target='billing') → success
            press_digit(digits='1234') → error: No SIP session available for DTMF
            end_call() → success

        Arguments are sorted by key for deterministic rendering and
        diffability — matters when two calls of the same tool show up
        next to each other in the UI.

        Args:
            tool_name: The tool's registered name.
            arguments: Tool-call arguments dict. ``repr()`` is used on
                values so strings are quoted; non-string values render
                as ``str(value)`` form via ``%r``.
            status: One of ``"success"``, ``"error"``, ``"timeout"``,
                ``"cancelled"`` — matches Layer 4's
                :class:`~app.tools.result.ToolStatus` enum values.
            error: When ``status != "success"``, an optional human-
                readable error string appended after the arrow.
        """
        args_str = ", ".join(f"{key}={value!r}" for key, value in sorted(arguments.items()))
        content = f"{tool_name}({args_str}) → {status}"
        if error and status != "success":
            content += f": {error}"
        await self._append(speaker="tool", content=content, timestamp=timestamp)

    async def _append(
        self,
        *,
        speaker: str,
        content: str,
        timestamp: datetime | None,
    ) -> None:
        if speaker not in _VALID_SPEAKERS:
            # Producer bug — fail loudly. Unknown speakers would
            # break the frontend's per-speaker styling silently.
            raise ValueError(
                f"Invalid speaker {speaker!r}; must be one of {sorted(_VALID_SPEAKERS)}"
            )
        ts = timestamp if timestamp is not None else datetime.now(UTC)
        async with self._lock:
            self._turns.append(
                TranscriptTurn(
                    turn_number=len(self._turns) + 1,
                    speaker=speaker,
                    content=content,
                    timestamp=ts,
                )
            )

    def to_list(self) -> list[dict[str, Any]]:
        """Snapshot the accumulator as a serializable list.

        Returns a fresh list of plain dicts — safe to mutate without
        affecting the accumulator. The accumulator itself is not
        cleared; calling :meth:`to_list` is non-destructive so the
        end-of-call writer can re-snapshot if a second write is
        triggered (post-call analyses path).
        """
        return [turn.to_dict() for turn in self._turns]

    def turn_count(self) -> int:
        """Total finalized turns recorded so far."""
        return len(self._turns)
