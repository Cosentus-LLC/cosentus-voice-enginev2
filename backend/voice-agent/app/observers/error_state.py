"""Per-call error holder — populated by ErrorObserver, read by Layer 8.

When a Pipecat ``ErrorFrame`` propagates through the pipeline,
:class:`~app.observers.error_observer.ErrorObserver` records the
details here. Layer 8's end-of-call ``finally`` block reads
:attr:`ErrorState.last_error` when constructing the
``CallRecord.error`` field — picking up errors that v1 silently
dropped (Bedrock validation, AssemblyAI disconnects, ElevenLabs
quota, transport hiccups).

No singleton. One :class:`ErrorState` per call, constructed at
pipeline-build time and passed both to :class:`ErrorObserver` and
to whatever code path constructs the final :class:`~app.persistence.call_record.CallRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass

# Match Layer 6's ``CallRecord.error`` truncation cap. Frontend
# rendering looks bad when stack traces or 50-line API responses
# end up in the error column. The cap is the same value used by
# :func:`app.persistence.call_record._truncate_error`.
_ERROR_TRUNCATION_CHARS = 1000


@dataclass
class ErrorState:
    """Holder for the most recent ``ErrorFrame`` observed on the pipeline.

    Last-write-wins. Multiple errors during a call (e.g. an STT
    websocket reconnect followed by a Bedrock validation) overwrite
    earlier values — the typical operator triage interest is "what
    finally killed this call", which is the most recent error.
    """

    last_error: str | None = None
    """Human-readable error string from ``ErrorFrame.error``,
    truncated to :data:`_ERROR_TRUNCATION_CHARS`. ``None`` when no
    ``ErrorFrame`` has been observed yet."""

    last_error_type: str | None = None
    """``type(ErrorFrame.exception).__name__`` when the frame
    carried a Python exception. ``None`` when the frame had no
    ``exception`` attribute (Pipecat permits ``ErrorFrame(error="...")``
    without a backing exception)."""

    last_error_fatal: bool = False
    """``True`` when the observed frame was a
    :class:`~pipecat.frames.frames.FatalErrorFrame` (or set
    ``fatal=True``). Layer 8 may use this to escalate logging / alert
    routing; the call still terminates the same way."""

    def record(
        self,
        error: str,
        exception: Exception | None = None,
        fatal: bool = False,
    ) -> None:
        """Record an observed error. Last write wins.

        Args:
            error: ``ErrorFrame.error`` text. Truncated to
                :data:`_ERROR_TRUNCATION_CHARS` to keep the
                ``voice_calls.error`` column UI-friendly.
            exception: Optional underlying Python exception
                (``ErrorFrame.exception``).
                :attr:`last_error_type` is captured from this.
            fatal: ``True`` for :class:`~pipecat.frames.frames.FatalErrorFrame`.
        """
        self.last_error = error[:_ERROR_TRUNCATION_CHARS] if error else None
        self.last_error_type = type(exception).__name__ if exception else None
        self.last_error_fatal = fatal

    def has_error(self) -> bool:
        """Return ``True`` if any error has been recorded."""
        return self.last_error is not None
