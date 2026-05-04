"""Observe ErrorFrames; record latest into ErrorState holder.

Layer 7's other half. v1 had no equivalent — Pipecat-internal
``ErrorFrame`` events (the Bedrock validation error in walking-
skeleton round 3 was the canonical example) never made it to v1's
``CallRecord.error`` because v1's ``service_main.py`` ``finally``
block only captured exceptions from the surrounding
``try``/``except``. Frames flowing through the pipeline never
reached that scope.

v2 closes that gap with this observer:

* :class:`~pipecat.frames.frames.ErrorFrame` →
  :meth:`~app.observers.error_state.ErrorState.record`
* :class:`~pipecat.frames.frames.FatalErrorFrame` (subclass) → same,
  with ``fatal=True``

What this observer does NOT do
------------------------------

* It does NOT cancel the pipeline. Pipecat already handles
  ``FatalErrorFrame`` lifecycle via its own task management
  (``PipelineTask`` cancels the runner on fatal errors). Cancelling
  here would race that path and risk a double-cancel.
* It does NOT retry, redirect, or transform the error. Layer 7 is
  read-only — record + observe.
* It does NOT push frames. ``BaseObserver`` doesn't have a
  ``push_frame`` method; observers are notification-only.

Layer 8's end-of-call ``finally`` block reads
:attr:`ErrorState.last_error` and uses it as the value for
``CallRecord.error`` when the pipeline coroutine itself didn't
catch a Python exception (the typical case for Pipecat-internal
errors that surface as ``ErrorFrame``).
"""

from __future__ import annotations

import structlog
from pipecat.frames.frames import ErrorFrame, FatalErrorFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

from app.observers.error_state import ErrorState

logger = structlog.get_logger(__name__)


class ErrorObserver(BaseObserver):
    """Records the latest ``ErrorFrame`` into a per-call :class:`ErrorState`.

    Last-write-wins: when multiple errors fire during a call (e.g.
    a Bedrock validation followed by a transport reconnect), the
    most recent overwrites earlier values — matches operator
    triage interest ("what finally killed the call").
    """

    def __init__(self, error_state: ErrorState) -> None:
        super().__init__()
        self._error_state = error_state
        # Per-frame.id dedup. Same rationale as ``TranscriptObserver``:
        # ``on_push_frame`` fires per processor hop, so a single
        # ``ErrorFrame`` triggers N calls. ``ErrorFrames`` aren't
        # broadcast (no callers of ``broadcast_frame(ErrorFrame, ...)``
        # exist in Pipecat 1.1.0), so ``frame.id`` is the right dedup
        # key. Bounded growth isn't a concern here — error frames are
        # rare relative to text frames.
        self._seen_frame_ids: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        """Inspect the frame; record if it's an ``ErrorFrame``."""
        frame = data.frame

        if not isinstance(frame, ErrorFrame):
            return

        if frame.id in self._seen_frame_ids:
            return
        self._seen_frame_ids.add(frame.id)

        # ``isinstance(..., FatalErrorFrame)`` is the first-class check.
        # ``getattr(..., "fatal", False)`` is a defensive fallback in
        # case a future Pipecat version exposes ``fatal=True`` on a
        # plain ``ErrorFrame`` without subclassing.
        fatal = isinstance(frame, FatalErrorFrame) or getattr(frame, "fatal", False)
        exception = getattr(frame, "exception", None)

        self._error_state.record(
            error=frame.error,
            exception=exception,
            fatal=fatal,
        )

        logger.warning(
            "error_frame_observed",
            error=frame.error[:200] if frame.error else "",
            fatal=fatal,
            exception_type=type(exception).__name__ if exception else None,
        )
