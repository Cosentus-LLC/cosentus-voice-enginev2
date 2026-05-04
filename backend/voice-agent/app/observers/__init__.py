"""Layer 7 — Pipecat observers that bridge frame events to Layer 6.

Two :class:`~pipecat.observers.base_observer.BaseObserver` subclasses:

* :class:`TranscriptObserver` — observes user / assistant text frames
  and routes turns to a per-call
  :class:`~app.persistence.transcript.TranscriptAccumulator`.
* :class:`ErrorObserver` — observes ``ErrorFrame`` and records the
  latest error into a per-call :class:`ErrorState` holder so Layer 8's
  end-of-call ``finally`` block can populate ``CallRecord.error``.

Tool turns are NOT captured by an observer — Layer 4's tool handler
closure (built by Layer 8) calls
:meth:`TranscriptAccumulator.append_tool_turn` directly. End-of-call
write triggering also stays in Layer 8's ``finally`` block, not in
an observer.
"""

from app.observers.error_observer import ErrorObserver
from app.observers.error_state import ErrorState
from app.observers.transcript_observer import TranscriptObserver

__all__ = [
    "ErrorObserver",
    "ErrorState",
    "TranscriptObserver",
]
