"""Layer 7 — Pipecat observers + aggregator-event hookups for Layer 6.

* :func:`wire_transcript_handlers` — attaches
  ``on_user_turn_stopped`` / ``on_assistant_turn_stopped`` handlers
  to a Pipecat
  :class:`~pipecat.processors.aggregators.llm_response_universal.LLMContextAggregatorPair`
  so user and assistant turns flow into a per-call
  :class:`~app.persistence.transcript.TranscriptAccumulator`.
* :class:`ErrorObserver` — :class:`~pipecat.observers.base_observer.BaseObserver`
  subclass that observes ``ErrorFrame`` and records the latest error
  into a per-call :class:`ErrorState` holder so Layer 8's
  end-of-call ``finally`` block can populate ``CallRecord.error``.

Tool turns are NOT captured by either path — Layer 4's tool handler
closure (built by Layer 8) calls
:meth:`TranscriptAccumulator.append_tool_turn` directly. End-of-call
write triggering also stays in Layer 8's ``finally`` block.
"""

from app.observers.error_observer import ErrorObserver
from app.observers.error_state import ErrorState
from app.observers.transcript_observer import wire_transcript_handlers

__all__ = [
    "ErrorObserver",
    "ErrorState",
    "wire_transcript_handlers",
]
