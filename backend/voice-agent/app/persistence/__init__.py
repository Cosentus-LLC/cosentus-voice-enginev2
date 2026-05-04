"""Layer 6 — call persistence.

End-of-call writes to ``voice_calls`` via the lambda. Per call:

* :class:`TranscriptAccumulator` collects user / assistant / tool turns
  in memory while the call runs (Layer 7 / Layer 4 producers).
* :class:`CallRecord` is the end-of-call snapshot serialized to the
  lambda's ``POST /api/calls`` envelope.
* :func:`write_call_record` invokes the lambda — best-effort, never
  raises.
* :func:`trigger_auto_actions` invokes ``POST /api/auto-actions`` for
  derived costs / scores / tasks.
* :func:`run_post_call_analyses` runs Bedrock structured extraction
  over the transcript + case_data using the agent's configured field
  schema.

Layer 7 (observers) plugs the accumulator into the Pipecat pipeline.
Layer 4 (tool executor) already calls back into the accumulator when
a tool finishes. Layer 8 (pipeline builder) wires the end-of-call
flow.
"""

from app.persistence.call_record import CallRecord
from app.persistence.call_writer import trigger_auto_actions, write_call_record
from app.persistence.post_call import run_post_call_analyses
from app.persistence.transcript import TranscriptAccumulator, TranscriptTurn

__all__ = [
    "CallRecord",
    "TranscriptAccumulator",
    "TranscriptTurn",
    "run_post_call_analyses",
    "trigger_auto_actions",
    "write_call_record",
]
