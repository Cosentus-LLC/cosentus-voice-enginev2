"""``end_call`` tool — clean pipeline termination via :class:`EndTaskFrame`.

Pushes an :class:`EndTaskFrame` upstream. PipelineTask catches it,
flushes the in-flight queue (so the LLM's "Thanks, have a good
one" TTS audio finishes playing), then emits an ``EndFrame``
downstream that drains the rest of the pipeline.

Why ``EndTaskFrame`` upstream instead of ``EndFrame`` downstream?

The earlier v2 revision queued ``EndFrame`` directly via
``context.queue_frame``. That works in the happy case but races
with concurrent ``InterruptionFrame`` traffic: see open Pipecat
issue `#3757 <https://github.com/pipecat-ai/pipecat/issues/3757>`_
("EndFrame can hang the pipeline indefinitely when racing with
InterruptionFrame"). The Pipecat-canonical pattern for tools
requesting graceful shutdown is ``EndTaskFrame`` upstream — see
the `Pipecat end-pipeline guide
<https://docs.pipecat.ai/guides/fundamentals/end-pipeline>`_.

This is the same architectural realignment that closed Phase 2
for hangup paths in ``app/bot/bot.py`` (where ``queue_frames(
[EndFrame()])`` was replaced by ``task.cancel(reason=...)``).

``cancel_on_interruption`` uses Pipecat's documented default
(``True``), but the result explicitly sets ``run_llm=False``. Once
``EndTaskFrame`` is queued, the call is terminal; a follow-up LLM
run can leak step machinery or summaries after the goodbye while
the pipeline is draining.

The ``reason`` argument is audit-log only — the call ends
regardless of value. We accept it to keep the LLM's tool-use
schema human-readable; many system prompts encourage Claude to
state a reason for clarity in transcripts.
"""

from __future__ import annotations

import structlog
from pipecat.frames.frames import EndTaskFrame
from pipecat.processors.frame_processor import FrameDirection

from app.tools.context import ToolContext
from app.tools.result import ToolResult, error_result, success_result
from app.tools.schema import ToolDefinition, ToolParameter

logger = structlog.get_logger(__name__)


DESCRIPTION_DEFAULT = (
    "End the current phone call. Use this ONLY after the conversation "
    "has reached a natural conclusion: the customer's issue is fully "
    "resolved, you have confirmed they don't need further help, and "
    "you have said goodbye. Never hang up while the customer is still "
    "speaking or has unresolved questions."
)


async def end_call_executor(
    arguments: dict,
    context: ToolContext,
) -> ToolResult:
    """Push an :class:`EndTaskFrame` upstream to terminate the pipeline."""
    reason = (arguments.get("reason") or "").strip() or "Conversation concluded"

    if context.queue_frame is None:
        return error_result("No frame queue available; cannot end call")

    try:
        # Upstream EndTaskFrame: the canonical Pipecat pattern for a
        # tool to request graceful shutdown. PipelineTask receives
        # the upstream frame, flushes any in-flight TTS, then emits
        # EndFrame downstream to tear down processors. Avoids the
        # EndFrame hang race in pipecat issue #3757.
        await context.queue_frame(
            EndTaskFrame(reason=f"end_call_tool:{reason}"),
            FrameDirection.UPSTREAM,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "end_call_failed",
            reason=reason,
            error=str(exc),
            call_id=context.call_id,
        )
        return error_result("Unable to end the call. Please try again.")

    logger.info(
        "end_call_initiated",
        reason=reason,
        call_id=context.call_id,
    )

    return success_result(data={"reason": reason, "call_ended": True}, run_llm=False)


END_CALL = ToolDefinition(
    name="end_call",
    description=DESCRIPTION_DEFAULT,
    parameters=[
        # ``reason`` is optional — Aurora's lambda schema accepts
        # ``end_call`` with no arguments at all. Forcing it would
        # diverge from the schema and burden the LLM with inventing
        # a reason on every hangup.
        ToolParameter(
            name="reason",
            type="string",
            description=(
                "Optional brief reason for ending the call (audit log "
                "only); the call ends regardless of value."
            ),
            required=False,
        ),
    ],
    executor=end_call_executor,
    # EndTaskFrame queueing itself is fast; the timeout bounds any
    # transport-internal delay in accepting the frame.
    timeout_secs=10.0,
    # ``cancel_on_interruption`` and ``run_llm`` use the Pipecat
    # defaults (both ``True``). See module docstring.
)
