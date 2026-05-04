"""``end_call`` tool — clean pipeline termination via :class:`EndFrame`.

Queues an :class:`EndFrame` into the pipeline. Pipecat drains every
in-flight frame (including the LLM's "Thanks, have a good one"
TTS audio) before tearing down the transport, so the caller hears
the full closing line before disconnect.

The ``reason`` argument is audit-log only — the call ends
regardless of value. We accept it to keep the LLM's tool-use
schema human-readable; many system prompts encourage Claude to
state a reason for clarity in transcripts.

``run_llm=False`` is critical: the LLM has already spoken the
closing line on the turn that emitted this tool call. Running the
LLM again would (a) add ~2 s of dead air before the actual
disconnect and (b) fail with "Unable to send messages before
joining" once the transport has closed.
"""

from __future__ import annotations

import structlog
from pipecat.frames.frames import EndFrame

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
    """Queue an :class:`EndFrame` to terminate the pipeline."""
    reason = (arguments.get("reason") or "").strip() or "Conversation concluded"

    if context.queue_frame is None:
        return error_result("No frame queue available; cannot end call")

    try:
        await context.queue_frame(EndFrame())
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

    return success_result(
        data={"reason": reason, "call_ended": True},
        # See module docstring — the LLM's closing line was spoken
        # on the same turn that emitted this tool call. Running the
        # LLM again would race the transport close.
        run_llm=False,
    )


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
    # EndFrame queueing itself is fast; the timeout bounds any
    # transport-internal delay in accepting the frame.
    timeout_secs=10.0,
    # State change — once EndFrame is queued, cancellation has no
    # meaning.
    cancel_on_interruption=False,
)
