"""``transfer_call`` tool — warm SIP transfer via Daily.

Hands the caller off to a per-agent-configured target phone number
using Daily's ``sip_call_transfer`` (warm transfer). Daily keeps
the original SIP leg anchored, dials the target, and bridges the
audio. We chose ``sip_call_transfer`` over ``sip_refer`` for
launch reliability — REFER requires originating-carrier REFER
support that we haven't validated against Cosentus's PSTN routing.

Per-agent config (from ``AgentConfig.tools[].settings``):

    {
        "targets": {
            "billing_supervisor": "+13105551234",
            "after_hours_voicemail": "+13105559876"
        }
    }

The agent's target names are surfaced to the LLM as an enum on the
``target`` parameter (applied at registry-build time via
:meth:`ToolDefinition.with_enum`), so Claude can only emit valid
target names — no free-text guessing.
"""

from __future__ import annotations

import structlog

from app.tools.context import ToolContext
from app.tools.result import ToolResult, error_result, success_result
from app.tools.schema import ToolDefinition, ToolParameter

logger = structlog.get_logger(__name__)


# Default LLM-facing description. Aurora's per-agent description
# overrides this at registration time; this fallback only applies
# when the agent didn't supply one.
DESCRIPTION_DEFAULT = (
    "Transfer the caller to a specific pre-configured target. "
    "Only use when the caller explicitly requests a transfer, "
    "when you cannot help with their request, or when the issue "
    "requires a human. Do not transfer without first confirming "
    "with the caller."
)


async def transfer_call_executor(
    arguments: dict,
    context: ToolContext,
) -> ToolResult:
    """Execute a Daily SIP call transfer.

    Validates inputs, resolves the target name to a phone number
    via the agent's settings, and calls
    ``transport.sip_call_transfer`` with the SIP-correct envelope.
    Returns ``run_llm=False`` on success — the LLM should have
    already spoken its hand-off line in the same turn that
    triggered this tool call.
    """
    target_name = (arguments.get("target") or "").strip()
    if not target_name:
        return error_result("target argument is required")

    targets = context.tool_settings.get("targets")
    if not isinstance(targets, dict) or not targets:
        return error_result("transfer_call has no configured targets for this agent")

    phone_number = targets.get(target_name)
    if not phone_number:
        available = sorted(targets.keys())
        return error_result(f"Unknown transfer target {target_name!r}. Available: {available}")

    if not context.sip_session_id:
        return error_result("No SIP session available for transfer")

    if context.transport is None:
        return error_result("No transport available for transfer")

    # Mask the destination in logs — call records are HIPAA-adjacent.
    destination_mask = (
        "****" + phone_number[-4:]
        if isinstance(phone_number, str) and len(phone_number) >= 4
        else "***"
    )

    try:
        # Daily's sip_call_transfer payload: capital-P ``toEndPoint``
        # is required by the Daily SDK. ``sessionId`` identifies the
        # SIP leg in the room. Warm transfer — Daily anchors the
        # original leg and bridges the new one.
        await context.transport.sip_call_transfer(
            {
                "sessionId": context.sip_session_id,
                "toEndPoint": phone_number,
            }
        )
    except Exception as exc:  # noqa: BLE001 — convert all transport errors
        logger.exception(
            "transfer_call_failed",
            target_name=target_name,
            destination=destination_mask,
            sip_session_id=context.sip_session_id,
            call_id=context.call_id,
            error=str(exc),
        )
        return error_result(
            "I'm unable to complete the transfer right now. "
            "Let me try to help you myself, or you can call back in a moment."
        )

    logger.info(
        "transfer_call_initiated",
        target_name=target_name,
        destination=destination_mask,
        sip_session_id=context.sip_session_id,
        call_id=context.call_id,
    )

    return success_result(
        data={"transferred_to": target_name},
        # Don't run the LLM after — the call is now on a different
        # leg and the LLM running would only add latency before the
        # actual bridge takes effect. The hand-off line should have
        # been spoken in the same LLM turn that emitted the tool call.
        run_llm=False,
    )


TRANSFER_CALL = ToolDefinition(
    name="transfer_call",
    description=DESCRIPTION_DEFAULT,
    parameters=[
        # The ``target`` parameter is declared as a plain string
        # here; ``build_registry_for_call`` rewrites it with an
        # ``enum=[...target_names...]`` constraint per-agent.
        ToolParameter(
            name="target",
            type="string",
            description=(
                "Name of the transfer target. Must exactly match one of "
                "the configured targets for this agent."
            ),
            required=True,
        ),
    ],
    executor=transfer_call_executor,
    # Daily SIP REFER / transfer negotiation can take 3-10 seconds
    # with slow carriers; 30s covers worst-case + retry margin.
    timeout_secs=30.0,
    # State change — partial cancellation would leave the bridge in
    # an undefined state. Don't cancel mid-flight.
    cancel_on_interruption=False,
)
