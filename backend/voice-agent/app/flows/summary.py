"""Dialogue-only summary adapter for Pipecat Flows context resets."""

from __future__ import annotations

from typing import Any

import structlog
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat_flows.adapters import LLMAdapter

logger = structlog.get_logger(__name__)

_SUMMARY_LABEL = "Prior dialogue summary for assistant memory"
_DIALOGUE_ROLES = {"user", "assistant"}


class DialogueOnlyFlowAdapter(LLMAdapter):
    """Keep Flows summaries scoped to caller/agent dialogue.

    Pipecat-Flows 1.1.1 does not expose an adapter injection hook, but its
    ``FlowManager`` stores the adapter on the instance. ``build_flow_manager``
    installs this adapter there so RESET_WITH_SUMMARY does not summarize
    system/developer/task/tool machinery.
    """

    def format_summary_message(self, summary: str) -> dict:
        """Format the running summary without the user-visible stock prefix."""
        return {"role": "developer", "content": f"{_SUMMARY_LABEL}:\n{summary}"}

    async def generate_summary(
        self,
        llm: Any,
        summary_prompt: str,
        context: LLMContext,
    ) -> str | None:
        """Generate a summary from standard user/assistant text turns only."""
        try:
            transcript = _format_dialogue_for_summary(context.get_messages())
            if not transcript:
                return None

            summary_context = LLMContext(
                messages=[{"role": "user", "content": f"Conversation history:\n{transcript}"}]
            )
            return await llm.run_inference(summary_context, system_instruction=summary_prompt)
        except Exception as exc:  # noqa: BLE001 - summary failure must not break the call
            logger.error("flow_summary_generation_failed", error=str(exc), exc_info=True)
            return None


def _format_dialogue_for_summary(messages: list[dict]) -> str:
    """Return a plain transcript from user/assistant text messages only."""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, LLMSpecificMessage):
            continue
        role = message.get("role")
        if role not in _DIALOGUE_ROLES:
            continue
        text = _message_text(message.get("content")).strip()
        if text:
            parts.append(f"{role.upper()}: {text}")
    return "\n\n".join(parts)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text") or ""))
        return " ".join(text_parts)
    return ""
