"""Tests for the dialogue-only Pipecat Flows summary adapter."""

from __future__ import annotations

from app.flows.summary import DialogueOnlyFlowAdapter
from pipecat.processors.aggregators.llm_context import LLMContext


class _FakeLLM:
    def __init__(self) -> None:
        self.summary_context: LLMContext | None = None
        self.system_instruction: str | None = None

    async def run_inference(
        self,
        summary_context: LLMContext,
        *,
        system_instruction: str,
    ) -> str:
        self.summary_context = summary_context
        self.system_instruction = system_instruction
        return "Representative confirmed fax submission."


async def test_flow_summary_adapter_filters_internal_roles_and_tool_calls():
    adapter = DialogueOnlyFlowAdapter()
    llm = _FakeLLM()
    context = LLMContext(
        messages=[
            {"role": "system", "content": "When done, call greeting_done."},
            {"role": "developer", "content": "Use verify_identity first."},
            {"role": "user", "content": "The claim was denied for CO-16."},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I will ask what is needed."}],
                "tool_calls": [{"function": {"name": "transfer_call", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "end_call result"},
        ]
    )

    summary = await adapter.generate_summary(llm, "SUMMARY PROMPT", context)

    assert summary == "Representative confirmed fax submission."
    assert llm.system_instruction == "SUMMARY PROMPT"
    assert llm.summary_context is not None
    transcript = llm.summary_context.messages[0]["content"]
    assert "The claim was denied for CO-16." in transcript
    assert "I will ask what is needed." in transcript
    assert "call greeting_done" not in transcript
    assert "verify_identity" not in transcript
    assert "transfer_call" not in transcript
    assert "end_call" not in transcript


def test_flow_summary_message_label_does_not_use_public_summary_phrase():
    adapter = DialogueOnlyFlowAdapter()

    message = adapter.format_summary_message("fax confirmed")

    assert message["role"] == "developer"
    assert "Prior dialogue summary for assistant memory" in message["content"]
    assert "Here's a summary of the conversation" not in message["content"]
    assert "Conversation summary so far" not in message["content"]
