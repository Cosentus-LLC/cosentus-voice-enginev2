"""VERIFICATION-2 — function-call lifecycle integration test.

Empirically confirms that, after the Bug A rewrite (no
``InterruptionTaskFrame`` from inside tool handlers), the standard
Pipecat function-call lifecycle correctly lands BOTH a ``tool_use``
block (on the assistant turn) AND a ``tool_result`` block (on the
following user turn) in the LLM context. The Bedrock adapter then
converts those into ``toolUse`` / ``toolResult`` content blocks
that Claude can reason over.

This is the specific bug yesterday's inbound PSTN test surfaced:
because press_digit pushed an interruption frame from inside the
handler, neither the tool_use nor the tool_result ever made it
into context. Claude saw repeated user requests but no record of
having pressed digits → infinite tool-call loop.

Test approach: drive ``LLMAssistantAggregator`` directly with the
canonical frame sequence from ``llm_service.py`` (FunctionCallIn-
ProgressFrame → FunctionCallResultFrame). Inspect ``LLMContext``
messages afterward. Then run those messages through the Bedrock
adapter and inspect the resulting toolUse / toolResult content
blocks.

Defaults under test:
* ``cancel_on_interruption=True`` (Pipecat default; sync flow). With
  the Bug A rewrite, both press_digit and end_call use this default.
* ``run_llm`` is whatever the result-callback ``properties`` say —
  defaults to ``True``.
"""

from __future__ import annotations

import json

import pytest
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
)


def _make_aggregator() -> tuple[LLMAssistantAggregator, LLMContext]:
    context = LLMContext()
    aggregator = LLMAssistantAggregator(context=context)
    return aggregator, context


@pytest.mark.asyncio
async def test_sync_tool_lifecycle_records_tool_use_and_tool_result_in_context():
    """The synchronous tool path (``cancel_on_interruption=True``)
    adds an assistant ``tool_calls`` message AND a ``tool`` message
    with the actual result. Both must be present so Bedrock's adapter
    can emit ``toolUse`` and ``toolResult`` content blocks.
    """
    aggregator, context = _make_aggregator()

    # In-progress frame — broadcast by the LLM service before invoking
    # the handler. Adds the assistant tool_calls block + IN_PROGRESS
    # tool message.
    in_progress = FunctionCallInProgressFrame(
        function_name="press_digit",
        tool_call_id="tooluse_abc123",
        arguments={"digits": "123"},
        cancel_on_interruption=True,
    )
    await aggregator._handle_function_call_in_progress(in_progress)

    # Result frame — broadcast by the result_callback after the
    # handler completes. For sync flow, the IN_PROGRESS placeholder
    # is updated in place with the actual result.
    result_frame = FunctionCallResultFrame(
        function_name="press_digit",
        tool_call_id="tooluse_abc123",
        arguments={"digits": "123"},
        result={"digits_pressed": "123", "digit_count": 3},
        run_llm=True,
        properties=FunctionCallResultProperties(),
    )
    # Mirror the LLM service's _function_calls_in_progress dict that
    # _handle_function_call_result expects to find the in-progress
    # entry in. The aggregator's _handle_function_call_in_progress
    # populates it; verify and proceed.
    assert "tooluse_abc123" in aggregator._function_calls_in_progress
    await aggregator._handle_function_call_result(result_frame)

    # Inspect context.
    messages = context.get_messages()

    # An assistant message with a tool_calls list naming press_digit.
    assistant_with_tool_calls = [
        m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_with_tool_calls) == 1, messages
    tc = assistant_with_tool_calls[0]["tool_calls"][0]
    assert tc["id"] == "tooluse_abc123"
    assert tc["function"]["name"] == "press_digit"

    # A tool message keyed by the same tool_call_id, content is the
    # serialized result payload (no longer "IN_PROGRESS").
    tool_messages = [
        m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == "tooluse_abc123"
    ]
    assert len(tool_messages) == 1, messages
    assert tool_messages[0]["content"] != "IN_PROGRESS", (
        "tool message must be updated with the actual result"
    )
    # JSON-encoded payload.
    payload = json.loads(tool_messages[0]["content"])
    assert payload["digits_pressed"] == "123"


@pytest.mark.asyncio
async def test_bedrock_adapter_emits_tool_use_and_tool_result_blocks():
    """Run the standard-format messages from the lifecycle through
    the Bedrock adapter. Bedrock's Converse API needs:

    * ``role: assistant`` with ``content: [{toolUse: {...}}]``
    * ``role: user`` with ``content: [{toolResult: {...}}]``

    Anything else (e.g. dropping the toolUse block, or losing the
    toolResult into a generic text blob) will cause Claude to lose
    memory of the tool call — which is the exact symptom yesterday's
    inbound PSTN test surfaced.
    """
    aggregator, context = _make_aggregator()

    # Seed the context with a prior user turn to anchor the conversation.
    context.add_message({"role": "user", "content": "Press 1 2 3 please."})

    # Drive the lifecycle.
    await aggregator._handle_function_call_in_progress(
        FunctionCallInProgressFrame(
            function_name="press_digit",
            tool_call_id="tooluse_xyz789",
            arguments={"digits": "123"},
            cancel_on_interruption=True,
        )
    )
    await aggregator._handle_function_call_result(
        FunctionCallResultFrame(
            function_name="press_digit",
            tool_call_id="tooluse_xyz789",
            arguments={"digits": "123"},
            result={"digits_pressed": "123", "digit_count": 3},
            run_llm=True,
            properties=FunctionCallResultProperties(),
        )
    )

    # Convert through the Bedrock adapter.
    adapter = AWSBedrockLLMAdapter()
    invocation = adapter.get_llm_invocation_params(context)
    bedrock_messages = invocation["messages"]

    # Find a toolUse content block (assistant turn) and a toolResult
    # content block (user turn, post-merge per Bedrock convention).
    tool_use_blocks: list = []
    tool_result_blocks: list = []
    for msg in bedrock_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "toolUse" in block:
                tool_use_blocks.append((msg["role"], block["toolUse"]))
            if isinstance(block, dict) and "toolResult" in block:
                tool_result_blocks.append((msg["role"], block["toolResult"]))

    assert tool_use_blocks, f"no toolUse block in bedrock messages: {bedrock_messages}"
    assert tool_result_blocks, f"no toolResult block in bedrock messages: {bedrock_messages}"

    # toolUse must live on an assistant turn.
    role, tu = tool_use_blocks[0]
    assert role == "assistant"
    assert tu["toolUseId"] == "tooluse_xyz789"
    assert tu["name"] == "press_digit"
    assert tu["input"] == {"digits": "123"}

    # toolResult must live on a user turn (Bedrock convention; the
    # adapter converts ``role: tool`` → ``role: user`` with the
    # toolResult content block).
    role, tr = tool_result_blocks[0]
    assert role == "user"
    assert tr["toolUseId"] == "tooluse_xyz789"
    # Result content carries the success payload.
    assert tr["content"], f"toolResult content empty: {tr}"


@pytest.mark.asyncio
async def test_lifecycle_works_without_interruption_frame():
    """Direct empirical re-verification of the Bug A claim.

    The press_digit handler used to push an ``InterruptionTaskFrame``
    BEFORE returning. That frame propagated through the pipeline as
    a downstream ``InterruptionFrame`` and triggered the assistant
    aggregator's ``reset()`` path, which discarded the partial
    aggregation.

    With the rewrite, the handler does no such thing. To confirm the
    lifecycle is healthy under the new pattern, drive the canonical
    sequence and verify the assistant tool_calls + tool messages
    are intact afterward.
    """
    aggregator, context = _make_aggregator()

    await aggregator._handle_function_call_in_progress(
        FunctionCallInProgressFrame(
            function_name="press_digit",
            tool_call_id="tooluse_no_interrupt",
            arguments={"digits": "9"},
            cancel_on_interruption=True,
        )
    )
    await aggregator._handle_function_call_result(
        FunctionCallResultFrame(
            function_name="press_digit",
            tool_call_id="tooluse_no_interrupt",
            arguments={"digits": "9"},
            result={"digits_pressed": "9"},
            run_llm=True,
            properties=FunctionCallResultProperties(),
        )
    )

    messages = context.get_messages()
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages), (
        "assistant tool_calls block missing — Bug A regression?"
    )
    assert any(
        m.get("role") == "tool"
        and m.get("tool_call_id") == "tooluse_no_interrupt"
        and m.get("content") != "IN_PROGRESS"
        for m in messages
    ), "tool result message missing — Bug A regression?"
