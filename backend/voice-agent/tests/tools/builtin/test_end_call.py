"""Tests for app.tools.builtin.end_call."""

from __future__ import annotations

from unittest.mock import AsyncMock

from app.tools.builtin.end_call import END_CALL, end_call_executor
from app.tools.context import ToolContext
from app.tools.result import ToolStatus
from pipecat.frames.frames import EndFrame


def _ctx(*, queue_frame: AsyncMock | None = None) -> tuple[ToolContext, AsyncMock]:
    qf = queue_frame or AsyncMock(return_value=None)
    return (
        ToolContext(
            call_id="call-1",
            session_id="call-1",
            sip_session_id="sip-leg-1",
            queue_frame=qf,
        ),
        qf,
    )


class TestEndCallExecutor:
    async def test_queues_end_frame(self):
        ctx, qf = _ctx()
        result = await end_call_executor({"reason": "done"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        qf.assert_awaited_once()
        frame = qf.call_args.args[0]
        assert isinstance(frame, EndFrame)

    async def test_run_llm_false(self):
        # Critical: LLM has already spoken the closing line on the
        # tool-calling turn. Running it again would race the
        # transport close.
        ctx, _ = _ctx()
        result = await end_call_executor({}, ctx)
        assert result.status is ToolStatus.SUCCESS
        assert result.run_llm is False

    async def test_default_reason_when_not_provided(self):
        ctx, _ = _ctx()
        result = await end_call_executor({}, ctx)
        assert result.status is ToolStatus.SUCCESS
        assert (result.data or {}).get("reason") == "Conversation concluded"

    async def test_preserves_provided_reason_in_data(self):
        ctx, _ = _ctx()
        result = await end_call_executor({"reason": "Customer issue resolved"}, ctx)
        assert (result.data or {}).get("reason") == "Customer issue resolved"

    async def test_whitespace_only_reason_falls_back_to_default(self):
        ctx, _ = _ctx()
        result = await end_call_executor({"reason": "   "}, ctx)
        assert (result.data or {}).get("reason") == "Conversation concluded"

    async def test_no_queue_frame_returns_error(self):
        ctx = ToolContext(call_id="call-1", queue_frame=None)
        result = await end_call_executor({}, ctx)
        assert result.status is ToolStatus.ERROR

    async def test_queue_frame_exception_returns_error(self):
        qf = AsyncMock(side_effect=RuntimeError("queue died"))
        ctx, _ = _ctx(queue_frame=qf)
        result = await end_call_executor({}, ctx)
        assert result.status is ToolStatus.ERROR


class TestEndCallDefinition:
    def test_name_matches_aurora(self):
        assert END_CALL.name == "end_call"

    def test_reason_param_optional(self):
        reason = next(p for p in END_CALL.parameters if p.name == "reason")
        assert reason.required is False

    def test_does_not_cancel_on_interruption(self):
        assert END_CALL.cancel_on_interruption is False
