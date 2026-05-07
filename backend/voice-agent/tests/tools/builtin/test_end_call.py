"""Tests for app.tools.builtin.end_call.

After the May 2026 rewrite, end_call pushes ``EndTaskFrame`` upstream
instead of queueing ``EndFrame`` downstream. This is the Pipecat-
canonical pattern for tool-initiated graceful shutdown and avoids
the open ``EndFrame`` hang race in pipecat issue #3757.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from app.tools.builtin.end_call import END_CALL, end_call_executor
from app.tools.context import ToolContext
from app.tools.result import ToolStatus
from pipecat.frames.frames import (
    CancelTaskFrame,
    EndTaskFrame,
    InterruptionFrame,
    InterruptionTaskFrame,
)
from pipecat.processors.frame_processor import FrameDirection


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
    async def test_pushes_end_task_frame_upstream(self):
        """v2 fix: previously queued ``EndFrame`` downstream which
        races with ``InterruptionFrame`` traffic (pipecat #3757).
        New pattern is ``EndTaskFrame`` UPSTREAM — PipelineTask
        catches it, flushes the queue, then emits ``EndFrame``
        downstream cleanly.
        """
        ctx, qf = _ctx()
        result = await end_call_executor({"reason": "done"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        qf.assert_awaited_once()
        frame = qf.await_args.args[0]
        assert isinstance(frame, EndTaskFrame)

        # Direction must be UPSTREAM. Accept either positional or
        # keyword form.
        direction = (
            qf.await_args.args[1]
            if len(qf.await_args.args) > 1
            else qf.await_args.kwargs.get("direction")
        )
        assert direction is FrameDirection.UPSTREAM

    async def test_end_task_frame_carries_reason(self):
        ctx, qf = _ctx()
        await end_call_executor({"reason": "Customer satisfied"}, ctx)
        frame = qf.await_args.args[0]
        assert isinstance(frame, EndTaskFrame)
        assert "Customer satisfied" in str(frame.reason or "")

    async def test_run_llm_default_true(self):
        """v2 fix: previously ``run_llm=False``. With ``EndTaskFrame``
        upstream the pipeline drains gracefully; letting the LLM
        re-fire is fine and lets Claude record the end-call decision
        in conversation history before shutdown.
        """
        ctx, _ = _ctx()
        result = await end_call_executor({}, ctx)
        assert result.status is ToolStatus.SUCCESS
        assert result.run_llm is True

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

    async def test_handler_does_not_push_pipeline_control_frames(self):
        """REGRESSION GUARD.

        end_call must not push ``InterruptionTaskFrame`` /
        ``InterruptionFrame`` / ``CancelTaskFrame`` — only the
        documented ``EndTaskFrame``. The Bug A audit established
        that custom interruption frames inside tools break the
        function-call lifecycle.
        """
        ctx, qf = _ctx()
        await end_call_executor({"reason": "test"}, ctx)
        forbidden_types = (
            InterruptionTaskFrame,
            InterruptionFrame,
            CancelTaskFrame,
        )
        for call in qf.await_args_list:
            frame = call.args[0]
            assert not isinstance(frame, forbidden_types), (
                f"end_call must not push {type(frame).__name__}"
            )


class TestEndCallDefinition:
    def test_name_matches_aurora(self):
        assert END_CALL.name == "end_call"

    def test_reason_param_optional(self):
        reason = next(p for p in END_CALL.parameters if p.name == "reason")
        assert reason.required is False

    def test_uses_default_cancel_on_interruption_true(self):
        """v2 fix: previously ``cancel_on_interruption=False``. The
        new EndTaskFrame-upstream pattern doesn't need partial-
        cancellation protection — once PipelineTask receives it,
        the shutdown is atomic.
        """
        assert END_CALL.cancel_on_interruption is True
