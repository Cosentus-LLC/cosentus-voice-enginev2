"""Tests for app.tools.executor — ToolExecutor."""

from __future__ import annotations

import asyncio

from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.result import (
    ToolResult,
    ToolStatus,
    success_result,
)
from app.tools.schema import ToolDefinition, ToolParameter


def _make_tool(
    name: str,
    executor,
    *,
    timeout_secs: float = 1.0,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        parameters=[ToolParameter(name="x", type="string", description="x")],
        executor=executor,
        timeout_secs=timeout_secs,
    )


def _ctx() -> ToolContext:
    return ToolContext(call_id="test-call-1", session_id="test-call-1")


class TestToolExecutor:
    async def test_returns_success_on_happy_path(self):
        async def ok(args: dict, ctx) -> ToolResult:
            return success_result(data={"echo": args["x"]})

        registry = ToolRegistry()
        registry.register(_make_tool("ok_tool", ok))

        result = await ToolExecutor(registry).execute("ok_tool", {"x": "hi"}, _ctx())
        assert result.status is ToolStatus.SUCCESS
        assert result.data == {"echo": "hi"}

    async def test_returns_error_for_unknown_tool(self):
        registry = ToolRegistry()
        result = await ToolExecutor(registry).execute("missing", {}, _ctx())
        assert result.status is ToolStatus.ERROR
        assert "Unknown tool" in (result.error or "")

    async def test_returns_timeout_when_executor_exceeds_timeout(self):
        async def slow(args: dict, ctx) -> ToolResult:
            await asyncio.sleep(2.0)
            return success_result()

        registry = ToolRegistry()
        registry.register(
            _make_tool("slow_tool", slow, timeout_secs=0.05),
        )

        result = await ToolExecutor(registry).execute("slow_tool", {}, _ctx())
        assert result.status is ToolStatus.TIMEOUT
        assert "timed out" in (result.error or "").lower()

    async def test_returns_cancelled_on_cancelled_error(self):
        async def cancelled_executor(args: dict, ctx) -> ToolResult:
            raise asyncio.CancelledError()

        registry = ToolRegistry()
        registry.register(_make_tool("cancel_tool", cancelled_executor))

        result = await ToolExecutor(registry).execute("cancel_tool", {}, _ctx())
        assert result.status is ToolStatus.CANCELLED

    async def test_returns_error_on_unhandled_exception(self):
        async def raising(args: dict, ctx) -> ToolResult:
            raise RuntimeError("boom")

        registry = ToolRegistry()
        registry.register(_make_tool("boom_tool", raising))

        result = await ToolExecutor(registry).execute("boom_tool", {}, _ctx())
        assert result.status is ToolStatus.ERROR
        assert "boom" in (result.error or "")

    async def test_uses_tool_specific_timeout(self):
        # Verifies executor honors the tool's configured timeout, not
        # a hardcoded global.
        async def slow(args: dict, ctx) -> ToolResult:
            await asyncio.sleep(0.5)
            return success_result()

        registry = ToolRegistry()
        registry.register(_make_tool("fast_to", slow, timeout_secs=0.1))
        registry.register(_make_tool("slow_to", slow, timeout_secs=2.0))

        fast_result = await ToolExecutor(registry).execute("fast_to", {}, _ctx())
        slow_result = await ToolExecutor(registry).execute("slow_to", {}, _ctx())
        assert fast_result.status is ToolStatus.TIMEOUT
        assert slow_result.status is ToolStatus.SUCCESS
