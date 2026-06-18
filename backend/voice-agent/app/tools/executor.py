"""Tool executor — wraps tool invocations with timeout and error handling.

Single responsibility: take a tool name and arguments, run the
tool's executor, and translate every possible outcome (success,
timeout, cancellation, exception) into a :class:`ToolResult`. Never
raises.

The executor is constructed once per registry (per call) and used
by Layer 8's tool handler closures.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from app.observability.tracing import set_span_attrs, tool_span
from app.tools.context import ToolContext
from app.tools.registry import ToolRegistry
from app.tools.result import (
    ToolResult,
    cancelled_result,
    error_result,
    timeout_result,
)

logger = structlog.get_logger(__name__)


class ToolExecutor:
    """Executes tools registered in a :class:`ToolRegistry`.

    Behavior contract:

    * Returns a :class:`ToolResult` for every input. Never raises
      out of :meth:`execute`.
    * Honors each tool's ``timeout_secs`` via
      :func:`asyncio.wait_for`. On timeout the in-flight tool task
      is cancelled and a :func:`~app.tools.result.timeout_result`
      is returned.
    * Treats :class:`asyncio.CancelledError` as a normal outcome
      (e.g. user interrupted mid-call) and returns
      :func:`~app.tools.result.cancelled_result`.
    * Any other exception is logged at ``error`` level with the
      traceback and returned as
      :func:`~app.tools.result.error_result`.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Run one tool invocation. Never raises.

        Args:
            tool_name: Name of the tool to execute. Must be
                registered in the executor's :class:`ToolRegistry`.
            arguments: LLM-supplied arguments dict.
            context: Per-call :class:`ToolContext`. Layer 8 builds
                this fresh for every invocation.

        Returns:
            :class:`ToolResult`.
        """
        tool = self._registry.get(tool_name)
        if tool is None:
            logger.error(
                "tool_not_found",
                tool_name=tool_name,
                available=self._registry.names(),
                call_id=context.call_id,
            )
            return error_result(f"Unknown tool: {tool_name}")

        # ``voice.tool`` span (#13), parented to the call root via the
        # explicitly-propagated context. PHI-free: only the tool name,
        # status, run_llm flag, timing, and error *type* — never the
        # arguments or result data.
        with tool_span(tool_name=tool_name, parent_context=context.otel_context) as span:
            start = time.monotonic()

            def _duration_ms() -> float:
                return round((time.monotonic() - start) * 1000, 2)

            try:
                result = await asyncio.wait_for(
                    tool.executor(arguments, context),
                    timeout=tool.timeout_secs,
                )
            except TimeoutError:
                logger.error(
                    "tool_timeout",
                    tool_name=tool_name,
                    timeout_secs=tool.timeout_secs,
                    call_id=context.call_id,
                )
                set_span_attrs(
                    span,
                    {"voice.tool.status": "timeout", "voice.tool.duration_ms": _duration_ms()},
                )
                return timeout_result()
            except asyncio.CancelledError:
                logger.warning(
                    "tool_cancelled",
                    tool_name=tool_name,
                    call_id=context.call_id,
                )
                set_span_attrs(
                    span,
                    {"voice.tool.status": "cancelled", "voice.tool.duration_ms": _duration_ms()},
                )
                return cancelled_result()
            except Exception as exc:  # noqa: BLE001 — last-resort wrapping
                logger.exception(
                    "tool_error",
                    tool_name=tool_name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    call_id=context.call_id,
                )
                set_span_attrs(
                    span,
                    {
                        "voice.tool.status": "error",
                        "voice.tool.error_type": type(exc).__name__,
                        "voice.tool.duration_ms": _duration_ms(),
                    },
                )
                return error_result(str(exc))

            logger.info(
                "tool_executed",
                tool_name=tool_name,
                status=result.status.value,
                run_llm=result.run_llm,
                call_id=context.call_id,
            )
            set_span_attrs(
                span,
                {
                    "voice.tool.status": result.status.value,
                    "voice.tool.run_llm": result.run_llm,
                    "voice.tool.duration_ms": _duration_ms(),
                },
            )
            return result
