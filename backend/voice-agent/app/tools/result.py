"""Tool execution results — what the executor returns to the handler.

The result tells the handler four things: did the tool succeed, what
data should be sent back to the LLM, whether the LLM should be
re-invoked after the tool runs, and (for deterministic tools)
whether to bypass the LLM and speak a pre-formatted line directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolStatus(Enum):
    """Outcome of a tool invocation."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolResult:
    """The structured outcome of one tool execution.

    Attributes:
        status: Outcome category (see :class:`ToolStatus`).
        data: Optional payload passed back to the LLM as the function
            result. Skipped on non-success paths in favor of
            ``error``.
        error: Human-readable error message. Populated on every
            non-success path.
        run_llm: Whether the LLM should be invoked after the tool
            returns. ``True`` is the default and matches Pipecat's
            normal flow. ``False`` is critical for ``end_call``
            (avoid the post-EndFrame LLM call that would fail) and
            ``press_digit`` (let the IVR's response become the next
            user turn instead of the bot speaking).
        spoken_response: If set on a successful result with
            ``run_llm=False``, the handler should push a
            ``TTSSpeakFrame`` with this text directly into the
            pipeline — bypassing the LLM round-trip entirely. None
            of the v2 platform tools currently use this path; it
            exists for future deterministic-response tools.
    """

    status: ToolStatus
    data: dict | None = None
    error: str | None = None
    run_llm: bool = True
    spoken_response: str | None = None


def success_result(
    data: dict | None = None,
    *,
    run_llm: bool = True,
    spoken_response: str | None = None,
) -> ToolResult:
    """Construct a successful :class:`ToolResult`."""
    return ToolResult(
        status=ToolStatus.SUCCESS,
        data=data,
        run_llm=run_llm,
        spoken_response=spoken_response,
    )


def error_result(error: str) -> ToolResult:
    """Construct an error :class:`ToolResult` with a human-readable message."""
    return ToolResult(status=ToolStatus.ERROR, error=error)


def timeout_result() -> ToolResult:
    """Construct a timeout :class:`ToolResult`."""
    return ToolResult(
        status=ToolStatus.TIMEOUT,
        error="Tool execution timed out",
    )


def cancelled_result() -> ToolResult:
    """Construct a cancellation :class:`ToolResult`."""
    return ToolResult(
        status=ToolStatus.CANCELLED,
        error="Tool execution was cancelled",
    )
