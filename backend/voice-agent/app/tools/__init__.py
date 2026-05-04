"""Tools layer — three platform tools, registry, executor.

Public surface:

* :class:`ToolDefinition`, :class:`ToolParameter` — schema (immutable)
* :class:`ToolResult`, :class:`ToolStatus`, :func:`success_result`,
  :func:`error_result`, :func:`timeout_result`,
  :func:`cancelled_result` — outcomes
* :class:`ToolContext` — runtime info passed to executors
* :class:`ToolRegistry` — per-call catalog
* :class:`ToolExecutor` — timeout / error / cancel wrapper
* :func:`parse_disabled_tools` — CSV parser (Settings.disabled_tools)
* :func:`build_registry_for_call` — Layer 8's entry point
* :data:`BUILTIN_TOOLS` — the three platform tools

Per-tool helpers (rarely needed outside Layer 4 / tests):

* :data:`TRANSFER_CALL`, :data:`PRESS_DIGIT`, :data:`END_CALL`
* :func:`transfer_call_executor`, :func:`press_digit_executor`,
  :func:`end_call_executor`
"""

from app.tools.builtin import (
    BUILTIN_TOOLS,
    END_CALL,
    PRESS_DIGIT,
    TRANSFER_CALL,
    end_call_executor,
    press_digit_executor,
    transfer_call_executor,
)
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.registry import (
    ToolRegistry,
    build_registry_for_call,
    parse_disabled_tools,
)
from app.tools.result import (
    ToolResult,
    ToolStatus,
    cancelled_result,
    error_result,
    success_result,
    timeout_result,
)
from app.tools.schema import ToolDefinition, ToolParameter

__all__ = [
    "BUILTIN_TOOLS",
    "END_CALL",
    "PRESS_DIGIT",
    "TRANSFER_CALL",
    "ToolContext",
    "ToolDefinition",
    "ToolExecutor",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "ToolStatus",
    "build_registry_for_call",
    "cancelled_result",
    "end_call_executor",
    "error_result",
    "parse_disabled_tools",
    "press_digit_executor",
    "success_result",
    "timeout_result",
    "transfer_call_executor",
]
