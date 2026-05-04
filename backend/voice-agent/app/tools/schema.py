"""Tool schema — immutable definitions of what each tool exposes to the LLM.

A :class:`ToolDefinition` carries everything the rest of Layer 4
needs for one tool: the LLM-facing name and description, the
argument schema (one or more :class:`ToolParameter`), the executor
that runs when the tool is invoked, and timeout / cancellation
policy.

Definitions are immutable (frozen dataclasses). Per-agent
customizations — overriding the description, adding an enum
constraint to a parameter — are applied via ``with_description`` /
``with_enum`` which return new definitions rather than mutating
the original. This keeps the catalog (:data:`builtin.catalog.BUILTIN_TOOLS`)
safe to reuse across calls and across tests.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipecat.adapters.schemas.function_schema import FunctionSchema

if TYPE_CHECKING:
    from app.tools.context import ToolContext
    from app.tools.result import ToolResult


# Tool executors are async callables; type kept loose so test mocks
# work without subclassing.
ToolExecutorFn = Callable[[dict[str, Any], "ToolContext"], Awaitable["ToolResult"]]


@dataclass(frozen=True)
class ToolParameter:
    """One argument the LLM passes when invoking a tool.

    Maps directly to a JSON Schema property. ``enum`` is the
    constraint Pipecat / Bedrock enforce most reliably — used by
    ``transfer_call`` to limit the ``target`` argument to the
    agent's configured target names.
    """

    name: str
    type: str  # "string" | "number" | "integer" | "boolean"
    description: str
    required: bool = True
    enum: list[str] | None = None
    pattern: str | None = None  # JSON Schema regex


@dataclass(frozen=True)
class ToolDefinition:
    """Complete platform-tool definition.

    Attributes:
        name: LLM-facing identifier (e.g. ``"transfer_call"``).
            Must match Aurora's ``VALID_TOOL_TYPES``.
        description: Default LLM-facing description. Aurora's
            per-agent ``description`` overrides this at registration
            time via :meth:`with_description`.
        parameters: Tuple-of-arguments schema.
        executor: Async function that runs when the LLM invokes
            this tool. Returns a :class:`ToolResult`.
        timeout_secs: Hard deadline for the executor. After this
            elapses, :class:`~app.tools.executor.ToolExecutor`
            cancels and returns ``timeout_result()``.
        cancel_on_interruption: Whether Pipecat should cancel an
            in-flight tool call when the user interrupts. ``True``
            for cheap idempotent tools; ``False`` for tools whose
            partial execution leaves state changes (transfer,
            DTMF, end-call).
    """

    name: str
    description: str
    parameters: list[ToolParameter]
    executor: ToolExecutorFn
    timeout_secs: float = 30.0
    cancel_on_interruption: bool = True

    def to_function_schema(self) -> FunctionSchema:
        """Convert to a Pipecat :class:`FunctionSchema` for LLM registration.

        Pipecat's adapter layer (Bedrock / Anthropic / OpenAI)
        translates :class:`FunctionSchema` into the vendor-specific
        tool format at request time. We just produce the universal
        intermediate.
        """
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for param in self.parameters:
            entry: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                entry["enum"] = list(param.enum)
            if param.pattern:
                entry["pattern"] = param.pattern
            properties[param.name] = entry
            if param.required:
                required.append(param.name)
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=properties,
            required=required,
        )

    def with_description(self, new_description: str) -> ToolDefinition:
        """Return a copy with a different LLM-facing description.

        Used to apply the per-agent description from
        ``AgentConfig.tools[].description`` over the platform default.
        """
        return dataclasses.replace(self, description=new_description)

    def with_enum(self, parameter_name: str, enum: list[str]) -> ToolDefinition:
        """Return a copy with an enum constraint on one parameter.

        Used by ``transfer_call`` registration: the ``target``
        parameter is dynamically constrained to the agent's
        configured target names so Claude can only emit valid
        values.

        Raises:
            KeyError: if ``parameter_name`` is not an existing
                parameter on this definition.
        """
        new_params = []
        found = False
        for param in self.parameters:
            if param.name == parameter_name:
                new_params.append(dataclasses.replace(param, enum=list(enum)))
                found = True
            else:
                new_params.append(param)
        if not found:
            raise KeyError(f"Tool {self.name!r} has no parameter {parameter_name!r}")
        return dataclasses.replace(self, parameters=new_params)
