"""Tool registry — per-call catalog of tools the LLM can invoke.

A :class:`ToolRegistry` holds the survivors after Layer 4's
filtering: the platform tools the operator hasn't disabled, that
the agent has opted into. Layer 8 builds one registry per
pipeline / per call (no global singleton — see Layer 2's
dependency-injection rationale) and hands it to the LLM service
for ``register_function`` registration.

Public functions:

* :func:`parse_disabled_tools` — split a CSV from
  ``Settings.disabled_tools`` into a set of names. (Closes
  tech-debt log entry 5: parsing happens at this Layer 4 boundary,
  not in Layer 2's ``Settings``.)
* :func:`build_registry_for_call` — the canonical Layer-8-facing
  factory: takes an ``AgentConfig`` + ``Settings``, returns a
  fully-built :class:`ToolRegistry`.
"""

from __future__ import annotations

import structlog
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from app.config.agent_config import AgentConfig
from app.config.settings import Settings
from app.tools.schema import ToolDefinition

logger = structlog.get_logger(__name__)


class ToolRegistry:
    """Per-call collection of registered tools.

    Constructed once per pipeline build, populated from the
    platform catalog filtered against operator and agent config,
    then locked. Layer 8 reads :meth:`to_tools_schema` and
    :meth:`all` / :meth:`get_settings` from it.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._tool_settings: dict[str, dict] = {}
        self._locked: bool = False

    def register(
        self,
        tool: ToolDefinition,
        settings: dict | None = None,
    ) -> None:
        """Add a tool to the registry.

        Args:
            tool: The :class:`ToolDefinition` to register. Already
                customized (per-agent description, enum, etc.) by
                the caller.
            settings: Per-agent ``settings`` payload from
                ``AgentConfig.tools[].settings`` — handed to the
                executor at invocation time via
                :class:`~app.tools.context.ToolContext.tool_settings`.
                Empty dict if the agent didn't configure any.

        Raises:
            ValueError: If the tool name is empty or already
                registered.
            RuntimeError: If the registry has been locked.
        """
        if self._locked:
            raise RuntimeError("Registry is locked; cannot register more tools")
        if not tool.name:
            raise ValueError("Tool name cannot be empty")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        self._tool_settings[tool.name] = settings or {}

    def get(self, name: str) -> ToolDefinition | None:
        """Return the registered tool with this name, or ``None``."""
        return self._tools.get(name)

    def get_settings(self, name: str) -> dict:
        """Return the per-agent settings for ``name``, or ``{}``."""
        return self._tool_settings.get(name, {})

    def all(self) -> list[ToolDefinition]:
        """Return every registered tool, in registration order."""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """Return every registered tool name."""
        return list(self._tools.keys())

    def lock(self) -> None:
        """Freeze the registry — no more registrations allowed."""
        self._locked = True

    def is_locked(self) -> bool:
        """Whether :meth:`lock` has been called."""
        return self._locked

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def to_tools_schema(self) -> ToolsSchema:
        """Build a Pipecat :class:`ToolsSchema` from every registered tool.

        Layer 8 hands the result to :class:`LLMContext` so the LLM
        sees the tool catalog when it generates a response.
        """
        return ToolsSchema(
            standard_tools=[t.to_function_schema() for t in self._tools.values()],
        )


def parse_disabled_tools(csv_string: str | None) -> set[str]:
    """Parse the ``disabled_tools`` CSV into a set of tool names.

    Whitespace is stripped from each entry and empty entries are
    dropped. Closes tech-debt log entry 5: Settings stores the raw
    CSV string; Layer 4 owns the parse.

    Args:
        csv_string: Raw CSV from
            :class:`~app.config.settings.Settings.disabled_tools`,
            or ``None``.

    Returns:
        Set of tool names to skip when building the registry.
    """
    if not csv_string:
        return set()
    return {name.strip() for name in csv_string.split(",") if name.strip()}


def build_registry_for_call(
    agent: AgentConfig,
    settings: Settings,
) -> ToolRegistry:
    """Build a per-call :class:`ToolRegistry` from agent + platform config.

    Filtering pipeline (in order):

    1. Drop any tool listed in ``Settings.disabled_tools`` — the
       operator kill-switch.
    2. Drop any tool not opted into by the agent — i.e., not
       present in ``AgentConfig.tools[]``.
    3. Apply the per-agent description override
       (``AgentConfig.tools[].description``) over the platform
       default. Empty / whitespace-only descriptions fall back to
       the default.
    4. Apply the ``transfer_call`` special case: if the agent's
       ``settings.targets`` is a non-empty dict, dynamically add
       an enum constraint to the ``target`` parameter so Claude
       is hard-limited to valid target names.

    The registry is locked before return — Layer 8 cannot mutate
    it after construction.

    Args:
        agent: The per-call agent config from the lambda.
        settings: The platform :class:`~app.config.settings.Settings`
            instance.

    Returns:
        A fully-built, locked :class:`ToolRegistry` ready for
        :meth:`ToolRegistry.to_tools_schema` consumption by Layer 8.
    """
    # Imported here to avoid a circular: catalog imports schema,
    # schema is imported by registry, but registry imports catalog
    # only at use time — keeps the module-load graph clean.
    from app.tools.builtin.catalog import BUILTIN_TOOLS

    disabled = parse_disabled_tools(settings.disabled_tools)
    agent_tools_by_type = {t.type: t for t in agent.tools}

    registry = ToolRegistry()

    for tool_name, tool_def in BUILTIN_TOOLS.items():
        if tool_name in disabled:
            logger.info(
                "tool_skipped",
                tool=tool_name,
                reason="disabled_by_config",
                agent=agent.name,
            )
            continue
        if tool_name not in agent_tools_by_type:
            logger.info(
                "tool_skipped",
                tool=tool_name,
                reason="not_in_agent_config",
                agent=agent.name,
            )
            continue

        agent_tool = agent_tools_by_type[tool_name]
        agent_settings = agent_tool.settings or {}

        # Per-agent description override. Whitespace-only is treated
        # as empty so a stray space in Aurora doesn't blank out the
        # platform default.
        override = (agent_tool.description or "").strip()
        tool = tool_def.with_description(override) if override else tool_def

        # transfer_call's target enum is per-agent: each agent has
        # its own set of named targets in settings["targets"]. Hard
        # constraint via enum is more reliable than relying on
        # Claude to remember the target names from the description.
        if tool_name == "transfer_call":
            targets = agent_settings.get("targets")
            if isinstance(targets, dict) and targets:
                tool = tool.with_enum("target", sorted(targets.keys()))

        registry.register(tool, settings=agent_settings)
        logger.info(
            "tool_registered",
            tool=tool_name,
            agent=agent.name,
            settings_keys=sorted(agent_settings.keys()),
        )

    registry.lock()
    return registry
