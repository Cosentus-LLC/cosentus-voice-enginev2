"""Tests for app.tools.registry — ToolRegistry + build_registry_for_call."""

from __future__ import annotations

import pytest
from app.config.agent_config import AgentConfig, ToolConfig
from app.config.settings import Settings
from app.tools.registry import (
    ToolRegistry,
    build_registry_for_call,
)
from app.tools.result import success_result
from app.tools.schema import ToolDefinition, ToolParameter
from pipecat.adapters.schemas.tools_schema import ToolsSchema


async def _noop_executor(args: dict, ctx) -> object:
    return success_result()


def _tool_def(name: str = "demo_tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        parameters=[
            ToolParameter(name="x", type="string", description="x"),
        ],
        executor=_noop_executor,
    )


# ── ToolRegistry — direct API ───────────────────────────────────────────────


class TestToolRegistry:
    def test_register_adds_tool(self):
        registry = ToolRegistry()
        tool = _tool_def("tool_a")
        registry.register(tool)
        assert registry.get("tool_a") is tool
        assert "tool_a" in registry
        assert len(registry) == 1

    def test_register_with_settings(self):
        registry = ToolRegistry()
        registry.register(_tool_def("tool_a"), settings={"k": "v"})
        assert registry.get_settings("tool_a") == {"k": "v"}

    def test_get_returns_none_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get("missing") is None

    def test_get_settings_returns_empty_dict_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get_settings("missing") == {}

    def test_register_raises_on_duplicate_name(self):
        registry = ToolRegistry()
        registry.register(_tool_def("tool_a"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_tool_def("tool_a"))

    def test_register_raises_on_empty_name(self):
        registry = ToolRegistry()
        empty_named = ToolDefinition(
            name="",
            description="d",
            parameters=[],
            executor=_noop_executor,
        )
        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register(empty_named)

    def test_register_raises_on_locked_registry(self):
        registry = ToolRegistry()
        registry.register(_tool_def("tool_a"))
        registry.lock()
        with pytest.raises(RuntimeError, match="locked"):
            registry.register(_tool_def("tool_b"))

    def test_lock_marks_registry_locked(self):
        registry = ToolRegistry()
        assert registry.is_locked() is False
        registry.lock()
        assert registry.is_locked() is True

    def test_all_returns_registration_order(self):
        registry = ToolRegistry()
        registry.register(_tool_def("alpha"))
        registry.register(_tool_def("beta"))
        registry.register(_tool_def("gamma"))
        assert [t.name for t in registry.all()] == ["alpha", "beta", "gamma"]

    def test_to_tools_schema_returns_pipecat_object_with_all_tools(self):
        registry = ToolRegistry()
        registry.register(_tool_def("tool_a"))
        registry.register(_tool_def("tool_b"))
        schema = registry.to_tools_schema()
        assert isinstance(schema, ToolsSchema)
        names = sorted(t.name for t in schema.standard_tools)
        assert names == ["tool_a", "tool_b"]


# ── build_registry_for_call — integration ──────────────────────────────────


@pytest.fixture
def settings_no_disabled(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "test-lambda")
    monkeypatch.setenv("API_KEY_SECRET_ARN", "arn:test")
    monkeypatch.delenv("DISABLED_TOOLS", raising=False)
    return Settings(_env_file=None)


def _agent(*tools: ToolConfig) -> AgentConfig:
    return AgentConfig(name="test-agent", tools=list(tools))


class TestBuildRegistryForCall:
    def test_no_agent_tools_means_empty_registry(self, settings_no_disabled: Settings):
        registry = build_registry_for_call(_agent(), settings_no_disabled)
        assert registry.names() == []
        assert registry.is_locked()

    def test_agent_opt_in_registers_tool(self, settings_no_disabled: Settings):
        agent = _agent(ToolConfig(type="end_call"))
        registry = build_registry_for_call(agent, settings_no_disabled)
        assert registry.names() == ["end_call"]

    def test_unknown_tool_type_in_agent_config_silently_skipped(
        self, settings_no_disabled: Settings
    ):
        # Aurora may surface a tool name we don't ship — drop it,
        # don't crash.
        agent = _agent(ToolConfig(type="not_a_real_tool"))
        registry = build_registry_for_call(agent, settings_no_disabled)
        assert registry.names() == []

    def test_disabled_tool_is_filtered(
        self, monkeypatch: pytest.MonkeyPatch, settings_no_disabled: Settings
    ):
        # Disable end_call via Settings.disabled_tools even though
        # the agent has opted into it.
        monkeypatch.setenv("DISABLED_TOOLS", "end_call")
        settings = Settings(_env_file=None)

        agent = _agent(
            ToolConfig(type="end_call"),
            ToolConfig(type="press_digit"),
        )
        registry = build_registry_for_call(agent, settings)
        assert "end_call" not in registry.names()
        assert "press_digit" in registry.names()

    def test_per_agent_description_overrides_default(self, settings_no_disabled: Settings):
        custom = "Custom press_digit description for Chris."
        agent = _agent(
            ToolConfig(type="press_digit", description=custom),
        )
        registry = build_registry_for_call(agent, settings_no_disabled)
        assert registry.get("press_digit").description == custom

    def test_empty_or_whitespace_description_falls_back_to_default(
        self, settings_no_disabled: Settings
    ):
        # Empty string and whitespace-only should both leave the
        # platform default in place.
        from app.tools.builtin import PRESS_DIGIT

        for blank in ("", "   ", "\t\n"):
            agent = _agent(ToolConfig(type="press_digit", description=blank))
            registry = build_registry_for_call(agent, settings_no_disabled)
            assert registry.get("press_digit").description == PRESS_DIGIT.description

    def test_transfer_call_target_enum_applied_from_agent_settings(
        self, settings_no_disabled: Settings
    ):
        agent = _agent(
            ToolConfig(
                type="transfer_call",
                settings={
                    "targets": {
                        "billing": "+15551112222",
                        "after_hours": "+15553334444",
                    }
                },
            ),
        )
        registry = build_registry_for_call(agent, settings_no_disabled)
        target_param = next(
            p for p in registry.get("transfer_call").parameters if p.name == "target"
        )
        # Sorted for stability.
        assert target_param.enum == ["after_hours", "billing"]

    def test_transfer_call_no_targets_leaves_enum_unset(self, settings_no_disabled: Settings):
        # Agent didn't supply any targets — Claude gets the plain
        # string parameter (executor will error at call time).
        agent = _agent(ToolConfig(type="transfer_call"))
        registry = build_registry_for_call(agent, settings_no_disabled)
        target_param = next(
            p for p in registry.get("transfer_call").parameters if p.name == "target"
        )
        assert target_param.enum is None

    def test_per_agent_settings_stored_on_registry(self, settings_no_disabled: Settings):
        agent = _agent(
            ToolConfig(
                type="transfer_call",
                settings={"targets": {"x": "+15550000"}},
            ),
        )
        registry = build_registry_for_call(agent, settings_no_disabled)
        assert registry.get_settings("transfer_call") == {"targets": {"x": "+15550000"}}

    def test_registry_is_locked_after_build(self, settings_no_disabled: Settings):
        agent = _agent(ToolConfig(type="end_call"))
        registry = build_registry_for_call(agent, settings_no_disabled)
        assert registry.is_locked()
