"""Tests for app.tools.schema — ToolDefinition + ToolParameter."""

from __future__ import annotations

import dataclasses

import pytest
from app.tools.result import success_result
from app.tools.schema import ToolDefinition, ToolParameter
from pipecat.adapters.schemas.function_schema import FunctionSchema


async def _noop_executor(args: dict, ctx) -> object:
    return success_result()


def _tool(
    *,
    name: str = "demo_tool",
    description: str = "Demo description",
    parameters: list[ToolParameter] | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters
        or [
            ToolParameter(name="x", type="string", description="x param", required=True),
        ],
        executor=_noop_executor,
    )


class TestToolParameter:
    def test_constructs_with_required_fields(self):
        p = ToolParameter(
            name="digits",
            type="string",
            description="DTMF digits",
        )
        assert p.name == "digits"
        assert p.type == "string"
        assert p.description == "DTMF digits"
        assert p.required is True
        assert p.enum is None
        assert p.pattern is None

    def test_is_immutable(self):
        # frozen=True
        p = ToolParameter(name="x", type="string", description="d")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.name = "changed"  # type: ignore[misc]


class TestToolDefinition:
    def test_to_function_schema_basic(self):
        tool = _tool()
        schema = tool.to_function_schema()
        assert isinstance(schema, FunctionSchema)
        assert schema.name == "demo_tool"
        assert schema.description == "Demo description"
        assert schema.properties == {
            "x": {"type": "string", "description": "x param"},
        }
        assert schema.required == ["x"]

    def test_to_function_schema_with_enum_and_pattern(self):
        tool = _tool(
            parameters=[
                ToolParameter(
                    name="target",
                    type="string",
                    description="target",
                    required=True,
                    enum=["a", "b"],
                ),
                ToolParameter(
                    name="digits",
                    type="string",
                    description="digits",
                    required=True,
                    pattern=r"^[0-9]+$",
                ),
            ],
        )
        schema = tool.to_function_schema()
        assert schema.properties["target"]["enum"] == ["a", "b"]
        assert schema.properties["digits"]["pattern"] == r"^[0-9]+$"

    def test_to_function_schema_excludes_optional_from_required(self):
        tool = _tool(
            parameters=[
                ToolParameter(
                    name="opt",
                    type="string",
                    description="d",
                    required=False,
                ),
            ],
        )
        schema = tool.to_function_schema()
        assert schema.required == []

    def test_with_description_returns_copy(self):
        tool = _tool(description="original")
        new_tool = tool.with_description("new copy")
        assert tool.description == "original"
        assert new_tool.description == "new copy"
        assert new_tool is not tool

    def test_with_enum_applies_to_named_param(self):
        tool = _tool(
            parameters=[
                ToolParameter(name="target", type="string", description="d"),
                ToolParameter(name="other", type="string", description="d"),
            ],
        )
        new_tool = tool.with_enum("target", ["a", "b", "c"])
        target = next(p for p in new_tool.parameters if p.name == "target")
        other = next(p for p in new_tool.parameters if p.name == "other")
        assert target.enum == ["a", "b", "c"]
        assert other.enum is None

    def test_with_enum_unknown_param_raises(self):
        tool = _tool()
        with pytest.raises(KeyError, match="not_a_param"):
            tool.with_enum("not_a_param", ["a"])

    def test_definition_is_immutable(self):
        tool = _tool()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tool.name = "changed"  # type: ignore[misc]
