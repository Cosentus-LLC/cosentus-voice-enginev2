"""Unit tests for the Flows scaffold (#41).

The scaffold proves the Flows ↔ pipeline wiring without introducing
any real conversation steps. These tests assert two things:

* constructing the ``FlowManager`` is side-effect-free (no frames
  queued) — the safety guarantee behind "opener intact"; and
* the trivial 2-step flow (``start`` → ``advance`` → ``end``) advances.

Collaborators are mocked: a real ``FlowManager`` is exercised against a
mock ``PipelineTask`` (frame-queue spy), mock LLM service, mock
aggregator, and mock transport.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from app.flows import build_flow_manager, build_scaffold_flow
from app.flows.scaffold import END_NODE, START_NODE, _advance, _end_node
from app.flows.summary import DialogueOnlyFlowAdapter
from pipecat_flows import FlowManager


def _mock_task() -> MagicMock:
    task = MagicMock()
    task.queue_frame = AsyncMock()
    task.queue_frames = AsyncMock()
    return task


def _build(task: MagicMock | None = None) -> tuple[FlowManager, MagicMock]:
    task = task or _mock_task()
    llm = MagicMock()
    llm.register_function = MagicMock()
    aggregator = MagicMock()
    transport = MagicMock()
    fm = build_flow_manager(
        task=task,
        llm=llm,
        context_aggregator=aggregator,
        transport=transport,
    )
    return fm, task


class TestConstruction:
    def test_build_flow_manager_returns_uninitialized_manager(self):
        fm, _ = _build()
        assert isinstance(fm, FlowManager)
        # No node has been set — initialize() was never called.
        assert fm.current_node is None

    def test_construction_queues_no_frames(self):
        """The safety guarantee: constructing the FlowManager must not
        touch the pipeline. Zero frames queued at build time means the
        opener-seeded LLMContext is never disturbed (flag-off path)."""
        task = _mock_task()
        _build(task)
        task.queue_frame.assert_not_called()
        task.queue_frames.assert_not_called()

    def test_build_flow_manager_uses_dialogue_only_summary_adapter(self):
        fm, _ = _build()
        assert isinstance(fm._adapter, DialogueOnlyFlowAdapter)


class TestScaffoldFlow:
    def test_build_scaffold_flow_is_start_node(self):
        node = build_scaffold_flow()
        assert node["name"] == START_NODE
        # Nodes never auto-respond, so the flow never queues a competing
        # LLMRunFrame against the opener.
        assert node["respond_immediately"] is False
        task_blob = " ".join(message["content"] for message in node["task_messages"])
        assert "Call advance" not in task_blob

    async def test_initialize_sets_start_node(self):
        fm, _ = _build()
        await fm.initialize(build_scaffold_flow())
        assert fm.current_node == START_NODE

    async def test_flow_advances_start_to_end(self):
        fm, _ = _build()
        await fm.initialize(build_scaffold_flow())
        assert fm.current_node == START_NODE

        # Drive the transition the way the advance function's result would.
        await fm.set_node_from_config(_end_node())
        assert fm.current_node == END_NODE

    async def test_advance_handler_returns_end_node(self):
        fm, _ = _build()
        result, next_node = await _advance({}, fm)
        assert result == {"status": "ok"}
        assert next_node is not None
        assert next_node["name"] == END_NODE


@pytest.mark.parametrize("node", [build_scaffold_flow(), _end_node()])
def test_scaffold_nodes_never_respond_immediately(node):
    """Both scaffold nodes must keep respond_immediately False so the
    flow never queues an LLMRunFrame that would double-trigger the
    opener's generation."""
    assert node["respond_immediately"] is False
