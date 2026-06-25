"""Unit tests for the Flows ordered steps + per-step context (16c, #43).

Two guarantees under test:

* **Ordered + un-skippable** — the chain advances ``navigate → … → wrap``
  in order; ``wrap`` is only reachable through ``reference_number``, whose
  advance handler refuses to transition until a non-blank reference number
  is supplied (a deterministic code check, not a prompt).
* **Bounded per-step context** — the first step RESETs and loads the
  hydrated prompt; every later step uses RESET_WITH_SUMMARY so per-turn
  input stays bounded with a running summary carried forward.

Tests build nodes + call handlers directly (the ``test_identity_gate``
pattern) — they never drive the live ``FlowManager._update_llm_context``,
so the ``RESET_WITH_SUMMARY`` ``DeprecationWarning`` is not triggered and
``filterwarnings = error`` stays green.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from app.flows.steps import (
    DEADLINE,
    NAVIGATE,
    NAVIGATE_BASE_TASK,
    REFERENCE_NUMBER,
    REQUIRED_REFERENCE_FIELD,
    REQUIRED_REFERENCE_NODE_ID,
    STEPS,
    WRAP,
    build_navigate_task,
    build_step_chain,
    identity_gate_required_for_direction,
)
from app.knowledge.prefetch import PrefetchContext
from app.knowledge.semantic_cache import CacheHit
from pipecat_flows import ContextStrategy


def _fm() -> SimpleNamespace:
    """Minimal FlowManager stand-in: only ``.state`` is used by handlers."""
    return SimpleNamespace(state={})


def _registry(tool_names: list[str]) -> MagicMock:
    """A ToolRegistry stand-in whose tools surface as Flows functions.

    ``get(name).to_function_schema().name == name`` so the wrapped tool
    function keeps the tool's own name.
    """
    reg = MagicMock()
    reg.names.return_value = list(tool_names)

    def _get(name: str) -> MagicMock:
        spec = MagicMock()
        spec.cancel_on_interruption = False
        spec.timeout_secs = 30.0
        spec.to_function_schema.return_value = SimpleNamespace(
            name=name,
            description=f"{name} description",
            properties={},
            required=[],
        )
        return spec

    reg.get.side_effect = _get
    return reg


def _chain(
    tool_names: list[str] | None = None,
    hydrated_system: str = "HYDRATED-SYSTEM-PROMPT",
    flow_definition: dict | None = None,
):
    return build_step_chain(
        run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
        registry=_registry(tool_names or []),
        hydrated_system=hydrated_system,
        flow_definition=flow_definition,
    )


def _advance_fn(node):
    """The chain always appends the advance function last (after tools)."""
    return node["functions"][-1]


def _custom_flow() -> dict:
    return {
        "version": 1,
        "start": "intro",
        "nodes": [
            {
                "id": "intro",
                "type": "ask",
                "label": "Intro",
                "say": "Ask the representative to confirm the denial status.",
                "next": "route_submission",
            },
            {
                "id": "route_submission",
                "type": "branch",
                "label": "Route submission",
                "say": "Decide whether the payer requires fax or portal submission.",
                "branches": [
                    {"when": "Representative says fax is required", "to": "fax_path"},
                    {"when": "Representative says portal is required", "to": "portal_path"},
                ],
                "fallback": "fax_path",
            },
            {
                "id": "fax_path",
                "type": "ask",
                "say": "Confirm the fax number.",
                "capture": ["fax_number"],
                "next": REQUIRED_REFERENCE_NODE_ID,
            },
            {
                "id": "portal_path",
                "type": "ask",
                "say": "Confirm the portal name.",
                "capture": ["portal_name"],
                "next": REQUIRED_REFERENCE_NODE_ID,
            },
            {
                "id": REQUIRED_REFERENCE_NODE_ID,
                "type": "ask",
                "say": "Ask for a call reference number.",
                "capture": [REQUIRED_REFERENCE_FIELD],
                "required": True,
                "next": "done",
            },
            {"id": "done", "type": "end", "say": "Wrap up and end the call."},
        ],
    }


async def _walk_to(node, target_name):
    """Advance from ``node`` to ``target_name`` supplying no extra args.

    Works because every step before ``reference_number`` is un-gated.
    """
    fm = _fm()
    while node["name"] != target_name:
        _result, node = await _advance_fn(node).handler({}, fm)
        assert node is not None, f"stalled before reaching {target_name}"
    return node


# ── Ordering ─────────────────────────────────────────────────────────────


class TestDirectionAwareStart:
    def test_identity_gate_required_only_for_inbound(self):
        assert identity_gate_required_for_direction("inbound") is True
        assert identity_gate_required_for_direction("outbound") is False
        assert identity_gate_required_for_direction("browser") is False

    def test_unknown_direction_is_not_gated_by_default(self):
        assert identity_gate_required_for_direction("unknown") is False


class TestOrdering:
    @pytest.mark.asyncio
    async def test_chain_is_linear_and_in_step_order(self):
        node = _chain()
        fm = _fm()
        names: list[str] = []
        while True:
            names.append(node["name"])
            fns = node["functions"]
            if not fns:  # terminal node (wrap) has no advance
                break
            advance = _advance_fn(node)
            # Supply the required field for the only gated step.
            args = {"reference_number": "REF-12345"} if node["name"] == REFERENCE_NUMBER else {}
            _result, node = await advance.handler(args, fm)
            assert node is not None

        assert names == [s.name for s in STEPS]
        assert names[0] == NAVIGATE
        assert names[-1] == WRAP

    def test_first_node_is_navigate(self):
        assert _chain()["name"] == NAVIGATE

    def test_terminal_step_has_no_advance(self):
        # wrap advertises tools (if any) but no advance function.
        wrap = STEPS[-1]
        assert wrap.name == WRAP
        assert wrap.advance_name == ""


# ── Data-driven flow definitions (#5) ────────────────────────────────────


class TestDataDrivenFlow:
    @pytest.mark.asyncio
    async def test_custom_flow_builds_nodes_and_edges_from_runtime_definition(self):
        node = _chain(flow_definition=_custom_flow())
        fm = _fm()

        assert node["name"] == "intro"
        _result, node = await _advance_fn(node).handler({}, fm)
        assert node["name"] == "route_submission"
        _result, node = await _advance_fn(node).handler({"branch_to": "portal_path"}, fm)
        assert node["name"] == "portal_path"
        _result, node = await _advance_fn(node).handler({"portal_name": "Availity"}, fm)
        assert node["name"] == REQUIRED_REFERENCE_NODE_ID
        _result, node = await _advance_fn(node).handler({REQUIRED_REFERENCE_FIELD: "REF-1"}, fm)
        assert node["name"] == "done"

    @pytest.mark.asyncio
    async def test_no_flow_definition_uses_default_step_order(self):
        node = _chain(flow_definition=None)
        fm = _fm()
        names: list[str] = []
        while True:
            names.append(node["name"])
            if not node["functions"]:
                break
            args = {"reference_number": "REF-1"} if node["name"] == REFERENCE_NUMBER else {}
            _result, node = await _advance_fn(node).handler(args, fm)

        assert names == [s.name for s in STEPS]

    def test_malformed_flow_definition_falls_back_to_default_and_logs_warning(self, mocker):
        mock_logger = mocker.patch("app.flows.steps.logger")
        flow = {
            "version": 1,
            "start": "intro",
            "nodes": [{"id": "intro", "type": "end"}],
        }

        chain = _chain(flow_definition=flow)

        assert chain["name"] == NAVIGATE
        mock_logger.warning.assert_called_with(
            "flow_definition_invalid_default_flow",
            reason="validation_error",
            error_count=ANY,
            errors=ANY,
        )

    @pytest.mark.asyncio
    async def test_custom_reference_number_requires_call_reference(self):
        node = await _walk_to(_chain(flow_definition=_custom_flow()), REQUIRED_REFERENCE_NODE_ID)

        result, next_node = await _advance_fn(node).handler({REQUIRED_REFERENCE_FIELD: " "}, _fm())

        assert next_node is None
        assert result["status"] == "missing"
        assert result["field"] == REQUIRED_REFERENCE_FIELD

    @pytest.mark.asyncio
    async def test_custom_reference_number_records_both_state_keys(self):
        node = await _walk_to(_chain(flow_definition=_custom_flow()), REQUIRED_REFERENCE_NODE_ID)
        fm = _fm()

        result, next_node = await _advance_fn(node).handler(
            {REQUIRED_REFERENCE_FIELD: "REF-123"},
            fm,
        )

        assert next_node["name"] == "done"
        assert result[REQUIRED_REFERENCE_FIELD] == "REF-123"
        assert fm.state[REQUIRED_REFERENCE_FIELD] == "REF-123"
        assert fm.state[REFERENCE_NUMBER] == "REF-123"

    @pytest.mark.asyncio
    async def test_custom_branch_uses_selected_branch_target(self):
        flow = _custom_flow()
        _result, branch = await _advance_fn(_chain(flow_definition=flow)).handler({}, _fm())

        _result, next_node = await _advance_fn(branch).handler({"branch_to": "portal_path"}, _fm())

        assert next_node["name"] == "portal_path"

    @pytest.mark.asyncio
    async def test_custom_branch_missing_or_invalid_selection_uses_fallback(self):
        flow = _custom_flow()
        _result, branch = await _advance_fn(_chain(flow_definition=flow)).handler({}, _fm())

        _result, missing_next = await _advance_fn(branch).handler({}, _fm())
        _result, invalid_next = await _advance_fn(branch).handler({"branch_to": "nope"}, _fm())

        assert missing_next["name"] == "fax_path"
        assert invalid_next["name"] == "fax_path"

    @pytest.mark.asyncio
    async def test_custom_first_node_resets_and_later_nodes_use_summary(self):
        node = _chain(hydrated_system="HYDRATED-PHI-PROMPT", flow_definition=_custom_flow())

        assert node["role_message"] == "HYDRATED-PHI-PROMPT"
        assert node["context_strategy"].strategy == ContextStrategy.RESET
        _result, node = await _advance_fn(node).handler({}, _fm())
        assert "role_message" not in node
        assert node["context_strategy"].strategy == ContextStrategy.RESET_WITH_SUMMARY

    def test_custom_flow_advertises_tools_at_every_non_terminal_node(self):
        node = _chain(tool_names=["end_call", "transfer_call"], flow_definition=_custom_flow())

        names = [f.name for f in node["functions"]]

        assert "end_call" in names
        assert "transfer_call" in names
        assert _advance_fn(node).name == "advance_intro"

    @pytest.mark.asyncio
    async def test_custom_deadline_node_keeps_cached_knowledge_hint(self):
        flow = {
            "version": 1,
            "start": "deadline",
            "nodes": [
                {
                    "id": "deadline",
                    "type": "ask",
                    "say": "Confirm the timely filing deadline.",
                    "next": REQUIRED_REFERENCE_NODE_ID,
                },
                {
                    "id": REQUIRED_REFERENCE_NODE_ID,
                    "type": "ask",
                    "capture": [REQUIRED_REFERENCE_FIELD],
                    "required": True,
                    "next": "done",
                },
                {"id": "done", "type": "end"},
            ],
        }
        warmer = MagicMock()
        warmer.live_read.return_value = CacheHit(
            value="Aetna timely filing is 120 days.",
            similarity=1.0,
            query="timely filing limit for Aetna",
        )

        chain = build_step_chain(
            run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
            registry=_registry([]),
            hydrated_system="SYS",
            knowledge_warmer=warmer,
            knowledge_context=PrefetchContext(payer="Aetna"),
            flow_definition=flow,
        )

        assert "Known payer-level fact: Aetna timely filing is 120 days." in _navigate_task(chain)


# ── Un-skippable reference number ─────────────────────────────────────────


class TestReferenceNumberRequired:
    @pytest.mark.asyncio
    async def test_blank_reference_number_blocks_advance(self):
        node = await _walk_to(_chain(), REFERENCE_NUMBER)
        result, next_node = await _advance_fn(node).handler({"reference_number": "   "}, _fm())
        assert next_node is None  # stays on the node — un-skippable
        assert result["status"] == "missing"
        assert result["field"] == "reference_number"

    @pytest.mark.asyncio
    async def test_missing_reference_number_blocks_advance(self):
        node = await _walk_to(_chain(), REFERENCE_NUMBER)
        result, next_node = await _advance_fn(node).handler({}, _fm())
        assert next_node is None
        assert result["status"] == "missing"

    @pytest.mark.asyncio
    async def test_present_reference_number_advances_to_wrap_and_records_state(self):
        node = await _walk_to(_chain(), REFERENCE_NUMBER)
        fm = _fm()
        result, next_node = await _advance_fn(node).handler({"reference_number": "REF-9"}, fm)
        assert next_node is not None
        assert next_node["name"] == WRAP
        assert result["reference_number"] == "REF-9"
        assert fm.state["reference_number"] == "REF-9"

    @pytest.mark.asyncio
    async def test_cannot_reach_wrap_without_a_reference_number(self):
        # Walk supplying NO args anywhere: the call must stall at
        # reference_number and never reach wrap.
        node = _chain()
        fm = _fm()
        visited: list[str] = []
        for _ in range(len(STEPS) + 2):
            visited.append(node["name"])
            fns = node["functions"]
            if not fns:
                break
            _result, nxt = await _advance_fn(node).handler({}, fm)
            if nxt is None:
                break
            node = nxt
        assert WRAP not in visited
        assert visited[-1] == REFERENCE_NUMBER

    @pytest.mark.asyncio
    async def test_reference_number_advance_requires_the_field(self):
        node = await _walk_to(_chain(), REFERENCE_NUMBER)
        advance = _advance_fn(node)
        assert advance.required == ["reference_number"]
        assert "reference_number" in advance.properties


# ── Per-step bounded context ──────────────────────────────────────────────


class TestPerStepContext:
    def test_first_step_resets_and_loads_hydrated_prompt(self):
        chain = _chain(hydrated_system="HYDRATED-PHI-PROMPT")
        assert chain["role_message"] == "HYDRATED-PHI-PROMPT"
        assert chain["context_strategy"].strategy == ContextStrategy.RESET
        # RESET (not summary) so identity-gate chatter is dropped cleanly.
        assert chain["context_strategy"].summary_prompt is None

    @pytest.mark.asyncio
    async def test_later_steps_use_reset_with_summary(self):
        # Every step after the first carries a running summary forward.
        node = _chain()
        fm = _fm()
        first = True
        while True:
            if not first:
                assert node["context_strategy"].strategy == ContextStrategy.RESET_WITH_SUMMARY
                assert node["context_strategy"].summary_prompt  # non-empty
                # role_message persists from the first node; later nodes
                # don't re-send it.
                assert "role_message" not in node
            first = False
            fns = node["functions"]
            if not fns:
                break
            args = {"reference_number": "R-1"} if node["name"] == REFERENCE_NUMBER else {}
            _result, node = await _advance_fn(node).handler(args, fm)

    @pytest.mark.asyncio
    async def test_every_step_bounds_context_with_a_reset_family_strategy(self):
        node = _chain()
        fm = _fm()
        while True:
            assert node["context_strategy"].strategy in (
                ContextStrategy.RESET,
                ContextStrategy.RESET_WITH_SUMMARY,
            )
            fns = node["functions"]
            if not fns:
                break
            args = {"reference_number": "R-1"} if node["name"] == REFERENCE_NUMBER else {}
            _result, node = await _advance_fn(node).handler(args, fm)


# ── Tools delegate to the shared execution core ───────────────────────────


class TestStepTools:
    @pytest.mark.asyncio
    async def test_step_tool_delegates_to_run_tool_core(self):
        core = AsyncMock(return_value=({"ok": True}, False))
        chain = build_step_chain(
            run_tool_core=core,
            registry=_registry(["end_call"]),
            hydrated_system="X",
        )
        tool_fn = next(f for f in chain["functions"] if f.name == "end_call")
        payload, next_node = await tool_fn.handler({"reason": "done"}, _fm())

        core.assert_awaited_once_with("end_call", {"reason": "done"})
        assert payload == {"ok": True}
        assert next_node is None  # tools don't transition the flow

    def test_tools_advertised_at_every_step(self):
        chain = _chain(tool_names=["end_call", "transfer_call"])
        # navigate: 2 tools + 1 advance.
        names = [f.name for f in chain["functions"]]
        assert "end_call" in names
        assert "transfer_call" in names


# ── Verified IVR path feeding the navigate step (#17) ─────────────────────


def _navigate_task(chain) -> str:
    """The navigate node's task text (its single system task message)."""
    return chain["task_messages"][0]["content"]


class TestNavigateTask:
    def test_no_path_or_goal_is_byte_identical_to_base(self):
        # The default (listen-and-decide) navigate task must be unchanged.
        assert build_navigate_task() == NAVIGATE_BASE_TASK
        assert build_navigate_task("", "") == NAVIGATE_BASE_TASK
        assert build_navigate_task("   ", "   ") == NAVIGATE_BASE_TASK
        # And the chain's first node reflects that verbatim.
        assert _navigate_task(_chain()) == NAVIGATE_BASE_TASK

    def test_includes_verified_path_when_present(self):
        path = "1. Provider services — press 3\n2. Claims — press 1"
        task = build_navigate_task(ivr_path=path)
        assert NAVIGATE_BASE_TASK in task
        assert path in task

    def test_path_instructs_listen_and_decide_fallback(self):
        task = build_navigate_task(ivr_path="1. press 3")
        assert "navigate by ear" in task.lower()

    def test_includes_ivr_goal_when_present(self):
        task = build_navigate_task(ivr_goal="Reach a claims rep")
        assert NAVIGATE_BASE_TASK in task
        assert "Reach a claims rep" in task

    def test_goal_and_path_both_present(self):
        task = build_navigate_task(ivr_path="1. press 3", ivr_goal="Reach a rep")
        assert "Reach a rep" in task
        assert "1. press 3" in task

    @pytest.mark.asyncio
    async def test_build_step_chain_threads_path_into_navigate_node_only(self):
        chain = build_step_chain(
            run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
            registry=_registry([]),
            hydrated_system="SYS",
            ivr_path="1. Claims — press 1",
            ivr_goal="Reach a rep",
        )
        assert chain["name"] == NAVIGATE
        nav_task = _navigate_task(chain)
        assert "1. Claims — press 1" in nav_task
        assert "Reach a rep" in nav_task
        # Later steps keep their fixed task verbatim (path not leaked).
        _result, greet_node = await _advance_fn(chain).handler({}, _fm())
        assert "press 1" not in _navigate_task(greet_node)


class TestKnowledgeTaskHints:
    @pytest.mark.asyncio
    async def test_build_step_chain_without_knowledge_keeps_deadline_task_verbatim(self):
        chain = _chain()
        deadline = await _walk_to(chain, DEADLINE)

        expected = next(step.task for step in STEPS if step.name == DEADLINE)
        assert _navigate_task(deadline) == expected

    @pytest.mark.asyncio
    async def test_deadline_step_appends_cached_timely_filing_fact_on_hit(self):
        warmer = MagicMock()
        warmer.live_read.return_value = CacheHit(
            value="Aetna timely filing is 120 days.",
            similarity=1.0,
            query="timely filing limit for Aetna",
        )
        chain = build_step_chain(
            run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
            registry=_registry([]),
            hydrated_system="SYS",
            knowledge_warmer=warmer,
            knowledge_context=PrefetchContext(payer="Aetna"),
        )

        deadline = await _walk_to(chain, DEADLINE)
        task = _navigate_task(deadline)

        assert "Known payer-level fact: Aetna timely filing is 120 days." in task
        warmer.live_read.assert_called_with("timely filing limit for Aetna")

    @pytest.mark.asyncio
    async def test_deadline_step_miss_leaves_task_unchanged_and_continues(self):
        warmer = MagicMock()
        warmer.live_read.return_value = None
        chain = build_step_chain(
            run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
            registry=_registry([]),
            hydrated_system="SYS",
            knowledge_warmer=warmer,
            knowledge_context=PrefetchContext(payer="Aetna"),
        )

        deadline = await _walk_to(chain, DEADLINE)

        expected = next(step.task for step in STEPS if step.name == DEADLINE)
        assert _navigate_task(deadline) == expected

    @pytest.mark.asyncio
    async def test_knowledge_live_read_not_called_when_no_payer_context(self):
        warmer = MagicMock()
        chain = build_step_chain(
            run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
            registry=_registry([]),
            hydrated_system="SYS",
            knowledge_warmer=warmer,
            knowledge_context=PrefetchContext(payer=None),
        )

        await _walk_to(chain, DEADLINE)

        warmer.live_read.assert_not_called()
