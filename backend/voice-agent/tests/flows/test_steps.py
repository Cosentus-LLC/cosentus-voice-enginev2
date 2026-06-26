"""Unit tests for the Flows ordered steps + per-step context (16c, #43).

Two guarantees under test:

* **Ordered + un-skippable** — the default chain advances ``navigate → … →
  wrap`` in order; required-capture advance handlers refuse to transition
  until their required fields are supplied non-blank (a deterministic code
  check, not a prompt).
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
    ASK_NEEDS,
    CONFIRM_DENIAL,
    DEADLINE,
    FAX_PORTAL,
    GREET,
    GREETING_ALREADY_DONE_NOTE,
    GREETING_STATE_KEY,
    NAVIGATE,
    NAVIGATE_BASE_TASK,
    NEUTRAL_ADVANCE_NAME,
    NEUTRAL_ASSIST,
    NEUTRAL_CLOSE,
    REFERENCE_NUMBER,
    STEP_COMPLETION_ROLE_RULE,
    STEPS,
    WRAP,
    _build_default_step_chain,
    build_navigate_task,
    build_step_chain,
    identity_gate_required_for_direction,
)
from app.knowledge.prefetch import PrefetchContext
from app.knowledge.semantic_cache import CacheHit
from pipecat_flows import ContextStrategy

CALL_REFERENCE_FIELD = "call_reference"


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
    greeting_state: dict[str, bool] | None = None,
    include_ivr: bool = True,
    agent_id: str = "",
):
    return build_step_chain(
        run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
        registry=_registry(tool_names or []),
        hydrated_system=hydrated_system,
        flow_definition=flow_definition,
        greeting_state=greeting_state,
        include_ivr=include_ivr,
        agent_id=agent_id,
    )


def _default_chain(
    tool_names: list[str] | None = None,
    hydrated_system: str = "HYDRATED-SYSTEM-PROMPT",
    greeting_state: dict[str, bool] | None = None,
    include_ivr: bool = True,
    ivr_path: str = "",
    ivr_goal: str = "",
    knowledge_warmer=None,
    knowledge_context=None,
):
    """Build the claim 8-step chain directly.

    Since #21, ``build_step_chain`` no longer routes a missing/invalid
    ``flow_definition`` to the claim chain — it falls back to the neutral
    chain. ``_build_default_step_chain`` is retained as the claim-flow
    seed-shape reference, so the claim-chain ordering/greeting/IVR/prefetch
    tests target it directly here.
    """
    return _build_default_step_chain(
        run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
        registry=_registry(tool_names or []),
        hydrated_system=hydrated_system,
        greeting_state=greeting_state,
        include_ivr=include_ivr,
        ivr_path=ivr_path,
        ivr_goal=ivr_goal,
        knowledge_warmer=knowledge_warmer,
        knowledge_context=knowledge_context,
    )


def _advance_fn(node):
    """The chain always appends the advance function last (after tools)."""
    return node["functions"][-1]


def _message_blob(node) -> str:
    return " ".join(message["content"] for message in node["task_messages"])


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
                "next": REFERENCE_NUMBER,
            },
            {
                "id": "portal_path",
                "type": "ask",
                "say": "Confirm the portal name.",
                "capture": ["portal_name"],
                "next": REFERENCE_NUMBER,
            },
            {
                "id": REFERENCE_NUMBER,
                "type": "ask",
                "say": "Ask for a call reference number.",
                "capture": [CALL_REFERENCE_FIELD],
                "required": True,
                "next": "done",
            },
            {"id": "done", "type": "end", "say": "Wrap up and end the call."},
        ],
    }


def _custom_flow_starting_at_navigate() -> dict:
    flow = _custom_flow()
    flow["start"] = NAVIGATE
    flow["nodes"] = [
        {
            "id": NAVIGATE,
            "type": "ask",
            "label": "Navigate",
            "ivr": True,
            "say": "Navigate the payer IVR.",
            "next": "intro",
        },
        *flow["nodes"],
    ]
    return flow


def _patient_balance_flow() -> dict:
    return {
        "version": 1,
        "start": "confirm_balance",
        "nodes": [
            {
                "id": "confirm_balance",
                "type": "ask",
                "label": "Confirm balance",
                "say": "Confirm the patient's current balance.",
                "capture": ["balance_amount"],
                "next": "payment_options",
            },
            {
                "id": "payment_options",
                "type": "ask",
                "label": "Payment options",
                "say": "Explain the available payment options.",
                "next": "done",
            },
            {"id": "done", "type": "end", "say": "Close the call."},
        ],
    }


def _required_balance_flow() -> dict:
    return {
        "version": 1,
        "start": "collect_balance_details",
        "nodes": [
            {
                "id": "collect_balance_details",
                "type": "ask",
                "label": "Collect balance details",
                "say": "Collect the patient's balance amount and due date.",
                "capture": ["balance_amount", "due_date"],
                "required": True,
                "next": "done",
            },
            {"id": "done", "type": "end", "say": "Close the call."},
        ],
    }


async def _walk_to(node, target_name):
    """Advance from ``node`` to ``target_name`` supplying no extra args.

    Works when every step before ``target_name`` is un-gated.
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
        node = _default_chain()
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
        assert _default_chain()["name"] == NAVIGATE

    def test_terminal_step_has_no_advance(self):
        # wrap advertises tools (if any) but no advance function.
        wrap = STEPS[-1]
        assert wrap.name == WRAP
        assert wrap.advance_name == ""


class TestIvrApplicability:
    def test_default_no_ivr_chain_starts_at_greet(self):
        node = _default_chain(include_ivr=False, greeting_state={GREETING_STATE_KEY: False})

        assert node["name"] == GREET
        assert node["role_message"].endswith(STEP_COMPLETION_ROLE_RULE)
        assert node["context_strategy"].strategy == ContextStrategy.RESET

    @pytest.mark.asyncio
    async def test_default_no_ivr_chain_omits_navigate_in_walk(self):
        node = _default_chain(include_ivr=False, greeting_state={GREETING_STATE_KEY: False})
        fm = _fm()
        names: list[str] = []

        while True:
            names.append(node["name"])
            if not node["functions"]:
                break
            args = {"reference_number": "REF-12345"} if node["name"] == REFERENCE_NUMBER else {}
            _result, node = await _advance_fn(node).handler(args, fm)
            assert node is not None

        assert NAVIGATE not in names
        assert names == [step.name for step in STEPS[1:]]

    def test_default_no_ivr_chain_respects_existing_greeting_state(self):
        node = _default_chain(include_ivr=False, greeting_state={GREETING_STATE_KEY: True})

        assert node["name"] == CONFIRM_DENIAL
        task_blob = _message_blob(node)
        assert GREETING_ALREADY_DONE_NOTE in task_blob
        assert "Introduce yourself" not in task_blob

    def test_data_driven_no_ivr_skips_starting_navigate_when_unambiguous(self):
        node = _chain(
            flow_definition=_custom_flow_starting_at_navigate(),
            include_ivr=False,
            greeting_state={GREETING_STATE_KEY: False},
        )

        assert node["name"] == "intro"
        assert NAVIGATE not in _message_blob(node)

    def test_data_driven_no_ivr_skips_any_ivr_flagged_start_node(self):
        # The skip is flag-driven: an arbitrarily-named ``ivr: true`` start
        # node with a single outgoing target is dropped for no-IVR calls.
        flow = {
            "version": 1,
            "start": "payer_menu_entry",
            "nodes": [
                {
                    "id": "payer_menu_entry",
                    "type": "ask",
                    "ivr": True,
                    "say": "Navigate to a claims representative.",
                    "next": "intro",
                },
                {
                    "id": "intro",
                    "type": "ask",
                    "say": "Ask the representative to confirm the denial.",
                    "next": "done",
                },
                {"id": "done", "type": "end"},
            ],
        }
        node = _chain(
            flow_definition=flow,
            include_ivr=False,
            greeting_state={GREETING_STATE_KEY: False},
        )

        assert node["name"] == "intro"
        assert "Navigate to a claims representative." not in _message_blob(node)


# ── Greeting state ───────────────────────────────────────────────────────


class TestGreetingState:
    @pytest.mark.asyncio
    async def test_speak_first_state_skips_greet_step_after_navigate(self):
        greeting_state = {GREETING_STATE_KEY: True}
        node = _default_chain(greeting_state=greeting_state)

        _result, next_node = await _advance_fn(node).handler({}, _fm())

        assert next_node["name"] == CONFIRM_DENIAL
        task_blob = _message_blob(next_node)
        assert GREETING_ALREADY_DONE_NOTE in task_blob
        assert "Introduce yourself" not in task_blob

    @pytest.mark.asyncio
    async def test_user_first_state_enters_greet_once_then_marks_greeted(self):
        greeting_state = {GREETING_STATE_KEY: False}
        fm = _fm()
        node = _default_chain(greeting_state=greeting_state)

        _result, greet_node = await _advance_fn(node).handler({}, fm)

        assert greet_node["name"] == GREET
        assert GREETING_ALREADY_DONE_NOTE not in _message_blob(greet_node)

        _result, next_node = await _advance_fn(greet_node).handler({}, fm)

        assert next_node["name"] == CONFIRM_DENIAL
        assert greeting_state[GREETING_STATE_KEY] is True
        assert fm.state[GREETING_STATE_KEY] is True
        assert GREETING_ALREADY_DONE_NOTE in _message_blob(next_node)

    @pytest.mark.asyncio
    async def test_no_regreeting_note_carries_across_later_default_steps(self):
        greeting_state = {GREETING_STATE_KEY: True}
        node = _default_chain(greeting_state=greeting_state)
        fm = _fm()
        visited: list[str] = []

        while True:
            visited.append(node["name"])
            if node["name"] != NAVIGATE:
                task_blob = _message_blob(node)
                assert GREETING_ALREADY_DONE_NOTE in task_blob
                assert "Introduce yourself" not in task_blob
            if not node["functions"]:
                break
            args = {"reference_number": "R-1"} if node["name"] == REFERENCE_NUMBER else {}
            _result, node = await _advance_fn(node).handler(args, fm)

        assert GREET not in visited
        assert visited == [
            NAVIGATE,
            CONFIRM_DENIAL,
            ASK_NEEDS,
            FAX_PORTAL,
            DEADLINE,
            REFERENCE_NUMBER,
            WRAP,
        ]

    @pytest.mark.asyncio
    async def test_default_chain_without_greeting_state_preserves_existing_order(self):
        _result, next_node = await _advance_fn(_default_chain()).handler({}, _fm())

        assert next_node["name"] == GREET


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
        assert node["name"] == REFERENCE_NUMBER
        _result, node = await _advance_fn(node).handler({CALL_REFERENCE_FIELD: "REF-1"}, fm)
        assert node["name"] == "done"

    @pytest.mark.asyncio
    async def test_non_claim_flow_without_reference_number_builds_and_reaches_end(self):
        node = _chain(flow_definition=_patient_balance_flow())
        fm = _fm()

        assert node["name"] == "confirm_balance"
        _result, node = await _advance_fn(node).handler({"balance_amount": "$42.17"}, fm)
        assert node["name"] == "payment_options"
        _result, node = await _advance_fn(node).handler({}, fm)
        assert node["name"] == "done"
        assert node["functions"] == []

    @pytest.mark.asyncio
    async def test_no_flow_definition_uses_neutral_fallback(self, mocker):
        # #21: a missing flow_definition runs the NEUTRAL chain (assist →
        # close) driven by the agent's own persona — never the claim 8-step.
        mock_logger = mocker.patch("app.flows.steps.logger")
        node = _chain(flow_definition=None, agent_id="agent-xyz")
        fm = _fm()
        names: list[str] = []
        while True:
            names.append(node["name"])
            if not node["functions"]:
                break
            _result, node = await _advance_fn(node).handler({}, fm)
            assert node is not None

        assert names == [NEUTRAL_ASSIST, NEUTRAL_CLOSE]
        # No claim step leaks into the neutral fallback.
        claim_names = {s.name for s in STEPS}
        assert not (set(names) & claim_names)
        assert NAVIGATE not in names
        assert REFERENCE_NUMBER not in names
        mock_logger.warning.assert_any_call(
            "flow_definition_missing_neutral_fallback",
            agent_id="agent-xyz",
            reason="missing",
        )

    def test_neutral_fallback_head_resets_and_loads_agent_persona(self):
        node = _chain(flow_definition=None, hydrated_system="AGENT-OWN-PERSONA")

        assert node["name"] == NEUTRAL_ASSIST
        assert node["context_strategy"].strategy == ContextStrategy.RESET
        assert "AGENT-OWN-PERSONA" in node["role_message"]
        assert node["role_message"].endswith(STEP_COMPLETION_ROLE_RULE)
        assert _advance_fn(node).name == NEUTRAL_ADVANCE_NAME

    def test_malformed_flow_definition_uses_neutral_fallback_and_logs_warning(self, mocker):
        mock_logger = mocker.patch("app.flows.steps.logger")
        flow = {
            "version": 1,
            "start": "intro",
            "nodes": [{"id": "intro", "type": "ask"}],
        }

        chain = _chain(flow_definition=flow, agent_id="agent-xyz")

        # Neutral fallback — NOT the claim chain.
        assert chain["name"] == NEUTRAL_ASSIST
        # The invalid-flow detail log still fires (why it was rejected) ...
        mock_logger.warning.assert_any_call(
            "flow_definition_invalid_default_flow",
            reason="validation_error",
            error_count=ANY,
            errors=ANY,
        )
        # ... alongside the action-taken log (neutral fallback + agent_id).
        mock_logger.warning.assert_any_call(
            "flow_definition_missing_neutral_fallback",
            agent_id="agent-xyz",
            reason="invalid",
        )

    @pytest.mark.asyncio
    async def test_neutral_close_node_is_terminal_and_bounds_context(self):
        node = _chain(flow_definition=None)

        _result, close = await _advance_fn(node).handler({}, _fm())

        assert close["name"] == NEUTRAL_CLOSE
        # Terminal: no advance (and no tools were registered) → empty functions.
        assert close["functions"] == []
        assert close["context_strategy"].strategy == ContextStrategy.RESET_WITH_SUMMARY

    def test_neutral_nodes_advertise_call_tools(self):
        # end_call (and every other registered tool) is advertised on the
        # neutral chain, so the agent can still end/transfer the call.
        node = _chain(tool_names=["end_call", "transfer_call"], flow_definition=None)

        assert "end_call" in {fn.name for fn in node["functions"]}
        assert NEUTRAL_ADVANCE_NAME in {fn.name for fn in node["functions"]}

    def test_no_ivr_ambiguous_ivr_start_uses_neutral_fallback(self, mocker):
        # An ``ivr: true`` start node with >1 outgoing target on a no-IVR call
        # cannot be unambiguously skipped — #21 routes it to the NEUTRAL chain
        # (previously: the claim 8-step).
        mock_logger = mocker.patch("app.flows.steps.logger")
        flow = {
            "version": 1,
            "start": "ivr_menu",
            "nodes": [
                {
                    "id": "ivr_menu",
                    "type": "branch",
                    "ivr": True,
                    "branches": [{"when": "claims line", "to": "intro"}],
                    "fallback": "other",
                },
                {"id": "intro", "type": "ask", "say": "Confirm denial.", "next": "done"},
                {"id": "other", "type": "ask", "say": "Confirm eligibility.", "next": "done"},
                {"id": "done", "type": "end"},
            ],
        }

        node = _chain(
            flow_definition=flow,
            include_ivr=False,
            greeting_state={GREETING_STATE_KEY: False},
        )

        assert node["name"] == NEUTRAL_ASSIST
        assert NAVIGATE not in node["name"]
        mock_logger.warning.assert_any_call(
            "flow_definition_no_ivr_start_ambiguous_neutral_fallback",
            node_id="ivr_menu",
            target_count=2,
        )

    @pytest.mark.asyncio
    async def test_seeded_claim_flow_definition_runs_full_order_unaffected(self):
        # A seeded Claims-style flow_definition still drives its full order;
        # #21 only changes the missing/invalid path, never a valid flow.
        node = _chain(flow_definition=_custom_flow_starting_at_navigate())
        fm = _fm()
        names: list[str] = []
        while True:
            names.append(node["name"])
            if not node["functions"]:
                break
            args = {CALL_REFERENCE_FIELD: "REF-1"} if node["name"] == REFERENCE_NUMBER else {}
            if node["name"] == "route_submission":
                args = {"branch_to": "fax_path"}
            _result, node = await _advance_fn(node).handler(args, fm)
            assert node is not None

        assert names[0] == NAVIGATE
        assert names[-1] == "done"
        assert NEUTRAL_ASSIST not in names

    @pytest.mark.asyncio
    async def test_required_call_reference_blocks_blank_capture(self):
        node = await _walk_to(_chain(flow_definition=_custom_flow()), REFERENCE_NUMBER)

        result, next_node = await _advance_fn(node).handler({CALL_REFERENCE_FIELD: " "}, _fm())

        assert next_node is None
        assert result["status"] == "missing"
        assert result["field"] == CALL_REFERENCE_FIELD

    @pytest.mark.asyncio
    async def test_call_reference_records_legacy_reference_state_key(self):
        node = await _walk_to(_chain(flow_definition=_custom_flow()), REFERENCE_NUMBER)
        fm = _fm()

        result, next_node = await _advance_fn(node).handler(
            {CALL_REFERENCE_FIELD: "REF-123"},
            fm,
        )

        assert next_node["name"] == "done"
        assert result[CALL_REFERENCE_FIELD] == "REF-123"
        assert fm.state[CALL_REFERENCE_FIELD] == "REF-123"
        assert fm.state[REFERENCE_NUMBER] == "REF-123"

    @pytest.mark.asyncio
    async def test_claim_flow_reference_node_keeps_legacy_advance_name(self):
        node = await _walk_to(_chain(flow_definition=_custom_flow()), REFERENCE_NUMBER)

        assert _advance_fn(node).name == "record_reference_number"

    @pytest.mark.asyncio
    async def test_required_capture_blocks_missing_or_blank_fields(self):
        node = _chain(flow_definition=_required_balance_flow())

        result, next_node = await _advance_fn(node).handler(
            {"balance_amount": "100.00", "due_date": "   "},
            _fm(),
        )
        assert next_node is None
        assert result == {"status": "missing", "field": "due_date"}

        result, next_node = await _advance_fn(node).handler(
            {"due_date": "2026-07-01"},
            _fm(),
        )
        assert next_node is None
        assert result == {"status": "missing", "field": "balance_amount"}

    @pytest.mark.asyncio
    async def test_required_capture_records_all_present_fields_and_advances(self):
        node = _chain(flow_definition=_required_balance_flow())
        fm = _fm()

        result, next_node = await _advance_fn(node).handler(
            {"balance_amount": "100.00", "due_date": "2026-07-01"},
            fm,
        )

        assert next_node["name"] == "done"
        assert result == {
            "status": "ok",
            "balance_amount": "100.00",
            "due_date": "2026-07-01",
        }
        assert fm.state["balance_amount"] == "100.00"
        assert fm.state["due_date"] == "2026-07-01"

    def test_required_capture_schema_marks_each_capture_field_required(self):
        node = _chain(flow_definition=_required_balance_flow())
        advance = _advance_fn(node)

        assert advance.required == ["balance_amount", "due_date"]
        assert "balance_amount" in advance.properties
        assert "due_date" in advance.properties

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

        assert "HYDRATED-PHI-PROMPT" in node["role_message"]
        assert STEP_COMPLETION_ROLE_RULE in node["role_message"]
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
    async def test_prefetch_flagged_node_keeps_cached_knowledge_hint(self):
        # An arbitrarily-named node opts into knowledge prefetch via the flag.
        flow = {
            "version": 1,
            "start": "appeal_window",
            "nodes": [
                {
                    "id": "appeal_window",
                    "type": "ask",
                    "prefetch": True,
                    "say": "Confirm the timely filing deadline.",
                    "next": REFERENCE_NUMBER,
                },
                {
                    "id": REFERENCE_NUMBER,
                    "type": "ask",
                    "capture": [CALL_REFERENCE_FIELD],
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

    @pytest.mark.asyncio
    async def test_unflagged_deadline_named_node_gets_no_knowledge_hint(self):
        # Behavior is flag-driven, not id-driven: a node named "deadline"
        # without ``prefetch: true`` must NOT receive the cached fact.
        flow = {
            "version": 1,
            "start": "deadline",
            "nodes": [
                {
                    "id": "deadline",
                    "type": "ask",
                    "say": "Confirm the timely filing deadline.",
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

        assert "Known payer-level fact" not in _navigate_task(chain)
        warmer.live_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_ivr_flagged_node_with_arbitrary_id_gets_path_injection(self):
        # IVR navigation can be placed on any node via ``ivr: true``.
        flow = {
            "version": 1,
            "start": "payer_menu_entry",
            "nodes": [
                {
                    "id": "payer_menu_entry",
                    "type": "ask",
                    "ivr": True,
                    "say": "Navigate to a claims representative.",
                    "next": "done",
                },
                {"id": "done", "type": "end"},
            ],
        }

        chain = build_step_chain(
            run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
            registry=_registry([]),
            hydrated_system="SYS",
            ivr_path="1. Claims — press 1",
            ivr_goal="Reach a rep",
            flow_definition=flow,
        )

        assert chain["name"] == "payer_menu_entry"
        task = _navigate_task(chain)
        assert NAVIGATE_BASE_TASK in task
        assert "1. Claims — press 1" in task
        assert "Reach a rep" in task

    @pytest.mark.asyncio
    async def test_unflagged_navigate_named_node_gets_no_path_injection(self):
        # A node named "navigate" without ``ivr: true`` uses its generic task
        # and never receives the verified IVR path.
        flow = {
            "version": 1,
            "start": NAVIGATE,
            "nodes": [
                {
                    "id": NAVIGATE,
                    "type": "ask",
                    "say": "Talk through the menu.",
                    "next": "done",
                },
                {"id": "done", "type": "end"},
            ],
        }

        chain = build_step_chain(
            run_tool_core=AsyncMock(return_value=({"status": "ok"}, False)),
            registry=_registry([]),
            hydrated_system="SYS",
            ivr_path="1. Claims — press 1",
            ivr_goal="Reach a rep",
            flow_definition=flow,
        )

        task = _navigate_task(chain)
        assert "Talk through the menu." in task
        assert "1. Claims — press 1" not in task
        assert NAVIGATE_BASE_TASK not in task

    @pytest.mark.asyncio
    async def test_claim_flow_flags_reproduce_default_ivr_and_prefetch_behavior(self):
        # The claim flow as data reproduces today's behavior via the flags,
        # regardless of node ids.
        flow = {
            "version": 1,
            "start": "menu",
            "nodes": [
                {
                    "id": "menu",
                    "type": "ask",
                    "ivr": True,
                    "say": "Navigate the payer IVR.",
                    "next": "appeal_window",
                },
                {
                    "id": "appeal_window",
                    "type": "ask",
                    "prefetch": True,
                    "say": "Confirm the timely filing deadline.",
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
            ivr_path="1. Claims — press 1",
            ivr_goal="Reach a rep",
            knowledge_warmer=warmer,
            knowledge_context=PrefetchContext(payer="Aetna"),
            flow_definition=flow,
        )

        # IVR node carries the verified path/goal.
        assert chain["name"] == "menu"
        assert "1. Claims — press 1" in _navigate_task(chain)
        assert "Reach a rep" in _navigate_task(chain)
        # Prefetch node carries the cached payer fact.
        appeal = await _walk_to(chain, "appeal_window")
        assert "Known payer-level fact: Aetna timely filing is 120 days." in _navigate_task(appeal)


# ── Un-skippable reference number ─────────────────────────────────────────


class TestReferenceNumberRequired:
    @pytest.mark.asyncio
    async def test_blank_reference_number_blocks_advance(self):
        node = await _walk_to(_default_chain(), REFERENCE_NUMBER)
        result, next_node = await _advance_fn(node).handler({"reference_number": "   "}, _fm())
        assert next_node is None  # stays on the node — un-skippable
        assert result["status"] == "missing"
        assert result["field"] == "reference_number"

    @pytest.mark.asyncio
    async def test_missing_reference_number_blocks_advance(self):
        node = await _walk_to(_default_chain(), REFERENCE_NUMBER)
        result, next_node = await _advance_fn(node).handler({}, _fm())
        assert next_node is None
        assert result["status"] == "missing"

    @pytest.mark.asyncio
    async def test_present_reference_number_advances_to_wrap_and_records_state(self):
        node = await _walk_to(_default_chain(), REFERENCE_NUMBER)
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
        node = _default_chain()
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
        node = await _walk_to(_default_chain(), REFERENCE_NUMBER)
        advance = _advance_fn(node)
        assert advance.required == ["reference_number"]
        assert "reference_number" in advance.properties


# ── Per-step bounded context ──────────────────────────────────────────────


class TestPerStepContext:
    def test_first_step_resets_and_loads_hydrated_prompt(self):
        chain = _chain(hydrated_system="HYDRATED-PHI-PROMPT")
        assert "HYDRATED-PHI-PROMPT" in chain["role_message"]
        assert STEP_COMPLETION_ROLE_RULE in chain["role_message"]
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


class TestInternalDirectiveLeakage:
    @pytest.mark.asyncio
    async def test_default_step_task_messages_do_not_name_internal_functions(self):
        forbidden = {
            "representative_reached",
            "greeting_done",
            "denial_reason_confirmed",
            "needs_identified",
            "submission_method_confirmed",
            "deadline_confirmed",
            "record_reference_number",
            "end_call",
            "transfer_call",
            "Conversation summary so far",
            "Here's a summary of the conversation",
        }
        node = _default_chain()
        fm = _fm()

        while True:
            task_blob = " ".join(message["content"] for message in node["task_messages"])
            for internal_name in forbidden:
                assert internal_name not in task_blob
            fns = node["functions"]
            if not fns:
                break
            args = {"reference_number": "R-1"} if node["name"] == REFERENCE_NUMBER else {}
            _result, node = await _advance_fn(node).handler(args, fm)

    def test_step_role_message_preserves_prompt_and_uses_generic_rule(self):
        chain = _default_chain(hydrated_system="HYDRATED")
        role_message = chain["role_message"]

        assert "HYDRATED" in role_message
        assert STEP_COMPLETION_ROLE_RULE in role_message
        for step in STEPS:
            if step.advance_name:
                assert step.advance_name not in role_message

    @pytest.mark.asyncio
    async def test_data_driven_task_messages_do_not_name_generated_advance_functions(self):
        forbidden = {
            "advance_intro",
            "advance_route_submission",
            "advance_fax_path",
            "advance_portal_path",
            "record_reference_number",
            "branch_to",
            "fax_path",
            "portal_path",
            "Conversation summary so far",
            "Here's a summary of the conversation",
        }
        node = _chain(flow_definition=_custom_flow())
        fm = _fm()

        while True:
            task_blob = " ".join(message["content"] for message in node["task_messages"])
            for internal_name in forbidden:
                assert internal_name not in task_blob
            fns = node["functions"]
            if not fns:
                break
            if node["name"] == "route_submission":
                args = {"branch_to": "fax_path"}
            elif node["name"] == REFERENCE_NUMBER:
                args = {CALL_REFERENCE_FIELD: "R-1"}
            else:
                args = {}
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

    @pytest.mark.asyncio
    async def test_terminal_step_tool_marks_call_ended_without_transition(self):
        core = AsyncMock(return_value=({"call_ended": True, "reason": "done"}, False))
        chain = build_step_chain(
            run_tool_core=core,
            registry=_registry(["end_call"]),
            hydrated_system="X",
        )
        fm = _fm()
        tool_fn = next(f for f in chain["functions"] if f.name == "end_call")

        payload, next_node = await tool_fn.handler({"reason": "done"}, fm)

        core.assert_awaited_once_with("end_call", {"reason": "done"})
        assert payload == {"call_ended": True, "reason": "done"}
        assert fm.state["call_ended"] is True
        assert next_node is None

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
        # And the claim chain's first node reflects that verbatim.
        assert _navigate_task(_default_chain()) == NAVIGATE_BASE_TASK

    def test_includes_verified_path_when_present(self):
        path = "1. Provider services — press 3\n2. Claims — press 1"
        task = build_navigate_task(ivr_path=path)
        assert NAVIGATE_BASE_TASK in task
        assert path in task

    def test_path_instructs_listen_and_decide_fallback(self):
        task = build_navigate_task(ivr_path="1. press 3")
        assert "navigate by ear" in task.lower()

    def test_navigate_task_instructs_wait_for_new_prompt_and_fallback(self):
        task = build_navigate_task()
        lower = task.lower()
        assert "wait for the next ivr prompt" in lower
        assert "never repeat the same digit" in lower
        assert "press 0 once" in lower
        assert "representative" in lower

    def test_navigate_task_affirms_keypad_capability_and_bans_inability_claims(self):
        lower = build_navigate_task().lower()
        assert "you can and should press keypad digits via the keypad tool" in lower
        assert "never tell the caller you cannot use the keypad" in lower
        assert "cannot process keypad menu options" in lower

    def test_includes_ivr_goal_when_present(self):
        task = build_navigate_task(ivr_goal="Reach a claims rep")
        assert NAVIGATE_BASE_TASK in task
        assert "Reach a claims rep" in task

    def test_goal_and_path_both_present(self):
        task = build_navigate_task(ivr_path="1. press 3", ivr_goal="Reach a rep")
        assert "Reach a rep" in task
        assert "1. press 3" in task

    @pytest.mark.asyncio
    async def test_default_chain_threads_path_into_navigate_node_only(self):
        chain = _default_chain(ivr_path="1. Claims — press 1", ivr_goal="Reach a rep")
        assert chain["name"] == NAVIGATE
        nav_task = _navigate_task(chain)
        assert "1. Claims — press 1" in nav_task
        assert "Reach a rep" in nav_task
        # Later steps keep their fixed task verbatim (path not leaked).
        _result, greet_node = await _advance_fn(chain).handler({}, _fm())
        assert "press 1" not in _navigate_task(greet_node)

    @pytest.mark.asyncio
    async def test_verified_path_mentions_wait_instructions_when_present(self):
        chain = _default_chain(ivr_path="1. Claims — press 1; wait 2.0s for the next IVR prompt")

        assert "wait 2.0s for the next IVR prompt" in _navigate_task(chain)
        assert "honor any wait instructions" in _navigate_task(chain)
        _result, greet_node = await _advance_fn(chain).handler({}, _fm())
        assert "wait 2.0s" not in _navigate_task(greet_node)


class TestKnowledgeTaskHints:
    @pytest.mark.asyncio
    async def test_build_step_chain_without_knowledge_keeps_deadline_task_verbatim(self):
        chain = _default_chain()
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
        chain = _default_chain(
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
        chain = _default_chain(
            knowledge_warmer=warmer,
            knowledge_context=PrefetchContext(payer="Aetna"),
        )

        deadline = await _walk_to(chain, DEADLINE)

        expected = next(step.task for step in STEPS if step.name == DEADLINE)
        assert _navigate_task(deadline) == expected

    @pytest.mark.asyncio
    async def test_knowledge_live_read_not_called_when_no_payer_context(self):
        warmer = MagicMock()
        chain = _default_chain(
            knowledge_warmer=warmer,
            knowledge_context=PrefetchContext(payer=None),
        )

        await _walk_to(chain, DEADLINE)

        warmer.live_read.assert_not_called()
