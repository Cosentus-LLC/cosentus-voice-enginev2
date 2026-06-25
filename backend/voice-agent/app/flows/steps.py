"""Pipecat Flows ordered call steps + per-step context (16c, #43).

The remaining, *post-verification* conversation as a fixed chain of
code-enforced Flows steps, built on the 16a scaffold
(:mod:`app.flows.scaffold`) and the 16b identity gate
(:mod:`app.flows.identity_gate`). Reaching here means
:func:`verify_against_case_data` already passed — so PHI is allowed
from this point on (the gate keeps everything PHI-free *before* it; see
:data:`app.flows.identity_gate.PRE_VERIFICATION_ROLE_MESSAGE`).

Code owns the workflow (two guarantees)
---------------------------------------
* **Ordered + un-skippable.** The steps are a *linear chain*:
  ``navigate → greet → confirm-denial → ask-needs → fax/portal →
  deadline → reference-number → wrap``. Each non-terminal node
  advertises exactly **one** advance function whose handler returns the
  *single* next node — Pipecat Flows narrows the LLM's tool schema per
  node, so the model literally cannot jump ahead. To reach ``wrap`` the
  call must pass through ``reference-number``, and that step's advance
  handler **refuses to transition** (returns ``next_node=None``) until a
  non-blank reference number is supplied. That refusal is a
  deterministic code check, not a prompt instruction.
* **Bounded per-step context.** Every step resets context so only the
  current step's task + a carried-forward summary stay in the LLM
  window — per-turn input stays flat on a 20-30 min call instead of
  accumulating the whole transcript. The *first* post-verification step
  (``navigate``) uses :attr:`ContextStrategy.RESET` to drop the
  identity-gate chatter cleanly and load the hydrated system prompt
  (``role_message``); every later step uses
  :attr:`ContextStrategy.RESET_WITH_SUMMARY` so the gist of the prior
  steps rides forward as a one-paragraph running summary.

  (Intra-step trimming for a single very long step — a sliding window
  over one node's turns — is **#22's** ``ContextWindow``, complementary
  to this per-node bounding. See the #43/#22 coordination note.)

The hydrated (PHI-bearing) system prompt is set **once**, as
``role_message`` on the ``navigate`` node; Flows persists a
``role_message`` across transitions until a node sets a new one, so the
later steps inherit it without re-sending it each turn.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from pipecat_flows import (
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
)
from pipecat_flows.types import ConsolidatedFunctionResult, NodeConfig

from app.knowledge.prefetch import PrefetchContext, PrefetchWarmer

logger = structlog.get_logger(__name__)

# Node identifiers — constants so bot.py wiring and tests reference them
# without restating string literals.
NAVIGATE = "navigate"
GREET = "greet"
CONFIRM_DENIAL = "confirm_denial_reason"
ASK_NEEDS = "ask_needs"
FAX_PORTAL = "fax_or_portal"
DEADLINE = "deadline"
REFERENCE_NUMBER = "reference_number"
WRAP = "wrap"

REQUIRED_REFERENCE_NODE_ID = REFERENCE_NUMBER
REQUIRED_REFERENCE_FIELD = "call_reference"
FLOW_NODE_TYPES = frozenset({"ask", "branch", "transfer", "end"})


def identity_gate_required_for_direction(direction: str) -> bool:
    """Return whether this call direction should start at the identity gate."""
    return direction == "inbound"


# The running-summary prompt used by RESET_WITH_SUMMARY on every step
# after the first. Post-verification, so it may reference claim/case
# details — the gate already barred all PHI before this point.
SUMMARY_PROMPT = (
    "You are summarizing a live medical-billing phone call for the assistant's "
    "own working memory, so it can continue the call without the full "
    "transcript. In 2-4 factual sentences capture: the claim and denial "
    "details discussed, what the representative said is required to resolve or "
    "appeal it, the confirmed fax number or payer portal and the submission "
    "deadline, and any reference or confirmation number obtained. Do not "
    "invent details that were not stated."
)


@dataclass(frozen=True)
class _Step:
    """One node in the ordered chain.

    ``advance_name`` is the (chain-unique) name of the single function
    this node advertises to move to the next step. ``required_field``,
    when set, makes that field a required string argument the handler
    must receive non-blank before it will transition — the un-skippable
    guarantee (used by ``reference_number``).
    """

    name: str
    task: str
    advance_name: str = ""
    advance_description: str = ""
    required_field: str | None = None
    required_description: str = ""


@dataclass(frozen=True)
class _FlowBranch:
    when: str
    to: str


@dataclass(frozen=True)
class _FlowNode:
    id: str
    type: str
    label: str
    say: str
    capture: tuple[str, ...]
    required: bool
    next: str | None
    branches: tuple[_FlowBranch, ...]
    fallback: str | None


@dataclass(frozen=True)
class _FlowDefinition:
    version: int
    start: str
    nodes: tuple[_FlowNode, ...]


# The ``navigate`` step's base task — used verbatim when no verified IVR
# path / goal is supplied, so the no-map path is byte-identical to the
# pre-#17 prompt. ``build_navigate_task`` augments it when a path/goal
# is present.
NAVIGATE_BASE_TASK = (
    "You have dialed the payer and reached their phone system. Navigate "
    "the IVR menus to reach a live claims representative — use the keypad "
    "tool to select menu options as needed, and stay on the line through "
    "any hold. Once you are speaking with a live representative, call "
    "representative_reached."
)


def build_navigate_task(ivr_path: str = "", ivr_goal: str = "") -> str:
    """Build the ``navigate`` step task, optionally guided by a map (#17).

    With neither argument this returns :data:`NAVIGATE_BASE_TASK`
    **verbatim** (today's listen-and-decide behavior). When present:

    * ``ivr_goal`` (per-agent, from ``AgentConfig.ivr_goal``) is stated
      as the navigation goal for this call.
    * ``ivr_path`` (the verified per-payer claims path, already rendered
      to numbered lines by
      :meth:`app.config.payer_knowledge.PayerIvrPath.as_navigation_text`)
      is appended with an instruction to follow it via the keypad tool —
      pressing the indicated digits and keying in the requested values —
      and an explicit instruction to **fall back to navigating by ear**
      when the live menu does not match the map. The ``press_digit``
      mechanism is unchanged; this only guides how the LLM calls it.
    """
    parts = [NAVIGATE_BASE_TASK]
    goal = ivr_goal.strip()
    if goal:
        parts.append(f"Your goal for this call: {goal}")
    path = ivr_path.strip()
    if path:
        parts.append(
            "A verified menu path for this payer is known. Follow it using the "
            "keypad tool: press the indicated digits, and for steps that ask you "
            "to key in a value (NPI, claim number, tax ID) enter that value from "
            "the call details. If the live menu does not match this path, ignore "
            "it and navigate by ear instead:\n" + path
        )
    return "\n\n".join(parts)


# The fixed denial-resolution workflow. Order here *is* the enforced
# order — index N's advance handler returns index N+1's node, nothing
# else. The terminal step (``wrap``) has no advance function.
STEPS: tuple[_Step, ...] = (
    _Step(
        name=NAVIGATE,
        task=NAVIGATE_BASE_TASK,
        advance_name="representative_reached",
        advance_description="Call once a live representative is on the line.",
    ),
    _Step(
        name=GREET,
        task=(
            "Greet the representative. Introduce yourself as calling on behalf of "
            "the provider about a denied claim, and say you'd like to resolve it. "
            "When you have introduced the call, call greeting_done."
        ),
        advance_name="greeting_done",
        advance_description="Call once you have greeted and introduced the call.",
    ),
    _Step(
        name=CONFIRM_DENIAL,
        task=(
            "Confirm the denial reason with the representative. Ask why the claim "
            "was denied and restate it back to confirm you understood it "
            "correctly. When the denial reason is confirmed, call "
            "denial_reason_confirmed."
        ),
        advance_name="denial_reason_confirmed",
        advance_description="Call once the denial reason is confirmed.",
    ),
    _Step(
        name=ASK_NEEDS,
        task=(
            "Ask the representative exactly what is needed to resolve or appeal "
            "the denial — for example corrected codes, medical records, an appeal "
            "form, or other documentation. When you understand what is required, "
            "call needs_identified."
        ),
        advance_name="needs_identified",
        advance_description="Call once you understand what is required to resolve the denial.",
    ),
    _Step(
        name=FAX_PORTAL,
        task=(
            "Determine how to submit the required information: confirm the correct "
            "fax number or the payer portal to use for the submission. When you "
            "have the submission method, call submission_method_confirmed."
        ),
        advance_name="submission_method_confirmed",
        advance_description="Call once the fax number or portal for submission is confirmed.",
    ),
    _Step(
        name=DEADLINE,
        task=(
            "Confirm the deadline by which the information must be submitted (the "
            "timely-filing or appeal window). When you have the deadline, call "
            "deadline_confirmed."
        ),
        advance_name="deadline_confirmed",
        advance_description="Call once the submission deadline is confirmed.",
    ),
    _Step(
        name=REFERENCE_NUMBER,
        task=(
            "Before ending, ask the representative for a call reference or "
            "confirmation number for this interaction. You must capture it. Once "
            "the representative gives it to you, call record_reference_number with "
            "the number exactly as stated."
        ),
        advance_name="record_reference_number",
        advance_description=(
            "Record the call reference / confirmation number. Call ONLY after the "
            "representative has provided it."
        ),
        required_field="reference_number",
        required_description="The reference or confirmation number the representative provided.",
    ),
    _Step(
        name=WRAP,
        task=(
            "Wrap up the call: thank the representative, briefly confirm the agreed "
            "next steps, the submission method and deadline, and the reference "
            "number you recorded, then politely close. Use the end_call tool when "
            "finished."
        ),
    ),
)

# ── Builder ──────────────────────────────────────────────────────────────

# The shared tool-execution core (``run_bot.run_tool_to_payload``):
# ``(tool_name, arguments) -> (payload, run_llm)``.
RunToolCore = Callable[[str, dict[str, Any]], "Any"]


def _advance_function(step: _Step, next_builder: Callable[[], NodeConfig]) -> FlowsFunctionSchema:
    """Build the single advance function for ``step``.

    The handler is a Flows consolidated function returning
    ``(result, next_node)``. When ``step.required_field`` is set, a blank
    value yields ``(_, None)`` — the flow stays on the current node so
    the LLM re-asks — making that step un-skippable. On success it
    records the captured value into ``flow_manager.state`` (PHI-safe: the
    value is state, never logged here) and returns the next node.
    """

    async def handler(args: FlowArgs, flow_manager: FlowManager) -> ConsolidatedFunctionResult:
        if step.required_field is not None:
            value = str(args.get(step.required_field) or "").strip()
            if not value:
                # Deterministic refusal — not a prompt instruction.
                return {"status": "missing", "field": step.required_field}, None
            flow_manager.state[step.required_field] = value
            return {"status": "ok", step.required_field: value}, next_builder()
        return {"status": "ok"}, next_builder()

    properties: dict[str, Any] = {}
    required: list[str] = []
    if step.required_field is not None:
        properties[step.required_field] = {
            "type": "string",
            "description": step.required_description,
        }
        required.append(step.required_field)

    return FlowsFunctionSchema(
        name=step.advance_name,
        description=step.advance_description,
        properties=properties,
        required=required,
        handler=handler,
    )


def _tool_functions(registry: Any, run_tool_core: RunToolCore) -> list[FlowsFunctionSchema]:
    """Wrap the call's real tools as Flows functions for a node.

    Mirrors the per-tool wrapping the 16b verified-node placeholder used:
    each tool delegates to the shared :func:`run_tool_to_payload` core so
    there is one execution + transcript-capture path. Advertised at every
    step so the bot can act (transfer, press digits, end the call)
    throughout — step progression is enforced by the advance chain, not
    by hiding tools.
    """
    tools: list[FlowsFunctionSchema] = []
    for name in registry.names():
        spec = registry.get(name)
        assert spec is not None  # iterating names() — must exist
        schema = spec.to_function_schema()

        def make_flow_tool_handler(tool_name: str):
            async def flow_tool_handler(args: FlowArgs, _flow_manager: FlowManager):
                payload, _run_llm = await run_tool_core(tool_name, dict(args))
                return payload, None

            return flow_tool_handler

        tools.append(
            FlowsFunctionSchema(
                name=schema.name,
                description=schema.description,
                properties=schema.properties,
                required=schema.required,
                handler=make_flow_tool_handler(name),
                cancel_on_interruption=spec.cancel_on_interruption,
                timeout_secs=spec.timeout_secs,
            )
        )
    return tools


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _string_or_none(value: Any) -> str | None:
    text = value.strip() if isinstance(value, str) else ""
    return text or None


def _normalize_flow_definition(raw: Any) -> _FlowDefinition | None:
    """Best-effort parse of the API's ``flow_definition`` contract.

    Returns ``None`` for non-objects or obviously malformed definitions so the
    caller can safely choose the default flow.
    """
    if not isinstance(raw, dict):
        return None
    version = raw.get("version")
    start = _string_or_none(raw.get("start"))
    raw_nodes = raw.get("nodes")
    if version != 1 or start is None or not isinstance(raw_nodes, list) or not raw_nodes:
        return None

    nodes: list[_FlowNode] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            return None
        raw_capture = raw_node.get("capture")
        raw_branches = raw_node.get("branches")
        capture = (
            tuple(field for field in raw_capture if isinstance(field, str))
            if isinstance(raw_capture, list)
            else ()
        )
        branches: tuple[_FlowBranch, ...] = ()
        if isinstance(raw_branches, list):
            parsed_branches: list[_FlowBranch] = []
            for raw_branch in raw_branches:
                if not isinstance(raw_branch, dict):
                    return None
                parsed_branches.append(
                    _FlowBranch(
                        when=_string_or_empty(raw_branch.get("when")),
                        to=_string_or_empty(raw_branch.get("to")),
                    )
                )
            branches = tuple(parsed_branches)
        nodes.append(
            _FlowNode(
                id=_string_or_empty(raw_node.get("id")),
                type=_string_or_empty(raw_node.get("type")),
                label=_string_or_empty(raw_node.get("label")),
                say=_string_or_empty(raw_node.get("say")),
                capture=capture,
                required=raw_node.get("required") is True,
                next=_string_or_none(raw_node.get("next")),
                branches=branches,
                fallback=_string_or_none(raw_node.get("fallback")),
            )
        )
    return _FlowDefinition(version=1, start=start, nodes=tuple(nodes))


def _definition_node_map(definition: _FlowDefinition) -> dict[str, _FlowNode]:
    nodes: dict[str, _FlowNode] = {}
    for node in definition.nodes:
        if node.id and node.id not in nodes:
            nodes[node.id] = node
    return nodes


def _outgoing_targets(node: _FlowNode) -> list[str]:
    targets: list[str] = []
    if node.next:
        targets.append(node.next)
    if node.fallback:
        targets.append(node.fallback)
    targets.extend(branch.to for branch in node.branches if branch.to)
    return targets


def _has_cycle(definition: _FlowDefinition, nodes_by_id: dict[str, _FlowNode]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        node = nodes_by_id.get(node_id)
        if node is None:
            return False
        visiting.add(node_id)
        for target in _outgoing_targets(node):
            if visit(target):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return visit(definition.start)


def _can_end_without_reference(
    definition: _FlowDefinition,
    nodes_by_id: dict[str, _FlowNode],
) -> bool:
    """Return True when any path reaches an end node before reference capture."""

    def visit(node_id: str, seen_reference: bool, path: frozenset[str]) -> bool:
        node = nodes_by_id.get(node_id)
        if node is None:
            return False
        next_seen_reference = seen_reference or node.id == REQUIRED_REFERENCE_NODE_ID
        if node.type == "end":
            return not next_seen_reference
        if node_id in path:
            return False
        next_path = path | {node_id}
        return any(
            visit(target, next_seen_reference, next_path) for target in _outgoing_targets(node)
        )

    return visit(definition.start, False, frozenset())


def _flow_definition_errors(definition: _FlowDefinition) -> list[str]:
    errors: list[str] = []
    nodes_by_id: dict[str, _FlowNode] = {}
    duplicate_ids: set[str] = set()
    for index, node in enumerate(definition.nodes):
        if not node.id:
            errors.append(f"nodes[{index}].id is required")
        elif node.id in nodes_by_id:
            duplicate_ids.add(node.id)
            errors.append(f"nodes[{index}].id duplicate node id {node.id!r}")
        else:
            nodes_by_id[node.id] = node
        if node.type not in FLOW_NODE_TYPES:
            errors.append(f"nodes[{index}].type must be one of {sorted(FLOW_NODE_TYPES)}")

    if definition.start not in nodes_by_id:
        errors.append(f"start references missing node {definition.start!r}")

    reference_node = nodes_by_id.get(REQUIRED_REFERENCE_NODE_ID)
    if reference_node is None:
        errors.append(f"nodes.{REQUIRED_REFERENCE_NODE_ID} is required")
    else:
        if reference_node.type != "ask":
            errors.append(f"nodes.{REQUIRED_REFERENCE_NODE_ID}.type must be ask")
        if reference_node.required is not True:
            errors.append(f"nodes.{REQUIRED_REFERENCE_NODE_ID}.required must be true")
        if REQUIRED_REFERENCE_FIELD not in reference_node.capture:
            errors.append(
                f"nodes.{REQUIRED_REFERENCE_NODE_ID}.capture must include "
                f"{REQUIRED_REFERENCE_FIELD}"
            )

    reachable: set[str] = set()

    def mark_reachable(node_id: str) -> None:
        if node_id in reachable:
            return
        node = nodes_by_id.get(node_id)
        if node is None:
            return
        reachable.add(node_id)
        for target in _outgoing_targets(node):
            mark_reachable(target)

    mark_reachable(definition.start)

    for node in definition.nodes:
        if not node.id or node.id in duplicate_ids:
            continue
        node_path = f"nodes.{node.id}"
        outgoing = _outgoing_targets(node)
        for target in outgoing:
            if target not in nodes_by_id:
                errors.append(f"{node_path} references missing node {target!r}")

        for branch_index, branch in enumerate(node.branches):
            if not branch.when.strip():
                errors.append(f"{node_path}.branches[{branch_index}].when is required")
            if not branch.to.strip():
                errors.append(f"{node_path}.branches[{branch_index}].to is required")

        if node.type == "end":
            if outgoing:
                errors.append(f"{node_path} end nodes cannot have outgoing transitions")
            continue

        if node.type == "branch":
            if not node.fallback:
                errors.append(f"{node_path}.fallback is required for branch nodes")
            if not node.branches:
                errors.append(f"{node_path}.branches must contain at least one branch")

        if not outgoing:
            errors.append(f"{node_path} non-end nodes must have an outgoing transition")

    for node in definition.nodes:
        if node.id and node.id in nodes_by_id and node.id not in reachable:
            errors.append(f"nodes.{node.id} is unreachable from start")

    if definition.start in nodes_by_id and _has_cycle(definition, nodes_by_id):
        errors.append("flow graph cannot contain cycles")
    if definition.start in nodes_by_id and _can_end_without_reference(definition, nodes_by_id):
        errors.append("every terminal path must pass through reference_number")

    return errors


def _usable_flow_definition(raw: Any) -> _FlowDefinition | None:
    if raw is None:
        return None
    definition = _normalize_flow_definition(raw)
    if definition is None:
        logger.warning(
            "flow_definition_invalid_default_flow",
            reason="malformed",
            error_count=1,
            errors=["flow_definition must match version/start/nodes shape"],
        )
        return None
    errors = _flow_definition_errors(definition)
    if errors:
        logger.warning(
            "flow_definition_invalid_default_flow",
            reason="validation_error",
            error_count=len(errors),
            errors=errors[:10],
        )
        return None
    return definition


_FUNCTION_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _advance_name_for_flow_node(node: _FlowNode) -> str:
    if node.id == REQUIRED_REFERENCE_NODE_ID:
        return "record_reference_number"
    suffix = _FUNCTION_NAME_RE.sub("_", node.id.strip()).strip("_").lower()
    return f"advance_{suffix or 'flow_step'}"


def _task_for_flow_node(*, node: _FlowNode, ivr_path: str, ivr_goal: str) -> str:
    if node.id == NAVIGATE:
        parts = [build_navigate_task(ivr_path=ivr_path, ivr_goal=ivr_goal)]
    else:
        label = node.label.strip() or node.id.replace("_", " ")
        prompt = node.say.strip() or f"Proceed with the {label} step."
        parts = [prompt]

    if node.capture:
        fields = ", ".join(field.replace("_", " ") for field in node.capture)
        parts.append(f"Capture these field(s) during this step when available: {fields}.")
    if node.id == REQUIRED_REFERENCE_NODE_ID:
        parts.append(
            "You must ask for and capture the call reference or confirmation number. "
            f"Once provided, call {_advance_name_for_flow_node(node)} with "
            f"{REQUIRED_REFERENCE_FIELD} exactly as stated."
        )
    elif node.type == "branch":
        branch_lines = [
            f"- {branch.when}: {branch.to}" for branch in node.branches if branch.when and branch.to
        ]
        fallback = node.fallback or ""
        parts.append(
            "Choose the next step by calling the advance function with branch_to. "
            "Use the fallback if none of the branch conditions match."
        )
        if branch_lines:
            parts.append("Branch options:\n" + "\n".join(branch_lines))
        if fallback:
            parts.append(f"Fallback branch_to: {fallback}")
    elif node.type == "transfer":
        parts.append("Use the transfer_call tool if the call should be transferred at this step.")
    elif node.type != "end":
        parts.append(f"When this step is complete, call {_advance_name_for_flow_node(node)}.")
    else:
        parts.append("Use the end_call tool when the call is finished.")
    return "\n\n".join(parts)


def _advance_function_for_flow_node(
    node: _FlowNode,
    build_next: Callable[[str], NodeConfig | None],
) -> FlowsFunctionSchema:
    async def handler(args: FlowArgs, flow_manager: FlowManager) -> ConsolidatedFunctionResult:
        captured: dict[str, str] = {}
        for field in node.capture:
            value = str(args.get(field) or "").strip()
            if (
                node.id == REQUIRED_REFERENCE_NODE_ID
                and field == REQUIRED_REFERENCE_FIELD
                and not value
            ):
                return {"status": "missing", "field": REQUIRED_REFERENCE_FIELD}, None
            if value:
                flow_manager.state[field] = value
                captured[field] = value
                if field == REQUIRED_REFERENCE_FIELD:
                    flow_manager.state[REFERENCE_NUMBER] = value

        target = node.next
        if node.type == "branch":
            allowed = {branch.to for branch in node.branches}
            if node.fallback:
                allowed.add(node.fallback)
            requested = str(args.get("branch_to") or "").strip()
            target = requested if requested in allowed else node.fallback
        elif target is None:
            target = node.fallback

        next_node = build_next(target) if target else None
        if target and next_node is None:
            logger.warning(
                "flow_definition_transition_missing_default_flow",
                node_id=node.id,
                target=target,
            )
        return {"status": "ok", **captured}, next_node

    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in node.capture:
        properties[field] = {
            "type": "string",
            "description": f"The {field.replace('_', ' ')} captured during this step.",
        }
    if node.id == REQUIRED_REFERENCE_NODE_ID:
        required.append(REQUIRED_REFERENCE_FIELD)
        properties[REQUIRED_REFERENCE_FIELD] = {
            "type": "string",
            "description": "The call reference or confirmation number the representative provided.",
        }
    if node.type == "branch":
        options = sorted(
            {branch.to for branch in node.branches if branch.to}
            | ({node.fallback} if node.fallback else set())
        )
        properties["branch_to"] = {
            "type": "string",
            "enum": options,
            "description": "The next node id selected from the branch conditions, or the fallback.",
        }

    return FlowsFunctionSchema(
        name=_advance_name_for_flow_node(node),
        description=f"Advance from the {node.label or node.id} flow step.",
        properties=properties,
        required=required,
        handler=handler,
    )


def _task_with_knowledge(
    *,
    step: _Step,
    task: str,
    knowledge_warmer: PrefetchWarmer | None,
    knowledge_context: PrefetchContext | None,
) -> str:
    """Append a cached payer-level fact when one is already warm.

    This is cache-only. A miss leaves the task text unchanged and the warmer
    fills in the background for a later turn.
    """
    if knowledge_warmer is None or knowledge_context is None:
        return task
    payer = (knowledge_context.payer or "").strip()
    if not payer or step.name != DEADLINE:
        return task

    query = f"timely filing limit for {payer}"
    try:
        hit = knowledge_warmer.live_read(query)
    except RuntimeError:
        # Unit tests can build nodes outside a running loop. Production has one,
        # but a missed background fill should never break node construction.
        return task
    if hit is None:
        return task
    return task + "\n\nKnown payer-level fact: " + hit.value


def _build_default_step_chain(
    *,
    run_tool_core: RunToolCore,
    registry: Any,
    hydrated_system: str,
    summary_prompt: str = SUMMARY_PROMPT,
    ivr_path: str = "",
    ivr_goal: str = "",
    knowledge_warmer: PrefetchWarmer | None = None,
    knowledge_context: PrefetchContext | None = None,
) -> NodeConfig:
    """Build the default post-verification step chain; return its first node.

    Passed to :func:`app.flows.identity_gate.build_identity_gate_flow` as
    ``verified_node`` — so the gate transitions into ``navigate`` only
    after a successful, code-checked verification.

    Args:
        run_tool_core: ``run_bot``'s shared tool-execution core
            (``(name, args) -> (payload, run_llm)``) — every step's tool
            functions delegate to it.
        registry: The call's :class:`~app.tools.registry.ToolRegistry`;
            its tools are re-advertised at each step (Flows narrows the
            schema per node).
        hydrated_system: The call's hydrated (PHI-bearing) system prompt.
            Set as ``role_message`` on the first step so PHI becomes
            available to the LLM **only** post-verification; persists
            across the later steps.
        summary_prompt: Prompt that drives the running summary on every
            step after the first (RESET_WITH_SUMMARY). Defaults to
            :data:`SUMMARY_PROMPT`.
        ivr_path: Verified per-payer claims IVR path (#17), pre-rendered
            to numbered lines. When non-blank it guides the ``navigate``
            step to follow the map via ``press_digit`` (fall back to
            by-ear when the live menu diverges). Blank → today's
            listen-and-decide navigate task.
        ivr_goal: Per-agent navigation goal (``AgentConfig.ivr_goal``),
            already hydrated. When non-blank it's stated as the goal in
            the ``navigate`` step. Blank → omitted.
        knowledge_warmer: Optional per-call warmer. When supplied, relevant
            step tasks may synchronously include cache hits.
        knowledge_context: PHI-minimized context for cache query construction.

    Returns:
        The ``navigate`` :class:`NodeConfig` — the head of the chain.
    """

    def build_node(index: int) -> NodeConfig:
        step = STEPS[index]
        is_first = index == 0
        is_last = index == len(STEPS) - 1

        functions: list[FlowsFunctionSchema] = _tool_functions(registry, run_tool_core)
        if not is_last:
            functions.append(_advance_function(step, lambda: build_node(index + 1)))

        # The navigate step (always index 0) optionally follows a verified
        # map; every other step uses its fixed task verbatim.
        task = (
            build_navigate_task(ivr_path=ivr_path, ivr_goal=ivr_goal)
            if step.name == NAVIGATE
            else step.task
        )
        task = _task_with_knowledge(
            step=step,
            task=task,
            knowledge_warmer=knowledge_warmer,
            knowledge_context=knowledge_context,
        )

        node: NodeConfig = {
            "name": step.name,
            "task_messages": [{"role": "system", "content": task}],
            "functions": functions,
            # Post-verification: the opener already fired during the gate,
            # so it's safe (and wanted) to respond as each step is entered.
            "respond_immediately": True,
        }

        if is_first:
            # Drop the identity-gate conversation entirely and load the
            # hydrated (PHI-bearing) prompt now that the caller is verified.
            node["role_message"] = hydrated_system
            node["context_strategy"] = ContextStrategyConfig(strategy=ContextStrategy.RESET)
        else:
            # Bound context per step, carrying a running summary forward.
            node["context_strategy"] = ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt=summary_prompt,
            )
        return node

    return build_node(0)


def _build_data_driven_step_chain(
    definition: _FlowDefinition,
    *,
    run_tool_core: RunToolCore,
    registry: Any,
    hydrated_system: str,
    summary_prompt: str,
    ivr_path: str,
    ivr_goal: str,
    knowledge_warmer: PrefetchWarmer | None,
    knowledge_context: PrefetchContext | None,
) -> NodeConfig:
    nodes_by_id = _definition_node_map(definition)

    def build_node(node_id: str, *, is_initial: bool = False) -> NodeConfig | None:
        step = nodes_by_id.get(node_id)
        if step is None:
            return None

        functions: list[FlowsFunctionSchema] = _tool_functions(registry, run_tool_core)
        if step.type != "end":
            functions.append(_advance_function_for_flow_node(step, build_node))

        task = _task_for_flow_node(node=step, ivr_path=ivr_path, ivr_goal=ivr_goal)
        task = _task_with_knowledge(
            step=_Step(name=step.id, task=task),
            task=task,
            knowledge_warmer=knowledge_warmer,
            knowledge_context=knowledge_context,
        )

        node: NodeConfig = {
            "name": step.id,
            "task_messages": [{"role": "system", "content": task}],
            "functions": functions,
            "respond_immediately": True,
        }
        if is_initial:
            node["role_message"] = hydrated_system
            node["context_strategy"] = ContextStrategyConfig(strategy=ContextStrategy.RESET)
        else:
            node["context_strategy"] = ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt=summary_prompt,
            )
        return node

    first = build_node(definition.start, is_initial=True)
    if first is None:
        logger.warning(
            "flow_definition_transition_missing_default_flow",
            node_id="__start__",
            target=definition.start,
        )
        return _build_default_step_chain(
            run_tool_core=run_tool_core,
            registry=registry,
            hydrated_system=hydrated_system,
            summary_prompt=summary_prompt,
            ivr_path=ivr_path,
            ivr_goal=ivr_goal,
            knowledge_warmer=knowledge_warmer,
            knowledge_context=knowledge_context,
        )
    return first


def build_step_chain(
    *,
    run_tool_core: RunToolCore,
    registry: Any,
    hydrated_system: str,
    summary_prompt: str = SUMMARY_PROMPT,
    ivr_path: str = "",
    ivr_goal: str = "",
    knowledge_warmer: PrefetchWarmer | None = None,
    knowledge_context: PrefetchContext | None = None,
    flow_definition: dict[str, Any] | None = None,
) -> NodeConfig:
    """Build the post-verification step chain; return its first node.

    When runtime-config provides a valid data-driven ``flow_definition``, the
    graph drives the nodes and transitions. Missing or invalid definitions
    safely fall back to the original fixed 8-step flow.
    """
    definition = _usable_flow_definition(flow_definition)
    if definition is None:
        return _build_default_step_chain(
            run_tool_core=run_tool_core,
            registry=registry,
            hydrated_system=hydrated_system,
            summary_prompt=summary_prompt,
            ivr_path=ivr_path,
            ivr_goal=ivr_goal,
            knowledge_warmer=knowledge_warmer,
            knowledge_context=knowledge_context,
        )
    return _build_data_driven_step_chain(
        definition,
        run_tool_core=run_tool_core,
        registry=registry,
        hydrated_system=hydrated_system,
        summary_prompt=summary_prompt,
        ivr_path=ivr_path,
        ivr_goal=ivr_goal,
        knowledge_warmer=knowledge_warmer,
        knowledge_context=knowledge_context,
    )


# Re-exported for tests / bot.py without restating the dataclass.
__all__ = [
    "ASK_NEEDS",
    "CONFIRM_DENIAL",
    "DEADLINE",
    "FAX_PORTAL",
    "GREET",
    "NAVIGATE",
    "NAVIGATE_BASE_TASK",
    "REFERENCE_NUMBER",
    "REQUIRED_REFERENCE_FIELD",
    "REQUIRED_REFERENCE_NODE_ID",
    "STEPS",
    "SUMMARY_PROMPT",
    "WRAP",
    "build_navigate_task",
    "build_step_chain",
    "identity_gate_required_for_direction",
]
