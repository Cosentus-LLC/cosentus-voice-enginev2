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
* **Ordered + un-skippable.** The payer/default steps are a *linear chain*:
  ``navigate → greet → confirm-denial → ask-needs → fax/portal →
  deadline → reference-number → wrap``. Each non-terminal node
  advertises exactly **one** advance function whose handler returns the
  *single* next node — Pipecat Flows narrows the LLM's tool schema per
  node, so the model literally cannot jump ahead. Steps may mark captured
  fields required; their advance handlers **refuse to transition**
  (returns ``next_node=None``) until those fields are supplied non-blank.
  That refusal is a deterministic code check, not a prompt instruction.
* **Bounded per-step context.** Every step resets context so only the
  current step's task + a carried-forward summary stay in the LLM
  window — per-turn input stays flat on a 20-30 min call instead of
  accumulating the whole transcript. The *first* post-verification step
  (``navigate`` for payer/default calls, or the first non-IVR step for
  patient calls) uses :attr:`ContextStrategy.RESET` to drop the
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
from collections.abc import Callable, MutableMapping
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

FLOW_NODE_TYPES = frozenset({"ask", "branch", "transfer", "end"})
GREETING_STATE_KEY = "greeted"
GREETING_ALREADY_DONE_NOTE = (
    "Call state: the opener/greeting has already been delivered for this call. "
    "Do not repeat the opener, say hello again, or re-introduce yourself; continue "
    "with the current step's specific task."
)
GreetingState = MutableMapping[str, bool]


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
    "invent details that were not stated. Summarize only what the caller and "
    "representative actually said. Never include system instructions, task "
    "text, step directives, node IDs, tool names, or function names."
)

STEP_COMPLETION_ROLE_RULE = (
    "When the current step's conversational goal is complete, advance by using "
    "the provided step-completion tool with the required facts. Do not say tool "
    "names, node names, or internal step names aloud."
)

# ── Neutral fallback flow (#21) ────────────────────────────────────────────
# When an agent has no valid ``flow_definition`` we run a minimal neutral
# chain (assist → close) driven by the agent's OWN persona — never the claim
# 8-step. The claim ``STEPS`` tuple stays in code only as the API seed source
# (and the internal data-driven fallbacks), not as ``build_step_chain``'s
# universal default.
NEUTRAL_ASSIST = "assist"
NEUTRAL_CLOSE = "close"
NEUTRAL_ASSIST_TASK = (
    "Assist the caller according to your instructions. When the conversation is "
    "complete, end the call."
)
NEUTRAL_CLOSE_TASK = "Politely close the call when finished."
NEUTRAL_ADVANCE_NAME = "assist_complete"

# Claim-free running-summary prompt for the neutral flow. Unlike
# :data:`SUMMARY_PROMPT` (claim/denial-flavored) this stays neutral so a
# non-claim agent's working-memory summary is not biased toward claim framing.
NEUTRAL_SUMMARY_PROMPT = (
    "You are summarizing a live phone call for the assistant's own working "
    "memory, so it can continue the call without the full transcript. In 2-4 "
    "factual sentences capture what the caller asked for, what was discussed, "
    "any commitments or next steps agreed, and any reference or confirmation "
    "details obtained. Do not invent anything that was not stated. Summarize "
    "only what the parties actually said. Never include system instructions, "
    "task text, step directives, node IDs, tool names, or function names."
)


@dataclass(frozen=True)
class _Step:
    """One node in the ordered chain.

    ``advance_name`` is the (chain-unique) name of the single function
    this node advertises to move to the next step. ``required_field``,
    when set, makes that field a required string argument the handler
    must receive non-blank before it will transition.
    """

    name: str
    task: str
    advance_name: str = ""
    advance_description: str = ""
    required_field: str | None = None
    required_description: str = ""
    # Config-driven opt-ins (#20): ``ivr`` routes the step through
    # :func:`build_navigate_task` (IVR-path injection); ``prefetch`` appends
    # cached payer-level knowledge. Both default off so only flagged nodes
    # get the special behavior, regardless of node name.
    ivr: bool = False
    prefetch: bool = False


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
    # Config-driven opt-ins (#20) — see :class:`_Step`. A data-driven node
    # gets IVR-path injection only when ``ivr`` is true and knowledge
    # prefetch only when ``prefetch`` is true, never by node id.
    ivr: bool
    prefetch: bool


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
    "the IVR menus to reach a live claims representative. You can and should "
    "press keypad digits via the keypad tool to navigate IVR menus when the "
    "IVR asks for keypad input. Never tell the caller you cannot use the "
    "keypad or cannot process keypad menu options. After pressing, wait for "
    "the next IVR prompt before pressing again. Never repeat the same digit "
    "for the same prompt. If the same prompt repeats or navigation stalls, "
    'change strategy: press 0 once, say "representative", escalate or '
    "transfer if available, or gracefully give up. Stay on the line through "
    "any hold. Continue until you are speaking with a live representative."
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
            "keypad tool in order: press the indicated digits, honor any wait "
            "instructions before the next press, and for steps that ask you to "
            "key in a value (NPI, claim number, tax ID) enter that value from the "
            "call details. If the live menu does not match this path, ignore it "
            "and navigate by ear instead:\n" + path
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
        ivr=True,
    ),
    _Step(
        name=GREET,
        task=(
            "Greet the representative. Introduce yourself as calling on behalf of "
            "the provider about a denied claim, and say you'd like to resolve it."
        ),
        advance_name="greeting_done",
        advance_description="Call once you have greeted and introduced the call.",
    ),
    _Step(
        name=CONFIRM_DENIAL,
        task=(
            "Confirm the denial reason with the representative. Ask why the claim "
            "was denied and restate it back to confirm you understood it "
            "correctly."
        ),
        advance_name="denial_reason_confirmed",
        advance_description="Call once the denial reason is confirmed.",
    ),
    _Step(
        name=ASK_NEEDS,
        task=(
            "Ask the representative exactly what is needed to resolve or appeal "
            "the denial — for example corrected codes, medical records, an appeal "
            "form, or other documentation."
        ),
        advance_name="needs_identified",
        advance_description="Call once you understand what is required to resolve the denial.",
    ),
    _Step(
        name=FAX_PORTAL,
        task=(
            "Determine how to submit the required information: confirm the correct "
            "fax number or the payer portal to use for the submission."
        ),
        advance_name="submission_method_confirmed",
        advance_description="Call once the fax number or portal for submission is confirmed.",
    ),
    _Step(
        name=DEADLINE,
        task=(
            "Confirm the deadline by which the information must be submitted (the "
            "timely-filing or appeal window)."
        ),
        advance_name="deadline_confirmed",
        advance_description="Call once the submission deadline is confirmed.",
        prefetch=True,
    ),
    _Step(
        name=REFERENCE_NUMBER,
        task=(
            "Before ending, ask the representative for a call reference or "
            "confirmation number for this interaction. You must capture the "
            "number exactly as stated before moving on."
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
            "number you recorded, then politely close the call."
        ),
    ),
)

# ── Builder ──────────────────────────────────────────────────────────────

# The shared tool-execution core (``run_bot.run_tool_to_payload``):
# ``(tool_name, arguments) -> (payload, run_llm)``.
RunToolCore = Callable[[str, dict[str, Any]], "Any"]


def _has_greeted(greeting_state: GreetingState | None) -> bool:
    return bool(greeting_state and greeting_state.get(GREETING_STATE_KEY))


def _mark_greeted(
    greeting_state: GreetingState | None,
    flow_manager: FlowManager | None = None,
) -> None:
    if greeting_state is not None:
        greeting_state[GREETING_STATE_KEY] = True
    if flow_manager is not None:
        flow_manager.state[GREETING_STATE_KEY] = True


def _task_messages(
    task: str,
    *,
    greeting_state: GreetingState | None,
    include_greeting_note: bool = True,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": task}]
    if include_greeting_note and _has_greeted(greeting_state):
        messages.append({"role": "system", "content": GREETING_ALREADY_DONE_NOTE})
    return messages


def _advance_function(
    step: _Step,
    next_builder: Callable[[FlowManager], NodeConfig],
    *,
    on_advance: Callable[[FlowManager], None] | None = None,
) -> FlowsFunctionSchema:
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
            if on_advance is not None:
                on_advance(flow_manager)
            return {"status": "ok", step.required_field: value}, next_builder(flow_manager)
        if on_advance is not None:
            on_advance(flow_manager)
        return {"status": "ok"}, next_builder(flow_manager)

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
                # Only literal JSON ``true`` opts in; missing/false/strings
                # stay disabled so bad config can't widen live-call behavior.
                ivr=raw_node.get("ivr") is True,
                prefetch=raw_node.get("prefetch") is True,
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
    if node.id == REFERENCE_NUMBER:
        return "record_reference_number"
    suffix = _FUNCTION_NAME_RE.sub("_", node.id.strip()).strip("_").lower()
    return f"advance_{suffix or 'flow_step'}"


def _required_capture_fields(node: _FlowNode) -> tuple[str, ...]:
    return node.capture if node.required else ()


def _record_captured_field(flow_manager: FlowManager, field: str, value: str) -> None:
    flow_manager.state[field] = value
    # Compatibility for existing claim-flow data that captures call_reference
    # while downstream code/tests also read the default-chain reference key.
    if field == "call_reference":
        flow_manager.state[REFERENCE_NUMBER] = value


def _task_for_flow_node(*, node: _FlowNode, ivr_path: str, ivr_goal: str, include_ivr: bool) -> str:
    if node.ivr and include_ivr:
        parts = [build_navigate_task(ivr_path=ivr_path, ivr_goal=ivr_goal)]
    else:
        label = node.label.strip() or node.id.replace("_", " ")
        prompt = node.say.strip() or f"Proceed with the {label} step."
        parts = [prompt]

    if node.capture:
        fields = ", ".join(field.replace("_", " ") for field in node.capture)
        parts.append(f"Capture these field(s) during this step when available: {fields}.")
        if node.required:
            parts.append("All captured field(s) for this step are required before moving on.")
    if node.type == "branch":
        branch_lines = [f"- {branch.when}" for branch in node.branches if branch.when and branch.to]
        fallback = node.fallback or ""
        parts.append(
            "Choose the matching branch condition. Use the fallback option if none "
            "of the branch conditions match."
        )
        if branch_lines:
            parts.append("Branch conditions:\n" + "\n".join(branch_lines))
        if fallback:
            parts.append("A fallback branch is available if no condition matches.")
    elif node.type == "transfer":
        parts.append("Transfer the call if this step cannot be completed with the current party.")
    elif node.type != "end":
        parts.append("When this step is complete, move to the next step.")
    else:
        parts.append("Politely close the call when finished.")
    return "\n\n".join(parts)


def _advance_function_for_flow_node(
    node: _FlowNode,
    build_next: Callable[[str], NodeConfig | None],
    *,
    greeting_state: GreetingState | None = None,
) -> FlowsFunctionSchema:
    required_fields = _required_capture_fields(node)
    required_field_set = set(required_fields)

    async def handler(args: FlowArgs, flow_manager: FlowManager) -> ConsolidatedFunctionResult:
        captured: dict[str, str] = {}
        for field in node.capture:
            value = str(args.get(field) or "").strip()
            if field in required_field_set and not value:
                return {"status": "missing", "field": field}, None
            if value:
                _record_captured_field(flow_manager, field, value)
                captured[field] = value

        target = node.next
        if node.type == "branch":
            allowed = {branch.to for branch in node.branches}
            if node.fallback:
                allowed.add(node.fallback)
            requested = str(args.get("branch_to") or "").strip()
            target = requested if requested in allowed else node.fallback
        elif target is None:
            target = node.fallback

        if node.id == GREET:
            _mark_greeted(greeting_state, flow_manager)

        next_node = build_next(target) if target else None
        if target and next_node is None:
            logger.warning(
                "flow_definition_transition_missing_default_flow",
                node_id=node.id,
                target=target,
            )
        return {"status": "ok", **captured}, next_node

    properties: dict[str, Any] = {}
    for field in node.capture:
        properties[field] = {
            "type": "string",
            "description": f"The {field.replace('_', ' ')} captured during this step.",
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
        required=list(required_fields),
        handler=handler,
    )


def _task_with_knowledge(
    *,
    prefetch: bool,
    task: str,
    knowledge_warmer: PrefetchWarmer | None,
    knowledge_context: PrefetchContext | None,
) -> str:
    """Append a cached payer-level fact when one is already warm.

    Only runs for nodes that opt in via ``prefetch`` (#20) — the cached
    payer facts are appended for any such node, regardless of its name.
    This is cache-only. A miss leaves the task text unchanged and the warmer
    fills in the background for a later turn.
    """
    if not prefetch or knowledge_warmer is None or knowledge_context is None:
        return task
    payer = (knowledge_context.payer or "").strip()
    if not payer:
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


def _role_message_for_step_chain(hydrated_system: str) -> str:
    base = hydrated_system.rstrip()
    if not base:
        return STEP_COMPLETION_ROLE_RULE
    return base + "\n\n" + STEP_COMPLETION_ROLE_RULE


def _build_default_step_chain(
    *,
    run_tool_core: RunToolCore,
    registry: Any,
    hydrated_system: str,
    summary_prompt: str = SUMMARY_PROMPT,
    ivr_path: str = "",
    ivr_goal: str = "",
    include_ivr: bool = True,
    knowledge_warmer: PrefetchWarmer | None = None,
    knowledge_context: PrefetchContext | None = None,
    greeting_state: GreetingState | None = None,
) -> NodeConfig:
    """Build the claim 8-step chain from :data:`STEPS`; return its first node.

    Retained as the **claim-flow seed-shape reference** only (#21): it mirrors
    the data-driven Claims ``flow_definition`` the API seeds, and the tests
    exercise it directly. It is **no longer** wired as a runtime fallback —
    ``build_step_chain`` and the data-driven edge cases route to
    :func:`_build_neutral_step_chain` instead, so a non-claim or misconfigured
    agent can never silently run claim steps.

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
        include_ivr: Whether to include the payer IVR ``navigate`` step.
            Patient calls set this false and begin at the first non-IVR
            step, while preserving the no-regreeting state.
        knowledge_warmer: Optional per-call warmer. When supplied, relevant
            step tasks may synchronously include cache hits.
        knowledge_context: PHI-minimized context for cache query construction.
        greeting_state: Per-call state that records whether the opener/greeting
            has already been delivered. When true, the default chain skips the
            ``greet`` node and carries a no-regreeting note across resets.

    Returns:
        The first :class:`NodeConfig` in the policy-selected chain.
    """
    steps = STEPS
    if not include_ivr:
        steps = STEPS[2:] if _has_greeted(greeting_state) else STEPS[1:]

    def build_node(index: int, *, include_greeting_note: bool = True) -> NodeConfig:
        step = steps[index]
        is_first = index == 0
        is_last = index == len(steps) - 1

        functions: list[FlowsFunctionSchema] = _tool_functions(registry, run_tool_core)
        if not is_last:
            if step.name == NAVIGATE:
                functions.append(
                    _advance_function(
                        step,
                        lambda _flow_manager: (
                            build_node(index + 2)
                            if _has_greeted(greeting_state)
                            else build_node(index + 1, include_greeting_note=False)
                        ),
                    )
                )
            elif step.name == GREET:
                functions.append(
                    _advance_function(
                        step,
                        lambda _flow_manager: build_node(index + 1),
                        on_advance=lambda flow_manager: _mark_greeted(
                            greeting_state,
                            flow_manager,
                        ),
                    )
                )
            else:
                functions.append(
                    _advance_function(step, lambda _flow_manager: build_node(index + 1))
                )

        # The IVR-flagged step (the navigate step, always index 0) optionally
        # follows a verified map; every other step uses its fixed task verbatim.
        task = build_navigate_task(ivr_path=ivr_path, ivr_goal=ivr_goal) if step.ivr else step.task
        task = _task_with_knowledge(
            prefetch=step.prefetch,
            task=task,
            knowledge_warmer=knowledge_warmer,
            knowledge_context=knowledge_context,
        )

        node: NodeConfig = {
            "name": step.name,
            "task_messages": _task_messages(
                task,
                greeting_state=greeting_state,
                include_greeting_note=include_greeting_note,
            ),
            "functions": functions,
            # Post-verification: the opener already fired during the gate,
            # so it's safe (and wanted) to respond as each step is entered.
            "respond_immediately": True,
        }

        if is_first:
            # Drop the identity-gate conversation entirely and load the
            # hydrated (PHI-bearing) prompt now that the caller is verified.
            node["role_message"] = _role_message_for_step_chain(hydrated_system)
            node["context_strategy"] = ContextStrategyConfig(strategy=ContextStrategy.RESET)
        else:
            # Bound context per step, carrying a running summary forward.
            node["context_strategy"] = ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt=summary_prompt,
            )
        return node

    return build_node(0)


def _build_neutral_step_chain(
    *,
    run_tool_core: RunToolCore,
    registry: Any,
    hydrated_system: str,
    greeting_state: GreetingState | None = None,
) -> NodeConfig:
    """Build the neutral fallback chain (assist → close); return its first node.

    Used whenever an agent has no usable ``flow_definition`` (#21). It runs the
    agent's OWN persona (``hydrated_system``) over a minimal two-node chain —
    one ``assist`` node that lets the agent help the caller per its own
    instructions, then a terminal ``close`` node — instead of the claim
    8-step. There are no claim-specific steps, no required ``reference_number``
    node, and no IVR/prefetch wiring; ``end_call`` (and every other call tool)
    is advertised at both nodes because they come from the registry via
    :func:`_tool_functions`.

    Args:
        run_tool_core: ``run_bot``'s shared tool-execution core.
        registry: The call's tool registry; its tools are re-advertised at
            each node.
        hydrated_system: The call's hydrated (PHI-bearing) system prompt. Set
            as ``role_message`` on the first node so PHI becomes available to
            the LLM only post-verification, identical to the other chains.
        greeting_state: Per-call greeting state; when already greeted the
            no-regreeting note rides along via :func:`_task_messages`.

    Returns:
        The first :class:`NodeConfig` (the ``assist`` node).
    """

    def build_close() -> NodeConfig:
        return {
            "name": NEUTRAL_CLOSE,
            "task_messages": _task_messages(NEUTRAL_CLOSE_TASK, greeting_state=greeting_state),
            "functions": _tool_functions(registry, run_tool_core),
            "respond_immediately": True,
            "context_strategy": ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt=NEUTRAL_SUMMARY_PROMPT,
            ),
        }

    assist_step = _Step(
        name=NEUTRAL_ASSIST,
        task=NEUTRAL_ASSIST_TASK,
        advance_name=NEUTRAL_ADVANCE_NAME,
        advance_description=(
            "Call once the caller has been assisted and the conversation is complete."
        ),
    )
    functions: list[FlowsFunctionSchema] = _tool_functions(registry, run_tool_core)
    functions.append(_advance_function(assist_step, lambda _flow_manager: build_close()))
    return {
        "name": NEUTRAL_ASSIST,
        "task_messages": _task_messages(NEUTRAL_ASSIST_TASK, greeting_state=greeting_state),
        "functions": functions,
        "respond_immediately": True,
        "role_message": _role_message_for_step_chain(hydrated_system),
        "context_strategy": ContextStrategyConfig(strategy=ContextStrategy.RESET),
    }


def _build_data_driven_step_chain(
    definition: _FlowDefinition,
    *,
    run_tool_core: RunToolCore,
    registry: Any,
    hydrated_system: str,
    summary_prompt: str,
    ivr_path: str,
    ivr_goal: str,
    include_ivr: bool,
    knowledge_warmer: PrefetchWarmer | None,
    knowledge_context: PrefetchContext | None,
    greeting_state: GreetingState | None,
) -> NodeConfig:
    nodes_by_id = _definition_node_map(definition)
    start_node_id = definition.start

    if not include_ivr:
        start_node = nodes_by_id.get(start_node_id)
        if start_node is not None and start_node.ivr:
            targets = _outgoing_targets(start_node)
            if len(targets) == 1:
                start_node_id = targets[0]
            else:
                logger.warning(
                    "flow_definition_no_ivr_start_ambiguous_neutral_fallback",
                    node_id=start_node.id,
                    target_count=len(targets),
                )
                return _build_neutral_step_chain(
                    run_tool_core=run_tool_core,
                    registry=registry,
                    hydrated_system=hydrated_system,
                    greeting_state=greeting_state,
                )

        if _has_greeted(greeting_state):
            start_node = nodes_by_id.get(start_node_id)
            if start_node is not None and start_node.id == GREET and start_node.next:
                start_node_id = start_node.next

    def build_node(node_id: str, *, is_initial: bool = False) -> NodeConfig | None:
        step = nodes_by_id.get(node_id)
        if step is None:
            return None

        functions: list[FlowsFunctionSchema] = _tool_functions(registry, run_tool_core)
        if step.type != "end":
            functions.append(
                _advance_function_for_flow_node(
                    step,
                    build_node,
                    greeting_state=greeting_state,
                )
            )

        task = _task_for_flow_node(
            node=step, ivr_path=ivr_path, ivr_goal=ivr_goal, include_ivr=include_ivr
        )
        task = _task_with_knowledge(
            prefetch=step.prefetch,
            task=task,
            knowledge_warmer=knowledge_warmer,
            knowledge_context=knowledge_context,
        )

        node: NodeConfig = {
            "name": step.id,
            "task_messages": _task_messages(task, greeting_state=greeting_state),
            "functions": functions,
            "respond_immediately": True,
        }
        if is_initial:
            node["role_message"] = _role_message_for_step_chain(hydrated_system)
            node["context_strategy"] = ContextStrategyConfig(strategy=ContextStrategy.RESET)
        else:
            node["context_strategy"] = ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt=summary_prompt,
            )
        return node

    first = build_node(start_node_id, is_initial=True)
    if first is None:
        logger.warning(
            "flow_definition_transition_missing_neutral_fallback",
            node_id="__start__",
            target=start_node_id,
        )
        return _build_neutral_step_chain(
            run_tool_core=run_tool_core,
            registry=registry,
            hydrated_system=hydrated_system,
            greeting_state=greeting_state,
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
    include_ivr: bool = True,
    knowledge_warmer: PrefetchWarmer | None = None,
    knowledge_context: PrefetchContext | None = None,
    flow_definition: dict[str, Any] | None = None,
    greeting_state: GreetingState | None = None,
    agent_id: str = "",
) -> NodeConfig:
    """Build the post-verification step chain; return its first node.

    When runtime-config provides a valid data-driven ``flow_definition``, the
    graph drives the nodes and transitions. Missing or invalid definitions
    safely fall back to the **neutral** chain (assist → close) driven by the
    agent's own persona (#21) — never the claim 8-step, so a non-claim or
    misconfigured agent can no longer silently run claim steps.
    ``include_ivr=False`` omits the payer IVR navigation step for patient
    calls (data-driven flows only).
    """
    definition = _usable_flow_definition(flow_definition)
    if definition is None:
        logger.warning(
            "flow_definition_missing_neutral_fallback",
            agent_id=agent_id,
            reason="missing" if flow_definition is None else "invalid",
        )
        return _build_neutral_step_chain(
            run_tool_core=run_tool_core,
            registry=registry,
            hydrated_system=hydrated_system,
            greeting_state=greeting_state,
        )
    return _build_data_driven_step_chain(
        definition,
        run_tool_core=run_tool_core,
        registry=registry,
        hydrated_system=hydrated_system,
        summary_prompt=summary_prompt,
        ivr_path=ivr_path,
        ivr_goal=ivr_goal,
        include_ivr=include_ivr,
        knowledge_warmer=knowledge_warmer,
        knowledge_context=knowledge_context,
        greeting_state=greeting_state,
    )


# Re-exported for tests / bot.py without restating the dataclass.
__all__ = [
    "ASK_NEEDS",
    "CONFIRM_DENIAL",
    "DEADLINE",
    "FAX_PORTAL",
    "GREET",
    "GREETING_ALREADY_DONE_NOTE",
    "GREETING_STATE_KEY",
    "NAVIGATE",
    "NAVIGATE_BASE_TASK",
    "NEUTRAL_ADVANCE_NAME",
    "NEUTRAL_ASSIST",
    "NEUTRAL_ASSIST_TASK",
    "NEUTRAL_CLOSE",
    "NEUTRAL_CLOSE_TASK",
    "REFERENCE_NUMBER",
    "STEPS",
    "STEP_COMPLETION_ROLE_RULE",
    "SUMMARY_PROMPT",
    "WRAP",
    "build_navigate_task",
    "build_step_chain",
    "identity_gate_required_for_direction",
]
