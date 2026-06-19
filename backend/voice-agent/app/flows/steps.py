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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pipecat_flows import (
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
)
from pipecat_flows.types import ConsolidatedFunctionResult, NodeConfig

from app.knowledge.prefetch import PrefetchContext, PrefetchWarmer

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
) -> NodeConfig:
    """Build the post-verification step chain; return its first node.

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
    "STEPS",
    "SUMMARY_PROMPT",
    "WRAP",
    "build_navigate_task",
    "build_step_chain",
]
