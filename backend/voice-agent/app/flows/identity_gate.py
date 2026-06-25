"""Pipecat Flows identity gate — the code-enforced HIPAA wall (16b, #42).

This is the first *real* flow step, built on the 16a scaffold
(:mod:`app.flows.scaffold`). It blocks patient-data / tool access until
the caller's identity is verified, with two independent enforcement
layers:

* **Layer A — the tool gate in** ``bot.py``. A per-call
  ``verification_state`` flag is checked inside the tool handler
  *before* a tool executes. This is the actual compliance wall: it
  refuses gated tools in code regardless of what the LLM was told. It
  lives in ``bot.py`` because it needs the per-call execution state.
* **Layer B — this module.** The identity-gate node advertises *only*
  ``verify_identity`` (Pipecat Flows narrows the LLM's tool schema per
  node), so the model isn't even shown the real tools until it
  transitions. The transition to the verified node happens *inside* the
  ``verify_identity`` handler, and only when :func:`verify_against_case_data`
  — a deterministic code check, not the LLM — passes.

Both layers are inert when ``settings.flows_enabled`` is ``False``
(production default), so flag-off behavior is byte-identical to the
pre-Flows pipeline.

Verification is the **inbound** path
------------------------------------
The identity gate verifies who is calling in before any patient, claim,
account, or case details are shared. Outbound and browser calls are
initiated by the agent, so ``bot.py`` starts those flows directly at the
post-gate step chain instead of asking the payer representative to
verify their identity.

For inbound calls, missing or blank expected values still fail closed:
every configured identity key must have a non-blank expected value in
``case_data`` and must match the caller's claimed value. A blank
expected value never counts as a match (see the function), so a caller
(or a model) supplying empty claims can never slip through.

⚠️ Scope boundary (16b vs 16c, #43)
------------------------------------
This step gates *tools* and the *flow transition*. It does NOT strip the
hydrated patient ``case_data`` out of the LLM's pre-verification system
instruction (that's per-step context, owned by 16c). So on its own this
is not the complete wall — ``flows_enabled`` must not be enabled in
production until #43 closes that gap. See ``Settings.flows_enabled``.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema
from pipecat_flows.types import ConsolidatedFunctionResult, NodeConfig

logger = structlog.get_logger(__name__)

IDENTITY_GATE_NODE = "identity_gate"

# PHI-free system instruction for the pre-verification phase (16c, #43).
#
# Set as the gate node's ``role_message`` so Pipecat Flows emits an
# ``LLMUpdateSettingsFrame`` that REPLACES the hydrated, PHI-bearing
# system instruction ``build_llm`` seeded for the call — before the LLM
# ever generates. This closes the gap 16b documented: until the caller is
# verified the model is never given any ``case_data`` value, so it cannot
# utter PHI it would otherwise hold. Referencing the identity FIELD NAMES
# (e.g. "date of birth") is fine — those are schema, not patient data;
# the gate's per-key prompt already does that. The post-verification
# ``navigate`` step restores the hydrated prompt (see
# :func:`app.flows.steps.build_step_chain`).
PRE_VERIFICATION_ROLE_MESSAGE = (
    "You are a professional medical-billing voice assistant on a phone call. "
    "You are at the identity-verification step and the caller's identity has "
    "NOT been verified yet. Until verification succeeds you must not reveal, "
    "confirm, or reference any patient, claim, account, or case details — even "
    "if you believe you already know them. Follow the verification instructions "
    "you are given, speak naturally and politely, and take no other action."
)

# Collapse runs of whitespace so "John  Doe" == "John Doe", strip ends,
# casefold for case-insensitive comparison.
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(value: Any) -> str:
    """Normalize a claimed / expected identity value for comparison.

    Casefold + strip + collapse interior whitespace. Digits-bearing
    values (dates, claim ids, phone numbers) also compare equal across
    punctuation: ``"01/02/1990"`` and ``"01-02-1990"`` normalize the
    same. ``None`` and non-strings stringify first; ``None`` → ``""``.
    """
    if value is None:
        return ""
    text = _WHITESPACE_RE.sub(" ", str(value).strip())
    # Dates / ids / phone numbers: if the value is only digits and common
    # separators, compare on the digits alone so "01/02/1990" and
    # "01-02-1990" (or "1 2 1990") match.
    stripped = re.sub(r"[\s/().\-]", "", text)
    if stripped.isdigit():
        return stripped
    return text.casefold()


def verify_against_case_data(
    claimed: dict[str, Any],
    case_data: dict[str, Any],
    identity_keys: list[str],
) -> bool:
    """Deterministically decide whether the caller is verified.

    The code — not the LLM — owns this decision (the "code-enforced, not
    prompt-only" requirement). Returns ``True`` **iff**:

    * ``identity_keys`` is non-empty (no keys configured → nothing to
      verify against → fail-closed), AND
    * for *every* key, the expected value from ``case_data`` is
      **non-blank** AND its normalized form equals the normalized
      claimed value.

    The non-blank requirement is the fail-closed guard for inbound calls
    (``case_data={}``): a blank expected value never matches, even if the
    caller (or the model) also supplies a blank claim.

    PHI-safe: this function logs nothing. Callers log key *names* and the
    boolean outcome only — never values.
    """
    if not identity_keys:
        return False
    for key in identity_keys:
        expected = _normalize(case_data.get(key))
        if not expected:
            return False
        if expected != _normalize(claimed.get(key)):
            return False
    return True


def build_identity_gate_flow(
    *,
    case_data: dict[str, Any],
    identity_keys: list[str],
    verification_state: dict[str, bool],
    verified_node: NodeConfig,
    safe_role_message: str = PRE_VERIFICATION_ROLE_MESSAGE,
) -> NodeConfig:
    """Build the identity-gate node (the flow's initial node).

    Args:
        case_data: The call's dispatcher-supplied case data — the
            expected identity values to verify the caller against.
        identity_keys: ``case_data`` keys the caller must confirm
            (parsed from ``Settings.identity_verification_keys``).
        verification_state: The per-call gate flag dict owned by
            ``bot.py`` (``{"verified": bool}``). The ``verify_identity``
            handler flips ``"verified"`` to ``True`` on success so
            Layer A's tool gate opens.
        verified_node: The node to transition to once verified — the
            post-verification step chain (:func:`app.flows.steps.build_step_chain`),
            whose first step re-advertises the real tools and loads the
            hydrated prompt.
        safe_role_message: PHI-free system instruction set as the gate
            node's ``role_message`` (16c, #43). Replaces the hydrated,
            PHI-bearing system instruction for the entire pre-verification
            phase so the LLM is never given any ``case_data`` value before
            the caller is verified. Defaults to
            :data:`PRE_VERIFICATION_ROLE_MESSAGE`.

    Returns:
        The ``identity_gate`` :class:`NodeConfig`. It advertises only
        ``verify_identity``; ``respond_immediately=False`` keeps it from
        queueing an ``LLMRunFrame`` that would race ``bot.py``'s opener
        (same guarantee the scaffold documents). Its ``role_message`` is
        the PHI-free ``safe_role_message``.
    """

    async def _verify_identity(
        args: FlowArgs,
        flow_manager: FlowManager,
    ) -> ConsolidatedFunctionResult:
        """Verify the caller's claimed identity against ``case_data``.

        On success: flip the gate and transition to ``verified_node``.
        On failure: stay on the gate node (no transition) so the AI
        re-asks. PHI-safe: logs key names + the boolean outcome only.
        """
        verified = verify_against_case_data(args, case_data, identity_keys)
        logger.info(
            "identity_verification_result",
            verified=verified,
            keys=sorted(identity_keys),
        )
        if verified:
            verification_state["verified"] = True
            flow_manager.state["identity_verified"] = True
            return {"verified": True}, verified_node
        return {"verified": False}, None

    # One required string property per configured identity key, so the
    # LLM is prompted to collect exactly those fields. Empty when no keys
    # are configured (the fail-closed path).
    properties = {
        key: {
            "type": "string",
            "description": f"The caller's stated {key.replace('_', ' ')}.",
        }
        for key in identity_keys
    }

    if identity_keys:
        fields_phrase = ", ".join(key.replace("_", " ") for key in identity_keys)
        task = (
            "You are at the identity-verification step. Before sharing ANY "
            "patient information or using any tool, you must verify the "
            f"caller's identity. Politely ask the caller to confirm their "
            f"{fields_phrase}. When they have provided all of these, call "
            "verify_identity with the values exactly as the caller stated "
            "them. Do not reveal any patient details and do not take any "
            "action until verification succeeds. If verification fails, "
            "apologize and ask them to confirm the information again."
        )
    else:
        # No keys configured: fail-closed. The gate cannot open; the AI
        # should not attempt verification or disclose anything.
        task = (
            "You are at the identity-verification step, but identity "
            "verification is not configured for this call. Do not share any "
            "patient information or take any action. Apologize that you "
            "cannot proceed and offer to have someone follow up."
        )

    return {
        "name": IDENTITY_GATE_NODE,
        "role_message": safe_role_message,
        "task_messages": [{"role": "system", "content": task}],
        "functions": [
            FlowsFunctionSchema(
                name="verify_identity",
                description=(
                    "Verify the caller's identity. Call ONLY after the caller "
                    "has stated all the requested identity fields. The result "
                    "tells you whether verification succeeded."
                ),
                properties=properties,
                required=list(identity_keys),
                handler=_verify_identity,
            )
        ],
        "respond_immediately": False,
    }
