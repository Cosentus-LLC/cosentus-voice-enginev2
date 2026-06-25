"""Pipecat Flows layer (EPIC #16).

* :mod:`app.flows.scaffold` — the 16a integration scaffold (#41):
  ``build_flow_manager`` (used every call) + the trivial 2-step flow.
* :mod:`app.flows.identity_gate` — the 16b identity gate (#42): the
  code-enforced step that blocks tool/data access until verified, and
  (16c) replaces the pre-verification system instruction with a PHI-free
  one.
* :mod:`app.flows.steps` — the 16c ordered, un-skippable post-verification
  steps (#43) with per-step bounded context and policy-driven IVR inclusion.
"""

from __future__ import annotations

from app.flows.identity_gate import (
    IDENTITY_GATE_NODE,
    PRE_VERIFICATION_ROLE_MESSAGE,
    build_identity_gate_flow,
    verify_against_case_data,
)
from app.flows.scaffold import build_flow_manager, build_scaffold_flow
from app.flows.steps import (
    GREETING_ALREADY_DONE_NOTE,
    GREETING_STATE_KEY,
    REFERENCE_NUMBER,
    REQUIRED_REFERENCE_FIELD,
    REQUIRED_REFERENCE_NODE_ID,
    STEPS,
    SUMMARY_PROMPT,
    build_step_chain,
    identity_gate_required_for_direction,
)

__all__ = [
    "IDENTITY_GATE_NODE",
    "GREETING_ALREADY_DONE_NOTE",
    "GREETING_STATE_KEY",
    "PRE_VERIFICATION_ROLE_MESSAGE",
    "REFERENCE_NUMBER",
    "REQUIRED_REFERENCE_FIELD",
    "REQUIRED_REFERENCE_NODE_ID",
    "STEPS",
    "SUMMARY_PROMPT",
    "build_flow_manager",
    "build_identity_gate_flow",
    "build_scaffold_flow",
    "build_step_chain",
    "identity_gate_required_for_direction",
    "verify_against_case_data",
]
