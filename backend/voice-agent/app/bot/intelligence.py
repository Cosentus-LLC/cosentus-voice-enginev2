"""PHI-free per-call intelligence metrics for dashboard aggregation."""

from __future__ import annotations

from app.flows import IDENTITY_GATE_NODE
from app.flows.steps import (
    ASK_NEEDS,
    CONFIRM_DENIAL,
    DEADLINE,
    FAX_PORTAL,
    GREET,
    NAVIGATE,
    REFERENCE_NUMBER,
    WRAP,
)

DASHBOARD_TERMINAL_STEPS = frozenset(
    {
        "greeting",
        "ivr_navigation",
        "identity_verification",
        "claim_lookup",
        "claim_status",
        "resolution",
        "reference_number",
        "wrap_up",
    }
)

_FLOW_NODE_TO_TERMINAL_STEP = {
    IDENTITY_GATE_NODE: "identity_verification",
    NAVIGATE: "ivr_navigation",
    GREET: "greeting",
    CONFIRM_DENIAL: "claim_status",
    ASK_NEEDS: "resolution",
    FAX_PORTAL: "resolution",
    DEADLINE: "resolution",
    REFERENCE_NUMBER: "reference_number",
    WRAP: "wrap_up",
}


def terminal_step_for_node(node_name: str | None, *, flows_enabled: bool) -> str:
    """Normalize the active flow node to the API dashboard terminal-step field."""
    if node_name in DASHBOARD_TERMINAL_STEPS:
        return node_name
    if node_name in _FLOW_NODE_TO_TERMINAL_STEP:
        return _FLOW_NODE_TO_TERMINAL_STEP[node_name]
    if node_name is not None:
        return node_name
    return "identity_verification" if flows_enabled else "greeting"
