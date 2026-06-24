"""Tests for PHI-free dashboard intelligence helpers."""

from __future__ import annotations

import pytest
from app.bot.intelligence import terminal_step_for_node


@pytest.mark.parametrize(
    ("node_name", "expected"),
    [
        ("identity_gate", "identity_verification"),
        ("navigate", "ivr_navigation"),
        ("greet", "greeting"),
        ("confirm_denial_reason", "claim_status"),
        ("ask_needs", "resolution"),
        ("fax_or_portal", "resolution"),
        ("deadline", "resolution"),
        ("reference_number", "reference_number"),
        ("wrap", "wrap_up"),
    ],
)
def test_terminal_step_for_node_maps_default_flow_nodes(node_name, expected):
    assert terminal_step_for_node(node_name, flows_enabled=True) == expected


def test_terminal_step_for_node_preserves_dashboard_step_names():
    assert terminal_step_for_node("claim_lookup", flows_enabled=True) == "claim_lookup"


def test_terminal_step_for_node_preserves_unknown_custom_node():
    assert terminal_step_for_node("custom_review", flows_enabled=True) == "custom_review"


def test_terminal_step_for_node_fallbacks_when_no_node_was_set():
    assert terminal_step_for_node(None, flows_enabled=True) == "identity_verification"
    assert terminal_step_for_node(None, flows_enabled=False) == "greeting"
