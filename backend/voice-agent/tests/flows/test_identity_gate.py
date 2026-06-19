"""Unit tests for the Flows identity gate (16b, #42).

Two surfaces:

* :func:`verify_against_case_data` — the deterministic, code-enforced
  verification predicate (the "not prompt-only" requirement), including
  the fail-closed guards (no keys configured; blank expected value, i.e.
  the inbound ``case_data={}`` path).
* :func:`build_identity_gate_flow` — the node advertises ONLY
  ``verify_identity`` and never auto-responds; its handler flips the
  shared ``verification_state`` and transitions to the verified node on
  success, and stays put on failure.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.flows import build_identity_gate_flow, verify_against_case_data
from app.flows.identity_gate import IDENTITY_GATE_NODE, PRE_VERIFICATION_ROLE_MESSAGE

_VERIFIED_NODE = {"name": "verified"}


def _fm() -> SimpleNamespace:
    """Minimal FlowManager stand-in: only ``.state`` is used."""
    return SimpleNamespace(state={})


# ── verify_against_case_data ─────────────────────────────────────────────


class TestVerifyAgainstCaseData:
    def test_matches_exact(self):
        assert verify_against_case_data(
            {"patient_name": "John Doe"}, {"patient_name": "John Doe"}, ["patient_name"]
        )

    def test_normalizes_case_and_whitespace(self):
        assert verify_against_case_data(
            {"patient_name": "  john   DOE "},
            {"patient_name": "John Doe"},
            ["patient_name"],
        )

    def test_date_digit_compare_ignores_separators(self):
        assert verify_against_case_data({"dob": "01-02-1990"}, {"dob": "01/02/1990"}, ["dob"])

    def test_rejects_mismatch(self):
        assert not verify_against_case_data(
            {"patient_name": "Jane Roe"}, {"patient_name": "John Doe"}, ["patient_name"]
        )

    def test_all_keys_must_match(self):
        claimed = {"patient_name": "John Doe", "dob": "01/02/1990"}
        case = {"patient_name": "John Doe", "dob": "12/31/1980"}
        assert not verify_against_case_data(claimed, case, ["patient_name", "dob"])

    def test_false_when_no_keys_configured(self):
        # Fail-closed: nothing to verify against.
        assert not verify_against_case_data({"x": "y"}, {"x": "y"}, [])

    def test_false_when_expected_blank_inbound(self):
        # Inbound: case_data={} → expected blank → never matches, even if
        # the caller (or model) also supplies a blank claim.
        assert not verify_against_case_data({"patient_name": ""}, {}, ["patient_name"])
        assert not verify_against_case_data({"patient_name": "John Doe"}, {}, ["patient_name"])

    def test_false_when_claim_missing(self):
        assert not verify_against_case_data({}, {"patient_name": "John Doe"}, ["patient_name"])


# ── build_identity_gate_flow ─────────────────────────────────────────────


class TestBuildIdentityGateFlow:
    def _node(self, **kw):
        defaults = dict(
            case_data={"patient_name": "John Doe"},
            identity_keys=["patient_name"],
            verification_state={"verified": False},
            verified_node=_VERIFIED_NODE,
        )
        defaults.update(kw)
        return build_identity_gate_flow(**defaults)

    def test_node_name_and_only_verify_identity(self):
        node = self._node()
        assert node["name"] == IDENTITY_GATE_NODE
        assert [f.name for f in node["functions"]] == ["verify_identity"]

    def test_never_responds_immediately(self):
        # Same opener-race guard the scaffold documents.
        assert self._node()["respond_immediately"] is False

    def test_role_message_defaults_to_phi_free_prompt(self):
        # 16c: the gate replaces the hydrated system instruction with a
        # PHI-free one for the whole pre-verification phase.
        assert self._node()["role_message"] == PRE_VERIFICATION_ROLE_MESSAGE

    def test_gate_node_contains_no_case_data_values(self):
        # Safety: with sentinel PHI in case_data, none of those values may
        # appear anywhere in the gate node's context (role_message or any
        # task message). Referencing the field NAMES (keys) is fine.
        sentinels = {
            "patient_name": "SENTINEL_NAME_ZZQ",
            "dob": "1911-11-11",
            "claim_id": "SENTINEL_CLAIM_7X7",
        }
        node = self._node(
            case_data=sentinels,
            identity_keys=["patient_name", "dob", "claim_id"],
        )
        blob = node["role_message"] + " ".join(m["content"] for m in node["task_messages"])
        for value in sentinels.values():
            assert value not in blob

    def test_custom_safe_role_message_is_used(self):
        node = self._node(safe_role_message="CUSTOM-SAFE")
        assert node["role_message"] == "CUSTOM-SAFE"

    def test_properties_track_identity_keys(self):
        node = self._node(identity_keys=["patient_name", "dob"])
        fn = node["functions"][0]
        assert set(fn.properties) == {"patient_name", "dob"}
        assert sorted(fn.required) == ["dob", "patient_name"]

    def test_empty_keys_produce_no_properties(self):
        node = self._node(identity_keys=[])
        fn = node["functions"][0]
        assert fn.properties == {}
        assert fn.required == []

    @pytest.mark.asyncio
    async def test_handler_pass_flips_state_and_transitions(self):
        state = {"verified": False}
        node = self._node(verification_state=state)
        handler = node["functions"][0].handler
        fm = _fm()

        result, next_node = await handler({"patient_name": "John Doe"}, fm)

        assert result == {"verified": True}
        assert next_node is _VERIFIED_NODE
        assert state["verified"] is True
        assert fm.state["identity_verified"] is True

    @pytest.mark.asyncio
    async def test_handler_fail_stays_on_node_and_leaves_state(self):
        state = {"verified": False}
        node = self._node(verification_state=state)
        handler = node["functions"][0].handler

        result, next_node = await handler({"patient_name": "Wrong Name"}, _fm())

        assert result == {"verified": False}
        assert next_node is None
        assert state["verified"] is False

    @pytest.mark.asyncio
    async def test_handler_fail_closed_when_no_keys(self):
        state = {"verified": False}
        node = self._node(identity_keys=[], verification_state=state)
        handler = node["functions"][0].handler

        result, next_node = await handler({}, _fm())

        assert result == {"verified": False}
        assert next_node is None
        assert state["verified"] is False
