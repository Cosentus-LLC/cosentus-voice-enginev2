"""Unit tests for the verified per-payer IVR path loader (#17).

Covers the model rendering (``press`` + ``action`` step kinds → numbered
prompt text) and the best-effort loader's success + every fall-back-to-
``None`` path (404, non-200, empty path, malformed JSON, validation
error, invoke error). The loader must never raise — a ``None`` means the
agent navigates by ear.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from app.config.payer_knowledge import (
    IvrStep,
    PayerIvrPath,
    load_payer_ivr_path,
)
from app.config.settings import Settings


def _settings() -> Settings:
    return Settings(
        voice_api_lambda_name="test-voice-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
    )


# A row with both step kinds: static DTMF presses and dynamic-entry actions.
_ROW = {
    "id": "uhc",
    "payer_id": "united-healthcare",
    "ivr_path_claims": [
        {"step": 1, "press": "3", "label": "Provider services"},
        {"step": 2, "press": "1", "label": "Claims"},
        {"step": 3, "action": "enter_npi", "label": "NPI prompt"},
        {"step": 4, "action": "enter_claim", "label": "Claim number prompt"},
    ],
}


def _invoke_response(*, status: int = 200, body: object | str | None = None) -> dict:
    """Build a boto3 invoke envelope whose Payload .read() yields the
    API-Gateway-proxy response (statusCode + JSON body string)."""
    if isinstance(body, str) or body is None:
        body_str = body if body is not None else ""
    else:
        body_str = json.dumps(body)
    envelope = {"statusCode": status, "body": body_str}
    return {"Payload": io.BytesIO(json.dumps(envelope).encode("utf-8"))}


def _patch_client(invoke_return=None, invoke_side_effect=None):
    """Patch the shared lambda client so no real boto3 call is made."""
    client = MagicMock()
    if invoke_side_effect is not None:
        client.invoke.side_effect = invoke_side_effect
    else:
        client.invoke.return_value = invoke_return
    return patch("app.config.payer_knowledge._get_lambda_client", return_value=client), client


# ── Model rendering ────────────────────────────────────────────────────────


class TestRendering:
    def test_renders_press_and_action_steps_as_numbered_text(self):
        path = PayerIvrPath.model_validate(_ROW)
        text = path.as_navigation_text()
        lines = text.splitlines()
        assert lines == [
            "1. Provider services — press 3",
            "2. Claims — press 1",
            "3. NPI prompt — key in the provider NPI",
            "4. Claim number prompt — key in the claim number",
        ]

    def test_unknown_action_uses_fallback_phrase(self):
        path = PayerIvrPath.model_validate(
            {"ivr_path_claims": [{"step": 1, "action": "enter_mystery", "label": "X"}]}
        )
        assert path.as_navigation_text() == "1. X — key in the requested value"

    def test_malformed_steps_are_skipped(self):
        # A step with neither press nor action renders nothing.
        path = PayerIvrPath.model_validate(
            {"ivr_path_claims": [{"label": "noise"}, {"press": "9", "label": "ok"}]}
        )
        assert path.as_navigation_text() == "1. ok — press 9"

    def test_step_number_falls_back_to_position_when_absent(self):
        path = PayerIvrPath.model_validate({"ivr_path_claims": [{"press": "1", "label": "a"}]})
        assert path.as_navigation_text() == "1. a — press 1"

    def test_null_ivr_path_claims_coerces_to_empty(self):
        path = PayerIvrPath.model_validate({"ivr_path_claims": None})
        assert path.ivr_path_claims == []
        assert path.as_navigation_text() == ""

    def test_blank_label_omitted_from_line(self):
        step = IvrStep(press="2", label="")
        path = PayerIvrPath(ivr_path_claims=[step])
        assert path.as_navigation_text() == "1. press 2"

    def test_renders_wait_ms_instruction(self):
        path = PayerIvrPath.model_validate(
            {"ivr_path_claims": [{"step": 1, "press": "3", "wait_ms": 2000}]}
        )
        assert path.as_navigation_text() == "1. press 3; wait 2.0s for the next IVR prompt"

    def test_renders_wait_seconds_instruction(self):
        path = PayerIvrPath.model_validate(
            {"ivr_path_claims": [{"step": 1, "press": "3", "waitSeconds": 1.5}]}
        )
        assert path.as_navigation_text() == "1. press 3; wait 1.5s for the next IVR prompt"

    def test_renders_wait_for_instruction(self):
        path = PayerIvrPath.model_validate(
            {"ivr_path_claims": [{"step": 1, "press": "3", "waitFor": "claims menu"}]}
        )
        assert path.as_navigation_text() == "1. press 3; wait for claims menu"


# ── Loader: success ──────────────────────────────────────────────────────────


class TestLoaderSuccess:
    @pytest.mark.asyncio
    async def test_returns_rendered_text_on_200(self):
        p, client = _patch_client(_invoke_response(body=_ROW))
        with p:
            text = await load_payer_ivr_path("united-healthcare", _settings())
        assert text is not None
        assert "1. Provider services — press 3" in text
        assert "key in the provider NPI" in text

    @pytest.mark.asyncio
    async def test_payer_id_is_url_encoded_into_the_path(self):
        # A human payer name with a space (today's payer_name lookup) is
        # encoded into /api/payers/:id.
        p, client = _patch_client(_invoke_response(body=_ROW))
        with p:
            await load_payer_ivr_path("United Healthcare", _settings())
        sent = json.loads(client.invoke.call_args.kwargs["Payload"])
        assert sent["path"] == "/api/payers/United%20Healthcare"
        assert sent["httpMethod"] == "GET"


# ── Loader: fall-back-to-None paths ──────────────────────────────────────────


class TestLoaderFallsBack:
    @pytest.mark.asyncio
    async def test_returns_none_on_404(self):
        p, _ = _patch_client(_invoke_response(status=404, body={"detail": "not found"}))
        with p:
            assert await load_payer_ivr_path("nope", _settings()) is None

    @pytest.mark.asyncio
    async def test_returns_none_on_non_200(self):
        p, _ = _patch_client(_invoke_response(status=500, body={"detail": "boom"}))
        with p:
            assert await load_payer_ivr_path("uhc", _settings()) is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_path(self):
        p, _ = _patch_client(_invoke_response(body={"ivr_path_claims": []}))
        with p:
            assert await load_payer_ivr_path("uhc", _settings()) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_steps_malformed(self):
        p, _ = _patch_client(_invoke_response(body={"ivr_path_claims": [{"label": "x"}]}))
        with p:
            assert await load_payer_ivr_path("uhc", _settings()) is None

    @pytest.mark.asyncio
    async def test_returns_none_on_malformed_body_json(self):
        p, _ = _patch_client(_invoke_response(body="{not json"))
        with p:
            assert await load_payer_ivr_path("uhc", _settings()) is None

    @pytest.mark.asyncio
    async def test_returns_none_on_validation_error(self):
        # ivr_path_claims is not a list → validation fails → None.
        p, _ = _patch_client(_invoke_response(body={"ivr_path_claims": "oops"}))
        with p:
            assert await load_payer_ivr_path("uhc", _settings()) is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invoke_error(self):
        p, _ = _patch_client(invoke_side_effect=RuntimeError("connection closed"))
        with p:
            assert await load_payer_ivr_path("uhc", _settings()) is None

    @pytest.mark.asyncio
    async def test_returns_none_on_unreadable_envelope(self):
        bad = MagicMock()
        bad.__getitem__ = MagicMock(side_effect=KeyError("Payload"))
        p, _ = _patch_client(bad)
        with p:
            assert await load_payer_ivr_path("uhc", _settings()) is None
