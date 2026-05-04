"""Tests for ``app/persistence/call_writer.py``.

The lambda is mocked at the boto3 ``invoke`` level so tests are pure
(no AWS calls). We assert the wire envelope shape, the success /
failure boolean, and the never-raise contract.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from app.config.settings import Settings
from app.persistence.call_record import CallRecord
from app.persistence.call_writer import (
    trigger_auto_actions,
    write_call_record,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def _settings() -> Settings:
    """Build a Settings without reading dotenv. Required env vars only."""
    return Settings(
        voice_api_lambda_name="test-voice-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
    )


def _record() -> CallRecord:
    return CallRecord(
        id="11111111-1111-1111-1111-111111111111",
        agent_name="test-agent",
        agent_display_name="Test Agent",
        from_number="+19494360836",
        target_number="+12098075018",
        direction="inbound",
        status="completed",
        started_at=datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 5, 4, 12, 5, 30, tzinfo=UTC),
        duration_secs=330,
        case_data={},
        transcript=[{"turn_number": 1, "speaker": "user", "content": "hi"}],
        session_id="daily-room-abc",
    )


def _ok_response(status_code: int = 201, body: str = '{"id":"abc"}') -> dict:
    """Synthesize a boto3 ``invoke`` response with a 2xx envelope."""
    envelope = json.dumps({"statusCode": status_code, "body": body}).encode("utf-8")
    return {"Payload": io.BytesIO(envelope)}


def _error_response(status_code: int = 500, body: str = '{"detail":"boom"}') -> dict:
    envelope = json.dumps({"statusCode": status_code, "body": body}).encode("utf-8")
    return {"Payload": io.BytesIO(envelope)}


# ── write_call_record success path ────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_returns_true_on_2xx():
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=_ok_response(201))
        ok = await write_call_record(_record(), _settings())
    assert ok is True


@pytest.mark.asyncio
async def test_write_returns_true_on_200_too():
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=_ok_response(200))
        ok = await write_call_record(_record(), _settings())
    assert ok is True


@pytest.mark.asyncio
async def test_write_uses_layer1_lambda_client():
    """The writer reuses Layer 1's _LAMBDA_CLIENT — assert via patch."""
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=_ok_response())
        await write_call_record(_record(), _settings())
        mock_lambda.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_envelope_shape_matches_lambda_contract():
    """Verify the API-Gateway-proxy envelope structure."""
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _ok_response()

    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(side_effect=capture)
        await write_call_record(_record(), _settings())

    assert captured["FunctionName"] == "test-voice-api"
    assert captured["InvocationType"] == "RequestResponse"

    payload = json.loads(captured["Payload"].decode("utf-8"))
    assert payload["httpMethod"] == "POST"
    assert payload["path"] == "/api/calls"
    assert payload["headers"]["Content-Type"] == "application/json"

    body = json.loads(payload["body"])
    assert body["id"] == "11111111-1111-1111-1111-111111111111"
    assert body["status"] == "completed"
    assert body["session_id"] == "daily-room-abc"


# ── write_call_record failure paths ───────────────────────────────────────


@pytest.mark.asyncio
async def test_write_returns_false_on_non_2xx():
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=_error_response(500))
        ok = await write_call_record(_record(), _settings())
    assert ok is False


@pytest.mark.asyncio
async def test_write_returns_false_on_invoke_exception():
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(side_effect=RuntimeError("connection reset"))
        ok = await write_call_record(_record(), _settings())
    assert ok is False


@pytest.mark.asyncio
async def test_write_returns_false_on_no_payload():
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value={})
        ok = await write_call_record(_record(), _settings())
    assert ok is False


@pytest.mark.asyncio
async def test_write_returns_false_on_malformed_envelope():
    """Lambda returned bytes that aren't JSON."""
    bad_envelope = {"Payload": io.BytesIO(b"not json {{{")}
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=bad_envelope)
        ok = await write_call_record(_record(), _settings())
    assert ok is False


@pytest.mark.asyncio
async def test_write_returns_false_when_envelope_is_a_list():
    """Some lambda misconfigurations return a non-dict JSON body."""
    bad_envelope = {"Payload": io.BytesIO(b'["not", "a", "dict"]')}
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=bad_envelope)
        ok = await write_call_record(_record(), _settings())
    assert ok is False


@pytest.mark.asyncio
async def test_write_never_raises_on_arbitrary_exceptions():
    """Anything bizarre at the boto3 layer should still resolve to False."""
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(side_effect=KeyboardInterrupt("simulated"))
        # KeyboardInterrupt is BaseException, NOT Exception — should propagate.
        with pytest.raises(KeyboardInterrupt):
            await write_call_record(_record(), _settings())


@pytest.mark.asyncio
async def test_write_swallows_exception_subclass():
    """Standard Exception subclasses → log + return False, never raise."""

    class CustomBoto3Error(Exception):
        pass

    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(side_effect=CustomBoto3Error("regional outage"))
        ok = await write_call_record(_record(), _settings())
    assert ok is False


# ── trigger_auto_actions ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_actions_returns_parsed_body_on_2xx():
    body_json = {"actions_taken": 3, "cost": "0.0024", "quality_score": 80}
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=_ok_response(200, body=json.dumps(body_json)))
        result = await trigger_auto_actions("call-id-1", _settings())
    assert result is not None
    assert result["actions_taken"] == 3
    assert result["cost"] == "0.0024"


@pytest.mark.asyncio
async def test_auto_actions_returns_none_on_non_2xx():
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=_error_response(404))
        result = await trigger_auto_actions("call-id-1", _settings())
    assert result is None


@pytest.mark.asyncio
async def test_auto_actions_returns_none_on_invoke_exception():
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(side_effect=RuntimeError("boom"))
        result = await trigger_auto_actions("call-id-1", _settings())
    assert result is None


@pytest.mark.asyncio
async def test_auto_actions_envelope_shape():
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _ok_response(200, body="{}")

    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(side_effect=capture)
        await trigger_auto_actions("call-id-xyz", _settings())

    payload = json.loads(captured["Payload"].decode("utf-8"))
    assert payload["path"] == "/api/auto-actions"
    body = json.loads(payload["body"])
    assert body == {"call_id": "call-id-xyz"}


@pytest.mark.asyncio
async def test_auto_actions_returns_none_on_unparseable_body():
    """A 2xx with non-JSON body still falls back to None."""
    bad_body = "not json {{{"
    with patch("app.persistence.call_writer._LAMBDA_CLIENT") as mock_lambda:
        mock_lambda.invoke = MagicMock(return_value=_ok_response(200, body=bad_body))
        result = await trigger_auto_actions("call-id-1", _settings())
    assert result is None
