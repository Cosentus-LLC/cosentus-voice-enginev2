"""Layer 6 — call persistence via lambda invoke.

End-of-call writes only. Best-effort, never-raise. Reuses Layer 1's
:data:`~app.config.agent_config._LAMBDA_CLIENT` for timeouts +
adaptive retries, closing v1's bare-boto3-client anti-pattern.

Two operations:

* :func:`write_call_record` — UPSERT a :class:`CallRecord` to
  ``voice_calls`` via ``POST /api/calls``. Idempotent on the lambda
  side (``ON CONFLICT (id) DO UPDATE``); typical flow calls this
  twice — once with empty ``post_call_analyses``, once after the
  Bedrock extraction completes.
* :func:`trigger_auto_actions` — kick off the lambda's derived-write
  pipeline (``voice_call_costs``, ``voice_call_scores``,
  ``voice_auto_actions``). The cost + score upserts are idempotent;
  ``voice_auto_actions`` inserts are NOT, so callers must invoke
  this AT MOST ONCE per terminal call.

Failure semantics
-----------------

A persistence failure must never propagate up and fail the call from
the caller's perspective. Both functions catch every exception, log
structured, and return ``False`` / ``None``. The recording webhook
(separate lambda code path) will still patch ``recording_path``
later regardless of whether the engine's write landed; if it didn't,
the webhook 404s harmlessly and operators can replay from
CloudWatch logs if needed.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

# Reuse Layer 1's hardened client. Single module-level boto3
# instance, shared across all invokes; explicit connect/read
# timeouts; adaptive retries. See agent_config.py for the AWS-doc
# rationale.
from app.config.agent_config import _LAMBDA_CLIENT
from app.config.settings import Settings
from app.persistence.call_record import CallRecord

logger = structlog.get_logger(__name__)

# Lambda routes by ``path`` in the API-Gateway-proxy envelope. Layer 1
# uses the same envelope shape for runtime-config GETs. Defining the
# routes as constants keeps the wire surface searchable and makes
# accidental drift visible.
_PATH_CALLS = "/api/calls"
_PATH_AUTO_ACTIONS = "/api/auto-actions"


async def write_call_record(record: CallRecord, settings: Settings) -> bool:
    """Upsert a :class:`CallRecord` to ``voice_calls`` via the lambda.

    Args:
        record: The end-of-call snapshot to persist.
        settings: Layer 2 settings — provides the lambda function name.

    Returns:
        ``True`` on a 2xx response, ``False`` on every failure path.
        **Never raises**: a persistence failure logs structured and
        propagates only via the boolean return.

    Idempotency:
        The lambda's ``POST /api/calls`` is ``INSERT … ON CONFLICT (id)
        DO UPDATE``. Safe to call N times for the same record. Typical
        usage is two writes per call — first with empty
        ``post_call_analyses``, second after Bedrock extraction lands.
        The lambda's UPSERT preserves the first-write transcript when
        the second write sends ``[]`` (empty-array guard); callers
        relying on this should not blank the transcript on the second
        write.
    """
    envelope = {
        "httpMethod": "POST",
        "path": _PATH_CALLS,
        "headers": {"Content-Type": "application/json"},
        "queryStringParameters": None,
        "body": json.dumps(record.to_lambda_body()),
    }

    try:
        response = await asyncio.to_thread(
            _LAMBDA_CLIENT.invoke,
            FunctionName=settings.voice_api_lambda_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(envelope).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; never propagate
        logger.error(
            "call_record_write_invoke_failed",
            call_id=record.id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return False

    payload, ok = _read_envelope(response, call_id=record.id)
    if not ok or payload is None:
        return False

    status_code = payload.get("statusCode", 0)
    if 200 <= status_code < 300:
        logger.info(
            "call_record_written",
            call_id=record.id,
            status=record.status,
            transcript_turns=len(record.transcript or []),
            has_analyses=bool(record.post_call_analyses),
            duration_secs=record.duration_secs or 0,
        )
        return True

    body_preview = _body_preview(payload)
    logger.error(
        "call_record_write_failed",
        call_id=record.id,
        status_code=status_code,
        body_preview=body_preview,
    )
    return False


async def trigger_auto_actions(call_id: str, settings: Settings) -> dict[str, Any] | None:
    """Trigger the lambda's derived-write pipeline for a saved call.

    Lambda reads the just-written ``voice_calls`` row (status,
    transcript, ``post_call_analyses``, ``case_data``) and computes:

    * ``voice_call_costs`` — telephony / STT / LLM / TTS unit costs
    * ``voice_call_scores`` — quality scoring (5-field BOOLEAN tally)
    * ``voice_auto_actions`` — task creation, denial routing, AR-call
      logging (depends on what fields are populated in
      ``post_call_analyses``)

    Args:
        call_id: The ``voice_calls.id`` to derive from. The lambda
            looks the row up by primary key.
        settings: Layer 2 settings.

    Returns:
        Parsed response body on 2xx (``actions_taken``, ``cost``,
        ``quality_score``, ``actions``), ``None`` on any failure.
        **Never raises.**

    Idempotency:
        ``voice_call_costs`` and ``voice_call_scores`` use ``ON
        CONFLICT (call_id) DO UPDATE`` — re-running is safe for them.
        ``voice_auto_actions`` inserts are unconditional — re-running
        will create duplicate task / log rows in the downstream
        ``tasks`` and ``ar_calls`` tables. **Caller must invoke this
        at most once per terminal call.** Layer 8 / 11 schedules this
        to fire from the call-end finally block, exactly once.
    """
    envelope = {
        "httpMethod": "POST",
        "path": _PATH_AUTO_ACTIONS,
        "headers": {"Content-Type": "application/json"},
        "queryStringParameters": None,
        "body": json.dumps({"call_id": call_id}),
    }

    try:
        response = await asyncio.to_thread(
            _LAMBDA_CLIENT.invoke,
            FunctionName=settings.voice_api_lambda_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(envelope).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.error(
            "auto_actions_invoke_failed",
            call_id=call_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    payload, ok = _read_envelope(response, call_id=call_id)
    if not ok or payload is None:
        return None

    status_code = payload.get("statusCode", 0)
    if not (200 <= status_code < 300):
        logger.error(
            "auto_actions_failed",
            call_id=call_id,
            status_code=status_code,
            body_preview=_body_preview(payload),
        )
        return None

    body_text = payload.get("body", "{}") or "{}"
    try:
        body = json.loads(body_text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.error(
            "auto_actions_body_unparseable",
            call_id=call_id,
            error=str(exc),
            body_preview=str(body_text)[:200],
        )
        return None

    logger.info(
        "auto_actions_completed",
        call_id=call_id,
        actions_taken=body.get("actions_taken"),
        cost=body.get("cost"),
        quality_score=body.get("quality_score"),
    )
    return body


def _read_envelope(response: dict[str, Any], *, call_id: str) -> tuple[dict[str, Any] | None, bool]:
    """Read the boto3 ``invoke`` response and parse the API-Gateway envelope.

    Returns ``(payload_dict, ok)``. ``ok`` is ``False`` if the
    payload is missing or the JSON is malformed. The boolean lets
    callers handle "no response at all" the same as "non-2xx
    response" without conflating the cases at the log line.
    """
    raw = response.get("Payload")
    if raw is None:
        logger.error("call_writer_no_payload", call_id=call_id)
        return None, False

    try:
        envelope_bytes = raw.read()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "call_writer_payload_read_failed",
            call_id=call_id,
            error=str(exc),
        )
        return None, False

    try:
        envelope = json.loads(envelope_bytes)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error(
            "call_writer_bad_envelope",
            call_id=call_id,
            error=str(exc),
        )
        return None, False

    if not isinstance(envelope, dict):
        logger.error(
            "call_writer_envelope_not_dict",
            call_id=call_id,
            envelope_type=type(envelope).__name__,
        )
        return None, False

    return envelope, True


def _body_preview(payload: dict[str, Any]) -> str:
    """Truncate the lambda's ``body`` field for log-line safety."""
    body = payload.get("body", "")
    if not isinstance(body, str):
        return ""
    return body[:300]
