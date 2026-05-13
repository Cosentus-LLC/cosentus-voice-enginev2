"""Voice API stub Lambda for staging.

Stands in for the production ``cosentus-voice-api`` Lambda so the
staging engine can run end-to-end calls without touching prod Aurora.
Every operation the engine invokes either returns a hardcoded sensible
default or accepts the write and discards it.

Routing
-------

The engine emits API-Gateway-proxy events with ``httpMethod`` and
``path``. This stub recognises three paths:

* ``GET  /api/agents/<id>/runtime-config`` → returns a minimal valid
  ``AgentConfig`` JSON envelope. Agent fields are static (Haiku model,
  short system prompt, ``press_digit`` + ``end_call`` tools) — enough
  to exercise the full call lifecycle without persistence.
* ``POST /api/calls`` → 200 with empty body. The engine's
  ``write_call_record`` is fire-and-forget on the engine side; the
  stub discards the payload, no Aurora touch.
* ``POST /api/auto-actions`` → 200 with a token success payload
  (``actions_taken=0``, ``cost=0``, ``quality_score=0``). The engine's
  ``trigger_auto_actions`` logs the response shape but doesn't act on
  it; the token keys ensure no KeyError on the engine side if it ever
  decides to.

Unknown paths return ``200`` with a ``warning: stub-ignored`` body —
loud enough to grep, quiet enough to not poison load-test metrics.

Why pure stdlib
---------------

The Lambda has no dependencies beyond the Python stdlib. ``json`` and
``re`` cover the entire surface. CDK deploys the file directly with
no packaging step; the deploy artifact is just this file. If we ever
need to mock more behaviour (latency injection, error injection),
add it inline here.

Logging
-------

Lambda's Python runtime auto-streams ``stdout`` to CloudWatch. Each
log line is a JSON object so CloudWatch Logs Insights can query
``op``, ``agent_id_or_name``, ``ms``, etc. directly.

Decommissioning
---------------

When the real staging cosentus-voice-api Lambda lands (Phase 6,
post-cutover), destroy the ``VoiceApiStubStack`` CDK stack and
repoint staging's ``VOICE_API_LAMBDA_NAME`` env var at the real one.
The engine code requires no change.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any


_RUNTIME_CONFIG_RE = re.compile(r"^/api/agents/(?P<id>[^/]+)/runtime-config$")
_PATH_CALLS = "/api/calls"
_PATH_AUTO_ACTIONS = "/api/auto-actions"


# ── AgentConfig response ────────────────────────────────────────────────────


def _build_agent_config(agent_id_or_name: str) -> dict[str, Any]:
    """Hardcoded staging-stub agent.

    The shape mirrors :class:`app.config.agent_config.AgentConfig`.
    Only required field is ``name`` — everything else uses sensible
    defaults the engine factory expects.

    The agent has two tools (``press_digit`` + ``end_call``) so the
    Wave 6 load harness can exercise the tool-dispatch path. Haiku
    model keeps Bedrock cost trivial. The system prompt is brief and
    the first message acknowledges the test framing.
    """
    return {
        "name": agent_id_or_name,
        "display_name": "Staging Stub Agent",
        "description": (
            "Cosentus voice-engine staging stub. Lives only in the staging "
            "stub Lambda. NOT a real Aurora row — anything you do on this "
            "call is ephemeral."
        ),
        "system_prompt": (
            "You are a friendly test assistant for the Cosentus voice "
            "engine. Keep responses brief — one or two sentences. If "
            "the caller asks you to press a digit, use the press_digit "
            "tool. If they ask you to end the call, use the end_call tool."
        ),
        "first_message": (
            "Hi, you've reached the staging stub agent. This call won't be "
            "saved anywhere. What can I help you with?"
        ),
        "ivr_goal": "",
        "speak_first": True,
        "llm": {
            "model": "claude-haiku-4-5",
            "max_tokens": 200,
            "temperature": 0.7,
        },
        "tts": {
            "voice_id": "",
            "model": "eleven_turbo_v2_5",
            "settings": {},
        },
        "stt": {"keywords": []},
        "tools": [
            {
                "type": "press_digit",
                "description": "Press a DTMF digit on the active call.",
                "settings": {},
            },
            {
                "type": "end_call",
                "description": "End the active call.",
                "settings": {},
            },
        ],
        "post_call_analyses": None,
        "_meta": {"agent_id": agent_id_or_name, "version": 0},
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ok(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """API-Gateway-proxy 200 response with JSON body."""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body or {}),
    }


def _log(op: str, **fields: Any) -> None:
    """One JSON line per request — CloudWatch Logs ingests structured."""
    entry: dict[str, Any] = {"op": op, **fields}
    print(json.dumps(entry, separators=(",", ":")))


# ── Entry point ─────────────────────────────────────────────────────────────


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """API-Gateway-proxy-style entry point.

    Route on ``event["path"]`` (not ``httpMethod`` — same path is
    only ever used with one method by the engine, so path-routing is
    enough).
    """
    started = time.time()
    path = event.get("path", "") or ""
    method = event.get("httpMethod", "") or ""

    # runtime-config GET
    match = _RUNTIME_CONFIG_RE.match(path)
    if match:
        agent_id_or_name = match.group("id")
        body = _build_agent_config(agent_id_or_name)
        _log(
            "GetRuntimeConfig",
            agent_id_or_name=agent_id_or_name,
            method=method,
            ms=int((time.time() - started) * 1000),
        )
        return _ok(body)

    # call-record write
    if path == _PATH_CALLS:
        # The engine sends record metadata in event["body"] as a JSON
        # string. We don't need to inspect it for the stub, but we
        # capture the call_id for log searchability.
        call_id = _extract_call_id(event.get("body"))
        _log(
            "WriteCallRecord",
            call_id=call_id,
            method=method,
            ms=int((time.time() - started) * 1000),
        )
        return _ok()

    # auto-actions derived writes
    if path == _PATH_AUTO_ACTIONS:
        call_id = _extract_call_id(event.get("body"))
        _log(
            "WriteAutoActions",
            call_id=call_id,
            method=method,
            ms=int((time.time() - started) * 1000),
        )
        # Token success payload so the engine's optional inspection
        # of the response body finds the expected keys.
        return _ok({"actions_taken": 0, "cost": 0, "quality_score": 0, "actions": []})

    # Unknown path: 200 + warning, not 404. We don't want load-test
    # harnesses to see false-positive 4xx noise.
    _log(
        "UnknownPath",
        path=path,
        method=method,
        ms=int((time.time() - started) * 1000),
    )
    return _ok({"warning": "stub-ignored", "path": path, "method": method})


def _extract_call_id(body: Any) -> str:
    """Best-effort ``call_id`` extraction from a JSON string body."""
    if not body:
        return ""
    if not isinstance(body, str):
        return ""
    try:
        parsed = json.loads(body)
    except (ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    # The engine sends call_id directly (auto-actions) OR id
    # (write_call_record uses CallRecord.id).
    return str(parsed.get("call_id") or parsed.get("id") or "")
