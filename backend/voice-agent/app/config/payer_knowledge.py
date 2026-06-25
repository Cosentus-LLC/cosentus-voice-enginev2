"""Verified per-payer IVR path — model + best-effort loader (#17).

The ``voice_payer_knowledge`` table holds operator-verified IVR menu
paths per payer (``ivr_path_claims``). The API exposes the raw row at
``GET /api/payers/:id`` (matchable by ``id`` or the ``payer_id`` slug),
returning ``200`` with the row or ``404 {"detail": ...}``. This module
fetches that row at call start and renders the claims path into
prompt-ready text the Flows ``navigate`` step injects, so the agent
**follows the verified map** via the existing ``press_digit`` tool
instead of navigating purely by ear.

Transport mirrors :mod:`app.config.agent_config` /
:func:`app.runner.server._lookup_inbound_agent`: a synchronous boto3
``Lambda.Invoke`` of the API-Gateway-proxy event, wrapped in
:func:`asyncio.to_thread`, reusing Layer 1's hardened, module-shared
client (timeouts + adaptive retry). It is **NOT** part of the
``runtime-config`` contract — it's a separate fetch, so no contract
change is involved.

Best-effort by design
---------------------

Unlike :func:`app.config.agent_config.load_agent_config` (which fails
the call on error), IVR navigation is *advisory*: a missing or
unreachable path simply means the agent navigates by ear, exactly as it
does today. So every failure path here — payer not found (the common
case until the API exposes a name lookup; see below), empty path,
malformed response, invoke error — is logged and returns ``None``; the
caller treats ``None`` as "no map, fall back to listen-and-decide". This
function never raises.

The ``payer_name`` lookup gap
-----------------------------

The dispatcher carries the payer as ``payer_name`` in ``case_data`` (a
human name, e.g. ``"United Healthcare"``), but ``GET /api/payers/:id``
matches by slug/``id``, not name. Until the API exposes a name-based
lookup, fetching by name returns ``404`` → ``None`` → by-ear fallback.
#17 ships functional-on-fallback today and goes fully live once that
endpoint lands (the API already has ``payer_name`` ILIKE internally).
We deliberately do NOT hardcode any name→slug mapping in the engine.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import quote

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config.agent_config import _get_lambda_client
from app.config.settings import Settings

logger = structlog.get_logger(__name__)

# action -> the phrase telling the agent what dynamic value to key in.
# These are the ``{step, action, label}`` step kinds (vs the static
# ``{step, press, label}`` DTMF steps). The agent keys the value in via
# the unchanged ``press_digit`` tool, reading it from its (post-
# verification, PHI-bearing) call context — we never thread the raw
# value through here.
_ACTION_PHRASES: dict[str, str] = {
    "enter_npi": "key in the provider NPI",
    "enter_claim": "key in the claim number",
    "enter_tax_id": "key in the provider tax ID",
}
_ACTION_FALLBACK_PHRASE = "key in the requested value"


class IvrStep(BaseModel):
    """One step in a verified IVR path.

    Two shapes share this model (``extra='ignore'`` drops any other
    columns the row carries):

    * **press** — ``{step, press, label}``: a static DTMF key to press.
    * **action** — ``{step, action, label}`` where ``action`` is one of
      :data:`_ACTION_PHRASES`: a dynamic value the agent keys in.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    step: int | None = None
    label: str = ""
    press: str = ""
    action: str = ""
    wait_ms: int | None = Field(default=None, alias="waitMs")
    wait_seconds: float | None = Field(default=None, alias="waitSeconds")
    wait_for: str = Field(default="", alias="waitFor")


class PayerIvrPath(BaseModel):
    """The slice of a ``voice_payer_knowledge`` row #17 reads.

    Only ``ivr_path_claims`` is modeled; ``extra='ignore'`` drops the
    rest of the row. A ``null`` JSON value for the list is coerced to an
    empty list so a payer row with no verified path validates cleanly
    (and renders to ``""`` → caller falls back).
    """

    model_config = ConfigDict(extra="ignore")

    ivr_path_claims: list[IvrStep] = Field(default_factory=list)

    @field_validator("ivr_path_claims", mode="before")
    @classmethod
    def _coerce_null_to_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    def as_navigation_text(self) -> str:
        """Render the claims path as numbered, prompt-ready lines.

        Each line states the menu label and the action: ``press N`` for
        DTMF steps, or the dynamic-entry phrase for ``action`` steps.
        Malformed steps (neither ``press`` nor a recognizable ``action``)
        are skipped. Returns ``""`` when nothing renders.
        """
        lines: list[str] = []
        for step in self.ivr_path_claims:
            verb = _step_verb(step)
            if verb is None:
                continue
            # Prefer the row's explicit step number; otherwise number by
            # rendered-line position so skipped/malformed steps don't leave
            # gaps in the displayed sequence.
            number = step.step if isinstance(step.step, int) and step.step > 0 else len(lines) + 1
            label = step.label.strip()
            wait = _step_wait_instruction(step)
            action = f"{verb}; {wait}" if wait else verb
            lines.append(f"{number}. {label} — {action}" if label else f"{number}. {action}")
        return "\n".join(lines)


def _step_verb(step: IvrStep) -> str | None:
    """The instruction verb for a step, or ``None`` if it is malformed."""
    press = step.press.strip()
    if press:
        return f"press {press}"
    action = step.action.strip()
    if action:
        return _ACTION_PHRASES.get(action, _ACTION_FALLBACK_PHRASE)
    return None


def _step_wait_instruction(step: IvrStep) -> str:
    """Render optional per-step wait guidance, preserving blank-by-default rows."""
    wait_for = step.wait_for.strip()
    if wait_for:
        return f"wait for {wait_for}"
    if isinstance(step.wait_ms, int) and step.wait_ms > 0:
        return f"wait {step.wait_ms / 1000:.1f}s for the next IVR prompt"
    if isinstance(step.wait_seconds, (int, float)) and step.wait_seconds > 0:
        return f"wait {float(step.wait_seconds):.1f}s for the next IVR prompt"
    return ""


def _build_proxy_event(payer_id: str) -> bytes:
    """API-Gateway-proxy event for ``GET /api/payers/:id``.

    ``payer_id`` is URL-encoded into the path (it may be a human payer
    name with spaces until the name-lookup endpoint lands).
    """
    payload: dict[str, Any] = {
        "httpMethod": "GET",
        "path": f"/api/payers/{quote(payer_id, safe='')}",
        "headers": {},
        "queryStringParameters": None,
        "body": None,
    }
    return json.dumps(payload).encode("utf-8")


async def load_payer_ivr_path(payer_id: str, settings: Settings) -> str | None:
    """Fetch a payer's verified claims IVR path as prompt-ready text.

    Args:
        payer_id: The payer identifier from ``case_data`` (today a
            ``payer_name``; see the module docstring's lookup-gap note).
            Passed verbatim into the ``/api/payers/:id`` path.
        settings: Provides ``voice_api_lambda_name`` + region (via the
            shared Layer-1 client).

    Returns:
        The rendered navigation text, or ``None`` on **any** failure or
        empty path. Never raises — IVR navigation is best-effort and a
        ``None`` simply means the agent navigates by ear.
    """
    client = _get_lambda_client(settings)
    payload = _build_proxy_event(payer_id)
    try:
        resp = await asyncio.to_thread(
            client.invoke,
            FunctionName=settings.voice_api_lambda_name,
            InvocationType="RequestResponse",
            Payload=payload,
        )
    except Exception as exc:  # noqa: BLE001 — log + fall back, never fail the call
        logger.warning(
            "payer_ivr_lookup_invoke_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    try:
        outer = json.loads(resp["Payload"].read())
    except Exception as exc:  # noqa: BLE001
        logger.warning("payer_ivr_lookup_bad_envelope", error=str(exc))
        return None

    status = outer.get("statusCode", 0)
    if status != 200:
        # 404 (payer not found / no name lookup yet) is the expected
        # common case today — info, not warning, so it doesn't read as a
        # fault. Other non-200s are surfaced louder.
        emit = logger.info if status == 404 else logger.warning
        emit("payer_ivr_lookup_non_200", status_code=status)
        return None

    body_text = outer.get("body", "{}") or "{}"
    try:
        body = json.loads(body_text)
    except (ValueError, json.JSONDecodeError):
        logger.warning("payer_ivr_lookup_bad_body")
        return None

    try:
        parsed = PayerIvrPath.model_validate(body)
    except ValidationError as exc:
        logger.warning("payer_ivr_lookup_validation_error", error=str(exc))
        return None

    text = parsed.as_navigation_text()
    if not text:
        logger.info("payer_ivr_lookup_empty_path")
        return None

    logger.info("payer_ivr_loaded", step_count=len(parsed.ivr_path_claims))
    return text
