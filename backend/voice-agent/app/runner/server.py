"""HTTP server (aiohttp) + graceful drain.

Five routes, all per-process:

* ``GET /health`` — liveness probe. Always 200 unless the process
  is broken. ECS uses this to decide whether to restart the
  container.
* ``GET /ready`` — readiness probe. 503 when draining or at
  capacity; 200 otherwise. NLB uses this to decide whether to
  route new calls to this task.
* ``GET /status`` — detailed manager state. Auth required (the
  output reveals operational details).
* ``POST /start`` — outbound or browser call spawn. Auth required.
  ``body.direction`` switches between the two.
* ``POST /daily-dialin-webhook`` — inbound PSTN call from Daily's
  SIP gateway. Verifies Daily's pinless dial-in HMAC signature
  before any lookup or call spawn.

:func:`graceful_drain` is the SIGTERM-driven shutdown coroutine.
Sets the manager's draining flag, polls active sessions for up to
110 seconds, cancels survivors (only the engine's spawned tasks —
not all asyncio tasks, which would kill the HTTP accept loop), and
releases ECS task protection.

aiohttp choice rationale (vs FastAPI): single event loop, lower
overhead, v1 + AWS sample production-proven, no FastAPI →
starlette → uvicorn dependency chain for what's effectively five
endpoints. ``fastapi`` is already a transitive dep (Layer 8 needs
it for ``pipecat.runner.types``) but we don't import it directly
here.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import json
import secrets
import time
from typing import TYPE_CHECKING, Any

import boto3
import structlog
from aiohttp import web

from app.config.settings import Settings
from app.runner.manager import CapacityRejected, PipelineManager
from app.runner.protection import TaskProtection

if TYPE_CHECKING:
    # MetricsEmitter is optional in graceful_drain (callable from
    # contexts that don't have one, e.g. tests). Import only for the
    # type hint to avoid pulling boto3-client construction into the
    # module import path.
    from app.runner.metrics import MetricsEmitter

# Typed AppKey instances. aiohttp 3.9+ recommends these for app[]
# storage to avoid the ``NotAppKeyWarning`` and to give type-checkers
# something to bite. Each key is module-scoped so route handlers
# import them implicitly via the same module.
APP_KEY_MANAGER: web.AppKey[PipelineManager] = web.AppKey("manager", PipelineManager)
APP_KEY_SETTINGS: web.AppKey[Settings] = web.AppKey("settings", Settings)
APP_KEY_API_KEY: web.AppKey[str] = web.AppKey("api_key", str)

logger = structlog.get_logger(__name__)


# Drain budget in seconds. Fargate's default ``stopTimeout`` is
# 120 s before SIGKILL; we leave a 10 s buffer for the post-drain
# protection release + session cleanup. Layer 11 may bump
# ``stopTimeout`` to 600 s for longer Cosentus IVR sessions; this
# default tracks the conservative 120 s case.
DEFAULT_DRAIN_BUDGET_SECS = 110

# Polling interval inside the drain coroutine.
_DRAIN_POLL_SECS = 5

# Window after cancelling survivor tasks for their finally blocks
# to fire (Layer 8's ``finalize_call`` must run to land a partial
# CallRecord). 2 s covers normal shutdown; longer would risk
# Fargate killing us before the drain returns.
_POST_CANCEL_GRACE_SECS = 2

# Daily pinless dial-in signs the room_creation_api webhook with
# HMAC-SHA256 over ``timestamp + "." + raw_json_body``.
_DAILY_DIALIN_TIMESTAMP_HEADER = "X-Pinless-Timestamp"
_DAILY_DIALIN_SIGNATURE_HEADER = "X-Pinless-Signature"
_DAILY_DIALIN_SIGNATURE_TOLERANCE_SECS = 300
_LOCAL_DIALIN_SIGNATURE_BYPASS_ENVS = {"local", "dev", "development", "test"}


# ── App factory ────────────────────────────────────────────────────────────


async def build_app(settings: Settings, manager: PipelineManager) -> web.Application:
    """Construct the aiohttp application and wire routes.

    Stores ``manager``, ``settings``, and the resolved API key on
    ``app[...]`` so route handlers can reach them via the request's
    app reference. No globals.
    """
    app = web.Application()
    app[APP_KEY_MANAGER] = manager
    app[APP_KEY_SETTINGS] = settings
    app[APP_KEY_API_KEY] = await _load_api_key(settings)

    app.router.add_get("/health", handle_health)
    app.router.add_get("/ready", handle_ready)
    app.router.add_get("/status", handle_status)
    app.router.add_post("/start", handle_start)
    app.router.add_post("/daily-dialin-webhook", handle_dialin_webhook)

    logger.info(
        "app_built",
        routes=["/health", "/ready", "/status", "/start", "/daily-dialin-webhook"],
        api_key_configured=bool(app[APP_KEY_API_KEY]),
    )
    return app


# ── Route handlers ────────────────────────────────────────────────────────


async def handle_health(request: web.Request) -> web.Response:
    """Liveness. Returns 200 while the process is running."""
    return web.json_response({"status": "healthy"})


async def handle_ready(request: web.Request) -> web.Response:
    """Readiness. 503 when draining or at capacity; 200 otherwise.

    Separate from ``/health`` so ECS doesn't kill at-capacity
    containers. NLB stops routing when this returns 503; ECS keeps
    the task running for in-progress calls.
    """
    manager: PipelineManager = request.app[APP_KEY_MANAGER]
    if manager.is_draining:
        return web.json_response(
            {"status": "draining", **manager.get_status()},
            status=503,
        )
    if manager.at_capacity:
        return web.json_response(
            {"status": "at_capacity", **manager.get_status()},
            status=503,
        )
    return web.json_response({"status": "ready", **manager.get_status()})


async def handle_status(request: web.Request) -> web.Response:
    """Detailed status. Auth required — output reveals operational state."""
    if not _check_api_key(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    manager: PipelineManager = request.app[APP_KEY_MANAGER]
    return web.json_response(manager.get_status())


async def handle_start(request: web.Request) -> web.Response:
    """Spawn an outbound or browser call. Auth required.

    Returns 202 Accepted immediately after ``asyncio.create_task``.
    The caller polls ``/status`` if it needs to know whether the
    call is live; the engine doesn't wait for the bot to join Daily
    before responding.
    """
    if not _check_api_key(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    manager: PipelineManager = request.app[APP_KEY_MANAGER]

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid_json"}, status=400)

    direction = body.get("direction", "outbound")
    agent_id = body.get("agent_id")
    if not agent_id:
        return web.json_response(
            {"error": "agent_id_required"},
            status=400,
        )

    try:
        if direction == "outbound":
            target_number = body.get("target_number")
            from_number = body.get("from_number")
            if not target_number or not from_number:
                return web.json_response(
                    {"error": "target_number_and_from_number_required"},
                    status=400,
                )
            result = await manager.start_outbound(
                agent_id=agent_id,
                target_number=target_number,
                from_number=from_number,
                case_data=body.get("case_data") or {},
                batch_id=body.get("batch_id"),
                batch_row_index=body.get("batch_row_index"),
            )
        elif direction == "browser":
            result = await manager.start_browser(
                agent_id=agent_id,
                case_data=body.get("case_data"),
            )
        else:
            return web.json_response(
                {"error": f"unknown_direction:{direction}"},
                status=400,
            )
    except CapacityRejected as exc:
        return web.json_response(
            {"status": "rejected", "reason": exc.reason},
            status=503,
        )

    response: dict[str, Any] = {
        "call_id": result.call_id,
        "room_name": result.room_name,
        "room_url": result.room_url,
        "status": "started",
    }
    if result.viewer_token is not None:
        response["viewer_token"] = result.viewer_token

    return web.json_response(response, status=202)


async def handle_dialin_webhook(request: web.Request) -> web.Response:
    """Daily dial-in webhook. Verifies Daily's pinless HMAC first.

    Returns the room URL and session ID Daily expects in its
    documented response shape — Daily's SIP gateway uses this to
    bridge the caller into the room.
    """
    manager: PipelineManager = request.app[APP_KEY_MANAGER]
    settings: Settings = request.app[APP_KEY_SETTINGS]
    raw_body = await request.read()

    if not _verify_daily_dialin_signature(request, raw_body, settings):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return web.json_response({"error": "invalid_json"}, status=400)

    from_number = body.get("From", "")
    to_number = body.get("To", "")
    call_id_external = body.get("callId", "")
    call_domain = body.get("callDomain", "")

    if not to_number:
        return web.json_response({"error": "to_required"}, status=400)

    # Phone-number → agent_id lookup via the API lambda.
    agent_id = await _lookup_inbound_agent(to_number, settings)
    if not agent_id:
        logger.warning(
            "dialin_no_agent_configured",
            to_number=to_number,
            from_number=from_number,
        )
        return web.json_response(
            {"error": "no_agent_configured", "to": to_number},
            status=503,
        )

    try:
        result = await manager.start_inbound(
            agent_id=agent_id,
            from_number=from_number,
            to_number=to_number,
            call_id_external=call_id_external,
            call_domain=call_domain,
        )
    except CapacityRejected as exc:
        return web.json_response(
            {"error": "at_capacity", "reason": exc.reason},
            status=503,
        )

    return web.json_response(
        {
            "dailyRoom": result.room_url,
            "sessionId": result.call_id,
        }
    )


# ── Auth helpers ──────────────────────────────────────────────────────────


def _check_api_key(request: web.Request) -> bool:
    """Constant-time API key check. Returns ``False`` to short-circuit.

    Empty configured key (local dev) means no auth — return ``True``
    so dev environments don't have to set the header. Production
    sets ``api_key_secret_arn``, the resolved secret is non-empty,
    and the header check uses ``secrets.compare_digest``.
    """
    expected = request.app.get(APP_KEY_API_KEY, "")
    if not expected:
        return True  # local dev — no auth
    provided = request.headers.get("X-API-Key", "")
    return secrets.compare_digest(provided, expected)


def _verify_daily_dialin_signature(
    request: web.Request,
    raw_body: bytes,
    settings: Settings,
    *,
    now: float | None = None,
) -> bool:
    """Verify Daily pinless dial-in HMAC before processing a call.

    Daily provides a Base64-encoded ``pinless_dialin.hmac`` secret and
    sends ``X-Pinless-Timestamp`` plus ``X-Pinless-Signature`` on each
    room-creation webhook. Local/test environments can omit the secret
    to keep unsigned development requests working; non-local
    environments fail closed when the secret is missing.
    """
    secret = settings.daily_dialin_webhook_hmac
    if not secret:
        environment = settings.environment.lower()
        if environment in _LOCAL_DIALIN_SIGNATURE_BYPASS_ENVS:
            logger.warning(
                "daily_dialin_signature_bypass_local",
                environment=settings.environment,
            )
            return True
        logger.error(
            "daily_dialin_signature_secret_missing",
            environment=settings.environment,
        )
        return False

    timestamp = request.headers.get(_DAILY_DIALIN_TIMESTAMP_HEADER, "")
    provided_signature = request.headers.get(_DAILY_DIALIN_SIGNATURE_HEADER, "")
    if not timestamp or not provided_signature:
        logger.warning(
            "daily_dialin_signature_missing_headers",
            has_timestamp=bool(timestamp),
            has_signature=bool(provided_signature),
        )
        return False

    if not _daily_dialin_timestamp_within_tolerance(timestamp, now=now):
        logger.warning("daily_dialin_signature_timestamp_rejected")
        return False

    expected_signature = _compute_daily_dialin_signature(secret, timestamp, raw_body)
    if expected_signature is None:
        logger.error("daily_dialin_signature_secret_invalid")
        return False

    if not hmac.compare_digest(provided_signature, expected_signature):
        logger.warning("daily_dialin_signature_mismatch")
        return False

    return True


def _compute_daily_dialin_signature(
    hmac_secret: str,
    timestamp: str,
    raw_body: bytes,
) -> str | None:
    """Return Daily's expected Base64 HMAC-SHA256 signature."""
    try:
        secret_bytes = base64.b64decode(hmac_secret, validate=True)
    except (binascii.Error, ValueError):
        return None

    signed_payload = timestamp.encode("utf-8") + b"." + raw_body
    digest = hmac.new(secret_bytes, signed_payload, "sha256").digest()
    return base64.b64encode(digest).decode("ascii")


def _daily_dialin_timestamp_within_tolerance(
    timestamp: str,
    *,
    now: float | None = None,
) -> bool:
    """Return whether a Daily timestamp is inside the replay window."""
    try:
        timestamp_value = float(timestamp.strip())
    except ValueError:
        return False

    # Be liberal on input: Daily docs name the header but do not state
    # seconds vs milliseconds. Support both numeric epoch formats.
    if timestamp_value > 10_000_000_000:
        timestamp_value = timestamp_value / 1000

    now_value = time.time() if now is None else now
    return abs(now_value - timestamp_value) <= _DAILY_DIALIN_SIGNATURE_TOLERANCE_SECS


async def _load_api_key(settings: Settings) -> str:
    """Resolve the API key from Secrets Manager.

    Empty-string ``api_key_secret_arn`` (the local-dev case) skips
    the AWS call entirely and returns ``""``, which
    :func:`_check_api_key` interprets as "auth disabled."

    The secret blob is treated as a JSON object with an ``api_key``
    field; a plain-string secret is also accepted. Production sets
    a JSON secret so other Cosentus services can extend with
    additional fields without rotating the secret name.
    """
    arn = settings.api_key_secret_arn
    if not arn:
        return ""
    client = boto3.client("secretsmanager", region_name=settings.aws_region)
    try:
        response = await asyncio.to_thread(
            client.get_secret_value,
            SecretId=arn,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed on any secret-load error
        logger.error(
            "api_key_secret_load_failed",
            arn=arn,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise RuntimeError("API key secret load failed") from exc

    raw = response.get("SecretString", "") or ""
    # Try JSON first, fall back to raw string.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            api_key = str(parsed.get("api_key") or parsed.get("apiKey") or "")
            if api_key:
                return api_key
            logger.error("api_key_secret_empty", arn=arn, secret_format="json")
            raise RuntimeError("API key secret resolved to an empty value")
    except (ValueError, json.JSONDecodeError):
        pass
    if raw:
        return raw
    logger.error("api_key_secret_empty", arn=arn, secret_format="raw")
    raise RuntimeError("API key secret resolved to an empty value")


# ── Phone-number lookup (inbound agent_id resolution) ────────────────────


async def _lookup_inbound_agent(to_number: str, settings: Settings) -> str | None:
    """Resolve ``to_number`` → inbound agent id via the API lambda.

    Reuses Layer 1's hardened ``_get_lambda_client(settings)`` for
    timeout + adaptive retry. Returns ``None`` on any failure path
    (lookup 404, lambda invoke error, malformed response). The
    caller (dialin webhook handler) maps ``None`` to a 503 so Daily
    retries — temporary issues self-heal; permanent issues page
    operations.

    The lambda returns the agent under a NESTED ``inbound_agent``
    object (mirrors the same shape exposed to the dashboard for
    list views), e.g.::

        {
            "id": "...",                      # phone number row id
            "number": "+12098075018",
            "inbound_agent": {
                "id": "<agent_id>",
                "name": "...",
                "display_name": "..."
            },
            "outbound_agent": null
        }

    First inbound test of v2 caught this — the prior implementation
    read the wrong key (``inbound_agent_id``, flat) and never
    spawned a bot for any real call.
    """
    from app.config.agent_config import _get_lambda_client

    client = _get_lambda_client(settings)
    envelope = {
        "httpMethod": "GET",
        "path": "/api/phone-numbers/lookup",
        "queryStringParameters": {"number": to_number},
        "headers": {},
        "body": None,
    }
    try:
        response = await asyncio.to_thread(
            client.invoke,
            FunctionName=settings.voice_api_lambda_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(envelope).encode("utf-8"),
        )
    except Exception as exc:  # noqa: BLE001 — log + return None
        logger.error(
            "phone_lookup_invoke_failed",
            to_number=to_number,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    try:
        payload_bytes = response["Payload"].read()
        payload = json.loads(payload_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "phone_lookup_bad_envelope",
            to_number=to_number,
            error=str(exc),
        )
        return None

    if payload.get("statusCode", 0) != 200:
        logger.warning(
            "phone_lookup_non_200",
            to_number=to_number,
            status_code=payload.get("statusCode"),
        )
        return None

    body_text = payload.get("body", "{}") or "{}"
    try:
        data = json.loads(body_text)
    except (ValueError, json.JSONDecodeError):
        return None
    inbound_agent = data.get("inbound_agent")
    if not isinstance(inbound_agent, dict):
        return None
    agent_id = inbound_agent.get("id")
    return agent_id if isinstance(agent_id, str) and agent_id else None


# ── Graceful drain ────────────────────────────────────────────────────────


async def graceful_drain(
    manager: PipelineManager,
    protection: TaskProtection,
    *,
    budget_secs: int = DEFAULT_DRAIN_BUDGET_SECS,
    metrics: MetricsEmitter | None = None,
) -> None:
    """SIGTERM-driven drain. Set draining flag, wait for active calls,
    cancel survivors, release protection.

    Cancels only the manager's spawned tasks
    (``manager.active_sessions.values()``), NOT every task on the
    asyncio loop. v1's ``for task in asyncio.all_tasks(loop)`` was
    a sledgehammer that killed the HTTP server's accept loop too —
    fine in practice (the process was about to exit anyway) but
    pollutes logs with bogus cancellations and races against the
    web.AppRunner's own cleanup. v2 ships fixed.

    The ``budget_secs`` matches Fargate's default 120 s
    ``stopTimeout`` minus a 10 s buffer for the post-drain
    protection release + cleanup work. Layer 11 may bump Fargate's
    stopTimeout for longer Cosentus IVR calls; pass a matching
    larger ``budget_secs`` here.

    ``metrics`` is optional. When supplied, the drain-timeout
    branch publishes a ``DrainTimeouts`` CloudWatch counter to
    feed Wave 3's alarm. Tests pass ``metrics=None`` and skip the
    emission path entirely.
    """
    logger.info(
        "drain_starting",
        active_sessions=manager.active_session_count,
        budget_secs=budget_secs,
    )
    await manager.shutdown()  # sets draining flag; future /ready returns 503

    drain_start = time.time()
    while manager.active_session_count > 0:
        elapsed = time.time() - drain_start
        if elapsed > budget_secs:
            remaining = manager.active_session_count
            logger.warning(
                "drain_timeout",
                remaining=remaining,
                elapsed_secs=round(elapsed, 1),
            )
            if metrics is not None:
                # Best-effort CloudWatch counter; emit_drain_timeout
                # swallows its own errors so a CloudWatch outage can't
                # block the cancel-and-exit path below.
                await metrics.emit_drain_timeout(remaining)
            # Cancel only the engine's tasks — NOT asyncio.all_tasks.
            for task in list(manager.active_sessions.values()):
                task.cancel()
            # Brief grace window for finally blocks to fire
            # (Layer 8's finalize_call writes a CallRecord even on
            # cancel).
            await asyncio.sleep(_POST_CANCEL_GRACE_SECS)
            break
        logger.info(
            "drain_waiting",
            remaining=manager.active_session_count,
            elapsed_secs=round(elapsed, 1),
        )
        await asyncio.sleep(_DRAIN_POLL_SECS)

    # Release protection so ECS can scale this task in.
    await protection.set_protected(False)
    await protection.close()
    logger.info("drain_complete", duration_secs=round(time.time() - drain_start, 1))
