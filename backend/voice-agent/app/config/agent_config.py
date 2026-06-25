"""Agent runtime config — Pydantic contract and Lambda loader.

This module owns the typed contract between the Fargate voice engine
and the cosentus-voice-api Lambda's
``GET /api/agents/:id/runtime-config`` endpoint. It defines the
Pydantic model tree that mirrors the JSON shape the Lambda returns,
and an async loader that fetches that config at call start.

Transport
---------

Lambda direct invoke only (boto3 ``Lambda.Invoke``, sync client
wrapped in :func:`asyncio.to_thread`). Same-account, IAM-native — no
API key, no API Gateway. v1 also exposed an HTTP fallback for local
dev and cross-account use; v2 deletes it. There is one path.

Synchronous boto3 inside ``to_thread`` is deliberate: aiobotocore
has produced intermittent "Connection closed" errors on Lambda
cold-start responses despite the Lambda having executed
successfully. Sync boto3 in a worker thread is battle-tested for
this exact call pattern, and a single ~200ms warm / ~1.2s cold
blocking call at session start is acceptable.

Failure mode
------------

Every failure path raises :class:`AgentConfigLoadError`. v2 does NOT
fall back to a stub agent on failure; that was a v1 pattern we
deliberately removed. The bot-runner Lambda's retry / dead-letter
handles a failed call.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, ClassVar

import boto3
import structlog
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config.settings import Settings

logger = structlog.get_logger(__name__)


class AgentConfigLoadError(Exception):
    """Raised when agent config cannot be loaded for any reason.

    v2 explicitly does NOT fall back to a stub agent on failure.
    Failed config load fails the call cleanly and bot-runner
    Lambda's retry handles it.
    """


# ── Pydantic models ─────────────────────────────────────────────────────────
#
# These mirror the JSON shape the cosentus-voice-api Lambda's
# ``buildRuntimeConfig`` returns. ``extra='ignore'`` on every model
# is deliberate: the lambda sends fields v2 chose not to model (per-
# agent provider/recording fields the v1 engine ignored anyway). See
# docs/v2-tech-debt-log.md entry 1 for the full list and the exit
# condition that lets us tighten this back to ``extra='forbid'``.
#
# Until Entry 1 closes (the lambda still ships those fields), we keep
# ``extra='ignore'`` so valid configs parse, but every model inherits
# :class:`_RuntimeConfigModel` which logs a **loud warning** for any
# field that is neither modeled nor on that model's
# ``_known_extra_fields`` allowlist. That turns silent contract drift
# — a renamed/added field the lambda starts sending — into a
# queryable ``agent_config_unknown_fields`` log line at parse time,
# instead of a default silently winning mid-call (the B2 failure
# mode). The known v1-era extras stay silent via the allowlist so we
# don't warn on every valid call.


class _RuntimeConfigModel(BaseModel):
    """Base for the runtime-config models: log-on-unknown-field guard.

    Keeps ``extra='ignore'`` (subclasses may extend ``model_config``,
    e.g. ``populate_by_name=True``; Pydantic merges it across
    inheritance) so a currently-valid payload still parses. A
    ``mode='before'`` validator inspects the raw incoming dict and
    emits a ``logger.warning('agent_config_unknown_fields', ...)`` for
    any key that is neither a modeled field/alias nor in this model's
    ``_known_extra_fields``. The call still proceeds — the unknown
    field is dropped as before — but the drift is now visible.

    ``_known_extra_fields`` is the per-subclass allowlist of fields the
    lambda is KNOWN to send but v2 deliberately doesn't model
    (docs/v2-tech-debt-log.md Entry 1). Drop entries as the lambda
    stops sending them; when every subclass's allowlist is empty and
    Entry 1 closes, we can flip to ``extra='forbid'`` (tracked as a
    follow-up to Entry 1 / B2).
    """

    model_config = ConfigDict(extra="ignore")

    _known_extra_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _allowed_field_names(cls) -> set[str]:
        """Modeled field names plus their wire aliases.

        Aliases (``_meta``, ``version``) must be included because the
        ``mode='before'`` validator sees the raw dict before Pydantic
        resolves aliases to attribute names.
        """
        names: set[str] = set()
        for name, info in cls.model_fields.items():
            names.add(name)
            if info.alias:
                names.add(info.alias)
        return names

    @model_validator(mode="before")
    @classmethod
    def _warn_on_unknown_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            allowed = cls._allowed_field_names()
            unknown = sorted(
                key for key in data if key not in allowed and key not in cls._known_extra_fields
            )
            if unknown:
                logger.warning(
                    "agent_config_unknown_fields",
                    model=cls.__name__,
                    unknown_fields=unknown,
                )
        return data


class LLMConfig(_RuntimeConfigModel):
    """LLM-side per-agent config.

    NOTE: ``provider`` and ``enable_prompt_caching`` are intentionally
    not modeled. v2 uses Bedrock Claude for every agent, and prompt
    caching is hardcoded ON in the LLM service factory (Layer 3) —
    no production agent ever needs caching off, so the per-agent
    toggle is dead. Both are allowlisted in ``_known_extra_fields`` so
    they're dropped silently; any *other* unknown field is logged. See
    docs/v2-tech-debt-log.md entry 1.
    """

    model_config = ConfigDict(extra="ignore")
    _known_extra_fields = frozenset({"provider", "enable_prompt_caching"})

    # Default Claude Haiku 4.5 for voice workloads (short turns, bounded
    # reasoning, tool-call-driven flows). Sonnet 4.6 is ~3x more
    # expensive per token; we opt into Sonnet explicitly for hard cases.
    # This default is mostly defensive — the lambda's
    # GetRuntimeConfig payload always includes llm.model, so this
    # fallback only fires if the lambda is misconfigured.
    model: str = "claude-haiku-4-5"
    max_tokens: int = 200
    temperature: float = 0.7


class TTSSettings(_RuntimeConfigModel):
    """ElevenLabs voice-tuning settings.

    Only ``stability`` and ``use_speaker_boost`` are modeled in v2.
    The lambda still sends ``similarity_boost``, ``style``, and
    ``speed``, but Cosentus's production fleet uses ElevenLabs
    defaults for those three. They're allowlisted in
    ``_known_extra_fields`` so they're dropped silently by
    ``extra='ignore'`` without a drift warning.
    """

    model_config = ConfigDict(extra="ignore")
    _known_extra_fields = frozenset({"similarity_boost", "style", "speed"})

    stability: float | None = None
    use_speaker_boost: bool | None = None


class TTSConfig(_RuntimeConfigModel):
    """ElevenLabs-side per-agent config.

    NOTE: ``provider`` is intentionally not modeled (allowlisted in
    ``_known_extra_fields``). v2 uses ElevenLabs for every agent.
    """

    model_config = ConfigDict(extra="ignore")
    _known_extra_fields = frozenset({"provider"})

    voice_id: str = ""
    # No Layer-1 default: the factory's _ELEVENLABS_DEFAULT_MODEL is the
    # single source of truth for the platform fallback (production value
    # eleven_flash_v2_5), mirroring voice_id above. An empty value here
    # falls through to that fallback in build_tts. Real agents always send
    # tts.model, so this only governs the omitted-field path.
    model: str = ""
    settings: TTSSettings = Field(default_factory=TTSSettings)


class STTConfig(_RuntimeConfigModel):
    """STT-side per-agent config.

    Only ``keywords`` is per-agent in v2. ``provider`` (always
    AssemblyAI) and ``language`` (always English) are platform-wide
    and allowlisted in ``_known_extra_fields`` so they're dropped
    silently by ``extra='ignore'``.
    """

    model_config = ConfigDict(extra="ignore")
    _known_extra_fields = frozenset({"provider", "language"})

    keywords: list[str] = Field(default_factory=list)


class ToolConfig(_RuntimeConfigModel):
    """One row in the agent's enabled-tools list."""

    model_config = ConfigDict(extra="ignore")

    type: str
    description: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)


class PostCallField(_RuntimeConfigModel):
    """One field in the per-agent post-call analysis schema."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: str = "text"
    description: str = ""
    format_examples: list[str] = Field(default_factory=list)
    choices: list[str] = Field(default_factory=list)


class PostCallConfig(_RuntimeConfigModel):
    """Per-agent post-call analysis schema."""

    model_config = ConfigDict(extra="ignore")

    # Default to the stronger model for the once-per-call OFFLINE
    # extraction (#20 — per-turn model routing). Post-call analysis is
    # structurally separate from the live pipeline, runs after the call
    # ends, and trades latency/cost for accuracy — so it defaults to
    # Sonnet, the inverse of the live LLMConfig.model default (Haiku).
    # This is engine fallback policy for the *omitted-field* path; the
    # runtime-config shape is unchanged and an explicit per-agent
    # ``post_call_analyses.model`` still wins (resolved in
    # ``run_post_call_analyses`` as ``pca_config.model or default``).
    model: str = "claude-sonnet-4-6"
    fields: list[PostCallField] = Field(default_factory=list)


class AgentConfigMeta(_RuntimeConfigModel):
    """Server-provided metadata for observability.

    The lambda sends ``_meta`` underscore-prefixed on the wire (the
    outer alias is on AgentConfig.meta); this submodel describes the
    inner shape.

    ``updated_at_ms`` is the Aurora row's ``updated_at`` column as
    unix milliseconds. The lambda still names this field ``version``
    on the wire — a historical artifact since the value was always
    wall-clock-derived, never a true monotonic version number. v2
    aliases ``version`` -> ``updated_at_ms`` so the data round-trips
    while internal callers see the honest name. When the lambda
    renames the field, the alias goes too. See
    docs/v2-tech-debt-log.md entry 2.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    agent_id: str = ""
    updated_at_ms: int = Field(default=0, alias="version")


class AgentConfig(_RuntimeConfigModel):
    """Top-level per-agent runtime config.

    Mirrors the lambda's ``GET /api/agents/:id/runtime-config``
    response shape, after dropping fields v2 doesn't read (see
    per-submodel notes and docs/v2-tech-debt-log.md entry 1).

    ``extra='ignore'`` lets v2 silently drop fields the lambda sends
    but v2 doesn't model — without this we would hard-fail every
    call on the v1-era fields the lambda still emits. The whole
    ``recording`` object is the one such top-level field today, so
    it's allowlisted in ``_known_extra_fields``; any other unmodeled
    top-level key is logged by the ``_RuntimeConfigModel`` guard.
    ``populate_by_name=True`` lets the ``meta`` field accept either
    the wire alias ``_meta`` or the Python attribute name.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    _known_extra_fields = frozenset({"recording"})

    name: str
    display_name: str = ""
    description: str = ""
    system_prompt: str = ""
    first_message: str = ""
    flow_definition: dict[str, Any] | None = None
    ivr_goal: str = ""
    identity_verification_keys: list[str] = Field(default_factory=list)
    call_kind: str | None = None
    """Per-agent call policy selector.

    ``payer`` means the bot is calling a payer phone system: skip patient
    identity verification and include the IVR navigate step. ``patient`` means
    the bot is calling a patient: verify identity before PHI/tool access and
    omit payer IVR navigation. The field is intentionally permissive so a bad
    runtime-config value falls back in ``bot.py`` instead of failing a call at
    parse time.
    """

    speak_first: bool = True
    """Whether the agent speaks first when the call starts.

    Three modes (matching Retell's pattern):

    * ``speak_first=False`` — user speaks first, the bot stays silent
      on connect and waits for transcribed input before generating
      anything.
    * ``speak_first=True`` AND ``first_message`` non-empty —
      static opener: the bot speaks ``first_message`` verbatim
      via TTS, no LLM round-trip.
    * ``speak_first=True`` AND ``first_message`` empty — dynamic
      opener: the LLM generates the first turn from
      ``system_prompt``.

    Defaults to ``True`` for backward compatibility — every existing
    Cosentus agent in production today implicitly speaks first
    (they all have a non-empty ``first_message``)."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tools: list[ToolConfig] = Field(default_factory=list)
    post_call_analyses: PostCallConfig | None = None
    meta: AgentConfigMeta = Field(default_factory=AgentConfigMeta, alias="_meta")


# ── Lambda loader ───────────────────────────────────────────────────────────
#
# Layer 1 reads env vars directly; Layer 2 (settings) doesn't exist
# yet. When it lands, the loader takes a settings object instead of
# calling os.environ.get itself. See docs/v2-tech-debt-log.md
# entry 3.

_LAMBDA_NAME_ENV = "VOICE_API_LAMBDA_NAME"
_REGION_ENV = "AWS_REGION"
_DEFAULT_REGION = "us-east-1"

# Lazy-initialized Lambda client. ``None`` until
# :func:`_get_lambda_client` runs the first time, at which point we
# construct the client using ``settings.aws_region`` (or the env-var
# fallback when no Settings is supplied) and cache it for every
# subsequent call.
#
# Why lazy:
#
# * AWS boto3 docs explicitly warn against creating clients inside
#   concurrent contexts: doing so can cause SSL interpreter failures
#   and response-ordering issues. Sync boto3 clients are thread-safe,
#   so the documented pattern is one client shared across worker
#   threads. We honor that — the cache makes the client effectively
#   module-shared after the first call.
# * We create through ``boto3.session.Session().client(...)`` rather
#   than ``boto3.client(...)`` to bypass the global DEFAULT_SESSION
#   entirely — that's the multithreaded pattern AWS documents.
#   Reference: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html
# * Explicit timeouts: default ``read_timeout`` is 60 s which would
#   hang the engine on a slow lambda or hot Aurora. 8 s read / 2 s
#   connect / 2 attempts adaptive matches the latency budget at call
#   start (warm ~200 ms, cold ~1.2 s).
#
# Closes Entry 4 of the tech debt log: the region is now bound from
# ``settings.aws_region`` at first use rather than from
# ``os.environ`` at import time. The function signature's promise —
# "settings drives the client" — is now actually true.
_LAMBDA_CLIENT: Any = None


def _get_lambda_client(settings: Settings | None = None) -> Any:
    """Return the module-shared lambda client, constructing it lazily.

    Idempotent — every call after the first returns the cached
    client. Region binding is taken from ``settings.aws_region`` when
    provided. When ``settings is None`` (Layer 1's pre-Layer-9 callers
    that haven't been wired through Settings yet — see tech debt
    Entry 3), we fall back to ``os.environ`` like the original
    pattern.

    The first call pays the construction cost (~1 ms on a warm
    interpreter). The boto3 sync client is thread-safe so the same
    instance is shared across the worker threads
    :func:`asyncio.to_thread` dispatches to.
    """
    global _LAMBDA_CLIENT
    if _LAMBDA_CLIENT is None:
        region = (
            settings.aws_region
            if settings is not None
            else os.environ.get(_REGION_ENV, _DEFAULT_REGION)
        )
        _LAMBDA_CLIENT = boto3.session.Session().client(
            "lambda",
            region_name=region,
            config=Config(
                connect_timeout=2.0,
                read_timeout=8.0,
                retries={"max_attempts": 2, "mode": "adaptive"},
            ),
        )
    return _LAMBDA_CLIENT


def _build_proxy_event(agent_id_or_name: str) -> bytes:
    """Construct the API-Gateway-proxy event the lambda expects."""
    payload: dict[str, Any] = {
        "httpMethod": "GET",
        "path": f"/api/agents/{agent_id_or_name}/runtime-config",
        "headers": {},
        "queryStringParameters": None,
        "body": None,
    }
    return json.dumps(payload).encode("utf-8")


def _invoke_lambda_sync(
    function_name: str,
    payload: bytes,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Synchronous boto3 ``Lambda.Invoke`` — runs in a worker thread.

    Resolves the module-shared client via :func:`_get_lambda_client`
    so the first call binds region from ``settings`` (when provided)
    rather than from the import-time env read.
    """
    client = _get_lambda_client(settings)
    return client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=payload,
    )


async def load_agent_config(
    agent_id_or_name: str,
    settings: Settings | None = None,
) -> AgentConfig:
    """Fetch one agent's runtime config from the voice-api Lambda.

    Args:
        agent_id_or_name: Agent UUID or name. Passed verbatim into
            the URL path the lambda routes on.
        settings: Optional :class:`~app.config.settings.Settings`.
            When provided, the lambda function name comes from
            ``settings.voice_api_lambda_name``. When ``None``, falls
            back to ``os.environ["VOICE_API_LAMBDA_NAME"]``. Layer 9
            (runtime) constructs Settings once at startup and passes
            it to every caller; the env-var fallback exits then. See
            docs/v2-tech-debt-log.md entry 3.

    Returns:
        Parsed :class:`AgentConfig`.

    Raises:
        AgentConfigLoadError: For any failure — missing config, lambda
            invoke error, ``FunctionError``, non-200 response status,
            malformed JSON in either envelope, or Pydantic
            validation failure. v2 does not fall back to a stub
            agent.
    """
    started = time.perf_counter()

    if settings is not None:
        function_name = settings.voice_api_lambda_name
    else:
        function_name = os.environ.get(_LAMBDA_NAME_ENV, "")

    if not function_name:
        logger.error(
            "agent_config_load_failed",
            reason="missing_env",
            env_var=_LAMBDA_NAME_ENV,
            agent_id_or_name=agent_id_or_name,
        )
        raise AgentConfigLoadError(f"{_LAMBDA_NAME_ENV} environment variable is required")

    payload = _build_proxy_event(agent_id_or_name)

    try:
        resp = await asyncio.to_thread(_invoke_lambda_sync, function_name, payload, settings)
    except (BotoCoreError, ClientError) as exc:
        load_time_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "agent_config_load_failed",
            reason="lambda_invoke_error",
            agent_id_or_name=agent_id_or_name,
            function_name=function_name,
            error=str(exc),
            error_type=type(exc).__name__,
            load_time_ms=load_time_ms,
        )
        raise AgentConfigLoadError(f"Lambda invoke failed for {agent_id_or_name}: {exc}") from exc

    # boto3 invoke envelope: {"Payload": StreamingBody, "FunctionError": str?, ...}
    raw_payload = resp["Payload"].read()
    if isinstance(raw_payload, (bytes, bytearray)):
        raw_payload = raw_payload.decode("utf-8")

    try:
        outer = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        load_time_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "agent_config_load_failed",
            reason="parse_error",
            stage="outer_json",
            agent_id_or_name=agent_id_or_name,
            error=str(exc),
            load_time_ms=load_time_ms,
        )
        raise AgentConfigLoadError(
            f"Lambda response was not valid JSON for {agent_id_or_name}"
        ) from exc

    if resp.get("FunctionError"):
        load_time_ms = (time.perf_counter() - started) * 1000
        message = outer.get("errorMessage", outer)
        logger.error(
            "agent_config_load_failed",
            reason="function_error",
            agent_id_or_name=agent_id_or_name,
            error=message,
            load_time_ms=load_time_ms,
        )
        raise AgentConfigLoadError(
            f"Lambda returned FunctionError for {agent_id_or_name}: {message}"
        )

    # Inner envelope is API-Gateway-proxy: {"statusCode", "body" (JSON str), "headers"}
    status = outer.get("statusCode", 500)
    body_str = outer.get("body", "") or ""

    if status == 404:
        load_time_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "agent_config_load_failed",
            reason="not_found",
            agent_id_or_name=agent_id_or_name,
            status=status,
            load_time_ms=load_time_ms,
        )
        raise AgentConfigLoadError(f"agent not found: {agent_id_or_name}")

    if status >= 400:
        load_time_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "agent_config_load_failed",
            reason="http_error",
            agent_id_or_name=agent_id_or_name,
            status=status,
            body_preview=body_str[:200],
            load_time_ms=load_time_ms,
        )
        raise AgentConfigLoadError(
            f"Lambda returned HTTP {status} for {agent_id_or_name}: {body_str[:200]}"
        )

    try:
        body = json.loads(body_str) if body_str else {}
    except json.JSONDecodeError as exc:
        load_time_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "agent_config_load_failed",
            reason="parse_error",
            stage="inner_json",
            agent_id_or_name=agent_id_or_name,
            error=str(exc),
            load_time_ms=load_time_ms,
        )
        raise AgentConfigLoadError(
            f"Lambda body was not valid JSON for {agent_id_or_name}"
        ) from exc

    try:
        config = AgentConfig.model_validate(body)
    except ValidationError as exc:
        load_time_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "agent_config_load_failed",
            reason="validation_error",
            agent_id_or_name=agent_id_or_name,
            error=str(exc),
            load_time_ms=load_time_ms,
        )
        raise AgentConfigLoadError(
            f"Lambda response did not match AgentConfig schema for {agent_id_or_name}: {exc}"
        ) from exc

    load_time_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "agent_config_loaded",
        transport="lambda_invoke",
        agent_id=config.meta.agent_id or config.name,
        agent_name=config.name,
        updated_at_ms=config.meta.updated_at_ms,
        tool_count=len(config.tools),
        llm_model=config.llm.model,
        tts_voice_id=config.tts.voice_id,
        load_time_ms=load_time_ms,
    )
    return config
