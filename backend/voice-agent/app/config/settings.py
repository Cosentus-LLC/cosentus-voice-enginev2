"""Platform-wide engine settings — env-var-backed, pydantic-settings.

This module owns every operator-tunable platform value. Per-agent
config (LLM model, voice ID, prompt, tools…) lives in :mod:`agent_config`
and comes from the API Lambda's runtime-config endpoint; this layer
covers the values that are *engine-wide*, not per-call.

Why env-vars only
-----------------

v1 used SSM Parameter Store for hot-tunable feature flags. Every
feature that justified that machinery (knowledge-base / RAG, A2A
capability registry, SageMaker provider routing) was dropped from
v2's surface. Pipecat's own deployment guidance uses plain env vars,
and Fargate task-definition redeploys are cheap, so the SSM
round-trip at every container boot buys nothing in v2.

Why no singleton
----------------

v1's ``ConfigService.get_config_service()`` was a module-level
singleton. v2 builds ``Settings`` once at process startup (in the
runtime layer) and passes it explicitly to anything that needs it.
Dependency injection rather than global state.

Locked-in technical choices live in code, not here
--------------------------------------------------

These are NOT settings fields, even though v1 sometimes treated them
as configurable:

- STT vendor (AssemblyAI), TTS vendor (ElevenLabs), LLM vendor (Bedrock)
- AssemblyAI sample rate (8000), encoding (pcm_s16le), turn mode (Mode 2)
- VAD thresholds, MinWords value, smart-turn analyzer choice

Changing any of those should require a code review, not an env-var
flip. They live in Layer 3 (services factory) and Layer 8 (pipeline
builder).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform-wide engine settings.

    Two fields are required (no default — :class:`Settings` raises
    :class:`pydantic.ValidationError` at construction if they aren't
    set in the environment). Everything else has a sensible default.

    Field names are lowercase Python convention; ``case_sensitive=False``
    makes both ``VOICE_API_LAMBDA_NAME`` (the standard env-var
    convention) and ``voice_api_lambda_name`` resolve to the same
    attribute.
    """

    # ── Required at boot ────────────────────────────────────────────────
    # No defaults. Settings() raises ValidationError if these aren't set.

    voice_api_lambda_name: str
    """The cosentus-voice-api Lambda function name (or alias) the engine
    invokes for ``runtime-config``, ``call-write``, etc. Production
    sets this to e.g. ``medcloud-voice-api:live``."""

    api_key_secret_arn: str
    """Secrets Manager ARN whose JSON blob holds Daily / ElevenLabs /
    AssemblyAI API keys. The secrets-loader reads this at boot and
    populates the matching env vars."""

    # ── Vendor API keys ─────────────────────────────────────────────────
    # The Fargate task definition injects these from Secrets Manager as
    # container env vars (infrastructure ``ecs-service.ts`` →
    # ``ASSEMBLYAI_API_KEY`` / ``ELEVENLABS_API_KEY``); pydantic-settings
    # reads them here at boot. Layer 3's ``build_stt`` / ``build_tts``
    # consume them from this object rather than touching ``os.environ``,
    # so ``Settings`` stays the single source of truth for secrets.

    assemblyai_api_key: str = ""
    """AssemblyAI streaming-STT API key. Empty default permits boot in
    local/dev/test without vendor access; ``build_stt`` raises a clear
    ``ValueError`` if it's still empty when a call actually needs it."""

    elevenlabs_api_key: str = ""
    """ElevenLabs TTS API key. Empty default permits boot in
    local/dev/test without vendor access; ``build_tts`` raises a clear
    ``ValueError`` if it's still empty when a call actually needs it."""

    # ── Infrastructure with sensible defaults ───────────────────────────

    aws_region: str = "us-east-1"
    """Region for every boto3 client. Matches the Fargate task's region."""

    environment: str = "production"
    """Deployment environment tag for structured logs and CloudWatch EMF."""

    log_level: str = "INFO"
    """Engine log level. ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``."""

    service_port: int = 8080
    """Port the FastAPI HTTP entrypoint binds on inside the Fargate task."""

    max_concurrent_calls: int = 6
    """Per-task maximum concurrent voice sessions. Auto-scaling adds
    tasks above this threshold; concurrency above this on a single
    task degrades audio quality. Initial value 6 is a starting point;
    Layer 9.5 scale testing will validate or adjust to 4 or 8."""

    # ── Layer 9 (runner) ────────────────────────────────────────────────

    daily_api_key: str = ""
    """Daily.co API key. Used by Layer 9's ``DailyRoomClient`` to
    create rooms and mint meeting tokens. Empty default permits
    boot in a test environment without Daily access; the runner
    itself fails fast at room-creation time if the key is unset."""

    daily_dialin_webhook_hmac: str = ""
    """Base64-encoded Daily ``pinless_dialin.hmac`` secret used to
    verify requests to ``/daily-dialin-webhook``. Empty default
    permits local dev / unit-test construction; production must
    provide it or the webhook fails closed."""

    daily_recording_webhook_hmac: str = ""
    """Base64-encoded Daily recording webhook HMAC secret used to
    verify requests to ``/daily-recording-webhook``. Production must
    provide it or the webhook fails closed."""

    recording_bucket: str = ""
    """S3 bucket Daily uses for cloud recording uploads. Set on each
    room's ``recordings_bucket.bucket_name`` property at room
    creation time. Empty default means recordings are disabled —
    the room is created without ``recordings_bucket``, falling back
    to Daily's default storage. Production sets this."""

    recording_role_arn: str = ""
    """IAM role ARN Daily assumes when writing recordings to the
    bucket. Required when ``recording_bucket`` is set; ignored
    otherwise. Set on each room's ``recordings_bucket.assume_role_arn``
    property at room creation time. The trust policy on the role
    grants Daily's signing principal sts:AssumeRole; the role's
    permission policy grants s3:PutObject on the bucket."""

    recording_region: str = "us-east-1"
    """AWS region of ``recording_bucket`` as passed to Daily's
    ``recordings_bucket.bucket_region`` property. Kept separate from
    ``aws_region`` so the engine can invoke regional AWS services from
    one region while Daily writes recordings to the bucket's region."""

    # ── Observability: per-call tracing (#13) ───────────────────────────
    # Off by default. An operator opts in by setting the endpoint + keys
    # (a human/infra step — the engine never provisions these). When off,
    # the tracing layer is a no-op and calls run identically. We export
    # OTLP/HTTP to self-hosted Langfuse so PHI-free spans stay in our infra.

    tracing_enabled: bool = False
    """Master switch for per-call OpenTelemetry tracing. When ``False``
    (default) ``app.observability`` is a no-op — zero overhead, no spans.
    Telemetry is observe-only and fail-open regardless of this flag."""

    otel_exporter_otlp_endpoint: str = ""
    """OTLP/HTTP traces endpoint for self-hosted Langfuse, e.g.
    ``https://langfuse.internal/api/public/otel/v1/traces``. Empty (default)
    keeps tracing off even if ``tracing_enabled`` is set."""

    langfuse_public_key: str = ""
    """Langfuse public key (the username half of OTLP Basic auth). Empty
    default permits boot without tracing configured."""

    langfuse_secret_key: str = ""
    """Langfuse secret key (the password half of OTLP Basic auth). Injected
    like the other vendor secrets; routed through Settings, never read from
    ``os.environ`` directly. Empty default permits boot without tracing."""

    # ── Operator kill-switch ────────────────────────────────────────────

    disabled_tools: str = ""
    """CSV of tool names to disable platform-wide. Empty means none.
    Useful when a single tool is misbehaving and needs to be turned
    off across every agent without redeploying. The tools layer
    parses the CSV — Settings stores it as a raw string."""

    required_case_data_keys: str = ""
    """CSV of ``case_data`` keys that MUST be present and non-blank
    before an OUTBOUND call dials (D2 defense-in-depth, #27). Empty
    (default) = no guard, today's behavior. The dispatcher is expected
    to supply these for outbound calls; if one is missing/blank the
    call is blocked rather than dialed with a blank patient name /
    claim id. Inbound calls (no dispatcher-supplied ``case_data``) are
    never subject to this. Layer 8 (``bot.py``) parses the CSV; Settings
    stores it as a raw string, mirroring ``disabled_tools``."""

    post_call_model_fallback_chain: str = ""
    """CSV of Bedrock model short-names the OFFLINE post-call analysis
    falls back to, in order, when the primary model hits a **retryable**
    Bedrock error (#20 — throttling / service-unavailable / model-not-ready
    / timeouts). Empty (default) = no failover, today's behavior: a
    retryable error on the primary returns ``{}`` like any other error.

    The per-agent ``post_call_analyses.model`` (or its Sonnet default)
    is always tried first; these are tried only after it, left-to-right.
    Entries are short-names resolved through ``_SHORT_TO_BEDROCK`` (same
    as the primary), so they MUST already be in the API allowlist +
    that map — no contract drift. Non-retryable errors (e.g.
    ``AccessDeniedException``, ``ValidationException``) never trigger
    failover; they fail fast as before. This is the **offline** path
    only — live mid-call cross-model failover is out of scope (tracked
    in #52). ``post_call.py`` parses the CSV; Settings stores the raw
    string, mirroring ``disabled_tools`` / ``required_case_data_keys``."""

    live_model_fallback_chain: str = ""
    """CSV of Bedrock model short-names the LIVE in-call LLM falls back to,
    in order, when the primary model hits a **retryable** Bedrock error
    (#52 — throttling / service-unavailable / model-not-ready / connect+read
    timeouts) *before any tokens stream*. Empty (default) = no failover,
    today's behavior: the live pipeline runs on a single model and a
    retryable error surfaces as an ErrorFrame exactly as before.

    Distinct from ``post_call_model_fallback_chain`` (the OFFLINE extraction
    path, #20): live and offline have different model preferences — live
    favors low-latency models (Haiku), offline favors accuracy (Sonnet) — so
    they do NOT share a chain. The per-agent ``llm.model`` (or its default) is
    always tried first; these are tried only after it, left-to-right.

    Entries are short-names resolved through ``resolve_bedrock_model_id``
    (same path as the primary), so they MUST already be in the API allowlist
    + ``_SHORT_TO_BEDROCK`` — no contract drift. An unknown short-name warns
    and passes through (Bedrock rejects it), mirroring the primary; it is not
    silently dropped. Failover fires only on **retryable** errors raised at
    the ``converse_stream`` call (before streaming); a non-retryable error
    (validation / auth) or an exhausted chain fails as today. Once the call
    fails over it **sticks** to the surviving model for the rest of that call
    (avoids re-paying a throttled primary's latency every turn); this is
    strictly call-scoped (per-call LLM service instance). Mid-stream failures
    (after tokens started) are out of scope. ``services/factory.py`` parses
    the CSV and builds the failover service; Settings stores the raw string,
    mirroring ``post_call_model_fallback_chain`` / ``disabled_tools``."""

    identity_verification_keys: str = ""
    """CSV of ``case_data`` keys the caller MUST confirm before the
    Pipecat Flows identity gate (#42) opens — the code-enforced HIPAA
    wall. Distinct from ``required_case_data_keys`` (which only checks a
    field is *present* before an outbound call dials): these are the
    fields the caller must *state correctly* to be verified. The
    ``verify_identity`` flow function compares the caller's claimed
    values against ``case_data`` for these keys, in code — not the LLM.

    Identity verification is the **outbound** path: outbound calls carry
    dispatcher-supplied ``case_data`` (patient name / DOB / claim id) to
    verify against. Inbound calls carry ``case_data={}`` — nothing to
    verify against — so the gate stays **fail-closed** (blocked) pending
    a future inbound-identity lookup, rather than silently opening.

    Empty (default) = no keys configured. When ``flows_enabled`` is on
    and this is empty, the gate has nothing to verify against and so
    **blocks everything** (fail-closed); ``bot.py`` logs this loudly at
    flow-build time. Operators enabling Flows MUST set this. Layer 8
    (``bot.py``) parses the CSV; Settings stores it as a raw string,
    mirroring ``required_case_data_keys`` / ``disabled_tools``."""

    payer_id_case_data_key: str = "payer_name"
    """The ``case_data`` key whose value identifies the payer for the
    verified-IVR-path fetch (#17). The engine reads this key from the
    dispatcher-supplied ``case_data`` and passes the value to
    ``GET /api/payers/:id`` to load that payer's verified claims IVR path,
    which the Flows ``navigate`` step then follows via ``press_digit``.

    Defaults to ``payer_name`` because the dispatcher carries the payer as
    ``payer_name`` (the same value that hydrates ``{{payer_name}}``). Note
    the lookup gap: ``/api/payers/:id`` matches by slug/id, not name, so a
    name fetch currently 404s and the agent falls back to navigating by
    ear — today's behavior — until the API exposes a name lookup. Blank /
    unset, or the key absent from ``case_data`` → no fetch, by-ear
    fallback. Only consulted when ``flows_enabled`` is on (the ``navigate``
    step exists only on the Flows path). ``bot.py`` reads the key; Settings
    stores it as a raw string, mirroring ``required_case_data_keys``."""

    # ── Feature flags ───────────────────────────────────────────────────

    context_summarization_enabled: bool = False
    """Rollout switch for bounded per-turn LLM context (#22). When
    ``False`` (default) the assistant aggregator runs with its stock
    parameters and the per-call ``LLMContext`` accumulates the whole
    conversation — today's behavior, byte-identical. When ``True``
    ``bot.py`` enables Pipecat's built-in
    :class:`~pipecat.processors.aggregators.llm_context_summarizer.LLMContextSummarizer`
    on the assistant aggregator: once the conversation crosses a token /
    message threshold (the knobs live in code in ``bot.py``, alongside
    the other tuned pipeline constants), older turns are folded into a
    running summary and only the last N turns are kept verbatim — so
    per-turn input stays bounded on a 20-30 min call instead of growing
    without limit.

    Off by default and independent of ``flows_enabled``: summarization
    adds a mid-call LLM round-trip that touches live-call timing, so it
    ships gated and is flipped on only after staging + eval (#5)
    validation. Complementary to the Flows per-step context bounding
    (#43): that bounds context *across* steps; this windows the turns
    *within* one long step (and is the only bound on the flows-off
    production path). The summary is produced by the call's own Bedrock
    model and lives in-context only — never logged, no new PHI surface."""

    flows_enabled: bool = False
    """Rollout switch for the Pipecat Flows layer (EPIC #16). When
    ``False`` (default) ``bot.py`` still *constructs* the per-call
    ``FlowManager`` — proving the wiring — but never initializes a node,
    so call behavior is byte-identical to the pre-Flows pipeline
    (opener + turn machinery untouched). When ``True`` the scaffold flow
    (#41) initializes its first node; 16b/16c build the real steps behind
    this same flag. This is a temporary rollout switch — flip it on and
    then remove it once Flows is the production path, so it doesn't
    linger as dead config.

    ✅ The compliance wall is complete (16b #42 + 16c #43). The identity
    gate gates *tools* + the *flow transition*, AND its PHI-free
    ``role_message`` replaces the hydrated system instruction for the
    whole pre-verification phase, so the model is never given any
    ``case_data`` value before the caller is verified; the hydrated
    prompt is restored only by the post-verification step chain. Enabling
    this still REQUIRES ``identity_verification_keys`` to be set, or the
    gate fail-closes and blocks every tool.

    ⚠️ Operationally still OFF by default: flip it on in production only
    after the full Flows path (gate + ordered steps) has been validated
    on staging."""

    knowledge_prefetch_enabled: bool = False
    """Rollout switch for the non-blocking knowledge prefetch skeleton (#56).

    When ``False`` (default), ``bot.py`` constructs no knowledge cache, no
    warmer, and no turn-boundary hook, so the production path remains
    byte-identical. When ``True``, each call gets its own in-memory semantic
    cache backed only by de-identified payer-level fixtures and local
    deterministic embeddings. Real PHI-bearing sources and network embeddings
    remain deferred behind interfaces."""

    knowledge_cache_ttl_secs: int = 300
    """Per-entry TTL for the per-call semantic cache. Applies only when
    ``knowledge_prefetch_enabled`` is on."""

    knowledge_cache_max_entries: int = 64
    """Maximum entries retained by each per-call semantic cache. Applies only
    when ``knowledge_prefetch_enabled`` is on."""

    model_config = SettingsConfigDict(
        # ``.env`` is for local dev only. Production Fargate sets env
        # vars via the task definition; .env is not present in the
        # container image.
        env_file=".env",
        env_file_encoding="utf-8",
        # Fargate sets a slew of AWS-managed env vars
        # (ECS_CONTAINER_METADATA_URI_V4, AWS_EXECUTION_ENV, etc.)
        # that aren't ours to model. Drop them silently.
        extra="ignore",
        # Standard env-var convention is uppercase; field names
        # follow Python convention. Both should resolve.
        case_sensitive=False,
    )
