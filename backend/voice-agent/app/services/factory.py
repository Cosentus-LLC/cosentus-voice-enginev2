"""Vendor service-object factory — STT, TTS, LLM construction.

Three pure functions, one Bedrock model-ID resolver. Each ``build_*``
returns a fully-configured Pipecat service object the pipeline
(Layer 8) hands directly to ``Pipeline(...)``.

Boundary
--------

This module *does not* know about pipelines, transports, turn-taking
strategies, or VAD analyzers. Those are Layer 8 concerns and pull
their own constants from Pipecat or from this module's siblings.
What lives here:

* The Mode-2 AssemblyAI incantation (``vad_force_turn_endpoint=False``,
  ``should_interrupt=False``, ``vad_threshold=0.3``, 8 kHz PCM —
  every parameter paid for in production debugging).
* The Bedrock model-ID translation table — Aurora stores
  ``"claude-haiku-4-5"`` shorthand; Bedrock requires the full
  inference profile ID. The mapping is the only authoritative
  source.
* Per-agent voice-tuning passthrough that respects "field unset
  means use vendor defaults" rather than clobbering them with
  ``None``.

Locked-in choices vs per-agent fields
-------------------------------------

Module-level constants (``_STT_*``, ``_ELEVENLABS_DEFAULT_*``) are
locked-in technical decisions that should require code review to
change. Per-agent fields (``agent.tts.voice_id``,
``agent.llm.temperature``) flow from the lambda's runtime-config
through ``AgentConfig`` and into vendor settings.

Behavior changes from v1
------------------------

Two intentional fixes vs ``backend/voice-agent/app/services/factory.py``
in v1's phase-8 branch:

1. ``agent.stt.keywords`` now applies on AssemblyAI calls. v1 read
   the field but only wired it into the Deepgram branch (which v2
   removed entirely), so on every production AssemblyAI call the
   keywords were silently dropped. v2 wires them into
   ``Settings.keyterms_prompt``.
2. Bedrock prompt caching is now active on every call. v1 logged
   ``agent.llm.enable_prompt_caching`` but never passed it to the
   ``AWSBedrockLLMService`` constructor — the flag was a no-op. v2
   hardcodes it ``True`` here so every Bedrock call benefits.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transcriptions.language import Language

from app.config.agent_config import AgentConfig
from app.config.settings import Settings

logger = structlog.get_logger(__name__)


# ── Locked-in AssemblyAI Mode-2 choices ────────────────────────────────────
#
# Each constant maps to one of v2's nine non-negotiable technical
# decisions (see docs/architecture/overview.md). Change any of these
# and you regress a production failure. v1 incident postmortems for
# the calls behind these values: ``2a7b2f94`` (claim-number chopping
# from Mode-1 endpointing) and ``cf423cd4`` (mid-bot interruption from
# AssemblyAI's broadcast_interruption bypass).

_STT_MODEL = "u3-rt-pro"
"""AssemblyAI Universal-3 Pro Streaming model — purpose-built for
alphanumeric / structured speech (claim numbers, member IDs, payer
names). Semantic turn-detection waits for terminal punctuation, so
mid-number pauses don't fragment turns."""

_STT_SAMPLE_RATE = 8000
"""Daily PSTN delivers 8 kHz PCM-linear. Override Pipecat's 16 kHz
default; we send the wire's actual rate, no upsampling."""

_STT_ENCODING = "pcm_s16le"
"""Daily PSTN is PCM-linear (16-bit), not μ-law. Match it exactly."""

_STT_VAD_THRESHOLD = 0.3
"""AssemblyAI's VAD activation threshold; must match the pipeline's
Silero confidence (also 0.3 — wired in Layer 8) to eliminate the
"dead zone" where AssemblyAI transcribes speech Silero hasn't
classified as voice."""

_STT_VAD_FORCE_TURN_ENDPOINT = False
"""Mode 2: AssemblyAI's native turn-endpoint detection drives turn
endings — we do NOT force endpointing from a Silero-VAD signal.
Mode 1 (``True``) was empirically shown to chop claim-number
recitations into one-word fragments (incident 2026-04-30, call
2a7b2f94)."""

_STT_SHOULD_INTERRUPT = False
"""Disable AssemblyAI's vendor-specific ``broadcast_interruption()``
bypass. With ``True`` (the Pipecat default), AAI fires interrupts
the instant SpeechStarted lands — before any transcript, outside
Pipecat's user_turn_strategies machinery. The bot was killed
mid-utterance on phantom user speech (incident 2026-04-30, call
cf423cd4). Interruption gating runs through Layer 8's
``MinWordsUserTurnStartStrategy`` instead."""


# ── ElevenLabs platform fallbacks ──────────────────────────────────────────

_ELEVENLABS_DEFAULT_VOICE_ID = "vW1NxlzqX8WROgpQAghR"
"""Platform-wide fallback voice when an agent doesn't set
``tts.voice_id``. v1 had three fallback layers (per-agent → SSM →
hardcoded); v2 dropped SSM, so the chain is per-agent → this."""

_ELEVENLABS_DEFAULT_MODEL = "eleven_flash_v2_5"
"""Platform-wide fallback model, and the single source of truth for
the TTS-model default. Matches v1's production-running value.
``AgentConfig.tts.model`` defaults to ``""`` on the Layer-1 side
(like ``voice_id``), so the chain is per-agent → this. Real agents
always send ``tts.model``; this only fires on the omitted/empty
path."""


# ── Bedrock fallback ───────────────────────────────────────────────────────

_BEDROCK_DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
"""Used when an agent's ``llm.model`` is empty. v1 also had an
``LLM_MODEL_ID`` env var fallback; v2 dropped it because Layer 2
owns env reads — direct ``os.environ`` access in Layer 3 violates
the boundary."""


# ── Bedrock model-ID resolution table ──────────────────────────────────────
#
# Aurora stores short model names ("claude-haiku-4-5"); Bedrock
# requires full inference profile IDs ("us.anthropic.claude-haiku-
# 4-5-20251001-v1:0"). This table is the only translation source.
#
# IMPORTANT: these are REAL Bedrock inference-profile IDs, not
# guesses. Verify any additions with:
#
#     aws bedrock list-inference-profiles --region us-east-1 \
#       --query 'inferenceProfileSummaries[?contains(inferenceProfileId,
#                 `claude`)].inferenceProfileId'
#
# Bedrock's ``ValidationException`` for a bad ID is swallowed by
# Pipecat's ``AWSBedrockLLMService``, producing a silent no-op
# pipeline (no LLM output, no errors in logs) — a typo here means
# every call goes to dead air. Production incident 2026-04-24 traces
# back to exactly this failure mode.
#
# SINGLE SOURCE OF TRUTH FOR THE ALLOWLIST = THE API, NOT HERE.
# The set of *accepted short model names* is owned by the API
# (``api-lambda-v2`` → ``lib/agent-schema.ts`` → ``VALID_LLM_MODELS``,
# a single Zod source). This table's KEYS are the engine's view of
# that allowlist and MUST stay in sync with it: a model added on one
# side only makes calls 400 (the 2026-05-27 prod incident — tech-debt
# Entry 19). The contract test
# ``tests/services/test_model_contract.py::test_model_map_matches_api_allowlist``
# fails the build the moment the two drift, so the divergence is
# caught at CI time rather than mid-call.

_SHORT_TO_BEDROCK: dict[str, str] = {
    # Sonnet 4.6's inference profile has no date suffix (unlike older
    # Claude versions). Don't "fix" this by adding one — it'll break
    # the call.
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-opus-4-5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
}


# ── Public functions ───────────────────────────────────────────────────────


# Aurora occasionally stores dated short forms ("claude-haiku-4-5-
# 20251001"). The lambda's POST /api/agents/:name/clone path inherits
# whatever ``post_call_analyses.model`` value was on the source agent;
# v2-tools-test ships with the dated form, so any agent cloned from
# it carries it forward. Stripping a trailing -YYYYMMDD before map
# lookup means a single map entry per Claude version covers both
# forms — and any future date-suffix variants land here automatically.
# Caught empirically in the v2 inbound PSTN test (call_id
# c18a181a-...): post_call_analyses.model was "claude-haiku-4-5-
# 20251001" → not in map → passed through to Bedrock → rejected
# with ValidationException.
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def resolve_bedrock_model_id(short_or_full: str) -> str:
    """Resolve an agent's ``llm.model`` value into a Bedrock inference profile ID.

    Rules:

    1. Empty / falsy → :data:`_BEDROCK_DEFAULT_MODEL`.
    2. Already a full Bedrock ID (contains a dot) → pass through.
    3. Known short form → mapped via :data:`_SHORT_TO_BEDROCK`.
    4. Short form with a trailing ``-YYYYMMDD`` suffix → strip the
       suffix and look up the un-dated form (e.g.
       ``claude-haiku-4-5-20251001`` → ``claude-haiku-4-5``).
    5. Unknown short form → log warning, pass through (Bedrock will
       reject with a clearer error than we can produce).

    See the module-level comment on :data:`_SHORT_TO_BEDROCK` for
    the production-incident background.
    """
    if not short_or_full:
        return _BEDROCK_DEFAULT_MODEL
    if "." in short_or_full:
        return short_or_full
    mapped = _SHORT_TO_BEDROCK.get(short_or_full)
    if mapped:
        return mapped
    # Try stripping a trailing -YYYYMMDD suffix.
    stripped = _DATE_SUFFIX_RE.sub("", short_or_full)
    if stripped != short_or_full:
        mapped = _SHORT_TO_BEDROCK.get(stripped)
        if mapped:
            return mapped
    logger.warning(
        "bedrock_model_id_unknown_short_form",
        input=short_or_full,
        hint=(
            "Add an entry to _SHORT_TO_BEDROCK in app/services/factory.py "
            "if this is a valid Aurora llm_model. Passing through as-is; "
            "Bedrock may reject."
        ),
    )
    return short_or_full


def build_stt(agent: AgentConfig, settings: Settings) -> AssemblyAISTTService:
    """Construct the AssemblyAI STT service with Mode-2 locked-in settings.

    The agent contributes only ``stt.keywords`` (wired into
    ``keyterms_prompt`` for AssemblyAI's keyword-boost feature). The
    rest are platform constants.

    The API key comes from Layer 2 ``Settings.assemblyai_api_key`` (the
    Fargate task definition injects it from Secrets Manager as the
    ``ASSEMBLYAI_API_KEY`` env var; ``Settings`` reads it at boot). This
    keeps secrets behind the ``Settings`` boundary rather than reading
    ``os.environ`` directly in Layer 3.
    """
    api_key = settings.assemblyai_api_key
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY not set in environment")

    settings_kwargs: dict[str, object] = {
        "model": _STT_MODEL,
        "language": Language.EN,
        "vad_threshold": _STT_VAD_THRESHOLD,
    }

    # Wire per-agent keywords. v1 read the field but only passed it
    # to the Deepgram branch (now removed); on AssemblyAI calls the
    # keywords were silently dropped. v2 closes that gap. Pipecat's
    # _NotGiven sentinel means "field unset"; we set the field only
    # when the agent actually configured keywords so empty-list
    # agents stay at the vendor default.
    if agent.stt.keywords:
        settings_kwargs["keyterms_prompt"] = list(agent.stt.keywords)

    return AssemblyAISTTService(
        api_key=api_key,
        sample_rate=_STT_SAMPLE_RATE,
        encoding=_STT_ENCODING,
        vad_force_turn_endpoint=_STT_VAD_FORCE_TURN_ENDPOINT,
        should_interrupt=_STT_SHOULD_INTERRUPT,
        settings=AssemblyAISTTService.Settings(**settings_kwargs),
    )


def build_tts(agent: AgentConfig, settings: Settings) -> ElevenLabsTTSService:
    """Construct the ElevenLabs WebSocket TTS service.

    Per-agent: ``tts.voice_id``, ``tts.model``,
    ``tts.settings.stability``, ``tts.settings.use_speaker_boost``.
    Each falls through to the platform default (or the vendor's own
    default) when unset.

    The API key comes from Layer 2 ``Settings.elevenlabs_api_key`` (the
    Fargate task definition injects it from Secrets Manager as the
    ``ELEVENLABS_API_KEY`` env var; ``Settings`` reads it at boot) —
    not from ``os.environ``.
    """
    api_key = settings.elevenlabs_api_key
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set in environment")

    voice = agent.tts.voice_id or _ELEVENLABS_DEFAULT_VOICE_ID
    model = agent.tts.model or _ELEVENLABS_DEFAULT_MODEL

    settings_kwargs: dict[str, object] = {
        "voice": voice,
        "model": model,
    }

    # Per-agent voice tuning. Only pass fields the agent set —
    # Pipecat's _NotGiven sentinel preserves vendor defaults when
    # the field is omitted, but None would clobber them.
    for field in ("stability", "use_speaker_boost"):
        value = getattr(agent.tts.settings, field, None)
        if value is not None:
            settings_kwargs[field] = value

    return ElevenLabsTTSService(
        api_key=api_key,
        settings=ElevenLabsTTSService.Settings(**settings_kwargs),
    )


# ── Live mid-call Bedrock model failover (#52) ─────────────────────────────
#
# When the live (in-call) Bedrock model is throttled / at capacity /
# not-ready, fail over to a backup Bedrock model so the call doesn't break.
# The seam is ``AWSBedrockLLMService._create_converse_stream`` (see
# _FailoverBedrockLLMService below): a *retryable* Bedrock error fires there,
# at the ``converse_stream`` call, BEFORE any tokens stream to TTS — so we can
# swap the model and retry without ever double-speaking. Mid-stream failures
# (after tokens started) are out of scope.

# Bedrock error codes that justify retrying the LIVE turn on the NEXT model in
# the fallback chain — transient / capacity conditions a different model (or
# its capacity pool) may survive. Anything else (AccessDeniedException,
# ValidationException, a bad model id) is a hard failure another model won't
# fix, so it propagates unchanged to Pipecat's existing push_error path.
#
# Intentionally a LOCAL copy of post_call._RETRYABLE_FAILOVER_ERROR_CODES
# (the OFFLINE path, #20): Layer 3 (this factory) must not import from Layer 6
# (post_call), and keeping it here avoids touching post_call.py. The two lists
# describe the same Bedrock transient-error surface; keep them in sync.
_RETRYABLE_FAILOVER_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
    }
)


def _is_retryable_bedrock_error(exc: Exception) -> bool:
    """True when ``exc`` is a transient Bedrock condition worth retrying on the
    NEXT model in the live fallback chain (#52).

    Retryable: connect / read timeouts (botocore ``*TimeoutError``) and the
    throttling / capacity error codes in :data:`_RETRYABLE_FAILOVER_ERROR_CODES`.
    Everything else — auth (``AccessDeniedException``), validation (a bad model
    id), etc. — is a hard failure another model won't fix, so the override
    re-raises and Pipecat handles it as today. Mirrors
    ``post_call._is_retryable_failover_error``.
    """
    if isinstance(exc, (ReadTimeoutError, ConnectTimeoutError)):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _RETRYABLE_FAILOVER_ERROR_CODES
    return False


def _parse_model_fallback_chain(csv_string: str | None) -> list[str]:
    """Parse the ``live_model_fallback_chain`` CSV into an ordered list of
    model short-names.

    Whitespace-stripped, empty entries dropped, order preserved — mirrors
    ``post_call._parse_fallback_chain`` and Layer 8's CSV parsers. Empty /
    unset → ``[]`` (no failover, today's behavior).
    """
    if not csv_string:
        return []
    return [m.strip() for m in csv_string.split(",") if m.strip()]


def resolve_live_model_chain(primary_short: str, settings: Settings) -> list[str]:
    """Build the ordered, de-duplicated list of Bedrock inference-profile IDs the
    LIVE in-call LLM should try (#52): the primary model first, then each
    fallback from ``Settings.live_model_fallback_chain``.

    Every entry is resolved through :func:`resolve_bedrock_model_id` (the same
    path as the primary), so short-names, dated short-names, and full IDs all
    normalize identically; an unknown short-name warns and passes through
    rather than being silently dropped. Duplicates (e.g. a fallback equal to
    the primary) are dropped so the same model is never re-tried. Always
    returns at least one element (the primary) — so an empty chain yields
    ``[primary]`` and :func:`build_llm` builds the plain service.
    """
    chain: list[str] = []
    seen: set[str] = set()
    for short in (
        primary_short,
        *_parse_model_fallback_chain(settings.live_model_fallback_chain),
    ):
        bedrock_id = resolve_bedrock_model_id(short)
        if bedrock_id not in seen:
            seen.add(bedrock_id)
            chain.append(bedrock_id)
    return chain


class _FailoverBedrockLLMService(AWSBedrockLLMService):
    """``AWSBedrockLLMService`` that fails over to a backup Bedrock model on a
    retryable error at the live ``converse_stream`` call (#52).

    Only constructed by :func:`build_llm` when ``Settings.live_model_fallback_chain``
    is non-empty; otherwise the plain ``AWSBedrockLLMService`` is built and this
    class is never involved (flag-off path is byte-identical).

    ⚠️ PINNED-1.1.0 INTERNAL: this overrides
    ``AWSBedrockLLMService._create_converse_stream`` — a Pipecat-private method.
    Safe under the exact ``pipecat-ai==1.1.0`` pin (see CLAUDE.md / #26); a
    Pipecat bump MUST re-validate this seam. The failover tests call this
    override directly, so a signature/behavior change fails the suite loudly.

    Failover semantics:

    * On a *retryable* Bedrock error (throttle / capacity / model-not-ready /
      connect+read timeout) with another model left in the chain, swap
      ``request_params["modelId"]`` to the next model and retry the call.
    * On a *non-retryable* error (validation / auth) or the last model in the
      chain, re-raise — Pipecat's ``_process_context`` handles it via
      ``push_error`` exactly as today.
    * After a successful failover the surviving model **sticks** for the rest
      of the call (``self._settings.model`` is updated), so subsequent turns
      start there instead of re-paying a throttled primary's latency. This is
      strictly call-scoped: ``build_llm`` constructs one service instance per
      call, so there is no cross-call leakage.

    Out of scope: mid-stream failures (errors after tokens have started
    streaming — the partial text is already on the TTS path), and the
    out-of-band ``run_inference`` summarization path (it uses ``converse``,
    not ``converse_stream``, and is best-effort off the critical path).
    """

    def __init__(self, *, fallback_model_ids: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Models to try AFTER the primary, in order. Already resolved Bedrock
        # inference-profile IDs (resolve_live_model_chain), de-duped against
        # the primary.
        self._fallback_model_ids = list(fallback_model_ids)

    async def _create_converse_stream(self, client: Any, request_params: dict[str, Any]) -> Any:
        # Ordered chain: whatever model the caller set first (normally the
        # primary, or the stuck-to fallback from a prior turn), then any
        # configured fallback not already at the head.
        current = request_params["modelId"]
        chain = [current, *[m for m in self._fallback_model_ids if m != current]]
        last_exc: Exception | None = None
        for index, model_id in enumerate(chain):
            request_params["modelId"] = model_id
            is_last = index == len(chain) - 1
            try:
                response = await super()._create_converse_stream(client, request_params)
            except Exception as exc:  # noqa: BLE001 — classify, then re-raise
                if _is_retryable_bedrock_error(exc) and not is_last:
                    logger.warning(
                        "live_llm_model_failover",
                        from_model=model_id,
                        to_model=chain[index + 1],
                        error_type=type(exc).__name__,
                    )
                    last_exc = exc
                    continue
                # Non-retryable, or no model left: let Pipecat's
                # _process_context handle it (push_error) as today.
                raise
            if index > 0:
                # Stick to the surviving model for the rest of this (call-
                # scoped) instance's turns.
                self._settings.model = model_id
                logger.warning(
                    "live_llm_model_failover_succeeded",
                    model=model_id,
                    models_tried=index + 1,
                )
            return response
        # Unreachable: the loop returns on success and re-raises on the last
        # model's failure. Re-raise defensively to satisfy the type checker.
        raise last_exc  # type: ignore[misc]


def build_llm(
    agent: AgentConfig,
    settings: Settings,
    system_instruction: str | None = None,
) -> AWSBedrockLLMService:
    """Construct the AWS Bedrock Claude LLM service.

    Per-agent: ``llm.model`` (resolved to a full Bedrock inference
    profile ID), ``llm.max_tokens``, ``llm.temperature``,
    ``system_prompt`` (passed via ``Settings.system_instruction``).

    Hardcoded: ``enable_prompt_caching=True``. v1 had a per-agent
    flag that was logged but never wired into the constructor — the
    field was a no-op. v2 turns prompt caching on for every call so
    long-form system prompts (Cosentus's are several KB) only pay
    full input cost on the first request and cached cost
    thereafter.

    Cache mechanics (verified for #21): Pipecat appends a single
    ``{"cachePoint": {"type": "default"}}`` to the **end of the
    Bedrock ``system`` block** — i.e. right after the system prompt
    and **before** the per-turn ``messages``. That is the correct
    breakpoint boundary: the cached prefix is exactly the system
    prompt, which is seeded once from ``system_instruction`` and is
    therefore byte-identical turn-to-turn **within a call**, so
    turn 2+ reads the cached prefix (``cache_read`` > 0). It is not
    stable **across** calls — Layer 5 hydration injects per-call
    values (e.g. ``{{current_time}}``) — so the win is within-call
    multi-turn, which is exactly what live calls need. The Bedrock
    ``cachePoint`` only supports the 5-minute ``default`` TTL (the
    1-hour TTL is a first-party-Anthropic-API feature, not available
    on Bedrock Converse); 5 minutes covers every turn of a live call
    since turns are seconds apart. Caveat: on Claude Haiku 4.5 (the
    live model) the minimum cacheable prefix is ~4096 tokens, so a
    short agent prompt silently won't cache even though the
    breakpoint is correct — Cosentus's multi-KB prompts clear it.
    Hit rate is observable via the ``voice.llm.tokens.cache_read`` /
    ``voice.llm.tokens.cache_creation`` span attributes (#28).

    The system prompt is passed via ``Settings.system_instruction``
    rather than as a ``role="system"`` entry in the
    :class:`LLMContext` messages list. Pipecat's context-management
    docs warn that a system message in the context can be lost
    during context summarization or full replacement; the Bedrock
    service in particular does not surface it as a system prompt at
    all — it is silently rendered as user input. Walking-skeleton
    run #1 reproduced this exactly (Claude responded to the 35 KB
    identity prompt as if it were a user message). The
    ``system_instruction`` path is the documented Pipecat pattern.

    Args:
        agent: The per-call agent config.
        settings: Layer 2 settings (provides ``aws_region``).
        system_instruction: Optional pre-hydrated system prompt
            override. When provided (e.g. by Layer 8 after running
            Layer 5's hydrator over ``agent.system_prompt``), this
            string is used verbatim. When ``None`` (default), falls
            back to ``agent.system_prompt`` raw — preserves
            backward compatibility with callers that don't yet
            hydrate.

    Region comes from Layer 2 ``Settings.aws_region``. Credentials
    use boto3's default chain (Fargate task IAM role) — no explicit
    keys passed.

    Live model failover (#52): when ``Settings.live_model_fallback_chain``
    is non-empty this returns a :class:`_FailoverBedrockLLMService` that
    fails the in-call turn over to the next Bedrock model on a retryable
    error (throttle / capacity / timeout) raised before any tokens stream.
    With the chain empty (default) the plain ``AWSBedrockLLMService`` is
    returned and behavior is unchanged.
    """
    model_id = resolve_bedrock_model_id(agent.llm.model)

    # Prefer the hydrated override; fall back to raw agent prompt
    # for backwards compatibility (existing callers and tests).
    effective_system_prompt = (
        system_instruction if system_instruction is not None else agent.system_prompt
    )

    llm_settings_kwargs: dict[str, object] = {
        # The system prompt always lives on the service via
        # system_instruction — not in LLMContext.messages — so it
        # survives context summarization and Bedrock honors it as
        # an actual system prompt.
        "system_instruction": effective_system_prompt,
        # Hardcoded ON. v1 had a per-agent flag that was always
        # logged but never wired — so caching was effectively off
        # platform-wide. Closing that gap here.
        "enable_prompt_caching": True,
    }

    if agent.llm.max_tokens is not None:
        llm_settings_kwargs["max_tokens"] = agent.llm.max_tokens
    if agent.llm.temperature is not None:
        llm_settings_kwargs["temperature"] = agent.llm.temperature

    # Live mid-call model failover (#52). ``chain[0]`` is always the resolved
    # primary (== model_id); any operator-configured fallbacks follow. With no
    # fallback configured (the default) the chain is just the primary, so we
    # build the plain service and the failover subclass is never involved —
    # the flag-off path is byte-identical to before.
    model_chain = resolve_live_model_chain(agent.llm.model, settings)
    service_kwargs: dict[str, Any] = {
        "model": model_id,
        "aws_region": settings.aws_region,
        "settings": AWSBedrockLLMService.Settings(**llm_settings_kwargs),
    }
    if len(model_chain) > 1:
        return _FailoverBedrockLLMService(
            fallback_model_ids=model_chain[1:],
            **service_kwargs,
        )
    return AWSBedrockLLMService(**service_kwargs)
