"""Post-call structured extraction over Bedrock.

After a call ends, Layer 6 reads the agent's per-agent
:class:`~app.config.agent_config.PostCallConfig` field schema and
runs a single Bedrock **Converse call with a forced tool** to extract
structured values from the transcript + ``case_data``. The model's
``toolUse.input`` arrives as an already-parsed object (no free-text
JSON parsing), which is validated against the field schema. The result
lands in
``voice_calls.post_call_analyses`` (JSONB) where the lambda's
``POST /api/auto-actions`` endpoint reads it to populate
``voice_call_costs``, ``voice_call_scores``, and
``voice_auto_actions``.

Field types
-----------

The lambda's wire-side validator restricts field types to ``text``
and ``selector``:

* ``text`` — free-form string, optionally with ``format_examples``.
* ``selector`` — a string from ``field.choices``. If the LLM picks
  something not in the list, we keep v1's behavior and prefix the
  output ``"invalid: <value>"`` so operators see what the model
  actually returned, rather than silently swapping in an empty
  string.

The brief's third type ``boolean`` was deliberately dropped: v1
doesn't support it, and the lambda's
``VALID_POST_CALL_FIELD_TYPES`` rejects it. When we want yes/no
extraction we'll add it as a coordinated lambda + engine change.

Failure semantics
-----------------

Best-effort, never raises:

* Empty / unconfigured fields → ``{}``
* Empty transcript → ``{}``
* Bedrock **retryable** error (throttling / service-unavailable /
  model-not-ready / timeout) → fail over to the next model in
  ``Settings.post_call_model_fallback_chain`` (#20); ``{}`` once the
  chain is exhausted. Empty chain (default) = no failover.
* Bedrock **non-retryable** error (auth, validation, bad model id) →
  ``{}`` immediately, no failover
* No structured tool output in the response → retry once (same model),
  then ``{}`` (not a failover trigger)
* Malformed tool input (not an object) → ``{}``

The two-write pattern in Layer 8 (write empty PCA, then re-write
with PCA after extraction) tolerates a ``{}`` here gracefully —
the call history row already exists with ``post_call_analyses={}``
and just stays at the default.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import boto3
import structlog
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    ReadTimeoutError,
)
from pydantic import BaseModel, ConfigDict, ValidationError, ValidationInfo, model_validator

from app.config.agent_config import AgentConfig, PostCallConfig, PostCallField
from app.config.settings import Settings
from app.observability.tracing import llm_span, set_span_attrs
from app.observers.usage_accumulator import UsageAccumulator
from app.services.factory import resolve_bedrock_model_id

logger = structlog.get_logger(__name__)


# Lazy-initialized Bedrock client. ``None`` until :func:`_get_bedrock_client`
# runs the first time, at which point we construct the client using
# ``settings.aws_region`` and cache it for every subsequent call.
#
# This pattern closes the Entry 11 anti-pattern (region captured at
# import time from ``os.environ`` regardless of what ``Settings`` says).
# Now the function signature's promise — "the settings parameter
# configures the client" — is actually true.
#
# Module-level cache is the AWS-documented multithreading pattern (one
# shared thread-safe client across worker threads), see
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html.
# The race-condition window during the first concurrent ``_get_*``
# calls creates at most one duplicate client which is GC'd; the global
# slot ends up with a single shared instance.
_BEDROCK_CLIENT: Any = None

# Default model when the agent's ``post_call_analyses.model`` is an
# explicit empty string. Defaults to the stronger model (#20 — per-turn
# model routing): the live turns run on Haiku for speed/cost, but the
# once-per-call OFFLINE extraction is off the latency-critical path, so
# it trades cost/latency for accuracy and defaults to Sonnet. Mirrors
# the ``PostCallConfig.model`` default (which covers the omitted-field
# path); agents with a specific need still override via the per-agent
# setting (``pca_config.model or _DEFAULT_PCA_MODEL`` below).
_DEFAULT_PCA_MODEL = "claude-sonnet-4-6"

# Bedrock error codes that justify retrying the extraction on the NEXT
# model in the fallback chain (#20). These are transient / capacity
# conditions where a different model (or its capacity pool) may succeed.
# Anything else — AccessDeniedException, ValidationException, a bad
# model id — is a hard failure that another model won't fix, so it fails
# fast (returns {}) exactly as before. Timeouts are handled separately
# via the botocore exception types (ReadTimeoutError / ConnectTimeoutError).
_RETRYABLE_FAILOVER_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
    }
)

# One retry when the model returns no usable structured output. Under a
# forced ``toolChoice`` the model almost always emits the tool, so the
# rare miss (a content filter, or the model answering in text) is what
# this guards. After the second miss we return ``{}`` rather than
# chaining further retries — if forcing the tool didn't work twice, a
# transcript-specific reason is at play and more retries won't fix it.
_MAX_EXTRACTION_RETRIES = 1

# The single forced tool the model must call to report its extraction.
_TOOL_NAME = "extract_post_call"

# Inference config for the extraction call. ``temperature=0`` is
# critical — variability in extraction defeats the purpose; we want
# the same transcript to produce the same structured output. v1
# also pinned to 0.0 for the same reason.
_EXTRACTION_TEMPERATURE = 0.0
_EXTRACTION_MAX_TOKENS = 2048


async def run_post_call_analyses(
    agent: AgentConfig,
    case_data: dict[str, Any],
    transcript: list[dict[str, Any]],
    settings: Settings,
    *,
    otel_parent_context: Any = None,
    usage_accumulator: UsageAccumulator | None = None,
) -> dict[str, Any]:
    """Run Bedrock structured extraction, return the field dict.

    Args:
        agent: The per-call agent config. Reads
            ``agent.post_call_analyses.model`` and
            ``agent.post_call_analyses.fields``.
        case_data: The hydration variables that fed the call. Passed
            to the LLM so it can cross-reference (e.g. confirm a
            transcript reference matches the dispatched ``claim_id``).
        transcript: Output of
            :meth:`~app.persistence.transcript.TranscriptAccumulator.to_list`
            — the full conversation including tool turns.
        settings: Layer 2 settings. Drives the lazy-initialized
            Bedrock client's region binding (see
            :func:`_get_bedrock_client`).
        otel_parent_context: The ``voice.call`` root span's context (#13).
            When provided, the post-call extraction is recorded as a child
            ``voice.post_call`` LLM span (model id / token usage / field
            counts / status only — no prompt, transcript, or values).
            ``None`` (default) when tracing is off.
        usage_accumulator: Optional per-call usage tally (#28). When provided,
            the extraction call's Converse ``usage`` (input/output tokens) is
            folded in on **every** attempt — including the retry — so the
            tokens this Bedrock call consumes are captured for cost even when
            extraction ultimately yields ``{}``. ``None`` (default) leaves cost
            capture inert.

    Returns:
        ``{field_name: value, ...}`` matching ``agent.post_call_analyses.fields``.
        Empty dict on any failure path. **Never raises.**
    """
    pca_config = agent.post_call_analyses
    if not pca_config or not pca_config.fields:
        logger.debug(
            "post_call_skipped_no_fields",
            agent_name=agent.name,
        )
        return {}

    if not transcript:
        logger.info(
            "post_call_skipped_empty_transcript",
            agent_name=agent.name,
            field_count=len(pca_config.fields),
        )
        return {}

    prompt = _build_extraction_prompt(pca_config, case_data, transcript)
    primary_short = pca_config.model or _DEFAULT_PCA_MODEL
    # Primary first, then any operator-configured offline fallbacks (#20).
    model_chain = _resolve_model_chain(primary_short, settings)

    with llm_span(
        operation="chat",
        model=model_chain[0],
        parent_context=otel_parent_context,
    ) as span:
        set_span_attrs(
            span,
            {
                "voice.post_call.field_count": len(pca_config.fields),
                "voice.post_call.model_chain_len": len(model_chain),
            },
        )

        last_error: str | None = None
        for model_index, bedrock_id in enumerate(model_chain):
            is_last_model = model_index == len(model_chain) - 1
            failover = False
            for attempt in range(_MAX_EXTRACTION_RETRIES + 1):
                try:
                    tool_input, usage = await _invoke_bedrock(
                        bedrock_id, prompt, pca_config, settings
                    )
                except (BotoCoreError, ClientError) as exc:
                    # ``response`` is only present (and a dict) on ClientError;
                    # BotoCoreError subclasses (timeouts) carry it as None or
                    # not at all — guard so api_code extraction can't blow up.
                    api_code = (
                        (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")
                    )
                    # Offline model failover (#20): a *retryable* Bedrock error
                    # (throttling / capacity / timeout) with another model left
                    # in the chain moves on to it rather than giving up. A
                    # non-retryable error, or the last model in the chain, fails
                    # fast → {}, exactly as before this change.
                    if _is_retryable_failover_error(exc) and not is_last_model:
                        logger.warning(
                            "post_call_model_failover",
                            agent_name=agent.name,
                            from_model=bedrock_id,
                            to_model=model_chain[model_index + 1],
                            error_type=type(exc).__name__,
                            api_code=api_code,
                            attempt=attempt + 1,
                        )
                        failover = True
                        break
                    logger.error(
                        "post_call_invoke_failed",
                        agent_name=agent.name,
                        model=bedrock_id,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        api_code=api_code,
                        attempt=attempt + 1,
                        models_tried=model_index + 1,
                    )
                    set_span_attrs(
                        span,
                        {
                            "voice.post_call.success": False,
                            "voice.post_call.attempts": attempt + 1,
                            "voice.post_call.models_tried": model_index + 1,
                            "voice.post_call.error_type": type(exc).__name__,
                        },
                    )
                    return {}
                except Exception as exc:  # noqa: BLE001 — never raise
                    logger.error(
                        "post_call_invoke_unexpected_error",
                        agent_name=agent.name,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        attempt=attempt + 1,
                    )
                    set_span_attrs(
                        span,
                        {
                            "voice.post_call.success": False,
                            "voice.post_call.attempts": attempt + 1,
                            "voice.post_call.models_tried": model_index + 1,
                            "voice.post_call.error_type": type(exc).__name__,
                        },
                    )
                    return {}

                _record_token_usage(span, usage)
                # Cost capture (#28): fold the extraction call's tokens into the
                # per-call tally. Runs on every attempt (the retry consumed tokens
                # too) and regardless of whether parsing below succeeds.
                if usage_accumulator is not None and usage:
                    usage_accumulator.add_llm_usage(
                        usage.get("inputTokens") or 0, usage.get("outputTokens") or 0
                    )

                parsed = (
                    _validate_tool_input(tool_input, pca_config) if tool_input is not None else None
                )
                if parsed is not None:
                    logger.info(
                        "post_call_completed",
                        agent_name=agent.name,
                        model=bedrock_id,
                        field_count=len(pca_config.fields),
                        fields_filled=len([v for v in parsed.values() if v]),
                        attempt=attempt + 1,
                        models_tried=model_index + 1,
                    )
                    set_span_attrs(
                        span,
                        {
                            "voice.post_call.success": True,
                            "voice.post_call.attempts": attempt + 1,
                            "voice.post_call.models_tried": model_index + 1,
                            "voice.post_call.fields_filled": len([v for v in parsed.values() if v]),
                        },
                    )
                    return parsed

                last_error = f"no_structured_output (attempt {attempt + 1})"
                logger.warning(
                    "post_call_no_structured_output",
                    agent_name=agent.name,
                    model=bedrock_id,
                    attempt=attempt + 1,
                    had_tool_input=tool_input is not None,
                )

            if failover:
                continue
            # Inner retries exhausted with no structured output. That is not a
            # retryable Bedrock error, so we do NOT fail over to another model
            # — same as before: give up and return {}.
            break

        logger.error(
            "post_call_exhausted_retries",
            agent_name=agent.name,
            last_error=last_error,
        )
        set_span_attrs(
            span,
            {
                "voice.post_call.success": False,
                "voice.post_call.attempts": _MAX_EXTRACTION_RETRIES + 1,
                "voice.post_call.error_type": "no_tool_output",
            },
        )
        return {}


def _is_retryable_failover_error(exc: Exception) -> bool:
    """True when ``exc`` is a transient Bedrock condition worth retrying on the
    NEXT model in the offline fallback chain (#20).

    Retryable: connect / read timeouts (botocore ``*TimeoutError``), and the
    throttling / capacity error codes in
    :data:`_RETRYABLE_FAILOVER_ERROR_CODES`. Everything else — auth
    (``AccessDeniedException``), validation (a bad model id), etc. — is a
    hard failure another model won't fix, so the caller fails fast.
    """
    if isinstance(exc, (ReadTimeoutError, ConnectTimeoutError)):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _RETRYABLE_FAILOVER_ERROR_CODES
    return False


def _parse_fallback_chain(csv_string: str | None) -> list[str]:
    """Parse the ``post_call_model_fallback_chain`` CSV into an ordered list of
    model short-names.

    Whitespace-stripped, empty entries dropped, order preserved — mirrors
    Layer 8's ``_parse_required_case_data_keys`` and Layer 4's
    ``parse_disabled_tools``. Empty / unset → ``[]`` (no failover, today's
    behavior).
    """
    if not csv_string:
        return []
    return [m.strip() for m in csv_string.split(",") if m.strip()]


def _resolve_model_chain(primary_short: str, settings: Settings) -> list[str]:
    """Build the ordered, de-duplicated list of Bedrock inference-profile IDs to
    try for the offline extraction (#20): the primary model first, then each
    fallback from ``Settings.post_call_model_fallback_chain``.

    Every entry is resolved through :func:`resolve_bedrock_model_id` (the same
    path as the primary), so short-names, dated short-names, and full IDs all
    normalize identically. Duplicates (e.g. a fallback equal to the primary)
    are dropped so the same model is never re-tried. Always returns at least
    one element (the primary).
    """
    chain: list[str] = []
    seen: set[str] = set()
    for short in (
        primary_short,
        *_parse_fallback_chain(settings.post_call_model_fallback_chain),
    ):
        bedrock_id = resolve_bedrock_model_id(short)
        if bedrock_id not in seen:
            seen.add(bedrock_id)
            chain.append(bedrock_id)
    return chain


def _record_token_usage(span: Any, usage: dict[str, Any]) -> None:
    """Attach Bedrock Converse token usage to the post-call span. PHI-free."""
    if not usage:
        return
    set_span_attrs(
        span,
        {
            "gen_ai.usage.input_tokens": usage.get("inputTokens"),
            "gen_ai.usage.output_tokens": usage.get("outputTokens"),
            "gen_ai.usage.total_tokens": usage.get("totalTokens"),
        },
    )


def _build_extraction_prompt(
    pca_config: PostCallConfig,
    case_data: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> str:
    """Build the extraction prompt fed to Bedrock.

    Sections are explicitly labeled so the model knows what's input vs
    what to produce. The field-list section calls out type + choices for
    each field; the structured output itself is enforced by the forced
    ``extract_post_call`` tool schema (see :func:`_build_tool_config`),
    not by free-text instructions.

    The transcript is rendered with turn numbers and speakers so the
    model can reason about ordering ("the operator said X *after*
    confirming the claim ID").
    """
    lines: list[str] = [
        "You are extracting structured data from a completed phone call.",
        f"Read the transcript + case data and call the {_TOOL_NAME} tool with the values.",
        "",
        "## Case data passed into the call",
        json.dumps(case_data, indent=2) if case_data else "(none)",
        "",
        "## Transcript",
    ]
    for turn in transcript:
        turn_num = turn.get("turn_number", "?")
        speaker = turn.get("speaker", "unknown")
        content = turn.get("content", "")
        lines.append(f"[{turn_num}] {speaker}: {content}")

    lines.extend(["", "## Fields to extract"])
    for f in pca_config.fields:
        if f.type == "selector":
            choices = ", ".join(f.choices) if f.choices else "(no choices)"
            lines.append(f"- {f.name} (selector — pick ONE of: {choices})")
        else:
            # Treat anything non-selector as ``text``. v1 wire types
            # are constrained to ``text``/``selector`` by the lambda
            # validator, so this covers the entire valid surface.
            lines.append(f"- {f.name} (text)")
        if f.description:
            lines.append(f"    {f.description}")
        if f.format_examples:
            lines.append(f"    Example format: {f.format_examples[0]!r}")

    lines.extend(
        [
            "",
            "## Output format",
            f"Call the {_TOOL_NAME} tool. Its parameters are the field names above.",
            "",
            "If a field cannot be determined from the transcript, pass an "
            "empty string for that field.",
            "For selector fields, pass one of the listed choices verbatim.",
        ]
    )
    return "\n".join(lines)


def _get_bedrock_client(settings: Settings) -> Any:
    """Return the module-shared ``bedrock-runtime`` client, constructing it
    lazily on first call using ``settings.aws_region``.

    Idempotent — every call after the first returns the cached client.
    The first call pays the construction cost (~1 ms on a warm
    interpreter). The boto3 sync client is thread-safe so the same
    instance is shared across the worker threads
    :func:`asyncio.to_thread` dispatches to.
    """
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        _BEDROCK_CLIENT = boto3.session.Session().client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            config=BotoConfig(
                # Connect quickly — Bedrock's regional endpoint is fast
                # when healthy. Read takes longer; structured extraction
                # with transcript context can run 2-8 s on Haiku.
                connect_timeout=5.0,
                read_timeout=30.0,
                retries={"max_attempts": 2, "mode": "adaptive"},
            ),
        )
    return _BEDROCK_CLIENT


def _build_tool_config(pca_config: PostCallConfig) -> dict[str, Any]:
    """Build the Converse ``toolConfig`` for one forced extraction tool.

    The tool's ``inputSchema`` is generated **dynamically** from
    ``pca_config.fields`` (the per-agent schema): every field is a
    string; ``selector`` fields with choices add an ``enum`` so the
    model is constrained to a listed value. No field is marked
    ``required`` — an omitted field coerces to ``""`` in
    :func:`_validate_tool_input`, preserving the "missing → empty"
    contract. ``toolChoice`` forces the model to emit exactly this tool,
    so the response is structured data rather than free text. Mirrors the
    live pipeline's schema-building (``app/tools/schema.py``).

    Callers guarantee ``pca_config.fields`` is non-empty (the early-out
    in :func:`run_post_call_analyses`), so the tool always has ≥1
    property — Bedrock rejects an empty tool list.
    """
    properties: dict[str, dict[str, Any]] = {}
    for f in pca_config.fields:
        entry: dict[str, Any] = {"type": "string", "description": f.description or f.name}
        if f.type == "selector" and f.choices:
            entry["enum"] = list(f.choices)
        properties[f.name] = entry
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": _TOOL_NAME,
                    "description": "Record the structured values extracted from the call.",
                    "inputSchema": {"json": {"type": "object", "properties": properties}},
                }
            }
        ],
        "toolChoice": {"tool": {"name": _TOOL_NAME}},
    }


async def _invoke_bedrock(
    model_id: str, prompt: str, pca_config: PostCallConfig, settings: Settings
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Call Converse forcing the extraction tool; return ``(tool_input, usage)``.

    Synchronous boto3 wrapped in :func:`asyncio.to_thread` so it plays
    nicely with the async pipeline coroutine. Returns the first
    ``toolUse.input`` object from the response — an **already-parsed
    dict** (no free-text JSON parsing) — alongside the Converse ``usage``
    dict (``{}`` when absent) for the post-call span. Returns
    ``(None, usage)`` if the response carries no tool use (rare under a
    forced ``toolChoice`` — a content filter, or the model answering in
    text); the caller retries, then returns ``{}``.
    """
    client = _get_bedrock_client(settings)
    response = await asyncio.to_thread(
        client.converse,
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": _EXTRACTION_MAX_TOKENS,
            "temperature": _EXTRACTION_TEMPERATURE,
        },
        toolConfig=_build_tool_config(pca_config),
    )
    usage = response.get("usage") or {}
    blocks = response.get("output", {}).get("message", {}).get("content") or []
    for block in blocks:
        tool_use = block.get("toolUse")
        if tool_use and isinstance(tool_use.get("input"), dict):
            return tool_use["input"], usage
    return None, usage


class _ExtractionEnvelope(BaseModel):
    """Pydantic validator for the model's ``toolUse.input``.

    Field names are dynamic per agent (and need not be valid Python
    identifiers), so the model carries no declared fields — it accepts
    arbitrary keys (``extra="allow"``) and a ``mode="before"`` validator
    coerces the raw input to the field schema passed via validation
    ``context``. A non-object input raises (→ ``ValidationError`` →
    ``None`` in :func:`_validate_tool_input`).
    """

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _coerce_to_schema(cls, data: Any, info: ValidationInfo) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("tool input must be a JSON object")
        fields: list[PostCallField] = (info.context or {}).get("fields", [])
        result: dict[str, Any] = {}
        for f in fields:
            raw_value = data.get(f.name)
            if f.type == "selector":
                value_str = str(raw_value) if raw_value is not None else ""
                if not value_str:
                    result[f.name] = ""
                elif value_str in f.choices:
                    result[f.name] = value_str
                else:
                    # v1 semantics: keep the model's value with an
                    # ``invalid:`` marker rather than swallowing it, so
                    # operators triaging a call see what the LLM said.
                    result[f.name] = f"invalid: {value_str}"
            else:
                # Default branch covers ``text`` and any future type the
                # lambda may add — coerce to string, blank if missing.
                result[f.name] = str(raw_value) if raw_value is not None else ""
        return result


def _validate_tool_input(
    tool_input: dict[str, Any], pca_config: PostCallConfig
) -> dict[str, Any] | None:
    """Validate + coerce ``toolUse.input`` to the field schema.

    Returns:
        * ``dict`` keyed by field name with validated values — the final
          output, ``{field: value}`` for every configured field.
        * ``None`` — the input was not a usable object. Caller may retry.

    Same value semantics as the old free-text parser (selector invalid →
    ``"invalid: "`` prefix; empty / missing / non-string → ``str`` or
    ``""``), but driven by structured tool output, so there is no JSON
    parsing to fail.
    """
    try:
        validated = _ExtractionEnvelope.model_validate(
            tool_input, context={"fields": pca_config.fields}
        )
    except ValidationError:
        return None
    return validated.model_dump()
