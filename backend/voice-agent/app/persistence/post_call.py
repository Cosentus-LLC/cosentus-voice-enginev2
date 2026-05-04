"""Post-call structured extraction over Bedrock.

After a call ends, Layer 6 reads the agent's per-agent
:class:`~app.config.agent_config.PostCallConfig` field schema and
runs a single Bedrock LLM call to extract structured values from the
transcript + ``case_data``. The result lands in
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
* Bedrock API error → ``{}`` (no model-fallback chain — keep one
  primary model, fast fail)
* Invalid JSON in response → retry once, then ``{}``
* Non-dict JSON shape → ``{}``

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
from botocore.exceptions import BotoCoreError, ClientError

from app.config.agent_config import AgentConfig, PostCallConfig
from app.config.settings import Settings
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

# Default model when the agent's ``post_call_analyses.model`` is
# empty. Haiku is the right default — fast, cheap, and structured
# extraction over a 5-minute transcript fits well inside its context
# window. Agents with denser extraction needs override via the
# per-agent setting.
_DEFAULT_PCA_MODEL = "claude-haiku-4-5"

# One retry on a JSON-parse failure. After the second failure we
# return ``{}`` rather than chaining further retries — the typical
# parse failure is the model wrapping output in markdown despite our
# instructions; if a strip-and-retry doesn't help, the model is
# producing malformed JSON for a transcript-specific reason and
# more retries won't fix it.
_MAX_PARSE_RETRIES = 1

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
    model_short = pca_config.model or _DEFAULT_PCA_MODEL
    bedrock_id = resolve_bedrock_model_id(model_short)

    last_error: str | None = None
    for attempt in range(_MAX_PARSE_RETRIES + 1):
        try:
            raw = await _invoke_bedrock(bedrock_id, prompt, settings)
        except (BotoCoreError, ClientError) as exc:
            api_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            logger.error(
                "post_call_invoke_failed",
                agent_name=agent.name,
                model=bedrock_id,
                error=str(exc),
                error_type=type(exc).__name__,
                api_code=api_code,
                attempt=attempt + 1,
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
            return {}

        parsed = _parse_and_validate(raw, pca_config)
        if parsed is not None:
            logger.info(
                "post_call_completed",
                agent_name=agent.name,
                model=bedrock_id,
                field_count=len(pca_config.fields),
                fields_filled=len([v for v in parsed.values() if v]),
                attempt=attempt + 1,
            )
            return parsed

        last_error = f"invalid_json (attempt {attempt + 1})"
        logger.warning(
            "post_call_invalid_json",
            agent_name=agent.name,
            attempt=attempt + 1,
            response_excerpt=raw[:300] if raw else "",
        )

    logger.error(
        "post_call_exhausted_retries",
        agent_name=agent.name,
        last_error=last_error,
    )
    return {}


def _build_extraction_prompt(
    pca_config: PostCallConfig,
    case_data: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> str:
    """Build the extraction prompt fed to Bedrock.

    Format optimized for v2's clean JSON output. Sections are
    explicitly labeled so the model knows what's input vs what to
    produce. The field-list section calls out type + choices for
    each field; we tell the model exactly what to return.

    The transcript is rendered with turn numbers and speakers so the
    model can reason about ordering ("the operator said X *after*
    confirming the claim ID").
    """
    lines: list[str] = [
        "You are extracting structured data from a completed phone call.",
        "Your job is to read the transcript + case data and return a single JSON object.",
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
            "Return ONLY a single JSON object whose keys are the field "
            "names above. No prose, no markdown code fences, no "
            "explanation — just JSON.",
            "",
            "If a field cannot be determined from the transcript, return "
            "an empty string for that field.",
            "For selector fields, return one of the listed choices verbatim.",
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


async def _invoke_bedrock(model_id: str, prompt: str, settings: Settings) -> str:
    """Call Bedrock Converse with a single user message; return text.

    Synchronous boto3 wrapped in :func:`asyncio.to_thread` so it
    plays nicely with the async pipeline coroutine. Returns the
    first text block from the response, ``""`` if no text was
    returned (rare; usually means the model hit a content filter).
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
    )
    output = response.get("output", {})
    message = output.get("message", {})
    blocks = message.get("content") or []
    for block in blocks:
        text = block.get("text")
        if isinstance(text, str) and text:
            return text
    return ""


def _parse_and_validate(raw: str, pca_config: PostCallConfig) -> dict[str, Any] | None:
    """Parse Bedrock's response and coerce to the field schema.

    Returns:
        * ``dict`` keyed by field name with validated values — caller
          should treat this as the final output.
        * ``None`` — JSON failed to parse. Caller may retry.

    Selector validation matches v1: invalid choices keep the model's
    output but get a ``"invalid: "`` prefix so operators see what
    actually came back. Empty / missing fields are coerced to ``""``.
    """
    if not raw:
        return None

    cleaned = raw.strip()
    # Strip markdown code fences if the model added them despite the
    # explicit "no markdown" instruction. Both ```json …``` and bare
    # ``` …``` patterns surface in the wild.
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()
        else:
            # Single-line fence-wrapped text; strip both ends.
            cleaned = cleaned.strip("`").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    result: dict[str, Any] = {}
    for f in pca_config.fields:
        raw_value = parsed.get(f.name)
        if f.type == "selector":
            value_str = str(raw_value) if raw_value is not None else ""
            if not value_str:
                result[f.name] = ""
            elif value_str in f.choices:
                result[f.name] = value_str
            else:
                # v1 semantics: keep the model's value with an
                # ``invalid:`` marker rather than swallowing it.
                # Operators triaging a call see what the LLM actually
                # said; switching to "" would hide the failure.
                result[f.name] = f"invalid: {value_str}"
        else:
            # Default branch covers ``text`` and any future type the
            # lambda may add — coerce to string, blank if missing.
            result[f.name] = str(raw_value) if raw_value is not None else ""

    return result
