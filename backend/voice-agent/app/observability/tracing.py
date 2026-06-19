"""Per-call OpenTelemetry tracing — PHI-free spans → self-hosted Langfuse (#13).

Why we roll our own spans
-------------------------

Pipecat 1.1.0 ships built-in GenAI tracing (``PipelineTask(enable_tracing=True)``)
but it embeds **PHI** directly in span attributes with no suppression flag:
``utils/tracing/service_attributes.py`` writes the full LLM ``input`` (messages)
+ ``output`` + ``gen_ai.system_instructions`` (our *hydrated* prompt, which
carries ``case_data``) on the LLM span, the ``transcript`` on the STT span, and
the spoken ``text`` on the TTS span. For a HIPAA-adjacent live-call system that's
a non-starter, so we do **not** enable Pipecat's tracer.

Instead this module emits our own spans where **PHI is never created in the first
place** — deny-by-default, only ids / timings / counts / token-usage / model-id /
status / error-type ever reach a span. Per-stage live STT→LLM→TTS timing comes
from Pipecat's separate, fully-numeric *metrics* path
(:mod:`app.observers.metrics_observer`), not its tracer.

Span model (one trace per call)
-------------------------------

* ``voice.call`` — root orchestration span (per call), keyed by ``call_id`` /
  ``session_id``. Created in :func:`~app.bot.bot.run_bot`.
* ``voice.tool`` — one per tool execution (:mod:`app.tools.executor`), child of
  the root via an explicitly-propagated OTel context.
* ``voice.post_call`` — the post-call extraction LLM call
  (:mod:`app.persistence.post_call`), child of the root.

Children parent to the root by an **explicit** ``Context`` captured with
:func:`context_of` and threaded through ``ToolContext.otel_context`` /
``finalize_call(otel_parent_context=…)`` — never via ambient contextvar
propagation, which is fragile across Pipecat's internal task boundaries.

Fail-open / fail-safe
---------------------

Tracing is observe-only. Every function here is wrapped so a telemetry failure
**never** breaks a call: when tracing is disabled, the OTel SDK isn't installed,
or setup/export fails, the helpers degrade to no-ops (spans are ``None``) and the
pipeline behaves identically. Defaults keep tracing **off** — an operator opts in
by setting the Langfuse endpoint + keys (see :class:`~app.config.settings.Settings`).
"""

from __future__ import annotations

import base64
import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from app.config.settings import Settings

logger = structlog.get_logger(__name__)

# OpenTelemetry is an optional import surface: if the SDK isn't present the whole
# module degrades to no-ops. Mirrors Pipecat's own ``OPENTELEMETRY_AVAILABLE``
# guard so a slimmed-down environment (or a future dep removal) can't crash boot.
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only in OTel-less environments
    _OTEL_AVAILABLE = False

_TRACER_NAME = "app.observability"
_SERVICE_NAME = "voice-engine"

# Module-scoped tracer + provider, installed by :func:`init_tracing`. ``None``
# means tracing is off — every helper short-circuits to a no-op. We hold our own
# provider reference (rather than relying on the global tracer provider) so the
# helpers and :func:`shutdown_tracing` stay decoupled from process-global state.
_TRACER: Any = None
_PROVIDER: Any = None
_INITIALIZED = False


def is_tracing_available() -> bool:
    """Whether the OpenTelemetry SDK is importable in this environment."""
    return _OTEL_AVAILABLE


def init_tracing(settings: Settings) -> bool:
    """Set up the OTel SDK + OTLP→Langfuse exporter. Idempotent; fail-open.

    Called once at process boot (:func:`app.main.amain`). Returns ``True`` when
    tracing is live, ``False`` (no-op) when disabled, the SDK is missing, or
    setup fails — in every ``False`` case the span helpers degrade to no-ops and
    calls run unchanged.

    Args:
        settings: Engine settings. Reads ``tracing_enabled``,
            ``otel_exporter_otlp_endpoint``, ``langfuse_public_key`` /
            ``langfuse_secret_key`` (for OTLP Basic auth), and ``environment``
            (the ``deployment.environment`` resource attribute).

    Returns:
        ``True`` if tracing was initialized and is exporting; ``False`` otherwise.
    """
    global _TRACER, _PROVIDER, _INITIALIZED

    if _INITIALIZED:
        return _TRACER is not None
    _INITIALIZED = True

    if not settings.tracing_enabled:
        logger.info("tracing_disabled")
        return False

    if not _OTEL_AVAILABLE:
        logger.warning("tracing_unavailable_otel_not_installed")
        return False

    if not settings.otel_exporter_otlp_endpoint:
        logger.warning("tracing_enabled_but_no_endpoint")
        return False

    try:
        resource = Resource.create(
            {
                "service.name": _SERVICE_NAME,
                "deployment.environment": settings.environment,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            headers=_auth_headers(settings),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _PROVIDER = provider
        _TRACER = provider.get_tracer(_TRACER_NAME)
        logger.info(
            "tracing_initialized",
            endpoint=settings.otel_exporter_otlp_endpoint,
            service_name=_SERVICE_NAME,
            environment=settings.environment,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — fail-open: never break boot
        logger.error(
            "tracing_init_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        _TRACER = None
        _PROVIDER = None
        return False


def shutdown_tracing() -> None:
    """Flush + shut down the tracer provider. Fail-open; called on drain."""
    if _PROVIDER is None:
        return
    try:
        _PROVIDER.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.warning("tracing_shutdown_failed", error=str(exc))


def _auth_headers(settings: Settings) -> dict[str, str]:
    """Build the OTLP Basic-auth header for self-hosted Langfuse.

    Langfuse's OTLP endpoint authenticates with
    ``Authorization: Basic base64(public_key:secret_key)``. Returns an empty
    dict when either key is unset, so a misconfigured-but-enabled deployment
    still exports (and fails visibly at the collector) rather than crashing here.
    """
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if not public_key or not secret_key:
        return {}
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def context_of(span: Any) -> Any:
    """Return the OTel ``Context`` for *span*, for explicit child parenting.

    Returns ``None`` when *span* is ``None`` or OTel is unavailable — callers
    pass the result straight back as ``tool_span(parent_context=…)`` / the
    ``otel_parent_context`` kwarg, where ``None`` means "no parent / no-op".
    """
    if span is None or not _OTEL_AVAILABLE:
        return None
    try:
        return trace.set_span_in_context(span)
    except Exception:  # noqa: BLE001
        return None


def set_span_attrs(span: Any, attributes: dict[str, Any]) -> None:
    """Set attributes on *span*, guarding ``None`` and swallowing failures.

    Keys are dotted (``"voice.tool.status"``), so this takes a dict rather than
    ``**kwargs``. No-op when *span* is ``None`` (tracing off). **Callers must
    only pass PHI-free values** — ids / timings / counts / status / error-type.
    """
    if span is None:
        return
    try:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("set_span_attrs_failed", error=str(exc))


def start_call_span(
    *,
    call_id: str,
    session_id: str | None,
    agent_id: str,
    direction: str,
) -> Any:
    """Start the per-call root ``voice.call`` span. Returns the span or ``None``.

    The caller owns the lifecycle: capture the parent context with
    :func:`context_of`, then close the span with :func:`end_call_span` in its
    ``finally``. Returns ``None`` (no-op) when tracing is off.
    """
    if _TRACER is None:
        return None
    try:
        span = _TRACER.start_span("voice.call")
        attrs: dict[str, Any] = {
            "voice.call_id": call_id,
            "voice.agent_id": agent_id,
            "voice.direction": direction,
        }
        if session_id:
            attrs["voice.session_id"] = session_id
        set_span_attrs(span, attrs)
        return span
    except Exception as exc:  # noqa: BLE001
        logger.debug("start_call_span_failed", error=str(exc))
        return None


def end_call_span(
    span: Any,
    *,
    end_status: str,
    duration_secs: int | None = None,
    transcript_turns: int | None = None,
    error_type: str | None = None,
) -> None:
    """Set terminal attributes on the ``voice.call`` span and end it. Fail-open.

    ``error_type`` is the exception *class name* only — never the message text,
    which can carry PHI.
    """
    if span is None:
        return
    set_span_attrs(
        span,
        {
            "voice.end_status": end_status,
            "voice.duration_secs": duration_secs,
            "voice.transcript_turns": transcript_turns,
            "voice.error_type": error_type,
        },
    )
    with contextlib.suppress(Exception):
        span.end()


@contextlib.contextmanager
def tool_span(*, tool_name: str, parent_context: Any = None) -> Iterator[Any]:
    """Context manager wrapping one tool execution as a ``voice.tool`` span.

    Yields the span (or ``None`` when tracing is off). Parents to *parent_context*
    (the call root). Only the tool *name* is recorded here — callers add
    status / timing / error-type via :func:`set_span_attrs`; arguments and
    results are PHI and must never be attached.
    """
    if _TRACER is None:
        yield None
        return
    span = None
    try:
        span = _TRACER.start_span("voice.tool", context=parent_context)
        set_span_attrs(
            span,
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool_name,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("tool_span_start_failed", error=str(exc))
        yield None
        return
    try:
        yield span
    finally:
        with contextlib.suppress(Exception):
            span.end()


@contextlib.contextmanager
def llm_span(*, operation: str, model: str, parent_context: Any = None) -> Iterator[Any]:
    """Context manager for an LLM call (the post-call extraction) as ``voice.post_call``.

    Yields the span (or ``None``). Sets GenAI-convention ``gen_ai.operation.name``
    + ``gen_ai.request.model`` (the model id is allowed). Callers add token-usage
    / field counts / status — never the prompt, transcript, or extracted values.
    """
    if _TRACER is None:
        yield None
        return
    span = None
    try:
        span = _TRACER.start_span("voice.post_call", context=parent_context)
        set_span_attrs(
            span,
            {
                "gen_ai.operation.name": operation,
                "gen_ai.request.model": model,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("llm_span_start_failed", error=str(exc))
        yield None
        return
    try:
        yield span
    finally:
        with contextlib.suppress(Exception):
            span.end()
