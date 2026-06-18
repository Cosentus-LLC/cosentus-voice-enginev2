"""Observability — per-call OpenTelemetry tracing (PHI-free) for the engine.

See :mod:`app.observability.tracing` for the design rationale (why we emit our
own spans instead of Pipecat's PHI-bearing tracer) and the span model.
"""

from __future__ import annotations

from app.observability.tracing import (
    context_of,
    end_call_span,
    init_tracing,
    is_tracing_available,
    llm_span,
    set_span_attrs,
    shutdown_tracing,
    start_call_span,
    tool_span,
)

__all__ = [
    "context_of",
    "end_call_span",
    "init_tracing",
    "is_tracing_available",
    "llm_span",
    "set_span_attrs",
    "shutdown_tracing",
    "start_call_span",
    "tool_span",
]
