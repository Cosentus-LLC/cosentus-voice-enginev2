"""Tests for app/observability/tracing.py + the metrics observer (#13).

Coverage:

* ``init_tracing`` fail-open / no-op semantics (disabled, no endpoint).
* OTLP Basic-auth header construction for self-hosted Langfuse.
* The PHI-free span helpers (``voice.call`` / ``voice.tool`` / ``voice.post_call``)
  and that they correlate into a single trace via explicit context propagation.
* The metrics observer folding numeric per-stage timing onto the call span.
* The hard PHI guarantee: a simulated call (call + tool + post-call + metrics)
  emits **no** attribute value containing injected PHI sentinels.
* Exporter failure is fail-open (never raises into a call).

Spans are captured with OTel's :class:`InMemorySpanExporter` wired through a
``SimpleSpanProcessor`` (synchronous export on span end), installed onto the
tracing module's tracer. An autouse fixture saves/restores the module's
tracer state so tests don't leak the global provider into one another.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from app.config.agent_config import AgentConfig, LLMConfig, PostCallConfig, PostCallField
from app.config.settings import Settings
from app.observability import tracing
from app.observability.tracing import (
    _auth_headers,
    context_of,
    end_call_span,
    init_tracing,
    llm_span,
    set_span_attrs,
    start_call_span,
    tool_span,
)
from app.observers.metrics_observer import MetricsObserver
from app.observers.usage_accumulator import UsageAccumulator
from app.persistence.post_call import run_post_call_analyses
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult, error_result, success_result
from app.tools.schema import ToolDefinition, ToolParameter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)

# PHI sentinels — if any of these strings ever land in a span attribute the
# deny-by-default contract has been broken.
PHI_NAME = "Jane-Doe-SENTINEL"
PHI_DOB = "1980-01-02-SENTINEL"
PHI_TRANSCRIPT = "my-social-is-SENTINEL-12345"
PHI_TOOL_ARG = "tool-arg-SENTINEL"


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_tracing_state():
    """Save/restore the tracing module's installed-tracer state per test."""
    saved = (tracing._TRACER, tracing._PROVIDER, tracing._INITIALIZED)
    yield
    tracing._TRACER, tracing._PROVIDER, tracing._INITIALIZED = saved


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    """Install an in-memory tracer onto the module and return the exporter."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    tracing._TRACER = provider.get_tracer("test")
    tracing._PROVIDER = provider
    return exp


def _settings(**overrides) -> Settings:
    base = {
        "voice_api_lambda_name": "test-voice-api",
        "api_key_secret_arn": "arn:aws:secretsmanager:us-east-1:0:secret:test",
    }
    base.update(overrides)
    return Settings(**base)


def _all_attr_values(exp: InMemorySpanExporter) -> list:
    """Every attribute value across every finished span (for PHI scanning)."""
    values: list = []
    for span in exp.get_finished_spans():
        for v in (span.attributes or {}).values():
            values.append(v)
    return values


def _spans_by_name(exp: InMemorySpanExporter) -> dict[str, object]:
    return {s.name: s for s in exp.get_finished_spans()}


# ── init_tracing: fail-open / no-op ───────────────────────────────────────


class TestInitTracing:
    def test_disabled_is_noop(self):
        tracing._TRACER = None
        tracing._PROVIDER = None
        tracing._INITIALIZED = False
        assert init_tracing(_settings(tracing_enabled=False)) is False
        # No tracer installed → span helpers are no-ops.
        assert (
            start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound") is None
        )

    def test_enabled_without_endpoint_is_noop(self):
        tracing._TRACER = None
        tracing._PROVIDER = None
        tracing._INITIALIZED = False
        assert init_tracing(_settings(tracing_enabled=True)) is False

    def test_idempotent(self):
        tracing._TRACER = None
        tracing._PROVIDER = None
        tracing._INITIALIZED = False
        # First call (disabled) sets _INITIALIZED; second returns the same.
        assert init_tracing(_settings(tracing_enabled=False)) is False
        assert init_tracing(_settings(tracing_enabled=True)) is False


class TestAuthHeaders:
    def test_empty_when_keys_missing(self):
        assert _auth_headers(_settings()) == {}

    def test_basic_auth_base64(self):
        headers = _auth_headers(
            _settings(langfuse_public_key="pk-abc", langfuse_secret_key="sk-xyz")
        )
        # base64("pk-abc:sk-xyz")
        import base64

        expected = base64.b64encode(b"pk-abc:sk-xyz").decode("ascii")
        assert headers == {"Authorization": f"Basic {expected}"}


# ── Span helpers: attributes + correlation ────────────────────────────────


class TestCallSpan:
    def test_call_span_attributes_allowlisted(self, exporter):
        span = start_call_span(
            call_id="call-1", session_id="sess-1", agent_id="agent-9", direction="outbound"
        )
        end_call_span(
            span,
            end_status="completed",
            duration_secs=42,
            transcript_turns=7,
            error_type=None,
        )
        finished = _spans_by_name(exporter)["voice.call"]
        attrs = dict(finished.attributes)
        assert attrs == {
            "voice.call_id": "call-1",
            "voice.session_id": "sess-1",
            "voice.agent_id": "agent-9",
            "voice.direction": "outbound",
            "voice.end_status": "completed",
            "voice.duration_secs": 42,
            "voice.transcript_turns": 7,
        }
        # error_type=None must not be set.
        assert "voice.error_type" not in attrs

    def test_error_type_set_when_present(self, exporter):
        span = start_call_span(call_id="c", session_id=None, agent_id="a", direction="inbound")
        end_call_span(span, end_status="failed", error_type="RuntimeError")
        attrs = dict(_spans_by_name(exporter)["voice.call"].attributes)
        assert attrs["voice.error_type"] == "RuntimeError"
        # session_id=None omitted.
        assert "voice.session_id" not in attrs


class TestToolSpanCorrelation:
    def test_tool_span_parented_to_call(self, exporter):
        call = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        ctx = context_of(call)
        with tool_span(tool_name="end_call", parent_context=ctx) as ts:
            set_span_attrs(ts, {"voice.tool.status": "success"})
        end_call_span(call, end_status="completed")

        spans = _spans_by_name(exporter)
        call_span = spans["voice.call"]
        tool = spans["voice.tool"]
        # Same trace, tool is a child of the call root.
        assert tool.context.trace_id == call_span.context.trace_id
        assert tool.parent is not None
        assert tool.parent.span_id == call_span.context.span_id
        assert tool.attributes["gen_ai.tool.name"] == "end_call"


# ── post_call LLM span ─────────────────────────────────────────────────────


def _pca_agent() -> AgentConfig:
    return AgentConfig(
        name="test-agent",
        display_name="Test",
        system_prompt="You are a test agent.",
        llm=LLMConfig(model="claude-sonnet-4-6"),
        post_call_analyses=PostCallConfig(
            model="claude-haiku-4-5",
            fields=[PostCallField(name="summary", type="text", description="A summary.")],
        ),
    )


def _bedrock_response_with_usage(tool_input: dict) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu-1",
                            "name": "extract_post_call",
                            "input": tool_input,
                        }
                    }
                ],
            }
        },
        "usage": {"inputTokens": 123, "outputTokens": 45, "totalTokens": 168},
    }


class TestPostCallSpan:
    @pytest.mark.asyncio
    async def test_post_call_span_correlated_and_token_usage(self, exporter):
        call = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        ctx = context_of(call)
        transcript = [{"turn_number": 1, "speaker": "user", "content": "hello"}]

        with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
            mock_br.converse = MagicMock(
                return_value=_bedrock_response_with_usage({"summary": "done"})
            )
            result = await run_post_call_analyses(
                _pca_agent(), {}, transcript, _settings(), otel_parent_context=ctx
            )
        end_call_span(call, end_status="completed")

        assert result == {"summary": "done"}
        spans = _spans_by_name(exporter)
        pc = spans["voice.post_call"]
        # Correlated under the call root.
        assert pc.context.trace_id == spans["voice.call"].context.trace_id
        assert pc.parent.span_id == spans["voice.call"].context.span_id
        # Token usage + counts + status, model id — no content.
        attrs = dict(pc.attributes)
        assert attrs["gen_ai.request.model"]  # resolved bedrock id present
        assert attrs["gen_ai.usage.input_tokens"] == 123
        assert attrs["gen_ai.usage.output_tokens"] == 45
        assert attrs["gen_ai.usage.total_tokens"] == 168
        assert attrs["voice.post_call.field_count"] == 1
        assert attrs["voice.post_call.fields_filled"] == 1
        assert attrs["voice.post_call.success"] is True

    @pytest.mark.asyncio
    async def test_post_call_span_records_failure(self, exporter):
        from botocore.exceptions import ClientError

        call = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        ctx = context_of(call)
        err = ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse")
        with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
            mock_br.converse = MagicMock(side_effect=err)
            result = await run_post_call_analyses(
                _pca_agent(),
                {},
                [{"turn_number": 1, "speaker": "user", "content": "x"}],
                _settings(),
                otel_parent_context=ctx,
            )
        end_call_span(call, end_status="completed")

        assert result == {}
        attrs = dict(_spans_by_name(exporter)["voice.post_call"].attributes)
        assert attrs["voice.post_call.success"] is False
        assert attrs["voice.post_call.error_type"] == "ClientError"


# ── Metrics observer ────────────────────────────────────────────────────────


def _metrics_frame(*data) -> MetricsFrame:
    return MetricsFrame(data=list(data))


def _pushed(frame):
    fp = MagicMock()
    fp.frame = frame
    return fp


class TestMetricsObserver:
    @pytest.mark.asyncio
    async def test_folds_per_stage_timing_onto_span(self, exporter):
        obs = MetricsObserver(processor_stage={"STT#0": "stt", "LLM#0": "llm", "TTS#0": "tts"})
        await obs.on_push_frame(
            _pushed(
                _metrics_frame(
                    TTFBMetricsData(processor="LLM#0", value=0.5),
                    TTFBMetricsData(processor="STT#0", value=0.1),
                    ProcessingMetricsData(processor="TTS#0", value=0.2),
                    LLMUsageMetricsData(
                        processor="LLM#0",
                        value=LLMTokenUsage(
                            prompt_tokens=100, completion_tokens=20, total_tokens=120
                        ),
                    ),
                    TTSUsageMetricsData(processor="TTS#0", value=88),
                    TurnMetricsData(
                        processor="turn",
                        is_complete=True,
                        probability=0.9,
                        e2e_processing_time_ms=300.0,
                    ),
                )
            )
        )
        span = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        obs.write_to_span(span)
        end_call_span(span, end_status="completed")

        attrs = dict(_spans_by_name(exporter)["voice.call"].attributes)
        assert attrs["voice.llm.ttfb_ms.avg"] == 500.0
        assert attrs["voice.stt.ttfb_ms.avg"] == 100.0
        assert attrs["voice.tts.processing_ms.avg"] == 200.0
        assert attrs["voice.llm.tokens.prompt"] == 100
        assert attrs["voice.llm.tokens.completion"] == 20
        assert attrs["voice.llm.tokens.total"] == 120
        assert attrs["voice.tts.chars"] == 88
        assert attrs["voice.turns.count"] == 1
        assert attrs["voice.turns.e2e_ms.avg"] == 300.0

    @pytest.mark.asyncio
    async def test_dedups_by_frame_id(self, exporter):
        obs = MetricsObserver(processor_stage={"LLM#0": "llm"})
        frame = _metrics_frame(TTFBMetricsData(processor="LLM#0", value=0.4))
        await obs.on_push_frame(_pushed(frame))
        await obs.on_push_frame(_pushed(frame))  # same frame.id → ignored
        span = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        obs.write_to_span(span)
        end_call_span(span, end_status="completed")
        attrs = dict(_spans_by_name(exporter)["voice.call"].attributes)
        assert attrs["voice.llm.ttfb.count"] == 1

    @pytest.mark.asyncio
    async def test_ignores_unmapped_processor_timing(self, exporter):
        obs = MetricsObserver(processor_stage={"LLM#0": "llm"})
        await obs.on_push_frame(
            _pushed(_metrics_frame(TTFBMetricsData(processor="Transport#0", value=0.9)))
        )
        span = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        obs.write_to_span(span)
        end_call_span(span, end_status="completed")
        attrs = dict(_spans_by_name(exporter)["voice.call"].attributes)
        # No stage timing emitted for an unmapped processor.
        assert not any(k.startswith("voice.stt.") or k.startswith("voice.tts.") for k in attrs)
        assert "voice.llm.ttfb_ms.avg" not in attrs

    def test_write_to_span_none_is_noop(self):
        obs = MetricsObserver(processor_stage={})
        obs.write_to_span(None)  # must not raise

    @pytest.mark.asyncio
    async def test_feeds_usage_accumulator_llm_and_tts(self):
        """#28 — live LLM tokens (prompt→in, completion→out) + TTS chars are
        folded into the usage tally, in addition to the span attributes."""
        usage = UsageAccumulator()
        obs = MetricsObserver(processor_stage={"LLM#0": "llm"}, usage_accumulator=usage)
        await obs.on_push_frame(
            _pushed(
                _metrics_frame(
                    LLMUsageMetricsData(
                        processor="LLM#0",
                        value=LLMTokenUsage(
                            prompt_tokens=100, completion_tokens=20, total_tokens=120
                        ),
                    ),
                    TTSUsageMetricsData(processor="TTS#0", value=88),
                )
            )
        )
        totals = usage.totals()
        assert totals.llm_tokens_in == 100
        assert totals.llm_tokens_out == 20
        assert totals.tts_chars == 88

    @pytest.mark.asyncio
    async def test_no_accumulator_is_inert(self):
        """Default (no accumulator) records span metrics but no cost tally — and
        must not raise on usage frames."""
        obs = MetricsObserver(processor_stage={"LLM#0": "llm"})
        await obs.on_push_frame(
            _pushed(_metrics_frame(TTSUsageMetricsData(processor="TTS#0", value=5)))
        )  # no exception, nowhere for the tally to go

    @pytest.mark.asyncio
    async def test_average_llm_ttfb_ms_returns_integer_average(self):
        obs = MetricsObserver(processor_stage={"LLM#0": "llm"})
        await obs.on_push_frame(
            _pushed(
                _metrics_frame(
                    TTFBMetricsData(processor="LLM#0", value=0.5),
                    TTFBMetricsData(processor="LLM#0", value=1.001),
                )
            )
        )

        assert obs.average_llm_ttfb_ms() == 750

    @pytest.mark.asyncio
    async def test_average_llm_ttfb_ms_returns_none_without_llm_sample(self):
        obs = MetricsObserver(processor_stage={"STT#0": "stt"})
        await obs.on_push_frame(
            _pushed(_metrics_frame(TTFBMetricsData(processor="STT#0", value=0.1)))
        )

        assert obs.average_llm_ttfb_ms() is None


# ── The hard PHI guarantee (acceptance) ────────────────────────────────────


class TestNoPhiInAnyAttribute:
    @pytest.mark.asyncio
    async def test_simulated_call_emits_correlated_trace_with_no_phi(self, exporter):
        """A simulated call (call root + tool + post-call + metrics) produces a
        single correlated trace whose attribute values contain NO PHI."""
        case_data = {"patient_name": PHI_NAME, "dob": PHI_DOB}

        # 1. Call root span.
        call = start_call_span(
            call_id="call-xyz", session_id="sess-xyz", agent_id="agent-1", direction="inbound"
        )
        root_ctx = context_of(call)

        # 2. Tool execution through the real executor — args + result carry PHI.
        async def echo(args: dict, ctx) -> ToolResult:
            return success_result(data={"leaked": args["x"]})

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="lookup",
                description="lookup",
                parameters=[ToolParameter(name="x", type="string", description="x")],
                executor=echo,
                timeout_secs=1.0,
            )
        )
        tool_ctx = ToolContext(call_id="call-xyz", session_id="sess-xyz", otel_context=root_ctx)
        await ToolExecutor(registry).execute("lookup", {"x": PHI_TOOL_ARG}, tool_ctx)

        # 3. Metrics observer (numeric only).
        obs = MetricsObserver(processor_stage={"LLM#0": "llm"})
        await obs.on_push_frame(
            _pushed(_metrics_frame(TTFBMetricsData(processor="LLM#0", value=0.5)))
        )
        obs.write_to_span(call)

        # 4. Post-call extraction — transcript + case_data + extracted values are PHI.
        transcript = [{"turn_number": 1, "speaker": "user", "content": PHI_TRANSCRIPT}]
        with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
            mock_br.converse = MagicMock(
                return_value=_bedrock_response_with_usage({"summary": PHI_NAME})
            )
            await run_post_call_analyses(
                _pca_agent(), case_data, transcript, _settings(), otel_parent_context=root_ctx
            )

        end_call_span(call, end_status="completed", transcript_turns=len(transcript))

        # All three spans share one trace.
        spans = _spans_by_name(exporter)
        assert {"voice.call", "voice.tool", "voice.post_call"} <= set(spans)
        trace_ids = {s.context.trace_id for s in exporter.get_finished_spans()}
        assert len(trace_ids) == 1

        # No attribute value contains any PHI sentinel.
        sentinels = (PHI_NAME, PHI_DOB, PHI_TRANSCRIPT, PHI_TOOL_ARG)
        for value in _all_attr_values(exporter):
            text = str(value)
            for sentinel in sentinels:
                assert sentinel not in text, f"PHI leaked into span attribute: {value!r}"

    @pytest.mark.asyncio
    async def test_tool_error_records_type_not_message(self, exporter):
        """A tool failure records only the status/error-type, never the message
        (which can carry PHI)."""

        async def boom(args: dict, ctx) -> ToolResult:
            raise ValueError(f"failed for {PHI_NAME}")

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="boom",
                description="boom",
                parameters=[ToolParameter(name="x", type="string", description="x")],
                executor=boom,
                timeout_secs=1.0,
            )
        )
        call = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        ctx = ToolContext(call_id="c", otel_context=context_of(call))
        result = await ToolExecutor(registry).execute("boom", {"x": "v"}, ctx)
        end_call_span(call, end_status="completed")

        assert result.status == error_result("x").status
        attrs = dict(_spans_by_name(exporter)["voice.tool"].attributes)
        assert attrs["voice.tool.status"] == "error"
        assert attrs["voice.tool.error_type"] == "ValueError"
        for value in _all_attr_values(exporter):
            assert PHI_NAME not in str(value)


# ── Fail-open ──────────────────────────────────────────────────────────────


class TestFailOpen:
    def test_exporter_failure_does_not_raise(self):
        """A broken exporter (raises on export) must not propagate out of the
        span helpers — telemetry is fail-open."""

        class _BrokenExporter(InMemorySpanExporter):
            def export(self, spans):
                raise RuntimeError("collector down")

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_BrokenExporter()))
        tracing._TRACER = provider.get_tracer("test")
        tracing._PROVIDER = provider

        span = start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound")
        # end triggers the (broken) export synchronously; must be swallowed.
        end_call_span(span, end_status="completed")

    def test_helpers_noop_when_tracer_uninstalled(self):
        tracing._TRACER = None
        assert (
            start_call_span(call_id="c", session_id="s", agent_id="a", direction="inbound") is None
        )
        with tool_span(tool_name="t", parent_context=None) as ts:
            assert ts is None
        with llm_span(operation="chat", model="m", parent_context=None) as ls:
            assert ls is None
        # set_span_attrs / end_call_span on None are no-ops.
        set_span_attrs(None, {"k": "v"})
        end_call_span(None, end_status="completed")
