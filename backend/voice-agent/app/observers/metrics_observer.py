"""Observe Pipecat ``MetricsFrame``s; fold per-stage timing onto the call span (#13).

This is the PHI-free path to #13's core goal (§1): a per-call view of *which
stage is slow* — STT vs LLM vs TTS. Pipecat's metrics pipeline is entirely
**numeric** and completely separate from its PHI-bearing GenAI tracer (see
:mod:`app.observability.tracing` for why we don't use the latter). With
``enable_metrics`` / ``enable_usage_metrics`` already on (Layer 8's
``PipelineTask``), every service emits ``MetricsFrame``s carrying:

* :class:`~pipecat.metrics.metrics.TTFBMetricsData` — time-to-first-byte (s).
* :class:`~pipecat.metrics.metrics.ProcessingMetricsData` — processing time (s).
* :class:`~pipecat.metrics.metrics.LLMUsageMetricsData` — prompt / completion /
  cache token counts.
* :class:`~pipecat.metrics.metrics.TTSUsageMetricsData` — characters synthesized.
* :class:`~pipecat.metrics.metrics.TurnMetricsData` — turn count + end-to-end
  processing time (ms).

This observer accumulates those numbers per call and, at finalize, writes
aggregated attributes (``voice.<stage>.ttfb_ms.*``, ``voice.llm.tokens.*``,
``voice.tts.chars``, ``voice.turns.*``) onto the ``voice.call`` span via
:meth:`write_to_span`. Pure numbers + a fixed stage label — no transcript,
no message content, no PHI.

Stage attribution is exact: ``MetricsData.processor`` equals the emitting
processor's ``FrameProcessor.name`` (e.g. ``"AWSBedrockLLMService#0"``). Layer 8
builds the ``processor_stage`` map from the live service instances'
``.name`` properties, so there's no brittle class-name substring matching.

Like the other Layer 7 observers this is **read-only** and best-effort: it never
pushes frames, never cancels, and never raises out of ``on_push_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from app.observability.tracing import set_span_attrs
from app.observers.usage_accumulator import UsageAccumulator

logger = structlog.get_logger(__name__)


@dataclass
class _StageAgg:
    """Running aggregate of one timing metric for one stage (count / sum / max)."""

    count: int = 0
    sum_s: float = 0.0
    max_s: float = 0.0

    def add(self, value_s: float) -> None:
        self.count += 1
        self.sum_s += value_s
        if value_s > self.max_s:
            self.max_s = value_s

    def avg_ms(self) -> float:
        return round((self.sum_s / self.count) * 1000, 2) if self.count else 0.0

    def max_ms(self) -> float:
        return round(self.max_s * 1000, 2)


@dataclass
class _LLMTokens:
    prompt: int = 0
    completion: int = 0
    total: int = 0
    cache_read: int = 0
    cache_creation: int = 0

    @property
    def any(self) -> bool:
        return bool(self.prompt or self.completion or self.total)


@dataclass
class _CallMetrics:
    """Per-call accumulator. One instance per :class:`MetricsObserver`."""

    ttfb: dict[str, _StageAgg] = field(
        default_factory=lambda: {s: _StageAgg() for s in ("stt", "llm", "tts")}
    )
    processing: dict[str, _StageAgg] = field(
        default_factory=lambda: {s: _StageAgg() for s in ("stt", "llm", "tts")}
    )
    llm_tokens: _LLMTokens = field(default_factory=_LLMTokens)
    tts_chars: int = 0
    turn_count: int = 0
    turn_e2e: _StageAgg = field(default_factory=_StageAgg)


class MetricsObserver(BaseObserver):
    """Accumulates Pipecat metrics per call; folds them onto the call span.

    Args:
        processor_stage: Map of ``FrameProcessor.name`` → stage label
            (``"stt"`` / ``"llm"`` / ``"tts"``). Built by Layer 8 from the live
            STT / LLM / TTS service instances. Metrics from processors not in the
            map (e.g. transport, aggregators) are ignored for per-stage timing;
            usage metrics (LLM tokens, TTS chars, turns) are recorded regardless.
        usage_accumulator: Optional per-call usage tally (#28). When provided,
            each LLM token / TTS character metric is also folded in (mapped to
            billing ``in``/``out`` tokens + chars) so :func:`finalize_call` can
            persist real usage on the :class:`CallRecord`. ``None`` (the default,
            e.g. when tracing is off) leaves cost capture inert — the span
            attributes below are unaffected either way.
    """

    def __init__(
        self,
        *,
        processor_stage: dict[str, str],
        usage_accumulator: UsageAccumulator | None = None,
    ) -> None:
        super().__init__()
        self._processor_stage = processor_stage
        self._usage = usage_accumulator
        self._metrics = _CallMetrics()
        # ``on_push_frame`` fires once per processor hop, so a single
        # ``MetricsFrame`` is observed N times. Dedup by frame.id — same
        # rationale as ``ErrorObserver`` / ``TranscriptObserver``.
        self._seen_frame_ids: set[int] = set()

    async def on_push_frame(self, data: FramePushed) -> None:
        """Fold any ``MetricsFrame`` into the per-call accumulator. Never raises."""
        frame = data.frame
        if not isinstance(frame, MetricsFrame):
            return
        if frame.id in self._seen_frame_ids:
            return
        self._seen_frame_ids.add(frame.id)

        try:
            for item in frame.data:
                self._record(item)
        except Exception as exc:  # noqa: BLE001 — observer must never break the call
            logger.debug("metrics_observe_failed", error=str(exc))

    def _record(self, item: Any) -> None:
        # ``SmartTurnMetricsData`` subclasses ``TurnMetricsData``, so the turn
        # check is placed first / catches both. Stage timing metrics are only
        # attributed when the emitting processor maps to a known stage.
        if isinstance(item, TurnMetricsData):
            self._metrics.turn_count += 1
            self._metrics.turn_e2e.add(item.e2e_processing_time_ms / 1000.0)
        elif isinstance(item, LLMUsageMetricsData):
            usage = item.value
            self._metrics.llm_tokens.prompt += usage.prompt_tokens or 0
            self._metrics.llm_tokens.completion += usage.completion_tokens or 0
            self._metrics.llm_tokens.total += usage.total_tokens or 0
            self._metrics.llm_tokens.cache_read += usage.cache_read_input_tokens or 0
            self._metrics.llm_tokens.cache_creation += usage.cache_creation_input_tokens or 0
            # Cost capture (#28): live-pipeline LLM usage → in/out tokens.
            # Cache tokens stay on the span only (prompt caching isn't wired
            # yet), keeping the persisted shape the API's columns model.
            if self._usage is not None:
                self._usage.add_llm_usage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
        elif isinstance(item, TTSUsageMetricsData):
            self._metrics.tts_chars += int(item.value or 0)
            if self._usage is not None:  # cost capture (#28)
                self._usage.add_tts_chars(int(item.value or 0))
        elif isinstance(item, TTFBMetricsData):
            stage = self._processor_stage.get(item.processor)
            if stage:
                self._metrics.ttfb[stage].add(float(item.value))
        elif isinstance(item, ProcessingMetricsData):
            stage = self._processor_stage.get(item.processor)
            if stage:
                self._metrics.processing[stage].add(float(item.value))

    def write_to_span(self, span: Any) -> None:
        """Fold the accumulated metrics onto *span* as PHI-free attributes.

        Called once by Layer 8 at finalize, just before the ``voice.call`` span
        ends. No-op when *span* is ``None`` (tracing off). Only stages/metrics
        with samples are written, so the attribute set reflects what actually ran.
        """
        if span is None:
            return

        attrs: dict[str, Any] = {}
        m = self._metrics

        for stage in ("stt", "llm", "tts"):
            ttfb = m.ttfb[stage]
            if ttfb.count:
                attrs[f"voice.{stage}.ttfb_ms.avg"] = ttfb.avg_ms()
                attrs[f"voice.{stage}.ttfb_ms.max"] = ttfb.max_ms()
                attrs[f"voice.{stage}.ttfb.count"] = ttfb.count
            proc = m.processing[stage]
            if proc.count:
                attrs[f"voice.{stage}.processing_ms.avg"] = proc.avg_ms()
                attrs[f"voice.{stage}.processing_ms.max"] = proc.max_ms()

        if m.llm_tokens.any:
            attrs["voice.llm.tokens.prompt"] = m.llm_tokens.prompt
            attrs["voice.llm.tokens.completion"] = m.llm_tokens.completion
            attrs["voice.llm.tokens.total"] = m.llm_tokens.total
            if m.llm_tokens.cache_read:
                attrs["voice.llm.tokens.cache_read"] = m.llm_tokens.cache_read
            if m.llm_tokens.cache_creation:
                attrs["voice.llm.tokens.cache_creation"] = m.llm_tokens.cache_creation

        if m.tts_chars:
            attrs["voice.tts.chars"] = m.tts_chars

        if m.turn_count:
            attrs["voice.turns.count"] = m.turn_count
            attrs["voice.turns.e2e_ms.avg"] = m.turn_e2e.avg_ms()
            attrs["voice.turns.e2e_ms.max"] = m.turn_e2e.max_ms()

        set_span_attrs(span, attrs)

    def average_llm_ttfb_ms(self) -> int | None:
        """Return average live LLM time-to-first-byte in ms, if observed."""
        ttfb = self._metrics.ttfb["llm"]
        if not ttfb.count:
            return None
        return round(ttfb.avg_ms())
