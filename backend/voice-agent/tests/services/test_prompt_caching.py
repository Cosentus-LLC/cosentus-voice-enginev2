"""Tests for Bedrock prompt caching — verify + lock-in (#21).

Caching is already enabled (``build_llm`` hardcodes ``enable_prompt_caching=True``)
and the cache breakpoint is placed by Pipecat, not by our code. What these tests
pin down is the part the issue asks us to *verify*:

* the cache breakpoint (``{"cachePoint": {"type": "default"}}``) lands at the END
  of the Bedrock ``system`` block — the stable-prefix boundary, AFTER the system
  prompt and BEFORE the per-turn ``messages``;
* the cached prefix (that ``system`` block) is **byte-identical turn-to-turn**, so
  turn 2+ hits the cache (``cache_read`` > 0) — the acceptance criterion;
* nothing dynamic (the growing ``messages``) sits inside the cached prefix;
* the ``enable_prompt_caching`` flag is what drives the cachePoint insertion;
* the ``cache_read`` / ``cache_creation`` usage tokens (the #28 readout the
  hit-rate is measured from) are folded onto the metrics observer.

The cachePoint insertion lives inside Pipecat's ``AWSBedrockLLMService._process_context``
(pinned 1.1.0). Rather than re-implement that snippet (which would test a copy,
not the real thing), these tests drive the real ``_process_context`` against a
mocked Bedrock client and assert on the ``request_params`` it actually builds —
so a Pipecat change to the cache-point boundary fails the suite loudly (same
philosophy as ``test_llm_failover.py`` driving ``_create_converse_stream`` directly).

⚠️ Invariant for #22 (context trimming): trimming operates on ``messages``, never
on the ``system`` block — keep it that way or the cached prefix stops being stable
and turn-2+ cache hits disappear. ``test_cached_system_prefix_byte_stable_across_turns``
guards the system half of that invariant.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.config.agent_config import AgentConfig, LLMConfig, STTConfig, TTSConfig, TTSSettings
from app.config.settings import Settings
from app.observers.metrics_observer import MetricsObserver
from app.services.factory import build_llm
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws.llm import AWSBedrockLLMService

_CACHE_POINT = {"cachePoint": {"type": "default"}}

# A system prompt long enough to be representative; the exact length doesn't
# matter to these structural tests (the live 4096-token Haiku minimum can't be
# asserted without a real call — see the build_llm docstring note).
_SYSTEM_PROMPT = "You are Chris, a Cosentus billing specialist.\n" * 40


# ── Fixtures / helpers ──────────────────────────────────────────────────────


def _settings(**overrides) -> Settings:
    base = {
        "voice_api_lambda_name": "test-lambda",
        "api_key_secret_arn": "arn:test",
        "aws_region": "us-west-2",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _agent(*, system_prompt: str = _SYSTEM_PROMPT) -> AgentConfig:
    return AgentConfig(
        name="test-agent",
        system_prompt=system_prompt,
        llm=LLMConfig(model="claude-haiku-4-5", max_tokens=200, temperature=0.7),
        tts=TTSConfig(voice_id="", model="", settings=TTSSettings()),
        stt=STTConfig(keywords=[]),
    )


def _ctx(*user_turns: str) -> LLMContext:
    """Build an LLMContext with ``user_turns`` user messages (simulating a call
    that has accumulated N turns of history)."""
    ctx = LLMContext()
    for turn in user_turns:
        ctx.add_message({"role": "user", "content": turn})
    return ctx


def _conversation(n_user_turns: int) -> LLMContext:
    """Build an LLMContext for a realistic ``n_user_turns``-turn call: alternating
    user/assistant messages. Alternating roles means Bedrock keeps them as
    distinct messages (consecutive same-role messages get merged), so the
    ``messages`` list grows turn-to-turn — which is what makes the cached-prefix
    stability assertion meaningful rather than trivial."""
    ctx = LLMContext()
    for i in range(n_user_turns):
        ctx.add_message({"role": "user", "content": f"caller question {i}"})
        if i < n_user_turns - 1:
            ctx.add_message({"role": "assistant", "content": f"agent reply {i}"})
    return ctx


async def _empty_stream():
    """An async iterator that yields no events — stands in for a Bedrock
    converse_stream response so ``_process_context`` exits cleanly after the
    request is assembled."""
    return
    yield  # pragma: no cover — makes this an async generator


async def _capture_request_params(
    service: AWSBedrockLLMService, context: LLMContext
) -> dict[str, Any]:
    """Drive the REAL ``_process_context`` and return the ``request_params`` it
    builds — including Pipecat's cachePoint insertion — without a live call.

    ``push_frame`` is no-op'd to isolate request assembly (and silence the
    not-started frame-processor log); the AWS session client is a stub async
    context manager; ``_create_converse_stream`` records the params and returns
    an empty stream so the consume loop is a no-op.
    """
    captured: dict[str, Any] = {}

    async def _fake_create_stream(self_, client, request_params):  # noqa: ANN001
        captured.clear()
        captured.update(request_params)
        return {"stream": _empty_stream()}

    @asynccontextmanager
    async def _fake_client(**_kwargs):
        yield MagicMock()

    service._aws_session = MagicMock()
    service._aws_session.client = _fake_client

    with (
        patch.object(service, "push_frame", AsyncMock()),
        patch.object(AWSBedrockLLMService, "_create_converse_stream", _fake_create_stream),
    ):
        await service._process_context(context)

    return captured


# ── cache breakpoint placement ──────────────────────────────────────────────


class TestCachePointPlacement:
    async def test_cachepoint_appended_at_end_of_system_block(self):
        # The breakpoint must sit at the stable-prefix boundary: after the
        # system prompt, as the LAST element of the system list.
        service = build_llm(_agent(), _settings())
        params = await _capture_request_params(service, _ctx("hello"))

        system = params["system"]
        assert system[-1] == _CACHE_POINT
        # Everything before the cachePoint is the system prompt text — and the
        # prompt passed through verbatim (no truncation / transformation).
        assert system[0] == {"text": _SYSTEM_PROMPT}
        assert sum(1 for item in system if "cachePoint" in item) == 1

    async def test_no_cachepoint_in_dynamic_messages(self):
        # Per-turn messages live AFTER the breakpoint and must never carry a
        # cachePoint themselves — only the stable system prefix is cached.
        service = build_llm(_agent(), _settings())
        params = await _capture_request_params(service, _ctx("turn one", "turn two"))

        for message in params["messages"]:
            content = message.get("content")
            if isinstance(content, list):
                assert all("cachePoint" not in block for block in content)

    async def test_caching_disabled_inserts_no_cachepoint(self):
        # Negative control: the enable_prompt_caching flag is what drives the
        # insertion. Flip it off on the built service and the cachePoint is gone.
        service = build_llm(_agent(), _settings())
        service._settings.enable_prompt_caching = False

        params = await _capture_request_params(service, _ctx("hello"))

        assert all("cachePoint" not in item for item in params["system"])


# ── cached-prefix stability across turns (the acceptance criterion) ─────────


class TestCachedPrefixStability:
    async def test_cached_system_prefix_byte_stable_across_turns(self):
        # The whole point of caching: across turns of one call the cached prefix
        # (the system block, cachePoint included) is byte-identical, so turn 2+
        # reads the cache. The messages list grows turn-to-turn — proving the
        # stability isn't a trivial identical-input artifact.
        service = build_llm(_agent(), _settings())

        turn1 = await _capture_request_params(service, _conversation(1))
        turn2 = await _capture_request_params(service, _conversation(2))
        turn3 = await _capture_request_params(service, _conversation(3))

        # Cached prefix identical every turn.
        assert turn1["system"] == turn2["system"] == turn3["system"]
        # ...and it actually ends in the cachePoint (not just trivially equal).
        assert turn1["system"][-1] == _CACHE_POINT

        # Dynamic content really is growing — the stability above is meaningful.
        assert len(turn1["messages"]) < len(turn2["messages"]) < len(turn3["messages"])


# ── usage readout the hit-rate is measured from (#28) ───────────────────────


def _push(frame) -> FramePushed:
    return FramePushed(
        source=MagicMock(),
        destination=MagicMock(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


class TestCacheUsageMetrics:
    async def test_cache_tokens_folded_onto_metrics(self):
        # cache_read / cache_creation are the operator's hit-rate readout — they
        # must survive the observer and land on the per-call accumulator (and,
        # via write_to_span, on voice.llm.tokens.cache_*).
        observer = MetricsObserver(processor_stage={})
        usage = LLMTokenUsage(
            prompt_tokens=1000,
            completion_tokens=50,
            total_tokens=1050,
            cache_read_input_tokens=900,
            cache_creation_input_tokens=100,
        )
        frame = MetricsFrame(
            data=[LLMUsageMetricsData(processor="AWSBedrockLLMService#0", value=usage)]
        )

        await observer.on_push_frame(_push(frame))

        assert observer._metrics.llm_tokens.cache_read == 900
        assert observer._metrics.llm_tokens.cache_creation == 100
