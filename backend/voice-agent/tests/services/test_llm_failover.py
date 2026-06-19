"""Tests for live mid-call Bedrock model failover (#52).

Covers the two surfaces of the feature:

* ``build_llm`` wiring — empty ``live_model_fallback_chain`` builds the plain
  ``AWSBedrockLLMService`` (flag-off path byte-identical); a non-empty chain
  builds ``_FailoverBedrockLLMService`` with the resolved fallback IDs.
* ``resolve_live_model_chain`` — primary-first, resolved, de-duped, ordered.
* ``_FailoverBedrockLLMService._create_converse_stream`` — the override that
  fails the live turn over to the next Bedrock model on a *retryable* error
  raised at the ``converse_stream`` call, sticks to the surviving model, and
  re-raises non-retryable / chain-exhausted errors unchanged.

The override is called directly (not through the whole pipeline) so a Pipecat
signature change to ``_create_converse_stream`` — a pinned-1.1.0 internal —
fails the suite loudly (see _FailoverBedrockLLMService docstring).
"""

from __future__ import annotations

import pytest
from app.config.agent_config import AgentConfig, LLMConfig, STTConfig, TTSConfig, TTSSettings
from app.config.settings import Settings
from app.services.factory import (
    _FailoverBedrockLLMService,
    _is_retryable_bedrock_error,
    _parse_model_fallback_chain,
    build_llm,
    resolve_live_model_chain,
)
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
from pipecat.services.aws.llm import AWSBedrockLLMService

# Resolved Bedrock inference-profile IDs the test models map to (mirrors
# _SHORT_TO_BEDROCK). Using the short-names through the real resolver keeps the
# test honest about the allowlist contract.
_HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_SONNET_4_6 = "us.anthropic.claude-sonnet-4-6"
_SONNET_4_5 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


# ── Fixtures / helpers ──────────────────────────────────────────────────────


def _settings(**overrides) -> Settings:
    base = {
        "voice_api_lambda_name": "test-lambda",
        "api_key_secret_arn": "arn:test",
        "aws_region": "us-west-2",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _agent(*, llm_model: str = "claude-haiku-4-5") -> AgentConfig:
    return AgentConfig(
        name="test-agent",
        system_prompt="You are a test agent.",
        llm=LLMConfig(model=llm_model, max_tokens=200, temperature=0.7),
        tts=TTSConfig(voice_id="", model="", settings=TTSSettings()),
        stt=STTConfig(keywords=[]),
    )


def _make_service(*, primary: str = _HAIKU, fallbacks: list[str]) -> _FailoverBedrockLLMService:
    """Construct the failover service directly for override-level tests."""
    return _FailoverBedrockLLMService(
        model=primary,
        aws_region="us-west-2",
        settings=AWSBedrockLLMService.Settings(
            system_instruction="sys",
            enable_prompt_caching=True,
        ),
        fallback_model_ids=fallbacks,
    )


def _throttle() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "ConverseStream",
    )


def _validation() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": "bad model"}},
        "ConverseStream",
    )


def _access_denied() -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
        "ConverseStream",
    )


class _FakeBedrockClient:
    """Stand-in for the aioboto3 bedrock-runtime client.

    ``behavior`` maps a modelId to either ``("raise", exc)`` or
    ``("ok", sentinel)``; ``converse_stream`` records the modelId it was called
    with and acts on the mapping.
    """

    def __init__(self, behavior: dict[str, tuple[str, object]]):
        self._behavior = behavior
        self.calls: list[str] = []

    async def converse_stream(self, **kwargs):
        model_id = kwargs["modelId"]
        self.calls.append(model_id)
        action, value = self._behavior[model_id]
        if action == "raise":
            raise value
        return value


def _request_params(model_id: str) -> dict[str, object]:
    return {"modelId": model_id, "messages": [], "additionalModelRequestFields": {}}


# ── _is_retryable_bedrock_error ─────────────────────────────────────────────


class TestIsRetryable:
    @pytest.mark.parametrize(
        "exc",
        [
            _throttle(),
            ClientError({"Error": {"Code": "ServiceUnavailableException"}}, "op"),
            ClientError({"Error": {"Code": "ModelNotReadyException"}}, "op"),
            ReadTimeoutError(endpoint_url="https://bedrock-runtime"),
            ConnectTimeoutError(endpoint_url="https://bedrock-runtime"),
        ],
    )
    def test_retryable(self, exc):
        assert _is_retryable_bedrock_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            _validation(),
            _access_denied(),
            ValueError("not a bedrock error"),
        ],
    )
    def test_not_retryable(self, exc):
        assert _is_retryable_bedrock_error(exc) is False


# ── resolve_live_model_chain / _parse_model_fallback_chain ──────────────────


class TestResolveLiveModelChain:
    def test_empty_chain_returns_only_primary(self):
        chain = resolve_live_model_chain("claude-haiku-4-5", _settings())
        assert chain == [_HAIKU]

    def test_primary_first_then_fallbacks_resolved(self):
        settings = _settings(live_model_fallback_chain="claude-sonnet-4-6, claude-sonnet-4-5")
        chain = resolve_live_model_chain("claude-haiku-4-5", settings)
        assert chain == [_HAIKU, _SONNET_4_6, _SONNET_4_5]

    def test_duplicate_fallback_equal_to_primary_dropped(self):
        settings = _settings(live_model_fallback_chain="claude-haiku-4-5, claude-sonnet-4-6")
        chain = resolve_live_model_chain("claude-haiku-4-5", settings)
        assert chain == [_HAIKU, _SONNET_4_6]

    def test_duplicate_fallbacks_deduped_order_preserved(self):
        settings = _settings(
            live_model_fallback_chain="claude-sonnet-4-6, claude-sonnet-4-6, claude-sonnet-4-5"
        )
        chain = resolve_live_model_chain("claude-haiku-4-5", settings)
        assert chain == [_HAIKU, _SONNET_4_6, _SONNET_4_5]

    def test_unknown_fallback_passes_through_not_dropped(self):
        # Mirrors the offline path (#20): unknown short-names warn + pass
        # through (Bedrock rejects), not silently dropped.
        settings = _settings(live_model_fallback_chain="claude-mystery-9-9")
        chain = resolve_live_model_chain("claude-haiku-4-5", settings)
        assert chain == [_HAIKU, "claude-mystery-9-9"]

    def test_parse_strips_whitespace_and_drops_empties(self):
        assert _parse_model_fallback_chain(" a , ,b ,, c ") == ["a", "b", "c"]
        assert _parse_model_fallback_chain("") == []
        assert _parse_model_fallback_chain(None) == []


# ── build_llm wiring ────────────────────────────────────────────────────────


class TestBuildLlmFailoverWiring:
    def test_no_chain_returns_plain_service(self):
        # Flag-off path: empty chain → plain service, failover class never
        # involved (behavior byte-identical to before #52).
        service = build_llm(_agent(), _settings())
        assert type(service) is AWSBedrockLLMService
        assert not isinstance(service, _FailoverBedrockLLMService)

    def test_with_chain_returns_failover_service_with_resolved_fallbacks(self):
        settings = _settings(live_model_fallback_chain="claude-sonnet-4-6, claude-sonnet-4-5")
        service = build_llm(_agent(llm_model="claude-haiku-4-5"), settings)
        assert isinstance(service, _FailoverBedrockLLMService)
        # Primary stays on the service; only the post-primary models are
        # carried as fallbacks, resolved + in order.
        assert service._settings.model == _HAIKU
        assert service._fallback_model_ids == [_SONNET_4_6, _SONNET_4_5]

    def test_chain_with_only_primary_returns_plain_service(self):
        # A fallback equal to the primary de-dupes to a length-1 chain → plain.
        settings = _settings(live_model_fallback_chain="claude-haiku-4-5")
        service = build_llm(_agent(llm_model="claude-haiku-4-5"), settings)
        assert type(service) is AWSBedrockLLMService


# ── _create_converse_stream failover behavior ───────────────────────────────


class TestCreateConverseStreamFailover:
    async def test_failover_advances_on_throttling(self):
        service = _make_service(primary=_HAIKU, fallbacks=[_SONNET_4_6])
        sentinel = object()
        client = _FakeBedrockClient({_HAIKU: ("raise", _throttle()), _SONNET_4_6: ("ok", sentinel)})
        params = _request_params(_HAIKU)

        result = await service._create_converse_stream(client, params)

        assert result is sentinel
        assert client.calls == [_HAIKU, _SONNET_4_6]
        assert params["modelId"] == _SONNET_4_6

    async def test_failover_on_read_timeout(self):
        service = _make_service(primary=_HAIKU, fallbacks=[_SONNET_4_6])
        sentinel = object()
        client = _FakeBedrockClient(
            {
                _HAIKU: ("raise", ReadTimeoutError(endpoint_url="https://bedrock-runtime")),
                _SONNET_4_6: ("ok", sentinel),
            }
        )

        result = await service._create_converse_stream(client, _request_params(_HAIKU))

        assert result is sentinel
        assert client.calls == [_HAIKU, _SONNET_4_6]

    async def test_failover_walks_multiple_models(self):
        service = _make_service(primary=_HAIKU, fallbacks=[_SONNET_4_6, _SONNET_4_5])
        sentinel = object()
        client = _FakeBedrockClient(
            {
                _HAIKU: ("raise", _throttle()),
                _SONNET_4_6: ("raise", _throttle()),
                _SONNET_4_5: ("ok", sentinel),
            }
        )

        result = await service._create_converse_stream(client, _request_params(_HAIKU))

        assert result is sentinel
        assert client.calls == [_HAIKU, _SONNET_4_6, _SONNET_4_5]

    @pytest.mark.parametrize("exc_factory", [_validation, _access_denied])
    async def test_non_retryable_error_does_not_failover(self, exc_factory):
        service = _make_service(primary=_HAIKU, fallbacks=[_SONNET_4_6])
        client = _FakeBedrockClient(
            {_HAIKU: ("raise", exc_factory()), _SONNET_4_6: ("ok", object())}
        )

        with pytest.raises(ClientError):
            await service._create_converse_stream(client, _request_params(_HAIKU))

        # Fallback never attempted — non-retryable fails fast, as today.
        assert client.calls == [_HAIKU]

    async def test_chain_exhausted_raises_last_error(self):
        # Every model throttles → the last error propagates to Pipecat's
        # _process_context (push_error), exactly as the single-model path does.
        service = _make_service(primary=_HAIKU, fallbacks=[_SONNET_4_6])
        client = _FakeBedrockClient(
            {_HAIKU: ("raise", _throttle()), _SONNET_4_6: ("raise", _throttle())}
        )

        with pytest.raises(ClientError) as excinfo:
            await service._create_converse_stream(client, _request_params(_HAIKU))

        assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"
        assert client.calls == [_HAIKU, _SONNET_4_6]

    async def test_failover_sticks_model_for_subsequent_turns(self):
        # After a failover the surviving model is pinned onto the service so the
        # NEXT turn starts there instead of re-hitting the throttled primary.
        service = _make_service(primary=_HAIKU, fallbacks=[_SONNET_4_6])
        sentinel = object()
        client = _FakeBedrockClient({_HAIKU: ("raise", _throttle()), _SONNET_4_6: ("ok", sentinel)})

        await service._create_converse_stream(client, _request_params(_HAIKU))
        assert service._settings.model == _SONNET_4_6

        # Simulate the next turn: _process_context would build request_params
        # from self._settings.model (now the fallback), so it goes straight to
        # the surviving model with no wasted primary round-trip.
        next_client = _FakeBedrockClient({_SONNET_4_6: ("ok", sentinel)})
        await service._create_converse_stream(next_client, _request_params(service._settings.model))
        assert next_client.calls == [_SONNET_4_6]

    async def test_success_on_primary_does_not_change_model(self):
        # No failover → primary stays primary, no stick.
        service = _make_service(primary=_HAIKU, fallbacks=[_SONNET_4_6])
        sentinel = object()
        client = _FakeBedrockClient({_HAIKU: ("ok", sentinel)})

        result = await service._create_converse_stream(client, _request_params(_HAIKU))

        assert result is sentinel
        assert client.calls == [_HAIKU]
        assert service._settings.model == _HAIKU
