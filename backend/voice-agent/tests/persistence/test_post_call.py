"""Tests for ``app/persistence/post_call.py``.

Post-call extraction runs through a Bedrock Converse **forced tool**: the
model's ``toolUse.input`` arrives as an already-parsed object, validated
against the per-agent field schema. Bedrock is mocked here so the gate
stays offline/deterministic; live field-accuracy is validated on staging.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
from app.config.agent_config import (
    AgentConfig,
    LLMConfig,
    PostCallConfig,
    PostCallField,
)
from app.config.settings import Settings
from app.observers.usage_accumulator import UsageAccumulator
from app.persistence import post_call as post_call_module
from app.persistence.post_call import (
    _build_extraction_prompt,
    _build_tool_config,
    _is_retryable_failover_error,
    _resolve_model_chain,
    _safe_property_keys,
    _validate_tool_input,
    run_post_call_analyses,
)
from botocore.exceptions import ClientError, ReadTimeoutError

# ── Test fixtures ──────────────────────────────────────────────────────────


def _settings() -> Settings:
    return Settings(
        voice_api_lambda_name="test-voice-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
    )


def _settings_with_chain(chain: str) -> Settings:
    """Settings with an offline model fallback chain configured (#20)."""
    return Settings(
        voice_api_lambda_name="test-voice-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
        post_call_model_fallback_chain=chain,
    )


def _throttling_error() -> ClientError:
    """A retryable Bedrock throttling error (triggers offline failover)."""
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse"
    )


def _pca_agent(model: str) -> AgentConfig:
    """Agent with a single-field post-call config pinned to ``model``."""
    return AgentConfig(
        name="test-agent",
        display_name="Test",
        system_prompt="You are a test agent.",
        post_call_analyses=PostCallConfig(
            model=model,
            fields=[PostCallField(name="summary", type="text", description="A summary.")],
        ),
    )


def _agent(fields: list[PostCallField] | None = None) -> AgentConfig:
    pca = PostCallConfig(model="claude-haiku-4-5", fields=fields) if fields is not None else None
    return AgentConfig(
        name="test-agent",
        display_name="Test",
        system_prompt="You are a test agent.",
        llm=LLMConfig(model="claude-sonnet-4-6"),
        post_call_analyses=pca,
    )


def _transcript() -> list[dict]:
    return [
        {
            "turn_number": 1,
            "speaker": "user",
            "content": "I need to check claim 12345.",
            "timestamp": "2026-05-04T12:00:00+00:00",
        },
        {
            "turn_number": 2,
            "speaker": "assistant",
            "content": "Of course. The claim is paid.",
            "timestamp": "2026-05-04T12:00:05+00:00",
        },
    ]


def _tool_response(tool_input: dict) -> dict:
    """A Converse response where the model called the extraction tool."""
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
        }
    }


def _text_response(text: str) -> dict:
    """A Converse response with no tool use — the model answered in text."""
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}}


def _with_usage(response: dict, *, in_tokens: int, out_tokens: int) -> dict:
    """Attach a Converse ``usage`` block (as Bedrock returns it)."""
    return {
        **response,
        "usage": {
            "inputTokens": in_tokens,
            "outputTokens": out_tokens,
            "totalTokens": in_tokens + out_tokens,
        },
    }


# ── Skip / no-op paths ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_empty_when_no_pca_configured():
    agent = _agent(fields=None)
    result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}


@pytest.mark.asyncio
async def test_returns_empty_when_fields_list_is_empty():
    agent = _agent(fields=[])
    result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}


@pytest.mark.asyncio
async def test_returns_empty_when_transcript_is_empty():
    agent = _agent(fields=[PostCallField(name="summary", type="text", description="A summary.")])
    result = await run_post_call_analyses(agent, {}, [], _settings())
    assert result == {}


# ── Happy paths ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_text_field_extracted():
    agent = _agent(
        fields=[
            PostCallField(name="summary", type="text", description="A summary."),
        ]
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            return_value=_tool_response({"summary": "Claim 12345 is paid."})
        )
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"summary": "Claim 12345 is paid."}


@pytest.mark.asyncio
async def test_returns_selector_field_with_valid_choice():
    agent = _agent(
        fields=[
            PostCallField(
                name="outcome",
                type="selector",
                choices=["resolved", "escalated", "dropped"],
                description="The call outcome.",
            )
        ]
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_tool_response({"outcome": "resolved"}))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"outcome": "resolved"}


@pytest.mark.asyncio
async def test_returns_invalid_marker_for_off_list_selector_choice():
    """Invalid choice gets ``invalid: <value>`` prefix.

    Surfaces the LLM's actual output for operator triage rather than
    silently swallowing it.
    """
    agent = _agent(
        fields=[
            PostCallField(
                name="outcome",
                type="selector",
                choices=["resolved", "escalated"],
            )
        ]
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_tool_response({"outcome": "in_progress"}))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"outcome": "invalid: in_progress"}


@pytest.mark.asyncio
async def test_default_offline_model_is_sonnet():
    """When the per-agent ``post_call_analyses.model`` is empty, the offline
    extraction falls back to the stronger model (#20 — Sonnet), not the live
    Haiku. Assert the Bedrock Converse call targets the resolved Sonnet id."""
    agent = AgentConfig(
        name="test-agent",
        display_name="Test",
        system_prompt="You are a test agent.",
        post_call_analyses=PostCallConfig(
            model="",
            fields=[PostCallField(name="summary", type="text", description="A summary.")],
        ),
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_tool_response({"summary": "ok"}))
        await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert mock_br.converse.call_args.kwargs["modelId"] == "us.anthropic.claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_per_agent_model_overrides_default():
    """An explicit per-agent ``post_call_analyses.model`` still wins over the
    Sonnet default — operators who pinned a model keep it (#20)."""
    agent = AgentConfig(
        name="test-agent",
        display_name="Test",
        system_prompt="You are a test agent.",
        post_call_analyses=PostCallConfig(
            model="claude-haiku-4-5",
            fields=[PostCallField(name="summary", type="text", description="A summary.")],
        ),
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_tool_response({"summary": "ok"}))
        await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert (
        mock_br.converse.call_args.kwargs["modelId"]
        == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )


@pytest.mark.asyncio
async def test_handles_multiple_fields():
    agent = _agent(
        fields=[
            PostCallField(name="summary", type="text", description="Summary"),
            PostCallField(name="outcome", type="selector", choices=["resolved", "dropped"]),
        ]
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            return_value=_tool_response({"summary": "Done.", "outcome": "resolved"})
        )
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"summary": "Done.", "outcome": "resolved"}


@pytest.mark.asyncio
async def test_missing_field_in_response_defaults_to_empty():
    """Model omits a field → coerce to empty string, don't fail."""
    agent = _agent(
        fields=[
            PostCallField(name="summary", type="text"),
            PostCallField(name="reference_number", type="text"),
        ]
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_tool_response({"summary": "ok"}))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result["summary"] == "ok"
    assert result["reference_number"] == ""


# ── Error / retry paths ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_empty_on_bedrock_error():
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    err = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "no entitlement"}},
        "Converse",
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(side_effect=err)
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}


@pytest.mark.asyncio
async def test_retries_once_when_no_tool_use_block():
    """Model answers in text first (no tool use), then calls the tool."""
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            side_effect=[
                _text_response("I think the claim is paid."),
                _tool_response({"summary": "fixed it"}),
            ]
        )
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"summary": "fixed it"}
    assert mock_br.converse.call_count == 2


@pytest.mark.asyncio
async def test_returns_empty_after_retry_still_no_tool_use():
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_text_response("never calls the tool"))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}
    assert mock_br.converse.call_count == 2  # original + 1 retry


@pytest.mark.asyncio
async def test_returns_empty_when_tool_input_not_dict():
    """``toolUse.input`` that isn't an object is unusable → no tool input → {}."""
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    bad = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "x", "name": "extract_post_call", "input": [1, 2]}}
                ],
            }
        }
    }
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=bad)
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}
    assert mock_br.converse.call_count == 2


@pytest.mark.asyncio
async def test_never_raises_on_unexpected_exception():
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(side_effect=RuntimeError("simulated unexpected"))
        # Should not raise; should return {}.
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}


# ── Offline model failover (#20) ────────────────────────────────────────────

_SONNET_ID = "us.anthropic.claude-sonnet-4-6"
_HAIKU_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.mark.asyncio
async def test_fails_over_to_next_model_on_throttling():
    """A retryable error (throttling) on the primary moves to the next model
    in the chain, which succeeds. Converse is hit on both models, in order."""
    agent = _pca_agent("claude-sonnet-4-6")
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            side_effect=[_throttling_error(), _tool_response({"summary": "ok"})]
        )
        result = await run_post_call_analyses(
            agent, {}, _transcript(), _settings_with_chain("claude-haiku-4-5")
        )
    assert result == {"summary": "ok"}
    assert mock_br.converse.call_count == 2
    assert mock_br.converse.call_args_list[0].kwargs["modelId"] == _SONNET_ID
    assert mock_br.converse.call_args_list[1].kwargs["modelId"] == _HAIKU_ID


@pytest.mark.asyncio
async def test_fails_over_on_read_timeout():
    """A botocore read timeout is retryable → fail over to the next model."""
    agent = _pca_agent("claude-sonnet-4-6")
    timeout = ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            side_effect=[timeout, _tool_response({"summary": "recovered"})]
        )
        result = await run_post_call_analyses(
            agent, {}, _transcript(), _settings_with_chain("claude-haiku-4-5")
        )
    assert result == {"summary": "recovered"}
    assert mock_br.converse.call_count == 2


@pytest.mark.asyncio
async def test_no_failover_on_non_retryable_error_even_with_chain():
    """A non-retryable error (AccessDenied) fails fast → {} — no failover,
    even with a fallback chain configured. Only the primary is hit."""
    agent = _pca_agent("claude-sonnet-4-6")
    err = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "no entitlement"}}, "Converse"
    )
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(side_effect=err)
        result = await run_post_call_analyses(
            agent, {}, _transcript(), _settings_with_chain("claude-haiku-4-5")
        )
    assert result == {}
    assert mock_br.converse.call_count == 1
    assert mock_br.converse.call_args_list[0].kwargs["modelId"] == _SONNET_ID


@pytest.mark.asyncio
async def test_chain_exhausted_returns_empty():
    """Retryable errors on every model in the chain → {} once exhausted.
    Both models are tried; the never-raises {} contract holds."""
    agent = _pca_agent("claude-sonnet-4-6")
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(side_effect=[_throttling_error(), _throttling_error()])
        result = await run_post_call_analyses(
            agent, {}, _transcript(), _settings_with_chain("claude-haiku-4-5")
        )
    assert result == {}
    assert mock_br.converse.call_count == 2


@pytest.mark.asyncio
async def test_empty_chain_no_failover_on_throttling():
    """With no fallback chain (default), a retryable error returns {} after
    the primary — today's behavior, byte-for-byte."""
    agent = _pca_agent("claude-sonnet-4-6")
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(side_effect=_throttling_error())
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}
    assert mock_br.converse.call_count == 1


@pytest.mark.asyncio
async def test_no_failover_on_no_structured_output():
    """A no-tool-output result is NOT a retryable error, so it retries on the
    SAME model (once) and then gives up — the chain is not consulted."""
    agent = _pca_agent("claude-sonnet-4-6")
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_text_response("no tool call"))
        result = await run_post_call_analyses(
            agent, {}, _transcript(), _settings_with_chain("claude-haiku-4-5")
        )
    assert result == {}
    # original + 1 same-model retry; the Haiku fallback is never tried.
    assert mock_br.converse.call_count == 2
    assert all(c.kwargs["modelId"] == _SONNET_ID for c in mock_br.converse.call_args_list)


def test_is_retryable_failover_error_classification():
    """Throttling / capacity codes + timeouts are retryable; auth/validation
    and unrelated exceptions are not."""
    assert _is_retryable_failover_error(_throttling_error()) is True
    assert (
        _is_retryable_failover_error(
            ClientError(
                {"Error": {"Code": "ServiceUnavailableException", "Message": "x"}}, "Converse"
            )
        )
        is True
    )
    assert _is_retryable_failover_error(ReadTimeoutError(endpoint_url="https://x")) is True
    assert (
        _is_retryable_failover_error(
            ClientError({"Error": {"Code": "ValidationException", "Message": "x"}}, "Converse")
        )
        is False
    )
    assert _is_retryable_failover_error(RuntimeError("boom")) is False


def test_resolve_model_chain_dedupes_and_preserves_order():
    """Primary first, then fallbacks left-to-right; resolved to Bedrock ids
    and de-duplicated (a fallback equal to the primary is dropped)."""
    chain = _resolve_model_chain(
        "claude-sonnet-4-6", _settings_with_chain("claude-haiku-4-5, claude-sonnet-4-6")
    )
    assert chain == [_SONNET_ID, _HAIKU_ID]


def test_resolve_model_chain_primary_only_when_chain_empty():
    chain = _resolve_model_chain("claude-sonnet-4-6", _settings())
    assert chain == [_SONNET_ID]


# ── Usage capture for cost (#28) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_feeds_usage_accumulator_on_success():
    """The extraction call's Converse usage is folded into the tally."""
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    usage = UsageAccumulator()
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            return_value=_with_usage(
                _tool_response({"summary": "ok"}), in_tokens=900, out_tokens=40
            )
        )
        await run_post_call_analyses(agent, {}, _transcript(), _settings(), usage_accumulator=usage)
    totals = usage.totals()
    assert totals.llm_tokens_in == 900
    assert totals.llm_tokens_out == 40
    assert totals.tts_chars == 0  # post-call has no TTS


@pytest.mark.asyncio
async def test_feeds_usage_accumulator_on_every_attempt():
    """The retry consumed tokens too — both attempts' usage is captured."""
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    usage = UsageAccumulator()
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            side_effect=[
                _with_usage(_text_response("no tool"), in_tokens=100, out_tokens=10),
                _with_usage(_tool_response({"summary": "ok"}), in_tokens=120, out_tokens=15),
            ]
        )
        await run_post_call_analyses(agent, {}, _transcript(), _settings(), usage_accumulator=usage)
    totals = usage.totals()
    assert totals.llm_tokens_in == 220
    assert totals.llm_tokens_out == 25


@pytest.mark.asyncio
async def test_usage_accumulator_none_is_safe():
    """Default (no accumulator) — extraction still works, nothing to feed."""
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(
            return_value=_with_usage(_tool_response({"summary": "ok"}), in_tokens=5, out_tokens=1)
        )
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"summary": "ok"}


# ── Tool input validation (replaces the old free-text JSON parser) ─────────


def _pca(*fields: PostCallField) -> PostCallConfig:
    return PostCallConfig(model="claude-haiku-4-5", fields=list(fields))


def test_validate_selector_wrong_case_marked_invalid():
    pca = _pca(PostCallField(name="outcome", type="selector", choices=["paid", "denied"]))
    assert _validate_tool_input({"outcome": "Paid"}, pca) == {"outcome": "invalid: Paid"}


def test_validate_out_of_enum_selector_marked_invalid():
    pca = _pca(PostCallField(name="outcome", type="selector", choices=["resolved", "escalated"]))
    assert _validate_tool_input({"outcome": "approved"}, pca) == {"outcome": "invalid: approved"}


def test_validate_missing_field_defaults_empty():
    pca = _pca(PostCallField(name="summary", type="text"), PostCallField(name="ref", type="text"))
    assert _validate_tool_input({"summary": "ok"}, pca) == {"summary": "ok", "ref": ""}


def test_validate_ignores_extra_keys():
    pca = _pca(PostCallField(name="summary", type="text"))
    assert _validate_tool_input({"summary": "ok", "sentiment": "positive"}, pca) == {
        "summary": "ok"
    }


def test_validate_coerces_non_string_to_str():
    """Model returns an int for a text field; coerce to str rather than failing."""
    pca = _pca(PostCallField(name="count", type="text"))
    assert _validate_tool_input({"count": 42}, pca) == {"count": "42"}


def test_validate_returns_none_on_non_dict_input():
    pca = _pca(PostCallField(name="summary", type="text"))
    assert _validate_tool_input([1, 2, 3], pca) is None


# ── Tool config built dynamically from the field schema ────────────────────


def test_build_tool_config_selector_has_enum():
    pca = _pca(
        PostCallField(name="outcome", type="selector", choices=["resolved", "escalated"]),
        PostCallField(name="summary", type="text"),
    )
    props = _build_tool_config(pca)["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]
    assert props["outcome"]["enum"] == ["resolved", "escalated"]
    assert props["outcome"]["type"] == "string"
    assert "enum" not in props["summary"]
    assert props["summary"]["type"] == "string"


def test_build_tool_config_forces_tool_and_marks_nothing_required():
    pca = _pca(PostCallField(name="summary", type="text"))
    cfg = _build_tool_config(pca)
    assert cfg["toolChoice"] == {"tool": {"name": "extract_post_call"}}
    json_schema = cfg["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert "required" not in json_schema


def test_build_tool_config_selector_without_choices_is_plain_string():
    pca = _pca(PostCallField(name="outcome", type="selector", choices=[]))
    props = _build_tool_config(pca)["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]
    assert "enum" not in props["outcome"]


# ── Prompt construction ───────────────────────────────────────────────────


def test_prompt_includes_case_data():
    pca = _pca(PostCallField(name="summary", type="text"))
    prompt = _build_extraction_prompt(pca, {"claim_id": "12345"}, _transcript())
    assert "claim_id" in prompt
    assert "12345" in prompt


def test_prompt_includes_all_transcript_turns():
    pca = _pca(PostCallField(name="summary", type="text"))
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "I need to check claim 12345." in prompt
    assert "Of course. The claim is paid." in prompt


def test_prompt_renders_selector_with_choices():
    pca = _pca(
        PostCallField(
            name="outcome",
            type="selector",
            choices=["resolved", "escalated"],
            description="What happened?",
        )
    )
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "selector" in prompt
    assert "resolved" in prompt
    assert "escalated" in prompt
    assert "What happened?" in prompt


def test_prompt_renders_text_field_clean():
    pca = _pca(PostCallField(name="summary", type="text", description="2 sentences."))
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "summary (text)" in prompt
    assert "2 sentences." in prompt


def test_prompt_includes_format_examples():
    pca = _pca(PostCallField(name="reference", type="text", format_examples=["REF-12345"]))
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "REF-12345" in prompt


def test_prompt_handles_empty_case_data():
    pca = _pca(PostCallField(name="summary", type="text"))
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "(none)" in prompt


def test_prompt_instructs_tool_call():
    pca = _pca(PostCallField(name="summary", type="text"))
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "extract_post_call" in prompt


# ── Lazy-init Bedrock client binding ───────────────────────────────────────


@pytest.mark.asyncio
async def test_bedrock_client_binds_region_from_settings_on_first_call():
    """The lazy-init pattern must build the client from ``settings.aws_region``,
    not from the env-var-at-import. Closes Entry 11."""
    # Reset the module-level cache so the next ``_get_bedrock_client``
    # call constructs a fresh client.
    post_call_module._BEDROCK_CLIENT = None

    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    settings = Settings(
        voice_api_lambda_name="test-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
        aws_region="us-west-2",
    )

    captured = {}

    def fake_client(service: str, region_name: str, config):
        captured["service"] = service
        captured["region"] = region_name
        mock = MagicMock()
        mock.converse.return_value = _tool_response({"summary": "ok"})
        return mock

    fake_session = MagicMock()
    fake_session.client = MagicMock(side_effect=fake_client)
    with patch(
        "app.persistence.post_call.boto3.session.Session",
        return_value=fake_session,
    ):
        await run_post_call_analyses(agent, {}, _transcript(), settings)

    assert captured["service"] == "bedrock-runtime"
    assert captured["region"] == "us-west-2"
    # Reset for subsequent tests so no module-level state leaks.
    post_call_module._BEDROCK_CLIENT = None


# ── Bedrock property-key sanitization (field names with spaces, etc.) ─────────
#
# Regression for the staging finding: chris-claim-status's post-call field is
# named "call summary" (with a space). Bedrock rejects tool property keys
# outside ^[a-zA-Z0-9_.-]{1,64}$ with a ValidationException that fails the
# entire extraction (so post_call_analyses persisted as {}).


def test_build_tool_config_sanitizes_field_name_with_space():
    """The wire key must be Bedrock-valid, not the raw human field name."""
    cfg = _build_tool_config(
        PostCallConfig(
            model="claude-haiku-4-5",
            fields=[PostCallField(name="call summary", type="text", description="Notes.")],
        )
    )
    props = cfg["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"]
    assert "call summary" not in props  # raw name (with space) would be rejected
    assert "call_summary" in props


def test_validate_tool_input_maps_sanitized_key_back_to_field_name():
    """Model answers under the sanitized key → output keeps the original name."""
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="call summary", type="text", description="Notes.")],
    )
    out = _validate_tool_input({"call_summary": "Claim 12345 denied CO16."}, pca)
    assert out == {"call summary": "Claim 12345 denied CO16."}


def test_safe_property_keys_are_bedrock_valid_and_unique():
    """Disallowed chars → '_', capped at 64, and de-duplicated."""
    pattern = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
    fields = [
        PostCallField(name="call summary", type="text"),
        PostCallField(name="call/summary", type="text"),  # same base → must dedupe
        PostCallField(name="résumé #", type="text"),
        PostCallField(name="x" * 80, type="text"),  # over Bedrock's 64-char cap
    ]
    keys = _safe_property_keys(fields)
    assert all(pattern.match(k) for k in keys), keys
    assert len(set(keys)) == len(keys)  # all unique
