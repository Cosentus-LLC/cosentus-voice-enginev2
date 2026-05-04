"""Tests for ``app/persistence/post_call.py``."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from app.config.agent_config import (
    AgentConfig,
    LLMConfig,
    PostCallConfig,
    PostCallField,
)
from app.config.settings import Settings
from app.persistence.post_call import (
    _build_extraction_prompt,
    _parse_and_validate,
    run_post_call_analyses,
)
from botocore.exceptions import ClientError

# ── Test fixtures ──────────────────────────────────────────────────────────


def _settings() -> Settings:
    return Settings(
        voice_api_lambda_name="test-voice-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
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


def _bedrock_response(text: str) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        }
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
    bedrock_text = json.dumps({"summary": "Claim 12345 is paid."})
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_bedrock_response(bedrock_text))
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
    bedrock_text = json.dumps({"outcome": "resolved"})
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_bedrock_response(bedrock_text))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"outcome": "resolved"}


@pytest.mark.asyncio
async def test_returns_invalid_marker_for_off_list_selector_choice():
    """v1 semantics: invalid choice gets ``invalid: <value>`` prefix.

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
    bedrock_text = json.dumps({"outcome": "in_progress"})
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_bedrock_response(bedrock_text))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"outcome": "invalid: in_progress"}


@pytest.mark.asyncio
async def test_handles_multiple_fields():
    agent = _agent(
        fields=[
            PostCallField(name="summary", type="text", description="Summary"),
            PostCallField(name="outcome", type="selector", choices=["resolved", "dropped"]),
        ]
    )
    bedrock_text = json.dumps({"summary": "Done.", "outcome": "resolved"})
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_bedrock_response(bedrock_text))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"summary": "Done.", "outcome": "resolved"}


@pytest.mark.asyncio
async def test_missing_field_in_response_defaults_to_empty():
    """Bedrock omits a field → coerce to empty string, don't fail."""
    agent = _agent(
        fields=[
            PostCallField(name="summary", type="text"),
            PostCallField(name="reference_number", type="text"),
        ]
    )
    bedrock_text = json.dumps({"summary": "ok"})  # reference_number missing
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_bedrock_response(bedrock_text))
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
async def test_retries_once_on_invalid_json():
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    invalid = "this isn't json"
    valid = json.dumps({"summary": "fixed it"})
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        # First call returns garbage, second call returns valid JSON.
        mock_br.converse = MagicMock(
            side_effect=[
                _bedrock_response(invalid),
                _bedrock_response(valid),
            ]
        )
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {"summary": "fixed it"}
    assert mock_br.converse.call_count == 2


@pytest.mark.asyncio
async def test_returns_empty_after_retry_still_invalid_json():
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_bedrock_response("never going to be JSON"))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}
    assert mock_br.converse.call_count == 2  # original + 1 retry


@pytest.mark.asyncio
async def test_returns_empty_on_non_dict_json():
    """LLM returns a JSON array, not an object — invalid for our schema."""
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(return_value=_bedrock_response("[1, 2, 3]"))
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}


@pytest.mark.asyncio
async def test_never_raises_on_unexpected_exception():
    agent = _agent(fields=[PostCallField(name="summary", type="text")])
    with patch("app.persistence.post_call._BEDROCK_CLIENT") as mock_br:
        mock_br.converse = MagicMock(side_effect=RuntimeError("simulated unexpected"))
        # Should not raise; should return {}.
        result = await run_post_call_analyses(agent, {}, _transcript(), _settings())
    assert result == {}


# ── Markdown-fence stripping ───────────────────────────────────────────────


def test_parse_strips_json_code_fence():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    raw = '```json\n{"summary": "ok"}\n```'
    result = _parse_and_validate(raw, pca)
    assert result == {"summary": "ok"}


def test_parse_strips_bare_code_fence():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    raw = '```\n{"summary": "ok"}\n```'
    result = _parse_and_validate(raw, pca)
    assert result == {"summary": "ok"}


def test_parse_handles_no_fence():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    raw = '{"summary": "ok"}'
    result = _parse_and_validate(raw, pca)
    assert result == {"summary": "ok"}


def test_parse_returns_none_on_completely_invalid_json():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    assert _parse_and_validate("totally not json", pca) is None


def test_parse_returns_none_on_empty_string():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    assert _parse_and_validate("", pca) is None


def test_parse_coerces_non_string_text_field():
    """LLM returns an int for a text field; coerce to str rather than failing."""
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="count", type="text")],
    )
    raw = '{"count": 42}'
    result = _parse_and_validate(raw, pca)
    assert result == {"count": "42"}


# ── Prompt construction ───────────────────────────────────────────────────


def test_prompt_includes_case_data():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    prompt = _build_extraction_prompt(pca, {"claim_id": "12345"}, _transcript())
    assert "claim_id" in prompt
    assert "12345" in prompt


def test_prompt_includes_all_transcript_turns():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "I need to check claim 12345." in prompt
    assert "Of course. The claim is paid." in prompt


def test_prompt_renders_selector_with_choices():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[
            PostCallField(
                name="outcome",
                type="selector",
                choices=["resolved", "escalated"],
                description="What happened?",
            )
        ],
    )
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "selector" in prompt
    assert "resolved" in prompt
    assert "escalated" in prompt
    assert "What happened?" in prompt


def test_prompt_renders_text_field_clean():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text", description="2 sentences.")],
    )
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "summary (text)" in prompt
    assert "2 sentences." in prompt


def test_prompt_includes_format_examples():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[
            PostCallField(
                name="reference",
                type="text",
                format_examples=["REF-12345"],
            )
        ],
    )
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "REF-12345" in prompt


def test_prompt_handles_empty_case_data():
    pca = PostCallConfig(
        model="claude-haiku-4-5",
        fields=[PostCallField(name="summary", type="text")],
    )
    prompt = _build_extraction_prompt(pca, {}, _transcript())
    assert "(none)" in prompt
