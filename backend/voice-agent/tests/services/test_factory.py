"""Tests for app.services.factory — STT/TTS/LLM construction + Bedrock resolver."""

from __future__ import annotations

import pytest
from app.config.agent_config import (
    AgentConfig,
    LLMConfig,
    STTConfig,
    TTSConfig,
    TTSSettings,
)
from app.config.settings import Settings
from app.services.factory import (
    _BEDROCK_DEFAULT_MODEL,
    _ELEVENLABS_DEFAULT_MODEL,
    _ELEVENLABS_DEFAULT_VOICE_ID,
    _SHORT_TO_BEDROCK,
    _STT_ENCODING,
    _STT_MODEL,
    _STT_SAMPLE_RATE,
    _STT_SHOULD_INTERRUPT,
    _STT_VAD_FORCE_TURN_ENDPOINT,
    _STT_VAD_THRESHOLD,
    build_llm,
    build_stt,
    build_tts,
    resolve_bedrock_model_id,
)
from pipecat.transcriptions.language import Language

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def assemblyai_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-aa-key")
    return "test-aa-key"


@pytest.fixture
def elevenlabs_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-el-key")
    return "test-el-key"


@pytest.fixture
def settings_fixture(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Layer 2 Settings instance with a known region."""
    monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "test-lambda")
    monkeypatch.setenv("API_KEY_SECRET_ARN", "arn:test")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    return Settings(_env_file=None)


def _agent(
    *,
    name: str = "test-agent",
    system_prompt: str = "You are a test agent.",
    llm_model: str = "claude-haiku-4-5",
    llm_max_tokens: int | None = None,
    llm_temperature: float | None = None,
    tts_voice_id: str = "",
    tts_model: str = "",
    tts_stability: float | None = None,
    tts_use_speaker_boost: bool | None = None,
    stt_keywords: list[str] | None = None,
) -> AgentConfig:
    """Construct a minimal AgentConfig for tests."""
    return AgentConfig(
        name=name,
        system_prompt=system_prompt,
        llm=LLMConfig(
            model=llm_model,
            max_tokens=llm_max_tokens if llm_max_tokens is not None else 200,
            temperature=llm_temperature if llm_temperature is not None else 0.7,
        ),
        tts=TTSConfig(
            voice_id=tts_voice_id,
            model=tts_model or "eleven_turbo_v2_5",
            settings=TTSSettings(
                stability=tts_stability,
                use_speaker_boost=tts_use_speaker_boost,
            ),
        ),
        stt=STTConfig(keywords=stt_keywords or []),
    )


# ── resolve_bedrock_model_id ────────────────────────────────────────────────


class TestResolveBedrockModelId:
    def test_empty_string_returns_default_haiku(self):
        assert resolve_bedrock_model_id("") == _BEDROCK_DEFAULT_MODEL

    def test_full_bedrock_id_passes_through_unchanged(self):
        # Anything with a "." is assumed to already be a full Bedrock ID.
        full = "us.anthropic.claude-sonnet-4-7-20260101-v1:0"
        assert resolve_bedrock_model_id(full) == full

    def test_known_short_form_haiku_4_5(self):
        # The everyday case — Aurora stores "claude-haiku-4-5", we
        # need the full inference profile ID.
        assert (
            resolve_bedrock_model_id("claude-haiku-4-5")
            == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )

    def test_sonnet_4_6_resolves_with_no_date_suffix(self):
        # Production-paid edge case. Sonnet 4.6's profile has no
        # date suffix; "fixing" this would break every Sonnet 4-6
        # call (incident 2026-04-24).
        result = resolve_bedrock_model_id("claude-sonnet-4-6")
        assert result == "us.anthropic.claude-sonnet-4-6"
        assert "2026" not in result and "v1:0" not in result, (
            "Sonnet 4-6 must resolve to the bare profile ID"
        )

    def test_unknown_short_form_passes_through_with_warning(self, caplog):
        # Unknown short forms aren't fatal — the resolver lets
        # Bedrock produce its own canonical error rather than
        # guessing at the right ID.
        result = resolve_bedrock_model_id("claude-haiku-99-9")
        assert result == "claude-haiku-99-9"

    def test_every_short_form_in_table_resolves(self):
        # Sanity: every entry in _SHORT_TO_BEDROCK is reachable via
        # the resolver. Catches accidental table-key typos.
        for short, expected in _SHORT_TO_BEDROCK.items():
            assert resolve_bedrock_model_id(short) == expected

    def test_table_does_not_contain_dot_for_short_keys(self):
        # The "contains-dot → pass through" branch fires before the
        # table lookup. If a short-form key ever had a dot, it'd
        # never reach the lookup. Guard against that.
        for short in _SHORT_TO_BEDROCK:
            assert "." not in short, f"short key {short!r} contains a dot"


# ── build_stt ───────────────────────────────────────────────────────────────


class TestBuildStt:
    def test_returns_assemblyai_service_with_locked_in_constructor_args(
        self, mocker, assemblyai_env: str
    ):
        mock_cls = mocker.patch("app.services.factory.AssemblyAISTTService")

        agent = _agent()
        result = build_stt(agent)

        assert result is mock_cls.return_value
        mock_cls.assert_called_once()

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["api_key"] == assemblyai_env
        assert kwargs["sample_rate"] == _STT_SAMPLE_RATE == 8000
        assert kwargs["encoding"] == _STT_ENCODING == "pcm_s16le"
        assert kwargs["vad_force_turn_endpoint"] is _STT_VAD_FORCE_TURN_ENDPOINT is False
        assert kwargs["should_interrupt"] is _STT_SHOULD_INTERRUPT is False
        # settings was constructed via the patched Settings inner class
        assert kwargs["settings"] is mock_cls.Settings.return_value

    def test_settings_has_locked_in_model_language_and_threshold(self, mocker, assemblyai_env: str):
        mock_cls = mocker.patch("app.services.factory.AssemblyAISTTService")

        build_stt(_agent())

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["model"] == _STT_MODEL == "u3-rt-pro"
        assert settings_kwargs["language"] == Language.EN
        assert settings_kwargs["vad_threshold"] == _STT_VAD_THRESHOLD == 0.3

    def test_settings_has_keyterms_prompt_when_keywords_set(self, mocker, assemblyai_env: str):
        mock_cls = mocker.patch("app.services.factory.AssemblyAISTTService")

        build_stt(_agent(stt_keywords=["claim", "patient", "deductible"]))

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["keyterms_prompt"] == [
            "claim",
            "patient",
            "deductible",
        ]

    def test_settings_omits_keyterms_prompt_when_keywords_empty(self, mocker, assemblyai_env: str):
        # Pipecat's _NotGiven sentinel means "field unset" → vendor
        # default. Passing keyterms_prompt=[] would override that
        # with a literal empty list. Don't.
        mock_cls = mocker.patch("app.services.factory.AssemblyAISTTService")

        build_stt(_agent(stt_keywords=[]))

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert "keyterms_prompt" not in settings_kwargs

    def test_raises_when_assemblyai_api_key_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ASSEMBLYAI_API_KEY"):
            build_stt(_agent())


# ── build_tts ───────────────────────────────────────────────────────────────


class TestBuildTts:
    def test_returns_elevenlabs_service_with_settings(self, mocker, elevenlabs_env: str):
        mock_cls = mocker.patch("app.services.factory.ElevenLabsTTSService")

        result = build_tts(_agent(tts_voice_id="agent-voice"))

        assert result is mock_cls.return_value
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["api_key"] == elevenlabs_env
        assert kwargs["settings"] is mock_cls.Settings.return_value

    def test_voice_falls_through_agent_then_default(self, mocker, elevenlabs_env: str):
        mock_cls = mocker.patch("app.services.factory.ElevenLabsTTSService")

        # Per-agent voice_id is used.
        build_tts(_agent(tts_voice_id="agent-voice-id"))
        assert mock_cls.Settings.call_args.kwargs["voice"] == "agent-voice-id"

        # Empty per-agent voice_id falls back to the platform default.
        mock_cls.reset_mock()
        build_tts(_agent(tts_voice_id=""))
        assert mock_cls.Settings.call_args.kwargs["voice"] == _ELEVENLABS_DEFAULT_VOICE_ID
        assert _ELEVENLABS_DEFAULT_VOICE_ID == "vW1NxlzqX8WROgpQAghR"

    def test_model_falls_through_agent_then_default(self, mocker, elevenlabs_env: str):
        mock_cls = mocker.patch("app.services.factory.ElevenLabsTTSService")

        # Per-agent model is used.
        build_tts(_agent(tts_model="eleven_multilingual_v2"))
        assert mock_cls.Settings.call_args.kwargs["model"] == "eleven_multilingual_v2"

        # Empty per-agent model falls back to the platform default.
        # The AgentConfig default is "eleven_turbo_v2_5"; the factory's
        # platform default is "eleven_flash_v2_5". Both get covered:
        # AgentConfig produces "eleven_turbo_v2_5" by default → factory
        # uses it directly.
        mock_cls.reset_mock()
        agent = _agent(tts_model="")
        # _agent(tts_model="") falls through to "eleven_turbo_v2_5" in the
        # AgentConfig default — exercise that path.
        agent.tts.model = ""  # Force empty to test the factory's "or default" path.
        build_tts(agent)
        assert (
            mock_cls.Settings.call_args.kwargs["model"]
            == _ELEVENLABS_DEFAULT_MODEL
            == "eleven_flash_v2_5"
        )

    def test_settings_includes_stability_when_set(self, mocker, elevenlabs_env: str):
        mock_cls = mocker.patch("app.services.factory.ElevenLabsTTSService")

        build_tts(_agent(tts_stability=0.7))

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["stability"] == 0.7

    def test_settings_includes_use_speaker_boost_when_set(self, mocker, elevenlabs_env: str):
        mock_cls = mocker.patch("app.services.factory.ElevenLabsTTSService")

        build_tts(_agent(tts_use_speaker_boost=True))

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["use_speaker_boost"] is True

    def test_settings_omits_voice_tuning_when_unset(self, mocker, elevenlabs_env: str):
        # None values must NOT be forwarded — Pipecat's _NotGiven
        # sentinel preserves vendor defaults; None would clobber.
        mock_cls = mocker.patch("app.services.factory.ElevenLabsTTSService")

        build_tts(
            _agent(tts_stability=None, tts_use_speaker_boost=None),
        )

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert "stability" not in settings_kwargs
        assert "use_speaker_boost" not in settings_kwargs

    def test_raises_when_elevenlabs_api_key_missing(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            build_tts(_agent())


# ── build_llm ───────────────────────────────────────────────────────────────


class TestBuildLlm:
    def test_returns_bedrock_service_with_resolved_model(self, mocker, settings_fixture: Settings):
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        result = build_llm(_agent(llm_model="claude-haiku-4-5"), settings_fixture)

        assert result is mock_cls.return_value
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert kwargs["aws_region"] == settings_fixture.aws_region == "us-west-2"
        assert kwargs["settings"] is mock_cls.Settings.return_value

    def test_aws_region_comes_from_settings_not_env(
        self, mocker, monkeypatch: pytest.MonkeyPatch, settings_fixture: Settings
    ):
        # Layer 2 Settings is the source of truth for region —
        # changing the env after Settings is constructed should NOT
        # affect the region passed to Bedrock.
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")
        monkeypatch.setenv("AWS_REGION", "ap-southeast-2")

        build_llm(_agent(), settings_fixture)

        assert mock_cls.call_args.kwargs["aws_region"] == "us-west-2"

    def test_enable_prompt_caching_always_true(self, mocker, settings_fixture: Settings):
        # v1 logged the per-agent flag but never wired it; v2
        # hardcodes ON. Verifying the constant.
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        build_llm(_agent(), settings_fixture)

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["enable_prompt_caching"] is True

    def test_max_tokens_and_temperature_passed_when_set(self, mocker, settings_fixture: Settings):
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        build_llm(
            _agent(llm_max_tokens=1500, llm_temperature=0.2),
            settings_fixture,
        )

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["max_tokens"] == 1500
        assert settings_kwargs["temperature"] == 0.2

    def test_max_tokens_and_temperature_omitted_when_none(self, mocker, settings_fixture: Settings):
        # Pipecat's _NotGiven sentinel preserves vendor defaults;
        # explicit None would clobber them. Build an agent with the
        # LLM defaults overridden to None to test this.
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        agent = _agent()
        agent.llm.max_tokens = None  # type: ignore[assignment]
        agent.llm.temperature = None  # type: ignore[assignment]

        build_llm(agent, settings_fixture)

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert "max_tokens" not in settings_kwargs
        assert "temperature" not in settings_kwargs

    def test_unknown_model_short_form_passes_through(self, mocker, settings_fixture: Settings):
        # Unknown short forms aren't fatal at the factory layer —
        # Bedrock will reject them with its own error. Verifies the
        # resolver hooks into build_llm correctly.
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        build_llm(_agent(llm_model="claude-mystery-9-9"), settings_fixture)

        assert mock_cls.call_args.kwargs["model"] == "claude-mystery-9-9"

    def test_empty_model_uses_hardcoded_default(self, mocker, settings_fixture: Settings):
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        build_llm(_agent(llm_model=""), settings_fixture)

        assert mock_cls.call_args.kwargs["model"] == _BEDROCK_DEFAULT_MODEL

    def test_system_instruction_passed_from_agent(self, mocker, settings_fixture: Settings):
        # Surfaced by walking-skeleton run #1: putting the system
        # prompt as a role="system" message in LLMContext got silently
        # rendered as user input by Bedrock. Pipecat's documented
        # pattern is Settings.system_instruction, which is what
        # build_llm now uses.
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        prompt = "You are Chris, a billing specialist."
        build_llm(_agent(system_prompt=prompt), settings_fixture)

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["system_instruction"] == prompt

    def test_system_instruction_always_passed_unconditionally(
        self, mocker, settings_fixture: Settings
    ):
        # Even an empty system_prompt is forwarded — the field is
        # always set on Settings, never elided. This keeps the
        # behavior predictable: nothing in build_llm decides whether
        # the system prompt "deserves" to be passed.
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        build_llm(_agent(system_prompt=""), settings_fixture)

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert "system_instruction" in settings_kwargs
        assert settings_kwargs["system_instruction"] == ""

    def test_multiline_system_prompt_passed_unchanged(self, mocker, settings_fixture: Settings):
        # Cosentus's prompts are multi-KB, multi-section markdown
        # with headers, lists, and template variables. Verify the
        # full string passes through intact — no truncation, no
        # stripping, no transformation.
        mock_cls = mocker.patch("app.services.factory.AWSBedrockLLMService")

        prompt = (
            "# Identity\n"
            "You are Chris.\n\n"
            "# Rules\n"
            "1. One question at a time.\n"
            "2. {{template_variable}} stays literal.\n"
        )
        build_llm(_agent(system_prompt=prompt), settings_fixture)

        settings_kwargs = mock_cls.Settings.call_args.kwargs
        assert settings_kwargs["system_instruction"] == prompt
        assert "{{template_variable}}" in settings_kwargs["system_instruction"]
