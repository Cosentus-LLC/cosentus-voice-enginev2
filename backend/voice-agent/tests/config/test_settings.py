"""Tests for app.config.settings — env-var-backed Settings model."""

from __future__ import annotations

import pytest
from app.config.settings import Settings
from pydantic import ValidationError

# ── Helpers ─────────────────────────────────────────────────────────────────


# Every env var Settings cares about. Stripped at the start of every test
# so values leaked from the parent shell or a stray .env can't silently
# satisfy a "required" field.
_ALL_SETTINGS_ENV = (
    "VOICE_API_LAMBDA_NAME",
    "API_KEY_SECRET_ARN",
    "ASSEMBLYAI_API_KEY",
    "ELEVENLABS_API_KEY",
    "AWS_REGION",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "SERVICE_PORT",
    "MAX_CONCURRENT_CALLS",
    "DAILY_DIALIN_WEBHOOK_HMAC",
    "DAILY_RECORDING_WEBHOOK_HMAC",
    "RECORDING_BUCKET",
    "RECORDING_ROLE_ARN",
    "RECORDING_REGION",
    "DISABLED_TOOLS",
    "KNOWLEDGE_PREFETCH_ENABLED",
    "KNOWLEDGE_CACHE_TTL_SECS",
    "KNOWLEDGE_CACHE_MAX_ENTRIES",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every Settings env var before each test."""
    for var in _ALL_SETTINGS_ENV:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)


def _set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the two required env vars for tests that need a successful load."""
    monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "medcloud-voice-api:test")
    monkeypatch.setenv(
        "API_KEY_SECRET_ARN",
        "arn:aws:secretsmanager:us-east-1:123456789012:secret:test-AbCdEf",
    )


def _settings() -> Settings:
    """Construct Settings without reading .env (avoids local-file contamination)."""
    return Settings(_env_file=None)


# ── Required-field validation ───────────────────────────────────────────────


class TestSettingsRequiredFields:
    def test_loads_cleanly_when_all_required_set(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        s = _settings()

        assert s.voice_api_lambda_name == "medcloud-voice-api:test"
        assert s.api_key_secret_arn.startswith("arn:aws:secretsmanager:")

    def test_missing_voice_api_lambda_name_raises(self, monkeypatch: pytest.MonkeyPatch):
        # Only the secret ARN is set; voice_api_lambda_name is missing.
        monkeypatch.setenv("API_KEY_SECRET_ARN", "arn:test")

        with pytest.raises(ValidationError) as exc_info:
            _settings()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("voice_api_lambda_name",) for e in errors)

    def test_missing_api_key_secret_arn_raises(self, monkeypatch: pytest.MonkeyPatch):
        # Only the lambda name is set; api_key_secret_arn is missing.
        monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "test-lambda")

        with pytest.raises(ValidationError) as exc_info:
            _settings()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("api_key_secret_arn",) for e in errors)


# ── Defaults ────────────────────────────────────────────────────────────────


class TestSettingsDefaults:
    def test_optional_fields_use_documented_defaults(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        # Strip any host-environment fields so we test pure defaults.
        for env in (
            "AWS_REGION",
            "ENVIRONMENT",
            "LOG_LEVEL",
            "SERVICE_PORT",
            "MAX_CONCURRENT_CALLS",
            "DAILY_DIALIN_WEBHOOK_HMAC",
            "DISABLED_TOOLS",
            "POST_CALL_MODEL_FALLBACK_CHAIN",
            "DAILY_API_KEY",
            "ASSEMBLYAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "DAILY_RECORDING_WEBHOOK_HMAC",
            "RECORDING_BUCKET",
            "RECORDING_ROLE_ARN",
            "RECORDING_REGION",
        ):
            monkeypatch.delenv(env, raising=False)
        s = _settings()

        assert s.aws_region == "us-east-1"
        assert s.environment == "production"
        assert s.log_level == "INFO"
        assert s.service_port == 8080
        # Layer 9 starting point; Layer 9.5 scale test will validate.
        assert s.max_concurrent_calls == 6
        assert s.disabled_tools == ""
        # #20 offline failover is opt-in; empty default = no fallback chain.
        assert s.post_call_model_fallback_chain == ""
        # Layer 9 fields default to empty so local dev / unit tests
        # don't require Daily / S3 access just to construct Settings.
        assert s.daily_api_key == ""
        assert s.daily_dialin_webhook_hmac == ""
        assert s.daily_recording_webhook_hmac == ""
        assert s.recording_bucket == ""
        assert s.recording_role_arn == ""
        assert s.recording_region == "us-east-1"
        # Vendor API keys default to empty too — build_stt / build_tts
        # raise a clear ValueError if they're still empty at call time.
        assert s.assemblyai_api_key == ""
        assert s.elevenlabs_api_key == ""
        # #56 knowledge prefetch is opt-in and bounded per call.
        assert s.knowledge_prefetch_enabled is False
        assert s.knowledge_cache_ttl_secs == 300
        assert s.knowledge_cache_max_entries == 64


class TestSettingsFeatureFlags:
    def test_context_summarization_defaults_off(self, monkeypatch: pytest.MonkeyPatch):
        """#22 bounded-context summarization ships gated and off by default,
        so the flow-off production path is byte-identical to pre-#22."""
        _set_required(monkeypatch)
        monkeypatch.delenv("CONTEXT_SUMMARIZATION_ENABLED", raising=False)
        assert _settings().context_summarization_enabled is False

    def test_context_summarization_enables_from_env(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CONTEXT_SUMMARIZATION_ENABLED", "true")
        assert _settings().context_summarization_enabled is True

    def test_knowledge_prefetch_defaults_off(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        monkeypatch.delenv("KNOWLEDGE_PREFETCH_ENABLED", raising=False)
        assert _settings().knowledge_prefetch_enabled is False

    def test_knowledge_prefetch_enables_from_env(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("KNOWLEDGE_PREFETCH_ENABLED", "true")
        assert _settings().knowledge_prefetch_enabled is True

    def test_knowledge_cache_knobs_coerce_from_env_strings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        _set_required(monkeypatch)
        monkeypatch.setenv("KNOWLEDGE_CACHE_TTL_SECS", "120")
        monkeypatch.setenv("KNOWLEDGE_CACHE_MAX_ENTRIES", "12")
        settings = _settings()

        assert settings.knowledge_cache_ttl_secs == 120
        assert settings.knowledge_cache_max_entries == 12


# ── Type coercion ───────────────────────────────────────────────────────────


class TestSettingsCoercion:
    def test_int_fields_coerce_from_env_strings(self, monkeypatch: pytest.MonkeyPatch):
        # Env vars are always strings; pydantic coerces.
        _set_required(monkeypatch)
        monkeypatch.setenv("SERVICE_PORT", "9000")
        monkeypatch.setenv("MAX_CONCURRENT_CALLS", "12")

        s = _settings()
        assert s.service_port == 9000
        assert isinstance(s.service_port, int)
        assert s.max_concurrent_calls == 12
        assert isinstance(s.max_concurrent_calls, int)

    def test_int_field_rejects_non_numeric_string(self, monkeypatch: pytest.MonkeyPatch):
        # Pydantic should fail loudly if an int field gets garbage.
        _set_required(monkeypatch)
        monkeypatch.setenv("SERVICE_PORT", "not-a-number")

        with pytest.raises(ValidationError):
            _settings()


# ── Case insensitivity ──────────────────────────────────────────────────────


class TestSettingsCaseInsensitive:
    def test_uppercase_env_vars_bind(self, monkeypatch: pytest.MonkeyPatch):
        # Standard env-var convention.
        _set_required(monkeypatch)
        monkeypatch.setenv("DAILY_DIALIN_WEBHOOK_HMAC", "base64-secret")
        monkeypatch.setenv("DAILY_RECORDING_WEBHOOK_HMAC", "recording-secret")
        monkeypatch.setenv("RECORDING_BUCKET", "recording-bucket")
        monkeypatch.setenv("RECORDING_ROLE_ARN", "arn:aws:iam::123:role/Daily")
        monkeypatch.setenv("RECORDING_REGION", "us-west-2")
        s = _settings()
        assert s.voice_api_lambda_name == "medcloud-voice-api:test"
        assert s.daily_dialin_webhook_hmac == "base64-secret"
        assert s.daily_recording_webhook_hmac == "recording-secret"
        assert s.recording_bucket == "recording-bucket"
        assert s.recording_role_arn == "arn:aws:iam::123:role/Daily"
        assert s.recording_region == "us-west-2"

    def test_lowercase_env_vars_also_bind(self, monkeypatch: pytest.MonkeyPatch):
        # case_sensitive=False: lowercase resolves to the same field.
        monkeypatch.setenv("voice_api_lambda_name", "lc-lambda")
        monkeypatch.setenv("api_key_secret_arn", "arn:lc")

        s = _settings()
        assert s.voice_api_lambda_name == "lc-lambda"
        assert s.api_key_secret_arn == "arn:lc"


# ── Extra-field handling ────────────────────────────────────────────────────


class TestSettingsExtraIgnore:
    def test_unrelated_env_vars_silently_ignored(self, monkeypatch: pytest.MonkeyPatch):
        # Fargate sets a slew of AWS-managed env vars Settings doesn't
        # model. extra='ignore' keeps construction clean.
        _set_required(monkeypatch)
        monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "https://ecs/foo")
        monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_ECS_FARGATE")
        monkeypatch.setenv("RANDOM_NOISE_VAR", "x")

        # No exception; settings still loads.
        s = _settings()
        assert s.voice_api_lambda_name == "medcloud-voice-api:test"


# ── Vendor API keys ─────────────────────────────────────────────────────────


class TestSettingsVendorApiKeys:
    def test_read_from_env(self, monkeypatch: pytest.MonkeyPatch):
        # The Fargate task definition injects these from Secrets
        # Manager as env vars; pydantic-settings reads them at boot.
        _set_required(monkeypatch)
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aa-from-env")
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-from-env")

        s = _settings()
        assert s.assemblyai_api_key == "aa-from-env"
        assert s.elevenlabs_api_key == "el-from-env"

    def test_default_empty_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        s = _settings()
        assert s.assemblyai_api_key == ""
        assert s.elevenlabs_api_key == ""


# ── disabled_tools handling ─────────────────────────────────────────────────


class TestSettingsDisabledTools:
    def test_stored_as_raw_string(self, monkeypatch: pytest.MonkeyPatch):
        # Settings doesn't parse the CSV — the consumer in the tools
        # layer (Layer 4) does. Settings just preserves the string.
        _set_required(monkeypatch)
        monkeypatch.setenv("DISABLED_TOOLS", "transfer_call,press_digit")

        s = _settings()
        assert s.disabled_tools == "transfer_call,press_digit"

    def test_default_empty(self, monkeypatch: pytest.MonkeyPatch):
        _set_required(monkeypatch)
        s = _settings()
        assert s.disabled_tools == ""
