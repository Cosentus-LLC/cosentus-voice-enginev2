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
    "AWS_REGION",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "SERVICE_PORT",
    "MAX_CONCURRENT_CALLS",
    "DISABLED_TOOLS",
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
        s = _settings()

        assert s.aws_region == "us-east-1"
        assert s.environment == "production"
        assert s.log_level == "INFO"
        assert s.service_port == 8080
        assert s.max_concurrent_calls == 4
        assert s.disabled_tools == ""


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
        s = _settings()
        assert s.voice_api_lambda_name == "medcloud-voice-api:test"

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
