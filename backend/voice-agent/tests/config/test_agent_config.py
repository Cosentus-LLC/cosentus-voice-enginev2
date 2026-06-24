"""Tests for app.config.agent_config — Pydantic models + Lambda loader."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from app.config.agent_config import (
    AgentConfig,
    AgentConfigLoadError,
    load_agent_config,
)
from app.config.settings import Settings
from botocore.exceptions import ClientError, ReadTimeoutError
from pydantic import ValidationError

# ── Fixtures ────────────────────────────────────────────────────────────────


def _runtime_config_json() -> dict[str, Any]:
    """Realistic GET /api/agents/:id/runtime-config response body.

    Shape source: v1's contract-trace audit (handoff #4 — what the
    lambda's ``buildRuntimeConfig`` actually returns). Includes
    fields v2 doesn't model so we can verify ``extra='ignore'``.
    """
    return {
        "name": "chris-claim-status",
        "display_name": "Chris (Claim Status)",
        "description": "Outbound claim-status agent",
        "system_prompt": "You are a Cosentus voice agent...",
        "first_message": "Hi, I'm calling about a claim status.",
        "flow_definition": None,
        "ivr_goal": "Reach a claims rep",
        "llm": {
            "provider": "anthropic",  # extra in v2
            "model": "claude-sonnet-4-6",
            "max_tokens": 390,
            "temperature": 0.2,
            "enable_prompt_caching": True,  # extra in v2
        },
        "tts": {
            "provider": "elevenlabs",  # extra in v2
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "model": "eleven_turbo_v2_5",
            "settings": {
                "stability": 0.75,
                "similarity_boost": 0.8,  # extra in v2
                "style": 0.0,  # extra in v2
                "use_speaker_boost": False,
                "speed": 1.0,  # extra in v2
            },
        },
        "stt": {
            "provider": "assemblyai",  # extra in v2
            "language": "en",  # extra in v2
            "keywords": ["claim", "patient", "deductible"],
        },
        "tools": [
            {"type": "end_call", "description": "End the call", "settings": {}},
            {
                "type": "transfer_call",
                "description": "Transfer the call",
                "settings": {"destination": "+15555550100"},
            },
        ],
        "recording": {"enabled": True, "channels": 2},  # whole object extra
        "post_call_analyses": {
            "model": "claude-haiku-4-5-20251001",
            "fields": [
                {
                    "name": "claim_status",
                    "type": "enum",
                    "description": "Status reported by payer",
                    "format_examples": [],
                    "choices": ["paid", "denied", "pending"],
                }
            ],
        },
        "_meta": {
            "agent_id": "00000000-0000-4000-8000-000000000001",
            "version": 1730486400000,
        },
    }


def _mock_invoke_response(
    status_code: int,
    body: Any,
    function_error: str | None = None,
) -> dict[str, Any]:
    """Build a fake boto3 ``Lambda.Invoke`` response.

    boto3 returns an envelope ``{"Payload": StreamingBody, ...}``
    where ``StreamingBody.read()`` yields the JSON-encoded
    API-Gateway-proxy response: ``{"statusCode", "body" (str), ...}``.
    """
    body_str = body if isinstance(body, str) else json.dumps(body)
    outer = {"statusCode": status_code, "body": body_str, "headers": {}}
    payload_bytes = json.dumps(outer).encode("utf-8")

    payload_mock = MagicMock()
    payload_mock.read.return_value = payload_bytes

    response: dict[str, Any] = {"Payload": payload_mock}
    if function_error:
        response["FunctionError"] = function_error
    return response


# ── Pydantic model tests ────────────────────────────────────────────────────


class TestAgentConfigParse:
    def test_realistic_lambda_response_parses_cleanly(self):
        cfg = AgentConfig.model_validate(_runtime_config_json())

        assert cfg.name == "chris-claim-status"
        assert cfg.display_name == "Chris (Claim Status)"
        assert cfg.system_prompt == "You are a Cosentus voice agent..."
        assert cfg.first_message == "Hi, I'm calling about a claim status."
        assert cfg.flow_definition is None
        assert cfg.ivr_goal == "Reach a claims rep"

        assert cfg.llm.model == "claude-sonnet-4-6"
        assert cfg.llm.max_tokens == 390
        assert cfg.llm.temperature == 0.2

        assert cfg.tts.voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert cfg.tts.model == "eleven_turbo_v2_5"
        assert cfg.tts.settings.stability == 0.75
        assert cfg.tts.settings.use_speaker_boost is False

        assert cfg.stt.keywords == ["claim", "patient", "deductible"]

        assert len(cfg.tools) == 2
        assert cfg.tools[0].type == "end_call"
        assert cfg.tools[1].type == "transfer_call"
        assert cfg.tools[1].settings == {"destination": "+15555550100"}

        assert cfg.post_call_analyses is not None
        assert cfg.post_call_analyses.model == "claude-haiku-4-5-20251001"
        assert len(cfg.post_call_analyses.fields) == 1
        assert cfg.post_call_analyses.fields[0].name == "claim_status"
        assert cfg.post_call_analyses.fields[0].choices == [
            "paid",
            "denied",
            "pending",
        ]

        assert cfg.meta.agent_id == "00000000-0000-4000-8000-000000000001"
        assert cfg.meta.updated_at_ms == 1730486400000

    def test_post_call_analyses_can_be_null(self):
        body = _runtime_config_json()
        body["post_call_analyses"] = None
        cfg = AgentConfig.model_validate(body)
        assert cfg.post_call_analyses is None

    def test_flow_definition_object_parses(self):
        flow = {
            "version": 1,
            "start": "reference_number",
            "nodes": [
                {
                    "id": "reference_number",
                    "type": "ask",
                    "capture": ["call_reference"],
                    "required": True,
                    "next": "done",
                },
                {"id": "done", "type": "end"},
            ],
        }
        body = _runtime_config_json()
        body["flow_definition"] = flow

        cfg = AgentConfig.model_validate(body)

        assert cfg.flow_definition == flow

    def test_flow_definition_is_modeled_in_dump(self):
        flow = {"version": 1, "start": "reference_number", "nodes": []}
        cfg = AgentConfig.model_validate({"name": "agent", "flow_definition": flow})

        assert cfg.model_dump()["flow_definition"] == flow

    def test_meta_alias_underscore_meta_deserializes(self):
        # The lambda sends ``_meta`` (wire convention); the Python
        # attribute is ``meta``. populate_by_name=True + alias="_meta"
        # makes the wire form work.
        body = {
            "name": "test",
            "_meta": {"agent_id": "abc-123", "version": 999},
        }
        cfg = AgentConfig.model_validate(body)
        assert cfg.meta.agent_id == "abc-123"
        assert cfg.meta.updated_at_ms == 999

    def test_meta_can_also_be_passed_by_python_name(self):
        # Tests aren't forced to mirror the wire's _meta convention.
        cfg = AgentConfig.model_validate(
            {
                "name": "test",
                "meta": {"agent_id": "abc-123", "updated_at_ms": 999},
            }
        )
        assert cfg.meta.agent_id == "abc-123"
        assert cfg.meta.updated_at_ms == 999

    def test_extra_fields_silently_dropped(self):
        # extra='ignore' lets v2 drop v1-era fields the lambda still
        # sends. See docs/v2-tech-debt-log.md entry 1.
        body = _runtime_config_json()

        # Sanity check: the fixture really does include the extras.
        assert "provider" in body["llm"]
        assert "enable_prompt_caching" in body["llm"]
        assert "provider" in body["tts"]
        assert "similarity_boost" in body["tts"]["settings"]
        assert "speed" in body["tts"]["settings"]
        assert "provider" in body["stt"]
        assert "language" in body["stt"]
        assert "recording" in body

        # All silently dropped — no exceptions.
        cfg = AgentConfig.model_validate(body)

        # And dumping the model produces only modeled fields.
        dumped = cfg.model_dump(by_alias=True)
        assert "provider" not in dumped["llm"]
        assert "enable_prompt_caching" not in dumped["llm"]
        assert "provider" not in dumped["tts"]
        assert "similarity_boost" not in dumped["tts"]["settings"]
        assert "style" not in dumped["tts"]["settings"]
        assert "speed" not in dumped["tts"]["settings"]
        assert "provider" not in dumped["stt"]
        assert "language" not in dumped["stt"]
        assert "recording" not in dumped


class TestUnknownFieldLogging:
    """Log-on-unknown guard (B2 / tech-debt Entry 1).

    We keep ``extra='ignore'`` so valid configs still parse, but a
    ``mode='before'`` validator on ``_RuntimeConfigModel`` logs a loud
    ``agent_config_unknown_fields`` warning for any field that is
    neither modeled nor on the per-model ``_known_extra_fields``
    allowlist — so contract drift surfaces at parse time instead of a
    default silently winning mid-call.
    """

    def _unknown_field_warnings(self, mock_logger) -> list:
        return [
            call
            for call in mock_logger.warning.call_args_list
            if call.args and call.args[0] == "agent_config_unknown_fields"
        ]

    def test_unknown_top_level_field_logged_loudly(self, mocker):
        mock_logger = mocker.patch("app.config.agent_config.logger")

        AgentConfig.model_validate({"name": "agent", "surprise_field": 1})

        warnings = self._unknown_field_warnings(mock_logger)
        assert len(warnings) == 1
        assert warnings[0].kwargs["model"] == "AgentConfig"
        assert "surprise_field" in warnings[0].kwargs["unknown_fields"]

    def test_unknown_nested_field_logged_with_submodel_name(self, mocker):
        mock_logger = mocker.patch("app.config.agent_config.logger")

        AgentConfig.model_validate(
            {"name": "agent", "llm": {"model": "claude-haiku-4-5", "frobnicate": True}}
        )

        warnings = self._unknown_field_warnings(mock_logger)
        assert len(warnings) == 1
        assert warnings[0].kwargs["model"] == "LLMConfig"
        assert warnings[0].kwargs["unknown_fields"] == ["frobnicate"]

    def test_known_extra_fields_not_logged(self, mocker):
        # The realistic fixture carries every Entry-1 extra
        # (llm.provider, tts.settings.style, recording, ...). None of
        # them should produce a drift warning — they're allowlisted.
        mock_logger = mocker.patch("app.config.agent_config.logger")

        AgentConfig.model_validate(_runtime_config_json())

        assert self._unknown_field_warnings(mock_logger) == []

    def test_flow_definition_is_not_logged_as_unknown(self, mocker):
        mock_logger = mocker.patch("app.config.agent_config.logger")

        AgentConfig.model_validate(
            {
                "name": "agent",
                "flow_definition": {"version": 1, "start": "reference_number", "nodes": []},
            }
        )

        assert self._unknown_field_warnings(mock_logger) == []

    def test_known_good_config_still_parses(self):
        # Acceptance criterion: a currently-valid payload still parses
        # cleanly — the guard only logs, never raises.
        cfg = AgentConfig.model_validate(_runtime_config_json())
        assert cfg.name == "chris-claim-status"


class TestAgentConfigSpeakFirst:
    def test_default_is_true_when_field_missing_from_lambda(self):
        # Lambda hasn't deployed the speak_first column yet → field
        # is absent from the response. AgentConfig defaults to True
        # (matches every existing production agent's implicit behavior).
        cfg = AgentConfig.model_validate({"name": "agent", "first_message": "Hello!"})
        assert cfg.speak_first is True

    def test_explicit_true_loads(self):
        cfg = AgentConfig.model_validate({"name": "agent", "speak_first": True})
        assert cfg.speak_first is True

    def test_explicit_false_loads(self):
        cfg = AgentConfig.model_validate({"name": "agent", "speak_first": False})
        assert cfg.speak_first is False

    def test_realistic_response_with_speak_first(self):
        body = _runtime_config_json()
        body["speak_first"] = False
        cfg = AgentConfig.model_validate(body)
        assert cfg.speak_first is False
        assert cfg.name == "chris-claim-status"


class TestLLMConfigDefaults:
    """Regression guard for the engine-side default LLM model.

    The ``LLMConfig.model`` Pydantic default fires only when the lambda
    payload omits the ``llm.model`` field entirely (defensive — in
    practice the lambda always populates it). We pin the default to
    Haiku 4.5 to match the cost-conscious defaults strategy across the
    platform; changing this should be intentional, so the test exists
    to make any default-flip visible.
    """

    def test_llm_model_default_is_haiku_4_5(self):
        from app.config.agent_config import LLMConfig

        cfg = LLMConfig()
        assert cfg.model == "claude-haiku-4-5"

    def test_agent_config_without_llm_section_defaults_to_haiku(self):
        cfg = AgentConfig.model_validate(
            {
                # Minimum payload — name is required, everything else
                # falls through to defaults.
                "name": "test-defaults",
            }
        )
        assert cfg.llm.model == "claude-haiku-4-5"

    def test_tts_model_default_is_empty(self):
        """TTSConfig.model carries no Layer-1 default — the factory's
        _ELEVENLABS_DEFAULT_MODEL is the single source of truth for the
        platform TTS-model fallback (eleven_flash_v2_5). An empty value
        here falls through to that fallback in build_tts, mirroring
        voice_id. This test makes any reintroduction of a Layer-1
        default visible (see tech-debt Entry 8 / B4)."""
        from app.config.agent_config import TTSConfig

        assert TTSConfig().model == ""

    def test_post_call_model_default_is_sonnet(self):
        """The offline post-call extraction defaults to the STRONGER model
        (#20). It runs once per call, off the latency-critical live path,
        so it trades cost/latency for accuracy — the inverse of the live
        ``LLMConfig.model`` Haiku default. Fires only when the lambda omits
        ``post_call_analyses.model``; an explicit value still wins. This
        test makes any future default-flip visible."""
        from app.config.agent_config import PostCallConfig

        assert PostCallConfig().model == "claude-sonnet-4-6"


class TestAgentConfigValidation:
    def test_empty_config_fails_validation_name_required(self):
        with pytest.raises(ValidationError) as exc:
            AgentConfig.model_validate({})
        errors = exc.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)

    def test_tool_config_type_required(self):
        with pytest.raises(ValidationError) as exc:
            AgentConfig.model_validate(
                {
                    "name": "test",
                    "tools": [{"description": "missing type"}],
                }
            )
        errors = exc.value.errors()
        assert any(e["loc"] == ("tools", 0, "type") for e in errors)

    def test_post_call_field_name_required(self):
        with pytest.raises(ValidationError) as exc:
            AgentConfig.model_validate(
                {
                    "name": "test",
                    "post_call_analyses": {
                        "model": "claude-haiku-4-5-20251001",
                        "fields": [{"description": "missing name"}],
                    },
                }
            )
        errors = exc.value.errors()
        assert any(e["loc"] == ("post_call_analyses", "fields", 0, "name") for e in errors)


# ── Loader tests ────────────────────────────────────────────────────────────


@pytest.fixture
def voice_api_lambda_env(monkeypatch: pytest.MonkeyPatch) -> str:
    name = "medcloud-voice-api:test"
    monkeypatch.setenv("VOICE_API_LAMBDA_NAME", name)
    return name


class TestLoadAgentConfig:
    async def test_success_returns_parsed_config(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = _mock_invoke_response(200, _runtime_config_json())

        cfg = await load_agent_config("chris-claim-status")

        assert isinstance(cfg, AgentConfig)
        assert cfg.name == "chris-claim-status"
        assert cfg.meta.agent_id == "00000000-0000-4000-8000-000000000001"
        assert cfg.meta.updated_at_ms == 1730486400000

        # Lambda invoked with the right shape. ``_invoke_lambda_sync``
        # takes (function_name, payload, settings) — settings is the
        # third arg, ``None`` here because no Settings object is passed.
        function_name, payload_bytes, settings_arg = mock_invoke.call_args.args
        assert function_name == voice_api_lambda_env
        assert settings_arg is None
        event = json.loads(payload_bytes)
        assert event["httpMethod"] == "GET"
        assert event["path"] == "/api/agents/chris-claim-status/runtime-config"

    async def test_404_raises_agent_not_found(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = _mock_invoke_response(404, {"detail": "Agent not found"})

        with pytest.raises(AgentConfigLoadError, match="agent not found"):
            await load_agent_config("nope")

    async def test_500_raises_with_status(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = _mock_invoke_response(500, {"detail": "DB unreachable"})

        with pytest.raises(AgentConfigLoadError, match="HTTP 500"):
            await load_agent_config("chris-claim-status")

    async def test_malformed_outer_json_raises(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        # The boto3 envelope's Payload is not valid JSON.
        payload_mock = MagicMock()
        payload_mock.read.return_value = b"not-json-at-all"
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = {"Payload": payload_mock}

        with pytest.raises(AgentConfigLoadError, match="not valid JSON"):
            await load_agent_config("chris-claim-status")

    async def test_malformed_inner_body_json_raises(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        # Outer envelope is fine; inner ``body`` string isn't valid JSON.
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = _mock_invoke_response(200, "not-{json")

        with pytest.raises(AgentConfigLoadError, match="body was not valid JSON"):
            await load_agent_config("chris-claim-status")

    async def test_pydantic_validation_failure_raises(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        # 200 with valid JSON body but missing the required ``name`` field.
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = _mock_invoke_response(200, {})

        with pytest.raises(AgentConfigLoadError, match="schema"):
            await load_agent_config("anything")

    async def test_function_error_raises(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        # Lambda execution itself failed; boto3 surfaces FunctionError.
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = _mock_invoke_response(
            200,
            {"errorMessage": "KeyError: 'agent_id'"},
            function_error="Unhandled",
        )

        with pytest.raises(AgentConfigLoadError, match="FunctionError"):
            await load_agent_config("chris-claim-status")

    async def test_missing_lambda_name_env_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("VOICE_API_LAMBDA_NAME", raising=False)

        with pytest.raises(AgentConfigLoadError, match="VOICE_API_LAMBDA_NAME"):
            await load_agent_config("chris-claim-status")

    async def test_boto_client_error_raises_wrapping_original(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        original = ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Lambda not found",
                }
            },
            "Invoke",
        )
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.side_effect = original

        with pytest.raises(AgentConfigLoadError) as exc_info:
            await load_agent_config("chris-claim-status")
        assert exc_info.value.__cause__ is original

    async def test_lambda_invoke_timeout_raises(
        self,
        mocker,
        voice_api_lambda_env: str,
    ):
        # Lambda read-timeout (8 s on the module-level client) surfaces
        # as a ReadTimeoutError. It's a BotoCoreError subclass, so the
        # loader's catch-all wraps it in AgentConfigLoadError just like
        # a ClientError. Verifies the wrapping plus __cause__ chain so
        # operators can read the original timeout in tracebacks.
        timeout = ReadTimeoutError(
            endpoint_url="https://lambda.us-east-1.amazonaws.com/2015-03-31/functions/test"
        )
        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.side_effect = timeout

        with pytest.raises(AgentConfigLoadError, match="Lambda invoke failed") as exc_info:
            await load_agent_config("chris-claim-status")
        assert exc_info.value.__cause__ is timeout

    async def test_settings_object_overrides_env_for_lambda_name(
        self,
        mocker,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Verifies the Layer 2 integration: when a Settings instance is
        # passed, voice_api_lambda_name comes from settings, not env.
        # Strongest form: build Settings, then change env, prove the
        # captured Settings value (not the changed env value) is used.
        monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "from-settings")
        monkeypatch.setenv(
            "API_KEY_SECRET_ARN",
            "arn:aws:secretsmanager:us-east-1:123:secret:test",
        )
        settings = Settings(_env_file=None)

        # Flip env after Settings is constructed. If the loader read
        # env directly we'd see "from-env" below.
        monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "from-env-shouldnt-be-used")

        mock_invoke = mocker.patch("app.config.agent_config._invoke_lambda_sync")
        mock_invoke.return_value = _mock_invoke_response(200, _runtime_config_json())

        await load_agent_config("chris-claim-status", settings=settings)

        # Third positional arg is the Settings object, which the
        # lazy-init lambda client uses to bind region.
        function_name, _payload, settings_arg = mock_invoke.call_args.args
        assert function_name == "from-settings"
        assert settings_arg is settings


class TestLambdaClientLazyInit:
    """Closes Entry 4: ``_LAMBDA_CLIENT`` now binds region from ``settings.aws_region``."""

    def test_lazy_init_uses_settings_region(self, monkeypatch):
        """First call to ``_get_lambda_client(settings)`` must construct
        the boto3 client with ``settings.aws_region`` — even when the
        env var disagrees."""
        from app.config import agent_config as agent_config_module

        # Reset the cache and set a misleading env var. If the lazy-init
        # fell back to env reads we'd see "us-east-1" below.
        agent_config_module._LAMBDA_CLIENT = None
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        settings = Settings(
            voice_api_lambda_name="test-lambda",
            api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
            aws_region="eu-central-1",
        )

        captured = {}

        def fake_client(service: str, region_name: str, config):
            captured["service"] = service
            captured["region"] = region_name
            return MagicMock()

        fake_session = MagicMock()
        fake_session.client = MagicMock(side_effect=fake_client)
        original_session = agent_config_module.boto3.session.Session
        try:
            agent_config_module.boto3.session.Session = MagicMock(return_value=fake_session)
            agent_config_module._get_lambda_client(settings)
        finally:
            agent_config_module.boto3.session.Session = original_session
            agent_config_module._LAMBDA_CLIENT = None

        assert captured["service"] == "lambda"
        assert captured["region"] == "eu-central-1"

    def test_lazy_init_falls_back_to_env_when_no_settings(self, monkeypatch):
        """Pre-Layer-9 callers without a Settings object still get a
        working client via the env-var fallback (tech debt Entry 3)."""
        from app.config import agent_config as agent_config_module

        agent_config_module._LAMBDA_CLIENT = None
        monkeypatch.setenv("AWS_REGION", "ap-south-1")

        captured = {}

        def fake_client(service: str, region_name: str, config):
            captured["region"] = region_name
            return MagicMock()

        fake_session = MagicMock()
        fake_session.client = MagicMock(side_effect=fake_client)
        original_session = agent_config_module.boto3.session.Session
        try:
            agent_config_module.boto3.session.Session = MagicMock(return_value=fake_session)
            agent_config_module._get_lambda_client(None)
        finally:
            agent_config_module.boto3.session.Session = original_session
            agent_config_module._LAMBDA_CLIENT = None

        assert captured["region"] == "ap-south-1"

    def test_lazy_init_caches_client_across_calls(self):
        """After the first call, subsequent calls return the same instance."""
        from app.config import agent_config as agent_config_module

        agent_config_module._LAMBDA_CLIENT = None

        construct_count = 0

        def fake_client(service: str, region_name: str, config):
            nonlocal construct_count
            construct_count += 1
            return MagicMock(name=f"client-{construct_count}")

        fake_session = MagicMock()
        fake_session.client = MagicMock(side_effect=fake_client)
        original_session = agent_config_module.boto3.session.Session

        settings = Settings(
            voice_api_lambda_name="t",
            api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:t",
            aws_region="us-east-1",
        )

        try:
            agent_config_module.boto3.session.Session = MagicMock(return_value=fake_session)
            client_a = agent_config_module._get_lambda_client(settings)
            client_b = agent_config_module._get_lambda_client(settings)
        finally:
            agent_config_module.boto3.session.Session = original_session
            agent_config_module._LAMBDA_CLIENT = None

        assert client_a is client_b
        assert construct_count == 1
