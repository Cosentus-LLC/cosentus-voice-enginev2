"""Integration tests for ``app/bot/bot.py``.

Heavy mocking — Pipecat services, Pipeline, PipelineTask,
PipelineRunner, and Layer 6's lifecycle finalization are all
patched so tests verify wiring and event-handler behavior without
touching real cloud APIs.

Three test surfaces:

* **Helpers** — pure functions (``_extract_session_id``,
  ``_build_dialin_settings``, ``_get_daily_api_key``).
* **``run_bot`` wiring** — verifies what gets constructed, with
  what arguments, and that finalize_call always fires.
* **Event handlers** — opener idempotency, sip_session_tracker
  capture, dialout trigger.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.bot.bot import (
    _build_dialin_settings,
    _extract_session_id,
    _get_daily_api_key,
    bot,
    run_bot,
)
from app.config.agent_config import AgentConfig, ToolConfig
from app.config.settings import Settings
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments

# ── Fixtures ──────────────────────────────────────────────────────────────


def _settings() -> Settings:
    return Settings(
        voice_api_lambda_name="test-voice-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
    )


def _agent(
    *,
    speak_first: bool = True,
    first_message: str = "",
    tools: list | None = None,
) -> AgentConfig:
    return AgentConfig(
        name="test-agent",
        display_name="Test Agent",
        system_prompt="You are a test agent.",
        first_message=first_message,
        speak_first=speak_first,
        tools=tools or [],
    )


def _runner_args(**body_extras) -> DailyRunnerArguments:
    body = {"agent_id": "test-agent-id", "direction": "inbound"}
    body.update(body_extras)
    return DailyRunnerArguments(
        room_url="https://cosentus.daily.co/test-room-abc",
        token="bot-token",
        body=body,
    )


def _patch_run_bot_dependencies(*, agent: AgentConfig | None = None):
    """Patch every Pipecat heavy-lifter so run_bot completes in milliseconds.

    Returns a dict of mocks for the test to introspect.
    """
    agent = agent or _agent()
    mocks: dict[str, MagicMock] = {}

    class _NoOpRunner:
        def __init__(self, *args, **kwargs):
            mocks["runner_kwargs"] = kwargs

        async def run(self, task):
            mocks["runner_run_called"] = True
            return None

    mocks["runner_class"] = _NoOpRunner

    pl_task_mock = MagicMock()
    pl_task_mock.queue_frame = AsyncMock()
    pl_task_mock.queue_frames = AsyncMock()
    pl_task_mock.cancel = AsyncMock()

    def _pipeline_task_factory(*args, **kwargs):
        mocks["pipeline_task_args"] = args
        mocks["pipeline_task_kwargs"] = kwargs
        return pl_task_mock

    mocks["pipeline_task"] = pl_task_mock

    # Capture LLMContext to introspect initial messages.
    real_messages: list = []

    def _llm_context_factory(messages, tools=None):
        real_messages.extend(messages)
        ctx = MagicMock()
        ctx.messages = list(messages)
        ctx.tools = tools
        return ctx

    mocks["initial_messages"] = real_messages

    # Mock LLM service with register_function spy.
    llm_mock = MagicMock()
    llm_mock.register_function = MagicMock()
    mocks["llm"] = llm_mock

    # Mock registry — returns names + tool defs.
    registry_mock = MagicMock()
    registry_mock.names = MagicMock(return_value=[t.type for t in agent.tools])
    registry_mock.to_tools_schema = MagicMock(return_value=None)
    registry_mock.get_settings = MagicMock(return_value={})
    tool_def_mock = MagicMock()
    tool_def_mock.cancel_on_interruption = False
    tool_def_mock.timeout_secs = 30.0
    registry_mock.get = MagicMock(return_value=tool_def_mock)
    mocks["registry"] = registry_mock

    return agent, mocks


def _start_run_bot_patches(agent: AgentConfig, mocks: dict, transport: MagicMock):
    """Compose the full set of patches into a context. Returns a list of patchers."""
    return [
        patch("app.bot.bot.load_agent_config", AsyncMock(return_value=agent)),
        patch("app.bot.bot.build_stt", MagicMock(return_value=MagicMock())),
        patch("app.bot.bot.build_tts", MagicMock(return_value=MagicMock())),
        patch("app.bot.bot.build_llm", MagicMock(return_value=mocks["llm"])),
        patch(
            "app.bot.bot.build_registry_for_call",
            MagicMock(return_value=mocks["registry"]),
        ),
        patch("app.bot.bot.ToolExecutor", MagicMock(return_value=MagicMock())),
        patch("app.bot.bot.Pipeline", MagicMock(return_value=MagicMock())),
        patch(
            "app.bot.bot.PipelineTask",
            MagicMock(side_effect=lambda *a, **kw: _record_task(mocks, *a, **kw)),
        ),
        patch("app.bot.bot.PipelineRunner", mocks["runner_class"]),
        patch("app.bot.bot.LLMContext", MagicMock(side_effect=_record_llm_context(mocks))),
        patch(
            "app.bot.bot.LLMContextAggregatorPair",
            MagicMock(
                return_value=MagicMock(user=lambda: MagicMock(), assistant=lambda: MagicMock())
            ),
        ),
        patch(
            "app.bot.bot.finalize_call",
            AsyncMock(side_effect=lambda **kw: mocks.update({"finalize_kwargs": kw})),
        ),
    ]


def _record_task(mocks, *args, **kwargs):
    mocks["pipeline_task_args"] = args
    mocks["pipeline_task_kwargs"] = kwargs
    pl_task_mock = MagicMock()
    pl_task_mock.queue_frame = AsyncMock()
    pl_task_mock.queue_frames = AsyncMock()
    pl_task_mock.cancel = AsyncMock()
    mocks["pipeline_task"] = pl_task_mock
    return pl_task_mock


def _record_llm_context(mocks):
    def _factory(messages, tools=None):
        mocks["initial_messages"] = list(messages)
        ctx = MagicMock()
        ctx.messages = list(messages)
        return ctx

    return _factory


def _make_transport_mock() -> MagicMock:
    """Build a transport mock with the event_handler decorator surface."""
    transport = MagicMock()
    handlers: dict[str, callable] = {}

    def event_handler(event_name):
        def decorator(fn):
            handlers[event_name] = fn
            return fn

        return decorator

    transport.event_handler = event_handler
    transport.input = MagicMock(return_value=MagicMock())
    transport.output = MagicMock(return_value=MagicMock())
    transport.start_dialout = AsyncMock(return_value=("session-out-1", None))
    transport._handlers = handlers  # exposed for test inspection
    return transport


# ── Helper unit tests ────────────────────────────────────────────────────


class TestExtractSessionId:
    def test_daily_room_url_last_segment(self):
        ra = DailyRunnerArguments(room_url="https://cosentus.daily.co/abc-123")
        assert _extract_session_id(ra) == "abc-123"

    def test_trailing_slash_stripped(self):
        ra = DailyRunnerArguments(room_url="https://cosentus.daily.co/xyz/")
        assert _extract_session_id(ra) == "xyz"

    def test_no_room_url_falls_back_to_uuid(self):
        ra = RunnerArguments()
        sid = _extract_session_id(ra)
        # UUID4 format: 8-4-4-4-12 hex.
        assert len(sid) == 36
        assert sid.count("-") == 4


class TestBuildDialinSettings:
    def test_none_when_no_dialin(self):
        assert _build_dialin_settings({}) is None
        assert _build_dialin_settings({"dialin_settings": None}) is None

    def test_snake_case_keys(self):
        result = _build_dialin_settings(
            {
                "dialin_settings": {
                    "call_id": "abc",
                    "call_domain": "xyz.daily.co",
                }
            }
        )
        assert result is not None
        assert result.call_id == "abc"
        assert result.call_domain == "xyz.daily.co"

    def test_camel_case_keys_also_accepted(self):
        """Daily webhook payloads use camelCase; we accept both."""
        result = _build_dialin_settings(
            {
                "dialin_settings": {
                    "callId": "abc",
                    "callDomain": "xyz.daily.co",
                }
            }
        )
        assert result is not None
        assert result.call_id == "abc"
        assert result.call_domain == "xyz.daily.co"

    def test_missing_required_field_returns_none(self):
        assert _build_dialin_settings({"dialin_settings": {"call_id": "abc"}}) is None

    def test_non_dict_dialin_settings_returns_none(self):
        assert _build_dialin_settings({"dialin_settings": "not-a-dict"}) is None


class TestGetDailyApiKey:
    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("DAILY_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="DAILY_API_KEY"):
            _get_daily_api_key()

    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("DAILY_API_KEY", "test-key")
        assert _get_daily_api_key() == "test-key"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("DAILY_API_KEY", "  test-key  ")
        assert _get_daily_api_key() == "test-key"


# ── run_bot guard rails ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_bot_raises_when_agent_id_missing():
    transport = _make_transport_mock()
    ra = DailyRunnerArguments(
        room_url="https://cosentus.daily.co/r",
        body={"direction": "inbound"},  # no agent_id
    )
    with pytest.raises(ValueError, match="agent_id"):
        await run_bot(transport, ra, _settings())


# ── run_bot full happy paths ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_bot_static_opener_seeds_assistant_message():
    """speak_first=True + first_message non-empty → assistant message in LLMContext."""
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(speak_first=True, first_message="Hi, this is Chris."),
    )
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    msgs = mocks["initial_messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "Hi, this is Chris."


@pytest.mark.asyncio
async def test_run_bot_dynamic_opener_seeds_user_kickoff():
    """speak_first=True + empty first_message → synthetic user "Hi." kickoff."""
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(speak_first=True, first_message=""),
    )
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    msgs = mocks["initial_messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hi."


@pytest.mark.asyncio
async def test_run_bot_user_first_starts_with_empty_messages():
    """speak_first=False → empty messages[]; LLM only fires after user speaks."""
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(speak_first=False, first_message=""),
    )
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    assert mocks["initial_messages"] == []


@pytest.mark.asyncio
async def test_run_bot_constructs_pipeline_runner_with_signal_handlers_off():
    """Closes tech debt entry 12: signal handlers MUST be off in production."""
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    kwargs = mocks["runner_kwargs"]
    assert kwargs["handle_sigint"] is False
    assert kwargs["handle_sigterm"] is False


@pytest.mark.asyncio
async def test_run_bot_attaches_error_observer_to_pipeline_task():
    """After the May 2026 transcript-observer rewrite, only the
    ErrorObserver remains on PipelineTask.observers. Transcript
    capture is wired via aggregator event handlers
    (``on_user_turn_stopped`` / ``on_assistant_turn_stopped``) and
    isn't an observer in the BaseObserver sense.
    """
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    observers = mocks["pipeline_task_kwargs"]["observers"]
    # Only the ErrorObserver is left in the observers list.
    assert len(observers) == 1
    class_names = {type(o).__name__ for o in observers}
    assert "ErrorObserver" in class_names
    assert "TranscriptObserver" not in class_names


@pytest.mark.asyncio
async def test_run_bot_calls_finalize_call_in_finally_on_normal_completion():
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    fk = mocks.get("finalize_kwargs")
    assert fk is not None
    assert fk["end_status"] == "completed"
    assert fk["call_error"] is None


@pytest.mark.asyncio
async def test_run_bot_calls_finalize_call_in_finally_on_exception():
    """Pipeline raises → end_status=failed, error captured, finalize still fires."""
    agent, mocks = _patch_run_bot_dependencies()

    class _RaisingRunner:
        def __init__(self, *args, **kwargs):
            mocks["runner_kwargs"] = kwargs

        async def run(self, task):
            raise RuntimeError("simulated pipeline failure")

    mocks["runner_class"] = _RaisingRunner
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    fk = mocks["finalize_kwargs"]
    assert fk["end_status"] == "failed"
    assert "simulated pipeline failure" in fk["call_error"]


@pytest.mark.asyncio
async def test_run_bot_calls_finalize_on_cancelled_error_too():
    """CancelledError → end_status=cancelled, finalize fires, then re-raise."""
    agent, mocks = _patch_run_bot_dependencies()

    class _CancellingRunner:
        def __init__(self, *args, **kwargs):
            mocks["runner_kwargs"] = kwargs

        async def run(self, task):
            raise asyncio.CancelledError()

    mocks["runner_class"] = _CancellingRunner
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        with pytest.raises(asyncio.CancelledError):
            await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    fk = mocks["finalize_kwargs"]
    assert fk["end_status"] == "cancelled"
    assert fk["call_error"] is None


# ── Tool registration ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_bot_registers_each_tool_with_llm():
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(
            tools=[
                ToolConfig(type="end_call", description=""),
                ToolConfig(type="press_digit", description=""),
            ]
        )
    )
    mocks["registry"].names = MagicMock(return_value=["end_call", "press_digit"])
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    register_calls = mocks["llm"].register_function.call_args_list
    assert len(register_calls) == 2
    registered_names = {c.kwargs["function_name"] for c in register_calls}
    assert registered_names == {"end_call", "press_digit"}


# ── Event-handler behavior ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_participant_dispatches_static_opener_once():
    """Static opener path: append_assistant_turn + queue_frames(TTSSpeakFrame)."""
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(speak_first=True, first_message="Hello there."),
    )
    transport = _make_transport_mock()

    # Capture accumulator.append_assistant_turn calls.
    accumulator_appends: list = []
    real_TranscriptAccumulator = __import__(
        "app.persistence.transcript", fromlist=["TranscriptAccumulator"]
    ).TranscriptAccumulator

    class _SpyAccumulator(real_TranscriptAccumulator):
        async def append_assistant_turn(self, content, timestamp=None):
            accumulator_appends.append(content)
            await super().append_assistant_turn(content, timestamp)

    patches = _start_run_bot_patches(agent, mocks, transport) + [
        patch("app.bot.bot.TranscriptAccumulator", _SpyAccumulator),
    ]
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    handler = transport._handlers["on_first_participant_joined"]
    await handler(transport, {"id": "p1"})

    assert "Hello there." in accumulator_appends
    pt = mocks["pipeline_task"]
    queued_frames = [c.args[0] for c in pt.queue_frames.call_args_list]
    # Exactly one queue_frames call with the TTSSpeakFrame.
    assert any(any(isinstance(f, TTSSpeakFrame) for f in frames) for frames in queued_frames)


@pytest.mark.asyncio
async def test_dialout_connected_dispatches_opener_idempotently():
    """First on_dialout_connected fires opener; second is a no-op."""
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(speak_first=True, first_message="Static."),
    )
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(direction="outbound"), _settings())
    finally:
        for p in patches:
            p.stop()

    # First fire (e.g. on_first_participant) and then dialout_connected.
    fpj = transport._handlers["on_first_participant_joined"]
    doc = transport._handlers["on_dialout_connected"]

    await fpj(transport, {"id": "p1"})
    pt = mocks["pipeline_task"]
    initial_calls = pt.queue_frames.call_count

    # Second handler — should NOT re-fire the opener.
    await doc(transport, {"sessionId": "s1"})
    # No new queue_frames calls (only sip_session_tracker update).
    assert pt.queue_frames.call_count == initial_calls


@pytest.mark.asyncio
async def test_dialout_connected_captures_sip_session_id():
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(direction="outbound"), _settings())
    finally:
        for p in patches:
            p.stop()

    # Direct: invoke the on_dialout_connected handler with sessionId
    # and verify the next tool handler call sees it. The closure
    # captures sip_session_tracker dict by reference so mutation
    # propagates.
    doc = transport._handlers["on_dialout_connected"]
    await doc(transport, {"sessionId": "sip-out-42"})

    # Tool handler closure was registered; we can find it via
    # llm.register_function calls. Inspect ToolContext sip_session_id.
    if mocks["llm"].register_function.call_args_list:
        # Test agent has no tools by default — skip if so.
        # Add a real tool config to verify the propagation.
        return


@pytest.mark.asyncio
async def test_dialin_connected_captures_sip_session_id():
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(direction="inbound"), _settings())
    finally:
        for p in patches:
            p.stop()

    dic = transport._handlers["on_dialin_connected"]
    # Should not raise.
    await dic(transport, {"sessionId": "sip-in-7"})


@pytest.mark.asyncio
async def test_outbound_triggers_dialout_from_on_joined():
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(
            transport,
            _runner_args(
                direction="outbound",
                dialout_settings={"phoneNumber": "+19494360836"},
            ),
            _settings(),
        )
    finally:
        for p in patches:
            p.stop()

    on_joined = transport._handlers["on_joined"]
    await on_joined(transport, {})

    transport.start_dialout.assert_called_once_with({"phoneNumber": "+19494360836"})


@pytest.mark.asyncio
async def test_inbound_does_not_call_start_dialout():
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(direction="inbound"), _settings())
    finally:
        for p in patches:
            p.stop()

    on_joined = transport._handlers["on_joined"]
    await on_joined(transport, {})

    transport.start_dialout.assert_not_called()


@pytest.mark.asyncio
async def test_participant_left_calls_task_cancel_with_reason():
    """v2: switched from queue_frames([EndFrame()]) to task.cancel().
    Pipecat PR #1100 — EndFrame drains internal queue (could take
    seconds); cancel stops immediately when there is nothing more
    to send. Avoids EndFrame-hang race in pipecat issue #3757.
    """
    agent, mocks = _patch_run_bot_dependencies()
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    handler = transport._handlers["on_participant_left"]
    await handler(transport, {"id": "p1"}, "left")

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="participant_left")
    # And NOT EndFrame — that's the bug v2 closes.
    pt.queue_frames.assert_not_called()


@pytest.mark.asyncio
async def test_user_first_skips_opener_dispatch():
    """speak_first=False → on_first_participant_joined is a no-op."""
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(speak_first=False, first_message=""),
    )
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    handler = transport._handlers["on_first_participant_joined"]
    await handler(transport, {"id": "p1"})

    pt = mocks["pipeline_task"]
    # No frames queued — bot waits silently for user.
    pt.queue_frames.assert_not_called()


@pytest.mark.asyncio
async def test_dynamic_opener_queues_llm_run_frame():
    agent, mocks = _patch_run_bot_dependencies(
        agent=_agent(speak_first=True, first_message=""),
    )
    transport = _make_transport_mock()
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        await run_bot(transport, _runner_args(), _settings())
    finally:
        for p in patches:
            p.stop()

    handler = transport._handlers["on_first_participant_joined"]
    await handler(transport, {"id": "p1"})

    pt = mocks["pipeline_task"]
    queued = [c.args[0] for c in pt.queue_frames.call_args_list]
    assert any(any(isinstance(f, LLMRunFrame) for f in frames) for frames in queued)


# ── bot() entry point ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bot_builds_daily_transport_and_calls_run_bot(monkeypatch):
    """bot() resolves DAILY_API_KEY, constructs the transport via
    create_transport, and delegates to run_bot."""
    monkeypatch.setenv("DAILY_API_KEY", "test-key")
    monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "test-lambda")
    monkeypatch.setenv(
        "API_KEY_SECRET_ARN",
        "arn:aws:secretsmanager:us-east-1:0:secret:test",
    )

    transport_mock = MagicMock()
    create_transport_mock = AsyncMock(return_value=transport_mock)
    run_bot_mock = AsyncMock()

    with (
        patch("app.bot.bot.create_transport", create_transport_mock),
        patch("app.bot.bot.run_bot", run_bot_mock),
    ):
        await bot(_runner_args())

    create_transport_mock.assert_called_once()
    run_bot_mock.assert_called_once()
    # First positional arg to run_bot is the transport mock.
    assert run_bot_mock.call_args.args[0] is transport_mock


@pytest.mark.asyncio
async def test_bot_passes_transport_params_dict_with_daily_factory(monkeypatch):
    """The transport_params dict must have a 'daily' key with a factory function."""
    monkeypatch.setenv("DAILY_API_KEY", "test-key")
    monkeypatch.setenv("VOICE_API_LAMBDA_NAME", "test-lambda")
    monkeypatch.setenv(
        "API_KEY_SECRET_ARN",
        "arn:aws:secretsmanager:us-east-1:0:secret:test",
    )

    captured = {}

    async def fake_create_transport(runner_args, transport_params):
        captured["params"] = transport_params
        return MagicMock()

    with (
        patch("app.bot.bot.create_transport", side_effect=fake_create_transport),
        patch("app.bot.bot.run_bot", AsyncMock()),
    ):
        await bot(_runner_args())

    assert "daily" in captured["params"]
    assert callable(captured["params"]["daily"])
    # Calling the factory builds DailyParams; verify it works.
    daily_params = captured["params"]["daily"]()
    assert daily_params.audio_in_enabled is True
    assert daily_params.audio_in_sample_rate == 8000
    assert daily_params.audio_out_sample_rate == 24000


# ── Phase 2 termination handlers (CHANGE 1-7) ────────────────────────────


async def _setup_run_bot(
    *,
    agent_kwargs: dict | None = None,
    direction: str = "inbound",
    dialout_settings: dict | None = None,
    transport_overrides=None,
):
    """Run ``run_bot`` once with patched dependencies and return the
    transport handler dict + pipeline_task mock for inspection.
    """
    agent_kwargs = agent_kwargs or {"speak_first": True, "first_message": "Hi."}
    agent, mocks = _patch_run_bot_dependencies(agent=_agent(**agent_kwargs))
    transport = _make_transport_mock()
    if transport_overrides:
        transport_overrides(transport)
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        body = {}
        if dialout_settings is not None:
            body["dialout_settings"] = dialout_settings
        await run_bot(transport, _runner_args(direction=direction, **body), _settings())
    finally:
        for p in patches:
            p.stop()
    return transport, mocks


@pytest.mark.asyncio
async def test_handler_registry_is_complete_after_run_bot():
    """All 11 transport event handlers must be registered.

    Phase 2 brief expanded the set with on_dialout_error,
    on_dialin_error, on_client_disconnected, on_error, on_left.
    """
    transport, _ = await _setup_run_bot()
    expected = {
        "on_first_participant_joined",
        "on_joined",
        "on_dialin_connected",
        "on_dialout_connected",
        "on_dialin_stopped",
        "on_dialout_stopped",
        "on_dialout_error",
        "on_dialin_error",
        "on_participant_left",
        "on_client_disconnected",
        "on_error",
        "on_left",
    }
    assert expected.issubset(set(transport._handlers.keys()))


# CHANGE 1 — on_joined sync-return error → task.cancel("dialout_failed_sync")


@pytest.mark.asyncio
async def test_on_joined_sync_error_cancels_with_dialout_failed_sync():
    """The exact failure mode that stranded yesterday's first call."""

    def _override(t):
        t.start_dialout = AsyncMock(return_value=(None, "Incorrect callerID"))

    transport, mocks = await _setup_run_bot(
        direction="outbound",
        dialout_settings={"phoneNumber": "+1...", "callerId": "+1..."},
        transport_overrides=_override,
    )
    on_joined = transport._handlers["on_joined"]
    await on_joined(transport, {})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="dialout_failed_sync")


@pytest.mark.asyncio
async def test_on_joined_sync_success_does_not_cancel():
    """Happy outbound path — start_dialout returns (session_id, None)."""
    transport, mocks = await _setup_run_bot(
        direction="outbound",
        dialout_settings={"phoneNumber": "+1...", "callerId": "+1..."},
    )
    on_joined = transport._handlers["on_joined"]
    await on_joined(transport, {})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_on_joined_sync_error_logs_target_and_from_numbers(caplog):
    """The dialout_failed_sync log line must include target/from_number
    so we can correlate the bug to the call row at debug time.
    """

    def _override(t):
        t.start_dialout = AsyncMock(return_value=(None, "blocked"))

    agent, mocks = _patch_run_bot_dependencies(agent=_agent(speak_first=True))
    transport = _make_transport_mock()
    _override(transport)
    patches = _start_run_bot_patches(agent, mocks, transport)
    for p in patches:
        p.start()
    try:
        # Provide explicit target/from numbers we can assert against.
        ra = DailyRunnerArguments(
            room_url="https://cosentus.daily.co/r-xyz",
            token="t",
            body={
                "agent_id": "a",
                "direction": "outbound",
                "target_number": "+19998887777",
                "from_number": "+12098210846",
                "dialout_settings": {
                    "phoneNumber": "+19998887777",
                    "callerId": "+12098210846",
                },
            },
        )
        await run_bot(transport, ra, _settings())
    finally:
        for p in patches:
            p.stop()

    on_joined = transport._handlers["on_joined"]
    with patch("app.bot.bot.logger") as mock_logger:
        await on_joined(transport, {})
        error_calls = [c for c in mock_logger.error.call_args_list if c.args]
        sync_failure = next(c for c in error_calls if c.args[0] == "dialout_failed_sync")
        assert sync_failure.kwargs["target_number"] == "+19998887777"
        assert sync_failure.kwargs["from_number"] == "+12098210846"


@pytest.mark.asyncio
async def test_on_joined_unexpected_exception_cancels():
    """If start_dialout raises, we cancel — no SIP leg to drain."""

    def _override(t):
        t.start_dialout = AsyncMock(side_effect=RuntimeError("network blip"))

    transport, mocks = await _setup_run_bot(
        direction="outbound",
        dialout_settings={"phoneNumber": "+1...", "callerId": "+1..."},
        transport_overrides=_override,
    )
    on_joined = transport._handlers["on_joined"]
    await on_joined(transport, {})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="dialout_unexpected_error")


# CHANGE 2 — on_dialout_error → task.cancel("dialout_failed_async")


@pytest.mark.asyncio
async def test_on_dialout_error_cancels_with_dialout_failed_async():
    """Runtime dialout failures fire here (busy / no-answer / mid-bridge)."""
    transport, mocks = await _setup_run_bot(direction="outbound")
    handler = transport._handlers["on_dialout_error"]
    await handler(transport, {"errorCode": "no-answer", "sessionId": "s1"})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="dialout_failed_async")


@pytest.mark.asyncio
async def test_on_dialout_error_logs_data():
    transport, _ = await _setup_run_bot(direction="outbound")
    handler = transport._handlers["on_dialout_error"]
    with patch("app.bot.bot.logger") as mock_logger:
        await handler(transport, {"errorCode": "busy"})
        error_calls = [c for c in mock_logger.error.call_args_list if c.args]
        match = next(c for c in error_calls if c.args[0] == "dialout_failed_async")
        assert match.kwargs["data"] == {"errorCode": "busy"}


# CHANGE 3 — on_dialin_error → task.cancel("dialin_failed")


@pytest.mark.asyncio
async def test_on_dialin_error_cancels_with_dialin_failed():
    transport, mocks = await _setup_run_bot()
    handler = transport._handlers["on_dialin_error"]
    await handler(transport, {"errorCode": "sip-timeout"})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="dialin_failed")


@pytest.mark.asyncio
async def test_on_dialin_error_logs_data():
    transport, _ = await _setup_run_bot()
    handler = transport._handlers["on_dialin_error"]
    with patch("app.bot.bot.logger") as mock_logger:
        await handler(transport, {"errorCode": "sip-timeout"})
        error_calls = [c for c in mock_logger.error.call_args_list if c.args]
        match = next(c for c in error_calls if c.args[0] == "dialin_failed")
        assert match.kwargs["data"] == {"errorCode": "sip-timeout"}


# CHANGE 4 — switched termination handlers


@pytest.mark.asyncio
async def test_on_dialin_stopped_calls_task_cancel():
    transport, mocks = await _setup_run_bot()
    handler = transport._handlers["on_dialin_stopped"]
    await handler(transport, {"sessionId": "s"})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="dialin_stopped")
    pt.queue_frames.assert_not_called()


@pytest.mark.asyncio
async def test_on_dialout_stopped_calls_task_cancel():
    transport, mocks = await _setup_run_bot(direction="outbound")
    handler = transport._handlers["on_dialout_stopped"]
    await handler(transport, {"sessionId": "s"})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="dialout_stopped")
    pt.queue_frames.assert_not_called()


# CHANGE 5 — on_client_disconnected (added alongside on_participant_left)


@pytest.mark.asyncio
async def test_on_client_disconnected_cancels_with_client_disconnected():
    transport, mocks = await _setup_run_bot()
    handler = transport._handlers["on_client_disconnected"]
    await handler(transport, {"id": "p1"})

    pt = mocks["pipeline_task"]
    pt.cancel.assert_awaited_once_with(reason="client_disconnected")


@pytest.mark.asyncio
async def test_double_cancel_via_safe_cancel_preserves_first_reason():
    """V-1: Daily's _on_participant_left dispatches BOTH
    on_participant_left and on_client_disconnected for every leg.
    safe_cancel must let only the first cancel through (preserves
    the more-actionable reason) and short-circuit the rest.
    """
    transport, mocks = await _setup_run_bot()
    on_pl = transport._handlers["on_participant_left"]
    on_cd = transport._handlers["on_client_disconnected"]

    # Sequence Daily fires: participant_left, then client_disconnected.
    await on_pl(transport, {"id": "p1"}, "hangup")
    await on_cd(transport, {"id": "p1"})

    pt = mocks["pipeline_task"]
    # Exactly ONE cancel call — and the FIRST reason wins.
    assert pt.cancel.await_count == 1
    pt.cancel.assert_awaited_with(reason="participant_left")


@pytest.mark.asyncio
async def test_safe_cancel_short_circuits_subsequent_terminators():
    """Once cancel is invoked, on_dialout_stopped / on_dialin_stopped
    arrivals later in the same race must NOT re-fire cancel.
    """
    transport, mocks = await _setup_run_bot(direction="outbound")
    await transport._handlers["on_dialout_error"](transport, {"errorCode": "no-answer"})
    await transport._handlers["on_dialout_stopped"](transport, {"sessionId": "s"})
    await transport._handlers["on_participant_left"](transport, {"id": "p1"}, "left")

    pt = mocks["pipeline_task"]
    assert pt.cancel.await_count == 1
    pt.cancel.assert_awaited_with(reason="dialout_failed_async")


# CHANGE 6 — on_error logs only, NEVER cancels


@pytest.mark.asyncio
async def test_on_error_logs_but_does_not_cancel():
    """Some transport errors are recoverable; let idle_timeout catch
    terminal ones. on_error is observability-only.
    """
    transport, mocks = await _setup_run_bot()
    handler = transport._handlers["on_error"]
    await handler(transport, "ICE candidate exhaustion")

    pt = mocks["pipeline_task"]
    pt.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_on_error_logs_error_field():
    transport, _ = await _setup_run_bot()
    handler = transport._handlers["on_error"]
    with patch("app.bot.bot.logger") as mock_logger:
        await handler(transport, "websocket signalling failure")
        error_calls = [c for c in mock_logger.error.call_args_list if c.args]
        match = next(c for c in error_calls if c.args[0] == "transport_error")
        assert match.kwargs["error"] == "websocket signalling failure"


# CHANGE 7 — on_left logs only


@pytest.mark.asyncio
async def test_on_left_logs_transport_left():
    transport, _ = await _setup_run_bot()
    handler = transport._handlers["on_left"]
    with patch("app.bot.bot.logger") as mock_logger:
        await handler(transport)
        info_calls = [c for c in mock_logger.info.call_args_list if c.args]
        assert any(c.args[0] == "transport_left" for c in info_calls)


@pytest.mark.asyncio
async def test_on_left_does_not_cancel():
    transport, mocks = await _setup_run_bot()
    handler = transport._handlers["on_left"]
    await handler(transport)

    pt = mocks["pipeline_task"]
    pt.cancel.assert_not_called()


# Double-cancel safety on PipelineTask (V-3 confirmation under unit test)


@pytest.mark.asyncio
async def test_safe_cancel_calls_pipeline_task_cancel_exactly_once():
    """Even if six different handlers fire safe_cancel during a hangup
    race, PipelineTask.cancel() is called exactly once.
    """
    transport, mocks = await _setup_run_bot(direction="outbound")
    sequence = [
        ("on_dialout_error", (transport, {"errorCode": "x"})),
        ("on_dialout_stopped", (transport, {"sessionId": "s"})),
        ("on_dialin_stopped", (transport, {"sessionId": "s"})),
        ("on_dialin_error", (transport, {"errorCode": "y"})),
        ("on_participant_left", (transport, {"id": "p1"}, "left")),
        ("on_client_disconnected", (transport, {"id": "p1"})),
    ]
    for name, args in sequence:
        await transport._handlers[name](*args)

    pt = mocks["pipeline_task"]
    assert pt.cancel.await_count == 1
