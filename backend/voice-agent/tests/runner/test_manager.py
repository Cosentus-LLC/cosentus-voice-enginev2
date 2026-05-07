"""Tests for ``app/runner/manager.py``.

Mocks the bot, Daily client, and TaskProtection so tests run in
milliseconds. Verifies:

* spawn lifecycle (room creation → task creation → dict insertion)
* dict-boundary lifecycle (0→1 acquire, 1→0 release)
* heartbeat coroutine starts on first call, stops when empty
* capacity gating + draining gating
* _wrapped_bot pops dict on success / exception / cancellation
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.config.settings import Settings
from app.runner.daily_rooms import DailyRoom
from app.runner.manager import (
    CallSpawnResult,
    CapacityRejected,
    PipelineManager,
)
from app.runner.protection import TaskProtection

# ── Fixtures ──────────────────────────────────────────────────────────────


def _settings(*, max_concurrent: int = 6) -> Settings:
    return Settings(
        voice_api_lambda_name="test-voice-api",
        api_key_secret_arn="arn:aws:secretsmanager:us-east-1:0:secret:test",
        max_concurrent_calls=max_concurrent,
    )


def _daily_mock(
    *,
    inbound_room: DailyRoom | None = None,
    outbound_room: DailyRoom | None = None,
    browser_room: DailyRoom | None = None,
    caller_id_uuid: str | None = "stub-uuid-for-tests",
) -> MagicMock:
    inbound_room = inbound_room or DailyRoom(
        url="https://x.daily.co/in", name="in", sip_uri="sip:in@x"
    )
    outbound_room = outbound_room or DailyRoom(url="https://x.daily.co/out", name="out")
    browser_room = browser_room or DailyRoom(url="https://x.daily.co/br", name="br")
    daily = MagicMock()
    daily.create_inbound_room = AsyncMock(return_value=inbound_room)
    daily.create_outbound_room = AsyncMock(return_value=outbound_room)
    daily.create_browser_room = AsyncMock(return_value=browser_room)
    daily.mint_token = AsyncMock(return_value="bot.token.jwt")
    # Default: every from_number resolves to a stub UUID so existing
    # outbound tests don't hit the unresolved-fallback path. Tests
    # that exercise the fallback override this explicitly.
    daily.get_phone_number_uuid = AsyncMock(return_value=caller_id_uuid)
    daily.close = AsyncMock()
    return daily


def _protection_mock() -> MagicMock:
    p = MagicMock(spec=TaskProtection)
    p.set_protected = AsyncMock(return_value=True)
    p.renew_if_protected = AsyncMock(return_value=True)
    p.close = AsyncMock()
    p.is_available = True
    p.is_protected = False
    return p


# ── Status accessors ─────────────────────────────────────────────────────


def test_get_status_initial():
    m = PipelineManager(_settings(), _daily_mock(), _protection_mock())
    status = m.get_status()
    assert status["active_sessions"] == 0
    assert status["max_concurrent"] == 6
    assert status["draining"] is False


def test_at_capacity_false_when_empty():
    m = PipelineManager(_settings(), _daily_mock(), _protection_mock())
    assert m.at_capacity is False


def test_at_capacity_true_when_full():
    m = PipelineManager(_settings(max_concurrent=2), _daily_mock(), _protection_mock())
    m._active_sessions["a"] = MagicMock()
    m._active_sessions["b"] = MagicMock()
    assert m.at_capacity is True


# ── _reject_if_unavailable ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_if_draining():
    m = PipelineManager(_settings(), _daily_mock(), _protection_mock())
    m._draining = True
    with pytest.raises(CapacityRejected) as exc_info:
        await m.start_browser(agent_id="x")
    assert exc_info.value.reason == "draining"


@pytest.mark.asyncio
async def test_reject_if_at_capacity():
    m = PipelineManager(_settings(max_concurrent=1), _daily_mock(), _protection_mock())
    m._active_sessions["existing"] = MagicMock()
    with pytest.raises(CapacityRejected) as exc_info:
        await m.start_browser(agent_id="x")
    assert exc_info.value.reason == "at_capacity"


# ── start_outbound ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_outbound_creates_room_and_spawns():
    daily = _daily_mock()
    protection = _protection_mock()
    m = PipelineManager(_settings(), daily, protection)

    fake_bot = AsyncMock()
    with patch("app.runner.manager.bot", fake_bot):
        result = await m.start_outbound(
            agent_id="agent-1",
            target_number="+19494360836",
            from_number="+12098075018",
            case_data={"k": "v"},
        )

        assert isinstance(result, CallSpawnResult)
        assert result.room_name == "out"
        assert result.room_url == "https://x.daily.co/out"
        daily.create_outbound_room.assert_awaited_once()
        daily.mint_token.assert_awaited_once_with("out")

        # Wait inside the patch context so the spawned task sees the
        # patched bot. The patch unwinds when this with-block exits.
        await asyncio.sleep(0.05)
        fake_bot.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_outbound_resolves_caller_id_e164_to_uuid():
    """Empirically verified 2026-05-07: Daily's ``dialOut/start``
    rejects ``callerId`` in E.164 form (``+15559999999``) with
    ``"Incorrect callerID! No phone number maps to: ..."`` and
    accepts the matching purchased-phone-number UUID. The manager
    resolves E.164 → UUID and passes the UUID to
    ``dialout_settings.callerId``; ``body.from_number`` keeps the
    E.164 for Aurora storage and human-readable logs.
    """
    daily = _daily_mock(caller_id_uuid="resolved-daily-uuid-abc")
    m = PipelineManager(_settings(), daily, _protection_mock())

    captured_args = []

    async def fake_bot(runner_args):
        captured_args.append(runner_args)

    with patch("app.runner.manager.bot", fake_bot):
        await m.start_outbound(
            agent_id="a",
            target_number="+15551234567",
            from_number="+15559999999",
            case_data={},
        )
        await asyncio.sleep(0.05)

    assert len(captured_args) == 1
    body = captured_args[0].body
    assert body["direction"] == "outbound"
    # E.164 — kept for Aurora storage + transcripts.
    assert body["from_number"] == "+15559999999"
    # UUID — what Daily expects in its dialOut/start callerId field.
    assert body["dialout_settings"]["phoneNumber"] == "+15551234567"
    assert body["dialout_settings"]["callerId"] == "resolved-daily-uuid-abc"
    # The resolver was actually consulted.
    daily.get_phone_number_uuid.assert_awaited_once_with("+15559999999")


@pytest.mark.asyncio
async def test_start_outbound_falls_back_to_e164_on_unresolved_caller_id():
    """If the resolver returns ``None`` (E.164 not in Daily's
    purchased-numbers list), the manager passes the E.164 through
    unchanged. Daily then returns ``"Incorrect callerID..."`` and
    Phase 2's ``dialout_failed_sync`` handler cancels the bot
    cleanly. Don't fail-fast in the manager — keep the existing
    termination + CallRecord path.
    """
    daily = _daily_mock(caller_id_uuid=None)
    m = PipelineManager(_settings(), daily, _protection_mock())

    captured_args = []

    async def fake_bot(runner_args):
        captured_args.append(runner_args)

    with patch("app.runner.manager.bot", fake_bot):
        await m.start_outbound(
            agent_id="a",
            target_number="+15551234567",
            from_number="+19998887777",
            case_data={},
        )
        await asyncio.sleep(0.05)

    assert len(captured_args) == 1
    body = captured_args[0].body
    # callerId falls back to the E.164 form when no UUID resolves.
    assert body["dialout_settings"]["callerId"] == "+19998887777"


# ── start_browser ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_browser_returns_viewer_token():
    daily = _daily_mock()
    # The mint_token mock returns "bot.token.jwt" for both calls;
    # the second call is the viewer token. Override to differentiate.
    daily.mint_token = AsyncMock(side_effect=["bot.jwt", "viewer.jwt"])
    m = PipelineManager(_settings(), daily, _protection_mock())

    with patch("app.runner.manager.bot", AsyncMock()):
        result = await m.start_browser(agent_id="agent-1")
        await asyncio.sleep(0.05)

    assert result.viewer_token == "viewer.jwt"
    daily.create_browser_room.assert_awaited_once()
    assert daily.mint_token.call_count == 2


@pytest.mark.asyncio
async def test_start_browser_sets_direction_browser():
    daily = _daily_mock()
    m = PipelineManager(_settings(), daily, _protection_mock())

    captured = []

    async def fake_bot(runner_args):
        captured.append(runner_args.body)

    with patch("app.runner.manager.bot", fake_bot):
        await m.start_browser(agent_id="x")
        await asyncio.sleep(0.05)

    assert captured[0]["direction"] == "browser"


# ── start_inbound ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_inbound_creates_sip_room_and_spawns():
    daily = _daily_mock()
    m = PipelineManager(_settings(), daily, _protection_mock())

    captured = []

    async def fake_bot(runner_args):
        captured.append(runner_args.body)

    with patch("app.runner.manager.bot", fake_bot):
        await m.start_inbound(
            agent_id="agent-1",
            from_number="+19494360836",
            to_number="+12098075018",
            call_id_external="ext-call-id",
            call_domain="cosentus.daily.co",
        )
        await asyncio.sleep(0.05)

    daily.create_inbound_room.assert_awaited_once()
    body = captured[0]
    assert body["direction"] == "inbound"
    assert body["dialin_settings"]["call_id"] == "ext-call-id"
    assert body["dialin_settings"]["call_domain"] == "cosentus.daily.co"


# ── Dict-boundary protection lifecycle ───────────────────────────────────


@pytest.mark.asyncio
async def test_zero_to_one_acquires_protection():
    daily = _daily_mock()
    protection = _protection_mock()
    m = PipelineManager(_settings(), daily, protection)

    # Use a slow fake bot so the dict isn't empty by the time we
    # check protection.set_protected was called.
    bot_event = asyncio.Event()

    async def slow_bot(runner_args):
        await bot_event.wait()

    with patch("app.runner.manager.bot", slow_bot):
        await m.start_browser(agent_id="x")
        # Boundary should have triggered acquire by this point.
        protection.set_protected.assert_awaited_once_with(True)

        # Cleanup: release the bot, wait, ensure released.
        bot_event.set()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_one_to_zero_releases_protection():
    daily = _daily_mock()
    protection = _protection_mock()
    # Make is_protected toggle as set_protected is called.
    state = {"protected": False}

    async def _set(val, **kwargs):
        state["protected"] = val
        protection.is_protected = val
        return True

    protection.set_protected = AsyncMock(side_effect=_set)
    m = PipelineManager(_settings(), daily, protection)

    fast_bot = AsyncMock()
    with patch("app.runner.manager.bot", fast_bot):
        await m.start_browser(agent_id="x")
        await asyncio.sleep(0.05)  # let the task run + clean up

    # Two calls: True on entry, False on exit.
    calls = [c.args[0] for c in protection.set_protected.call_args_list]
    assert calls == [True, False]


@pytest.mark.asyncio
async def test_second_concurrent_call_does_not_re_acquire():
    daily = _daily_mock()
    protection = _protection_mock()
    m = PipelineManager(_settings(), daily, protection)

    bot_event = asyncio.Event()

    async def slow_bot(runner_args):
        await bot_event.wait()

    with patch("app.runner.manager.bot", slow_bot):
        await m.start_browser(agent_id="a")
        await m.start_browser(agent_id="b")
        # Only ONE acquire call total.
        assert protection.set_protected.await_count == 1
        bot_event.set()
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_wrapped_bot_pops_dict_on_exception():
    daily = _daily_mock()
    m = PipelineManager(_settings(), daily, _protection_mock())

    async def crashing_bot(runner_args):
        raise RuntimeError("bot exploded")

    with patch("app.runner.manager.bot", crashing_bot):
        await m.start_browser(agent_id="x")
        await asyncio.sleep(0.05)

    assert m.active_session_count == 0


@pytest.mark.asyncio
async def test_wrapped_bot_pops_dict_on_cancellation():
    daily = _daily_mock()
    m = PipelineManager(_settings(), daily, _protection_mock())

    bot_event = asyncio.Event()

    async def slow_bot(runner_args):
        await bot_event.wait()

    with patch("app.runner.manager.bot", slow_bot):
        await m.start_browser(agent_id="x")
        # Yield so the spawned task starts running and reaches the
        # bot_event.wait() suspension point before we cancel.
        await asyncio.sleep(0.05)
        tasks = list(m.active_sessions.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert m.active_session_count == 0


# ── shutdown ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_sets_draining():
    m = PipelineManager(_settings(), _daily_mock(), _protection_mock())
    assert m.is_draining is False
    await m.shutdown()
    assert m.is_draining is True


# ── _heartbeat_loop per-iteration error handling (Phase 2 #8) ───────────


@pytest.mark.asyncio
async def test_heartbeat_iteration_error_does_not_kill_loop(monkeypatch):
    """Pre-fix: a single ``renew_if_protected`` exception killed the
    loop for the call's lifetime (up to 30 min until ECS expired
    protection). v2 logs the error and retries on the next tick.
    """
    monkeypatch.setattr("app.runner.manager._HEARTBEAT_INTERVAL_SECS", 0)

    daily = _daily_mock()
    protection = _protection_mock()
    call_count = {"n": 0}

    async def flaky_renew():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient throttling")
        # Subsequent iterations succeed.
        return True

    protection.renew_if_protected = AsyncMock(side_effect=flaky_renew)
    m = PipelineManager(_settings(), daily, protection)

    # Stage active sessions so the loop has work to do, then drive
    # several iterations.
    m._active_sessions["call-a"] = MagicMock()
    loop_task = asyncio.create_task(m._heartbeat_loop())

    # Yield enough times for at least 3 iterations to land.
    for _ in range(20):
        await asyncio.sleep(0)
        if call_count["n"] >= 3:
            break

    # Drain the active sessions to let the loop exit cleanly.
    m._active_sessions.clear()
    await asyncio.wait_for(loop_task, timeout=1.0)

    assert call_count["n"] >= 2  # one error + at least one retry


@pytest.mark.asyncio
async def test_heartbeat_cancelled_error_exits_cleanly(monkeypatch):
    """SIGTERM cancels the heartbeat. CancelledError must return
    cleanly — never re-raise (re-raise would propagate up to the
    spawn site where the task was created and emit a noisy traceback).
    """
    monkeypatch.setattr("app.runner.manager._HEARTBEAT_INTERVAL_SECS", 0)

    daily = _daily_mock()
    protection = _protection_mock()
    m = PipelineManager(_settings(), daily, protection)

    m._active_sessions["call-a"] = MagicMock()
    loop_task = asyncio.create_task(m._heartbeat_loop())
    await asyncio.sleep(0)  # let it start

    loop_task.cancel()

    # Should NOT raise — the loop catches CancelledError and returns.
    result = await loop_task
    assert result is None


@pytest.mark.asyncio
async def test_heartbeat_exits_when_active_sessions_empty(monkeypatch):
    """Loop terminates naturally when the dict empties — the next
    spawn 0→1 transition restarts a fresh task.
    """
    monkeypatch.setattr("app.runner.manager._HEARTBEAT_INTERVAL_SECS", 0)

    daily = _daily_mock()
    protection = _protection_mock()
    m = PipelineManager(_settings(), daily, protection)
    # Empty by construction — loop should return immediately.
    await asyncio.wait_for(m._heartbeat_loop(), timeout=1.0)
    protection.renew_if_protected.assert_not_called()
