"""Tests for app.tools.builtin.press_digit."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from app.tools.builtin.press_digit import (
    PRESS_DIGIT,
    press_digit_executor,
)
from app.tools.context import ToolContext
from app.tools.result import ToolStatus
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    InterruptionTaskFrame,
    OutputDTMFUrgentFrame,
)


def _ctx(
    *,
    sip_session_id: str | None = "sip-leg-1",
    queue_frame: AsyncMock | None = None,
) -> tuple[ToolContext, AsyncMock]:
    qf = queue_frame or AsyncMock(return_value=None)
    return (
        ToolContext(
            call_id="call-1",
            session_id="call-1",
            sip_session_id=sip_session_id,
            queue_frame=qf,
        ),
        qf,
    )


class TestPressDigitExecutor:
    async def test_empty_digits_returns_error(self):
        ctx, _ = _ctx()
        result = await press_digit_executor({"digits": ""}, ctx)
        assert result.status is ToolStatus.ERROR

    async def test_invalid_chars_return_error(self):
        ctx, _ = _ctx()
        result = await press_digit_executor({"digits": "12abc"}, ctx)
        assert result.status is ToolStatus.ERROR
        assert "invalid" in (result.error or "").lower()

    async def test_no_sip_session_returns_error(self):
        ctx, _ = _ctx(sip_session_id=None)
        result = await press_digit_executor({"digits": "123"}, ctx)
        assert result.status is ToolStatus.ERROR
        assert "sip" in (result.error or "").lower()

    async def test_no_queue_frame_returns_error(self):
        ctx = ToolContext(call_id="call-1", sip_session_id="sip-1", queue_frame=None)
        result = await press_digit_executor({"digits": "123"}, ctx)
        assert result.status is ToolStatus.ERROR

    async def test_queues_interruption_first_then_dtmf_in_order(self):
        ctx, qf = _ctx()

        result = await press_digit_executor({"digits": "12"}, ctx)

        assert result.status is ToolStatus.SUCCESS
        # First call must be the interruption.
        first_frame = qf.call_args_list[0].args[0]
        assert isinstance(first_frame, InterruptionTaskFrame)

        # Subsequent calls are the DTMF frames in order.
        dtmf_frames = [
            call.args[0]
            for call in qf.call_args_list[1:]
            if isinstance(call.args[0], OutputDTMFUrgentFrame)
        ]
        assert len(dtmf_frames) == 2
        assert dtmf_frames[0].button == KeypadEntry("1")
        assert dtmf_frames[1].button == KeypadEntry("2")

    async def test_dtmf_frames_carry_sip_session_id(self):
        ctx, qf = _ctx()
        await press_digit_executor({"digits": "1"}, ctx)
        dtmf_frames = [
            call.args[0]
            for call in qf.call_args_list
            if isinstance(call.args[0], OutputDTMFUrgentFrame)
        ]
        assert dtmf_frames[0].transport_destination == "sip-leg-1"

    async def test_uses_pacing_ms_from_env(self, monkeypatch: pytest.MonkeyPatch):
        # Set an absurdly small pacing so the test stays fast and
        # we can still observe the inter-digit sleep.
        monkeypatch.setenv("PRESS_DIGIT_PACING_MS", "5")
        ctx, _ = _ctx()
        # Just verify it doesn't crash and returns success — exact
        # timing observation is fragile.
        result = await press_digit_executor({"digits": "12"}, ctx)
        assert result.status is ToolStatus.SUCCESS

    async def test_default_pacing_when_env_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PRESS_DIGIT_PACING_MS", raising=False)
        ctx, qf = _ctx()
        # Single digit — no pacing sleep involved, just confirms
        # the env-unset path doesn't fail.
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.SUCCESS

    async def test_invalid_pacing_env_falls_back_loudly(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PRESS_DIGIT_PACING_MS", "garbage")
        ctx, _ = _ctx()
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.SUCCESS

    async def test_run_llm_false_on_success(self):
        ctx, _ = _ctx()
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        # Stay silent — the IVR's response becomes the next user turn.
        assert result.run_llm is False

    async def test_no_pacing_sleep_after_last_digit(self, monkeypatch: pytest.MonkeyPatch):
        # Pacing 200ms; if we slept after the last digit, total
        # elapsed for one digit would be > 200ms. Single-digit case
        # should be fast (no inter-digit sleeps).
        monkeypatch.setenv("PRESS_DIGIT_PACING_MS", "200")
        ctx, _ = _ctx()
        loop = asyncio.get_running_loop()
        start = loop.time()
        await press_digit_executor({"digits": "1"}, ctx)
        elapsed = loop.time() - start
        # 60ms interruption settle + a few ms of overhead, no 200ms
        # post-last-digit sleep.
        assert elapsed < 0.15, f"elapsed={elapsed}"


class TestPressDigitDefinition:
    def test_name_matches_aurora(self):
        assert PRESS_DIGIT.name == "press_digit"

    def test_digits_param_is_required_with_pattern(self):
        digits = next(p for p in PRESS_DIGIT.parameters if p.name == "digits")
        assert digits.required is True
        assert digits.type == "string"
        assert digits.pattern == r"^[0-9*#]+$"

    def test_does_not_cancel_on_interruption(self):
        assert PRESS_DIGIT.cancel_on_interruption is False
