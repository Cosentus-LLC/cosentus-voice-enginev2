"""Tests for app.tools.builtin.press_digit.

After the Bug A rewrite (May 2026), press_digit follows the standard
Pipecat tool pattern:

* No ``InterruptionTaskFrame`` from inside the handler.
* No per-digit ``OutputDTMFUrgentFrame`` queueing.
* Single ``transport.send_dtmf({...})`` call — Daily paces internally.
* ``cancel_on_interruption`` and ``run_llm`` use Pipecat defaults
  (both ``True``).

The regression test ``test_handler_does_not_push_pipeline_control_frames``
is the empirical guard against the bug recurring: if anyone re-adds
an ``InterruptionTaskFrame`` (or any pipeline-control frame) to the
handler, the test fails before the broken behavior reaches a real
PSTN call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from app.tools.builtin.press_digit import (
    PRESS_DIGIT,
    press_digit_executor,
)
from app.tools.context import ToolContext
from app.tools.result import ToolStatus
from pipecat.frames.frames import (
    CancelTaskFrame,
    InterruptionFrame,
    InterruptionTaskFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
)


def _ctx(
    *,
    sip_session_id: str | None = "sip-leg-1",
    transport: MagicMock | None = None,
    queue_frame: AsyncMock | None = None,
    message_history: list[dict] | None = None,
    ivr_navigation_state: dict | None = None,
) -> tuple[ToolContext, MagicMock, AsyncMock]:
    if transport is None:
        transport = MagicMock()
        transport.send_dtmf = AsyncMock(return_value=None)
    qf = queue_frame or AsyncMock(return_value=None)
    return (
        ToolContext(
            call_id="call-1",
            session_id="call-1",
            sip_session_id=sip_session_id,
            transport=transport,
            queue_frame=qf,
            message_history=message_history or [],
            ivr_navigation_state=ivr_navigation_state if ivr_navigation_state is not None else {},
        ),
        transport,
        qf,
    )


class TestPressDigitExecutor:
    async def test_empty_digits_returns_error(self):
        ctx, transport, _ = _ctx()
        result = await press_digit_executor({"digits": ""}, ctx)
        assert result.status is ToolStatus.ERROR
        transport.send_dtmf.assert_not_awaited()

    async def test_invalid_chars_return_error(self):
        ctx, transport, _ = _ctx()
        result = await press_digit_executor({"digits": "12abc"}, ctx)
        assert result.status is ToolStatus.ERROR
        assert "invalid" in (result.error or "").lower()
        transport.send_dtmf.assert_not_awaited()

    async def test_no_sip_session_returns_error(self):
        ctx, transport, _ = _ctx(sip_session_id=None)
        result = await press_digit_executor({"digits": "123"}, ctx)
        assert result.status is ToolStatus.ERROR
        assert "sip" in (result.error or "").lower()
        transport.send_dtmf.assert_not_awaited()

    async def test_no_transport_returns_error(self):
        ctx = ToolContext(
            call_id="call-1",
            sip_session_id="sip-1",
            transport=None,
            queue_frame=AsyncMock(),
        )
        result = await press_digit_executor({"digits": "123"}, ctx)
        assert result.status is ToolStatus.ERROR
        assert "transport" in (result.error or "").lower()

    async def test_calls_send_dtmf_with_correct_payload(self):
        ctx, transport, _ = _ctx()
        result = await press_digit_executor({"digits": "123"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        transport.send_dtmf.assert_awaited_once()
        settings = transport.send_dtmf.await_args.args[0]
        assert settings["tones"] == "123"
        assert settings["sessionId"] == "sip-leg-1"
        assert settings["digitDurationMs"] == 120

    async def test_passes_full_digit_string_in_one_call(self):
        """v2 fix: previously queued one OutputDTMFUrgentFrame per
        digit. New behavior: single send_dtmf with the whole string —
        Daily paces internally.
        """
        ctx, transport, _ = _ctx()
        await press_digit_executor({"digits": "*1#23"}, ctx)
        assert transport.send_dtmf.await_count == 1
        assert transport.send_dtmf.await_args.args[0]["tones"] == "*1#23"

    async def test_uses_pacing_ms_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PRESS_DIGIT_PACING_MS", "75")
        ctx, transport, _ = _ctx()
        await press_digit_executor({"digits": "1"}, ctx)
        assert transport.send_dtmf.await_args.args[0]["digitDurationMs"] == 75

    async def test_default_pacing_when_env_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("PRESS_DIGIT_PACING_MS", raising=False)
        ctx, transport, _ = _ctx()
        await press_digit_executor({"digits": "1"}, ctx)
        assert transport.send_dtmf.await_args.args[0]["digitDurationMs"] == 120

    async def test_invalid_pacing_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PRESS_DIGIT_PACING_MS", "garbage")
        ctx, transport, _ = _ctx()
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        assert transport.send_dtmf.await_args.args[0]["digitDurationMs"] == 120

    async def test_out_of_range_pacing_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PRESS_DIGIT_PACING_MS", "9999")
        ctx, transport, _ = _ctx()
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        assert transport.send_dtmf.await_args.args[0]["digitDurationMs"] == 120

    async def test_run_llm_default_true_so_claude_can_confirm(self):
        """v2 fix: previously ``run_llm=False``. With the new pattern
        the LLM is re-invoked after the tool result lands, so Claude
        can confirm the press in conversation. This is what gives
        Claude conversational memory of having pressed digits.
        """
        ctx, _, _ = _ctx()
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        assert result.run_llm is True

    async def test_success_data_includes_digits_pressed(self):
        ctx, _, _ = _ctx()
        result = await press_digit_executor({"digits": "456"}, ctx)
        assert result.status is ToolStatus.SUCCESS
        assert result.data is not None
        assert result.data["digits_pressed"] == "456"
        assert result.data["digit_count"] == 3

    async def test_blocks_same_digit_without_new_ivr_prompt(self):
        state: dict = {}
        prompt = [{"role": "user", "content": "For claims, press 1."}]
        ctx, transport, _ = _ctx(message_history=prompt, ivr_navigation_state=state)

        first = await press_digit_executor({"digits": "1"}, ctx)
        second = await press_digit_executor({"digits": "1"}, ctx)

        assert first.status is ToolStatus.SUCCESS
        assert second.status is ToolStatus.ERROR
        assert "new IVR prompt" in (second.error or "")
        assert transport.send_dtmf.await_count == 1

    async def test_allows_same_digit_after_new_ivr_prompt(self):
        state: dict = {}
        first_prompt = [{"role": "user", "content": "For claims, press 1."}]
        ctx, transport, _ = _ctx(message_history=first_prompt, ivr_navigation_state=state)

        first = await press_digit_executor({"digits": "1"}, ctx)
        ctx.message_history = [{"role": "user", "content": "For eligibility, press 1."}]
        second = await press_digit_executor({"digits": "1"}, ctx)

        assert first.status is ToolStatus.SUCCESS
        assert second.status is ToolStatus.SUCCESS
        assert transport.send_dtmf.await_count == 2

    async def test_blocks_repeated_fallback_until_new_prompt(self):
        state: dict = {"last_digits": "1", "last_prompt_norm": "for claims, press 1."}
        prompt = [{"role": "user", "content": "For claims, press 1."}]
        ctx, transport, _ = _ctx(message_history=prompt, ivr_navigation_state=state)

        first = await press_digit_executor({"digits": "0"}, ctx)
        second = await press_digit_executor({"digits": "0"}, ctx)
        ctx.message_history = [{"role": "user", "content": "Please hold while I transfer you."}]
        third = await press_digit_executor({"digits": "0"}, ctx)

        assert first.status is ToolStatus.SUCCESS
        assert second.status is ToolStatus.ERROR
        assert "already pressed" in (second.error or "")
        assert third.status is ToolStatus.SUCCESS
        assert transport.send_dtmf.await_count == 2

    async def test_greenlund_replay_blocks_twelve_repeated_ones(self):
        state: dict = {}
        prompt = [{"role": "user", "content": "For claims status, press 1."}]
        ctx, transport, _ = _ctx(message_history=prompt, ivr_navigation_state=state)

        results = [await press_digit_executor({"digits": "1"}, ctx) for _ in range(12)]

        assert results[0].status is ToolStatus.SUCCESS
        assert [result.status for result in results[1:]] == [ToolStatus.ERROR] * 11
        assert transport.send_dtmf.await_count == 1

    async def test_guard_does_not_log_prompt_text(self, mocker):
        mock_logger = mocker.patch("app.tools.builtin.press_digit.logger")
        state: dict = {}
        prompt_text = "For secret claim ABC123, press 1."
        ctx, _, _ = _ctx(
            message_history=[{"role": "user", "content": prompt_text}],
            ivr_navigation_state=state,
        )

        await press_digit_executor({"digits": "1"}, ctx)
        result = await press_digit_executor({"digits": "1"}, ctx)

        assert result.status is ToolStatus.ERROR
        assert mock_logger.warning.call_count == 1
        assert prompt_text not in repr(mock_logger.warning.call_args)

    async def test_send_dtmf_exception_returns_error(self):
        transport = MagicMock()
        transport.send_dtmf = AsyncMock(side_effect=RuntimeError("daily down"))
        ctx, _, _ = _ctx(transport=transport)
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.ERROR
        assert "DTMF" in (result.error or "")

    async def test_send_dtmf_returning_error_string_is_propagated(self):
        """Daily SDK returns a non-empty error description on
        SIP-level failure (e.g., session ID stale). Surface it.
        """
        transport = MagicMock()
        transport.send_dtmf = AsyncMock(return_value="invalid-session-id")
        ctx, _, _ = _ctx(transport=transport)
        result = await press_digit_executor({"digits": "1"}, ctx)
        assert result.status is ToolStatus.ERROR
        assert "invalid-session-id" in (result.error or "")

    async def test_handler_does_not_push_pipeline_control_frames(self):
        """REGRESSION GUARD for Bug A.

        Earlier v2 revisions pushed ``InterruptionTaskFrame`` from
        inside the handler to clear pending TTS before sending
        tones. That broke the function-call lifecycle: tool_use /
        tool_result blocks vanished from LLM context. The standard
        Pipecat tool pattern is: handlers do NOT push pipeline-
        control frames. If this test fails, someone re-introduced
        the anti-pattern.
        """
        ctx, _, qf = _ctx()
        await press_digit_executor({"digits": "123"}, ctx)

        # The handler may legitimately not call queue_frame at all
        # under the new design (it goes through transport.send_dtmf).
        # If queue_frame WAS called, none of the calls may carry
        # any pipeline-control frame.
        forbidden_types = (
            InterruptionTaskFrame,
            InterruptionFrame,
            CancelTaskFrame,
            OutputDTMFFrame,
            OutputDTMFUrgentFrame,
        )
        for call in qf.await_args_list:
            frame = call.args[0]
            assert not isinstance(frame, forbidden_types), (
                f"press_digit handler must not push {type(frame).__name__} — "
                "see module docstring for the bug history."
            )


class TestPressDigitDefinition:
    def test_name_matches_aurora(self):
        assert PRESS_DIGIT.name == "press_digit"

    def test_digits_param_is_required_with_pattern(self):
        digits = next(p for p in PRESS_DIGIT.parameters if p.name == "digits")
        assert digits.required is True
        assert digits.type == "string"
        assert digits.pattern == r"^[0-9*#]+$"

    def test_uses_default_cancel_on_interruption_true(self):
        """v2 fix: previously ``cancel_on_interruption=False`` to
        support the homegrown TTS-clearing pattern. With that hack
        gone, the documented Pipecat default ``True`` is correct.
        """
        assert PRESS_DIGIT.cancel_on_interruption is True
