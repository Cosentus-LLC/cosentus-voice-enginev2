"""Tests for app.tools.builtin.transfer_call."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from app.tools.builtin.transfer_call import (
    TRANSFER_CALL,
    transfer_call_executor,
)
from app.tools.context import ToolContext
from app.tools.result import ToolStatus


def _ctx(
    *,
    sip_session_id: str | None = "sip-leg-1",
    targets: dict | None = None,
    transport: object | None = None,
) -> ToolContext:
    if transport is None:
        transport = MagicMock()
        transport.sip_call_transfer = AsyncMock(return_value=None)
    return ToolContext(
        call_id="call-1",
        session_id="call-1",
        sip_session_id=sip_session_id,
        transport=transport,
        tool_settings={"targets": targets if targets is not None else {"billing": "+15551234567"}},
    )


class TestTransferCallExecutor:
    async def test_missing_target_returns_error(self):
        result = await transfer_call_executor({"target": ""}, _ctx())
        assert result.status is ToolStatus.ERROR
        assert "target" in (result.error or "").lower()

    async def test_no_targets_configured_returns_error(self):
        result = await transfer_call_executor(
            {"target": "billing"},
            _ctx(targets={}),
        )
        assert result.status is ToolStatus.ERROR
        assert "no configured targets" in (result.error or "").lower()

    async def test_unknown_target_returns_error_listing_available(self):
        result = await transfer_call_executor(
            {"target": "mystery"},
            _ctx(targets={"billing": "+1555", "after_hours": "+1666"}),
        )
        assert result.status is ToolStatus.ERROR
        # Lists known targets so the LLM can pick a valid one if it
        # ever bypasses the enum constraint.
        assert "billing" in (result.error or "")
        assert "after_hours" in (result.error or "")

    async def test_no_sip_session_returns_error(self):
        result = await transfer_call_executor(
            {"target": "billing"},
            _ctx(sip_session_id=None),
        )
        assert result.status is ToolStatus.ERROR
        assert "sip" in (result.error or "").lower()

    async def test_calls_sip_call_transfer_with_correct_envelope(self):
        transport = MagicMock()
        transport.sip_call_transfer = AsyncMock(return_value=None)
        ctx = _ctx(transport=transport, targets={"billing": "+15551234567"})

        result = await transfer_call_executor({"target": "billing"}, ctx)

        assert result.status is ToolStatus.SUCCESS
        transport.sip_call_transfer.assert_awaited_once()
        payload = transport.sip_call_transfer.call_args.args[0]
        # Capital-P toEndPoint per Daily SDK contract.
        assert payload == {
            "sessionId": "sip-leg-1",
            "toEndPoint": "+15551234567",
        }

    async def test_run_llm_false_on_success(self):
        result = await transfer_call_executor(
            {"target": "billing"},
            _ctx(),
        )
        assert result.status is ToolStatus.SUCCESS
        # Hand-off line was already spoken in the LLM turn that
        # triggered this tool call; running LLM again would only
        # add latency.
        assert result.run_llm is False

    async def test_transport_exception_returns_error_with_friendly_message(self):
        transport = MagicMock()
        transport.sip_call_transfer = AsyncMock(side_effect=RuntimeError("Daily SDK exploded"))
        ctx = _ctx(transport=transport)

        result = await transfer_call_executor({"target": "billing"}, ctx)

        assert result.status is ToolStatus.ERROR
        # Surface a polite caller-friendly message, not the raw
        # transport exception.
        assert "transfer" in (result.error or "").lower()

    async def test_handler_does_not_push_pipeline_control_frames(self):
        """REGRESSION GUARD against the Bug A anti-pattern.

        The May 2026 Bug A audit established: tool handlers must
        not push ``InterruptionTaskFrame`` / ``InterruptionFrame``
        / ``CancelTaskFrame`` / any pipeline-control frame, because
        doing so disrupts the function-call lifecycle and makes
        tool_use / tool_result blocks vanish from the LLM context.
        transfer_call already follows the standard pattern (calls
        ``transport.sip_call_transfer`` directly, no frame pushes);
        this test fails the build if anyone re-introduces the
        anti-pattern.
        """
        from pipecat.frames.frames import (
            CancelTaskFrame,
            InterruptionFrame,
            InterruptionTaskFrame,
        )

        queue_frame_mock = AsyncMock()
        transport = MagicMock()
        transport.sip_call_transfer = AsyncMock(return_value=None)
        ctx = ToolContext(
            call_id="call-1",
            session_id="call-1",
            sip_session_id="sip-leg-1",
            transport=transport,
            queue_frame=queue_frame_mock,
            tool_settings={"targets": {"billing": "+15551234567"}},
        )

        await transfer_call_executor({"target": "billing"}, ctx)

        forbidden_types = (
            InterruptionTaskFrame,
            InterruptionFrame,
            CancelTaskFrame,
        )
        for call in queue_frame_mock.await_args_list:
            frame = call.args[0]
            assert not isinstance(frame, forbidden_types), (
                f"transfer_call must not push {type(frame).__name__}"
            )


class TestTransferCallDefinition:
    def test_name_matches_aurora_valid_tool_types(self):
        # Aurora's VALID_TOOL_TYPES uses "transfer_call".
        assert TRANSFER_CALL.name == "transfer_call"

    def test_target_parameter_is_required_and_string(self):
        target = next(p for p in TRANSFER_CALL.parameters if p.name == "target")
        assert target.required is True
        assert target.type == "string"
        # No platform-default enum — added per-agent at registry-build time.
        assert target.enum is None

    def test_long_timeout_for_carrier_negotiation(self):
        # SIP REFER negotiation can take 3-10 seconds; tool needs
        # generous headroom.
        assert TRANSFER_CALL.timeout_secs >= 30.0

    def test_does_not_cancel_on_interruption(self):
        # Partial cancellation would leave the bridge in an undefined
        # state — disallow.
        assert TRANSFER_CALL.cancel_on_interruption is False
