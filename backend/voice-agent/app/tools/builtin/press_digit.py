"""``press_digit`` tool — DTMF tones via Daily transport.

Lets the LLM programmatically press keypad digits to navigate IVR
menus on the far end. Implementation calls Daily's
``transport.send_dtmf({"tones": ..., "sessionId": ..., "digitDurationMs": ...})``
once with the full digit string — Daily paces the tones internally
at the configured ``digitDurationMs`` cadence.

History note: an earlier v2 revision pushed an
:class:`~pipecat.frames.frames.InterruptionTaskFrame` from inside the
handler to clear pending TTS before sending tones. That broke the
function-call lifecycle: ``FunctionCallInProgressFrame`` and
``FunctionCallResultFrame`` (both ``UninterruptibleFrame``) failed
to land in the assistant aggregator because the interruption fired
mid-broadcast, so the tool_use / tool_result blocks never made it
into the LLM context. Claude lost memory of every press_digit call,
producing a multi-press loop on follow-up turns ("Did you do it?"
→ Claude calls press_digit again because it has no record of the
first call). Verified empirically in the v2 inbound PSTN test
(call_id ``c18a181a-...``).

The fix per the standard Pipecat tool pattern is: NO custom
interruption frames in the handler. Just send the DTMF and call
``result_callback``. Pipecat's framework handles ordering. References:

* `Pipecat function-calling docs <https://docs.pipecat.ai/pipecat/learn/function-calling>`_
* `Pipecat issue #3661 <https://github.com/pipecat-ai/pipecat/issues/3661>`_
  (function calls + interruptions are fragile)
* `Pipecat-flows hardening writeup
  <https://dev.to/kollaikalrupesh/hardening-pipecat-a-month-of-fixing-what-matters-44l>`_

The tool's ``cancel_on_interruption`` and ``run_llm`` both stay at
the documented defaults (``True``). With ``run_llm=True``, the LLM
re-fires after the tool result lands so Claude can confirm the
press in conversation; this keeps the tool history alive in
context. The 120 ms ``digitDurationMs`` matches v1's production
calibration: some IVRs (Aetna, older UHC carrier paths) miss tones
that arrive < 60 ms apart — their detectors require ~50 ms of tone
audio + a 50 ms gap. 120 ms is the safe default. Override via
``PRESS_DIGIT_PACING_MS`` for IVRs that need a different cadence.
"""

from __future__ import annotations

import os
import re

import structlog

from app.tools.context import ToolContext
from app.tools.result import ToolResult, error_result, success_result
from app.tools.schema import ToolDefinition, ToolParameter

logger = structlog.get_logger(__name__)


DESCRIPTION_DEFAULT = (
    "Press DTMF digits on the phone keypad to navigate IVR menus. "
    "Valid input: digits 0-9, *, and #. Multi-digit input is sent "
    "as a sequence (e.g. '1234' presses four digits in order). "
    "Use only when the IVR has prompted for keypad input. After "
    "pressing, you will receive a tool result confirming the press; "
    "wait for the IVR's response before pressing again."
)


# Pre-validated set of DTMF chars Pipecat / Daily accept.
_VALID_DTMF_RE = re.compile(r"^[0-9*#]+$")

_DEFAULT_PACING_MS = 120


def _read_pacing_ms() -> int:
    """Read ``PRESS_DIGIT_PACING_MS`` from env, fall back to 120.

    Function (not module-level constant) so operators can flip the
    value via env-var without a process restart of import-once
    semantics. Out-of-range / non-numeric values fall back loudly.
    """
    raw = os.environ.get("PRESS_DIGIT_PACING_MS", "").strip()
    if not raw:
        return _DEFAULT_PACING_MS
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "press_digit_pacing_ms_invalid",
            value=raw,
            using_default_ms=_DEFAULT_PACING_MS,
        )
        return _DEFAULT_PACING_MS
    if val < 0 or val > 2000:
        logger.warning(
            "press_digit_pacing_ms_out_of_range",
            value=val,
            using_default_ms=_DEFAULT_PACING_MS,
        )
        return _DEFAULT_PACING_MS
    return val


async def press_digit_executor(
    arguments: dict,
    context: ToolContext,
) -> ToolResult:
    """Send DTMF tones through the active transport.

    Flow:

    1. Validate digit string (empty → error; invalid chars → error).
    2. Validate SIP session + transport (required for DTMF routing).
    3. Single ``transport.send_dtmf({...})`` call — Daily paces
       internally at ``digitDurationMs``.
    4. Return success with ``digits_pressed`` so the LLM can confirm
       in conversation when ``run_llm=True`` (the default) re-fires
       it.
    """
    digits = (arguments.get("digits") or "").strip()
    if not digits:
        return error_result("digits argument is required")
    if not _VALID_DTMF_RE.match(digits):
        return error_result(f"Invalid digits {digits!r}. Allowed: 0-9, *, #.")
    if not context.sip_session_id:
        return error_result("No SIP session available for DTMF")
    if context.transport is None:
        return error_result("No transport available for DTMF")

    pacing_ms = _read_pacing_ms()

    # One call. Daily's send_dtmf accepts the full tone string and
    # paces internally per ``digitDurationMs``. No frame queueing,
    # no pre-DTMF interruption frame, no manual pacing sleep — that
    # entire pattern was incompatible with Pipecat's function-call
    # lifecycle. See module docstring.
    settings: dict = {
        "tones": digits,
        "sessionId": context.sip_session_id,
        "digitDurationMs": pacing_ms,
    }
    try:
        error = await context.transport.send_dtmf(settings)
    except Exception as exc:  # noqa: BLE001 — surface the failure
        logger.exception(
            "press_digit_send_failed",
            error=str(exc),
            digits=digits,
            sip_session_id=context.sip_session_id,
            call_id=context.call_id,
        )
        return error_result(
            "Unable to send DTMF tones. Please try again or ask "
            "the customer to press the digits directly."
        )

    if error:
        # Daily SDK returns the error string as the awaited value.
        logger.error(
            "press_digit_daily_error",
            error=str(error),
            digits=digits,
            sip_session_id=context.sip_session_id,
            call_id=context.call_id,
        )
        return error_result(f"DTMF send refused by Daily: {error}. Please try again.")

    logger.info(
        "press_digit_completed",
        digit_count=len(digits),
        digit_duration_ms=pacing_ms,
        sip_session_id=context.sip_session_id,
        call_id=context.call_id,
    )

    return success_result(
        data={"digits_pressed": digits, "digit_count": len(digits)},
    )


PRESS_DIGIT = ToolDefinition(
    name="press_digit",
    description=DESCRIPTION_DEFAULT,
    parameters=[
        ToolParameter(
            name="digits",
            type="string",
            description=(
                "DTMF digits to press. Valid: 0-9, *, #. Examples: '1', '4567', '#', '*1'."
            ),
            required=True,
            pattern=r"^[0-9*#]+$",
        ),
    ],
    executor=press_digit_executor,
    # 15s covers a 30-digit account number at 120 ms pacing with
    # generous margin (30 * 0.12 + jitter ≈ 4 s).
    timeout_secs=15.0,
    # ``cancel_on_interruption`` and ``run_llm`` intentionally use
    # the Pipecat defaults (both ``True``). Earlier v2 revisions
    # set them to ``False`` to support a homegrown TTS-clearing
    # pattern that broke the function-call lifecycle. See module
    # docstring for the empirical bug history.
)
