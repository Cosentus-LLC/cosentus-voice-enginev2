"""``press_digit`` tool — DTMF tones via Daily transport.

Lets the LLM programmatically press keypad digits to navigate IVR
menus on the far end. Implementation queues
:class:`OutputDTMFUrgentFrame` per digit, paced via the
``PRESS_DIGIT_PACING_MS`` env var (default 120 ms).

Two pieces of paid-for production knowledge are preserved from v1:

1. **Pre-DTMF interruption.** Claude often emits conversational
   text alongside a tool-use block ("Sure, one moment" +
   ``press_digit``). On a SIP call the DTMF is out-of-band so the
   far end hears only the text. We push an
   :class:`InterruptionTaskFrame` BEFORE the tones to cut any
   in-flight TTS, then wait 60 ms for the cancel to propagate
   through STT → LLM → TTS → Transport before the first tone
   lands. Without this, ~300 ms of TTS audio leaks alongside the
   tones on WebRTC dev rooms.

2. **120 ms inter-digit pacing.** Some IVRs (Aetna, older UHC
   carrier paths) miss digits when tones arrive < 60 ms apart —
   their detectors require ~50 ms of tone audio + a 50 ms gap. v1
   shipped 120 ms as a safe default. Override via
   ``PRESS_DIGIT_PACING_MS`` for IVRs that need a different
   cadence; no redeploy needed.

The tool returns ``run_llm=False`` so the bot stays silent after
pressing — the IVR's response becomes the next user turn.
"""

from __future__ import annotations

import asyncio
import os
import re

import structlog
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import InterruptionTaskFrame, OutputDTMFUrgentFrame

from app.tools.context import ToolContext
from app.tools.result import ToolResult, error_result, success_result
from app.tools.schema import ToolDefinition, ToolParameter

logger = structlog.get_logger(__name__)


DESCRIPTION_DEFAULT = (
    "Press DTMF digits on the phone keypad to navigate IVR menus. "
    "Valid input: digits 0-9, *, and #. Multi-digit input is sent "
    "as a sequence (e.g. '1234' presses four digits in order). "
    "Use only when the IVR has prompted for keypad input. After "
    "pressing, wait silently for the IVR's response — do NOT speak "
    "between the press and the response."
)


# Pre-validated set of DTMF chars Pipecat / Daily accept.
_VALID_DTMF_RE = re.compile(r"^[0-9*#]+$")

# Settle delay after the pre-DTMF TTS interruption. 60 ms is enough
# for the InterruptionFrame to propagate STT → LLM → TTS → Transport
# in the measured pipeline before the first DTMF lands.
_INTERRUPTION_SETTLE_SECS = 0.060

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
    2. Push :class:`InterruptionTaskFrame` to cut any in-flight TTS.
    3. Sleep 60 ms to let the interruption propagate.
    4. Queue one :class:`OutputDTMFUrgentFrame` per digit, with
       configurable inter-digit pacing.
    5. Return ``run_llm=False`` so the bot stays silent for the
       IVR's response.
    """
    digits = (arguments.get("digits") or "").strip()
    if not digits:
        return error_result("digits argument is required")
    if not _VALID_DTMF_RE.match(digits):
        return error_result(f"Invalid digits {digits!r}. Allowed: 0-9, *, #.")
    if not context.sip_session_id:
        return error_result("No SIP session available for DTMF")
    if context.queue_frame is None:
        return error_result("No frame queue available; cannot send DTMF")

    pacing_ms = _read_pacing_ms()
    pacing_secs = pacing_ms / 1000.0

    # Cut any in-flight TTS so Claude's "sure, one moment" filler
    # doesn't leak into the SIP audio. Failures here are logged but
    # don't block the DTMF — the tones will still go through.
    try:
        await context.queue_frame(InterruptionTaskFrame())
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning(
            "press_digit_interruption_queue_failed",
            error=str(exc),
            call_id=context.call_id,
        )

    # Let the interruption propagate before the first DTMF lands.
    await asyncio.sleep(_INTERRUPTION_SETTLE_SECS)

    # Queue digits one at a time. Daily's send_dtmf accepts a single
    # tone per call; transport_destination is the dataclass field on
    # the frame (set after construction — Pipecat 1.1.0's
    # OutputDTMFUrgentFrame doesn't accept it as a constructor kwarg).
    last_index = len(digits) - 1
    for i, digit_char in enumerate(digits):
        frame = OutputDTMFUrgentFrame(button=KeypadEntry(digit_char))
        frame.transport_destination = context.sip_session_id
        try:
            await context.queue_frame(frame)
        except Exception as exc:  # noqa: BLE001 — surface the failure
            logger.exception(
                "press_digit_queue_failed",
                digits_pressed_so_far=i,
                error=str(exc),
                call_id=context.call_id,
            )
            return error_result(
                "Unable to send DTMF tones. Please try again or ask "
                "the customer to press the digits directly."
            )
        # Pace between digits, but skip the sleep after the last one
        # so we don't add dead time before returning to the LLM.
        if i < last_index and pacing_ms > 0:
            await asyncio.sleep(pacing_secs)

    logger.info(
        "press_digit_completed",
        digit_count=len(digits),
        pacing_ms=pacing_ms,
        sip_session_id=context.sip_session_id,
        call_id=context.call_id,
    )

    return success_result(
        data={"digits_pressed": digits, "digit_count": len(digits)},
        # Stay silent — the IVR's response becomes the next user turn.
        run_llm=False,
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
    # generous margin (30 * 0.12 + interruption + jitter ≈ 4 s).
    timeout_secs=15.0,
    # Once the first tone has been queued, partial cancellation
    # leaves the IVR in mid-input — let it run.
    cancel_on_interruption=False,
)
