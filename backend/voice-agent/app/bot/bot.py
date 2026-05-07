"""Layer 8 — pipeline composition for one call.

Pipecat foundational two-function pattern:

* :func:`bot(runner_args)` — entry point. Builds a per-call
  transport from :class:`~pipecat.runner.types.RunnerArguments` via
  :func:`~pipecat.runner.utils.create_transport`, then delegates to
  :func:`run_bot`. Today the ``transport_params`` dict has a single
  ``"daily"`` entry; browser / telephony entries can be added later
  with no run_bot changes.
* :func:`run_bot(transport, runner_args, settings)` — transport-
  agnostic. Composes Layers 1-7 into a Pipecat pipeline, runs the
  conversation, and fires :func:`~app.bot.lifecycle.finalize_call`
  in a ``finally`` block.

Per-call isolation is by construction: every per-call dependency
(``AgentConfig``, services, accumulator, error_state, observers,
event handlers, tool handler closures) lives as a local in
``run_bot`` and is captured by closures that never leak across
calls. Only module-level boto3 clients (Layer 1's
``_LAMBDA_CLIENT``, Layer 6's ``_BEDROCK_CLIENT``) persist between
calls — both thread-safe per AWS docs.

The bot file does NOT import HTTP infrastructure. Layer 9 (future)
constructs :class:`~pipecat.runner.types.DailyRunnerArguments` for
HTTP / SQS / webhook triggers and calls ``bot(runner_args)`` via
``asyncio.create_task``. Multiple concurrent calls in one Fargate
process is supported and verified.

Three opener modes (Layer 5):

* ``speak_first=False`` — bot waits for the user to speak first.
* ``speak_first=True`` and non-empty ``first_message`` — static
  TTS opener; LLM doesn't run until the user replies.
* ``speak_first=True`` and empty ``first_message`` — dynamic
  opener; LLM generates the greeting from ``system_prompt``.

All three are dispatched in :func:`on_first_participant_joined` (and
:func:`on_dialout_connected` as a defensive backup). An idempotency
guard ensures the opener fires exactly once per call.

Bugs v2 ships fixed (vs v1):

* :class:`PipelineRunner` constructed with ``handle_sigint=False``
  and ``handle_sigterm=False``. v1 inherited the default ``True``,
  which registers a process-wide signal handler that gets clobbered
  by the next runner construction. With concurrent calls in one
  process, only the most-recently-constructed runner's tasks
  receive SIGINT. v2's Layer 9 (when it lands) owns process signals.
  See tech debt log entry 12.

* ``on_first_participant_joined`` for static-opener delivery, not
  the v1 / walking-skeleton ``on_client_connected`` (which fires
  per participant and would double-fire on multi-participant calls).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyDialinSettings, DailyParams
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from app.bot.lifecycle import finalize_call
from app.config.agent_config import load_agent_config
from app.config.settings import Settings
from app.hydration.hydrator import hydrate_prompt
from app.observers.error_observer import ErrorObserver
from app.observers.error_state import ErrorState
from app.observers.transcript_observer import wire_transcript_handlers
from app.persistence.transcript import TranscriptAccumulator
from app.services.factory import build_llm, build_stt, build_tts
from app.tools.executor import ToolExecutor
from app.tools.registry import build_registry_for_call
from app.tools.result import ToolResult, ToolStatus

logger = structlog.get_logger(__name__)


# ── Locked-in pipeline knobs (matches walking-skeleton round 3) ───────────
#
# Audio rates are uniform across direction:
#
# * input 8 kHz — matches Layer 3's STT lock-in (PSTN-correct,
#   AssemblyAI Universal-3 Pro Streaming). Daily resamples browser
#   audio (16 kHz native) down to 8 kHz so the same setting works
#   everywhere. Without this, browser audio mismatches the STT
#   sample rate and produces garbage transcripts (reproduced in
#   walking-skeleton round 1; see tech debt entry 9).
#
# * output 24 kHz — ElevenLabs flash_v2_5 / turbo_v2_5 emit 24 kHz.
#   Daily handles downsampling to 8 kHz for the SIP leg. Forcing
#   24 kHz at the transport keeps the bot-to-Daily path full-fidelity
#   and lets Daily own the per-direction resampling.
_AUDIO_IN_SAMPLE_RATE = 8000
_AUDIO_OUT_SAMPLE_RATE = 24000

# Smart-turn stop threshold. Default is 3.0s; we override to 0.8s
# matching v1 production baseline (the single biggest contributor
# to "the bot feels sluggish" complaints).
_SMART_TURN_STOP_SECS = 0.8

# Silero VAD parameters tuned for AssemblyAI Mode-2 turn detection.
# confidence=0.3 aligns Silero with AssemblyAI's vad_threshold so
# both layers agree on speech presence. stop_secs=0.2 gives a tight
# grip on end-of-speech.
_VAD_CONFIDENCE = 0.3
_VAD_STOP_SECS = 0.2

# MinWords start strategy — when bot is silent fires after 1 word;
# when bot is speaking requires 3 words to interrupt. Subsumes
# both backchannel filtering ("yeah", "uh-huh") and soft-utterance
# fallback ("Hello?"). See v1 pipeline_ecs.py:643-680 for the full
# rationale and the production calls behind these values.
_MIN_WORDS_INTERRUPT_THRESHOLD = 3

# Default idle timeout — Pipecat tears down the pipeline after
# this many seconds of no audio in either direction. 300s matches
# walking-skeleton round 3 and v1.
_DEFAULT_IDLE_TIMEOUT_SECS = 300


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    settings: Settings,
) -> None:
    """Run one call's pipeline. Transport-agnostic.

    Args:
        transport: Pre-constructed transport (Daily / SmallWebRTC /
            FastAPI websocket). Built by :func:`bot` or by Layer 9.
        runner_args: Pipecat runner arguments. ``runner_args.body``
            carries our per-call dict (agent_id, direction,
            target_number, from_number, case_data, batch_id,
            batch_row_index, dialin_settings, dialout_settings).
        settings: Layer 2 settings. Provides ``aws_region``,
            ``voice_api_lambda_name``, etc.

    Raises:
        ValueError: When ``body.agent_id`` is missing — without an
            agent we have nothing to run.
        asyncio.CancelledError: Re-raised after ``finalize_call``
            so Layer 9's task tracking sees the cancel.

    Other exceptions are caught, captured into ``CallRecord.error``,
    and **not** re-raised — the caller (Layer 9) sees clean
    completion with the row populated.
    """
    body = runner_args.body or {}

    agent_id = body.get("agent_id")
    if not agent_id:
        raise ValueError("runner_args.body.agent_id is required")

    direction = body.get("direction", "inbound")
    target_number = body.get("target_number") or ""
    from_number = body.get("from_number") or ""
    case_data: dict[str, Any] = body.get("case_data") or {}
    batch_id = body.get("batch_id")
    batch_row_index = body.get("batch_row_index")
    dialout_settings = body.get("dialout_settings")

    call_id = str(uuid.uuid4())
    call_started_at = datetime.now(UTC)
    end_status = "completed"
    call_error: str | None = None
    session_id = _extract_session_id(runner_args)

    structlog.contextvars.bind_contextvars(call_id=call_id, session_id=session_id)
    logger.info(
        "call_starting",
        call_id=call_id,
        agent_id=agent_id,
        direction=direction,
        session_id=session_id,
        has_dialout_settings=bool(dialout_settings),
        case_data_keys=sorted(case_data.keys()),
    )

    # ── Layer 1: load agent ────────────────────────────────────────────
    agent = await load_agent_config(agent_id, settings=settings)

    # ── Layer 5: hydrate prompts ───────────────────────────────────────
    hydrated_system = hydrate_prompt(agent.system_prompt, case_data)
    hydrated_first = hydrate_prompt(agent.first_message, case_data) if agent.first_message else ""

    # ── Layer 3: services (STT / TTS / LLM) ────────────────────────────
    stt = build_stt(agent)
    tts = build_tts(agent)
    llm = build_llm(agent, settings, system_instruction=hydrated_system)

    # ── Layer 4: tools ────────────────────────────────────────────────
    registry = build_registry_for_call(agent, settings)
    executor = ToolExecutor(registry)

    # ── Layers 6 + 7: accumulators + error observer ────────────────────
    # User/assistant transcript turns now flow via Pipecat's canonical
    # aggregator events (see Layer 7 ``wire_transcript_handlers``);
    # the wire-up itself happens after the aggregator pair is built
    # below. Tool turns are still appended directly by Layer 4's tool
    # handler closure. The error observer remains a BaseObserver
    # subclass attached to PipelineTask — Pipecat doesn't ship an
    # error-state-recording primitive.
    accumulator = TranscriptAccumulator()
    error_state = ErrorState()
    error_observer = ErrorObserver(error_state)

    # ── Per-call mutable closure state ────────────────────────────────
    # ``sip_session_tracker`` carries the Daily SIP session id from
    # whichever connect event fires (dialin or dialout) over to the
    # tool handlers (transfer_call / press_digit need it). Per-call
    # local; never shared across concurrent calls.
    sip_session_tracker: dict[str, str | None] = {"session_id": None}
    # Idempotency guard — both ``on_first_participant_joined`` and
    # ``on_dialout_connected`` route to the opener path. Only the
    # first to fire delivers the opener; the rest no-op.
    opener_state: dict[str, bool] = {"dispatched": False}

    # ── LLMContext seeding (Layer 5 modes) ────────────────────────────
    # Three branches matching the three opener modes. The user-first
    # branch starts with empty messages — the LLM only runs after the
    # user's first transcribed turn, by which point a real user
    # message exists in context.
    initial_messages: list[dict[str, Any]] = []
    if agent.speak_first and hydrated_first:
        # Static opener: pre-seed as an assistant turn so future LLM
        # calls have the greeting in their conversation history.
        initial_messages.append({"role": "assistant", "content": hydrated_first})
    elif agent.speak_first and not hydrated_first:
        # Dynamic opener: Bedrock's ConverseStream API rejects empty
        # messages[]. Synthetic kickoff "Hi." gives Bedrock something
        # to respond to; Claude treats it as the caller's hello and
        # generates the greeting from system_instruction. See tech
        # debt entry 10.
        initial_messages.append({"role": "user", "content": "Hi."})

    context = LLMContext(initial_messages, tools=registry.to_tools_schema())

    # ── Locked-in turn machinery (Mode 2 + smart-turn) ────────────────
    # VAD goes ONLY on the aggregator's user_params. DailyParams
    # silently accepts a vad_analyzer kwarg (Pydantic v2 default
    # extra="ignore") but ignores the value — verified empirically.
    smart_turn = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=_SMART_TURN_STOP_SECS),
    )
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(stop_secs=_VAD_STOP_SECS, confidence=_VAD_CONFIDENCE),
    )
    aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            user_turn_strategies=UserTurnStrategies(
                start=[
                    MinWordsUserTurnStartStrategy(
                        min_words=_MIN_WORDS_INTERRUPT_THRESHOLD,
                    ),
                ],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)],
            ),
        ),
    )

    # Wire Layer 7's transcript capture to the aggregator's canonical
    # ``on_user_turn_stopped`` / ``on_assistant_turn_stopped`` events.
    # Captures Pipecat's ``AssistantTurnStoppedMessage.interrupted``
    # flag — propagates to ``TranscriptTurn.interrupted`` so analysts
    # can see which assistant turns were cut off by user barge-in.
    wire_transcript_handlers(aggregator_pair, accumulator)

    # ── Pipeline assembly ─────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregator_pair.user(),
            llm,
            tts,
            transport.output(),
            aggregator_pair.assistant(),
        ]
    )

    # ── PipelineTask ──────────────────────────────────────────────────
    # Built-in TurnTrackingObserver + UserBotLatencyObserver auto-
    # attach when enable_metrics=True (Pipecat 1.1.0 internals).
    # Layer 7's ErrorObserver rides alongside; transcript capture is
    # wired via aggregator event handlers above and isn't attached
    # to PipelineTask.observers.
    idle_timeout = (
        getattr(runner_args, "pipeline_idle_timeout_secs", None) or _DEFAULT_IDLE_TIMEOUT_SECS
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[error_observer],
        idle_timeout_secs=idle_timeout,
    )

    # ── Tool handler closure factory ──────────────────────────────────
    # Layer 4's executor wired into Pipecat's register_function. The
    # closure captures all per-call dependencies. Tool-turn
    # accumulation lives here (Layer 7) — synchronous with execution,
    # not via observer. Never raises out of the closure.
    def make_tool_handler(tool_name: str):
        async def tool_handler(params: FunctionCallParams) -> None:
            tool_context = registry.get(tool_name)  # noqa: F841 — sanity
            from app.tools.context import ToolContext

            ctx = ToolContext(
                call_id=call_id,
                session_id=session_id,
                sip_session_id=sip_session_tracker["session_id"],
                transport=transport,
                queue_frame=task.queue_frame,
                tool_settings=registry.get_settings(tool_name),
            )

            try:
                result = await executor.execute(
                    tool_name,
                    dict(params.arguments),
                    ctx,
                )
            except Exception as exc:  # noqa: BLE001 — synthesize a failure
                logger.error(
                    "tool_handler_unexpected_error",
                    tool=tool_name,
                    error=str(exc)[:500],
                    error_type=type(exc).__name__,
                )
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    error=str(exc)[:500],
                    run_llm=True,
                )

            # Layer 7's tool-turn capture. Synchronous with execution,
            # all data already in scope. Best-effort — a transcript
            # append failure never breaks the call.
            try:
                await accumulator.append_tool_turn(
                    tool_name=tool_name,
                    arguments=dict(params.arguments),
                    status=result.status.value,
                    error=result.error,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "tool_turn_append_failed",
                    tool=tool_name,
                    error=str(exc)[:200],
                )

            logger.info(
                "tool_call_completed",
                tool=tool_name,
                status=result.status.value,
                arguments=dict(params.arguments),
                error=result.error,
                run_llm=result.run_llm,
            )

            props = FunctionCallResultProperties(run_llm=result.run_llm)
            if result.status.value == "success":
                payload = result.data or {"status": "ok"}
            else:
                payload = {"error": result.error or "tool failed"}
            await params.result_callback(payload, properties=props)

        return tool_handler

    for tool_name in registry.names():
        spec = registry.get(tool_name)
        assert spec is not None  # iterating names() — must exist
        llm.register_function(
            function_name=tool_name,
            handler=make_tool_handler(tool_name),
            cancel_on_interruption=spec.cancel_on_interruption,
            timeout_secs=spec.timeout_secs,
        )

    # ── Static-opener dispatch (idempotent) ───────────────────────────
    async def deliver_opener_if_needed() -> None:
        """Route to the opener path. Fires at most once per call."""
        if opener_state["dispatched"]:
            return
        opener_state["dispatched"] = True

        if not agent.speak_first:
            logger.info("opener_skipped_user_first", call_id=call_id)
            return

        if hydrated_first:
            # Static opener — append to accumulator FIRST so the
            # transcript reflects what the bot said; LLMFullResponse*
            # frames don't fire for TTSSpeakFrame, so TranscriptObserver
            # won't catch this turn on its own.
            await accumulator.append_assistant_turn(hydrated_first)
            await task.queue_frames([TTSSpeakFrame(hydrated_first)])
            logger.info(
                "opener_static_dispatched",
                call_id=call_id,
                chars=len(hydrated_first),
            )
            return

        # Dynamic opener — kickoff seed is already in
        # LLMContext.messages above; LLMRunFrame triggers generation.
        await task.queue_frames([LLMRunFrame()])
        logger.info("opener_dynamic_dispatched", call_id=call_id)

    # ── Cancellation guard ────────────────────────────────────────────
    # PipelineTask.cancel() is itself idempotent (see ``_finished`` /
    # ``_cancelled`` guards in pipecat 1.1.0 task.py:520-617). The
    # closure-local guard here is cheap insurance with two extra
    # benefits beyond Pipecat's own:
    #
    # 1. Preserves the FIRST cancel reason — more actionable than
    #    the second. When the caller hangs up, both
    #    ``on_participant_left`` and ``on_client_disconnected``
    #    fire (Daily's ``_on_participant_left`` dispatches both
    #    sequentially — verified in Pipecat
    #    transports/daily/transport.py:2958-2964). Without this
    #    guard, the cancel reason that lands on the pipeline is
    #    "client_disconnected" (the second alias) instead of
    #    "participant_left" (the actual semantic event).
    # 2. Suppresses redundant log noise — only the first cancel
    #    path logs its triggering event; subsequent ones short-
    #    circuit silently.
    cancel_state: dict[str, bool] = {"cancelled": False}

    async def safe_cancel(reason: str) -> None:
        if cancel_state["cancelled"]:
            return
        cancel_state["cancelled"] = True
        await task.cancel(reason=reason)

    # ── Event handlers ────────────────────────────────────────────────

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport_, participant):
        logger.info(
            "first_participant_joined",
            call_id=call_id,
            participant_id=participant.get("id") if participant else None,
        )
        await deliver_opener_if_needed()

    @transport.event_handler("on_joined")
    async def on_joined(transport_, data):
        logger.info("bot_joined_room", call_id=call_id)
        # Outbound: we joined the room, now ring the callee. Daily's
        # start_dialout creates a SIP leg from the bot's connection;
        # REST API can't initiate this from outside.
        if direction == "outbound" and dialout_settings:
            # ``callerId`` here is the Daily phone-number-record
            # UUID (set by Layer 9 manager.start_outbound). Logging
            # alongside ``from_number`` (E.164) makes "why is the
            # callee seeing this number?" debuggable in production.
            caller_id_uuid = (dialout_settings or {}).get("callerId")
            try:
                dialout_session_id, error = await transport_.start_dialout(
                    dialout_settings,
                )
                if error:
                    # Sync-return error — the dialout request was
                    # refused before any ringing happened (e.g.
                    # caller-id UUID not authorized in Daily,
                    # malformed phoneNumber). There's no SIP leg to
                    # drain. Cancel immediately so the bot doesn't
                    # strand in an empty room until idle_timeout.
                    logger.error(
                        "dialout_failed_sync",
                        call_id=call_id,
                        error=str(error),
                        target_number=target_number,
                        from_number=from_number,
                        caller_id_uuid=caller_id_uuid,
                    )
                    await safe_cancel("dialout_failed_sync")
                    return
                logger.info(
                    "dialout_initiated",
                    call_id=call_id,
                    dialout_session_id=dialout_session_id,
                    target_number=target_number,
                    from_number=from_number,
                    caller_id_uuid=caller_id_uuid,
                )
            except Exception as exc:  # noqa: BLE001
                # Unexpected exception (network blip, SDK bug, etc.).
                # Cancel for the same reason as the sync-return path:
                # without a SIP leg there's nothing to drain.
                logger.exception(
                    "dialout_unexpected_error",
                    call_id=call_id,
                    error=str(exc),
                    target_number=target_number,
                    from_number=from_number,
                    caller_id_uuid=caller_id_uuid,
                )
                await safe_cancel("dialout_unexpected_error")

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport_, data):
        sid = data.get("sessionId") if isinstance(data, dict) else None
        sip_session_tracker["session_id"] = sid
        logger.info(
            "dialin_connected",
            call_id=call_id,
            sip_session_id=sid,
        )

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport_, data):
        sid = data.get("sessionId") if isinstance(data, dict) else None
        sip_session_tracker["session_id"] = sid
        logger.info(
            "dialout_connected",
            call_id=call_id,
            sip_session_id=sid,
        )
        # Defensive secondary trigger. Empirical Round 4 will confirm
        # which event actually fires when the callee picks up; the
        # idempotency guard in deliver_opener_if_needed makes both
        # safe.
        await deliver_opener_if_needed()

    @transport.event_handler("on_dialin_stopped")
    async def on_dialin_stopped(transport_, data):
        # Inbound SIP leg disconnected (caller hung up).
        # task.cancel() over EndFrame: Pipecat PR #1100 — "EndFrame
        # drains internal queue, could take seconds; cancel stops
        # immediately when there is nothing more to send."
        # Avoids the EndFrame hang race in pipecat issue #3757.
        logger.info("dialin_stopped", call_id=call_id, data=data)
        await safe_cancel("dialin_stopped")

    @transport.event_handler("on_dialout_stopped")
    async def on_dialout_stopped(transport_, data):
        # Outbound SIP leg ended (callee hung up or RUNTIME failure
        # like busy / no-answer-timeout / SIP REFER drop). Same
        # rationale as on_dialin_stopped.
        logger.info("dialout_stopped", call_id=call_id, data=data)
        await safe_cancel("dialout_stopped")

    @transport.event_handler("on_dialout_error")
    async def on_dialout_error(transport_, data):
        # Distinct from the sync-return error in on_joined. This is
        # the RUNTIME failure path — start_dialout returned success
        # but a later phase (ringing, pickup, mid-bridge) failed.
        # Fires 5-60+ seconds after start_dialout. Without this
        # handler the bot strands until idle_timeout (5 min).
        logger.error(
            "dialout_failed_async",
            call_id=call_id,
            data=data,
        )
        await safe_cancel("dialout_failed_async")

    @transport.event_handler("on_dialin_error")
    async def on_dialin_error(transport_, data):
        # Symmetric with on_dialout_error. Inbound SIP bridge
        # failures: handshake fail, mid-call drop, etc.
        logger.error(
            "dialin_failed",
            call_id=call_id,
            data=data,
        )
        await safe_cancel("dialin_failed")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport_, participant, reason):
        logger.info(
            "participant_left",
            call_id=call_id,
            participant_id=participant.get("id") if participant else None,
            reason=reason,
        )
        await safe_cancel("participant_left")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_, client):
        # Daily's ``_on_participant_left`` (transport.py:2958-2964)
        # dispatches BOTH ``on_participant_left`` AND
        # ``on_client_disconnected`` for every leg type — PSTN,
        # browser, bot. We register both as defense-in-depth in
        # case Pipecat's alias mapping ever changes. The
        # safe_cancel guard ensures only the first one drives the
        # actual cancel; the second short-circuits.
        logger.info("client_disconnected", call_id=call_id)
        await safe_cancel("client_disconnected")

    @transport.event_handler("on_error")
    async def on_transport_error(transport_, error):
        # Generic transport-level errors — websocket signalling
        # failure, ICE candidate exhaustion, Daily-API-internal
        # errors. Distinct from ErrorFrames in the pipeline (which
        # Layer 7's ErrorObserver catches).
        #
        # Logging only — do NOT cancel here. Some transport errors
        # are recoverable (transient signalling glitches). Let the
        # pipeline's idle_timeout catch terminal failures so we
        # don't kill calls that recover on their own.
        logger.error(
            "transport_error",
            call_id=call_id,
            error=str(error),
        )

    @transport.event_handler("on_left")
    async def on_left(transport_):
        # Receipt that the transport actually finished tearing down
        # before run_bot's finally block fires. Helps diagnose stuck
        # cleanups in production.
        logger.info("transport_left", call_id=call_id)

    # ── Run pipeline + finalize ───────────────────────────────────────
    # PipelineRunner with both signal flags off — Layer 9 owns
    # process-level signals. v1 / AWS sample inherit the default
    # ``handle_sigint=True`` which clobbers prior runners' handlers
    # under concurrent calls. See tech debt entry 12.
    runner = PipelineRunner(
        handle_sigint=False,
        handle_sigterm=False,
    )

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        # Layer 9 cancelling our task (deploy / scale-in) — record
        # status, finalize, then re-raise so the asyncio task tracking
        # sees the cancellation.
        end_status = "cancelled"
        logger.info("pipeline_cancelled", call_id=call_id)
        raise
    except Exception as exc:  # noqa: BLE001 — capture for CallRecord.error
        end_status = "failed"
        call_error = str(exc)[:1000]
        logger.error(
            "pipeline_run_failed",
            call_id=call_id,
            error=call_error,
            error_type=type(exc).__name__,
        )
    finally:
        try:
            await finalize_call(
                call_id=call_id,
                agent=agent,
                accumulator=accumulator,
                error_state=error_state,
                case_data=case_data,
                started_at=call_started_at,
                end_status=end_status,
                call_error=call_error,
                direction=direction,
                target_number=target_number,
                from_number=from_number,
                session_id=session_id,
                batch_id=batch_id,
                batch_row_index=batch_row_index,
                settings=settings,
            )
        except Exception:  # noqa: BLE001 — never let finalize crash run_bot
            logger.exception(
                "finalize_call_unexpected_error",
                call_id=call_id,
            )
        structlog.contextvars.unbind_contextvars("call_id", "session_id")


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point. Builds the per-call transport, calls :func:`run_bot`.

    Layer 9 (or :func:`pipecat.runner.run.main` in dev) calls this
    via ``asyncio.create_task(bot(runner_args))`` per spawn. The
    function returns when the call completes, fails, or is
    cancelled — Layer 9's task tracking observes either the return
    or the :exc:`asyncio.CancelledError`.
    """
    settings = Settings()  # env-driven, cheap, per-call construction is fine
    body = runner_args.body or {}

    transport_params = {
        "daily": lambda: DailyParams(
            api_key=_get_daily_api_key(),
            audio_in_enabled=True,
            audio_in_sample_rate=_AUDIO_IN_SAMPLE_RATE,
            audio_out_enabled=True,
            audio_out_sample_rate=_AUDIO_OUT_SAMPLE_RATE,
            # End-silence padding after EndFrame. v1 forced 0 so the
            # caller doesn't hear dead air after the bot says goodbye
            # and queues end_call. Keeping it.
            audio_out_end_silence_secs=0,
            # We do not use Daily's transcription. Layer 3's
            # AssemblyAI Mode 2 is the locked-in STT.
            transcription_enabled=False,
            dialin_settings=_build_dialin_settings(body),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, settings)


# ── Helpers ─────────────────────────────────────────────────────────────

_DAILY_API_KEY_ENV = "DAILY_API_KEY"


def _get_daily_api_key() -> str:
    """Resolve ``DAILY_API_KEY`` from the environment.

    Layer 2's ``Settings`` doesn't model Daily-specific credentials
    yet — Daily is a transport concern, not a service concern. Read
    direct here. When Layer 9 lands, it can plumb a typed
    ``Settings.daily_api_key`` through; this stays the fallback.
    """
    import os

    key = os.environ.get(_DAILY_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{_DAILY_API_KEY_ENV} environment variable is required for Daily transport"
        )
    return key


def _extract_session_id(runner_args: RunnerArguments) -> str:
    """Derive the recording-webhook binding (= Daily room name).

    For :class:`~pipecat.runner.types.DailyRunnerArguments`, the
    last URL segment of ``room_url`` is the room name Daily uses
    in the ``recording.ready-to-download`` webhook. Layer 6's
    :class:`CallRecord.session_id` matches this, so the lambda
    webhook handler can join recording artifacts to the call row.

    Non-Daily transports get a UUID fallback. There's no recording
    webhook for browser calls anyway.
    """
    room_url = getattr(runner_args, "room_url", None)
    if room_url:
        return str(room_url).rstrip("/").split("/")[-1]
    return str(uuid.uuid4())


def _build_dialin_settings(body: dict[str, Any]) -> DailyDialinSettings | None:
    """Convert ``body.dialin_settings`` dict to :class:`DailyDialinSettings`.

    Layer 9's webhook handler populates this for inbound PSTN calls
    from Daily's ``daily-dialin-webhook`` payload. Outbound and
    browser calls have ``dialin_settings=None``.
    """
    raw = body.get("dialin_settings")
    if not raw or not isinstance(raw, dict):
        return None
    call_id = raw.get("call_id") or raw.get("callId") or ""
    call_domain = raw.get("call_domain") or raw.get("callDomain") or ""
    if not call_id or not call_domain:
        return None
    return DailyDialinSettings(call_id=call_id, call_domain=call_domain)
