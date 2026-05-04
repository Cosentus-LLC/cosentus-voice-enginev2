"""Walking skeleton — validate v2 Layers 1-4 composition with Pipecat 1.1.0.

Not production code. The minimum end-to-end pipeline that uses every
v2 layer shipped so far (config / settings / service factory /
tools) on top of a real Daily room with a real lambda-loaded agent.
Used to surface composition issues before Layers 5-12 land.

Will be deleted when Layer 8 (pipeline builder) ships.

How to run
----------

Prereqs:

* AWS creds with access to ``medcloud-voice-api:live`` and the
  Secrets Manager API-key blob. Boto3 default credential chain.
* ``backend/voice-agent/scripts/.env.skeleton`` populated (see
  ``scripts/README.md``). Gitignored.
* The repo's venv with ``pipecat-ai[assemblyai,elevenlabs,aws,daily]``
  plus ``python-dotenv``.

Run from the repo root::

    source .venv/bin/activate
    python backend/voice-agent/scripts/walking_skeleton.py

The script prints a Daily room URL plus a three-test script. Open
the URL in your browser, join with mic enabled, and follow the
test prompts. Ctrl-C or hang up the browser to clean up.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import aiohttp
import structlog
from dotenv import load_dotenv

# Spike-only sys.path bootstrap so `from app.*` works when this is
# run directly from the repo root. pytest gets the same treatment
# via pyproject's [tool.pytest.ini_options].pythonpath; outside
# pytest the package isn't installed editable, so we add the parent
# of `app/` (i.e. `backend/voice-agent/`) here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Layer 1, 2, 3, 4
from app.config import Settings, load_agent_config  # noqa: E402
from app.services import build_llm, build_stt, build_tts  # noqa: E402
from app.tools import (  # noqa: E402
    ToolContext,
    ToolExecutor,
    build_registry_for_call,
)

# Pipecat 1.1.0
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallResultProperties,
)
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.daily.utils import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

logger = structlog.get_logger("walking_skeleton")


# Round 2 default: v2-tools-test agent (synthetic, with all three
# Layer 4 tools enabled). Override via env to test other agents.
AGENT_NAME = os.environ.get("SKELETON_AGENT", "v2-tools-test")


def _resolve_sip_session_id(transport: DailyTransport) -> str | None:
    """Best-effort look up the SIP participant id on the transport.

    Browser-only test rooms have no SIP leg, so this returns
    ``None`` — and ``transfer_call`` / ``press_digit`` will return a
    graceful "no SIP session" error. That IS the round-2 expected
    behavior; we're validating tool firing, not full SIP audio.
    """
    try:
        participants = getattr(transport, "_participants", None) or {}
        for pid, participant in participants.items():
            info = participant.get("info") if isinstance(participant, dict) else None
            if isinstance(info, dict) and info.get("isSip"):
                return pid
    except Exception:  # noqa: BLE001 — best-effort lookup
        return None
    return None


async def main() -> None:
    # ── 1. Load .env.skeleton ──────────────────────────────────────────────
    env_file = Path(__file__).resolve().parent / ".env.skeleton"
    if not env_file.exists():
        sys.exit(f"Missing {env_file}. Populate it per scripts/README.md before running.")
    load_dotenv(env_file)
    logger.info("env_loaded", file=str(env_file))

    # ── 2. Build Layer 2 Settings (raises if required env missing) ─────────
    settings = Settings(_env_file=None)
    logger.info(
        "settings_loaded",
        aws_region=settings.aws_region,
        voice_api_lambda_name=settings.voice_api_lambda_name,
    )

    daily_api_key = os.environ.get("DAILY_API_KEY")
    if not daily_api_key:
        sys.exit("DAILY_API_KEY missing from .env.skeleton")

    # ── 3. Create a one-off Daily room + meeting token ─────────────────────
    async with aiohttp.ClientSession() as http:
        rest = DailyRESTHelper(
            daily_api_key=daily_api_key,
            daily_api_url="https://api.daily.co/v1",
            aiohttp_session=http,
        )
        room_params = DailyRoomParams(
            properties=DailyRoomProperties(
                exp=int(time.time()) + 3600,  # 60-minute room
                start_video_off=True,
                enable_chat=False,
                enable_prejoin_ui=False,
                eject_at_room_exp=True,
            ),
        )
        room = await rest.create_room(room_params)
        token = await rest.get_token(room.url, expiry_time=3600)
        logger.info("daily_room_created", url=room.url, expires_in_secs=3600)

        print("\n" + "=" * 64)
        print("Daily room URL:")
        print(f"  {room.url}")
        print("=" * 64)
        print()
        print("TEST SCRIPT — try these three things in any order:")
        print()
        print("TEST 1 — END CALL")
        print("  Say: 'OK we're done, please hang up.'")
        print("  Expected: agent says goodbye, call ends cleanly.")
        print("  Validate in logs: 'end_call_initiated' followed")
        print("                    by clean pipeline shutdown.")
        print()
        print("TEST 2 — TRANSFER")
        print("  Say: 'Please transfer me to the user.'")
        print("  Expected: agent says 'transferring you now',")
        print("            transfer_call_failed in logs because")
        print("            no SIP session for browser audio.")
        print("  This is EXPECTED. The validation is that the")
        print("  tool fired correctly, the API call shape is")
        print("  right, and the error message is graceful.")
        print()
        print("TEST 3 — PRESS DIGIT")
        print("  Say: 'Press 1234 on the keypad.'")
        print("  Expected: agent says 'pressing 1234 now',")
        print("            press_digit_completed in logs with")
        print("            digit_count=4 (or press_digit_failed")
        print("            for browser audio — same expected")
        print("            behavior as transfer).")
        print()
        try:
            input("Press Enter when joined ... ")
        except EOFError:
            sys.exit("No stdin attached; rerun in a terminal.")

        # ── 4. Load real agent from real lambda ────────────────────────────
        logger.info("loading_agent_config", name=AGENT_NAME)
        agent = await load_agent_config(AGENT_NAME, settings=settings)
        logger.info(
            "agent_loaded",
            name=agent.name,
            display_name=agent.display_name,
            llm_model=agent.llm.model,
            tts_voice_id=agent.tts.voice_id,
            tts_model=agent.tts.model,
            system_prompt_chars=len(agent.system_prompt),
            first_message=agent.first_message,
            tools=[t.type for t in agent.tools],
        )

        # ── 5. Build STT / TTS / LLM via Layer 3 ───────────────────────────
        # Layer 3's build_llm now passes system_instruction correctly
        # (fix shipped 2026-05-04 in commit d8c4721); the round-1
        # local override is gone.
        stt = build_stt(agent)
        tts = build_tts(agent)
        llm = build_llm(agent, settings=settings)
        logger.info(
            "services_built",
            stt=type(stt).__name__,
            tts=type(tts).__name__,
            llm=type(llm).__name__,
        )

        # ── 6. Build Layer 4 tool registry + executor ──────────────────────
        # Filters BUILTIN_TOOLS through (a) Settings.disabled_tools
        # and (b) the agent's tools[] opt-in list. Per-agent
        # description overrides + transfer_call's target enum are
        # applied here.
        registry = build_registry_for_call(agent, settings)
        executor = ToolExecutor(registry)
        logger.info(
            "tools_registered",
            tool_count=len(registry.names()),
            tools=registry.names(),
        )

        # ── 7. Seed the LLM context + attach tool catalog ──────────────────
        # The system prompt lives on the LLM service via
        # system_instruction (Layer 3); messages just seeds the
        # bot's first-turn opener. tools= attaches the Layer 4
        # registry's ToolsSchema so Claude sees the catalog when
        # generating responses.
        messages: list[dict] = [
            {"role": "assistant", "content": agent.first_message},
        ]
        context = LLMContext(messages, tools=registry.to_tools_schema())

        # ── 8. Aggregator pair with the locked-in turn machinery ───────────
        smart_turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams())
        vad = SileroVADAnalyzer(
            params=VADParams(stop_secs=0.2, confidence=0.3),
        )
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=vad,
                user_turn_strategies=UserTurnStrategies(
                    start=[MinWordsUserTurnStartStrategy(min_words=3)],
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn),
                    ],
                ),
            ),
        )

        # ── 9. Daily transport with explicit 8 kHz audio_in ────────────────
        # Round 1 finding: Daily defaults to 16 kHz browser audio,
        # but Layer 3's STT is hardcoded to 8 kHz (PSTN). Force the
        # match (tech-debt entry 9).
        transport = DailyTransport(
            room.url,
            token,
            "v2 walking skeleton",
            DailyParams(
                audio_in_enabled=True,
                audio_in_sample_rate=8000,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                vad_analyzer=vad,
            ),
        )

        # ── 10. Pipeline ───────────────────────────────────────────────────
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                aggregators.user(),
                llm,
                tts,
                transport.output(),
                aggregators.assistant(),
            ]
        )

        # ── 11. PipelineTask ───────────────────────────────────────────────
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=300,
        )

        # ── 12. Register a Pipecat handler for each Layer-4 tool ───────────
        # Layer 8 will own this glue eventually (build the
        # ToolContext, route results back to Pipecat's callback,
        # honor run_llm). For the spike, inline it here so the
        # tool-firing path runs end-to-end.
        def make_tool_handler(tool_name: str):
            async def tool_handler(params: FunctionCallParams) -> None:
                tool_context = ToolContext(
                    call_id=os.environ.get("SESSION_ID") or room.url,
                    session_id=os.environ.get("SESSION_ID") or room.url,
                    sip_session_id=_resolve_sip_session_id(transport),
                    transport=transport,
                    queue_frame=task.queue_frame,
                    tool_settings=registry.get_settings(tool_name),
                )

                result = await executor.execute(
                    tool_name,
                    dict(params.arguments),
                    tool_context,
                )

                logger.info(
                    "tool_call_completed",
                    tool=tool_name,
                    status=result.status.value,
                    arguments=dict(params.arguments),
                    error=result.error,
                    run_llm=result.run_llm,
                )

                # Convert ToolResult into Pipecat's expected callback
                # shape. run_llm controls whether the LLM speaks
                # again after the tool result.
                props = FunctionCallResultProperties(run_llm=result.run_llm)
                if result.status.value == "success":
                    payload = result.data or {"status": "ok"}
                else:
                    payload = {"error": result.error or "tool failed"}
                await params.result_callback(payload, properties=props)

            return tool_handler

        for tool_name in registry.names():
            tool_def = registry.get(tool_name)
            assert tool_def is not None  # we just iterated registry.names()
            llm.register_function(
                function_name=tool_name,
                handler=make_tool_handler(tool_name),
                cancel_on_interruption=tool_def.cancel_on_interruption,
                timeout_secs=tool_def.timeout_secs,
            )
            logger.info(
                "tool_handler_registered",
                tool=tool_name,
                cancel_on_interruption=tool_def.cancel_on_interruption,
                timeout_secs=tool_def.timeout_secs,
            )

        # ── 13. Event handlers ─────────────────────────────────────────────
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport_, client):
            logger.info("client_connected", client=client)
            # Speak first_message via TTS directly (round-1 pattern).
            await task.queue_frames([TTSSpeakFrame(agent.first_message)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport_, client):
            logger.info("client_disconnected, cancelling task")
            await task.cancel()

        # ── 14. Run ────────────────────────────────────────────────────────
        runner = PipelineRunner(handle_sigint=True)
        logger.info("pipeline_running, talk to the bot now")
        await runner.run(task)
        logger.info("pipeline_done")


if __name__ == "__main__":
    asyncio.run(main())
