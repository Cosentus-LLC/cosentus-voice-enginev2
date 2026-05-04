"""Walking skeleton — validate v2 Layers 1-3 composition with Pipecat 1.1.0.

Not production code. The minimum end-to-end pipeline that uses every
v2 layer shipped so far (config / settings / service factory) on top
of a real Daily room with a real lambda-loaded agent. Used to surface
composition issues before Layers 4-12 land.

Will be deleted when Layer 8 (pipeline builder) ships.

How to run
----------

Prereqs:

* AWS creds with access to ``medcloud-voice-api:live`` and the
  Secrets Manager API-key blob. Boto3 default credential chain.
* ``backend/voice-agent/scripts/.env.skeleton`` populated with the
  five env vars below (see ``scripts/README.md``). Gitignored.
* The repo's venv with ``pipecat-ai[assemblyai,elevenlabs,aws,daily]``
  plus ``python-dotenv``.

Run from the repo root::

    source .venv/bin/activate
    python backend/voice-agent/scripts/walking_skeleton.py

The script prints a Daily room URL, then waits for you to press
Enter once you've joined in your browser. Have a 30-second
conversation with the agent and Ctrl-C to clean up.
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

# Layer 1, 2, 3
from app.config import Settings, load_agent_config  # noqa: E402
from app.services import build_stt, build_tts  # noqa: E402
from app.services.factory import resolve_bedrock_model_id  # noqa: E402

# Pipecat 1.1.0
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.aws.llm import AWSBedrockLLMService
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


AGENT_NAME = os.environ.get("SKELETON_AGENT", "chris-claim-status")


async def main() -> None:
    # ── 1. Load .env.skeleton ──────────────────────────────────────────────
    env_file = Path(__file__).resolve().parent / ".env.skeleton"
    if not env_file.exists():
        sys.exit(
            f"Missing {env_file}. Populate it per scripts/README.md "
            "before running."
        )
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
    # The aiohttp session lives for the full script lifetime; the
    # transport keeps its own connection separately.
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
        print("Open this URL in your browser, join with mic enabled,")
        print("then come back here and press Enter to start the bot.\n")
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

        # ── 5. Build STT / TTS via Layer 3; build LLM directly here ────────
        # build_llm doesn't currently accept system_instruction, but
        # Pipecat 1.1.0's Bedrock service requires the system prompt
        # to come via Settings.system_instruction — putting it as a
        # role="system" message in the LLMContext gets silently
        # converted to role="user" and Claude responds to it as user
        # input (skeleton run #1 reproduced this exactly).
        #
        # When this lands in Layer 3 properly, build_llm will accept
        # system_instruction and the spike-only override goes away.
        stt = build_stt(agent)
        tts = build_tts(agent)
        llm = AWSBedrockLLMService(
            aws_region=settings.aws_region,
            settings=AWSBedrockLLMService.Settings(
                model=resolve_bedrock_model_id(agent.llm.model),
                max_tokens=agent.llm.max_tokens,
                temperature=agent.llm.temperature,
                system_instruction=agent.system_prompt,
                enable_prompt_caching=True,
            ),
        )
        logger.info(
            "services_built",
            stt=type(stt).__name__,
            tts=type(tts).__name__,
            llm=type(llm).__name__,
            system_instruction_chars=len(agent.system_prompt),
        )

        # ── 6. Seed the LLM context with the bot's first_message ───────────
        # The system prompt now lives on the service (system_instruction);
        # the messages list seeds the conversation with the bot's
        # opener so when the user responds, Claude has context that
        # the call already started.
        messages: list[dict] = [
            {"role": "assistant", "content": agent.first_message},
        ]
        context = LLMContext(messages)

        # ── 7. Aggregator pair with the locked-in turn machinery ───────────
        # Mirrors v2's project brief:
        #   - MinWordsUserTurnStartStrategy(min_words=3) as the SOLE
        #     start strategy (no VAD start strategy)
        #   - LocalSmartTurnAnalyzerV3 as the stop analyzer
        #   - Silero confidence=0.3, matching AssemblyAI's vad_threshold
        smart_turn = LocalSmartTurnAnalyzerV3(params=SmartTurnParams())
        vad = SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.2,
                # Aligned with AssemblyAI Mode 2 vad_threshold so the
                # two analyzers agree on what is/isn't speech.
                confidence=0.3,
            )
        )
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=vad,
                user_turn_strategies=UserTurnStrategies(
                    start=[MinWordsUserTurnStartStrategy(min_words=3)],
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(
                            turn_analyzer=smart_turn,
                        )
                    ],
                ),
            ),
        )

        # ── 8. Daily transport with the room we just created ───────────────
        # audio_in_sample_rate=8000 forces Daily to deliver 8 kHz audio,
        # matching the Layer 3 AssemblyAI factory's locked-in 8 kHz
        # config (Cosentus's PSTN target). Without this, Daily browser
        # delivers 16 kHz; AssemblyAI interprets the bytes at 8 kHz
        # speed and produces nonsense transcripts (skeleton run #1
        # reproduced this — "hello" got transcribed as "I know").
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

        # ── 9. Wire the pipeline ───────────────────────────────────────────
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

        # ── 10. PipelineTask with metrics enabled ──────────────────────────
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=300,
        )

        # ── 11. Event handlers ─────────────────────────────────────────────
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport_, client):
            logger.info("client_connected", client=client)
            # Speak first_message via TTS directly. We do NOT push
            # LLMRunFrame here because the messages list now starts
            # with first_message as a seeded assistant turn — calling
            # the LLM with no user message would either fail (Bedrock
            # rejects empty user turn) or have Claude improvise the
            # opener again, defeating the point of having
            # first_message at all.
            _ = LLMRunFrame  # silence unused-import
            await task.queue_frames([TTSSpeakFrame(agent.first_message)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport_, client):
            logger.info("client_disconnected, cancelling task")
            await task.cancel()

        # ── 12. Run ────────────────────────────────────────────────────────
        runner = PipelineRunner(handle_sigint=True)
        logger.info("pipeline_running, talk to the bot now")
        await runner.run(task)
        logger.info("pipeline_done")


if __name__ == "__main__":
    asyncio.run(main())
