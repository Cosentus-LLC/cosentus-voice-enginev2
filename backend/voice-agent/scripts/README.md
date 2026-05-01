# Walking skeleton

Walking-skeleton spike. Not production. Used to validate Layers
1–3 + Pipecat 1.1.0 composition (notably the new
`LLMContextAggregatorPair` / `UserTurnStrategies` shape) on a real
Daily call before building Layers 4–12. Delete when Layer 8 ships.

## Files

- `walking_skeleton.py` — the script. ~200 lines. Creates a Daily
  room, loads `chris-claim-status` from the real API Lambda, builds
  STT/TTS/LLM via Layer 3, wires the new aggregator pair with
  `MinWordsUserTurnStartStrategy(min_words=3)` and
  `LocalSmartTurnAnalyzerV3`, runs the pipeline.
- `.env.skeleton` — env file consumed by the script. **Gitignored.**
  Populate with the values below before running.
- This README.

## Prerequisites

1. AWS credentials (default profile / SSO) with permission to
   invoke `medcloud-voice-api:live` and read the API-key Secrets
   Manager secret. Boto3's default credential chain is used; no
   keys go in `.env.skeleton`.
2. `pipecat-ai[assemblyai,elevenlabs,aws,daily]` plus
   `python-dotenv` installed in the repo's venv. Already in
   `pyproject.toml` after Layer 3.

## `.env.skeleton` template

```dotenv
# Layer 2 Settings — required
VOICE_API_LAMBDA_NAME=medcloud-voice-api:live
API_KEY_SECRET_ARN=arn:aws:secretsmanager:us-east-1:825269749545:secret:SecretsApiKeySecretCEC8F618-MqoY7x3uc0N3-dOqjuy
AWS_REGION=us-east-1

# Vendor API keys — pulled from Secrets Manager into the env so the
# Layer 3 factory's os.environ.get() reads find them. Dump:
#
#   aws secretsmanager get-secret-value \
#     --secret-id $API_KEY_SECRET_ARN \
#     --query 'SecretString' --output text
#
# Then paste the three values below.
DAILY_API_KEY=…
ASSEMBLYAI_API_KEY=…
ELEVENLABS_API_KEY=…
```

## Run

```bash
source .venv/bin/activate
python backend/voice-agent/scripts/walking_skeleton.py
```

Script flow:

1. Loads `.env.skeleton`.
2. Creates a one-off Daily room (60-min expiry).
3. Prints the room URL and waits for you to press Enter.
4. Loads `chris-claim-status` config from the lambda, builds the
   pipeline, and starts.
5. The bot runs until you Ctrl-C or hang up the browser.

Open the URL, join with mic enabled, press Enter back in the
terminal, and have a conversation.

## What this validates

In order of importance:

1. The new Pipecat 1.0+ aggregator pattern composes with our
   `AgentConfig` (`LLMContextAggregatorPair` + `UserTurnStrategies`).
2. Layer 1's `load_agent_config` flows into Layer 3's factory
   without surprises.
3. AssemblyAI Mode 2 incantation behaves correctly on a real call
   (no chopped numbers, no false interrupts).
4. Bedrock prompt caching works in streaming mode (we hardcoded
   `enable_prompt_caching=True` in Layer 3 without ever testing).
5. `LocalSmartTurnAnalyzerV3` works with Daily browser audio
   (16 kHz; the 8 kHz PSTN concern is tracked separately).
6. End-to-end latency is subjectively acceptable.

Findings are recorded in the spike commit message and follow-up
notes; see git log for the spike(skeleton) commit.

## Findings

See the commit message for `spike(skeleton)`. Once Layer 8 ships,
this directory and all its files are deleted.
