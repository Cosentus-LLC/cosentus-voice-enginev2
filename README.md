# cosentus-voice-engine

The Cosentus voice agent platform: AI voice agents that handle phone calls
for medical billing — outbound to insurance companies for claim status,
inbound for billing support. The engine is a [Pipecat](https://github.com/pipecat-ai/pipecat)
pipeline running on AWS Fargate, fronted by Daily.co PSTN, with
AssemblyAI Universal-3 Pro Streaming for STT, ElevenLabs for TTS, and
AWS Bedrock Claude for LLM. Per-agent configuration lives in Aurora
Postgres; recordings land in S3 with SSE-KMS.

## The four layers of the platform

- **Data** — Aurora Postgres holds agent configs, call records, batch
  state. Owned by a sibling repo's API Lambda; this engine reads via
  HTTP, never directly.
- **API** — A separate Lambda exposes CRUD over agents, batches, calls.
  Not in this repo.
- **Engine** — *This repo.* The Pipecat pipeline that runs each call,
  the HTTP entrypoint Fargate exposes behind an internet-facing ALB
  (which Daily's webhooks hit directly), and the CDK that ships it all.
- **Telephony** — Daily.co provides PSTN, SIP, recording, and the
  WebRTC transport into Pipecat.

## Status

Built fresh in May 2026 to replace `cosentus-voice-engine` (the AWS
sample-voice-agent fork). The v1 still runs production today. v2
exists to shed the upstream fork's cruft (SageMaker stack, A2A
capability agents, dual local/ECS pipelines) and rebuild around the
choices that actually survived contact with production calls.

For the full system spec see [`docs/architecture/overview.md`](docs/architecture/overview.md).
For the layer-by-layer rebuild plan see [`docs/architecture/migration-from-v1.md`](docs/architecture/migration-from-v1.md).

## Dev setup

See [`SETUP.md`](SETUP.md). Quick orientation:

- Python: ≥ 3.11 minimum, **3.12 recommended** (matches Pipecat's own
  guidance).
- Package manager: **[uv](https://docs.astral.sh/uv/)** is the
  primary path — Pipecat's docs and CLI now recommend it for speed
  and lockfile semantics. `pip` still works as a fallback.
- Install with `uv sync --extra dev --python 3.12` (the full gate
  lives in `SETUP.md`).
