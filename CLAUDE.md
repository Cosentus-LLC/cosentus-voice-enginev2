# Agent instructions — `voice-engine-v2`

This is the Cosentus voice-platform **engine** — a Python **Pipecat** pipeline on AWS Fargate
that runs live phone calls (Daily PSTN · AssemblyAI STT · AWS Bedrock Claude LLM · ElevenLabs TTS).
It is a **HIPAA-adjacent medical-billing system that handles LIVE patient calls** — be careful.

## What this repo is for

A working copy where we **modernize the engine**, incrementally. Scope is whatever each issue
defines — **not** limited to a fixed list. Examples of the kinds of work expected: an
**evaluation loop** (measure call quality), **structured conversation flows** (Pipecat Flows),
**structured AI outputs** (BAML), **LLM cost/latency** work (per-turn model routing, Bedrock
prompt caching, context trimming, streaming), the payer-knowledge / IVR-path wiring, and the
security/correctness fixes from the review.

Do big changes **incrementally**: one issue / one PR at a time, **tests green** before each PR.
Changes are **additive** — the audio foundation (WebRTC/Daily, one-container-per-call, smart
turn detection, tuned interrupt settings) is correct; build the AI-product layer on top of it.

## Hard rules

- **Never deploy.** No `cdk deploy`, no Docker build/push to ECR, no `aws` CLI. Deploys
  (Fargate staging/prod) are manual and human-only.
- **Never touch production or v1.** This is live-call infrastructure — a bad change can break
  real patient calls. You build + test only; a human deploys to **staging** first.
- **Do NOT widen the Pipecat pin.** `pipecat-ai==1.1.0` is exact on purpose — 1.2.x breaks
  ElevenLabs TTS (1008 "voice_settings" error). Do not change it without re-validating TTS
  end-to-end; the Dockerfile installs `--frozen` from `uv.lock`.
- **The engine reads agent config from the API over HTTP** (the `runtime-config` shape) — it
  **never** touches the database directly, and it must keep consuming the API's existing shape.
  Coordinate before assuming any contract change.
- **Don't add dependencies, change infra/CDK, or touch secrets/env vars without asking.**
- **Route secrets/config through the settings object** — don't read env vars directly in new code.
- If a change is risky, large, touches the live audio pipeline, or could affect a real call —
  **stop and ask first.**

## Workflow

**When you are given an issue number (e.g. "do #1" / "implement #1"), follow this routine BY
DEFAULT — without being told each time:**

1. Read the GitHub issue: `gh issue view <number>`. Its acceptance criteria + Context/Architecture
   sections are the spec — use them, don't guess.
2. **Post a DETAILED implementation plan** as an issue comment and **wait for approval before
   writing any code.** It must meet the **Plan requirements** below — vague plans are rejected.
3. Implement, scoped to the issue. When introducing a new pattern (Pipecat Flows, BAML, a
   model-routing layer), apply it consistently and explain it in the PR.
4. Run the **gate** (below) until green.
5. Open a PR; the body **must** contain `Closes #<n>`; explain what changed and why.

## Issue lifecycle (labels)

The `status:` label is the single signal for "what's happening here." Keep it in sync:

- **Only start issues labelled `status: ready`.** If it's `status: blocked` or `needs-arch`,
  **do NOT write code** — comment what's blocking it (the dependency issue #, missing access, or
  unfinished spec) and **stop.** Never vaguely "start anyway."
- **On pickup** (after your plan is approved): `gh issue edit <n> --remove-label "status: ready" --add-label "status: in-progress"`.
- **PR body MUST contain `Closes #<n>`** (auto-closes the issue + advances the status automation).
- **If you discover a blocker mid-work**, add `status: blocked`, comment exactly what's blocking,
  and stop — no half-finished PR.

## Blockers must be explicit (in every issue + when you discover one)

A `🚫 Blockers` note must spell out, **with issue numbers**:
1. **Dependency blockers** — "Blocked by #N — do NOT start until #N is shipped and merged."
2. **File-conflict blockers** — if it edits a file another open issue also edits (e.g. `bot.py`,
   `factory.py`, `agent_config.py`), say so: "edits `bot.py` → serial with #X/#Y; don't run in
   parallel or they merge-conflict." Mark which issues are a one-at-a-time chain vs parallel-safe.
3. **Interlinks** — related/upstream/downstream issues, so the chain is visible from the issue alone.

## Issues are self-contained

Implement from the **issue body** — it carries the full spec (files, approach, edge cases, named
tests). If an issue links `MODERNIZATION-PLAN.md §X`, read that section for extra rationale, but
don't depend on it: the body is the source of truth.

## Plan requirements (the plan MUST be concrete — not vague)

Read the relevant code FIRST, then write a plan a reviewer could follow line-by-line. Include **all**:
1. **Files** — exact paths you will add / modify / delete.
2. **Per-file changes** — specific functions, classes, and **signatures** by name (not "add types").
3. **Behavior to preserve** — list each behavior **and how** you keep it.
4. **Edge cases** — interruptions, hold music, transfers, missing data, error paths.
5. **Tests** — the specific `pytest` cases you'll add or keep green, **named**.
6. **Risks / open questions** — anything ambiguous. **Ask before coding; do not guess.**
7. **Verification** — the exact commands (the gate below).

**Banned:** vague verbs with no specifics ("clean up", "improve", "handle properly", "preserve
behavior" *without saying how*). **If you can't be specific, you don't understand the code yet.**

## Branch naming

`<prefix>/issue-<number>-<short-kebab-summary>` — prefix by type: `feat` (new), `fix` (bug),
`chore` (tooling/config), `refactor`. Summary 2–4 words, lowercase, hyphenated; always include
the issue number. e.g. `feat/issue-5-eval-loop-spike`, `fix/issue-1-auth-fail-closed`.

## Setup / gate

- **Install:** `uv sync --extra dev --python 3.12` — uv manages Python; **use 3.12**
  (Pipecat 1.1.0 imports `audioop`, removed in 3.13+, so 3.13/3.14 break it). If `uv` is missing:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **Gate before every PR — all must pass:** `uv run ruff check .` · `uv run ruff format --check .` · `uv run pytest`
- **Ruff excludes pre-existing operational paths** (`backend/voice-agent/scripts`, the infra stub
  `infrastructure/lambdas/voice-api-stub`) — see `pyproject.toml [tool.ruff]`. So the gate is green
  on app + test code. **If the gate ever fails on a file you did NOT touch, STOP and flag it — do
  not broaden your PR to fix unrelated files.**
- Tests live in `backend/voice-agent/tests`. **Add or update tests for anything you change.** Keep
  `pytest`'s `filterwarnings = error` green (warnings fail the suite).
