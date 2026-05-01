# v2 tech debt log

Open issues we know about and have decided to defer. Each entry has
context (what), rationale (why we accepted it), and an exit
condition (when we close it). Order is chronological; numbers are
stable references for code comments.

## Entry 1: Lambda `runtime-config` carries fields v2 doesn't model

**Context.** The cosentus-voice-api Lambda's
`GET /api/agents/:id/runtime-config` endpoint returns several fields
that v2's `AgentConfig` does not model:

- `llm.provider`, `llm.enable_prompt_caching`
- `tts.provider`, `tts.settings.similarity_boost`,
  `tts.settings.style`, `tts.settings.speed`
- `stt.provider`, `stt.language`
- `recording.enabled`, `recording.channels` (the entire
  `recording` object)

v2 silently drops them via `extra='ignore'` on the Pydantic models.

**Why we accepted this.** v2 deliberately collapses several v1
choices that were per-agent on paper but platform-wide in practice.
v1's contract-trace audit (handoff #6) confirmed the engine ignored
these fields at runtime anyway — env vars and Daily defaults won.
v2 names that truth in the type system: per-agent provider/recording
fields are not modeled because they are not honored. Changing the
lambda's response contract to drop the fields is a separate repo's
work and out of scope for v2's greenfield rebuild.

**Cost.** Minor. The lambda continues to ship bytes we throw away;
a future contract change in either direction (lambda removes the
fields, or v2 wires them through) will be a coordinated edit.

**Exit condition.** Close this entry when the cosentus-voice-api
Lambda repo stops sending the dropped fields (or marks them
deprecated and the next contract revision removes them). At that
point we drop `extra='ignore'` from the AgentConfig submodels so
contract drift produces loud failures instead of silent ones.

**Layer / file.** Layer 1 — `backend/voice-agent/app/config/agent_config.py`.

## Entry 2: `AgentConfigMeta.version` → `updated_at_ms` alias

**Context.** The lambda's `_meta.version` field is Aurora's
`updated_at` column rendered as unix milliseconds — not a real
version number. v2's `AgentConfigMeta` exposes the value as
`updated_at_ms` for clarity, with a Pydantic field alias on the
wire name `version` (`Field(default=0, alias="version")`). The
alias is the only reason the data round-trips; without it,
`extra='ignore'` would drop the value silently and every call
would log `updated_at_ms=0`.

**Why we accepted this.** Aligning the lambda contract is part of
the lambda repo's P5 cleanup work (see v1's contract-trace audit).
Decoupled from v2's greenfield rebuild — v2 papers over the
upstream misnaming so internal callers see the honest name today.

**Cost.** Minor. A workaround that papers over upstream misnaming.
Anyone reading `AgentConfigMeta` for the first time has to follow
the alias to understand the wire format.

**Exit condition.** Close when the lambda repo's P5 cleanup
renames the field on the wire. At that point we drop the
`Field(alias="version")` so `updated_at_ms` is just the wire
field name.

**Layer / file.** Layer 1 — `backend/voice-agent/app/config/agent_config.py`
(`AgentConfigMeta`).

## Entry 3: Layer 1 falls back to `os.environ` when no Settings is passed

**Context.** `app/config/agent_config.py` historically reached into
`os.environ` for two values: `VOICE_API_LAMBDA_NAME` and
`AWS_REGION`. With Layer 2 (`Settings`) shipped, the loader now
accepts an optional `Settings` parameter — when provided,
`voice_api_lambda_name` comes from settings; when `None`, the
loader falls back to `os.environ` for backwards compatibility.

The module-level `_LAMBDA_CLIENT` still reads `AWS_REGION` from
`os.environ` because the client is constructed at module import,
*before* any caller could construct or pass a `Settings` instance.

**Updated 2026-05-01 (Layer 2 ships).** `Settings` now exists.
`load_agent_config(agent_id_or_name, settings=...)` is the
recommended path. The env-var fallback remains so existing tests
and any pre-Layer-9 caller still work.

**Why the fallback remains.** Layer 9 (runtime) is the layer that
will construct `Settings` once at process startup and pass it
through every Layer-1 call site. Removing the fallback now would
require each caller to plumb Settings through manually before
Layer 9 lands.

**Cost.** Two `os.environ.get` calls in the loader and one in the
module-level `_LAMBDA_CLIENT` constructor.

**Exit condition.** When Layer 9 (`app/runtime/`) lands and the
runtime layer:

1. Constructs `Settings()` once at process startup; and
2. Passes `settings` to every `load_agent_config` call site.

At that point we drop the `settings is None` branch from
`load_agent_config` and require a `Settings` argument, and we
refactor the module-level `_LAMBDA_CLIENT` to be lazily constructed
on first use using `settings.aws_region`.

**Layer / file.** Layer 1 — `backend/voice-agent/app/config/agent_config.py`.
