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

**Related: `_meta.version` aliasing.** The lambda sends
`_meta.version` (Aurora's `updated_at` as unix millis, named
"version" for historical reasons). v2's `AgentConfigMeta` exposes
this as `updated_at_ms` for clarity, with a Pydantic field alias on
the wire name `version`. The alias is the only reason this value
round-trips; without it, `extra='ignore'` would drop it silently.

**Exit condition.** Close this entry when the cosentus-voice-api
Lambda repo:

1. Stops sending the dropped fields (or marks them deprecated and
   the next contract revision removes them); and
2. Renames `_meta.version` to `_meta.updated_at_ms`.

At that point we drop `extra='ignore'` from the AgentConfig
submodels (so contract drift produces loud failures) and remove the
`Field(alias="version")` from `AgentConfigMeta.updated_at_ms`.

## Entry 2: Layer 1 reads env vars directly; Layer 2 will own settings

**Context.** `app/config/agent_config.py` reads `AWS_REGION` and
`VOICE_API_LAMBDA_NAME` directly from `os.environ` inside the
loader. v2 plans a dedicated settings layer (Layer 2) that owns all
env-and-SSM bootstrap; the loader should consume from it rather
than reach into `os.environ` itself.

**Why we accepted this.** Layer 2 doesn't exist yet. Adding inline
`os.environ.get(...)` calls is a two-line workaround; abstracting
prematurely would violate the "no premature abstraction" principle.

**Exit condition.** When Layer 2 (`app/config/settings.py`) lands,
refactor `load_agent_config` to take a `Settings` instance (or
read from a module-level resolved settings object) instead of
calling `os.environ.get` itself.
