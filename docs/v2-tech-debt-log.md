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

**What.** `load_agent_config` and the module-level `_LAMBDA_CLIENT`
both read `os.environ` directly when no `Settings` object is
provided. As of Layer 2, `Settings` exists, but Layers 3–8 are
written before Layer 9 (runtime). Until Layer 9 lands, callers
either pass `Settings` explicitly or fall back to env reads.

**Why we shipped it.** Layer 9 (runtime) is the layer that will
construct `Settings` once at process startup and pass it through
every Layer-1 call site. Removing the fallback now would require
each Layer 3–8 consumer to plumb a Settings instance through
itself before Layer 9 has a place to construct one. Keeping the
fallback lets each layer ship cleanly and Layer 9 wire everything
up at the end.

**Cost.** Two `os.environ.get` paths in `load_agent_config` (the
`settings is None` branch) plus one in the module-level
`_LAMBDA_CLIENT` constructor (covered in entry 4 too — the AWS
region specifically). Mild duplication of the "where do values
come from" question across two layers.

**Exit condition.** When Layer 9 (`app/runtime/`) lands and:

1. Constructs `Settings()` once at process startup; and
2. Passes `settings` to every `load_agent_config` call site.

At that point we drop the `settings is None` branch from
`load_agent_config` and require a `Settings` argument. The
module-level `_LAMBDA_CLIENT`'s region migration is tracked
separately in entry 4.

**Layer / file.** Layer 1 — `backend/voice-agent/app/config/agent_config.py`.

## Entry 4: `_LAMBDA_CLIENT` region captured at import time

**What.** `agent_config.py` constructs the boto3 lambda client at
module import using `os.environ.get("AWS_REGION", "us-east-1")`.
This happens before any `Settings` object exists.

**Why we shipped it.** A module-level client is the AWS-documented
multithreading pattern (one shared thread-safe client across
worker threads), and that requires the region at construction. The
import order in v2 — Layer 1 modules load before any Layer 9
runtime can run — means there is no `Settings` instance available
at import time without inverting the boot sequence.

**Cost.** Region cannot be changed without a process restart.
Theoretical debt: AWS region is set once at Fargate deploy time
and never changes during the life of a task, so this is more a
shape/cleanliness concern than an operational one.

**Revisit when.** Layer 9 wires up. Two options at that point:

1. Lazily construct the lambda client on first use using
   `settings.aws_region`. The client is then still module-shared
   (created once, kept across calls), just deferred. Trade-off:
   first call pays the construction cost (~1ms).
2. Accept that AWS region is a deploy-time concern and document
   the import-time read as intentional. No code change.

**Layer / file.** Layer 1 —
`backend/voice-agent/app/config/agent_config.py` module-level
`_LAMBDA_CLIENT`.

## Entry 5: `disabled_tools` stored as raw CSV string

**What.** `Settings.disabled_tools` is typed as `str`, not
`list[str]`. Pydantic Settings supports JSON-encoded list strings
from env vars (e.g. `'["a","b"]'` → `["a", "b"]`) but not bare
CSVs without a custom validator. The operator-friendly format is
`"tool1,tool2"`, and `Settings` stores it verbatim.

**Why we shipped it.** A custom field validator is premature
complexity for a field whose only consumer is Layer 4 (tools).
Splitting the string at the consumption site is a one-liner.

**Cost.** `split(",")` lives in Layer 4 rather than centralized
in `Settings`. Mild concern that nothing forces operators to use
the exact format the consumer expects (whitespace, empty entries,
unknown tool names all pass `Settings` validation untouched).

**Revisit when.** Layer 4 ships. If the parsing turns out to need
non-trivial work (whitespace stripping, empty-entry filtering,
validation against the registered tool catalog), promote to a
Pydantic field validator on `Settings.disabled_tools` so the
parsed `list[str]` is the public API and the wire format stays
human-friendly CSV.

**Layer / file.** Layer 2 —
`backend/voice-agent/app/config/settings.py` (`disabled_tools`
field).

## Entry 6: `audioop` deprecation filter in pytest config

**What.** `pyproject.toml` carries a `filterwarnings` entry
suppressing `'audioop' is deprecated` `DeprecationWarning`. The
filter exists because Pipecat 1.1.0 still imports the stdlib
`audioop` module, which is deprecated in Python 3.12 and **removed
in Python 3.13**. With `filterwarnings = ["error"]` the warning was
breaking pytest collection at Pipecat-import time.

**Why we shipped it.** Pipecat upstream issue #709 tracks the
migration off `audioop`. As of Pipecat 1.1.0 (released
2026-04-27) the migration hasn't shipped. Adding a narrow filter
unblocks our tests on Python 3.12 without modifying upstream code
or pinning a dev fork.

**Cost.** The filter masks the symptom but doesn't fix the
disease. Hard upgrade ceiling at Python 3.12 — moving to 3.13 will
fail with `ImportError`, not a warning. We're locked to 3.12
until Pipecat ships an `audioop` migration or we add `audioop-lts`
as an explicit dependency to provide the module ourselves.

**Exit condition.** Close when either:

1. Pipecat ships a version that doesn't import `audioop` (track
   upstream issue #709); or
2. We decide to upgrade Python past 3.12, in which case add
   `audioop-lts` to `pyproject.toml` dependencies.

Drop the filter from `pyproject.toml` when either path lands.

**Layer / file.** Layer 3 — `pyproject.toml` `[tool.pytest.ini_options].filterwarnings` list.

## Entry 7: API keys read directly from `os.environ` in Layer 3

**What.** `app/services/factory.py`'s `build_stt` and `build_tts`
read `ASSEMBLYAI_API_KEY` and `ELEVENLABS_API_KEY` directly via
`os.environ.get()`. This bypasses the Layer 2 `Settings` boundary
that's supposed to be the sole source of truth for environment
configuration.

**Why we shipped it.** API keys arrive in env via `secrets_loader`
at boot — that's Layer 6, not yet built. Pipecat's services expect
the keys via constructor arguments (`api_key=...`). Plumbing the
keys through `Settings` would require either a separate `Secrets`
object or extending `Settings` to model API keys; both are
premature before Layer 6 establishes the secrets boundary.

**Cost.** Two `os.environ.get` reads in Layer 3 (one per service)
violate the "Layer 2 owns env" principle established at Layer 2.
If env reads spread further during Layers 4–8, the boundary erodes
and Layer 9's settings-everywhere refactor gets larger.

**Exit condition.** Layer 6 (`secrets_loader`) ships and we decide
the secrets boundary. Three options on the table at that point:

1. API keys live as fields on `Settings` (loaded from env, which
   `secrets_loader` populated at boot).
2. API keys live on a separate `Secrets` object that `secrets_loader`
   constructs and the runtime layer passes to factories.
3. The current pattern is sanctioned: `secrets_loader` writes to
   env at boot; services read env. (Less clean, but matches v1's
   operational model and Pipecat's own assumption that API keys
   come from env.)

If (1) or (2) is chosen, drop the `os.environ.get` reads from
`build_stt` / `build_tts`.

**Layer / file.** Layer 3 — `app/services/factory.py`
(`build_stt`, `build_tts`).

## Entry 8: TTS model has two disagreeing defaults

**What.** `AgentConfig.tts.model` defaults to `"eleven_turbo_v2_5"`
in Layer 1's Pydantic model. The factory's `_ELEVENLABS_DEFAULT_MODEL`
is `"eleven_flash_v2_5"`. The Layer 1 default kicks in when the
lambda omits the `tts.model` field entirely; the factory fallback
kicks in when the field is present but empty. Two defaults for the
same hole.

**Why we shipped it.** Defense in depth. The lambda *should* always
populate `tts.model` from Aurora, but if it doesn't, both layers
have a fallback. The factory's `flash_v2_5` matches v1's
production-running value — that's the deployment-safe default.
Layer 1's `turbo_v2_5` is what agents created in v2 will use as
their starting model.

**Cost.** Mild architectural smell. Reading the code, it's not
immediately obvious which default applies in which scenario, so a
future maintainer might "fix" the disagreement and break one of the
safety nets.

**Exit condition.** Layer 12 (prod cutover) when we have production
logs confirming the lambda always sends `tts.model`. At that point
one of the two defaults can be deleted (Layer 1's, since it would
be unreachable) or both harmonized to the same value.

**Layer / file.** Layer 1 — `app/config/agent_config.py`
(`TTSConfig.model` default); Layer 3 — `app/services/factory.py`
(`_ELEVENLABS_DEFAULT_MODEL`).

## Entry 9: `DailyParams` must explicitly set `audio_in_sample_rate=8000`

**What.** Daily browser-based audio is 16 kHz by default. v2's
AssemblyAI STT is hardcoded to 8 kHz (PSTN-correct via Layer 3
locked-in choice). Without explicit `audio_in_sample_rate=8000`
on `DailyParams`, browser audio mismatches the STT sample rate
and produces garbage transcripts ("hello" → "I know", "could you
hear me?" → "could you go away?" — both reproduced in the walking
skeleton's run #1).

**Why we shipped it.** Discovery during walking-skeleton testing.
Layer 8 (pipeline builder) hasn't been written yet, so the fix
lands when `DailyParams` is constructed there. Production PSTN
calls deliver 8 kHz audio natively, so prod isn't actually broken
today — but any browser-based testing, dev calls, or hybrid
scenarios will hit this without the explicit setting.

**Cost.** Easy to forget. If Layer 8's brief doesn't explicitly
call this out, walking-skeleton-style failures repeat on the
first dev test. Entry exists primarily to make sure Layer 8
doesn't repeat the discovery.

**Exit condition.** Layer 8 ships with
`DailyParams(audio_in_sample_rate=8000, ...)` and the explicit
setting is covered by a pipeline test. At that point this entry
can be marked closed.

**Layer / file.** Layer 8 — pipeline builder, `DailyTransport` /
`DailyParams` construction.
