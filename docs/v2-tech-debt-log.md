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

## Entry 4: ~~`_LAMBDA_CLIENT` region captured at import time~~

**Closed:** 2026-05-04 in a Layer 6 follow-up commit alongside
Entry 11. ``app/config/agent_config.py`` now uses a
``_get_lambda_client(settings)`` lazy-init helper: the boto3 lambda
client is constructed on first call using ``settings.aws_region``
(or the env-var fallback when no Settings is supplied — tech debt
Entry 3 still tracks that), then cached at module level for every
subsequent call. ``_invoke_lambda_sync`` takes ``settings`` as a
third positional arg; ``call_writer.py`` passes its caller's
settings through. The function signature's promise — "settings
drives the client" — is now actually true.

---

<details>
<summary>Original entry (kept for history)</summary>

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

</details>

## Entry 5: ~~`disabled_tools` stored as raw CSV string~~

**Closed:** 2026-05-04 in Layer 4 (commit `36ba285`).
`parse_disabled_tools()` in `app/tools/registry.py` implements the
parse-on-construct pattern at the consumer boundary, exactly as the
entry's exit condition specified — whitespace stripping, empty-entry
filtering, and CSV-to-`list[str]` conversion all live at the
consumption site. `Settings.disabled_tools` stays a `str` on the wire
(operator-friendly), and Layer 4 owns the parse.

---

<details>
<summary>Original entry (kept for history)</summary>

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

</details>

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

## Entry 10: Bedrock dynamic-opener requires a synthetic kickoff message

**What.** AWS Bedrock's `ConverseStream` API rejects requests whose
`messages[]` array is empty or starts with an `assistant` turn:

> `ValidationException: A conversation must start with a user message.`

That makes the "dynamic opener" mode (`speak_first=True`,
`first_message=""`) impossible without a workaround — by definition
that mode runs the LLM before any caller has spoken, so there's no
natural user message to seed. v2 papers over this by inserting a
synthetic `{"role": "user", "content": "Hi."}` turn into the context
right before triggering the dynamic-opener `LLMRunFrame()`. Claude
treats it as the caller's "hello" and generates the greeting from
`system_instruction` as designed.

The walking skeleton currently carries the workaround inline; the
production pipeline (Layer 8) will need to do the same.

**Why we shipped it.** Discovery from walking skeleton round 3 —
empty `messages[]` reproduced the Bedrock validation error
immediately. The constraint is a Bedrock-side requirement (Anthropic
Claude's native API has the same rule, even though the Python SDK
documents a different one), so the only way to support
dynamic-opener mode is to seed something. "Hi." is the smallest
non-empty user message we could pick that doesn't bias Claude's
greeting (compared to e.g. "What can you help me with?", which would
prime an answer-shaped opener). Tested — Claude's first turn is
identical to what it generates for inbound PSTN where the user
actually says "hello".

**Cost.** Mild. Three concerns:

1. The synthetic turn appears in `LLMContext.messages` and any
   transcript recording reflecting it. Post-call analysis will see
   `[user: "Hi."]` as the first turn even though no one spoke.
   Acceptable because analyses run on the *transcribed* user audio
   (which is empty for this turn) — but anyone reading the raw LLM
   context dump will see the seeded value.
2. Tiny additional input tokens (~5) on every dynamic-opener call.
3. Couples Layer 8 / skeleton to a Bedrock-specific quirk. If we
   ever swap to OpenAI or another provider that accepts
   system-only contexts, the workaround becomes dead code.

**Exit condition.** Either (a) AWS adds support for empty / system-
only `messages[]` to ConverseStream — unlikely on their roadmap as
of now — or (b) v2 switches its primary LLM to a provider whose API
supports system-only first turns, at which point the kickoff seed
becomes conditional on `isinstance(llm, AWSBedrockLLMService)` or is
deleted entirely. Until then, Layer 8 just inherits the pattern.

**Layer / file.** Layer 8 (future — pipeline builder); currently in
`backend/voice-agent/scripts/walking_skeleton.py` (research artifact
only).

## Entry 11: ~~`_BEDROCK_CLIENT` region captured at import time~~

**Closed:** 2026-05-04 in a Layer 6 follow-up commit.
``app/persistence/post_call.py`` now uses a ``_get_bedrock_client(settings)``
lazy-init helper: the boto3 ``bedrock-runtime`` client is constructed
on first call using ``settings.aws_region``, then cached at module
level for subsequent calls. The function signature's promise — "the
settings parameter configures the client" — is now actually true.
Layer 1's ``_LAMBDA_CLIENT`` (Entry 4) gets the same treatment in a
parallel follow-up.

---

<details>
<summary>Original entry (kept for history)</summary>

**What.** `app/persistence/post_call.py` constructs the boto3
``bedrock-runtime`` client at module import using
``os.environ.get("AWS_REGION", "us-east-1")``. The
:func:`run_post_call_analyses` function accepts a ``settings:
Settings`` parameter, but ``settings.aws_region`` is **not** used to
configure the client — the region is bound before any ``Settings``
instance exists. Same anti-pattern as Entry 4 for Layer 1's
``_LAMBDA_CLIENT``, repeating in Layer 6.

**Why we shipped it.** Module-level boto3 clients are the AWS-
documented multithreading pattern (one shared thread-safe client
across worker threads), and that requires the region at
construction. Layer 6 imports cleanly before any Layer 9 runtime
can construct ``Settings``. Avoiding the import-time read would
require inverting the boot sequence or a lazy-init pattern, neither
of which Layer 6 was the right scope for.

In practice the two values agree at steady state — both
``settings.aws_region`` and the import-time env read pull from
``AWS_REGION`` — so production calls land on the right region. The
debt is the hidden coupling: the function signature suggests the
settings drive the client, but they don't.

**Cost.** Two concrete issues:

1. **Test isolation.** A test that overrides
   ``Settings(aws_region="us-west-2")`` without also setting the
   env var will still hit ``us-east-1`` (or whatever was in the
   env at import). The settings are silently ignored at the boto3
   layer.
2. **Layer-boundary smell.** Layer 6 advertises ``settings`` as
   the configuration source but actually consumes ``os.environ``
   directly. A future maintainer who flips
   ``Settings.aws_region`` will be surprised when the change
   doesn't take effect on Bedrock calls.

Operationally, AWS region is a deploy-time concern and never
changes during a Fargate task's life, so this is shape /
cleanliness debt, not a runtime risk.

**Exit condition.** Lazy-init pattern: defer construction to the
first call, using ``settings.aws_region`` from the
``run_post_call_analyses`` parameter. Sketch::

    _BEDROCK_CLIENT: BedrockClient | None = None
    _BEDROCK_CLIENT_LOCK = asyncio.Lock()

    async def _get_bedrock_client(settings: Settings) -> BedrockClient:
        global _BEDROCK_CLIENT
        if _BEDROCK_CLIENT is None:
            async with _BEDROCK_CLIENT_LOCK:
                if _BEDROCK_CLIENT is None:
                    _BEDROCK_CLIENT = boto3.session.Session().client(
                        "bedrock-runtime",
                        region_name=settings.aws_region,
                        config=BotoConfig(...),
                    )
        return _BEDROCK_CLIENT

The client stays module-shared (constructed once, reused across
calls); only the *binding* moment shifts. First call pays a
~1 ms construction cost. The same pattern should land on
``_LAMBDA_CLIENT`` in Layer 1 (Entry 4) at the same time so the
two clients have the same lifecycle story. Layer 9 (runtime) is
the natural place to apply both.

**Layer / file.** Layer 6 —
``backend/voice-agent/app/persistence/post_call.py`` module-level
``_BEDROCK_CLIENT``. Cross-references Entry 4.

</details>

## Entry 12: ~~`PipelineRunner` signal handlers conflict under concurrent calls~~

**Closed:** 2026-05-04 in Layer 8 (commit forthcoming).
``app/bot/bot.py::run_bot`` constructs ``PipelineRunner(handle_sigint=False,
handle_sigterm=False)`` so concurrent calls don't clobber each other's
signal handlers. Layer 9 (when it lands) will install a single
process-level handler that iterates active sessions and cancels each
task. v2 ships fixed; v1 / AWS sample ship the bug.

---

<details>
<summary>Original entry (kept for history)</summary>

**What.** Default ``PipelineRunner(handle_sigint=True)`` registers a
process-wide SIGINT handler via ``loop.add_signal_handler``, which
**replaces** (not appends) any prior handler. With N concurrent
calls in one Fargate process (``MAX_CONCURRENT_CALLS=4`` in
production), only the most-recently-constructed runner's tasks
receive SIGINT — earlier calls are leaked.

Verified empirically by reading
``.venv/lib/python3.12/site-packages/pipecat/pipeline/runner.py``
lines 33–62 (constructor) and 110–117 (signal-handler setup).
``add_signal_handler`` semantics confirmed via Python docs.

**Why we ship fixed.** The fix is straightforward: pass
``handle_sigint=False`` and ``handle_sigterm=False`` at runner
construction. Layer 9 (runtime) installs a single process-level
SIGTERM handler that iterates ``pipeline_manager.active_sessions``
and cancels each ``asyncio.Task``. Cost is one extra discipline
rule documented at the runner-construction site in ``run_bot``.

**Original bug source.** AWS sample-voice-agent
(``aws-solutions-library-samples/sample-voice-agent``) constructs
``PipelineRunner()`` with defaults at every call. v1 inherited the
pattern from the fork. v2 deviates explicitly and documents the
deviation inline.

**Empirical verification.** Done as part of Layer 8 verification
brief (see commits ``aecbd6d`` and ``78af92a`` lazy-init follow-ups
for context on similar concurrent-state issues we'd already
caught + fixed).

**Layer / file.** Layer 8 — ``backend/voice-agent/app/bot/bot.py``
``PipelineRunner`` construction site. Layer 9 (future) — process-
level signal handler.

</details>

## Entry 13: AssemblyAI WebSocket 1008 under burst load is documented vendor behavior, not a defect

**Context.** Layer 9.5 scale test surfaced ~80 log lines of
AssemblyAI WebSocket close 1008 (policy violation) when N=6
concurrent ``/start`` requests were fired in <1 second by the
synthetic load harness (``backend/voice-agent/scripts/scale_test.py``).
At first glance this looked like a vendor blocker — calls were
accepted by the engine but the AssemblyAI side closed the WS
mid-handshake.

**Root cause.** AssemblyAI Universal-Streaming v3 has *unlimited*
concurrent streams; what is rate-limited is *new streams per
minute*. The default baseline is 100 new streams/min, with
auto-scaling that grows the baseline by 10 % every 60 s once
utilization passes 70 %. Our harness opens N WebSockets in
milliseconds, so a 6-stream burst is treated as 6 toward the
per-minute new-stream cap and can briefly exceed the baseline
before auto-scaling reacts. Production traffic does not look
like this — real PSTN calls arrive paced over wall-clock time,
not in synthetic millisecond bursts.

**Why we accepted this (no code change).** The behavior is
documented vendor policy, not a defect in v2. Mitigation lives at
the account-configuration layer (raise the baseline before
production launch, sized to expected peak), not in the engine.
Adding artificial pacing to ``/start`` to "smooth" the harness
would mask the signal we *want* the harness to expose: that
vendor onboarding (paid tier, baseline raise, BAA) is its own
launch-readiness workstream.

**Verification.**

- AssemblyAI docs explicit on the model:
  ``https://www.assemblyai.com/docs/concepts/concurrency-limit``.
  "If you briefly exceed your current limit, new connections may
  return 1008 until it scales; baselines can be raised on
  request."
- Endpoint version verified as v3 Universal-Streaming, not legacy
  v2-realtime, via three independent checks:
  (1) Pipecat default
  ``api_endpoint_base_url="wss://streaming.assemblyai.com/v3/ws"``
  in ``.venv/.../pipecat/services/assemblyai/stt.py``;
  (2) ``backend/voice-agent/app/services/factory.py`` does not
  pass ``api_endpoint_base_url=`` and therefore accepts the
  default; (3) Pipecat's service refuses to start with
  ``vad_force_turn_endpoint=False`` on any non-Universal-3-Pro
  model — our Mode-2 + ``u3-rt-pro`` combination only resolves on
  the v3 endpoint, so a silent regression to v2-realtime is
  structurally impossible without a code change here.

**Effective wire URL on every call:**
``wss://streaming.assemblyai.com/v3/ws?sample_rate=8000&encoding=pcm_s16le&speech_model=u3-rt-pro&vad_threshold=0.3``.

**Cost.** Zero in v2 code. Cost is in vendor onboarding
(account tier verification, baseline raise request, BAA
follow-up), tracked below.

**Exit condition — production-readiness checklist.**

- [ ] Cosentus AssemblyAI account confirmed paid tier (billing
  dashboard).
- [ ] Baseline new-streams/min raised to ~2× expected peak new
  calls/min for headroom (email AssemblyAI Sales).
- [ ] BAA status confirmed with a human AssemblyAI rep (still
  open since the AI-sales-agent reply was non-authoritative).
- [x] v3 Universal-Streaming endpoint confirmed (2026-05-08,
  during Layer 10 VERIFICATION-0).

Close this entry once all three vendor-side items land and the
Layer 9.5 burst-test note is reframed in the scale-test report
from "vendor unknown" to "expected vendor behavior, mitigated by
baseline raise."

**Status.** Open, non-blocking for Layer 10 / Layer 11.

**Severity.** Low — vendor configuration, not code. Production
traffic patterns will not reproduce the harness burst pattern
that triggered the 1008s.

**Layer / file.** Vendor / account-side. No v2 code owns this;
referenced by the Layer 9.5 scale-test report and (future)
``docs/runbooks/production-launch-checklist.md``.

## Entry 14: Engine auth uses non-standard `X-API-Key` header instead of `Authorization: Bearer`

**Context.** ``app/runner/server.py::_check_api_key`` reads the
caller's API key from a custom ``X-API-Key`` header. The web
ecosystem (RFC 6750, every API gateway, every SDK client) defaults
to ``Authorization: Bearer <token>``. The mismatch wasn't obvious
until the first staging smoke test, where the deploy guide
defaulted to ``Authorization: Bearer`` and every ``/status``
request returned 401 — value was identical, header name wasn't.

**Why we accepted this.** v2 inherited the header name from v1
(forked AWS sample). Changing it now would require coordinating
with every caller (frontend dashboard, batch dispatcher, any
ad-hoc internal tooling). Low-impact bug-but-not-broken — once
the convention is known, everyone uses the right header.

**Cost.** One-time documentation friction per new operator. Will
show up again every time someone tries `curl` against the engine
for the first time.

**Exit condition.** Phase 5 follow-up after production launch:

1. Make ``_check_api_key`` accept BOTH headers, preferring
   ``Authorization: Bearer``. New callers use the canonical form;
   old callers keep working through deprecation.
2. Deprecate ``X-API-Key`` in a single release. Add a warning
   header to responses where it was used.
3. After one full call cycle (or 30 days, whichever is longer)
   without seeing ``X-API-Key`` in production logs, remove the
   compat shim.

While we're in there, also swap the comparison from ``==`` to
``hmac.compare_digest`` to eliminate the timing side channel.
Single string equality on a secret is a textbook footgun; the
exploit is impractical in our threat model but it's a 1-line fix
to remove.

**Severity.** Low — operator friction only, no real-world risk.

**Status.** Open, non-blocking.

**Layer / file.** Layer 9 — ``backend/voice-agent/app/runner/server.py``
(``_check_api_key`` and call sites).

## Entry 15: Silent fallback to "no-auth" when API-key secret load fails

**Context.** ``app/runner/server.py::_load_api_key`` catches every
exception from ``SecretsManager.get_secret_value`` and returns
``""``. Empty-string is then treated by ``_check_api_key`` as
"local-dev mode, auth disabled" — so any request passes without
header.

The intent was a clean local-dev story: if you didn't set
``API_KEY_SECRET_ARN``, you run without auth. The implementation
conflates two failure modes:

* "no ARN configured" (local dev) — should disable auth.
* "ARN configured, but Secrets Manager call failed" (prod IAM
  drift, network blip, secret deleted) — should fail loud, not
  silently disable auth.

In the second case, the engine boots and starts serving
**authenticated routes as if no auth were required**. We saw this
explicitly in the Layer 11 staging deploy code review when
auditing the auth path.

**Why we accepted this for the initial cutover.** Discovered too
late in the Layer 11 deploy cycle to rewire pre-staging-launch.
Risk is mitigated for now because: (a) staging IAM is granting the
role correctly today (verified by smoke test), (b) Secrets
Manager has a published 99.99% availability SLA, (c) the secret
isn't deleted by any automation.

**Cost.** A latent production-security regression if any of those
conditions break.

**Exit condition.** Phase 5 follow-up:

1. Split the two failure modes in ``_load_api_key``:
   * ``arn == ""`` ⇒ ``return ""`` (local-dev only-no-auth path,
     unchanged).
   * ``arn != ""`` and Secrets Manager call fails ⇒ **raise** so
     the task crashes at boot. ECS restarts the task; if the
     condition persists, the deploy circuit-breaker rolls back.
2. Add a startup-time assertion: in production environments
   (``settings.environment != "production"`` is for the
   non-prod path, ``"production"`` and ``"prod"`` mean we MUST
   have a non-empty key loaded). Fail-fast if the empty-string
   path is taken in those envs.
3. Add a smoke-test step to the future runbook: hit
   ``/status`` without auth from outside the VPC after every
   deploy. Expect 401. If 200, the bug is back.

**Severity.** Medium — quiet defaults that disable security are
worse than loud failures.

**Status.** Open, non-blocking for Layer 11 deploy / Wave 6 mock
load test. Should land before any real production traffic flows.

**Layer / file.** Layer 9 — ``backend/voice-agent/app/runner/server.py``
(``_load_api_key`` lines 301–339).
