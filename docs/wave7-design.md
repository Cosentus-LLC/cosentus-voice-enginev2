# Wave 7 — Concurrent Real-Audio Validation (design)

**Status:** DESIGN ONLY — not yet run. Awaiting go-ahead + ElevenLabs
quota top-up.

**Author:** 2026-06-01.

## Purpose

The last gate before vendor commitments (ElevenLabs Enterprise ~$22k/yr,
Daily HIPAA ~$500/mo). We have proven:

- **Wave 6** — infrastructure scales under load (autoscaling, capacity
  gates, crash recovery, task recycling) using *fail-fast mock* calls
  (no audio, ~1.7s lifetime).
- **2026-06-01 single real call** — one full real-audio conversation
  works end-to-end (STT → Bedrock → ElevenLabs TTS → audio both ways,
  call record + PCA to prod Aurora).

We have **NOT** proven **many simultaneous real-audio conversations**.
Wave 7 closes that gap: 10–20 concurrent calls with real audio flowing
through the full pipeline, on the current (non-BAA) vendor accounts.

**Exit criterion:** if Wave 7 passes, the BAAs get signed with
confidence. If it surfaces issues, we fix on current accounts and
re-test.

## What's different from Wave 6 (why mock load isn't enough)

| Dimension | Wave 6 mock | Wave 7 real-audio |
|---|---|---|
| Call lifetime | ~1.7s (fail-fast dialout) | ~2 min (full conversation) |
| STT | never engaged | AssemblyAI Universal-3 streaming, per call |
| LLM | never engaged | Bedrock Haiku, multiple turns per call |
| TTS | never engaged | ElevenLabs WS streaming, concurrent |
| Audio buffers | none | real PCM in/out, resampling, jitter buffers |
| Memory profile | Python objects only | + audio buffers, codec state, WS frames |
| Vendor concurrency limits | untested | **the whole point** |

Real audio exercises per-call vendor WebSocket connections (one
ElevenLabs WS + one AssemblyAI WS + Bedrock streaming per call) that
mock load never opened. Concurrency limits, throttling, and audio
quality under load are all invisible until now.

## Test method — synthetic audio injection (deferred Wave 6 scenario F)

Rather than dial 20 real phones, inject pre-recorded audio into each
call as a Daily WebRTC participant. This is the scenario F approach we
deferred from Wave 6.

### Call entry path: `direction: "browser"` (NOT PSTN dialout)

```
POST /start { "direction": "browser", "agent_id": "v2-tools-test" }
   → 202 { room_url, room_name, viewer_token, call_id }
```

- The engine bot joins the Daily room and waits for a participant.
- The harness joins the same room (using `viewer_token`) as a
  `daily-python` participant and **publishes pre-recorded WAV audio**
  as if it were the caller.
- Bot: STT transcribes injected audio → Bedrock LLM → ElevenLabs TTS →
  publishes bot audio to the room.
- Harness subscribes to the bot's audio track to (a) confirm audio
  actually arrived and (b) save a sample for spot-check.

Using `direction: "browser"` avoids the PSTN/SIP leg entirely — no
per-minute telephony charges, no dependency on the inbound-webhook
cutover. It still exercises STT + LLM + TTS + WebRTC media at
concurrency, which is exactly what we're validating.

### Scripted audio turns

A small set of pre-recorded WAV prompts (8 kHz mono, matching the
engine's `_AUDIO_IN_SAMPLE_RATE`) drives a deterministic conversation
that exercises the tool paths:

1. "Hi." (opener trigger)
2. "I have a question." (general turn → LLM + TTS)
3. "Can you press one for me?" (→ `press_digit` tool)
4. "Okay, goodbye." (→ `end_call` tool, clean termination)

~4 turns over ~2 minutes per call. Recorded once with a TTS voice or a
human; reused across all concurrent calls.

### Concurrency ramp

| Stage | Concurrent calls | Purpose |
|---|---|---|
| W7-0 | 1 | Real-audio baseline (TTFB, latency, quality reference) |
| W7-1 | 5 | First concurrency step — vendor WS behavior |
| W7-2 | 10 | Mid concurrency |
| W7-3 | 20 | Target concurrency (matches expected early-prod peak) |

Each stage holds for ~2 min (one full call cycle) plus a short ramp.
Run W7-0 → W7-3 sequentially, capturing metrics per stage.

## Harness placement

A single **EC2 instance in us-east-1** (e.g. `c5.xlarge`, 4 vCPU /
8 GB), close to Daily's media servers, runs the N `daily-python`
participants. Reasons:

- 20 concurrent WebRTC audio publishers + subscribers is CPU- and
  bandwidth-heavy; a laptop on residential Wi-Fi (which already caused
  flaky `/status` polling in Wave 6) is not reliable for media.
- Co-locating with Daily/AWS in us-east-1 removes home-network jitter
  as a confound when measuring audio quality.

Harness builds on the existing `backend/voice-agent/scripts/wave6/`
package (config, CloudWatch query helpers, scenario_base) + a new
`scripts/wave7/` with the audio-injection participant.

## Metrics captured (per stage)

| Metric | Source | Pass criterion (proposed) |
|---|---|---|
| ElevenLabs concurrent-stream errors | engine logs (1008 / WS close) | 0 unrecoverable; note any throttle/backoff |
| ElevenLabs TTS TTFB | engine pipecat metrics | p95 < 1.5s at N=20 (baseline was ~0.12s) |
| AssemblyAI 1008 rate | engine logs | bursts acceptable per tech-debt #13; no sustained failures |
| Bedrock throughput / throttling | engine logs + Bedrock CW metrics | 0 `ThrottlingException`; TTFB p95 < 2.5s |
| Audio delivered (bot → harness) | harness track subscription | 100% of calls receive bot audio |
| Audio quality spot-check | saved samples | intelligible, no garble/dropouts (manual) |
| CPU / memory per task | AWS/ECS | mem < 70%, CPU < 80% (Wave 6 Option I budget) |
| End-to-end turn latency | harness timestamps | p95 conversational turn < 3s |
| Degradation vs N=1 baseline | all of the above | no cliff between N=1 and N=20 |

## Vendor concurrency limits to watch (the real unknowns)

- **ElevenLabs** — concurrent request limit is **tier-dependent**.
  Creator/Pro allow a limited number of simultaneous TTS streams
  (historically ~5 on lower tiers, ~10–15 on Pro). **At N=20 we may hit
  the per-account concurrent-stream cap** → this is a key finding for
  the Enterprise decision (Enterprise raises it). If we hit it at
  N=10–20 on Pro, that's data *for* signing Enterprise, not a platform
  bug.
- **AssemblyAI** — per-minute new-stream limit (tech-debt #13). 20
  near-simultaneous stream opens may trip 1008 briefly; documented as
  expected vendor behavior, auto-scales.
- **Bedrock** — Haiku inference-profile RPM/TPM. 20 concurrent × a few
  turns is well within default quotas (verified during the earlier
  Bedrock-quota analysis), but we confirm no throttling.

## Cost estimate

The binding constraint is **ElevenLabs character quota**, not dollars.

### Per-call ElevenLabs usage
~2-min scripted call, bot speaks ~40–60s → **~1,000 characters/call**
(range 700–1,500).

### Per full ramp run (W7-0 through W7-3)
1 + 5 + 10 + 20 = **36 calls** × ~1,000 = **~36,000 characters/run**
(budget 50,000 with headroom).

### Dollar costs per run (negligible)
| Vendor | Usage | Cost |
|---|---|---|
| ElevenLabs | ~36k chars | (quota, see below) |
| AssemblyAI | ~36 calls × 2 min ≈ 1.2 hr streaming | ~$0.20–0.55 |
| Bedrock Haiku | ~36 calls × few turns | < $0.20 |
| Daily WebRTC | 2 participants × 2 min × 36 ≈ 144 part-min (no PSTN) | ~$0.60 |
| AWS Fargate | already running | $0 marginal |
| Harness EC2 c5.xlarge | ~4 hrs | ~$0.70 |
| **Total $/run** | | **~$1.50–2.50** |

Expect 3–5 runs across dev iteration + the graded run + a re-run:
**~$8–12 total dollars**, plus ElevenLabs quota.

### ElevenLabs quota — the actual gate

- Current: **creator tier, 100,002/mo, ~10,018 remaining** (resets ~9th
  of month to 100k).
- One full ramp run ≈ 36k chars. **The ~10k remaining is not enough for
  even one run.** Even after the monthly reset, creator's 100k allows
  ~2 full runs with no buffer — too tight for iterative testing.

**Minimum to run Wave 7 cleanly: bump to ElevenLabs Pro for one month.**

| Plan | Chars/mo | ~Price | Wave 7 runs it covers |
|---|---|---|---|
| Creator (current) | 100k | ~$22/mo | ~2 (too tight) |
| **Pro (recommended)** | **500k** | **~$99/mo** | **~13 full runs** |
| Scale | 2M | ~$330/mo | overkill for Wave 7 |

**Recommendation:** one month of **Pro (~$99)**. Gives ~500k chars =
ample headroom for all Wave 7 iterations + ongoing single-call testing,
AND raises the concurrent-stream limit (helping us actually reach N=20).
This is ~0.5% of the Enterprise annual commitment — the right amount to
spend to *de-risk* that commitment. Downgrade or convert to Enterprise
after Wave 7 based on results.

(Prices approximate — verify current ElevenLabs pricing at purchase.)

## Effort + timeline

| Task | Estimate |
|---|---|
| `scripts/wave7/` audio-injection participant (daily-python publish + subscribe) | ~1 day |
| Pre-recorded WAV prompt set + scripted turn driver | ~0.5 day |
| EC2 harness provisioning + run orchestration (build on wave6/) | ~0.5 day |
| Metrics capture + report generation | ~0.5 day |
| Execute ramp (W7-0→W7-3) + analysis | ~0.5 day |
| **Total** | **~3 days** |

## Open decisions before running

1. **ElevenLabs Pro bump** — needed before any meaningful run. ~$99,
   one month. (Or accept testing only at low concurrency on the ~10k
   remaining, which won't validate N=20.)
2. **Agent under test** — `v2-tools-test` (Haiku, exercises tools) is
   the natural choice; it also doubles as the Haiku voice-quality read
   we deferred. Confirm.
3. **Target concurrency** — design tops out at N=20. If early-prod peak
   expectation is higher, raise the ramp ceiling (cost scales linearly).
4. **Harness host** — EC2 c5.xlarge in us-east-1 (recommended) vs a
   strong local box. EC2 is more reliable for 20 concurrent media
   streams.

## Not in scope for Wave 7

- PSTN/SIP path at concurrency (uses `direction: browser` instead —
  the SIP leg was validated single-call; concurrency of the media
  pipeline is the variable under test).
- Inbound dial-in webhook cutover (separate task).
- HIPAA/BAA data handling (by definition pre-BAA; synthetic audio,
  no PHI).
