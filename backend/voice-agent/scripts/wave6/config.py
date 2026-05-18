"""Wave 6 harness configuration.

Reads values from env vars / CLI flags. The defaults are wired to the
deployed staging environment (the only target Wave 6 supports today).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# ── Staging targets ─────────────────────────────────────────────────────────

STAGING_HOSTNAME: Final = "staging.cosentusaibackend.com"
STAGING_BASE_URL: Final = f"https://{STAGING_HOSTNAME}"

AWS_REGION: Final = "us-east-1"
AWS_ACCOUNT: Final = "825269749545"

ECS_CLUSTER: Final = "cosentus-voice-engine-staging-cluster"
ECS_SERVICE: Final = "cosentus-voice-engine-staging-service"

CW_NAMESPACE: Final = "VoiceAgent/Pipeline"
CW_ENVIRONMENT_DIMENSION: Final = "staging"

# Per Wave 3 deploy: maxCapacity=5, max_concurrent_calls=6, target=4.2/task.
ECS_MAX_TASKS: Final = 5
PER_TASK_CONCURRENCY: Final = 6
SCALE_TARGET_SESSIONS_PER_TASK: Final = 4.2

STUB_LAMBDA_NAME: Final = "cosentus-voice-api-staging-stub"
STUB_AGENT_ID: Final = "staging-mock-1"

# ── Fast-fail call shape (used by scenarios a-e) ────────────────────────────
#
# Fake from_number → Daily rejects with "Incorrect callerID! No phone number
# maps to: ..." → engine's dialout_failed_sync handler cancels the bot
# within ~1.7 s. No real PSTN minute spent, no AssemblyAI / Bedrock /
# ElevenLabs invocation. Stub Lambda fields runtime-config + writes.

FAKE_FROM_NUMBER: Final = "+15555550100"
FAKE_TARGET_NUMBER: Final = "+19998887777"

# Empirical call lifetime from Wave 7 smoke (POST /start → finalize_call):
APPROX_CALL_LIFETIME_SECS: Final = 1.7


# ── Auth ────────────────────────────────────────────────────────────────────


def get_api_key() -> str:
    """Read the X-API-Key for /start auth.

    Sourced from ``STAGING_API_KEY`` env var. The value is the same
    string returned by the api-key Secrets Manager entry (Layer 11
    storage stack). Documented in 1Password under "Cosentus Voice
    Engine - staging /start auth".
    """
    key = os.environ.get("STAGING_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "STAGING_API_KEY env var not set. Export the value from "
            "1Password before running the harness:\n"
            "  export STAGING_API_KEY='<64-hex-chars>'"
        )
    return key


# ── Per-scenario output paths ───────────────────────────────────────────────


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for one Wave 6 harness run."""

    root: Path

    @property
    def report_md(self) -> Path:
        return self.root / "report.md"

    def scenario_json(self, name: str) -> Path:
        return self.root / f"scenario_{name}.json"

    def soak_heartbeat(self) -> Path:
        return self.root / "scenario_e_heartbeat.json"

    def soak_log(self) -> Path:
        return self.root / "scenario_e.jsonl"

    def engine_log_capture(self, scenario: str) -> Path:
        return self.root / f"engine_logs_{scenario}.txt"


def fresh_run_dir(timestamp_iso: str) -> RunPaths:
    """Make a wave6_results/<UTC-timestamp>/ directory."""
    base = (
        Path(__file__).resolve().parents[3]
        / "wave6_results"
        / timestamp_iso
    )
    base.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=base)


# ── Cost budgets per scenario (worst case; harness aborts if exceeded) ──────
#
# Estimated from the Wave 6 design doc plus a 2× safety multiplier. Each
# fast-fail call counts as ~$0.001 vendor charge (Daily room creation
# only — AssemblyAI / Bedrock / ElevenLabs are bypassed at the dialout
# fail). Soak scenario e is the dominant line item.

BUDGET_CALLS_PER_SCENARIO: Final = {
    "a": 600,    # 30 min × ~10 cpm avg + spike headroom
    "b": 100,    # 50 burst + 5 min sustained refire
    "c": 60,     # 5 min × ~3 cpm + replacement overhead
    "d": 200,    # 5 min × 30 concurrent + 50-call overflow burst
    "e": 30000,  # 4 h × ~12 calls/min sustained × 1.5× headroom
}
