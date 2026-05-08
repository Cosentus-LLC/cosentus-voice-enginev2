"""Layer 9.5 scale-test harness.

Spawns the v2 engine in a subprocess, fires concurrent ``/start``
POSTs against a fake destination (Daily accepts the dialout, the
carrier rejects with "Remote busy or did not answer", and Phase 2's
``safe_cancel("dialout_stopped")`` terminates the bot — total bot
lifetime ~3-5 s, no human bothered), and captures resource metrics
+ per-call latency to answer:

* Is ``max_concurrent_calls=6`` correct, or should it be tuned?
* Memory budget per task (informs Layer 11 Fargate sizing)
* FD budget per task (informs Layer 11 ulimits)
* Does Phase 2 SIGTERM drain finish within the 110 s budget when
  6 bots are mid-flight?
* Any vendor (Daily / Bedrock / ElevenLabs / Lambda) throttling at
  6× concurrency?

Usage::

    # Run all canonical scenarios (a-e) sequentially:
    python backend/voice-agent/scripts/scale_test.py --scenarios all

    # Run a subset:
    python backend/voice-agent/scripts/scale_test.py --scenarios a,b,c

    # Soak test (background, captures memory + FD growth over time):
    python backend/voice-agent/scripts/scale_test.py --soak --hours 24 --concurrency 3

Output lands in ``scale_test_results/<UTC-timestamp>/``:

* ``engine.log`` — engine stdout/stderr
* ``samples.jsonl`` — resource snapshots (RSS, FDs, CPU%) at 1 Hz
* ``scenario_<name>.json`` — per-scenario metrics + per-call latencies
* ``report.md`` — human-readable summary table
* ``soak.jsonl`` (soak test only) — per-cycle metrics

Design notes
------------

* No external deps (no ``psutil``). Uses ``ps`` / ``lsof`` subprocess
  calls for portable metrics. Slower than psutil but adequate for
  1 Hz sampling.
* The engine is a subprocess spawned via ``uv run python -m app.main``.
  ``uv`` wraps the actual Python, so the engine PID is the
  grandchild — discovered via ``lsof -tiTCP:8080``.
* Daily / Aurora / vendor APIs are real. Each scenario incurs real
  Daily room creations (~N rooms per N-call scenario); they auto-
  expire at 4 h so accumulation is bounded. Aurora writes are real
  and persistent.
* Fake target ``+19998887777`` is invalid for the carrier; Daily
  accepts the dialout, the carrier rejects within ~1 s, no real
  ringing happens.

Scenarios
---------

* ``a`` — N=1 baseline. Single call. Establishes per-call cost.
* ``b`` — N=3 light. 3 concurrent calls. 50% of max_concurrent.
* ``c`` — N=6 target. 6 concurrent calls. At max_concurrent.
* ``d`` — N=10 overflow. 10 concurrent calls. First 6 accepted,
  calls 7-10 should be 503 with reason=at_capacity.
* ``e`` — N=6 + SIGTERM mid-flight. 6 calls, then SIGTERM during
  active flight. Verifies Phase 2 drain budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp

# Add the repo's package root so we can import nothing from the
# engine itself — but the path is added defensively in case any
# future helper needs it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Constants ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_CWD = REPO_ROOT / "backend" / "voice-agent"
ENV_SKELETON_PATH = ENGINE_CWD / "scripts" / ".env.skeleton"

ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 8080
ENGINE_BASE = f"http://{ENGINE_HOST}:{ENGINE_PORT}"
ENGINE_HEALTH = f"{ENGINE_BASE}/health"
ENGINE_READY = f"{ENGINE_BASE}/ready"
ENGINE_STATUS = f"{ENGINE_BASE}/status"
ENGINE_START = f"{ENGINE_BASE}/start"

# Fake destination — Daily accepts the dialout, carrier rejects.
FAKE_TARGET = "+19998887777"
# Caller_id whose UUID we resolved + cached during the live test.
TEST_FROM_NUMBER = "+12098210846"
TEST_AGENT = "v2-tools-test"

ENGINE_STARTUP_TIMEOUT_SECS = 90
METRICS_POLL_INTERVAL_SECS = 1.0
POST_SCENARIO_SETTLE_SECS = 5
DRAIN_BUDGET_SECS = 110

# When firing N concurrent /start POSTs, allow this many seconds for
# all of them to land before declaring "done with the firing phase".
START_FIRING_BUDGET_SECS = 30

# Per-call timeout for the /start POST itself (the spawn handshake;
# the bot continues running after the response).
START_HTTP_TIMEOUT_SECS = 30


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class CallResult:
    """Outcome of one /start POST attempt."""

    call_index: int
    http_status: int
    latency_ms: float
    call_id: str | None = None
    rejected_reason: str | None = None
    error: str | None = None


@dataclass
class ResourceSnapshot:
    """One sample of engine subprocess metrics."""

    elapsed_secs: float
    rss_mb: float
    fd_count: int
    cpu_pct: float
    active_sessions: int


@dataclass
class ScenarioResult:
    """Aggregated outcome for one scenario."""

    name: str
    n_calls: int
    started_at: str
    finished_at: str
    duration_secs: float
    accepted: int
    rejected: int
    other: int
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_p99: float = 0.0
    latency_ms_max: float = 0.0
    peak_active_sessions: int = 0
    initial_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    final_rss_mb: float = 0.0
    rss_growth_mb: float = 0.0
    initial_fd_count: int = 0
    peak_fd_count: int = 0
    final_fd_count: int = 0
    fd_growth: int = 0
    peak_cpu_pct: float = 0.0
    notes: list[str] = field(default_factory=list)
    call_results: list[dict] = field(default_factory=list)


# ── Engine subprocess wrapper ──────────────────────────────────────────


class EngineProcess:
    """Manage the engine subprocess: spawn, discover pid, send signals."""

    def __init__(self, *, log_path: Path, env: dict | None = None):
        self.log_path = log_path
        self.env = env or {}
        self._proc: subprocess.Popen | None = None
        self._log_file: Any = None
        self._engine_pid: int | None = None

    async def start(self) -> None:
        """Spawn the engine and wait for ``/ready`` 200."""
        self._log_file = open(self.log_path, "wb")
        env = {**os.environ, **self.env}
        self._proc = subprocess.Popen(
            ["uv", "run", "python", "-m", "app.main"],
            cwd=str(ENGINE_CWD),
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Poll /ready until 200 or timeout.
        deadline = time.time() + ENGINE_STARTUP_TIMEOUT_SECS
        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
                try:
                    async with session.get(
                        ENGINE_READY, timeout=aiohttp.ClientTimeout(total=2.0)
                    ) as resp:
                        if resp.status == 200:
                            break
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError(
                    f"engine did not become ready within {ENGINE_STARTUP_TIMEOUT_SECS}s"
                )

        # Discover the actual engine PID (uv run wraps the python child).
        self._engine_pid = self._discover_engine_pid()
        if self._engine_pid is None:
            raise RuntimeError("could not discover engine pid via lsof on port 8080")

    def _discover_engine_pid(self) -> int | None:
        """Find the python process listening on the engine port."""
        try:
            result = subprocess.run(
                ["lsof", "-tiTCP:8080", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = result.stdout.strip().splitlines()
            if line:
                return int(line[0])
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass
        return None

    @property
    def engine_pid(self) -> int | None:
        return self._engine_pid

    @property
    def is_running(self) -> bool:
        if not self._proc:
            return False
        return self._proc.poll() is None

    def send_signal(self, sig: int) -> None:
        """Send a signal to the engine python child (not the uv wrapper)."""
        if self._engine_pid is None:
            return
        try:
            os.kill(self._engine_pid, sig)
        except ProcessLookupError:
            pass

    def stop(self, timeout: float = DRAIN_BUDGET_SECS + 30) -> int:
        """SIGTERM the engine; wait for clean exit; return the exit code.

        Returns -1 if the wrapper failed to exit within the timeout.
        """
        if self._proc is None:
            return 0
        if self._engine_pid is not None:
            try:
                os.kill(self._engine_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
            except OSError:
                pass
            return -1
        finally:
            if self._log_file:
                self._log_file.close()
                self._log_file = None


# ── Resource sampling ─────────────────────────────────────────────────


def _sample_rss_mb(pid: int) -> float:
    """Read RSS in MB via ``ps`` (kilobytes → MB)."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "rss="],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if out:
            return int(out) / 1024.0
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return 0.0


def _sample_cpu_pct(pid: int) -> float:
    """Read instantaneous CPU% via ``ps``.

    macOS ``ps`` returns cumulative CPU% which isn't great, but for
    relative comparison across scenarios it suffices.
    """
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu="],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if out:
            return float(out)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return 0.0


def _sample_fd_count(pid: int) -> int:
    """Count open file descriptors via ``lsof``."""
    try:
        result = subprocess.run(
            ["lsof", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # First line is the header; remaining are FDs.
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return max(0, len(lines) - 1)
    except (subprocess.TimeoutExpired, OSError):
        pass
    return 0


async def _sample_active_sessions(session: aiohttp.ClientSession) -> int:
    """Read current active_sessions from /ready (no auth)."""
    try:
        async with session.get(ENGINE_READY, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
            data = await resp.json()
            return int(data.get("active_sessions", 0))
    except (aiohttp.ClientError, TimeoutError, ValueError):
        return -1


@asynccontextmanager
async def metrics_sampler(
    *,
    pid: int,
    samples_path: Path,
    started_at: float,
    stop_event: asyncio.Event,
):
    """Background coroutine that samples resources every second.

    Writes JSON-lines to ``samples_path``. Stops when ``stop_event``
    is set.
    """
    samples: list[ResourceSnapshot] = []

    async def loop():
        async with aiohttp.ClientSession() as session:
            with open(samples_path, "w") as f:
                while not stop_event.is_set():
                    elapsed = time.time() - started_at
                    rss = _sample_rss_mb(pid)
                    cpu = _sample_cpu_pct(pid)
                    fds = _sample_fd_count(pid)
                    sessions = await _sample_active_sessions(session)
                    snap = ResourceSnapshot(
                        elapsed_secs=round(elapsed, 2),
                        rss_mb=rss,
                        fd_count=fds,
                        cpu_pct=cpu,
                        active_sessions=sessions,
                    )
                    samples.append(snap)
                    f.write(json.dumps(asdict(snap)) + "\n")
                    f.flush()
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(), timeout=METRICS_POLL_INTERVAL_SECS
                        )
                    except TimeoutError:
                        continue

    task = asyncio.create_task(loop())
    try:
        yield samples
    finally:
        stop_event.set()
        await task


# ── /start helpers ────────────────────────────────────────────────────


async def fire_start(
    session: aiohttp.ClientSession,
    *,
    call_index: int,
) -> CallResult:
    """Fire one /start POST and time it."""
    payload = {
        "direction": "outbound",
        "agent_id": TEST_AGENT,
        "target_number": FAKE_TARGET,
        "from_number": TEST_FROM_NUMBER,
        "case_data": {},
    }
    t0 = time.time()
    try:
        async with session.post(
            ENGINE_START,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=START_HTTP_TIMEOUT_SECS),
        ) as resp:
            latency_ms = (time.time() - t0) * 1000.0
            body: dict | None = None
            try:
                body = await resp.json()
            except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError):
                pass
            if resp.status == 202 and body:
                return CallResult(
                    call_index=call_index,
                    http_status=resp.status,
                    latency_ms=latency_ms,
                    call_id=body.get("call_id"),
                )
            if resp.status == 503 and body:
                return CallResult(
                    call_index=call_index,
                    http_status=resp.status,
                    latency_ms=latency_ms,
                    rejected_reason=body.get("reason") or body.get("error"),
                )
            return CallResult(
                call_index=call_index,
                http_status=resp.status,
                latency_ms=latency_ms,
                error=str(body)[:200] if body else "no body",
            )
    except (aiohttp.ClientError, TimeoutError) as exc:
        return CallResult(
            call_index=call_index,
            http_status=-1,
            latency_ms=(time.time() - t0) * 1000.0,
            error=str(exc),
        )


async def wait_for_active_sessions(
    session: aiohttp.ClientSession,
    *,
    target: int,
    op: str = "==",
    timeout: float = 30.0,
) -> int:
    """Poll /ready until active_sessions matches target.

    ``op`` is ``"=="``, ``">="``, or ``"<="`` — describing the
    relationship to wait for. Returns the last observed value.
    """
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        last = await _sample_active_sessions(session)
        if op == "==" and last == target:
            return last
        if op == ">=" and last >= target:
            return last
        if op == "<=" and last <= target:
            return last
        await asyncio.sleep(0.5)
    return last


# ── Scenario runners ─────────────────────────────────────────────────


async def run_scenario_concurrent(
    *,
    name: str,
    n_calls: int,
    engine: EngineProcess,
    output_dir: Path,
    sigterm_mid_flight: bool = False,
) -> ScenarioResult:
    """Generic scenario: fire N concurrent /start POSTs, capture
    metrics, wait for finalization (or SIGTERM mid-flight).
    """
    samples_path = output_dir / f"samples_{name}.jsonl"
    started_at_iso = datetime.now(UTC).isoformat()
    started_at = time.time()
    stop_metrics = asyncio.Event()

    pid = engine.engine_pid
    assert pid is not None, "engine pid not discovered"

    initial_rss = _sample_rss_mb(pid)
    initial_fds = _sample_fd_count(pid)

    notes: list[str] = []

    async with (
        aiohttp.ClientSession() as session,
        metrics_sampler(
            pid=pid,
            samples_path=samples_path,
            started_at=started_at,
            stop_event=stop_metrics,
        ) as samples,
    ):
        # Fire N /start POSTs concurrently.
        tasks = [asyncio.create_task(fire_start(session, call_index=i)) for i in range(n_calls)]
        results: list[CallResult] = await asyncio.gather(*tasks)

        peak_active = max(
            (s.active_sessions for s in samples if s.active_sessions >= 0),
            default=0,
        )

        if sigterm_mid_flight:
            # Wait until the engine has all accepted calls in flight.
            target_active = sum(1 for r in results if r.http_status == 202)
            observed = await wait_for_active_sessions(
                session,
                target=target_active,
                op=">=",
                timeout=10.0,
            )
            notes.append(f"pre_sigterm_active_sessions={observed} (target={target_active})")

            # Send SIGTERM.
            sigterm_at = time.time() - started_at
            engine.send_signal(signal.SIGTERM)
            notes.append(f"sigterm_sent_at_secs={sigterm_at:.2f}")

            # Wait for the engine process to exit cleanly.
            wait_start = time.time()
            while engine.is_running:
                if time.time() - wait_start > DRAIN_BUDGET_SECS + 20:
                    notes.append("drain_exceeded_budget — process still running")
                    break
                await asyncio.sleep(0.5)
            drain_secs = time.time() - wait_start
            notes.append(f"drain_secs={drain_secs:.2f}")
        else:
            # Wait for active_sessions to drop to 0 (all bots finalized).
            observed = await wait_for_active_sessions(
                session,
                target=0,
                op="==",
                timeout=60.0,
            )
            if observed != 0:
                notes.append(f"final_active_sessions={observed} (expected 0)")

        # Settle window for last samples + post-call metrics.
        await asyncio.sleep(POST_SCENARIO_SETTLE_SECS)

        peak_active = max(
            peak_active,
            *(s.active_sessions for s in samples if s.active_sessions >= 0),
        )

    # Stop signal already sent inside __aexit__; samples list is fully populated.

    finished_at = time.time()
    finished_at_iso = datetime.now(UTC).isoformat()

    if engine.is_running:
        final_rss = _sample_rss_mb(pid)
        final_fds = _sample_fd_count(pid)
    elif samples:
        final_rss = samples[-1].rss_mb
        final_fds = samples[-1].fd_count
    else:
        final_rss = 0.0
        final_fds = 0
    peak_rss = max((s.rss_mb for s in samples), default=initial_rss)
    peak_fd = max((s.fd_count for s in samples), default=initial_fds)
    peak_cpu = max((s.cpu_pct for s in samples), default=0.0)

    # Per-call latency stats, only for successful (or even errored) calls
    # — we want to know /start latency under load regardless of outcome.
    latencies = [r.latency_ms for r in results]
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)

    accepted = sum(1 for r in results if r.http_status == 202)
    rejected = sum(1 for r in results if r.http_status == 503)
    other = len(results) - accepted - rejected

    rejected_reasons: dict[str, int] = {}
    for r in results:
        if r.http_status == 503 and r.rejected_reason:
            rejected_reasons[r.rejected_reason] = rejected_reasons.get(r.rejected_reason, 0) + 1

    return ScenarioResult(
        name=name,
        n_calls=n_calls,
        started_at=started_at_iso,
        finished_at=finished_at_iso,
        duration_secs=round(finished_at - started_at, 2),
        accepted=accepted,
        rejected=rejected,
        other=other,
        rejected_reasons=rejected_reasons,
        latency_ms_p50=round(p50, 1),
        latency_ms_p95=round(p95, 1),
        latency_ms_p99=round(p99, 1),
        latency_ms_max=round(max(latencies, default=0.0), 1),
        peak_active_sessions=peak_active,
        initial_rss_mb=round(initial_rss, 1),
        peak_rss_mb=round(peak_rss, 1),
        final_rss_mb=round(final_rss, 1),
        rss_growth_mb=round(final_rss - initial_rss, 1),
        initial_fd_count=initial_fds,
        peak_fd_count=peak_fd,
        final_fd_count=final_fds,
        fd_growth=final_fds - initial_fds,
        peak_cpu_pct=round(peak_cpu, 1),
        notes=notes,
        call_results=[asdict(r) for r in results],
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    return statistics.quantiles(s, n=100, method="inclusive")[int(pct) - 1]


# ── Soak test ──────────────────────────────────────────────────────────


async def run_soak(
    *,
    engine: EngineProcess,
    output_dir: Path,
    hours: float,
    concurrency: int,
    cycle_pause_secs: float = 30.0,
) -> dict:
    """Run cycles of N concurrent calls for the configured duration.

    Each cycle: fire N /start POSTs, wait for active_sessions=0,
    sleep ``cycle_pause_secs``, repeat until ``hours`` elapse.
    Per-cycle metrics written to ``soak.jsonl``.
    """
    soak_path = output_dir / "soak.jsonl"
    samples_path = output_dir / "samples_soak.jsonl"
    started_at = time.time()
    stop_metrics = asyncio.Event()

    pid = engine.engine_pid
    assert pid is not None

    initial_rss = _sample_rss_mb(pid)
    initial_fds = _sample_fd_count(pid)

    duration_secs = hours * 3600
    deadline = started_at + duration_secs
    cycle_index = 0

    # Plain ``with`` for the JSONL file — ``open()`` returns a sync
    # context manager and can't be combined with ``async with``.
    soak_log = open(soak_path, "w")
    try:
        async with (
            aiohttp.ClientSession() as session,
            metrics_sampler(
                pid=pid,
                samples_path=samples_path,
                started_at=started_at,
                stop_event=stop_metrics,
            ) as _samples,
        ):
            while time.time() < deadline:
                cycle_index += 1
                cycle_started = time.time()
                tasks = [
                    asyncio.create_task(fire_start(session, call_index=i))
                    for i in range(concurrency)
                ]
                results = await asyncio.gather(*tasks)
                await wait_for_active_sessions(session, target=0, op="==", timeout=60.0)
                cycle_finished = time.time()

                cycle_record = {
                    "cycle_index": cycle_index,
                    "elapsed_hours": round((cycle_started - started_at) / 3600, 2),
                    "cycle_secs": round(cycle_finished - cycle_started, 2),
                    "accepted": sum(1 for r in results if r.http_status == 202),
                    "rejected": sum(1 for r in results if r.http_status == 503),
                    "rss_mb": round(_sample_rss_mb(pid), 1),
                    "fd_count": _sample_fd_count(pid),
                    "engine_running": engine.is_running,
                }
                soak_log.write(json.dumps(cycle_record) + "\n")
                soak_log.flush()
                print(f"[soak] {json.dumps(cycle_record)}")

                if not engine.is_running:
                    soak_log.write(
                        json.dumps(
                            {
                                "event": "engine_died",
                                "cycle_index": cycle_index,
                                "elapsed_hours": round((time.time() - started_at) / 3600, 2),
                            }
                        )
                        + "\n"
                    )
                    break

                await asyncio.sleep(cycle_pause_secs)
    finally:
        soak_log.close()

    final_rss = _sample_rss_mb(pid) if engine.is_running else 0
    final_fds = _sample_fd_count(pid) if engine.is_running else 0
    summary = {
        "duration_hours": round((time.time() - started_at) / 3600, 2),
        "concurrency": concurrency,
        "cycle_count": cycle_index,
        "initial_rss_mb": round(initial_rss, 1),
        "final_rss_mb": round(final_rss, 1),
        "rss_growth_mb": round(final_rss - initial_rss, 1),
        "initial_fd_count": initial_fds,
        "final_fd_count": final_fds,
        "fd_growth": final_fds - initial_fds,
        "engine_alive_at_end": engine.is_running,
    }
    return summary


# ── Env loading ────────────────────────────────────────────────────────


def load_engine_env() -> dict[str, str]:
    """Load env-var-style key=value pairs from .env.skeleton.

    Mirrors what ``set -a; . scripts/.env.skeleton; set +a`` does in
    the shell, so the engine subprocess sees DAILY_API_KEY etc.
    """
    env: dict[str, str] = {}
    if not ENV_SKELETON_PATH.exists():
        return env
    for line in ENV_SKELETON_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


# ── Output / report ────────────────────────────────────────────────────


def write_scenario_json(result: ScenarioResult, output_dir: Path) -> None:
    path = output_dir / f"scenario_{result.name}.json"
    path.write_text(json.dumps(asdict(result), indent=2))


def render_report(
    results: list[ScenarioResult],
    soak_summary: dict | None,
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Layer 9.5 Scale Test — Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append("## Scenarios")
    lines.append("")
    lines.append(
        "| Name | N | Accepted | Rejected | Latency p50/p95/p99/max (ms) | "
        "Peak active | RSS init→peak→final (MB) | RSS growth (MB) | "
        "FD init→peak→final | FD growth | Peak CPU% |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---|---:|---|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.name} | {r.n_calls} | {r.accepted} | {r.rejected} | "
            f"{r.latency_ms_p50}/{r.latency_ms_p95}/{r.latency_ms_p99}/{r.latency_ms_max} | "
            f"{r.peak_active_sessions} | "
            f"{r.initial_rss_mb}→{r.peak_rss_mb}→{r.final_rss_mb} | {r.rss_growth_mb} | "
            f"{r.initial_fd_count}→{r.peak_fd_count}→{r.final_fd_count} | {r.fd_growth} | "
            f"{r.peak_cpu_pct} |"
        )
    lines.append("")

    for r in results:
        if not r.notes and not r.rejected_reasons:
            continue
        lines.append(f"### Scenario `{r.name}` notes")
        lines.append("")
        for n in r.notes:
            lines.append(f"- {n}")
        if r.rejected_reasons:
            lines.append(f"- rejected_reasons: {dict(r.rejected_reasons)}")
        lines.append("")

    if soak_summary:
        lines.append("## Soak test")
        lines.append("")
        for k, v in soak_summary.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report)
    return report


# ── Main ───────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> int:
    output_root = REPO_ROOT / "scale_test_results"
    output_root.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    output_dir.mkdir()

    print(f"[harness] output dir: {output_dir}")

    env = load_engine_env()
    engine = EngineProcess(log_path=output_dir / "engine.log", env=env)

    print("[harness] starting engine subprocess...")
    await engine.start()
    print(f"[harness] engine ready, pid={engine.engine_pid}")

    scenario_map: dict[str, dict] = {
        "a": {"name": "a_n1_baseline", "n": 1, "sigterm": False},
        "b": {"name": "b_n3_light", "n": 3, "sigterm": False},
        "c": {"name": "c_n6_target", "n": 6, "sigterm": False},
        "d": {"name": "d_n10_overflow", "n": 10, "sigterm": False},
        "e": {"name": "e_n6_sigterm", "n": 6, "sigterm": True},
    }

    results: list[ScenarioResult] = []
    soak_summary: dict | None = None

    try:
        if args.soak:
            soak_summary = await run_soak(
                engine=engine,
                output_dir=output_dir,
                hours=args.hours,
                concurrency=args.concurrency,
            )
        else:
            requested = args.scenarios
            if "all" in requested or not requested:
                requested = ["a", "b", "c", "d", "e"]
            for code in requested:
                spec = scenario_map.get(code)
                if not spec:
                    print(f"[harness] unknown scenario {code!r} — skipping")
                    continue
                if spec["sigterm"] and not engine.is_running:
                    print(f"[harness] engine already exited; cannot run {spec['name']}")
                    continue
                print(f"[harness] running scenario {spec['name']}...")
                result = await run_scenario_concurrent(
                    name=spec["name"],
                    n_calls=spec["n"],
                    engine=engine,
                    output_dir=output_dir,
                    sigterm_mid_flight=spec["sigterm"],
                )
                results.append(result)
                write_scenario_json(result, output_dir)
                print(
                    f"[harness] {spec['name']}: accepted={result.accepted} "
                    f"rejected={result.rejected} "
                    f"peak_rss={result.peak_rss_mb}MB "
                    f"fd_growth={result.fd_growth}"
                )
    finally:
        if engine.is_running:
            print("[harness] stopping engine subprocess...")
            engine.stop()

    report = render_report(results, soak_summary, output_dir)
    print()
    print(report)
    print()
    print(f"[harness] full output: {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--scenarios",
        type=lambda s: [x.strip().lower() for x in s.split(",")],
        default=["all"],
        help="Comma-separated list (a,b,c,d,e) or 'all'.",
    )
    p.add_argument("--soak", action="store_true", help="Run the soak test instead of scenarios.")
    p.add_argument("--hours", type=float, default=24.0, help="Soak duration in hours.")
    p.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Soak concurrency (calls per cycle).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(main(args)))
