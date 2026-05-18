"""Async HTTPS /start poster — the per-call work unit.

One :class:`HttpCaller` instance, reused across scenarios. Each
``post_start()`` call fires one ``POST /start`` against the staging
ALB, captures the result + latency, and returns a structured outcome
the scenario can aggregate.

Why one shared aiohttp.ClientSession
------------------------------------

Reusing a session amortises TLS handshakes and DNS resolutions. At the
50-burst rate (5 connections/sec) this matters. The session is built
once per scenario and closed in ``aclose()``.

Outcomes the harness distinguishes
----------------------------------

* ``"accepted_202"``  — engine returned 202 + a ``call_id``. Concurrency
  budget consumed.
* ``"rejected_503"``  — engine returned 503 (capacity gate or draining).
  Body should include a ``reason``.
* ``"other_status"``  — anything else (401, 500, etc.). Logged with the
  body preview so failures are diagnosable.
* ``"timeout"``       — the request didn't return inside the per-call
  timeout. Counted distinctly.
* ``"error"``         — aiohttp raised before we got a response.

Each outcome carries the ``latency_ms`` so the scenario can build
P50/P95/P99 histograms.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from . import config


@dataclass(frozen=True)
class CallResult:
    """One ``/start`` POST outcome."""

    started_at_ms: int
    """Wall-clock unix millis at request issue."""
    status_code: int
    """HTTP status; -1 for timeout, -2 for connection error."""
    outcome: str
    """``accepted_202`` / ``rejected_503`` / ``other_status`` / ``timeout`` / ``error``."""
    latency_ms: float
    """Time from POST issue to response (any status) or timeout."""
    call_id: str | None
    """Engine-assigned call_id when 202; None otherwise."""
    body_preview: str
    """First 200 chars of the response body or error message."""

    def is_accepted(self) -> bool:
        return self.outcome == "accepted_202"

    def is_rejected_503(self) -> bool:
        return self.outcome == "rejected_503"


@dataclass
class CallResultBatch:
    """Aggregated results for one scenario phase or wave."""

    results: list[CallResult] = field(default_factory=list)

    def append(self, r: CallResult) -> None:
        self.results.append(r)

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def accepted(self) -> int:
        return sum(1 for r in self.results if r.is_accepted())

    @property
    def rejected_503(self) -> int:
        return sum(1 for r in self.results if r.is_rejected_503())

    @property
    def other_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            if r.outcome in ("accepted_202", "rejected_503"):
                continue
            key = (
                f"status_{r.status_code}"
                if r.outcome == "other_status"
                else r.outcome
            )
            out[key] = out.get(key, 0) + 1
        return out

    def latency_percentiles(self) -> dict[str, float]:
        """P50/P95/P99 latency in ms across all responses (any status)."""
        if not self.results:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ms = sorted(r.latency_ms for r in self.results)
        n = len(ms)

        def pick(p: float) -> float:
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return round(ms[idx], 1)

        return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


# ── Caller ──────────────────────────────────────────────────────────────────


class HttpCaller:
    """Async /start poster wrapping an aiohttp ClientSession.

    Use as an async context manager OR call ``aclose()`` manually after
    scenario completion.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        per_call_timeout_secs: float = 10.0,
        total_connection_limit: int = 200,
    ) -> None:
        self._base_url = base_url or config.STAGING_BASE_URL
        self._api_key = api_key or config.get_api_key()
        self._timeout = aiohttp.ClientTimeout(total=per_call_timeout_secs)
        self._connector = aiohttp.TCPConnector(limit=total_connection_limit)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "HttpCaller":
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ── /start ─────────────────────────────────────────────────────────────

    async def post_start(
        self,
        *,
        agent_id: str = config.STUB_AGENT_ID,
        target_number: str = config.FAKE_TARGET_NUMBER,
        from_number: str = config.FAKE_FROM_NUMBER,
        case_data: dict[str, Any] | None = None,
    ) -> CallResult:
        """Fire one outbound /start POST against staging.

        Defaults yield a fast-fail dialout: stub agent, fake numbers,
        ~1.7 s end-to-end lifecycle. Override for special scenarios.
        """
        assert self._session is not None, "Use as async context manager"

        body = {
            "direction": "outbound",
            "agent_id": agent_id,
            "target_number": target_number,
            "from_number": from_number,
            "case_data": case_data or {"wave6": True},
        }
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/start"
        started_at = time.monotonic()
        started_at_ms = int(time.time() * 1000)

        try:
            async with self._session.post(url, json=body, headers=headers) as resp:
                latency_ms = (time.monotonic() - started_at) * 1000
                try:
                    text = await resp.text()
                except Exception as exc:  # noqa: BLE001
                    text = f"<text-read-failed: {exc}>"
                preview = text[:200]
                call_id: str | None = None
                outcome: str
                if resp.status == 202:
                    outcome = "accepted_202"
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            call_id = str(parsed.get("call_id") or "") or None
                    except (ValueError, json.JSONDecodeError):
                        pass
                elif resp.status == 503:
                    outcome = "rejected_503"
                else:
                    outcome = "other_status"
                return CallResult(
                    started_at_ms=started_at_ms,
                    status_code=resp.status,
                    outcome=outcome,
                    latency_ms=round(latency_ms, 2),
                    call_id=call_id,
                    body_preview=preview,
                )
        except asyncio.TimeoutError:
            return CallResult(
                started_at_ms=started_at_ms,
                status_code=-1,
                outcome="timeout",
                latency_ms=round((time.monotonic() - started_at) * 1000, 2),
                call_id=None,
                body_preview="(timeout)",
            )
        except aiohttp.ClientError as exc:
            return CallResult(
                started_at_ms=started_at_ms,
                status_code=-2,
                outcome="error",
                latency_ms=round((time.monotonic() - started_at) * 1000, 2),
                call_id=None,
                body_preview=f"{type(exc).__name__}: {exc}"[:200],
            )

    # ── /status (operator-only — not used by scenarios mass-firing /start) ──

    async def get_status(self) -> dict[str, Any]:
        """Pull /status. Used for periodic live-state snapshots."""
        assert self._session is not None, "Use as async context manager"
        headers = {"X-API-Key": self._api_key}
        url = f"{self._base_url}/status"
        try:
            async with self._session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return {"_error": resp.status, "_body": (await resp.text())[:200]}
                return await resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"_error": str(exc)}

    # ── /ready (no auth) ────────────────────────────────────────────────────

    async def get_ready(self) -> tuple[int, dict[str, Any]]:
        """Pull /ready. Returns (status_code, body). 200 healthy / 503 not."""
        assert self._session is not None
        url = f"{self._base_url}/ready"
        try:
            async with self._session.get(url) as resp:
                try:
                    body = await resp.json()
                except Exception:  # noqa: BLE001
                    body = {"_raw": (await resp.text())[:200]}
                return resp.status, body
        except Exception as exc:  # noqa: BLE001
            return -2, {"_error": str(exc)}
