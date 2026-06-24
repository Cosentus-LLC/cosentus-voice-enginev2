"""End-of-call snapshot — what Layer 6 ships to the lambda.

:class:`CallRecord` is a dataclass whose attribute names match the
``voice_calls`` Aurora table 1:1 (modulo the auto-managed
``created_at`` and ``hidden`` columns). Renaming any attribute is a
breaking change for the call-history frontend, billing dashboards,
and the lambda contract — don't.

The dataclass enforces one thing the wire schema doesn't: the
``status`` value is validated against a fixed set at construction
time. Aurora keeps ``voice_calls.status`` as ``VARCHAR(30)`` with no
CHECK constraint, so the engine boundary is the only place we can
catch a typo before it reaches Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# v1's status vocabulary, harvested from the lambda's
# ROW_STATUS_MAP plus the values v1's pipeline writes
# (`completed` / `cancelled` / `failed`). Keep this set in sync with
# the lambda — drift would let an unknown status sneak through and
# bypass the engine-side check.
_VALID_STATUSES = frozenset(
    {
        "pending",
        "in_progress",
        "completed",
        "failed",
        "cancelled",
        "no_answer",
        "busy",
        "skipped",
    }
)

# Cap for the ``error`` column. Aurora's column is ``TEXT`` (no
# length limit) but operators use ``error`` for triage — pasting
# a 10 KB stack trace into the call-history UI would render badly.
# Match v1's truncation budget so v1 vs v2 errors look identical
# in the UI.
_ERROR_TRUNCATION_CHARS = 1000


@dataclass
class CallRecord:
    """End-of-call snapshot.

    Field order mirrors the lambda's ``INSERT INTO voice_calls`` column
    order so a side-by-side diff against ``index.mjs`` is trivial.
    """

    id: str
    """Engine-generated UUID at call start (= ``voice_calls.id``)."""

    agent_name: str
    """The agent's slug, NOT the display name. Matches
    ``voice_calls.agent_name`` (`VARCHAR(100)`)."""

    agent_display_name: str
    """Snapshot of the display name at call time. Aurora keeps this
    de-normalized so renaming the agent later doesn't rewrite call
    history."""

    from_number: str
    """E.164. For inbound: the caller. For outbound: our DID."""

    target_number: str
    """E.164. For inbound: our DID. For outbound: the callee. Aurora
    column is ``NOT NULL``."""

    direction: str
    """``"inbound"`` or ``"outbound"``."""

    status: str
    """One of :data:`_VALID_STATUSES`. Validated in :meth:`__post_init__`."""

    started_at: datetime
    """Call start. Engine-set at the top of the pipeline coroutine."""

    ended_at: datetime | None
    """Call end. ``None`` only when constructing a record before the
    pipeline tears down (rare; primarily for tests)."""

    duration_secs: int | None
    """Wall-clock seconds. ``None`` is serialized as ``0`` to match
    v1's behavior — frontend never sees null durations."""

    case_data: dict[str, Any]
    """Hydrator dict that fed the call's prompt + first message.
    Kept for forensics + post-call extraction."""

    transcript: list[dict[str, Any]]
    """Output of :meth:`~app.persistence.transcript.TranscriptAccumulator.to_list`.
    List of ``{turn_number, speaker, content, timestamp}`` dicts."""

    session_id: str
    """Daily room name — the binding for the recording webhook.
    Aurora indexes this; it is REQUIRED for the recording webhook to
    locate the row to patch ``recording_path`` on."""

    recording_path: str | None = None
    """Always ``None`` from the engine. The recording webhook
    (separate code path, lambda-side) patches this column once Daily
    finalizes the cloud recording. Engine never writes a value here."""

    post_call_analyses: dict[str, Any] = field(default_factory=dict)
    """Output of :func:`~app.persistence.post_call.run_post_call_analyses`.
    Empty dict means the agent didn't have ``post_call_analyses.fields``
    configured, or extraction failed."""

    error: str | None = None
    """Last error message — populated only on ``status="failed"``
    paths. Truncated to :data:`_ERROR_TRUNCATION_CHARS` on serialize
    to keep the UI tidy."""

    batch_id: str | None = None
    """Outbound-only. Set by the dispatcher when the call originated
    from a batch row; ``None`` for inbound. Frontend uses this to
    join calls back to their batch."""

    batch_row_index: int | None = None
    """Outbound-only. Index within the parent batch."""

    llm_tokens_in: int = 0
    """Real Bedrock input (prompt) tokens for the whole call — live
    pipeline turns + the post-call extraction (#28). Feeds the API's
    ``voice_call_costs.llm_tokens_in``. Defaults to ``0`` when metrics
    are unavailable, matching the API's estimate-fallback. **Emit half
    only:** the API must be taught to consume it (``api-lambda-v2`` #64)
    before cost capture is end-to-end live."""

    llm_tokens_out: int = 0
    """Real Bedrock output (completion) tokens for the whole call.
    Feeds ``voice_call_costs.llm_tokens_out``. See :attr:`llm_tokens_in`."""

    tts_chars: int = 0
    """Real characters synthesized by TTS for the whole call. Feeds
    ``voice_call_costs.tts_chars``. See :attr:`llm_tokens_in`."""

    terminal_step: str | None = None
    """PHI-free dashboard bucket for where the call ended in the flow."""

    transferred: bool = False
    """Whether the call successfully initiated a human transfer."""

    latency_ms: int | None = None
    """Representative per-call latency in milliseconds; currently average LLM TTFB."""

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status {self.status!r}. Must be one of {sorted(_VALID_STATUSES)}"
            )

    def to_lambda_body(self) -> dict[str, Any]:
        """Serialize to the lambda's ``POST /api/calls`` JSON body.

        ISO-8601 UTC timestamps everywhere. Naive datetimes are
        coerced to UTC at this boundary so a producer that forgets
        ``tzinfo`` still ends up with a timezone-aware wire value.

        ``updated_at`` is filled at serialize time (not stored on the
        dataclass) so a re-write — e.g. after post-call analyses
        complete — gets a fresh timestamp without the caller having
        to remember to bump it. The lambda honors the engine-supplied
        value rather than calling ``NOW()`` server-side.
        """
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "agent_display_name": self.agent_display_name,
            "from_number": self.from_number,
            "target_number": self.target_number,
            "direction": self.direction,
            "status": self.status,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "duration_secs": int(self.duration_secs) if self.duration_secs is not None else 0,
            "case_data": self.case_data,
            "transcript": self.transcript,
            "recording_path": self.recording_path,
            "post_call_analyses": self.post_call_analyses,
            "error": _truncate_error(self.error),
            "batch_id": self.batch_id,
            "batch_row_index": self.batch_row_index,
            "session_id": self.session_id,
            # Real usage for cost capture (#28). The lambda's call-upsert
            # schema is ``.passthrough()`` so these are accepted today, but
            # the API still estimates cost — it must add a raw-usage column
            # + consume these (api #64) for end-to-end capture.
            "llm_tokens_in": int(self.llm_tokens_in),
            "llm_tokens_out": int(self.llm_tokens_out),
            "tts_chars": int(self.tts_chars),
            "terminal_step": self.terminal_step,
            "transferred": bool(self.transferred),
            "latency_ms": int(self.latency_ms) if self.latency_ms is not None else None,
            "updated_at": datetime.now(UTC).isoformat(),
        }


def _iso(dt: datetime | None) -> str | None:
    """Render ``dt`` as an ISO-8601 string, coercing naive → UTC.

    Returns ``None`` if ``dt`` is ``None`` so the wire shape preserves
    nullability. The lambda accepts both ``null`` and a string for
    ``started_at`` / ``ended_at``.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _truncate_error(error: str | None) -> str | None:
    """Truncate to the wire-friendly cap; preserve ``None``."""
    if error is None:
        return None
    return error[:_ERROR_TRUNCATION_CHARS]
