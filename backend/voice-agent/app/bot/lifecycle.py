"""End-of-call orchestration: build CallRecord, fire Layer 6 writes.

Called from :func:`~app.bot.bot.run_bot`'s ``finally`` block — fires
once per call, no matter how the pipeline terminates (normal end,
exception, cancellation). Best-effort, **never raises**: a
persistence failure must not propagate out of the call coroutine.

Order of operations matches Layer 6's locked-in two-write pattern:

1. Build :class:`CallRecord` from accumulator + error_state + locals.
2. ``write_call_record`` — first write, full snapshot, empty
   ``post_call_analyses``.
3. If first write succeeded AND status is ``"completed"`` AND the
   agent has ``post_call_analyses`` configured: run Bedrock
   extraction, write again with the populated dict.
4. If first write succeeded: fire ``trigger_auto_actions`` exactly
   once. Lambda computes derived costs / scores / tasks.

Failure cascades:

* First write fails → no PCA, no auto-actions. CallRecord lost.
* PCA fails → ``run_post_call_analyses`` returns ``{}``; we don't
  re-write; auto-actions fires anyway with empty PCA (gets only
  telephony cost).
* Second write fails → PCA is computed but not persisted; logged.
* Auto-actions fails → logged; the row in ``voice_calls`` stays
  canonical.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from app.config.agent_config import AgentConfig
from app.config.settings import Settings
from app.observers.error_state import ErrorState
from app.persistence.call_record import CallRecord
from app.persistence.call_writer import trigger_auto_actions, write_call_record
from app.persistence.post_call import run_post_call_analyses
from app.persistence.transcript import TranscriptAccumulator

logger = structlog.get_logger(__name__)


async def finalize_call(
    *,
    call_id: str,
    agent: AgentConfig,
    accumulator: TranscriptAccumulator,
    error_state: ErrorState,
    case_data: dict[str, Any],
    started_at: datetime,
    end_status: str,
    call_error: str | None,
    direction: str,
    target_number: str,
    from_number: str,
    session_id: str,
    batch_id: str | None,
    batch_row_index: int | None,
    settings: Settings,
) -> None:
    """Build :class:`CallRecord`, fire Layer 6 writes, run PCA + auto-actions.

    All arguments are keyword-only because there are too many of
    them to remember positionally. Layer 8's ``run_bot`` finally
    block is the only caller; Layer 9 doesn't reach here directly.

    Never raises — wraps the whole flow in best-effort logging.
    """
    ended_at = datetime.now(UTC)
    duration = max(0, int((ended_at - started_at).total_seconds()))

    transcript = accumulator.to_list()
    # Caller-supplied error from try/except wins; fall back to any
    # ErrorFrame the observer recorded. This catches Pipecat-internal
    # errors that v1 silently dropped (Bedrock validation, AssemblyAI
    # disconnect, etc. — see Layer 7 ErrorObserver).
    final_error = call_error or error_state.last_error

    record = CallRecord(
        id=call_id,
        agent_name=agent.name,
        agent_display_name=agent.display_name,
        from_number=from_number,
        target_number=target_number,
        direction=direction,
        status=end_status,
        started_at=started_at,
        ended_at=ended_at,
        duration_secs=duration,
        case_data=case_data,
        transcript=transcript,
        # Engine never writes recording_path; Daily's recording
        # webhook patches it later via the lambda's
        # POST /api/calls/recording-update endpoint, keyed by
        # session_id (= Daily room name). See Layer 6 docs.
        recording_path=None,
        post_call_analyses={},
        error=final_error,
        batch_id=batch_id,
        batch_row_index=batch_row_index,
        session_id=session_id,
    )

    logger.info(
        "finalize_call_starting",
        call_id=call_id,
        status=end_status,
        duration_secs=duration,
        transcript_turns=len(transcript),
        has_error=bool(final_error),
        agent_name=agent.name,
    )

    first_ok = await write_call_record(record, settings)

    # PCA only on completed calls with configured analyses. Failed /
    # cancelled calls usually have malformed transcripts that confuse
    # the extraction LLM; auto-actions will still compute baseline
    # cost from transcript length so we don't lose billing fidelity.
    if (
        first_ok
        and end_status == "completed"
        and agent.post_call_analyses
        and agent.post_call_analyses.fields
    ):
        analyses = await run_post_call_analyses(
            agent,
            case_data,
            transcript,
            settings,
        )
        if analyses:
            record.post_call_analyses = analyses
            second_ok = await write_call_record(record, settings)
            logger.info(
                "post_call_analyses_persisted",
                call_id=call_id,
                second_write_ok=second_ok,
                fields_extracted=len(analyses),
            )

    # Trigger derived writes (cost / score / actions). The Lambda
    # endpoint is idempotent for cost/score (ON CONFLICT DO UPDATE)
    # but NOT for voice_auto_actions inserts — must be at-most-once
    # per call. Layer 8's finally fires exactly once, so this is safe.
    if first_ok:
        await trigger_auto_actions(call_id, settings)

    logger.info(
        "finalize_call_complete",
        call_id=call_id,
        first_write_ok=first_ok,
    )
