"""Per-call usage tally — real Bedrock tokens + TTS characters for cost capture (#28).

The engine has the ground-truth numbers a cost/ROI calculation needs but, until
now, dropped them: Pipecat emits LLM token counts and TTS character counts as
numeric ``MetricsFrame``s during the live pipeline, and the post-call extraction
(:func:`~app.persistence.post_call.run_post_call_analyses`) makes its own Bedrock
Converse call that reports ``usage``. Both were observed only for tracing (#13)
and then discarded; the API's ``voice_call_costs`` columns
(``llm_tokens_in/out``, ``tts_chars``) had nothing to populate them and could
only *estimate* cost.

:class:`UsageAccumulator` is the single tally both sources feed:

* the live pipeline — :class:`~app.observers.metrics_observer.MetricsObserver`
  folds each ``LLMUsageMetricsData`` / ``TTSUsageMetricsData`` in;
* the post-call extraction — :func:`run_post_call_analyses` adds its Converse
  ``usage`` (every attempt, including the retry).

At end-of-call :func:`~app.bot.lifecycle.finalize_call` reads :meth:`totals`
onto the :class:`~app.persistence.call_record.CallRecord`. This is a plain
observer-fed state container, same shape as :class:`~app.observers.error_state.ErrorState`
— it lives in ``observers/`` so ``persistence`` (which consumes it) imports
*from* observers, the established direction.

Numeric only — token counts and a character count, **no PHI**. No new vendor,
no new secret.

Concurrency: every mutation happens on the call's asyncio event loop — the
observer's ``on_push_frame`` callback and the ``finalize_call`` coroutine. The
post-call Bedrock call runs in a worker thread (``asyncio.to_thread``) but its
``usage`` is added back on the loop, not inside the thread. So no lock is needed.

**End-to-end cost capture is not live until the paired API change lands.** This
is the engine *emit* half: it now sends real usage on the wire. The API
(``api-lambda-v2`` #64) still computes costs via an *estimating* ``track_cost``
action and has no raw-usage column — the call upsert schema is ``.passthrough()``
so the new keys are accepted, but they are dropped until the API is taught to
consume them. Filed separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageTotals:
    """Immutable snapshot of a call's accumulated usage.

    Field names mirror :class:`~app.persistence.call_record.CallRecord`'s usage
    fields 1:1 so :func:`~app.bot.lifecycle.finalize_call` can copy them across
    without remapping.
    """

    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    tts_chars: int = 0


class UsageAccumulator:
    """Running tally of LLM tokens + TTS characters for one call.

    One instance per call, created in :func:`~app.bot.run_bot` and shared by the
    metrics observer (live pipeline) and the post-call extraction. All ``add_*``
    methods are total and never raise — a usage-tally error must never break a
    live patient call.
    """

    def __init__(self) -> None:
        self._llm_tokens_in = 0
        self._llm_tokens_out = 0
        self._tts_chars = 0

    def add_llm_usage(self, in_tokens: int, out_tokens: int) -> None:
        """Add one LLM call's input/output token counts.

        ``None`` and negative inputs coerce to ``0`` so a malformed metrics
        frame or an absent Converse ``usage`` dict can't corrupt the tally or
        produce a negative cost downstream.
        """
        self._llm_tokens_in += _non_negative(in_tokens)
        self._llm_tokens_out += _non_negative(out_tokens)

    def add_tts_chars(self, n: int) -> None:
        """Add one TTS synthesis's character count (``None``/negative → ``0``)."""
        self._tts_chars += _non_negative(n)

    def totals(self) -> UsageTotals:
        """Return an immutable snapshot of the accumulated usage so far."""
        return UsageTotals(
            llm_tokens_in=self._llm_tokens_in,
            llm_tokens_out=self._llm_tokens_out,
            tts_chars=self._tts_chars,
        )


def _non_negative(value: int | None) -> int:
    """Coerce ``None`` / negative / non-int to a non-negative ``int``."""
    try:
        n = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0
