"""Live-pipeline wiring helpers for knowledge prefetch (#56)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import structlog
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    UserTurnStoppedMessage,
)

from app.knowledge.prefetch import PrefetchContext, PrefetchWarmer

logger = structlog.get_logger(__name__)


def build_prefetch_context(
    *,
    case_data: Mapping[str, Any],
    payer_key: str,
    user_text: str = "",
) -> PrefetchContext:
    """Build the PHI-minimized context used by fixture-backed prefetch."""
    payer = ""
    key = payer_key.strip()
    if key:
        payer = str(case_data.get(key) or "").strip()
    return PrefetchContext(payer=payer or None, user_text=user_text)


def wire_knowledge_prefetch_handler(
    *,
    aggregator_pair: LLMContextAggregatorPair,
    warmer: PrefetchWarmer,
    case_data: Mapping[str, Any],
    payer_key: str,
) -> None:
    """Attach a turn-boundary warmer hook.

    The handler fires once per completed user turn. It schedules background
    fills and never awaits them.
    """
    user_aggregator = aggregator_pair.user()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def _on_user_turn_stopped(
        _aggregator,
        _strategy,
        message: UserTurnStoppedMessage,
    ) -> None:
        try:
            text = (message.content or "").strip()
            ctx = build_prefetch_context(
                case_data=case_data,
                payer_key=payer_key,
                user_text=text,
            )
            tasks = warmer.warm(ctx)
            logger.debug(
                "knowledge_prefetch_turn_boundary",
                case_data_keys=sorted(case_data.keys()),
                payer_key=payer_key,
                predicted_tasks=len(tasks),
                user_text_chars=len(text),
            )
        except Exception as exc:  # noqa: BLE001 — observation must never break calls
            logger.warning(
                "knowledge_prefetch_handler_failed",
                error_type=type(exc).__name__,
            )
