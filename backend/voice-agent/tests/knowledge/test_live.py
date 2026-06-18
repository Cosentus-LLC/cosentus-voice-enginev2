"""Tests for live knowledge-prefetch wiring (#56)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.knowledge.live import build_prefetch_context, wire_knowledge_prefetch_handler


def test_build_prefetch_context_uses_configured_payer_key() -> None:
    ctx = build_prefetch_context(
        case_data={"payer_name": " Aetna ", "patient_name": "Jane"},
        payer_key="payer_name",
        user_text="What is the filing limit?",
    )

    assert ctx.payer == "Aetna"
    assert ctx.user_text == "What is the filing limit?"


def test_build_prefetch_context_missing_payer_is_empty() -> None:
    ctx = build_prefetch_context(case_data={}, payer_key="payer_name")

    assert ctx.payer is None


class _UserAggregator:
    def __init__(self) -> None:
        self.handlers = {}

    def event_handler(self, event_name: str):
        def decorator(fn):
            self.handlers[event_name] = fn
            return fn

        return decorator


class _Pair:
    def __init__(self, user_aggregator: _UserAggregator) -> None:
        self._user_aggregator = user_aggregator

    def user(self) -> _UserAggregator:
        return self._user_aggregator


async def test_turn_boundary_handler_calls_warm_without_awaiting_tasks() -> None:
    user = _UserAggregator()
    warmer = MagicMock()
    warmer.warm.return_value = [MagicMock()]
    wire_knowledge_prefetch_handler(
        aggregator_pair=_Pair(user),
        warmer=warmer,
        case_data={"payer_name": "Aetna"},
        payer_key="payer_name",
    )

    await user.handlers["on_user_turn_stopped"](
        None,
        None,
        SimpleNamespace(content="deadline?", timestamp=None),
    )

    warmer.warm.assert_called_once()
    ctx = warmer.warm.call_args.args[0]
    assert ctx.payer == "Aetna"
    assert ctx.user_text == "deadline?"


async def test_turn_boundary_handler_does_not_log_case_data_values() -> None:
    user = _UserAggregator()
    warmer = MagicMock()
    warmer.warm.return_value = []
    wire_knowledge_prefetch_handler(
        aggregator_pair=_Pair(user),
        warmer=warmer,
        case_data={"payer_name": "Aetna", "patient_name": "Jane Doe"},
        payer_key="payer_name",
    )

    await user.handlers["on_user_turn_stopped"](
        None,
        None,
        SimpleNamespace(content="hello", timestamp=None),
    )

    # The hook may pass the payer into the warmer, but should not pass raw
    # case_data or patient fields into warm().
    args = warmer.warm.call_args.args
    assert len(args) == 1
    assert not hasattr(args[0], "case_data")
