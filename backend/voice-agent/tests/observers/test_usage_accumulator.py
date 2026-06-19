"""Tests for ``app/observers/usage_accumulator.py`` — per-call usage tally (#28)."""

from __future__ import annotations

from app.observers.usage_accumulator import UsageAccumulator, UsageTotals


def test_starts_at_zero():
    assert UsageAccumulator().totals() == UsageTotals(
        llm_tokens_in=0, llm_tokens_out=0, tts_chars=0
    )


def test_add_llm_usage_accumulates():
    acc = UsageAccumulator()
    acc.add_llm_usage(100, 20)
    acc.add_llm_usage(50, 5)
    totals = acc.totals()
    assert totals.llm_tokens_in == 150
    assert totals.llm_tokens_out == 25
    assert totals.tts_chars == 0


def test_add_tts_chars_accumulates():
    acc = UsageAccumulator()
    acc.add_tts_chars(88)
    acc.add_tts_chars(12)
    assert acc.totals().tts_chars == 100


def test_none_coerced_to_zero():
    """A missing Converse ``usage`` dict passes ``None`` — must not raise."""
    acc = UsageAccumulator()
    acc.add_llm_usage(None, None)  # type: ignore[arg-type]
    acc.add_tts_chars(None)  # type: ignore[arg-type]
    assert acc.totals() == UsageTotals()


def test_negative_coerced_to_zero():
    """A malformed metrics frame can't drive a count (or a cost) negative."""
    acc = UsageAccumulator()
    acc.add_llm_usage(-10, -5)
    acc.add_tts_chars(-3)
    assert acc.totals() == UsageTotals()


def test_non_numeric_coerced_to_zero():
    acc = UsageAccumulator()
    acc.add_llm_usage("oops", 7)  # type: ignore[arg-type]
    assert acc.totals().llm_tokens_in == 0
    assert acc.totals().llm_tokens_out == 7


def test_totals_is_immutable_snapshot():
    """``totals()`` returns a frozen snapshot — later adds don't mutate it."""
    acc = UsageAccumulator()
    acc.add_llm_usage(10, 2)
    snapshot = acc.totals()
    acc.add_llm_usage(90, 8)
    assert snapshot.llm_tokens_in == 10  # unchanged
    assert acc.totals().llm_tokens_in == 100
