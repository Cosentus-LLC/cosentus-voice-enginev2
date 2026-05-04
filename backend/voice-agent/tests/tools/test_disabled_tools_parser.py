"""Tests for app.tools.registry.parse_disabled_tools — closes tech-debt entry 5."""

from __future__ import annotations

from app.tools.registry import parse_disabled_tools


class TestParseDisabledTools:
    def test_empty_string_returns_empty_set(self):
        assert parse_disabled_tools("") == set()

    def test_none_returns_empty_set(self):
        assert parse_disabled_tools(None) == set()

    def test_single_name_returns_singleton(self):
        assert parse_disabled_tools("end_call") == {"end_call"}

    def test_csv_returns_set(self):
        assert parse_disabled_tools("end_call,press_digit") == {
            "end_call",
            "press_digit",
        }

    def test_strips_whitespace_around_each_entry(self):
        assert parse_disabled_tools("  end_call ,  press_digit  ") == {
            "end_call",
            "press_digit",
        }

    def test_drops_empty_entries_from_trailing_or_double_commas(self):
        assert parse_disabled_tools("end_call,,press_digit,") == {
            "end_call",
            "press_digit",
        }

    def test_pure_whitespace_returns_empty_set(self):
        assert parse_disabled_tools("   ,   ,   ") == set()

    def test_preserves_underscores_and_case(self):
        # We don't normalize case — match must be exact.
        assert parse_disabled_tools("End_Call,PRESS_DIGIT") == {
            "End_Call",
            "PRESS_DIGIT",
        }
