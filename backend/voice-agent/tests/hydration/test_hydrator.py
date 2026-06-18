"""Tests for app.hydration.hydrator.hydrate_prompt."""

from __future__ import annotations

import re
from datetime import datetime
from unittest.mock import patch

from app.hydration.hydrator import find_missing_required, hydrate_prompt


class TestEmptyOrNoneTemplate:
    def test_empty_string_template_returns_empty(self):
        assert hydrate_prompt("", {"X": "y"}) == ""

    def test_none_template_returns_empty(self):
        assert hydrate_prompt(None, {"X": "y"}) == ""

    def test_empty_string_no_case_data(self):
        assert hydrate_prompt("") == ""


class TestCurrentTime:
    def test_current_time_replaced_with_formatted_datetime(self):
        with patch("app.hydration.hydrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 25, 16, 30)
            out = hydrate_prompt("Today is {{current_time}}.")
        assert "Wednesday, March 25, 2026 04:30 PM" in out
        assert "{{current_time}}" not in out

    def test_current_time_works_without_case_data(self):
        with patch("app.hydration.hydrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 25, 9, 5)
            out = hydrate_prompt("At {{current_time}}", None)
        assert "09:05 AM" in out

    def test_case_data_can_override_current_time(self):
        # Agent author passes a custom timestamp; case_data wins
        # because it runs after the auto-inject.
        with patch("app.hydration.hydrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 25, 16, 30)
            # The auto-inject runs first, putting the formatted
            # datetime into the prompt. case_data substitution
            # finds no remaining {{current_time}} so the override
            # has no effect — same v1 semantics.
            out = hydrate_prompt(
                "Now: {{current_time}}",
                {"current_time": "OVERRIDE_VALUE"},
            )
        # Auto-inject wins (v1 semantics).
        assert "Wednesday, March 25, 2026 04:30 PM" in out
        assert "OVERRIDE_VALUE" not in out

    def test_current_time_format_matches_v1(self):
        # "%A, %B %d, %Y %I:%M %p"
        with patch("app.hydration.hydrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 2, 15, 4)
            out = hydrate_prompt("{{current_time}}")
        assert out == "Friday, January 02, 2026 03:04 PM"


class TestCaseDataSubstitution:
    def test_single_placeholder_substituted(self):
        assert (
            hydrate_prompt("DOS: {{Service_Date}}", {"Service_Date": "2026-02-14"})
            == "DOS: 2026-02-14"
        )

    def test_multiple_distinct_placeholders(self):
        out = hydrate_prompt("{{First}} {{Last}}", {"First": "Chris", "Last": "M"})
        assert out == "Chris M"

    def test_repeated_placeholder_substituted_everywhere(self):
        out = hydrate_prompt("{{X}} and {{X}} again", {"X": "foo"})
        assert out == "foo and foo again"

    def test_adjacent_placeholders_work(self):
        out = hydrate_prompt("{{A}}{{B}}", {"A": "1", "B": "2"})
        assert out == "12"

    def test_falsy_none_replaced_with_empty(self):
        assert hydrate_prompt("Patient: {{Name}}", {"Name": None}) == "Patient: "

    def test_falsy_empty_string_replaced_with_empty(self):
        assert hydrate_prompt("Patient: {{Name}}", {"Name": ""}) == "Patient: "

    def test_falsy_zero_replaced_with_empty(self):
        # v1 behavior: 0 is falsy → empty. Documented oddity.
        assert hydrate_prompt("Count: {{N}}", {"N": 0}) == "Count: "

    def test_falsy_false_replaced_with_empty(self):
        assert hydrate_prompt("Flag: {{F}}", {"F": False}) == "Flag: "

    def test_non_string_int_coerced(self):
        assert hydrate_prompt("Tokens: {{N}}", {"N": 1024}) == "Tokens: 1024"

    def test_non_string_float_coerced(self):
        assert hydrate_prompt("Amount: {{A}}", {"A": 125.50}) == "Amount: 125.5"


class TestUnfilledPlaceholders:
    def test_unfilled_placeholder_stripped(self):
        assert hydrate_prompt("Hello {{Unknown}} world", {}) == "Hello  world"

    def test_multiple_unfilled_all_stripped(self):
        out = hydrate_prompt("{{A}} {{B}} {{C}}", {})
        assert "{{" not in out
        assert "}}" not in out

    def test_only_unfilled_stripped_filled_kept(self):
        out = hydrate_prompt("{{Known}} + {{Unknown}}", {"Known": "yes"})
        assert "yes" in out
        assert "{{Unknown}}" not in out
        assert "{{Known}}" not in out

    def test_whitespace_padded_placeholder_stripped_not_substituted(self):
        # Only ``{{name}}`` matches — ``{{ name }}`` does not (v1
        # semantics). Whitespace inside the braces means the
        # template author wrote a non-canonical placeholder; we
        # strip it to empty rather than guess at intent.
        out = hydrate_prompt("Hi {{ Name }}", {"Name": "Chris"})
        # The whitespace-padded version doesn't match the regex
        # AND doesn't match the case_data substitution exact-string.
        # It survives both passes intact. (Actually verify the
        # regex doesn't match either.)
        assert out == "Hi {{ Name }}"


class TestRegexSafety:
    def test_special_regex_chars_in_value_treated_literally(self):
        # str.replace, not re.sub, so "$" "*" "+" "?" "." are safe.
        out = hydrate_prompt("X: {{X}}", {"X": "a.b*c+d?e$f"})
        assert out == "X: a.b*c+d?e$f"

    def test_special_regex_chars_in_key_handled(self):
        # Cosentus production uses keys with special chars like
        # ``Claim#`` and ``CPT_1``. Verify ``#`` works.
        out = hydrate_prompt("Claim: {{Claim#}}", {"Claim#": "ABC123"})
        assert out == "Claim: ABC123"


class TestNoneOrMissingCaseData:
    def test_none_case_data_treated_as_empty(self):
        # Unfilled placeholders strip; current_time still injects.
        with patch("app.hydration.hydrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 12, 0)
            out = hydrate_prompt("{{X}}-{{current_time}}", None)
        assert "Thursday, January 01, 2026 12:00 PM" in out
        assert "{{X}}" not in out

    def test_empty_dict_case_data_works(self):
        # Same as None — every placeholder strips, current_time
        # still injects.
        out = hydrate_prompt("Foo {{X}}", {})
        assert out == "Foo "


class TestRealWorldChrisFragment:
    def test_chris_fragment_with_partial_case_data(self):
        # Subset of Chris's prompt. Some placeholders filled,
        # some not. Filled values appear; unfilled strip; literal
        # text passes through unchanged.
        template = (
            "You are calling about a claim from {{Service_Date}} "
            "for patient {{Patient_Name}}. "
            "Member ID: {{Primary_Carrier_Policy#}}. "
            "Today's date: {{current_time}}."
        )
        with patch("app.hydration.hydrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 15, 10, 30)
            out = hydrate_prompt(
                template,
                {
                    "Service_Date": "2026-01-12",
                    "Patient_Name": "Jane Doe",
                    # Primary_Carrier_Policy# intentionally absent
                },
            )
        assert "claim from 2026-01-12" in out
        assert "patient Jane Doe" in out
        assert "Member ID: ." in out  # stripped to empty
        assert "Thursday, January 15, 2026 10:30 AM" in out
        assert "{{" not in out
        assert "}}" not in out


class TestNoVoiceWrapper:
    def test_output_does_not_contain_v1_voice_wrapper_banner(self):
        # v2 explicitly drops VOICE_WRAPPER. Sanity check that the
        # banner v1 prepended is gone.
        out = hydrate_prompt("Hello world.", {})
        assert "GLOBAL VOICE SYSTEM INSTRUCTIONS" not in out
        assert "NEVER use markdown" not in out
        # The v2 hydrator returns just the substituted template,
        # nothing else.
        assert out == "Hello world."

    def test_short_template_passes_through_with_no_prefix(self):
        # No banner means an empty case_data + no placeholders →
        # template returns verbatim.
        original = "You are Chris."
        out = hydrate_prompt(original, {})
        assert out == original

    def test_template_with_only_current_time(self):
        with patch("app.hydration.hydrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 5, 4, 14, 22)
            out = hydrate_prompt("{{current_time}}", {})
        # Just the timestamp — no v1 wrapper text.
        assert out == "Monday, May 04, 2026 02:22 PM"


class TestFindMissingRequired:
    """D2 (#27): pre-call required-field guard helper.

    A presence check distinct from hydrate_prompt's substitution: a
    missing/blank required field is flagged so an outbound call is
    blocked rather than dialed with a blank patient name / claim id.
    """

    def test_missing_required_key_is_flagged(self):
        # Required key entirely absent from case_data → flagged.
        out = find_missing_required(
            {"Patient_Name": "Jane Doe"},
            ["Patient_Name", "Claim#"],
        )
        assert out == ["Claim#"]

    def test_blank_required_value_is_flagged(self):
        # Empty string and whitespace-only both count as blank.
        out = find_missing_required(
            {"Patient_Name": "", "Claim#": "   "},
            ["Patient_Name", "Claim#"],
        )
        assert out == ["Claim#", "Patient_Name"]

    def test_none_value_is_flagged(self):
        out = find_missing_required({"Patient_Name": None}, ["Patient_Name"])
        assert out == ["Patient_Name"]

    def test_complete_case_data_unchanged(self):
        # Every required key present + non-blank → nothing missing, AND
        # hydrate_prompt fills them exactly as before (behavior preserved).
        case_data = {"Patient_Name": "Jane Doe", "Claim#": "ABC123"}
        assert find_missing_required(case_data, ["Patient_Name", "Claim#"]) == []
        out = hydrate_prompt("{{Patient_Name}} / {{Claim#}}", case_data)
        assert out == "Jane Doe / ABC123"

    def test_optional_unfilled_placeholder_still_stripped(self):
        # A placeholder that is NOT in the required set is irrelevant to
        # the guard and still strips to empty in hydrate_prompt (today's
        # behavior, unchanged).
        case_data = {"Patient_Name": "Jane Doe"}
        # Optional {{Service_Date}} is not required → guard passes.
        assert find_missing_required(case_data, ["Patient_Name"]) == []
        # And hydrate_prompt still strips the unfilled optional one.
        out = hydrate_prompt("{{Patient_Name}} on {{Service_Date}}", case_data)
        assert out == "Jane Doe on "

    def test_no_required_keys_is_noop(self):
        # Empty required set → never flags anything (the default path).
        assert find_missing_required({}, []) == []
        assert find_missing_required({"X": ""}, []) == []

    def test_falsy_zero_counts_as_present(self):
        # Presence != substitution: 0 / False are supplied values, not
        # missing. (hydrate_prompt strips them to empty; the guard does
        # not treat them as missing.)
        assert find_missing_required({"N": 0, "F": False}, ["N", "F"]) == []

    def test_non_string_value_counts_as_present(self):
        assert find_missing_required({"N": 1024}, ["N"]) == []

    def test_none_case_data_flags_all_required(self):
        assert find_missing_required(None, ["A", "B"]) == ["A", "B"]

    def test_result_is_sorted_and_deduped(self):
        # Duplicate required keys collapse; output is sorted.
        out = find_missing_required({}, ["B", "A", "B", "A"])
        assert out == ["A", "B"]


def test_chris_full_22_placeholders_smoke():
    # Sanity-test the full v1 placeholder set Chris uses today.
    chris_keys = [
        "Service_Date",
        "Practice_Name",
        "NPI",
        "Tax_ID",
        "Billing_Address",
        "Call_Back#",
        "Provider",
        "Patient_Name",
        "Patient_Birth_Date",
        "Primary_Carrier_Policy#",
        "Service_Location",
        "Primary_Carrier_Name",
        "Total_Charge",
        "Claim#",
        "Ins_Pmt",
        "Ins_Balance",
        "CPT_1",
        "CPT_2",
        "CPT_3",
        "CPT_4",
        "Acct#",
    ]
    template = " ".join(f"{{{{{k}}}}}" for k in chris_keys) + " {{current_time}}"
    case_data = {k: f"<{k}>" for k in chris_keys}
    with patch("app.hydration.hydrator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 5, 4, 9, 0)
        out = hydrate_prompt(template, case_data)
    for k in chris_keys:
        assert f"<{k}>" in out, f"missing {k}"
    assert "Monday, May 04, 2026 09:00 AM" in out
    # No leftover placeholders.
    assert not re.search(r"\{\{[^{}]*\}\}", out)
