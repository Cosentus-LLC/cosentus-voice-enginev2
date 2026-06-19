"""Prompt hydration — substitute ``{{var}}`` placeholders with case_data values.

v1's hydrator (``app/hydrator.py``) was the reference; v2's version is
leaner. Three differences from v1:

* **No ``VOICE_WRAPPER``** — v1 prepended a 5-line "you are a voice AI,
  no markdown" banner. Bedrock's ``system_instruction`` channel
  already signals "this is a system message" so Claude is less likely
  to leak markdown; per-agent system prompts already carry their own
  voice-output guidance (Chris's prompt explicitly says "Your
  responses are put through a text to speech funnel"). Dropping the
  wrapper removes ~250 redundant tokens from every call's input.
* **No platform variable schema or allowlist** — v1 was already
  wildcard-substitution; v2 keeps that property explicit.
* **Hydration is now system-prompt + first_message** — v1 only
  hydrated ``system_prompt``. v2 expands so an agent can use
  placeholders like ``{{Service_Date}}`` in the opener too.

The single platform-injected variable is ``{{current_time}}``. Every
other placeholder is filled from the ``case_data`` dict supplied by
the caller (the dispatcher / batch row in production; ``{}`` for
inbound / unkeyed scenarios). Unfilled placeholders strip to empty
rather than leaking literal ``{{name}}`` syntax to the LLM.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

# Match ``{{name}}`` exactly — no whitespace inside the braces. Same
# pattern as v1's hydrator. Whitespace-padded variants like
# ``{{ name }}`` do NOT match, so agents who write them get a strip-
# to-empty result and can fix their template; this prevents
# accidental matches against unrelated double-brace text.
_PLACEHOLDER_RE = re.compile(r"\{\{[^{}\s]*\}\}")

# Format mirrors v1: "Wednesday, March 25, 2026 04:30 PM".
_CURRENT_TIME_FORMAT = "%A, %B %d, %Y %I:%M %p"


class MissingRequiredCaseDataError(ValueError):
    """A required ``case_data`` field is missing/blank before a live call (D2).

    Raised by the pre-call guard (``app.bot.bot.run_bot``) when an
    OUTBOUND call's ``case_data`` is missing one or more keys an
    operator has declared required (``Settings.required_case_data_keys``).
    The call is blocked rather than dialed with a blank patient name /
    claim id.

    Subclasses :class:`ValueError` so it sits alongside ``run_bot``'s
    existing pre-flight ``ValueError`` (missing ``agent_id``) — any
    caller already catching ``ValueError`` keeps catching this. The
    message lists the offending **key names** (never values, which
    would be PHI).
    """


def find_missing_required(
    case_data: Mapping[str, Any] | None,
    required_keys: Iterable[str],
) -> list[str]:
    """Return the required keys that are absent or blank in ``case_data``.

    A key counts as missing when it is not present in ``case_data``,
    its value is ``None``, or ``str(value).strip()`` is empty
    (whitespace-only counts as blank). The result is sorted and
    de-duplicated so callers get a deterministic list for logging and
    tests.

    Note — this is a *presence* check, deliberately distinct from
    :func:`hydrate_prompt`'s *substitution* semantics. ``hydrate_prompt``
    strips any falsy value (``0``, ``False``) to empty when filling a
    template; here ``0`` / ``False`` count as **present** (``str(0)``
    is ``"0"``, not blank). A genuinely-supplied numeric/boolean field
    is not a missing field — and no real required field (patient name,
    claim id) is ever ``0``.

    Args:
        case_data: Mapping of placeholder name → value. ``None`` is
            treated as ``{}`` (so every required key is flagged).
        required_keys: The keys that must be present and non-blank.
            Empty / no keys → always returns ``[]`` (the no-op case).

    Returns:
        Sorted, de-duplicated list of missing/blank required keys.
        Empty list means every required key is satisfied.
    """
    data = case_data or {}
    missing: set[str] = set()
    for key in required_keys:
        if key not in data:
            missing.add(key)
            continue
        value = data[key]
        if value is None or str(value).strip() == "":
            missing.add(key)
    return sorted(missing)


def hydrate_prompt(
    template: str | None,
    case_data: Mapping[str, Any] | None = None,
) -> str:
    """Substitute ``{{var}}`` placeholders in ``template`` from ``case_data``.

    Algorithm (in order):

    1. ``{{current_time}}`` is auto-injected with a formatted
       ``datetime.now()``. Inserted *before* ``case_data`` is
       applied so an agent author can override the format by
       passing ``current_time`` explicitly in ``case_data``.
    2. For every key in ``case_data``, replace ``{{key}}`` with
       ``str(value)`` — or empty string if the value is falsy
       (``None``, ``""``, ``0``, ``False``, etc.). Matches v1
       semantics: a missing-but-present key isn't a literal "0".
    3. Any placeholder still present after substitution is stripped
       to empty. An agent author misspelling ``{{Patiient_Name}}``
       gets nothing in the output rather than literal template
       syntax leaking to the LLM.

    Args:
        template: The prompt template, possibly containing
            ``{{name}}`` placeholders. ``None`` and ``""`` both
            return ``""``.
        case_data: Mapping of placeholder name → value. ``None`` is
            treated as ``{}``. Keys are matched verbatim (case-
            sensitive); the platform does not normalize.

    Returns:
        The hydrated string. Always a ``str``; never ``None``.

    Examples:
        >>> hydrate_prompt("Hi {{Name}}", {"Name": "Chris"})
        'Hi Chris'
        >>> hydrate_prompt("Today: {{current_time}}", None)  # doctest: +ELLIPSIS
        'Today: ...'
        >>> hydrate_prompt("DOS: {{Service_Date}}", {})
        'DOS: '
    """
    if not template:
        return ""

    result = template

    # Auto-inject current_time first so case_data can override it.
    result = result.replace(
        "{{current_time}}",
        datetime.now().strftime(_CURRENT_TIME_FORMAT),
    )

    if case_data:
        for key, value in case_data.items():
            placeholder = f"{{{{{key}}}}}"
            replacement = str(value) if value else ""
            result = result.replace(placeholder, replacement)

    # Strip any remaining unfilled placeholders.
    result = _PLACEHOLDER_RE.sub("", result)

    return result
