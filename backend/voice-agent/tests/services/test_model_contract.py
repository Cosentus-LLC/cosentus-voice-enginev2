"""Contract test — the engine model map must not drift from the API allowlist.

The set of *accepted short model names* is owned by the API
(``api-lambda-v2`` → ``lib/agent-schema.ts`` → ``VALID_LLM_MODELS``, a
single Zod source). The engine's ``_SHORT_TO_BEDROCK`` table
(``app/services/factory.py``) is the engine's view of that allowlist:
it translates each accepted short name into a Bedrock inference-profile
ID. If a model is added on one side only, the two **drift** and live
calls 400 — exactly the 2026-05-27 production incident (tech-debt
Entry 19, where the Lambda's allowlist had three stale entries and the
Haiku migration 400'd until the lists were re-synced).

This test makes that drift impossible to ship silently: it fails at CI
time the moment the engine map's keys diverge from the checked-in
mirror of the API allowlist below. It is intentionally a *separate*
file from ``test_factory.py`` so the cross-repo contract is visible at
a glance.
"""

from __future__ import annotations

from app.services.factory import (
    _BEDROCK_DEFAULT_MODEL,
    _SHORT_TO_BEDROCK,
    resolve_bedrock_model_id,
)

# ─────────────────────────────────────────────────────────────────────────────
#  KEEP IN SYNC with api-lambda-v2 ``lib/agent-schema.ts`` → ``VALID_LLM_MODELS``.
#
#  This is a CHECKED-IN MIRROR of the API's allowlist, not the source.
#  The API is the single source of truth (Zod, after API #5). This is a
#  cross-repo contract: when a model is added/removed in the API's
#  ``VALID_LLM_MODELS``, update BOTH sides — this mirror AND
#  ``_SHORT_TO_BEDROCK`` in ``app/services/factory.py`` — in lockstep.
#
#  Bare short-names only. The API stores both bare and dated forms
#  (tech-debt Entry 19, Lambda commit ec17916), but the engine resolver
#  normalizes a trailing ``-YYYYMMDD`` to the bare form before lookup,
#  so the bare set is the contract surface that must match.
# ─────────────────────────────────────────────────────────────────────────────
_API_VALID_LLM_MODELS: frozenset[str] = frozenset(
    {
        "claude-sonnet-4",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
    }
)


def test_model_map_matches_api_allowlist():
    """The engine's accepted short-names (bare keys of ``_SHORT_TO_BEDROCK``)
    must exactly equal the API allowlist mirror. Fails on drift in either
    direction so neither side can add a model the other rejects.
    """
    engine_keys = set(_SHORT_TO_BEDROCK)
    api_models = set(_API_VALID_LLM_MODELS)

    engine_only = engine_keys - api_models
    api_only = api_models - engine_keys

    assert engine_keys == api_models, (
        "Engine model map has drifted from the API allowlist — calls for the "
        "mismatched model(s) will 400 (tech-debt Entry 19).\n"
        f"  In engine `_SHORT_TO_BEDROCK` but NOT in API allowlist: "
        f"{sorted(engine_only) or 'none'}\n"
        f"  In API allowlist but NOT in engine `_SHORT_TO_BEDROCK`: "
        f"{sorted(api_only) or 'none'}\n"
        "Fix: add/remove the model in BOTH api-lambda-v2 "
        "`lib/agent-schema.ts` (VALID_LLM_MODELS) and "
        "app/services/factory.py (_SHORT_TO_BEDROCK), then update the "
        "mirror in this test."
    )


def test_resolve_strips_dated_variant():
    """A dated short form (``-YYYYMMDD``) resolves to the same Bedrock
    inference profile as its un-dated form — so the API storing either
    form maps to one engine table entry (tech-debt Entry 19, the
    inbound-PSTN ValidationException).
    """
    dated = "claude-haiku-4-5-20251001"
    undated = "claude-haiku-4-5"
    assert resolve_bedrock_model_id(dated) == resolve_bedrock_model_id(undated)
    assert resolve_bedrock_model_id(dated) == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_unknown_model_passes_through_not_default():
    """An unknown short form is returned AS-IS — never silently swapped
    for ``_BEDROCK_DEFAULT_MODEL``. The silent-wrong-model swap is the
    real danger (a call would run on the wrong model with no error);
    passing through lets Bedrock reject with a clear error. Raising is
    intentionally out of scope (that's §0.6, and reverses the documented
    2026-04-24 pass-through fix).
    """
    unknown = "claude-mystery-9-9"
    result = resolve_bedrock_model_id(unknown)
    assert result == unknown
    assert result != _BEDROCK_DEFAULT_MODEL
