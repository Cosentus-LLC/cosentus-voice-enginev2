"""De-identified payer-knowledge fixtures for the #18 spike.

**No PHI.** Every fact here is a *payer-level*, publicly-known operational
detail (timely-filing windows, claims mailing addresses) — the kind of thing a
biller looks up regardless of which patient is calling. There is no patient
name, DOB, claim id, or any other identifier in this file. The real
system-of-record for these facts (#17) is out of scope for the spike; this
fixture stands in for it so the prefetch/cache pattern can be proven offline.

:func:`fixture_payer_lookup` deliberately simulates the latency of a real
network/RAG lookup (``_SIMULATED_LATENCY_S``) so tests can prove the live turn
never pays that cost on the warm path.
"""

from __future__ import annotations

import asyncio

from app.knowledge.interfaces import KnowledgeFetchContext

# Stand-in latency for a "real" payer-knowledge lookup (RAG/DB/HTTP). Kept small
# so the suite stays fast, but large relative to a cache ``get`` (microseconds),
# which is the whole point: the live turn must never pay this inline.
_SIMULATED_LATENCY_S = 0.05

# Public, payer-level facts only — see module docstring. Keyed by a normalized
# payer name. Values are short answer strings the agent could speak.
_PAYER_FACTS: dict[str, dict[str, str]] = {
    "aetna": {
        "timely_filing": "Aetna's timely filing limit for participating providers is 120 days "
        "from the date of service.",
        "claims_address": "Aetna paper claims go to PO Box 14079, Lexington, KY 40512.",
    },
    "cigna": {
        "timely_filing": "Cigna's timely filing limit is 90 days from the date of service for "
        "participating providers.",
        "claims_address": "Cigna paper claims go to PO Box 188061, Chattanooga, TN 37422.",
    },
    "united healthcare": {
        "timely_filing": "UnitedHealthcare's timely filing limit is 90 days from the date of "
        "service for most commercial plans.",
        "claims_address": "UnitedHealthcare paper claims go to PO Box 30555, Salt Lake City, "
        "UT 84130.",
    },
}


def _normalize_payer(name: str) -> str:
    return " ".join(name.lower().split())


async def fixture_payer_lookup(query: str) -> str:
    """Simulated slow payer-fact lookup. Returns an answer string for *query*.

    Recognizes queries shaped like ``"timely filing limit for <payer>"`` and
    ``"claims mailing address for <payer>"`` (the two lookup types the warmer
    predicts). Sleeps ``_SIMULATED_LATENCY_S`` to stand in for real network /
    RAG latency, then returns the matching fact, or a graceful "not found"
    string if the payer/fact isn't in the fixture.

    This is the function the **warmer** awaits off the live path — never the
    live turn.
    """
    await asyncio.sleep(_SIMULATED_LATENCY_S)
    lowered = query.lower()
    if "timely filing" in lowered:
        fact_key = "timely_filing"
    elif "claims" in lowered and "address" in lowered:
        fact_key = "claims_address"
    else:
        return f"No payer fact available for query: {query!r}."
    for payer, facts in _PAYER_FACTS.items():
        if payer in _normalize_payer(lowered):
            return facts[fact_key]
    return f"No payer fact available for query: {query!r}."


class DeidentifiedFixtureKnowledgeSource:
    """Default slow source backed by de-identified payer-level fixtures."""

    async def fetch(self, query: str, ctx: KnowledgeFetchContext | None = None) -> str | None:
        """Fetch a payer-level fixture answer.

        ``ctx`` is accepted to satisfy the production interface; this fixture
        uses only the query text and contains no PHI.
        """
        _ = ctx
        return await fixture_payer_lookup(query)
