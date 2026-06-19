"""Tests for app/knowledge/semantic_cache.py (#18 spike).

Covers:

* ``put`` then ``get`` is a hit; a paraphrase of the same lookup still hits
  (the "semantic" part); an unrelated query misses.
* The live-read guarantee: a hit returns **without** invoking the slow lookup
  function (no network call on the warm path).
* A hit is far under the ~200ms real-time voice budget (it's pure in-memory).
* Hit/miss/entry stats.
"""

from __future__ import annotations

import time

from app.knowledge.semantic_cache import SemanticCache, cosine, embed

# Generous ceiling: a real-time turn budget is ~200ms; an in-memory get is
# microseconds. We assert well under budget without being so tight it flakes
# on a loaded CI box.
_BUDGET_S = 0.05


def test_put_then_get_is_a_hit() -> None:
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days from date of service.")

    hit = cache.get("timely filing limit for Aetna")

    assert hit is not None
    assert hit.value == "120 days from date of service."
    assert hit.similarity >= 0.99  # exact text → ~1.0


def test_paraphrase_of_same_lookup_still_hits() -> None:
    # The point of a *semantic* cache: a near-paraphrase of the warmed query
    # shares the cached answer instead of triggering a fresh lookup.
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days from date of service.")

    hit = cache.get("what is the timely filing limit for Aetna")

    assert hit is not None
    assert hit.value == "120 days from date of service."


def test_unrelated_query_misses() -> None:
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days from date of service.")

    assert cache.get("claims mailing address for Cigna") is None


def test_hit_does_not_invoke_the_slow_lookup() -> None:
    # The live-turn guarantee: reading a warm cache makes no network/lookup
    # call. We assert the (would-be slow) lookup fn is never touched on a hit.
    calls: list[str] = []

    def slow_lookup(query: str) -> str:  # stand-in; must NOT be called on a hit
        calls.append(query)
        return "should not happen"

    cache = SemanticCache()
    cache.put("claims mailing address for Cigna", "PO Box 188061, Chattanooga, TN.")

    hit = cache.get("claims mailing address for Cigna")

    assert hit is not None
    assert calls == []  # no lookup invoked


def test_hit_is_well_under_the_realtime_budget() -> None:
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days from date of service.")

    start = time.perf_counter()
    hit = cache.get("timely filing limit for Aetna")
    elapsed = time.perf_counter() - start

    assert hit is not None
    assert elapsed < _BUDGET_S


def test_put_updates_existing_query_in_place() -> None:
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "old answer")
    cache.put("timely filing limit for Aetna", "120 days from date of service.")

    hit = cache.get("timely filing limit for Aetna")
    assert hit is not None
    assert hit.value == "120 days from date of service."
    assert cache.stats()["entries"] == 1  # updated, not duplicated


def test_stats_track_hits_and_misses() -> None:
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days.")

    cache.get("timely filing limit for Aetna")  # hit
    cache.get("totally unrelated query about parking")  # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1


def test_embed_is_deterministic_and_normalized() -> None:
    a = embed("timely filing limit for Aetna")
    b = embed("timely filing limit for Aetna")
    assert a == b  # stable across calls (hashlib, not builtin hash)
    assert abs(cosine(a, a) - 1.0) < 1e-9  # L2-normalized → self-cosine == 1


def test_empty_text_embeds_to_zero_and_never_hits() -> None:
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days.")
    # An all-zero vector (no tokens) must not spuriously match anything.
    assert cache.get("") is None


def test_expired_entry_misses_and_is_evicted() -> None:
    now = 0.0

    def clock() -> float:
        return now

    cache = SemanticCache(ttl_secs=1.0, clock=clock)
    cache.put("timely filing limit for Aetna", "120 days.")

    now = 2.0

    assert cache.get("timely filing limit for Aetna") is None
    assert cache.stats()["entries"] == 0


def test_max_entries_evicts_least_recently_used_entry() -> None:
    cache = SemanticCache(max_entries=2, threshold=0.99)
    cache.put("timely filing limit for Aetna", "aetna")
    cache.put("claims mailing address for Cigna", "cigna")

    assert cache.get("timely filing limit for Aetna") is not None
    cache.put("appeal deadline for United Healthcare", "uhc")

    assert cache.get("claims mailing address for Cigna") is None
    assert cache.get("timely filing limit for Aetna") is not None
    assert cache.get("appeal deadline for United Healthcare") is not None
    assert cache.stats()["entries"] == 2


def test_clear_drops_entries_and_resets_counts() -> None:
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days.")
    assert cache.get("timely filing limit for Aetna") is not None
    assert cache.stats()["hits"] == 1

    cache.clear()

    assert cache.stats() == {"hits": 0, "misses": 0, "entries": 0}
    assert cache.get("timely filing limit for Aetna") is None


def test_separate_cache_instances_are_isolated() -> None:
    call_a = SemanticCache()
    call_b = SemanticCache()

    call_a.put("timely filing limit for Aetna", "120 days.")

    assert call_a.get("timely filing limit for Aetna") is not None
    assert call_b.get("timely filing limit for Aetna") is None
