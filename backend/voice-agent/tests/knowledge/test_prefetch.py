"""Tests for app/knowledge/prefetch.py (#18 spike).

Covers:

* ``predict_lookups`` yields the expected payer-fact queries from context, and
  nothing when no payer is known.
* The warmer populates the cache *ahead of the turn*: after ``warm`` completes,
  a live read hits.
* The non-blocking guarantee: ``live_read`` on a miss returns immediately
  **without** awaiting the slow lookup, yet schedules a background fill that
  later lands (answer-now, fill-later).
* A failing lookup is swallowed (never breaks the call) and leaves a miss.
* Background fills are de-duped (no duplicate lookups for the same query).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.knowledge.fixtures import fixture_payer_lookup
from app.knowledge.interfaces import KnowledgeFetchContext
from app.knowledge.prefetch import PrefetchContext, PrefetchWarmer, predict_lookups
from app.knowledge.semantic_cache import LocalHashEmbeddingProvider, SemanticCache

# The fixture lookup sleeps this long; the live path must never pay it inline.
_SLOW_LOOKUP_S = 0.05


def test_predict_lookups_for_known_payer() -> None:
    queries = predict_lookups(PrefetchContext(payer="Aetna"))
    assert queries == [
        "timely filing limit for Aetna",
        "claims mailing address for Aetna",
    ]


def test_predict_lookups_without_payer_is_empty() -> None:
    assert predict_lookups(PrefetchContext(payer=None, user_text="hello")) == []
    assert predict_lookups(PrefetchContext(payer="   ")) == []


async def test_warm_populates_cache_ahead_of_turn() -> None:
    cache = SemanticCache()
    warmer = PrefetchWarmer(cache=cache, lookup_fn=fixture_payer_lookup)

    # Before warming, the live turn would miss.
    assert warmer.live_read("timely filing limit for Aetna") is None

    # Warm off the live path, then (only in the test) await the background tasks
    # to simulate "the caller is still talking" elapsing.
    tasks = warmer.warm(PrefetchContext(payer="Aetna"))
    await asyncio.gather(*tasks)

    # Now the live turn hits — the answer was fetched ahead of the turn.
    hit = warmer.live_read("timely filing limit for Aetna")
    assert hit is not None
    assert "120 days" in hit.value


async def test_live_read_miss_does_not_block_and_fills_later() -> None:
    cache = SemanticCache()
    warmer = PrefetchWarmer(cache=cache, lookup_fn=fixture_payer_lookup)

    # A miss must return immediately — far faster than the slow lookup itself.
    start = time.perf_counter()
    result = warmer.live_read("claims mailing address for Cigna")
    elapsed = time.perf_counter() - start

    assert result is None
    assert elapsed < _SLOW_LOOKUP_S  # did NOT await the slow lookup inline

    # But it scheduled a background fill; let it complete, then the next read hits.
    await asyncio.gather(*warmer._inflight.values())
    hit = warmer.live_read("claims mailing address for Cigna")
    assert hit is not None
    assert "PO Box 188061" in hit.value


async def test_failing_lookup_is_swallowed_and_leaves_a_miss() -> None:
    async def boom(query: str) -> str:
        raise RuntimeError("lookup backend down")

    cache = SemanticCache()
    warmer = PrefetchWarmer(cache=cache, lookup_fn=boom)

    tasks = warmer.warm(PrefetchContext(payer="Aetna"))
    await asyncio.gather(*tasks)  # must not raise

    # Failure degrades to a plain miss; the call is never impacted.
    assert warmer.live_read("timely filing limit for Aetna") is None
    assert cache.stats()["entries"] == 0


async def test_warm_dedupes_inflight_lookups() -> None:
    calls: list[str] = []

    async def counting_lookup(query: str) -> str:
        calls.append(query)
        await asyncio.sleep(_SLOW_LOOKUP_S)
        return "answer"

    cache = SemanticCache()
    warmer = PrefetchWarmer(cache=cache, lookup_fn=counting_lookup)

    # Two warms for the same payer back-to-back: the second must not relaunch
    # lookups already in flight from the first.
    tasks = warmer.warm(PrefetchContext(payer="Aetna"))
    tasks += warmer.warm(PrefetchContext(payer="Aetna"))
    await asyncio.gather(*tasks)

    # Exactly the two distinct predicted queries were fetched, once each.
    assert sorted(calls) == [
        "claims mailing address for Aetna",
        "timely filing limit for Aetna",
    ]


async def test_warm_skips_already_cached_queries() -> None:
    calls: list[str] = []

    async def counting_lookup(query: str) -> str:
        calls.append(query)
        return "answer"

    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "already known")
    warmer = PrefetchWarmer(cache=cache, lookup_fn=counting_lookup)

    tasks = warmer.warm(PrefetchContext(payer="Aetna"))
    await asyncio.gather(*tasks)

    # Only the not-yet-cached query was fetched.
    assert calls == ["claims mailing address for Aetna"]


@dataclass
class _CountingSource:
    calls: list[str]

    async def fetch(self, query: str, ctx: KnowledgeFetchContext | None = None) -> str | None:
        self.calls.append(query)
        await asyncio.sleep(_SLOW_LOOKUP_S)
        return "answer"


@dataclass
class _CountingEmbeddingProvider:
    calls: list[str]

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return await LocalHashEmbeddingProvider().embed(text)


def test_live_read_hit_does_not_call_source_or_embedding_provider() -> None:
    source_calls: list[str] = []
    embedding_calls: list[str] = []
    cache = SemanticCache()
    cache.put("timely filing limit for Aetna", "120 days.")
    warmer = PrefetchWarmer(
        cache=cache,
        knowledge_source=_CountingSource(source_calls),
        embedding_provider=_CountingEmbeddingProvider(embedding_calls),
    )

    hit = warmer.live_read("timely filing limit for Aetna")

    assert hit is not None
    assert source_calls == []
    assert embedding_calls == []


async def test_live_read_miss_returns_before_fetch_or_embed_complete() -> None:
    source_calls: list[str] = []
    embedding_calls: list[str] = []
    warmer = PrefetchWarmer(
        cache=SemanticCache(),
        knowledge_source=_CountingSource(source_calls),
        embedding_provider=_CountingEmbeddingProvider(embedding_calls),
    )

    start = time.perf_counter()
    result = warmer.live_read("timely filing limit for Aetna")
    elapsed = time.perf_counter() - start

    assert result is None
    assert elapsed < _SLOW_LOOKUP_S
    await asyncio.gather(*warmer._inflight.values())
    assert source_calls == ["timely filing limit for Aetna"]
    assert embedding_calls == ["timely filing limit for Aetna"]


async def test_aclose_cancels_inflight_tasks_and_clears_cache() -> None:
    never_finished = asyncio.Event()

    class _HangingSource:
        async def fetch(
            self,
            query: str,
            ctx: KnowledgeFetchContext | None = None,
        ) -> str | None:
            await never_finished.wait()
            return "answer"

    cache = SemanticCache()
    cache.put("already cached", "answer")
    warmer = PrefetchWarmer(
        cache=cache,
        knowledge_source=_HangingSource(),
        embedding_provider=LocalHashEmbeddingProvider(),
    )

    warmer.live_read("timely filing limit for Aetna")
    assert warmer._inflight

    await warmer.aclose()

    assert warmer._inflight == {}
    assert cache.stats()["entries"] == 0
