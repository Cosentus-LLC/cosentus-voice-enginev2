"""Predictive prefetch — the off-live-path "slow thinker" that warms the cache (#18).

While the caller is still talking, :func:`predict_lookups` guesses the lookups
the agent is about to need (for *this* payer / *this* claim) and
:class:`PrefetchWarmer` fetches them in the background and drops the answers into
the :class:`~app.knowledge.semantic_cache.SemanticCache`. By the time the agent
takes its turn, the answer is already cached.

The contract that keeps the call non-blocking
----------------------------------------------

* :meth:`PrefetchWarmer.warm` spawns ``asyncio`` background tasks and returns
  immediately. Callers **must not** ``await`` the slow lookup inline — they may
  ``await`` the returned tasks only from off-the-live-path code (e.g. tests, or
  a future "between turns" hook).
* :meth:`PrefetchWarmer.live_read` is what the **live turn** calls. It reads the
  cache only (synchronous, microseconds). On a hit it returns the answer; on a
  miss it returns ``None`` immediately **and** kicks off a background fill so
  the next turn can have it — answer-now, fill-in-a-beat-later. It never
  ``await``s the lookup.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from app.knowledge.interfaces import EmbeddingProvider, KnowledgeFetchContext, KnowledgeSource
from app.knowledge.semantic_cache import CacheHit, LocalHashEmbeddingProvider, SemanticCache

logger = structlog.get_logger(__name__)

# A slow lookup: query string in, answer string out. Async because the real one
# is network/RAG-bound; the spike's :func:`app.knowledge.fixtures.fixture_payer_lookup`
# fits this shape.
LookupFn = Callable[[str], Awaitable[str]]


@dataclass
class PrefetchContext:
    """What the warmer knows about the in-progress turn.

    Deliberately minimal for the spike: the payer (drives the one proven lookup
    type) plus optional claim id and the caller's partial utterance, which a
    richer predictor could mine. ``case_data``-style identifiers beyond these
    are intentionally absent — the spike predicts on payer alone.
    """

    payer: str | None = None
    claim_id: str | None = None
    user_text: str = ""


# The lookup types the spike knows how to predict + warm, as query templates.
# One payer can fan out to several queries; the warmer de-dupes against the
# cache before fetching. A real predictor would learn these from call flow.
_PAYER_LOOKUP_TEMPLATES: tuple[str, ...] = (
    "timely filing limit for {payer}",
    "claims mailing address for {payer}",
)


def predict_lookups(ctx: PrefetchContext) -> list[str]:
    """Predict the lookup queries the agent is likely to need next.

    Spike scope: when a payer is known, predict the payer-fact lookups
    (:data:`_PAYER_LOOKUP_TEMPLATES`). No payer → no prediction (empty list),
    which the warmer treats as "nothing to warm". Pure and side-effect-free so
    it is trivially testable; the warmer owns the I/O.
    """
    if not ctx.payer or not ctx.payer.strip():
        return []
    payer = ctx.payer.strip()
    return [template.format(payer=payer) for template in _PAYER_LOOKUP_TEMPLATES]


class PrefetchWarmer:
    """Runs predicted lookups off the live path and fills the semantic cache.

    Args:
        cache: the cache the live turn reads.
        knowledge_source: the slow source the warmer (never the live turn) awaits.
        embedding_provider: async embedder used only during background fills.
    """

    def __init__(
        self,
        cache: SemanticCache,
        lookup_fn: LookupFn | None = None,
        *,
        knowledge_source: KnowledgeSource | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.cache = cache
        if knowledge_source is None:
            if lookup_fn is None:
                raise TypeError("knowledge_source is required")
            knowledge_source = _LookupFnKnowledgeSource(lookup_fn)
        self.knowledge_source = knowledge_source
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        # Tracks in-flight fills keyed by query so a second warm()/live_read()
        # for the same query doesn't launch a duplicate lookup. Not for awaiting
        # by the live path — purely de-dup bookkeeping.
        self._inflight: dict[str, asyncio.Task[None]] = {}

    def warm(self, ctx: PrefetchContext) -> list[asyncio.Task[None]]:
        """Predict + background-fetch lookups for *ctx*. Returns the spawned tasks.

        Returns immediately — the lookups run as ``asyncio`` background tasks.
        Skips queries already cached or already in flight. The returned tasks
        are for off-live-path awaiting (tests / between-turn hooks) and
        observability; the live turn must not await them.
        """
        tasks: list[asyncio.Task[None]] = []
        for query in predict_lookups(ctx):
            task = self._spawn_fill(query, ctx)
            if task is not None:
                tasks.append(task)
        return tasks

    def live_read(self, query: str) -> CacheHit | None:
        """Live-turn read: cache only, never blocking.

        On a hit, returns the cached answer (microseconds). On a miss, returns
        ``None`` immediately and schedules a background fill so a later turn can
        have it — the agent answers now with what it has and fills in later.
        Never ``await``s the slow lookup.
        """
        hit = self.cache.get(query)
        if hit is not None:
            return hit
        # Miss: don't block. Kick off a background fill for next time.
        self._spawn_fill(query)
        return None

    async def aclose(self) -> None:
        """Cancel outstanding fills and clear the per-call cache."""
        tasks = [task for task in self._inflight.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inflight.clear()
        self.cache.clear()

    def _spawn_fill(
        self,
        query: str,
        ctx: PrefetchContext | None = None,
    ) -> asyncio.Task[None] | None:
        """Schedule a background fill for *query*, de-duped. ``None`` if skipped.

        Skips when the query is already cached or a fill is already in flight.
        Requires a running event loop (the live pipeline always has one); off
        it, this is a programming error and surfaces as ``RuntimeError`` from
        :func:`asyncio.get_running_loop`.
        """
        if query in self.cache:
            return None
        existing = self._inflight.get(query)
        if existing is not None and not existing.done():
            return existing
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._fill(query, ctx))
        self._inflight[query] = task
        return task

    async def _fill(self, query: str, ctx: PrefetchContext | None) -> None:
        """Run the slow lookup and store the answer. Fail-safe; off the live path.

        A lookup failure is logged (no PHI — payer-level query only) and
        swallowed: a failed warm must never propagate into the call, and the
        live turn simply sees a cache miss and degrades gracefully.
        """
        try:
            value = await self.knowledge_source.fetch(query, _to_fetch_context(ctx))
            if value is None:
                return
            vector = await self.embedding_provider.embed(query)
        except Exception as exc:  # noqa: BLE001 — a failed warm must not break the call
            logger.warning("prefetch_lookup_failed", query=query, error_type=type(exc).__name__)
            return
        finally:
            self._inflight.pop(query, None)
        self.cache.put(query, value, vector=vector)


@dataclass
class _LookupFnKnowledgeSource:
    lookup_fn: LookupFn

    async def fetch(self, query: str, ctx: KnowledgeFetchContext | None = None) -> str | None:
        _ = ctx
        return await self.lookup_fn(query)


def _to_fetch_context(ctx: PrefetchContext | None) -> KnowledgeFetchContext | None:
    if ctx is None:
        return None
    return KnowledgeFetchContext(
        payer=ctx.payer,
        claim_id=ctx.claim_id,
        user_text=ctx.user_text,
    )
