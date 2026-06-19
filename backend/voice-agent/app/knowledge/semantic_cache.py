"""Semantic cache — embed a query, return the nearest cached answer.

The live turn reads this cache and nothing else: :meth:`SemanticCache.get` is a
pure in-memory, synchronous nearest-neighbour lookup — **no network, no I/O, no
``await``** — so a hit costs microseconds and can never blow the real-time voice
budget. The (slow) real lookup that fills the cache happens off the live path in
:mod:`app.knowledge.prefetch`.

Embedding (spike scope)
-----------------------

Real embeddings infra is explicitly deferred. The default
:class:`LocalHashEmbeddingProvider` and :func:`embed` use a deterministic,
stdlib-only bag-of-words vector: lowercase alphanumeric tokens hashed into a
fixed-width vector, compared by cosine similarity. This keeps #56 dependency-
free while the production seams are in place.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import structlog

from app.knowledge.interfaces import IndexHit, SemanticIndex

logger = structlog.get_logger(__name__)

# Width of the hashed bag-of-words vector. Wide enough that distinct payer/claim
# vocabulary rarely collides in the spike's small query space; a real embedding
# model replaces this whole scheme.
_EMBED_DIM = 256

# Default cosine-similarity threshold for a hit. Tuned for the spike's
# token-hash embedding: paraphrases of the same lookup ("timely filing limit
# for Aetna" vs "what is Aetna's timely filing window") clear it, unrelated
# queries do not. A real embedding model would warrant its own threshold.
_DEFAULT_THRESHOLD = 0.6
_DEFAULT_TTL_SECS = 300.0
_DEFAULT_MAX_ENTRIES = 64

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def embed(text: str) -> list[float]:
    """Return a deterministic, L2-normalized bag-of-words vector for *text*.

    Stdlib-only spike embedding (see module docstring). Lowercases, extracts
    alphanumeric tokens, hashes each into ``_EMBED_DIM`` buckets, and L2
    normalizes so cosine similarity reduces to a dot product. An all-zero
    vector (no tokens) is returned as-is; :func:`cosine` treats it as similarity
    ``0`` with everything.
    """
    vec = [0.0] * _EMBED_DIM
    for token in _TOKEN_RE.findall(text.lower()):
        # SHA-1 of the token → a stable bucket. hashlib (not the builtin
        # ``hash``) so the embedding is stable across processes/runs, which a
        # cache shared across turns relies on.
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % _EMBED_DIM
        vec[bucket] += 1.0
    norm = math.sqrt(sum(component * component for component in vec))
    if norm == 0.0:
        return vec
    return [component / norm for component in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (dot product if L2-normed)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class LocalHashEmbeddingProvider:
    """Async wrapper around the deterministic local embedder.

    This default provider is safe for tests/dev and does not perform I/O.
    Real network providers remain deferred behind the :class:`EmbeddingProvider`
    interface and must not be called by the live-read path.
    """

    async def embed(self, text: str) -> list[float]:
        return embed(text)


@dataclass(frozen=True)
class CacheHit:
    """A semantic-cache hit: the stored answer and how close the match was.

    ``query`` is the *cached* query that matched (not the lookup query), so a
    caller can see what the answer was originally fetched for.
    """

    value: str
    similarity: float
    query: str


@dataclass
class _Entry:
    key: str
    value: str
    vector: list[float]
    expires_at: float


class InMemorySemanticIndex:
    """Bounded in-memory semantic index with TTL + LRU eviction.

    The ordered dict's iteration order is the LRU order. Reads that hit move the
    entry to the end; eviction pops from the front after expired entries are
    removed.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def upsert(
        self,
        key: str,
        vector: Sequence[float],
        value: str,
        *,
        expires_at: float,
    ) -> None:
        self._entries[key] = _Entry(
            key=key,
            value=value,
            vector=list(vector),
            expires_at=expires_at,
        )
        self._entries.move_to_end(key)

    def search(
        self,
        vector: Sequence[float],
        *,
        threshold: float,
        now: float,
    ) -> IndexHit | None:
        self.evict(now=now, max_entries=len(self._entries))
        best: _Entry | None = None
        best_sim = 0.0
        query_vec = list(vector)
        for entry in self._entries.values():
            sim = cosine(query_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is None or best_sim < threshold:
            return None
        self._entries.move_to_end(best.key)
        return IndexHit(key=best.key, value=best.value, similarity=best_sim)

    def evict(self, *, now: float, max_entries: int) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)

        while len(self._entries) > max(0, max_entries):
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self, *, now: float) -> Mapping[str, int]:
        self.evict(now=now, max_entries=len(self._entries))
        return {"entries": len(self._entries)}


class SemanticCache:
    """In-memory nearest-neighbour cache keyed by query embedding.

    This cache is meant to be a per-call instance. It is bounded by TTL and max
    entries, and its live-read surface stays synchronous/local.

    Args:
        threshold: Minimum cosine similarity for :meth:`get` to count as a hit.
        ttl_secs: Per-entry expiry window.
        max_entries: Maximum entries retained after LRU eviction.
        index: Semantic index implementation. Defaults to in-memory.
        clock: Monotonic clock, injectable for tests.
    """

    def __init__(
        self,
        *,
        threshold: float = _DEFAULT_THRESHOLD,
        ttl_secs: float = _DEFAULT_TTL_SECS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        index: SemanticIndex | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._ttl_secs = ttl_secs
        self._max_entries = max_entries
        self._index = index or InMemorySemanticIndex()
        self._clock = clock
        self._hits = 0
        self._misses = 0

    def put(self, query: str, value: str, *, vector: list[float] | None = None) -> None:
        """Store *value* under *query*'s embedding.

        If *query* already has an exact-text entry, its value is updated in
        place (the warmer can refresh a fact without growing the cache); a
        semantically-near-but-different query is stored as a new entry.
        """
        now = self._clock()
        self._index.upsert(
            query,
            vector if vector is not None else embed(query),
            value,
            expires_at=now + self._ttl_secs,
        )
        self._index.evict(now=now, max_entries=self._max_entries)

    def get(self, query: str) -> CacheHit | None:
        """Return the nearest cached answer at/above ``threshold``, else ``None``.

        Pure in-memory and synchronous — the live turn calls this and never
        blocks. Records a hit/miss for :meth:`stats`.
        """
        hit = self._index.search(
            embed(query),
            threshold=self._threshold,
            now=self._clock(),
        )
        if hit is not None:
            self._hits += 1
            return CacheHit(value=hit.value, similarity=hit.similarity, query=hit.key)
        self._misses += 1
        return None

    def __contains__(self, query: str) -> bool:
        """Whether *query* would hit (used by the warmer to skip re-fetching).

        Does not count toward hit/miss stats — it's a pre-fetch check, not a
        live read.
        """
        return (
            self._index.search(
                embed(query),
                threshold=self._threshold,
                now=self._clock(),
            )
            is not None
        )

    def clear(self) -> None:
        """Drop all entries and reset per-cache counters."""
        self._index.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int]:
        """Hit / miss / entry counts, for measurement and tests."""
        index_stats = dict(self._index.stats(now=self._clock()))
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": index_stats["entries"],
        }
