"""Don't-block-the-call knowledge lookups — predictive prefetch + semantic cache.

This is the flag-gated production skeleton for the "don't block the call"
pattern (VoiceAgentRAG / Regal). The real system-of-record for payer/claim
facts and real embeddings infra remain behind interfaces and are deferred.

The pattern
-----------

Live phone turns run on a ~200ms real-time budget. A naive mid-turn RAG/DB
lookup adds 50–300ms and freezes the conversation. Instead:

* A background **"slow thinker"** (:class:`~app.knowledge.prefetch.PrefetchWarmer`)
  runs *while the caller is talking*. :func:`~app.knowledge.prefetch.predict_lookups`
  guesses the likely next lookup (this payer, this claim) and the warmer
  pre-populates a :class:`~app.knowledge.semantic_cache.SemanticCache`.
* The **live turn only ever reads the cache** — sub-millisecond on a hit, and
  on a miss it returns immediately (answer-now, fill-in-a-beat-later). It
  **never** ``await``s the slow lookup inline, so the conversation never
  freezes on a lookup.

Safety
------

The live-call hook is off by default. Fixtures are **de-identified,
payer-level public facts only** (timely-filing windows, claims addresses) —
never patient data — so no PHI enters the repo.
"""

from __future__ import annotations

from app.knowledge.prefetch import (
    PrefetchContext,
    PrefetchWarmer,
    predict_lookups,
)
from app.knowledge.semantic_cache import (
    CacheHit,
    InMemorySemanticIndex,
    LocalHashEmbeddingProvider,
    SemanticCache,
)

__all__ = [
    "CacheHit",
    "InMemorySemanticIndex",
    "LocalHashEmbeddingProvider",
    "PrefetchContext",
    "PrefetchWarmer",
    "SemanticCache",
    "predict_lookups",
]
