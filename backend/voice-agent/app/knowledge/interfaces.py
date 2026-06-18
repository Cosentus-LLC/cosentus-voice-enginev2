"""Knowledge lookup interfaces for the non-blocking prefetch skeleton (#56)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class KnowledgeFetchContext:
    """PHI-safe context for a background knowledge fetch.

    ``case_data_keys`` carries key names only, never values. Real PHI-bearing
    sources remain behind this interface and are deferred from #56.
    """

    payer: str | None = None
    claim_id: str | None = None
    user_text: str = ""
    case_data_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexHit:
    """Result returned by a semantic index lookup."""

    key: str
    value: str
    similarity: float


class KnowledgeSource(Protocol):
    """Slow background lookup source."""

    async def fetch(self, query: str, ctx: KnowledgeFetchContext | None = None) -> str | None:
        """Fetch a knowledge answer for *query*, or ``None`` when absent."""


class EmbeddingProvider(Protocol):
    """Async embedding provider used off the live path."""

    async def embed(self, text: str) -> list[float]:
        """Embed *text* for storage in the live cache."""


class SemanticIndex(Protocol):
    """In-memory semantic index used by the per-call live cache."""

    def upsert(
        self,
        key: str,
        vector: Sequence[float],
        value: str,
        *,
        expires_at: float,
    ) -> None:
        """Insert or update one cached value."""

    def search(
        self,
        vector: Sequence[float],
        *,
        threshold: float,
        now: float,
    ) -> IndexHit | None:
        """Return the nearest non-expired hit at/above *threshold*."""

    def evict(self, *, now: float, max_entries: int) -> None:
        """Evict expired entries and then least-recently-used excess entries."""

    def clear(self) -> None:
        """Drop all indexed entries."""

    def stats(self, *, now: float) -> Mapping[str, int]:
        """Return index statistics after applying expiry."""
