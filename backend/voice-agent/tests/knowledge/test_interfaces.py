"""Tests for knowledge interfaces and default implementations (#56)."""

from __future__ import annotations

from app.knowledge.fixtures import DeidentifiedFixtureKnowledgeSource
from app.knowledge.interfaces import EmbeddingProvider, KnowledgeSource
from app.knowledge.semantic_cache import (
    InMemorySemanticIndex,
    LocalHashEmbeddingProvider,
    cosine,
    embed,
)


async def test_deidentified_fixture_source_implements_knowledge_source() -> None:
    source: KnowledgeSource = DeidentifiedFixtureKnowledgeSource()

    result = await source.fetch("timely filing limit for Aetna")

    assert result is not None
    assert "120 days" in result


async def test_local_hash_embedding_provider_matches_sync_embed() -> None:
    provider: EmbeddingProvider = LocalHashEmbeddingProvider()

    result = await provider.embed("timely filing limit for Aetna")

    assert result == embed("timely filing limit for Aetna")


def test_in_memory_semantic_index_upsert_and_search() -> None:
    index = InMemorySemanticIndex()
    vector = embed("claims mailing address for Cigna")

    index.upsert(
        "claims mailing address for Cigna",
        vector,
        "PO Box 188061",
        expires_at=10.0,
    )

    hit = index.search(vector, threshold=0.99, now=1.0)
    assert hit is not None
    assert hit.key == "claims mailing address for Cigna"
    assert hit.value == "PO Box 188061"
    assert abs(hit.similarity - cosine(vector, vector)) < 1e-9
