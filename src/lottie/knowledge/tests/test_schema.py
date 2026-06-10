from __future__ import annotations

import pytest
from pydantic import ValidationError

from lottie.knowledge.schema import (
    Chunk,
    DocStatus,
    Document,
    Embedding,
    KnowledgeLayer,
    RetrievalHit,
    RetrievalQuery,
)


def test_document_defaults_and_layer_enum() -> None:
    d = Document(
        id="g/conv",
        source="knowledge/global/conv.md",
        layer=KnowledgeLayer.GLOBAL,
        content="x",
    )
    assert d.tags == [] and d.depends_on == [] and d.frontmatter == {}


def test_retrieval_hit_carries_score() -> None:
    c = Chunk(id="g/conv#0", doc_id="g/conv", index=0, text="x", start=0, end=1)
    assert RetrievalHit(chunk=c, score=0.9).score == 0.9


def test_knowledge_layer_enum_values() -> None:
    assert KnowledgeLayer.GLOBAL.value == "global"
    assert KnowledgeLayer.PLATFORM.value == "platform"
    assert KnowledgeLayer.PROJECT.value == "project"
    assert KnowledgeLayer.MEMORY.value == "memory"
    assert KnowledgeLayer.DRAFT.value == "draft"


def test_retrieval_query_defaults() -> None:
    q = RetrievalQuery(text="find concepts")
    assert q.k == 5
    assert q.expand_graph is False
    assert q.layers == []
    assert q.tags == []


def test_document_status_defaults_to_draft() -> None:
    d = Document(
        id="g/conv",
        source="knowledge/global/conv.md",
        layer=KnowledgeLayer.GLOBAL,
        content="x",
    )
    assert d.status == DocStatus.DRAFT


def test_document_status_explicit_curated() -> None:
    d = Document(
        id="g/conv",
        source="knowledge/global/conv.md",
        layer=KnowledgeLayer.GLOBAL,
        content="x",
        status=DocStatus.CURATED,
    )
    assert d.status == DocStatus.CURATED


def test_embedding_valid_dim_matches_vector() -> None:
    """Embedding with matching dim and vector length is valid."""
    emb = Embedding(vector=[0.1, 0.2], model="m", dim=2)
    assert emb.dim == 2
    assert emb.vector == [0.1, 0.2]


def test_embedding_mismatched_dim_raises_validation_error() -> None:
    """Embedding with dim != len(vector) must raise ValidationError."""
    with pytest.raises(ValidationError, match="Embedding.dim=3 but vector has 2 elements"):
        Embedding(vector=[0.1, 0.2], model="m", dim=3)
