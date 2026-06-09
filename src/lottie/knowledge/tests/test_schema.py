from __future__ import annotations

from lottie.knowledge.schema import (
    Chunk,
    Document,
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
    assert KnowledgeLayer.GLOBAL == "global"
    assert KnowledgeLayer.PLATFORM == "platform"
    assert KnowledgeLayer.PROJECT == "project"
    assert KnowledgeLayer.MEMORY == "memory"
    assert KnowledgeLayer.DRAFT == "draft"


def test_retrieval_query_defaults() -> None:
    q = RetrievalQuery(text="find concepts")
    assert q.k == 5
    assert q.expand_graph is False
    assert q.layers == []
    assert q.tags == []
