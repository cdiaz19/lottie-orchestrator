from __future__ import annotations

from lottie.memory.schema import (
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    MemoryTier,
    RecallResult,
    ReflectionInput,
    ReflectionResult,
)


def test_tier_values() -> None:
    assert MemoryTier.WORKING.value == "working"
    assert MemoryTier.EPISODIC.value == "episodic"
    assert MemoryTier.SEMANTIC.value == "semantic"
    assert MemoryTier.PROCEDURAL.value == "procedural"


def test_record_defaults() -> None:
    rec = MemoryRecord(content="hello", namespace="demo")
    assert rec.tier is MemoryTier.EPISODIC
    assert rec.tags == []
    assert rec.metadata == {}
    assert rec.memory_id is None


def test_record_defaults_are_independent() -> None:
    a = MemoryRecord(content="a", namespace="demo")
    b = MemoryRecord(content="b", namespace="demo")
    a.tags.append("x")
    a.metadata["k"] = "v"
    assert b.tags == []
    assert b.metadata == {}


def test_query_defaults() -> None:
    q = MemoryQuery(text="find", namespace="demo")
    assert q.tier is None
    assert q.tags == []
    assert q.limit == 10


def test_hit_and_recall_result() -> None:
    rec = MemoryRecord(content="hello", namespace="demo")
    hit = MemoryHit(record=rec, score=1.0)
    result = RecallResult(hits=[hit])
    assert result.hits[0].record.content == "hello"
    assert result.hits[0].score == 1.0
    assert RecallResult().hits == []


def test_reflection_models() -> None:
    assert ReflectionInput(namespace="demo").limit == 50
    out = ReflectionResult()
    assert out.notes == []
    assert out.consolidated_count == 0
    assert out.written_ids == []
