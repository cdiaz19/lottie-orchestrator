from __future__ import annotations

import pytest

from lottie.memory.base import MemoryNotConfiguredError, NullMemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryQuery, MemoryRecord, MemoryTier


def _rec(
    content: str,
    *,
    namespace: str = "demo",
    tier: MemoryTier = MemoryTier.EPISODIC,
    tags: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        content=content, namespace=namespace, tier=tier, tags=tags or []
    )


def test_remember_assigns_id_and_sets_field() -> None:
    client = MockMemoryClient()
    mid = client.remember(_rec("hello"))
    assert mid == "demo-0"
    assert client.records[0].memory_id == "demo-0"
    assert client.remember(_rec("again")) == "demo-1"


def test_recall_substring_match() -> None:
    client = MockMemoryClient()
    client.remember(_rec("the cat sat"))
    client.remember(_rec("a dog barked"))
    hits = client.recall(MemoryQuery(text="cat", namespace="demo")).hits
    assert len(hits) == 1
    assert hits[0].record.content == "the cat sat"
    assert hits[0].score == 1.0


def test_recall_empty_text_matches_all() -> None:
    client = MockMemoryClient()
    client.remember(_rec("one"))
    client.remember(_rec("two"))
    hits = client.recall(MemoryQuery(text="", namespace="demo")).hits
    assert len(hits) == 2


def test_recall_filters_namespace_tier_tags() -> None:
    client = MockMemoryClient()
    client.remember(_rec("keep", namespace="a", tier=MemoryTier.EPISODIC, tags=["x"]))
    client.remember(_rec("other ns", namespace="b"))
    client.remember(_rec("wrong tier", namespace="a", tier=MemoryTier.SEMANTIC))
    client.remember(_rec("wrong tag", namespace="a", tags=["y"]))
    q = MemoryQuery(text="", namespace="a", tier=MemoryTier.EPISODIC, tags=["x"])
    hits = client.recall(q).hits
    assert [h.record.content for h in hits] == ["keep"]


def test_recall_limit_truncates() -> None:
    client = MockMemoryClient()
    for i in range(5):
        client.remember(_rec(f"item {i}"))
    hits = client.recall(MemoryQuery(text="", namespace="demo", limit=2)).hits
    assert len(hits) == 2


def test_forget_returns_true_then_false() -> None:
    client = MockMemoryClient()
    mid = client.remember(_rec("bye"))
    assert client.forget(mid) is True
    assert client.forget(mid) is False
    assert client.records == []


def test_seeded_records_get_ids() -> None:
    client = MockMemoryClient(records=[_rec("seed")])
    assert client.records[0].memory_id == "demo-0"


def test_null_client_raises_on_all_ops() -> None:
    client = NullMemoryClient()
    with pytest.raises(MemoryNotConfiguredError):
        client.remember(_rec("x"))
    with pytest.raises(MemoryNotConfiguredError):
        client.recall(MemoryQuery(text="", namespace="demo"))
    with pytest.raises(MemoryNotConfiguredError):
        client.forget("demo-0")
