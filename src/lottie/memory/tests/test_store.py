from pathlib import Path

import pytest

from lottie.memory.base import MemoryNotFoundError
from lottie.memory.schema import (
    MemoryOrigin,
    MemoryPatch,
    MemoryQuery,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
)
from lottie.memory.store import SqliteMemoryClient


def _client(tmp_path: Path) -> SqliteMemoryClient:
    return SqliteMemoryClient(tmp_path / "memory.db")


def test_remember_assigns_id_and_timestamps(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mid = client.remember(MemoryRecord(content="c", namespace="ns"))
    assert mid
    hits = client.recall(MemoryQuery(text="", namespace="ns")).hits
    assert len(hits) == 1
    rec = hits[0].record
    assert rec.memory_id == mid
    assert rec.created_at is not None and rec.updated_at is not None
    assert rec.status is MemoryStatus.ACTIVE


def test_recall_filters_namespace_tier_and_status(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.remember(
        MemoryRecord(content="a", namespace="ns", tier=MemoryTier.EPISODIC)
    )
    sid = client.remember(
        MemoryRecord(content="b", namespace="ns", tier=MemoryTier.SEMANTIC)
    )
    client.remember(MemoryRecord(content="c", namespace="other"))
    # namespace isolation
    result_ns = client.recall(MemoryQuery(text="", namespace="ns")).hits
    assert {h.record.content for h in result_ns} == {"a", "b"}
    # tier filter
    sem = client.recall(
        MemoryQuery(text="", namespace="ns", tier=MemoryTier.SEMANTIC)
    ).hits
    assert [h.record.content for h in sem] == ["b"]
    # deprecated rows are excluded
    client.update(sid, MemoryPatch(status=MemoryStatus.DEPRECATED))
    result_after = client.recall(MemoryQuery(text="", namespace="ns")).hits
    assert {h.record.content for h in result_after} == {"a"}


def test_recall_tag_match_any_and_score(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.remember(MemoryRecord(content="x", namespace="ns", tags=["p", "q"]))
    client.remember(MemoryRecord(content="y", namespace="ns", tags=["z"]))
    hits = client.recall(MemoryQuery(text="", namespace="ns", tags=["p"])).hits
    assert [h.record.content for h in hits] == ["x"]
    assert hits[0].score > 0.0


def test_recall_substring_and_limit(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.remember(MemoryRecord(content="alpha", namespace="ns"))
    client.remember(MemoryRecord(content="beta", namespace="ns"))
    result = client.recall(MemoryQuery(text="alp", namespace="ns")).hits
    assert [h.record.content for h in result] == ["alpha"]
    two = client.recall(MemoryQuery(text="", namespace="ns", limit=1)).hits
    assert len(two) == 1


def test_update_patches_and_refreshes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mid = client.remember(MemoryRecord(content="old", namespace="ns", tags=["a"]))
    updated = client.update(mid, MemoryPatch(content="new"))
    assert updated.content == "new"
    assert updated.tags == ["a"]
    assert updated.updated_at is not None and updated.created_at is not None
    with pytest.raises(MemoryNotFoundError):
        client.update("missing", MemoryPatch(content="x"))


def test_forget(tmp_path: Path) -> None:
    client = _client(tmp_path)
    mid = client.remember(MemoryRecord(content="c", namespace="ns"))
    assert client.forget(mid) is True
    assert client.forget(mid) is False
    assert client.recall(MemoryQuery(text="", namespace="ns")).hits == []


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    SqliteMemoryClient(path).remember(
        MemoryRecord(content="durable", namespace="ns", origin=MemoryOrigin.REFLECTION)
    )
    reopened = SqliteMemoryClient(path)
    hits = reopened.recall(MemoryQuery(text="", namespace="ns")).hits
    assert len(hits) == 1
    assert hits[0].record.origin is MemoryOrigin.REFLECTION
