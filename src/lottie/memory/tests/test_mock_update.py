import pytest

from lottie.memory.base import MemoryNotFoundError, NullMemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import (
    MemoryPatch,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
)


def test_mock_update_patches_only_supplied_fields() -> None:
    client = MockMemoryClient()
    mid = client.remember(
        MemoryRecord(content="old", namespace="ns", tags=["a"], tier=MemoryTier.SEMANTIC)
    )
    updated = client.update(mid, MemoryPatch(content="new", status=MemoryStatus.DEPRECATED))
    assert updated.content == "new"
    assert updated.status is MemoryStatus.DEPRECATED
    assert updated.tags == ["a"]                 # untouched
    assert updated.tier is MemoryTier.SEMANTIC    # untouched
    assert updated.memory_id == mid


def test_mock_update_unknown_id_raises() -> None:
    client = MockMemoryClient()
    with pytest.raises(MemoryNotFoundError):
        client.update("nope", MemoryPatch(content="x"))


def test_null_update_raises_not_configured() -> None:
    from lottie.memory.base import MemoryNotConfiguredError

    with pytest.raises(MemoryNotConfiguredError):
        NullMemoryClient().update("id", MemoryPatch(content="x"))
