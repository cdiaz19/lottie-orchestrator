"""In-memory `MemoryClient` for tests.

Stores records in a plain list, assigns deterministic ids, and does naive
substring/tag matching on recall — enough for agent integration tests without
a real store. Unit tests must never touch a real store (CLAUDE.md rule 5).
"""

from __future__ import annotations

from lottie.memory.base import MemoryClient, MemoryNotFoundError
from lottie.memory.schema import (
    MemoryHit,
    MemoryPatch,
    MemoryQuery,
    MemoryRecord,
    RecallResult,
)


class MockMemoryClient(MemoryClient):
    """Deterministic in-memory store. Inspect `self.records` in tests."""

    def __init__(self, records: list[MemoryRecord] | None = None) -> None:
        self.records: list[MemoryRecord] = []
        self._counter = 0
        for record in records or []:
            self.remember(record)

    def remember(self, record: MemoryRecord) -> str:
        memory_id = f"{record.namespace}-{self._counter}"
        self._counter += 1
        self.records.append(record.model_copy(update={"memory_id": memory_id}))
        return memory_id

    def recall(self, query: MemoryQuery) -> RecallResult:
        text = query.text.lower()
        tags = set(query.tags)
        hits: list[MemoryHit] = []
        for record in self.records:
            if record.namespace != query.namespace:
                continue
            if query.tier is not None and record.tier != query.tier:
                continue
            if tags and not (tags & set(record.tags)):
                continue
            if text and text not in record.content.lower():
                continue
            hits.append(MemoryHit(record=record, score=1.0))  # mock: no ranking
        return RecallResult(hits=hits[: query.limit])

    def update(self, memory_id: str, patch: MemoryPatch) -> MemoryRecord:
        for i, record in enumerate(self.records):
            if record.memory_id == memory_id:
                changes = patch.model_dump(exclude_none=True)
                updated = record.model_copy(update=changes)
                self.records[i] = updated
                return updated
        raise MemoryNotFoundError(memory_id)

    def forget(self, memory_id: str) -> bool:
        for i, record in enumerate(self.records):
            if record.memory_id == memory_id:
                del self.records[i]
                return True
        return False
