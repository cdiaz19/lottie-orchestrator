"""Provider-agnostic memory interface.

All agent memory access goes through `MemoryClient` (injected as
`self.memory` by `BaseAgent`); agent code never imports a store SDK directly.
This module depends only on `schema.py` so `lottie.core` can import it without
creating an import cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottie.memory.schema import MemoryPatch, MemoryQuery, MemoryRecord, RecallResult


class MemoryStoreError(Exception):
    """Base class for memory subsystem errors."""


class MemoryNotConfiguredError(MemoryStoreError):
    """Raised when an agent uses memory without a configured client."""


class MemoryNotFoundError(MemoryStoreError):
    """Raised when update/forget targets a memory_id that does not exist."""


class MemoryClient(ABC):
    """Abstract memory store. Swap implementations via config."""

    @abstractmethod
    def remember(self, record: MemoryRecord) -> str:
        """Persist `record`; return its assigned `memory_id`."""

    @abstractmethod
    def recall(self, query: MemoryQuery) -> RecallResult:
        """Return records matching `query`, ranked by relevance."""

    @abstractmethod
    def update(self, memory_id: str, patch: MemoryPatch) -> MemoryRecord:
        """Apply `patch` to the record with `memory_id`; return the updated record.

        Incremental only — unset patch fields are left unchanged. Raises
        `MemoryNotFoundError` if no such record exists.
        """

    @abstractmethod
    def forget(self, memory_id: str) -> bool:
        """Remove the record with `memory_id`; return True if one was removed."""


class NullMemoryClient(MemoryClient):
    """Default client for agents without memory configured. Fails loud."""

    _MSG = "memory not enabled for this agent — set memory.enabled in config.yaml"

    def remember(self, record: MemoryRecord) -> str:
        raise MemoryNotConfiguredError(self._MSG)

    def recall(self, query: MemoryQuery) -> RecallResult:
        raise MemoryNotConfiguredError(self._MSG)

    def update(self, memory_id: str, patch: MemoryPatch) -> MemoryRecord:
        raise MemoryNotConfiguredError(self._MSG)

    def forget(self, memory_id: str) -> bool:
        raise MemoryNotConfiguredError(self._MSG)
