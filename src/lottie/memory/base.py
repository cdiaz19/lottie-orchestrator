"""Provider-agnostic memory interface.

All agent memory access goes through `MemoryClient` (injected as
`self.memory` by `BaseAgent`); agent code never imports a store SDK directly.
This module depends only on `schema.py` so `lottie.core` can import it without
creating an import cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottie.memory.schema import MemoryQuery, MemoryRecord, RecallResult


class MemoryError(Exception):
    """Base class for memory subsystem errors."""


class MemoryNotConfiguredError(MemoryError):
    """Raised when an agent uses memory without a configured client."""


class MemoryClient(ABC):
    """Abstract memory store. Swap implementations via config."""

    @abstractmethod
    def remember(self, record: MemoryRecord) -> str:
        """Persist `record`; return its assigned `memory_id`."""

    @abstractmethod
    def recall(self, query: MemoryQuery) -> RecallResult:
        """Return records matching `query`, ranked by relevance."""

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

    def forget(self, memory_id: str) -> bool:
        raise MemoryNotConfiguredError(self._MSG)
