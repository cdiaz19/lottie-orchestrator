"""Pydantic models for the memory subsystem.

Pure data shapes — no logic, no imports beyond pydantic/stdlib. `base.py`,
`mock.py`, and `agent.py` all build on these.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class MemoryTier(StrEnum):
    """The four memory tiers (T0–T3)."""

    WORKING = "working"        # T0 — in-context, not persisted
    EPISODIC = "episodic"      # T1 — append-only event log
    SEMANTIC = "semantic"      # T2 — consolidated knowledge
    PROCEDURAL = "procedural"  # T3 — config/rules


class MemoryRecord(BaseModel):
    """A single stored memory."""

    content: str
    tier: MemoryTier = MemoryTier.EPISODIC
    namespace: str
    tags: list[str] = []
    metadata: dict[str, str] = {}
    memory_id: str | None = None  # assigned by MemoryClient.remember


class MemoryQuery(BaseModel):
    """A retrieval request against a namespace."""

    text: str
    namespace: str
    tier: MemoryTier | None = None  # None = any tier
    tags: list[str] = []            # match-any
    limit: int = 10


class MemoryHit(BaseModel):
    """One recalled record with its relevance score."""

    record: MemoryRecord
    score: float


class RecallResult(BaseModel):
    """Ordered hits for a query."""

    hits: list[MemoryHit] = []


class ReflectionInput(BaseModel):
    """Input to MemoryAgent: consolidate recent episodic memory."""

    namespace: str
    limit: int = 50


class ReflectionResult(BaseModel):
    """Output of MemoryAgent consolidation."""

    notes: list[str] = []
    consolidated_count: int = 0
    written_ids: list[str] = []
