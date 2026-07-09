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


class MemoryOrigin(StrEnum):
    """How a record entered the store (provenance root)."""

    REFLECTION = "reflection"  # written by the post-run Reflector (S2)
    DISTILL = "distill"        # written during skill distillation (S3)
    MANUAL = "manual"          # written directly / by MemoryAgent consolidation


class MemoryStatus(StrEnum):
    """Lifecycle. DEPRECATE is a soft delete — records are never dropped by the gateway."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"


class MemoryRecord(BaseModel):
    """A single stored memory."""

    content: str
    tier: MemoryTier = MemoryTier.EPISODIC
    namespace: str
    tags: list[str] = []
    metadata: dict[str, str] = {}
    memory_id: str | None = None  # assigned by MemoryClient.remember
    # --- provenance + lifecycle (V2 S0; enforced by the gateway in S1) ---
    origin: MemoryOrigin = MemoryOrigin.MANUAL
    source_agent: str | None = None
    run_id: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: float | None = None  # epoch seconds; assigned by the store on write
    updated_at: float | None = None  # epoch seconds; refreshed by the store on update


class MemoryQuery(BaseModel):
    """A retrieval request against a namespace."""

    text: str
    namespace: str
    tier: MemoryTier | None = None  # None = any tier
    tags: list[str] = []            # match-any
    limit: int = 10


class MemoryPatch(BaseModel):
    """Incremental update to a stored record. All fields optional; None = leave as-is.

    A patch NEVER replaces a whole record — only the supplied fields change. This
    is the ACE 'structured incremental update' primitive (never wholesale rewrite).
    """

    content: str | None = None
    tags: list[str] | None = None
    status: MemoryStatus | None = None
    metadata: dict[str, str] | None = None


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
