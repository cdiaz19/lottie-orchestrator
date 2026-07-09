"""Persistent SQLite `MemoryClient`.

Structured/tag recall only — no embeddings (CLAUDE.md rule 16: vectors only when a
corpus exceeds ~200 files). Imports only stdlib + pydantic + memory.schema/base, so
`lottie.core` can import memory without an import cycle.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from lottie.memory.base import MemoryClient, MemoryNotFoundError, NullMemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import (
    MemoryHit,
    MemoryOrigin,
    MemoryPatch,
    MemoryQuery,
    MemoryRecord,
    MemoryStatus,
    MemoryTier,
    RecallResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    memory_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    tier TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL,
    metadata TEXT NOT NULL,
    origin TEXT NOT NULL,
    source_agent TEXT,
    run_id TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_ns ON records (namespace, tier, status);
"""


class SqliteMemoryClient(MemoryClient):
    """Durable memory store at `path`. Structured recall, incremental update."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            self._conn = conn
        return self._conn

    def remember(self, record: MemoryRecord) -> str:
        now = time.time()
        memory_id = uuid.uuid4().hex
        conn = self._connect()
        conn.execute(
            "INSERT INTO records (memory_id, namespace, tier, content, tags, metadata, "
            "origin, source_agent, run_id, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory_id,
                record.namespace,
                record.tier.value,
                record.content,
                json.dumps(record.tags),
                json.dumps(record.metadata),
                record.origin.value,
                record.source_agent,
                record.run_id,
                record.status.value,
                now,
                now,
            ),
        )
        conn.commit()
        return memory_id

    def recall(self, query: MemoryQuery) -> RecallResult:
        conn = self._connect()
        sql = "SELECT * FROM records WHERE namespace = ? AND status = ?"
        params: list[object] = [query.namespace, MemoryStatus.ACTIVE.value]
        if query.tier is not None:
            sql += " AND tier = ?"
            params.append(query.tier.value)
        sql += " ORDER BY updated_at DESC, rowid DESC"
        rows = conn.execute(sql, params).fetchall()

        text = query.text.lower()
        want_tags = set(query.tags)
        hits: list[MemoryHit] = []
        for row in rows:
            record = self._row_to_record(row)
            if text and text not in record.content.lower():
                continue
            overlap = want_tags & set(record.tags)
            if want_tags and not overlap:
                continue
            score = len(overlap) / len(want_tags) if want_tags else 1.0
            hits.append(MemoryHit(record=record, score=score))
        return RecallResult(hits=hits[: query.limit])

    def update(self, memory_id: str, patch: MemoryPatch) -> MemoryRecord:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM records WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(memory_id)
        current = self._row_to_record(row)
        changes = patch.model_dump(exclude_none=True)
        updated = current.model_copy(update=changes)
        now = time.time()
        conn.execute(
            "UPDATE records SET content = ?, tags = ?, metadata = ?, status = ?, "
            "updated_at = ? WHERE memory_id = ?",
            (
                updated.content,
                json.dumps(updated.tags),
                json.dumps(updated.metadata),
                updated.status.value,
                now,
                memory_id,
            ),
        )
        conn.commit()
        updated.updated_at = now
        return updated

    def forget(self, memory_id: str) -> bool:
        conn = self._connect()
        cur = conn.execute("DELETE FROM records WHERE memory_id = ?", (memory_id,))
        conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            content=row["content"],
            tier=MemoryTier(row["tier"]),
            namespace=row["namespace"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            memory_id=row["memory_id"],
            origin=MemoryOrigin(row["origin"]),
            source_agent=row["source_agent"],
            run_id=row["run_id"],
            status=MemoryStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def build_memory_client(root: Path, *, backend: str, path: str) -> MemoryClient:
    """Construct a MemoryClient from primitive config (acyclic — no AgentConfig import).

    'sqlite' → durable store at <root>/<path>; 'mock' → in-memory; anything else
    (incl. 'null') → NullMemoryClient (fail-loud).
    """
    if backend == "sqlite":
        return SqliteMemoryClient(Path(root) / path)
    if backend == "mock":
        return MockMemoryClient()
    return NullMemoryClient()
