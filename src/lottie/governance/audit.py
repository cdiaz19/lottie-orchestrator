"""Audit trail logger. Append-only SQLite at <root>/.lottie/audit.db.

Best-effort: a failed write never breaks an agent run. Imports only stdlib +
pydantic + governance.schema, so `core -> governance` stays acyclic.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from lottie.governance.schema import AuditRecord

_DISABLE_VALUES = {"1", "true", "yes", "on"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    agent TEXT NOT NULL,
    provider TEXT,
    status TEXT NOT NULL,
    root INTEGER NOT NULL,
    input_sha256 TEXT NOT NULL,
    output_sha256 TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_agent_ts ON audit (agent, ts);
CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    amount REAL NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reservations_agent ON reservations (agent);
"""

# A reservation older than this (seconds) is assumed orphaned by a crashed run and swept
# before an admission decision, so a killed process cannot permanently shrink a budget.
_RESERVATION_TTL_S = 3600.0

_COLUMNS = (
    "ts, agent, provider, status, root, input_sha256, output_sha256, "
    "input_tokens, output_tokens, cost_usd, latency_ms, error"
)


def hash_model(model: BaseModel) -> str:
    """sha256 of a model's canonical JSON — fingerprints content without storing it."""
    return hashlib.sha256(model.model_dump_json().encode()).hexdigest()


class AuditLogger(ABC):
    """Sink for AuditRecords."""

    @abstractmethod
    def log(self, record: AuditRecord) -> None: ...


class NullAuditLogger(AuditLogger):
    """No-op sink (audit disabled)."""

    def log(self, record: AuditRecord) -> None:
        return


class SqliteAuditLogger(AuditLogger):
    """Append-only audit log in <root>/.lottie/audit.db. Exposes only log + query."""

    def __init__(self, root: Path) -> None:
        self._path = Path(root) / ".lottie" / "audit.db"
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.executescript(_SCHEMA)
            self._conn = conn
        return self._conn

    def log(self, record: AuditRecord) -> None:
        try:
            conn = self._connect()
            conn.execute(
                f"INSERT INTO audit ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.ts, record.agent, record.provider, record.status,
                    int(record.root), record.input_sha256, record.output_sha256,
                    record.input_tokens, record.output_tokens, record.cost_usd,
                    record.latency_ms, record.error,
                ),
            )
            conn.commit()
        except Exception as exc:  # broad catch — audit is best-effort, never break a run
            warnings.warn(f"audit log failed: {exc}", stacklevel=2)

    def query(
        self,
        *,
        agent: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[AuditRecord]:
        conn = self._connect()
        clauses: list[str] = []
        params: list[object] = []
        if agent is not None:
            clauses.append("agent = ?")
            params.append(agent)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM audit{where} ORDER BY ts DESC, id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [
            AuditRecord(
                ts=r[0], agent=r[1], provider=r[2], status=r[3], root=bool(r[4]),
                input_sha256=r[5], output_sha256=r[6], input_tokens=r[7],
                output_tokens=r[8], cost_usd=r[9], latency_ms=r[10], error=r[11],
            )
            for r in rows
        ]


    def total_cost(self, agent: str) -> float:
        """Sum of cost_usd across all recorded runs for `agent` (0.0 if none)."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM audit WHERE agent = ?", (agent,)
        ).fetchone()
        return float(row[0])

    def reserve(self, agent: str, amount: float, budget: float) -> int | None:
        """Atomically admit a run under budget, holding `amount` (fail-closed on races).

        Under one BEGIN IMMEDIATE transaction (file-level RESERVED lock — serializes across
        connections/processes): sweep orphaned reservations, sum committed spend + outstanding
        reservations, and admit only if `committed + reserved + amount <= budget`. Returns the
        reservation id on admission, or None if it would exceed budget (caller blocks). This
        closes the check-then-act TOCTOU: a concurrent unsettled reservation is counted here.
        """
        conn = self._connect()
        now = time.time()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "DELETE FROM reservations WHERE created < ?", (now - _RESERVATION_TTL_S,)
            )
            committed = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM audit WHERE agent = ?", (agent,)
            ).fetchone()[0]
            reserved = conn.execute(
                "SELECT COALESCE(SUM(amount), 0.0) FROM reservations WHERE agent = ?", (agent,)
            ).fetchone()[0]
            if float(committed) + float(reserved) + amount > budget:
                conn.rollback()
                return None
            cur = conn.execute(
                "INSERT INTO reservations (agent, amount, created) VALUES (?, ?, ?)",
                (agent, amount, now),
            )
            conn.commit()
            return int(cur.lastrowid) if cur.lastrowid is not None else None
        except Exception:
            conn.rollback()  # always release the RESERVED lock
            raise

    def settle(self, reservation_id: int) -> None:
        """Release a reservation once the run's real cost is recorded in `audit`. Best-effort:
        a failed delete must never break a run (a stale row is swept by TTL on the next reserve)."""
        try:
            conn = self._connect()
            conn.execute("DELETE FROM reservations WHERE id = ?", (reservation_id,))
            conn.commit()
        except Exception as exc:
            warnings.warn(f"reservation settle failed: {exc}", stacklevel=2)


def build_audit_logger(root: Path) -> AuditLogger:
    """NullAuditLogger when LOTTIE_DISABLE_AUDIT is set, else SqliteAuditLogger(root)."""
    if os.getenv("LOTTIE_DISABLE_AUDIT", "").lower() in _DISABLE_VALUES:
        return NullAuditLogger()
    return SqliteAuditLogger(Path(root))
