# V2 S0 — Memory Store Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a real persistent `MemoryClient` (SQLite) with an incremental `update`/patch op and provenance/lifecycle fields, wired into agents via config — the foundation V2 reflexive write-back (S1/S2) writes to.

**Architecture:** Extend `memory/schema.py` with provenance + a `MemoryPatch`; add `update()` to the `MemoryClient` ABC and its two existing impls; add `SqliteMemoryClient` (structured/tag recall, no vectors — rule 16); add a primitive-args `build_memory_client` factory + a `MemoryConfig` block on `AgentConfig`; inject the configured client through `instantiate_agent` via a new `BaseAgent.set_memory` setter (mirrors `set_policy`/`set_cost_gate`).

**Tech Stack:** Python 3.12+, Pydantic v2, stdlib `sqlite3`, `uv`, `pytest`, `mypy --strict`, `ruff`.

## Global Constraints

- **Rule 1:** never import an LLM SDK; not applicable here (no LLM calls in S0).
- **Rule 2:** all cross-boundary I/O is Pydantic v2 models (`memory/schema.py`). No raw dicts crossing the `MemoryClient` boundary.
- **Rule 5:** unit tests never touch a real store *for agent/skill logic*; the `SqliteMemoryClient` tests target the store directly and use `tmp_path` (never a shared `.lottie/`).
- **Rule 6 / 7b:** every file passes `mypy --strict`; local gate = CI: `uv run ruff check .`, `uv run mypy --strict src`, `uv run pytest -q` under `uv sync --dev --all-extras` before any push.
- **Rule 7:** conventional commits only.
- **Acyclic imports:** `memory/base.py` and `memory/store.py` import only stdlib + pydantic + `memory.schema` (so `lottie.core` may import memory without a cycle). `build_memory_client` takes **primitive args** (root/backend/path), never `AgentConfig` — same discipline as `build_cost_gate`/`build_policy_gate`.
- **Fail-closed:** memory disabled → agent keeps `NullMemoryClient` (raises on use). `update` on a missing id raises, never silently no-ops.
- **No new capabilities beyond the store:** S0 adds provenance/status *fields* and the `update` op, but NO gateway enforcement, NO reflection, NO SecurityGate-on-write — those are S1/S2. Provenance fields are plain optional data in S0.

---

## File Structure

- `src/lottie/memory/schema.py` — **modify**: add `MemoryOrigin`, `MemoryStatus` enums; provenance/lifecycle fields on `MemoryRecord`; new `MemoryPatch`.
- `src/lottie/memory/base.py` — **modify**: `update()` on the `MemoryClient` ABC + `NullMemoryClient`; new `MemoryNotFoundError`.
- `src/lottie/memory/mock.py` — **modify**: `MockMemoryClient.update()`.
- `src/lottie/memory/store.py` — **create**: `SqliteMemoryClient` + `build_memory_client`.
- `src/lottie/memory/__init__.py` — **modify**: export the new symbols.
- `src/lottie/project/config.py` — **modify**: `MemoryConfig` + `AgentConfig.memory`.
- `src/lottie/core/base_agent.py` — **modify**: `set_memory()` setter.
- `src/lottie/project/discovery.py` — **modify**: inject `build_memory_client(...)` in `instantiate_agent` when `config.memory.enabled`.
- `src/lottie/memory/tests/test_schema.py` — **create/modify**: schema defaults + patch.
- `src/lottie/memory/tests/test_store.py` — **create**: `SqliteMemoryClient` behavior.
- `src/lottie/memory/tests/test_mock_update.py` — **create**: mock `update` parity.
- `src/lottie/project/tests/` — **modify/create**: config + injection tests (path confirmed in Task 5 Step 1).

---

## Task 1: Schema — provenance, lifecycle, and `MemoryPatch`

**Files:**
- Modify: `src/lottie/memory/schema.py`
- Test: `src/lottie/memory/tests/test_schema.py`

**Interfaces:**
- Produces: `MemoryOrigin` (StrEnum: `REFLECTION="reflection"`, `DISTILL="distill"`, `MANUAL="manual"`); `MemoryStatus` (StrEnum: `ACTIVE="active"`, `DEPRECATED="deprecated"`); `MemoryRecord` gains `origin: MemoryOrigin = MemoryOrigin.MANUAL`, `source_agent: str | None = None`, `run_id: str | None = None`, `status: MemoryStatus = MemoryStatus.ACTIVE`, `created_at: float | None = None`, `updated_at: float | None = None`; `MemoryPatch(content: str | None = None, tags: list[str] | None = None, status: MemoryStatus | None = None, metadata: dict[str, str] | None = None)`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/test_schema.py` (append if it exists):

```python
from lottie.memory.schema import (
    MemoryOrigin,
    MemoryPatch,
    MemoryRecord,
    MemoryStatus,
)


def test_record_defaults_are_backward_compatible() -> None:
    # Existing callers construct with only content+namespace; new fields default.
    rec = MemoryRecord(content="hello", namespace="ns")
    assert rec.origin is MemoryOrigin.MANUAL
    assert rec.status is MemoryStatus.ACTIVE
    assert rec.source_agent is None
    assert rec.run_id is None
    assert rec.created_at is None
    assert rec.updated_at is None


def test_record_accepts_provenance() -> None:
    rec = MemoryRecord(
        content="c",
        namespace="ns",
        origin=MemoryOrigin.REFLECTION,
        source_agent="Digest",
        run_id="run-1",
    )
    assert rec.origin is MemoryOrigin.REFLECTION
    assert rec.source_agent == "Digest"


def test_patch_is_all_optional() -> None:
    empty = MemoryPatch()
    assert empty.content is None and empty.tags is None
    assert empty.status is None and empty.metadata is None
    patch = MemoryPatch(content="new", status=MemoryStatus.DEPRECATED)
    assert patch.content == "new"
    assert patch.status is MemoryStatus.DEPRECATED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_schema.py -q`
Expected: FAIL — `ImportError: cannot import name 'MemoryPatch'` (and `MemoryOrigin`/`MemoryStatus`).

- [ ] **Step 3: Write minimal implementation**

Edit `src/lottie/memory/schema.py`. Add the two enums after the existing `MemoryTier` enum:

```python
class MemoryOrigin(StrEnum):
    """How a record entered the store (provenance root)."""

    REFLECTION = "reflection"  # written by the post-run Reflector (S2)
    DISTILL = "distill"        # written during skill distillation (S3)
    MANUAL = "manual"          # written directly / by MemoryAgent consolidation


class MemoryStatus(StrEnum):
    """Lifecycle. DEPRECATE is a soft delete — records are never dropped by the gateway."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
```

Extend `MemoryRecord` — add the fields after the existing `memory_id`:

```python
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
```

Add `MemoryPatch` after `MemoryQuery`:

```python
class MemoryPatch(BaseModel):
    """Incremental update to a stored record. All fields optional; None = leave as-is.

    A patch NEVER replaces a whole record — only the supplied fields change. This
    is the ACE 'structured incremental update' primitive (never wholesale rewrite).
    """

    content: str | None = None
    tags: list[str] | None = None
    status: MemoryStatus | None = None
    metadata: dict[str, str] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/memory/tests/test_schema.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/memory/schema.py src/lottie/memory/tests/test_schema.py
git commit -m "feat(memory): provenance/lifecycle fields + MemoryPatch (V2 S0)"
```

---

## Task 2: `update` op on the ABC + Null + Mock clients

**Files:**
- Modify: `src/lottie/memory/base.py`, `src/lottie/memory/mock.py`
- Test: `src/lottie/memory/tests/test_mock_update.py`

**Interfaces:**
- Consumes: `MemoryPatch`, `MemoryRecord`, `MemoryStatus` (Task 1).
- Produces: `MemoryClient.update(memory_id: str, patch: MemoryPatch) -> MemoryRecord` (abstract); `MemoryNotFoundError(MemoryStoreError)`; `NullMemoryClient.update` (raises `MemoryNotConfiguredError`); `MockMemoryClient.update` (patches in place, raises `MemoryNotFoundError` on unknown id).

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/test_mock_update.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_mock_update.py -q`
Expected: FAIL — `ImportError: cannot import name 'MemoryNotFoundError'` / `AttributeError: 'MockMemoryClient' object has no attribute 'update'`.

- [ ] **Step 3: Write minimal implementation**

Edit `src/lottie/memory/base.py`. Add the error class after `MemoryNotConfiguredError`:

```python
class MemoryNotFoundError(MemoryStoreError):
    """Raised when update/forget targets a memory_id that does not exist."""
```

Update the `MemoryClient` import line at the top to include `MemoryPatch`:

```python
from lottie.memory.schema import MemoryPatch, MemoryQuery, MemoryRecord, RecallResult
```

Add the abstract method to `MemoryClient` (after `recall`, before `forget`):

```python
    @abstractmethod
    def update(self, memory_id: str, patch: MemoryPatch) -> MemoryRecord:
        """Apply `patch` to the record with `memory_id`; return the updated record.

        Incremental only — unset patch fields are left unchanged. Raises
        `MemoryNotFoundError` if no such record exists.
        """
```

Add the `NullMemoryClient` override (after `recall`, before `forget`):

```python
    def update(self, memory_id: str, patch: MemoryPatch) -> MemoryRecord:
        raise MemoryNotConfiguredError(self._MSG)
```

Edit `src/lottie/memory/mock.py`. Update the import to add `MemoryPatch`:

```python
from lottie.memory.schema import (
    MemoryHit,
    MemoryPatch,
    MemoryQuery,
    MemoryRecord,
    RecallResult,
)
```

Add `MemoryNotFoundError` to the base import:

```python
from lottie.memory.base import MemoryClient, MemoryNotFoundError
```

Add the method (after `recall`, before `forget`):

```python
    def update(self, memory_id: str, patch: MemoryPatch) -> MemoryRecord:
        for i, record in enumerate(self.records):
            if record.memory_id == memory_id:
                changes = patch.model_dump(exclude_none=True)
                updated = record.model_copy(update=changes)
                self.records[i] = updated
                return updated
        raise MemoryNotFoundError(memory_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/memory/tests/test_mock_update.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/memory/base.py src/lottie/memory/mock.py src/lottie/memory/tests/test_mock_update.py
git commit -m "feat(memory): incremental update() on MemoryClient ABC + Null + Mock (V2 S0)"
```

---

## Task 3: `SqliteMemoryClient`

**Files:**
- Create: `src/lottie/memory/store.py`
- Test: `src/lottie/memory/tests/test_store.py`

**Interfaces:**
- Consumes: `MemoryClient`, `MemoryNotFoundError` (base); all schema models (Task 1/2).
- Produces: `SqliteMemoryClient(path: Path)` implementing `remember`/`recall`/`update`/`forget`. `remember` assigns a uuid `memory_id`, stamps `created_at`/`updated_at`, persists to `<path>`. `recall` filters `namespace` + `status=ACTIVE` (+ optional `tier`) in SQL, then tag-match-any + substring in Python, scores by tag overlap, orders by `updated_at DESC`, applies `query.limit`. `update` patches supplied fields + refreshes `updated_at`. `forget` hard-deletes, returns bool. (Factory `build_memory_client` is Task 4.)

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/test_store.py`:

```python
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
    client.remember(MemoryRecord(content="a", namespace="ns", tier=MemoryTier.EPISODIC))
    sid = client.remember(MemoryRecord(content="b", namespace="ns", tier=MemoryTier.SEMANTIC))
    client.remember(MemoryRecord(content="c", namespace="other"))
    # namespace isolation
    assert {h.record.content for h in client.recall(MemoryQuery(text="", namespace="ns")).hits} == {"a", "b"}
    # tier filter
    sem = client.recall(MemoryQuery(text="", namespace="ns", tier=MemoryTier.SEMANTIC)).hits
    assert [h.record.content for h in sem] == ["b"]
    # deprecated rows are excluded
    client.update(sid, MemoryPatch(status=MemoryStatus.DEPRECATED))
    assert {h.record.content for h in client.recall(MemoryQuery(text="", namespace="ns")).hits} == {"a"}


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
    assert [h.record.content for h in client.recall(MemoryQuery(text="alp", namespace="ns")).hits] == ["alpha"]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.memory.store'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/memory/store.py`:

```python
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

from lottie.memory.base import MemoryClient, MemoryNotFoundError
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/memory/tests/test_store.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/memory/store.py src/lottie/memory/tests/test_store.py
git commit -m "feat(memory): SqliteMemoryClient — durable structured store (V2 S0)"
```

---

## Task 4: `MemoryConfig` + `build_memory_client` factory

**Files:**
- Modify: `src/lottie/project/config.py`, `src/lottie/memory/store.py`, `src/lottie/memory/__init__.py`
- Test: `src/lottie/memory/tests/test_factory.py`

**Interfaces:**
- Consumes: `SqliteMemoryClient` (Task 3), `NullMemoryClient` (base), `MockMemoryClient` (mock).
- Produces: `MemoryConfig(enabled: bool = False, backend: Literal["sqlite", "null", "mock"] = "sqlite", path: str = ".lottie/memory.db")`; `AgentConfig.memory: MemoryConfig = MemoryConfig()`; `build_memory_client(root: Path, *, backend: str, path: str) -> MemoryClient` — `"sqlite"` → `SqliteMemoryClient(root / path)`, `"mock"` → `MockMemoryClient()`, anything else → `NullMemoryClient()`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/test_factory.py`:

```python
from pathlib import Path

from lottie.memory.base import NullMemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.store import SqliteMemoryClient, build_memory_client


def test_factory_sqlite(tmp_path: Path) -> None:
    client = build_memory_client(tmp_path, backend="sqlite", path=".lottie/memory.db")
    assert isinstance(client, SqliteMemoryClient)


def test_factory_mock(tmp_path: Path) -> None:
    assert isinstance(build_memory_client(tmp_path, backend="mock", path="x"), MockMemoryClient)


def test_factory_null_default(tmp_path: Path) -> None:
    assert isinstance(build_memory_client(tmp_path, backend="null", path="x"), NullMemoryClient)
```

Create `src/lottie/project/tests/test_memory_config.py` (confirm dir exists — see Task 5 Step 1; if `src/lottie/project/tests/` is absent, create it with an empty `__init__.py` is NOT needed, pytest uses rootdir config):

```python
from lottie.project.config import AgentConfig, MemoryConfig


def test_agent_config_memory_defaults_off() -> None:
    cfg = AgentConfig(provider="mock")
    assert isinstance(cfg.memory, MemoryConfig)
    assert cfg.memory.enabled is False
    assert cfg.memory.backend == "sqlite"


def test_agent_config_memory_from_dict() -> None:
    cfg = AgentConfig.model_validate(
        {"provider": "mock", "memory": {"enabled": True, "backend": "mock"}}
    )
    assert cfg.memory.enabled is True
    assert cfg.memory.backend == "mock"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_factory.py src/lottie/project/tests/test_memory_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_memory_client'` and `cannot import name 'MemoryConfig'`.

- [ ] **Step 3: Write minimal implementation**

Append `build_memory_client` to `src/lottie/memory/store.py` (add `NullMemoryClient` to the base import and `MockMemoryClient` import at top):

```python
from lottie.memory.base import MemoryClient, MemoryNotFoundError, NullMemoryClient
from lottie.memory.mock import MockMemoryClient
```

At the end of the file:

```python
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
```

Edit `src/lottie/project/config.py`. Add `Literal` to the typing import at the top:

```python
from typing import Literal
```

Add `MemoryConfig` before `AgentConfig`:

```python
class MemoryConfig(BaseModel):
    """Per-agent memory store config. Disabled by default (agent keeps NullMemoryClient)."""

    enabled: bool = False
    backend: Literal["sqlite", "null", "mock"] = "sqlite"
    path: str = ".lottie/memory.db"  # resolved relative to the project root
```

Add the field to `AgentConfig` (after `max_turns`):

```python
    memory: MemoryConfig = MemoryConfig()
```

Edit `src/lottie/memory/__init__.py` — add exports:

```python
from lottie.memory.base import (
    MemoryClient,
    MemoryNotConfiguredError,
    MemoryNotFoundError,
    MemoryStoreError,
    NullMemoryClient,
)
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
    ReflectionInput,
    ReflectionResult,
)
from lottie.memory.store import SqliteMemoryClient, build_memory_client
```

Add `"MemoryNotFoundError"`, `"MemoryOrigin"`, `"MemoryPatch"`, `"MemoryStatus"`, `"SqliteMemoryClient"`, `"build_memory_client"` to `__all__` (keep it sorted).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/memory/tests/test_factory.py src/lottie/project/tests/test_memory_config.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/memory/store.py src/lottie/memory/__init__.py src/lottie/project/config.py src/lottie/memory/tests/test_factory.py src/lottie/project/tests/test_memory_config.py
git commit -m "feat(memory): MemoryConfig + build_memory_client factory (V2 S0)"
```

---

## Task 5: Inject the configured client via `instantiate_agent`

**Files:**
- Modify: `src/lottie/core/base_agent.py`, `src/lottie/project/discovery.py`
- Test: `src/lottie/project/tests/test_memory_injection.py`

**Interfaces:**
- Consumes: `build_memory_client` (Task 4), `AgentConfig.memory` (Task 4), the existing `instantiate_agent(...)` signature.
- Produces: `BaseAgent.set_memory(client: MemoryClient) -> None` (mirrors `set_policy`/`set_cost_gate`); `instantiate_agent` calls `agent.set_memory(build_memory_client(...))` when `config.memory.enabled` (else the agent keeps its default `NullMemoryClient`).

- [ ] **Step 1: Confirm the project test dir**

Run: `ls src/lottie/project/tests/ 2>/dev/null || echo "NO DIR"`
If `NO DIR`, create the directory (no `__init__.py` — the repo's pytest is rootdir-configured; confirm by checking a sibling test dir like `src/lottie/memory/tests/` has none). Put the Task 4 + Task 5 project tests there.

- [ ] **Step 2: Write the failing test**

Create `src/lottie/project/tests/test_memory_injection.py`:

```python
from pathlib import Path

from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.memory.base import NullMemoryClient
from lottie.memory.store import SqliteMemoryClient
from lottie.project.config import AgentConfig
from lottie.project.discovery import instantiate_agent


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


class _Echo(BaseAgent[_Input, _Output]):
    def _execute(self, data: _Input) -> _Output:
        return _Output(text=data.text)


def _cfg(**kw: object) -> AgentConfig:
    return AgentConfig.model_validate({"provider": "mock", **kw})


def test_memory_disabled_keeps_null_client(tmp_path: Path) -> None:
    agent = instantiate_agent(
        _Echo, llm=MockLLMProvider(["x"]), root=tmp_path, config=_cfg()
    )
    assert isinstance(agent.memory, NullMemoryClient)


def test_memory_enabled_injects_sqlite(tmp_path: Path) -> None:
    agent = instantiate_agent(
        _Echo,
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": True, "backend": "sqlite"}),
    )
    assert isinstance(agent.memory, SqliteMemoryClient)


def test_set_memory_setter() -> None:
    agent = _Echo(llm=MockLLMProvider(["x"]))
    store = SqliteMemoryClient(Path("/tmp/does-not-persist-here.db"))
    agent.set_memory(store)
    assert agent.memory is store
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest src/lottie/project/tests/test_memory_injection.py -q`
Expected: FAIL — `AttributeError: '_Echo' object has no attribute 'set_memory'`.

- [ ] **Step 4: Write minimal implementation**

Edit `src/lottie/core/base_agent.py`. Add the setter near the other `set_*` methods (search for `def set_policy`):

```python
    def set_memory(self, client: MemoryClient) -> None:
        """Attach a memory client (used by instantiate_agent when memory is enabled)."""
        self.memory = client
```

Edit `src/lottie/project/discovery.py`. Add the import near the other builders (top of file, with the governance builders):

```python
from lottie.memory.store import build_memory_client
```

In `instantiate_agent`, after the `agent.set_capability_gate(...)` line and before the `security_gate` block, add:

```python
    # V2 S0: attach a persistent memory client when the agent opts in. Disabled →
    # the agent keeps its default NullMemoryClient (fail-loud on use).
    if config.memory.enabled:
        agent.set_memory(
            build_memory_client(
                root, backend=config.memory.backend, path=config.memory.path
            )
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest src/lottie/project/tests/test_memory_injection.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/lottie/core/base_agent.py src/lottie/project/discovery.py src/lottie/project/tests/test_memory_injection.py
git commit -m "feat(memory): inject configured MemoryClient via instantiate_agent (V2 S0)"
```

---

## Task 6: Full gate + docs touch-up

**Files:**
- Modify: `src/lottie/memory/__init__.py` (verify exports), CLAUDE.md (no change required in S0 — the gateway rule lands in S1; skip).

- [ ] **Step 1: Run the full local gate (must match CI — rule 7b)**

```bash
uv sync --dev --all-extras
uv run ruff check .
uv run mypy --strict src
uv run pytest -q
```
Expected: ruff clean; mypy clean (file count grows by 1 — `store.py`); pytest all green (baseline 946 + new S0 tests, ~21 added).

- [ ] **Step 2: Fix any gate failures**

If mypy flags `sqlite3.Row` indexing as returning `Any`, the `_row_to_record` construction is still typed because each `MemoryRecord` field is explicitly typed at the call — no `# type: ignore` needed. If ruff flags import order, run `uv run ruff check . --fix` and re-review the diff before staging. Do NOT add `Any` (rule 6).

- [ ] **Step 3: Commit any gate fixes**

```bash
git add -A
git commit -m "chore(memory): satisfy mypy --strict + ruff for V2 S0"
```

*(Skip this commit if Steps 1 produced no changes.)*

---

## Lab round (R22) — separate `lottie-lab` PR, after S0 merges

Not part of this plan's commits (lab lives in the separate `lottie-lab` repo). After S0 merges to `main`, add Round 22: a driver that exercises `SqliteMemoryClient` through a real project config (`memory.enabled: true`) — remember/recall/update/forget, namespace isolation, `status=deprecated` exclusion, cross-instance persistence. Mirror the Round-8/Round-17 driver harness. Validate locally (lab CI stays red on `ORCH_REPO_TOKEN` — known non-bug).

---

## Self-Review

**Spec coverage (S0 rows of the epic spec §4 + §3.1/§3.2):**
- §3.1 provenance/lifecycle on `MemoryRecord` + `MemoryPatch` → Task 1. ✅
- §3.1 `MemoryDelta` → **out of S0 scope** (S1); intentionally not built here. ✅
- §3.2 `SqliteMemoryClient` + structured recall + `update` op → Tasks 2, 3. ✅
- §3.2 `build_memory_client` factory + inject via `instantiate_agent` → Tasks 4, 5. ✅
- §D1 defer Chroma/vectors → honored (structured recall only, noted in `store.py` docstring). ✅
- Fail-closed when disabled → Task 5 (`NullMemoryClient` retained). ✅

**Placeholder scan:** no TBD/TODO; every code step shows full code; commands have expected output. ✅

**Type consistency:** `MemoryPatch`/`MemoryOrigin`/`MemoryStatus` defined in Task 1 and imported identically in Tasks 2–5; `build_memory_client(root, *, backend, path)` signature identical in Task 4 def and Task 5 call; `set_memory` name consistent Task 5 def↔test. ✅

**Note on scope discipline:** `forget` remains a hard delete (ABC parity); the gateway's soft-DEPRECATE path is S1 and does not use `forget`. `recall` scoring is deterministic (tag-overlap ratio) so unit tests need no time mocking.
