# Governance Audit Trail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record every `BaseAgent.run` as an immutable, append-only `AuditRecord` in SQLite (`.lottie/audit.db`), content stored as sha256 hashes only, queryable via `lottie audit`.

**Architecture:** `governance/audit.py` holds an `AuditLogger` ABC + `SqliteAuditLogger` (append-only) + `NullAuditLogger` + `build_audit_logger(root)` factory. `BaseAgent.run` is overridden to wrap the existing instrumented `super().run`, compute input/output hashes, and write one record (success and failure), with a thread-local depth counter flagging top-level vs nested (mesh-worker) runs. `lottie audit` queries the db. Audit is best-effort (never breaks a run) and disabled in the test suite by default.

**Tech Stack:** Python 3.12, Pydantic v2, stdlib `sqlite3`/`hashlib`/`threading`, Typer + Rich, pytest, mypy --strict, ruff. Work from `/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator`, branch `feat/governance-audit-trail` (already checked out).

**Conventions:**
- Run tools from the project dir. mypy is not on the bare PATH — use `uv run mypy --strict ...`, `uv run pytest`, `uv run ruff check`.
- Conventional commits. Stage only the files each task names. Never `git add docs/` (already committed) or unrelated files.
- `RunMetrics` (from `lottie.core.metrics`) fields available on `self.last_metrics`: `name, kind, provider, timestamp (datetime), latency_ms, success (bool), input_tokens, output_tokens, cost_usd, error`.

---

### Task 1: `AuditRecord` schema

**Files:**
- Create: `src/lottie/governance/schema.py`
- Test: `src/lottie/governance/tests/__init__.py` (empty) + `src/lottie/governance/tests/test_schema.py`

- [ ] **Step 1: Failing test**

`src/lottie/governance/tests/test_schema.py`:
```python
from __future__ import annotations

from lottie.governance.schema import AuditRecord


def test_audit_record_minimal() -> None:
    r = AuditRecord(
        ts="2026-06-13T16:40:00+00:00",
        agent="echo",
        provider="mock/x",
        status="ok",
        root=True,
        input_sha256="a" * 64,
        output_sha256="b" * 64,
        input_tokens=1,
        output_tokens=2,
        cost_usd=0.001,
        latency_ms=12.5,
        error=None,
    )
    assert r.status == "ok" and r.root is True
    assert r.output_sha256 is not None


def test_audit_record_failure_has_no_output_hash() -> None:
    r = AuditRecord(
        ts="2026-06-13T16:40:00+00:00", agent="echo", provider=None, status="error",
        root=False, input_sha256="a" * 64, output_sha256=None, input_tokens=0,
        output_tokens=0, cost_usd=0.0, latency_ms=1.0, error="RuntimeError('boom')",
    )
    assert r.status == "error" and r.output_sha256 is None and r.error
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/governance/tests/test_schema.py -v` → module not found.

- [ ] **Step 3: Create the package + schema**

`src/lottie/governance/tests/__init__.py`: empty.
`src/lottie/governance/schema.py`:
```python
"""Typed contracts for the governance audit trail."""

from __future__ import annotations

from pydantic import BaseModel


class AuditRecord(BaseModel):
    """One immutable record of an agent run. Content is stored as sha256, never raw."""

    ts: str  # ISO-8601 UTC
    agent: str
    provider: str | None
    status: str  # "ok" | "error"
    root: bool  # True = top-level run; False = nested (e.g. a mesh worker)
    input_sha256: str
    output_sha256: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    error: str | None
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/governance/tests/test_schema.py -v` → 2 pass.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/governance && uv run ruff check src/lottie/governance` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/governance/schema.py src/lottie/governance/tests/__init__.py src/lottie/governance/tests/test_schema.py
git commit -m "feat(governance): AuditRecord schema for the audit trail"
```

---

### Task 2: `AuditLogger` + `SqliteAuditLogger` + `NullAuditLogger` + factory

**Files:**
- Create: `src/lottie/governance/audit.py`
- Test: `src/lottie/governance/tests/test_audit.py`

- [ ] **Step 1: Failing test**

`src/lottie/governance/tests/test_audit.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from lottie.governance.audit import (
    NullAuditLogger,
    SqliteAuditLogger,
    build_audit_logger,
    hash_model,
)
from lottie.governance.schema import AuditRecord
from pydantic import BaseModel


class _M(BaseModel):
    x: int


def _rec(agent: str = "echo", ts: str = "2026-06-13T10:00:00+00:00", status: str = "ok") -> AuditRecord:
    return AuditRecord(
        ts=ts, agent=agent, provider="mock/x", status=status, root=True,
        input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=1, output_tokens=2,
        cost_usd=0.0, latency_ms=1.0, error=None,
    )


def test_hash_model_is_sha256_of_json() -> None:
    import hashlib
    m = _M(x=5)
    assert hash_model(m) == hashlib.sha256(m.model_dump_json().encode()).hexdigest()


def test_log_then_query_roundtrip(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(_rec())
    rows = logger.query()
    assert len(rows) == 1 and rows[0].agent == "echo" and rows[0].status == "ok"
    assert (tmp_path / ".lottie" / "audit.db").is_file()


def test_query_is_newest_first_and_append_only(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(_rec(ts="2026-06-13T10:00:00+00:00"))
    logger.log(_rec(ts="2026-06-13T11:00:00+00:00"))
    rows = logger.query()
    assert [r.ts for r in rows] == ["2026-06-13T11:00:00+00:00", "2026-06-13T10:00:00+00:00"]


def test_query_filters_agent_since_and_limit(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(_rec(agent="a", ts="2026-06-13T09:00:00+00:00"))
    logger.log(_rec(agent="b", ts="2026-06-13T10:00:00+00:00"))
    logger.log(_rec(agent="b", ts="2026-06-13T11:00:00+00:00"))
    assert {r.agent for r in logger.query(agent="b")} == {"b"}
    assert len(logger.query(since="2026-06-13T10:00:00+00:00")) == 2
    assert len(logger.query(limit=1)) == 1


def test_query_empty_db_returns_empty(tmp_path: Path) -> None:
    assert SqliteAuditLogger(tmp_path).query() == []


def test_log_is_best_effort_on_bad_path(tmp_path: Path) -> None:
    # Point the db dir at a path that can't be created (a file where a dir must go).
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    logger = SqliteAuditLogger(blocker)  # blocker/.lottie/audit.db — parent is a file
    with pytest.warns(Warning):
        logger.log(_rec())  # must NOT raise


def test_null_logger_is_noop() -> None:
    NullAuditLogger().log(_rec())  # no raise, nothing persisted


def test_build_audit_logger_respects_disable_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")
    assert isinstance(build_audit_logger(tmp_path), NullAuditLogger)
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
    assert isinstance(build_audit_logger(tmp_path), SqliteAuditLogger)
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/governance/tests/test_audit.py -v` → module not found.

- [ ] **Step 3: Implement**

`src/lottie/governance/audit.py`:
```python
"""Audit trail logger. Append-only SQLite at <root>/.lottie/audit.db.

Best-effort: a failed write never breaks an agent run. Imports only stdlib +
pydantic + governance.schema, so `core -> governance` stays acyclic.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
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
"""

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
        except Exception as exc:  # noqa: BLE001 — audit is best-effort, never break a run
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


def build_audit_logger(root: Path) -> AuditLogger:
    """NullAuditLogger when LOTTIE_DISABLE_AUDIT is set, else SqliteAuditLogger(root)."""
    if os.getenv("LOTTIE_DISABLE_AUDIT", "").lower() in _DISABLE_VALUES:
        return NullAuditLogger()
    return SqliteAuditLogger(Path(root))
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/governance/tests/test_audit.py -v` → all pass.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/governance && uv run ruff check src/lottie/governance` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/governance/audit.py src/lottie/governance/tests/test_audit.py
git commit -m "feat(governance): append-only SqliteAuditLogger + Null + build factory"
```

---

### Task 3: Universal hook in `BaseAgent.run` + test-suite isolation

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Modify: `conftest.py` (disable audit for the whole suite)
- Test: `src/lottie/core/tests/test_base_agent_audit.py`

- [ ] **Step 1: Disable audit globally in the test suite**

In the root `conftest.py`, add an autouse fixture (alongside the existing ones):
```python
@pytest.fixture(autouse=True)
def _disable_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-off audit in tests; audit tests opt in by injecting a SqliteAuditLogger."""
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")
```

- [ ] **Step 2: Failing test**

`src/lottie/core/tests/test_base_agent_audit.py` (use an injected logger so it works regardless of the global disable):
```python
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.audit import SqliteAuditLogger, hash_model
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Echo(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a=f"echo:{data.q}")


class _Boom(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        raise RuntimeError("boom")


class _Outer(BaseAgent[_In, _Out]):
    """Runs an inner agent inside its own _execute (mimics a mesh top-level → worker)."""

    def __init__(self, llm: object, inner: _Echo, audit: object) -> None:
        super().__init__(llm, audit=audit)  # type: ignore[arg-type]
        self._inner = inner

    def _execute(self, data: _In) -> _Out:
        return self._inner.run(data)


def _llm() -> MockLLMProvider:
    return MockLLMProvider(["unused"])


def test_successful_run_writes_one_root_record(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Echo(_llm(), audit=logger)
    agent.run(_In(q="hi"))
    rows = logger.query()
    assert len(rows) == 1
    r = rows[0]
    assert r.status == "ok" and r.root is True
    assert r.input_sha256 == hash_model(_In(q="hi"))
    assert r.output_sha256 == hash_model(_Out(a="echo:hi"))
    assert r.latency_ms >= 0.0


def test_failed_run_writes_error_record_and_reraises(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Boom(_llm(), audit=logger)
    with pytest.raises(RuntimeError):
        agent.run(_In(q="hi"))
    rows = logger.query()
    assert len(rows) == 1
    assert rows[0].status == "error" and rows[0].output_sha256 is None and rows[0].error


def test_nested_run_flags_inner_non_root(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    inner = _Echo(_llm(), audit=logger)
    outer = _Outer(_llm(), inner, logger)
    outer.run(_In(q="hi"))
    rows = logger.query()  # newest-first; both agents named via class
    statuses = {(r.agent, r.root) for r in rows}
    assert ("_Echo", False) in statuses   # inner ran nested
    assert ("_Outer", True) in statuses   # outer is top-level
```

- [ ] **Step 3: Run, verify FAIL** — `uv run pytest src/lottie/core/tests/test_base_agent_audit.py -v` → fails (no `audit` kwarg / no records written).

- [ ] **Step 4: Implement the hook in `src/lottie/core/base_agent.py`**

Add imports at the top:
```python
import threading
...
from lottie.governance.audit import AuditLogger, build_audit_logger, hash_model
from lottie.governance.schema import AuditRecord
```
Add a module-level depth tracker (after imports, before the class):
```python
_audit_depth = threading.local()


def _depth() -> int:
    return getattr(_audit_depth, "value", 0)
```
Add `audit` to `__init__` and store the resolved logger (keep existing params/order; `audit` is keyword-only with a default, so subclasses like `MeshAgent` that call `super().__init__(...)` are unaffected):
```python
    def __init__(
        self,
        llm: LLMProvider,
        *,
        name: str | None = None,
        memory: MemoryClient | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self.llm = llm
        self.memory: MemoryClient = memory or NullMemoryClient()
        self._audit = audit if audit is not None else build_audit_logger(self._benchmarks_root)
```
Add the `run` override + `_write_audit` (place `run` above `complete`):
```python
    def run(self, data: InputT) -> OutputT:
        """Instrumented run (super) plus one immutable audit record (best-effort)."""
        _audit_depth.value = _depth() + 1
        is_root = _depth() == 1
        output: OutputT | None = None
        try:
            output = super().run(data)
            return output
        finally:
            try:
                self._write_audit(data, output, is_root)
            finally:
                _audit_depth.value = _depth() - 1

    def _write_audit(self, data: InputT, output: OutputT | None, is_root: bool) -> None:
        m = self.last_metrics
        if m is None:  # super().run always sets it, but stay defensive
            return
        record = AuditRecord(
            ts=m.timestamp.isoformat(),
            agent=m.name,
            provider=m.provider,
            status="ok" if m.success else "error",
            root=is_root,
            input_sha256=hash_model(data),
            output_sha256=hash_model(output) if output is not None else None,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
            cost_usd=m.cost_usd,
            latency_ms=m.latency_ms,
            error=m.error,
        )
        self._audit.log(record)
```

- [ ] **Step 5: Run the new tests + the WHOLE suite** — `uv run pytest src/lottie/core/tests/test_base_agent_audit.py -v` (3 pass), then `uv run pytest -q` (whole suite still green — the conftest disable keeps existing agent tests from writing audit.db). If any existing test breaks because a `BaseAgent` subclass constructor doesn't forward `**kwargs` or positionally passes args that now collide with `audit`, STOP and report — do not silently rewrite subclasses; `audit` is keyword-only so this should not happen.

- [ ] **Step 6: Gates** — `uv run mypy --strict src && uv run ruff check` → clean.

- [ ] **Step 7: Commit**
```bash
git add src/lottie/core/base_agent.py conftest.py src/lottie/core/tests/test_base_agent_audit.py
git commit -m "feat(core): audit every BaseAgent.run (universal hook, root-flagged, best-effort)"
```

---

### Task 4: `lottie audit` CLI

**Files:**
- Create: `src/lottie/cli/audit.py`
- Modify: `src/lottie/cli/app.py` (register the command)
- Test: `src/lottie/cli/tests/test_audit_cli.py`

- [ ] **Step 1: Failing test**

`src/lottie/cli/tests/test_audit_cli.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.schema import AuditRecord

runner = CliRunner()


def _seed(root: Path, agent: str = "echo") -> None:
    SqliteAuditLogger(root).log(
        AuditRecord(
            ts="2026-06-13T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=1,
            output_tokens=2, cost_usd=0.0, latency_ms=1.0, error=None,
        )
    )


def _init_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    return demo


def test_audit_lists_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _init_project(tmp_path, monkeypatch)
    _seed(demo)
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "echo" in result.output


def test_audit_empty_is_friendly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "no audit" in result.output.lower()


def test_audit_filters_by_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _init_project(tmp_path, monkeypatch)
    _seed(demo, agent="echo")
    _seed(demo, agent="other")
    result = runner.invoke(app, ["audit", "--agent", "other"])
    assert result.exit_code == 0
    assert "other" in result.output and "echo" not in result.output
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/cli/tests/test_audit_cli.py -v` → no `audit` command.

- [ ] **Step 3: Implement `src/lottie/cli/audit.py`**

```python
"""`lottie audit` — query the immutable agent-run audit log."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lottie.governance.audit import SqliteAuditLogger
from lottie.project.config import find_project_root

_console = Console()


def audit(
    agent: Annotated[
        str | None, typer.Option("--agent", help="Filter to one agent name.")
    ] = None,
    since: Annotated[
        str | None, typer.Option("--since", help="ISO-8601 lower bound on timestamp.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows.")] = 50,
) -> None:
    """Show recent agent-run audit records from .lottie/audit.db."""
    root = find_project_root()
    records = SqliteAuditLogger(root).query(agent=agent, since=since, limit=limit)
    if not records:
        _console.print("[dim]no audit records yet[/dim]")
        return
    table = Table(title="audit log")
    for col in ("ts", "agent", "status", "root", "tokens", "cost", "ms", "in/out"):
        table.add_column(col)
    for r in records:
        table.add_row(
            r.ts, r.agent, r.status, "Y" if r.root else "-",
            f"{r.input_tokens}/{r.output_tokens}", f"{r.cost_usd:.6f}",
            f"{r.latency_ms:.1f}", f"{r.input_sha256[:8]}/{(r.output_sha256 or '-')[:8]}",
        )
    _console.print(table)
```

- [ ] **Step 4: Register in `src/lottie/cli/app.py`**

Add `from lottie.cli.audit import audit` with the other imports, and `app.command("audit")(audit)` with the other `app.command(...)` lines.

- [ ] **Step 5: Run tests + gates** — `uv run pytest src/lottie/cli/tests/test_audit_cli.py -v` (3 pass), then `uv run mypy --strict src && uv run ruff check` → clean. If the empty-db test fails because `find_project_root` errors outside a project, confirm `_init_project` scaffolds one (it does); adjust assertions only to match the real Rich output wording, never to something trivially true.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/cli/audit.py src/lottie/cli/app.py src/lottie/cli/tests/test_audit_cli.py
git commit -m "feat(cli): lottie audit — query the immutable audit log"
```

---

## Self-review checklist (controller, before finishing)

- [ ] Spec coverage: every `BaseAgent.run` audited (ok + error), `root` correct for nested runs; sqlite append-only; hashes-only; `build_audit_logger` env toggle; `lottie audit` query; best-effort (no run broken by audit).
- [ ] `core → governance` acyclic (governance.audit imports no `BaseAgent`).
- [ ] Test suite disables audit globally; audit tests inject their own logger; whole suite green.
- [ ] Type names consistent: `AuditRecord`, `AuditLogger`, `SqliteAuditLogger`, `NullAuditLogger`, `build_audit_logger`, `hash_model`.
- [ ] `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` all green.
- [ ] Do NOT push — finish via finishing-a-development-branch, wait for the user.
```
