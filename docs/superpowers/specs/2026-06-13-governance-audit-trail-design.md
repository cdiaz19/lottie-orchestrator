# Governance — Audit Trail (first slice) — Design

> An immutable, append-only record of every agent run, written to SQLite under
> `.lottie/audit.db`, hooked universally into `BaseAgent.run`, queryable via `lottie audit`.
> Privacy-safe: input/output stored as sha256 hashes, never raw content.

- **Date:** 2026-06-13
- **Phase:** Governance, slice 1 of N. Later slices (own spec/plan/PR each): policy engine,
  cost governance, OpenTelemetry. This slice is the substrate they will log into.
- **Branch:** `feat/governance-audit-trail` (off `origin/main`).

---

## 1. Goal & scope

`src/lottie/governance/` is empty. Build an audit trail: every agent run produces one immutable
record (who/when/status/tokens/cost + input/output fingerprints) appended to a SQLite database, with
a `lottie audit` query CLI. Done when every `BaseAgent.run` (CLI, serve, mesh) writes a record,
records are append-only, `lottie audit` queries them, content is never stored raw, and the existing
suite stays green.

**Decisions (locked in brainstorming):** universal hook at `BaseAgent.run` · SQLite
`.lottie/audit.db` · sha256 hashes only (no raw content).

## 2. `AuditRecord` — `src/lottie/governance/schema.py`

```python
class AuditRecord(BaseModel):
    ts: str               # ISO-8601 UTC, e.g. "2026-06-13T16:40:00.123456+00:00"
    agent: str            # runnable name
    provider: str | None  # llm.model, or None
    status: str           # "ok" | "error"
    root: bool            # True = top-level run; False = nested (e.g. a mesh worker)
    input_sha256: str
    output_sha256: str | None   # None when the run failed (no output)
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    error: str | None     # repr(exc) when status == "error", else None
```

Hashes are `sha256(model.model_dump_json().encode()).hexdigest()`. `error` is the exception repr
already captured by the instrumentation (note: an exception message *could* echo input — acceptable
for a first slice; raw input/output themselves are never stored).

## 3. Logger — `src/lottie/governance/audit.py`

```python
class AuditLogger(ABC):
    @abstractmethod
    def log(self, record: AuditRecord) -> None: ...

class NullAuditLogger(AuditLogger):
    def log(self, record: AuditRecord) -> None:  # no-op
        return

class SqliteAuditLogger(AuditLogger):
    def __init__(self, root: Path) -> None: ...   # db at root/.lottie/audit.db
    def log(self, record: AuditRecord) -> None: ...   # one INSERT, best-effort
    def query(self, *, agent: str | None = None,
              since: str | None = None, limit: int = 50) -> list[AuditRecord]: ...
```

- **Schema** (created on first use, `CREATE TABLE IF NOT EXISTS audit (...)` + index on `(agent, ts)`).
  Columns map 1:1 to `AuditRecord` (booleans as `INTEGER`).
- **Append-only / immutable:** the class exposes only `log` (INSERT) and `query` (SELECT) — no
  UPDATE/DELETE. That is the immutability guarantee at the API layer.
- **Connection:** opened lazily on first `log`/`query`, `check_same_thread=False`; directory created
  with `mkdir(parents=True, exist_ok=True)`.
- **Best-effort:** `log` wraps its INSERT in `try/except Exception` and swallows (optionally a
  `warnings.warn`) — a locked/broken audit db must NEVER break an agent run. `query` does not swallow
  (CLI surfaces real errors).
- **`build_audit_logger(root: Path) -> AuditLogger`:** returns `NullAuditLogger()` when
  `LOTTIE_DISABLE_AUDIT` is truthy (mirrors `LOTTIE_DISABLE_BENCHMARKS`), else `SqliteAuditLogger(root)`.

`governance/audit.py` imports only stdlib (`sqlite3`, `hashlib`, `os`, `pathlib`, `warnings`),
`pydantic`, and `governance.schema` — never `core.BaseAgent`, so `core → governance` stays acyclic.

## 4. The universal hook — `BaseAgent.run` override (`src/lottie/core/base_agent.py`)

`InstrumentedRunnable.run` already times the run, accumulates tokens/cost, sets `self.last_metrics`,
and records benchmark metrics in a `finally` (success/error captured). It does NOT see input/output,
so audit (which needs hashes) taps one level up, in `BaseAgent.run`:

```python
# module-level
_audit_depth = threading.local()

def _depth() -> int:
    return getattr(_audit_depth, "value", 0)

class BaseAgent(...):
    def __init__(self, llm, *, name=None, memory=None, enable_benchmarks=None,
                 benchmarks_root=None, audit: AuditLogger | None = None) -> None:
        super().__init__(name=name, enable_benchmarks=enable_benchmarks,
                         benchmarks_root=benchmarks_root)
        self.llm = llm
        self.memory = memory or NullMemoryClient()
        self._audit = audit if audit is not None else build_audit_logger(self._benchmarks_root)

    def run(self, data: InputT) -> OutputT:
        _audit_depth.value = _depth() + 1
        is_root = _depth() == 1
        output: OutputT | None = None
        try:
            output = super().run(data)   # InstrumentedRunnable.run — sets last_metrics, may raise
            return output
        finally:
            try:
                self._write_audit(data, output, is_root)
            finally:
                _audit_depth.value = _depth() - 1
```

`_write_audit` builds an `AuditRecord` from `data`, `output`, and `self.last_metrics` (status/tokens/
cost/latency/error all from `last_metrics`, which the super-`finally` set on both paths), then calls
`self._audit.log(record)`. On failure `output is None` → `output_sha256 = None`, `status = "error"`,
`error = last_metrics.error`.

- **Root flag:** the thread-local depth makes the outermost agent run `root=True`; a mesh worker
  (a `BaseAgent` run inside `MeshAgent._execute`) is `root=False`. Workers are still recorded — the
  flag lets `lottie audit` filter to top-level runs.
- **Injectability:** `audit` is constructor-injectable (tests pass a `SqliteAuditLogger(tmp)` or a
  spy); default is resolved via `build_audit_logger`.
- **Skills untouched:** the hook is on `BaseAgent`, not `InstrumentedRunnable`, so skills are not
  audited.

`core/base_agent.py` gains `from lottie.governance.audit import AuditLogger, build_audit_logger` and
`import threading`. (Accepted instrumentation edge `core → governance`; acyclic per §3.)

## 5. `lottie audit` CLI — `src/lottie/cli/audit.py`

`lottie audit [--agent NAME] [--since ISO] [--limit N]` — resolves the project root
(`find_project_root`), opens `SqliteAuditLogger(root).query(...)`, renders a Rich table (ts, agent,
status, root, tokens, cost, latency, input/output hash prefix). Empty db → friendly "no audit
records yet" notice, exit 0. Registered on the main Typer app like `mesh`/`memory`.

## 6. Test isolation

Audit defaults ON in production. To keep the existing ~700-test suite fast and side-effect-free,
add an autouse fixture in the root `conftest.py` that sets `LOTTIE_DISABLE_AUDIT=1` for every test
(so default-constructed agents use `NullAuditLogger`). The new audit tests opt in explicitly by
injecting `SqliteAuditLogger(tmp_path)` (or by `monkeypatch.delenv("LOTTIE_DISABLE_AUDIT")` + a tmp
cwd). This mirrors the intent of the existing instrumentation while guaranteeing no test writes a
real `audit.db`.

## 7. Testing

- **SqliteAuditLogger** (unit): `log` then `query` round-trips a record; schema auto-creates;
  append-only — two `log`s yield two rows, `query` returns newest-first; `query` filters by `agent`,
  `since`, and honors `limit`; missing db dir is created; a `log` against an unwritable path does NOT
  raise (best-effort), while `query` on a fresh db returns `[]`.
- **build_audit_logger**: `LOTTIE_DISABLE_AUDIT=1` → `NullAuditLogger`; unset → `SqliteAuditLogger`.
- **BaseAgent hook** (integration, MockLLM, injected `SqliteAuditLogger(tmp)`): a successful run
  writes one record with `status="ok"`, `root=True`, correct input/output sha256 (assert against an
  independent hash of `model_dump_json()`), populated tokens/cost/latency; a failing run (provider
  raises) writes `status="error"`, `output_sha256=None`, non-null `error`, and the exception still
  propagates; a nested run (an agent whose `_execute` calls another agent's `run`) yields the inner
  record with `root=False` and the outer with `root=True`.
- **lottie audit CLI** (CliRunner): seed a temp project's `audit.db`, assert the table shows the
  agent + status; empty db → "no audit records" notice, exit 0; `--agent`/`--limit` honored.
- **Full gate:** `pytest -q`, `mypy --strict src`, `ruff check` all green; existing suite unaffected.

## 8. Out of scope (YAGNI / later slices)

- Policy engine (allow/deny/escalate), cost budgets/limits, OpenTelemetry — separate governance slices.
- Raw input/output capture (hashes only this slice); retention/rotation/pruning; cross-process log
  aggregation; per-skill auditing; auditing `MeshAgent.resume` beyond a normal run; signing/tamper-
  evidence beyond API-level append-only.

## 9. Definition of done

Every `BaseAgent.run` writes one immutable `AuditRecord` to `.lottie/audit.db` (success and failure,
`root` correct for nested mesh runs); content stored only as sha256; `AuditLogger` ABC +
`SqliteAuditLogger` + `NullAuditLogger` + `build_audit_logger`; `lottie audit` queries by
agent/since/limit; audit failures never break a run; test suite disables audit globally and the new
tests opt in; `pytest`/`mypy --strict src`/`ruff` green. Commit on the feature branch; do not push
until the user approves.
