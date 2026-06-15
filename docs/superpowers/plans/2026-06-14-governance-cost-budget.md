# Governance Cost Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-agent cumulative cost circuit-breaker: an agent that declares `budget_usd` is blocked (fail-closed, audited `status="budget_exceeded"`) from starting a run once its prior recorded spend in the audit ledger has reached the budget.

**Architecture:** `governance/cost.py` adds `BudgetExceeded` + `CostGate`/`NullCostGate` + `build_cost_gate`, mirroring `governance/policy.py`. The spend ledger is the existing audit db, read via a new read-only `SqliteAuditLogger.total_cost(agent)`. `BaseAgent.run` runs the cost check right after the policy check (policy first); a block writes a distinct audit record via a generalized `_write_block`. `instantiate_agent` attaches the gate from `config.budget_usd`.

**Tech Stack:** Python 3.12, Pydantic v2, stdlib `sqlite3`, pytest, mypy --strict, ruff. Branch `feat/governance-cost-budget` off `main` (the spec is on `docs/governance-cost-budget-spec`; this plan assumes implementation on a fresh branch off `main`, which already has audit #11 + policy #12 + the root-flag fix `e99d42e`). Tools via `uv run`.

**Key facts (verified against `main`):**
- `BaseAgent.run` (core/base_agent.py): `try: self._policy.check() except PolicyViolation as exc: self._write_policy_block(data, exc); raise`, then `_audit_depth.set(...)`, `super().run`, `_write_audit` in `finally`. `_write_policy_block(data, exc)` builds an `AuditRecord` with `status = "escalated" if isinstance(exc, PolicyEscalation) else "denied"`, `root=_depth() == 0`, zeroed metrics, wrapped in best-effort `try/except → warnings.warn`.
- `SqliteAuditLogger` (governance/audit.py) has `log` + `query`, a private `_connect()`, and an `audit` table with a `cost_usd REAL` column.
- `AgentConfig` (project/config.py): `provider`, `model_params`, `capabilities`, `policies`, `workers`, `interrupt_before`. `extra="ignore"`.
- `instantiate_agent` (project/discovery.py) tail builds the agent then `agent.set_policy(build_policy_gate(root, policies=..., capabilities=...))` and returns.
- `governance.cost` must import ONLY stdlib + `governance.audit` (no `core`/`project`) — acyclic.

---

### Task 1: `SqliteAuditLogger.total_cost`

**Files:**
- Modify: `src/lottie/governance/audit.py` (add one method)
- Test: `src/lottie/governance/tests/test_audit.py` (append)

- [ ] **Step 1: Append the failing tests** to `src/lottie/governance/tests/test_audit.py`:
```python
def test_total_cost_sums_one_agent(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    for cost in (0.01, 0.02, 0.03):
        logger.log(
            AuditRecord(
                ts="2026-06-14T10:00:00+00:00", agent="digest", provider="mock/x",
                status="ok", root=True, input_sha256="a" * 64, output_sha256="b" * 64,
                input_tokens=1, output_tokens=2, cost_usd=cost, latency_ms=1.0, error=None,
            )
        )
    assert abs(logger.total_cost("digest") - 0.06) < 1e-9


def test_total_cost_unknown_agent_is_zero(tmp_path: Path) -> None:
    assert SqliteAuditLogger(tmp_path).total_cost("nope") == 0.0


def test_total_cost_ignores_other_agents(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent="a", provider=None, status="ok",
            root=True, input_sha256="a" * 64, output_sha256=None, input_tokens=0,
            output_tokens=0, cost_usd=0.05, latency_ms=1.0, error=None,
        )
    )
    assert SqliteAuditLogger(tmp_path).total_cost("b") == 0.0
```
(`AuditRecord`, `SqliteAuditLogger`, `Path` are already imported in this test file — confirm; if not, add them.)

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/governance/tests/test_audit.py -k total_cost -v` → `AttributeError: ... has no attribute 'total_cost'`.

- [ ] **Step 3: Add the method** to `SqliteAuditLogger` in `src/lottie/governance/audit.py` (place it after `query`):
```python
    def total_cost(self, agent: str) -> float:
        """Sum of cost_usd across all recorded runs for `agent` (0.0 if none)."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM audit WHERE agent = ?", (agent,)
        ).fetchone()
        return float(row[0])
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/governance/tests/test_audit.py -v` → all pass.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/governance && uv run ruff check src/lottie/governance` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/governance/audit.py src/lottie/governance/tests/test_audit.py
git commit -m "feat(governance): SqliteAuditLogger.total_cost (spend ledger read)"
```

---

### Task 2: `governance/cost.py` — gate, error, factory

**Files:**
- Create: `src/lottie/governance/cost.py`
- Test: `src/lottie/governance/tests/test_cost.py`

- [ ] **Step 1: Failing test** — `src/lottie/governance/tests/test_cost.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from lottie.governance.audit import AuditRecord, SqliteAuditLogger
from lottie.governance.cost import (
    BudgetExceeded,
    CostGate,
    NullCostGate,
    build_cost_gate,
)


def _seed(root: Path, agent: str, cost: float) -> SqliteAuditLogger:
    logger = SqliteAuditLogger(root)
    logger.log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
            output_tokens=0, cost_usd=cost, latency_ms=1.0, error=None,
        )
    )
    return logger


def test_under_budget_passes(tmp_path: Path) -> None:
    ledger = _seed(tmp_path, "digest", 0.01)
    CostGate("digest", 0.10, ledger).check()  # no raise


def test_at_or_over_budget_blocks(tmp_path: Path) -> None:
    ledger = _seed(tmp_path, "digest", 0.10)
    with pytest.raises(BudgetExceeded):
        CostGate("digest", 0.10, ledger).check()  # spent == budget => block


def test_no_ledger_fails_closed() -> None:
    with pytest.raises(BudgetExceeded):
        CostGate("digest", 0.10, None).check()


def test_unreadable_ledger_fails_closed(tmp_path: Path) -> None:
    class _Boom(SqliteAuditLogger):
        def total_cost(self, agent: str) -> float:
            raise RuntimeError("db gone")

    with pytest.raises(BudgetExceeded):
        CostGate("digest", 0.10, _Boom(tmp_path)).check()


def test_null_gate_never_raises() -> None:
    NullCostGate().check()


def test_build_gate_no_budget_is_null(tmp_path: Path) -> None:
    assert isinstance(build_cost_gate(tmp_path, agent="x", budget_usd=None), NullCostGate)


def test_build_gate_with_budget_audit_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
    gate = build_cost_gate(tmp_path, agent="x", budget_usd=1.0)
    assert isinstance(gate, CostGate) and not isinstance(gate, NullCostGate)


def test_build_gate_audit_disabled_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")
    gate = build_cost_gate(tmp_path, agent="x", budget_usd=1.0)
    with pytest.raises(BudgetExceeded):  # ledger is None => fail closed
        gate.check()
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/governance/tests/test_cost.py -v` → module not found.

- [ ] **Step 3: Implement `src/lottie/governance/cost.py`:**
```python
"""Per-agent cumulative cost budget circuit-breaker.

Reads accrued spend from the audit ledger (SqliteAuditLogger.total_cost) and blocks
fail-closed when a configured budget is reached or unverifiable. Imports only stdlib +
governance.audit, so governance stays free of core/project deps (acyclic).
"""

from __future__ import annotations

from pathlib import Path

from lottie.governance.audit import SqliteAuditLogger, build_audit_logger


class BudgetExceeded(Exception):
    """A per-agent cumulative cost budget has been reached. Raised pre-run, fail-closed."""


class CostGate:
    """Blocks a run when an agent's prior recorded spend has reached its budget."""

    def __init__(self, agent: str, budget_usd: float, ledger: SqliteAuditLogger | None) -> None:
        self._agent = agent
        self._budget = budget_usd
        self._ledger = ledger  # None => no readable ledger => fail closed

    def check(self) -> None:
        """Raise BudgetExceeded if the budget is reached or unverifiable; else return."""
        if self._ledger is None:
            raise BudgetExceeded(
                f"agent {self._agent!r} has budget ${self._budget:.4f} but the audit "
                f"ledger is disabled - cannot verify spend (fail-closed)"
            )
        try:
            spent = self._ledger.total_cost(self._agent)
        except Exception as exc:  # noqa: BLE001 — ledger unreadable => fail closed, never 0
            raise BudgetExceeded(
                f"agent {self._agent!r}: audit ledger unreadable, cannot verify spend "
                f"(fail-closed): {exc}"
            ) from exc
        if spent >= self._budget:
            raise BudgetExceeded(
                f"agent {self._agent!r} reached its budget: spent ${spent:.4f} "
                f">= ${self._budget:.4f}"
            )


class NullCostGate(CostGate):
    """No-op gate — the BaseAgent default and the 'no budget declared' result."""

    def __init__(self) -> None:
        super().__init__("", 0.0, None)

    def check(self) -> None:
        return


def build_cost_gate(root: Path, *, agent: str, budget_usd: float | None) -> CostGate:
    """NullCostGate when no budget is declared, else a CostGate bound to the audit ledger.

    The ledger is a real SqliteAuditLogger only when audit is enabled; when audit is
    disabled (LOTTIE_DISABLE_AUDIT / NullAuditLogger) the ledger is None and
    CostGate.check() fails closed.
    """
    if budget_usd is None:
        return NullCostGate()
    audit = build_audit_logger(root)
    ledger = audit if isinstance(audit, SqliteAuditLogger) else None
    return CostGate(agent, budget_usd, ledger)
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/governance/tests/test_cost.py -v` → all pass. (If ruff flags `# noqa: BLE001` as unused because BLE isn't enabled, remove the comment but keep the broad `except`.)

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/governance && uv run ruff check src/lottie/governance` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/governance/cost.py src/lottie/governance/tests/test_cost.py
git commit -m "feat(governance): cost budget gate (cumulative circuit-breaker, fail-closed)"
```

---

### Task 3: Enforce the budget in `BaseAgent.run`

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Test: `src/lottie/core/tests/test_base_agent_cost.py`

- [ ] **Step 1: Failing test** — `src/lottie/core/tests/test_base_agent_cost.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.audit import AuditRecord, SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded, CostGate, NullCostGate
from lottie.governance.policy import PolicyDenied, PolicyGate
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Spy(BaseAgent[_In, _Out]):
    def __init__(self, llm: object, audit: object, name: str) -> None:
        super().__init__(llm, name=name, audit=audit)  # type: ignore[arg-type]
        self.ran = False

    def _execute(self, data: _In) -> _Out:
        self.ran = True
        return _Out(a=f"ok:{data.q}")


def _seed(logger: SqliteAuditLogger, agent: str, cost: float) -> None:
    logger.log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
            output_tokens=0, cost_usd=cost, latency_ms=1.0, error=None,
        )
    )


def test_over_budget_blocks_before_execute_and_audits(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Spy(MockLLMProvider(["x"]), logger, "digest")
    _seed(logger, "digest", 0.10)  # prior spend at budget
    agent.set_cost_gate(CostGate("digest", 0.10, logger))
    with pytest.raises(BudgetExceeded):
        agent.run(_In(q="hi"))
    assert agent.ran is False
    statuses = [r.status for r in SqliteAuditLogger(tmp_path).query(limit=20)]
    assert "budget_exceeded" in statuses


def test_under_budget_runs_normally(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Spy(MockLLMProvider(["x"]), logger, "digest")
    agent.set_cost_gate(CostGate("digest", 1.00, logger))  # no prior spend
    out = agent.run(_In(q="hi"))
    assert out.a == "ok:hi" and agent.ran is True


def test_default_cost_gate_is_null(tmp_path: Path) -> None:
    agent = _Spy(MockLLMProvider(["x"]), SqliteAuditLogger(tmp_path), "digest")
    assert isinstance(agent._cost, NullCostGate)


def test_policy_checked_before_budget(tmp_path: Path) -> None:
    # Agent both policy-denied AND over budget -> PolicyDenied wins (policy first).
    logger = SqliteAuditLogger(tmp_path)
    agent = _Spy(MockLLMProvider(["x"]), logger, "digest")
    _seed(logger, "digest", 0.10)
    agent.set_policy(PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()))
    agent.set_cost_gate(CostGate("digest", 0.10, logger))
    with pytest.raises(PolicyDenied):
        agent.run(_In(q="hi"))
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/core/tests/test_base_agent_cost.py -v` → fails (`set_cost_gate` / `_cost` missing; budget not enforced).

- [ ] **Step 3: Implement in `src/lottie/core/base_agent.py`**

Add the import (with the other `lottie.governance.*` imports):
```python
from lottie.governance.cost import BudgetExceeded, CostGate, NullCostGate
```
In `__init__`, after `self._policy: PolicyGate = NullPolicyGate()`, add:
```python
        self._cost: CostGate = NullCostGate()
```
Add a setter next to `set_policy`:
```python
    def set_cost_gate(self, gate: CostGate) -> None:
        """Attach a cost-budget gate (called by instantiate_agent for CLI/serve runs)."""
        self._cost = gate
```
Change the `run` pre-check block to check cost after policy and handle both blocks via a generalized `_write_block`:
```python
    def run(self, data: InputT) -> OutputT:
        """Policy + budget pre-checks, then instrumented run + audit (best-effort)."""
        try:
            self._policy.check()   # capability policy — checked FIRST (no I/O)
            self._cost.check()     # cumulative budget — checked SECOND (reads the ledger)
        except PolicyViolation as exc:
            self._write_block(data, exc, "escalated" if isinstance(exc, PolicyEscalation) else "denied")
            raise
        except BudgetExceeded as exc:
            self._write_block(data, exc, "budget_exceeded")
            raise
        token = _audit_depth.set(_depth() + 1)
        is_root = _depth() == 1
        output: OutputT | None = None
        try:
            output = super().run(data)
            return output
        finally:
            try:
                self._write_audit(data, output, is_root)
            finally:
                _audit_depth.reset(token)
```
Rename `_write_policy_block` to `_write_block` and take an explicit `status` (replacing the internal `status =` line). (Verified: the only caller is `run()` — no test or other module references `_write_policy_block` by name, so the rename is safe.)
```python
    def _write_block(self, data: InputT, exc: Exception, status: str) -> None:
        try:
            self._audit.log(
                AuditRecord(
                    ts=datetime.now(UTC).isoformat(),
                    agent=self.name,
                    provider=self.provider,
                    status=status,
                    # pre-check runs before the depth increment, so depth 0 == top-level here
                    root=_depth() == 0,
                    input_sha256=hash_model(data),
                    output_sha256=None,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    error=str(exc),
                )
            )
        except Exception as e:  # never let auditing convert/suppress the block
            warnings.warn(f"block audit failed: {e}", stacklevel=2)
```

- [ ] **Step 4: Run new tests + whole suite** — `uv run pytest src/lottie/core/tests/test_base_agent_cost.py -v` (4 pass); then `uv run pytest -q` (whole suite green — existing policy tests still see `denied`/`escalated` via `_write_block`; every agent defaults to `NullCostGate` so no new blocks).

- [ ] **Step 5: Gates** — `uv run mypy --strict src && uv run ruff check` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/core/base_agent.py src/lottie/core/tests/test_base_agent_cost.py
git commit -m "feat(core): enforce per-agent cost budget at BaseAgent.run; audit budget blocks"
```

---

### Task 4: `AgentConfig.budget_usd` + `instantiate_agent` wiring

**Files:**
- Modify: `src/lottie/project/config.py` (add `budget_usd`)
- Modify: `src/lottie/project/discovery.py` (attach the cost gate)
- Test: `src/lottie/project/tests/test_instantiate_cost.py`

- [ ] **Step 1: Failing test** — `src/lottie/project/tests/test_instantiate_cost.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.governance.audit import AuditRecord, SqliteAuditLogger
from lottie.governance.cost import BudgetExceeded
from lottie.llm import MockLLMProvider
from lottie.project.config import load_agent_config
from lottie.project.discovery import instantiate_agent, load_agent_class

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def _seed(demo: Path, agent: str, cost: float) -> None:
    SqliteAuditLogger(demo).log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=0,
            output_tokens=0, cost_usd=cost, latency_ms=1.0, error=None,
        )
    )


def test_instantiate_attaches_budget_that_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)  # audit must be readable
    demo = _scaffold(tmp_path, monkeypatch)
    cfg_path = demo / "agents" / "echo" / "config.yaml"
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8") + "budget_usd: 0.05\n", encoding="utf-8")
    cfg = load_agent_config(demo / "agents" / "echo")
    cls = load_agent_class(demo, "echo")
    agent = instantiate_agent(cls, llm=MockLLMProvider(["hi"]), root=demo, config=cfg)
    _seed(demo, agent.name, 0.05)  # prior spend at budget, under this agent's name
    from agents.echo.schema import EchoAgentInput  # type: ignore[import-not-found]

    with pytest.raises(BudgetExceeded):
        agent.run(EchoAgentInput(query="hi"))


def test_instantiate_no_budget_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    cfg = load_agent_config(demo / "agents" / "echo")
    cls = load_agent_class(demo, "echo")
    agent = instantiate_agent(cls, llm=MockLLMProvider(["hi"]), root=demo, config=cfg)
    from agents.echo.schema import EchoAgentInput  # type: ignore[import-not-found]

    assert agent.run(EchoAgentInput(query="hi")) is not None
```
> NOTE: the generated input class is `EchoAgentInput(query: str)` (confirmed in the policy slice). If a future scaffold renames it, adjust the test import, not production code. `agent.name` is the agent's metrics name — the test seeds spend under that exact name so `total_cost(agent.name)` matches.

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/project/tests/test_instantiate_cost.py -v` → `budget_usd` unknown / `BudgetExceeded` not raised.

- [ ] **Step 3: Add `budget_usd` to `AgentConfig`** in `src/lottie/project/config.py` (after `interrupt_before`):
```python
    budget_usd: float | None = None  # per-agent cumulative spend cap; None = unlimited
```

- [ ] **Step 4: Attach the gate in `instantiate_agent`** (`src/lottie/project/discovery.py`)

Add the import with the other `lottie.governance.*` import:
```python
from lottie.governance.cost import build_cost_gate
```
After the existing `agent.set_policy(...)` line and before `return agent`:
```python
    agent.set_cost_gate(
        build_cost_gate(root, agent=agent.name, budget_usd=config.budget_usd)
    )
    return agent
```

- [ ] **Step 5: Run new tests + whole suite** — `uv run pytest src/lottie/project/tests/test_instantiate_cost.py -v` (2 pass); `uv run pytest -q` (whole suite green; no agent declares `budget_usd` ⇒ `NullCostGate` everywhere).

- [ ] **Step 6: Gates** — `uv run mypy --strict src && uv run ruff check` → clean.

- [ ] **Step 7: Commit**
```bash
git add src/lottie/project/config.py src/lottie/project/discovery.py src/lottie/project/tests/test_instantiate_cost.py
git commit -m "feat(project): AgentConfig.budget_usd + instantiate_agent attaches the cost gate"
```

---

## Self-review checklist (controller, before finishing)

- [ ] Spec coverage: `total_cost` ledger read (§3); `BudgetExceeded`/`CostGate`/`NullCostGate`/`build_cost_gate` (§4); fail-closed when ledger None/unreadable (§5); enforced at `BaseAgent.run` after policy, blocked audit `status="budget_exceeded"` (§6); `budget_usd` config + `instantiate_agent` wiring (§7); block-on-prior-cumulative semantics (§2 — the gate compares `spent >= budget`, never estimates this run).
- [ ] `governance.cost` imports only stdlib + `governance.audit` (acyclic).
- [ ] Whole suite green; existing policy tests still get `denied`/`escalated` from the renamed `_write_block`; no agent newly blocked (every default is `NullCostGate`).
- [ ] Type/name consistency: `BudgetExceeded`, `CostGate`, `NullCostGate`, `build_cost_gate`, `total_cost`, `set_cost_gate`, `_write_block`, `budget_usd`, `status="budget_exceeded"`.
- [ ] `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green.
- [ ] Do NOT push — finish via finishing-a-development-branch, wait for the user. (Per-project budget + per-run caps + TOCTOU tightening remain deferred per spec §8.)
```
