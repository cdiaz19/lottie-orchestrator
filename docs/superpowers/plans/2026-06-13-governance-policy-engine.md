# Governance Policy Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce declarative allow/deny/escalate policy over an agent's declared **capabilities** at the run chokepoint: a `deny` match raises `PolicyDenied`, an `escalate` match raises `PolicyEscalation`, blocked runs are recorded in the audit trail with a distinct status, and empty/absent rules block nothing (backward-compatible).

**Architecture:** `governance/policy.py` holds `Policy` + loaders + `PolicyGate`/`NullPolicyGate` + `build_policy_gate` + the `PolicyViolation` hierarchy. `BaseAgent.run` (already overridden for audit) gains a policy pre-check before the instrumented `super().run`; a violation writes a `status="denied"/"escalated"` audit record then re-raises. `instantiate_agent` attaches the real gate (built from the agent's config) so CLI `lottie run` + serve enforce on the top-level agent/mesh.

**Tech Stack:** Python 3.12, Pydantic v2, `yaml`, pytest, mypy --strict, ruff. Branch `feat/governance-policy-engine` (already checked out, **stacked on `feat/governance-audit-trail`**). Tools via `uv run` (mypy/pytest/ruff not on bare PATH).

**Key facts:**
- `governance/audit.py` already provides `AuditLogger`, `hash_model`, and `AuditRecord` (status is a free `str`; we widen the vocabulary to `denied`/`escalated`).
- `BaseAgent.run` (core/base_agent.py) currently: increments `_audit_depth`, calls `super().run`, audits in `finally`. We add a policy pre-check ABOVE that block.
- `instantiate_agent(agent_cls, *, llm, root, config, enable_benchmarks=None)` (project/discovery.py) is the single construction seam for CLI + serve; it returns the agent via `from_project` or the plain constructor.
- `AgentConfig` has `.capabilities: list[str]` and `.policies: list[str]`.
- `policies/<name>.yaml` shape (from the `lottie init` template): `name`, `allow: []`, `deny: []`, `escalate: []`. The repo's own `policies/base.yaml` is **0 bytes** → must load as an empty policy.
- Layering: `governance.policy` imports ONLY stdlib + pydantic + yaml (NOT `lottie.project` / `lottie.core`). `build_policy_gate` therefore takes primitives, not `AgentConfig`.

---

### Task 1: `governance/policy.py` — model, loader, gate, errors, factory

**Files:**
- Create: `src/lottie/governance/policy.py`
- Test: `src/lottie/governance/tests/test_policy.py`

- [ ] **Step 1: Failing test**

`src/lottie/governance/tests/test_policy.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from lottie.governance.policy import (
    NullPolicyGate,
    Policy,
    PolicyConfigError,
    PolicyDenied,
    PolicyEscalation,
    PolicyGate,
    build_policy_gate,
    load_policy,
)


def _write(root: Path, name: str, body: str) -> None:
    (root / "policies").mkdir(parents=True, exist_ok=True)
    (root / "policies" / f"{name}.yaml").write_text(body, encoding="utf-8")


def test_load_policy_populated(tmp_path: Path) -> None:
    _write(tmp_path, "base", "name: base\ndeny: [shell]\nallow: [http]\nescalate: [fs]\n")
    p = load_policy(tmp_path, "base")
    assert p.deny == ["shell"] and p.allow == ["http"] and p.escalate == ["fs"]


def test_load_policy_empty_file_is_empty(tmp_path: Path) -> None:
    _write(tmp_path, "base", "")
    p = load_policy(tmp_path, "base")
    assert p.allow == [] and p.deny == [] and p.escalate == []


def test_load_policy_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(PolicyConfigError):
        load_policy(tmp_path, "nope")


def test_gate_clean_passes() -> None:
    PolicyGate(["http"], allow=set(), deny=set(), escalate=set()).check()  # no raise


def test_gate_deny_raises() -> None:
    with pytest.raises(PolicyDenied):
        PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()).check()


def test_gate_escalate_raises() -> None:
    with pytest.raises(PolicyEscalation):
        PolicyGate(["fs"], allow=set(), deny=set(), escalate={"fs"}).check()


def test_gate_deny_beats_escalate() -> None:
    with pytest.raises(PolicyDenied):
        PolicyGate(["x"], allow=set(), deny={"x"}, escalate={"x"}).check()


def test_gate_allow_whitelist_blocks_unlisted() -> None:
    with pytest.raises(PolicyDenied):
        PolicyGate(["http", "shell"], allow={"http"}, deny=set(), escalate=set()).check()


def test_gate_allow_superset_passes() -> None:
    PolicyGate(["http"], allow={"http", "fs"}, deny=set(), escalate=set()).check()


def test_null_gate_never_raises() -> None:
    NullPolicyGate().check()


def test_build_gate_no_policies_is_null(tmp_path: Path) -> None:
    assert isinstance(build_policy_gate(tmp_path, policies=[], capabilities=["x"]), NullPolicyGate)


def test_build_gate_merges_files(tmp_path: Path) -> None:
    _write(tmp_path, "a", "deny: [shell]\n")
    _write(tmp_path, "b", "escalate: [fs]\n")
    gate = build_policy_gate(tmp_path, policies=["a", "b"], capabilities=["fs"])
    with pytest.raises(PolicyEscalation):
        gate.check()
    gate2 = build_policy_gate(tmp_path, policies=["a", "b"], capabilities=["shell"])
    with pytest.raises(PolicyDenied):
        gate2.check()
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/governance/tests/test_policy.py -v` → module not found.

- [ ] **Step 3: Implement `src/lottie/governance/policy.py`**

```python
"""Capability policy engine: load allow/deny/escalate rules and evaluate them
against an agent's declared capabilities at the run chokepoint.

Imports only stdlib + pydantic + yaml so governance stays free of core/project deps.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel


class Policy(BaseModel):
    """One governance policy file: allow/deny/escalate over capability names."""

    name: str = ""
    allow: list[str] = []
    deny: list[str] = []
    escalate: list[str] = []


class PolicyViolation(Exception):
    """Base for a blocked run."""


class PolicyDenied(PolicyViolation):
    """A declared capability is denied (or not in a non-empty allow-list)."""


class PolicyEscalation(PolicyViolation):
    """A declared capability requires human approval (escalate). Blocked for now."""


class PolicyConfigError(Exception):
    """A declared policy file is missing or malformed."""


def load_policy(root: Path, name: str) -> Policy:
    """Read root/policies/<name>.yaml. Empty file → empty Policy; missing → error."""
    path = Path(root) / "policies" / f"{name}.yaml"
    if not path.is_file():
        raise PolicyConfigError(f"declared policy {name!r} not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return Policy(name=name)
    if not isinstance(raw, dict):
        raise PolicyConfigError(f"policy {name!r} must be a mapping, got {type(raw).__name__}")
    return Policy(
        name=str(raw.get("name", name)),
        allow=list(raw.get("allow") or []),
        deny=list(raw.get("deny") or []),
        escalate=list(raw.get("escalate") or []),
    )


class PolicyGate:
    """Evaluates an agent's declared capabilities against merged policy rules."""

    def __init__(
        self,
        capabilities: Iterable[str],
        *,
        allow: set[str],
        deny: set[str],
        escalate: set[str],
    ) -> None:
        self._caps = sorted(capabilities)
        self._allow = allow
        self._deny = deny
        self._escalate = escalate

    def check(self) -> None:
        """Raise on a violation; else return. Precedence: deny > escalate > allow-whitelist."""
        for cap in self._caps:
            if cap in self._deny:
                raise PolicyDenied(f"capability {cap!r} denied by policy")
        for cap in self._caps:
            if cap in self._escalate:
                raise PolicyEscalation(f"capability {cap!r} requires approval (escalate)")
        if self._allow:
            for cap in self._caps:
                if cap not in self._allow:
                    raise PolicyDenied(f"capability {cap!r} not in policy allow-list")


class NullPolicyGate(PolicyGate):
    """No-op gate — the BaseAgent default and the 'no policies declared' result."""

    def __init__(self) -> None:
        super().__init__([], allow=set(), deny=set(), escalate=set())

    def check(self) -> None:
        return


def build_policy_gate(
    root: Path, *, policies: list[str], capabilities: list[str]
) -> PolicyGate:
    """Merge every declared policy file (union) and bind capabilities.

    Returns NullPolicyGate when no policies are declared.
    """
    if not policies:
        return NullPolicyGate()
    allow: set[str] = set()
    deny: set[str] = set()
    escalate: set[str] = set()
    for name in policies:
        p = load_policy(root, name)
        allow |= set(p.allow)
        deny |= set(p.deny)
        escalate |= set(p.escalate)
    return PolicyGate(capabilities, allow=allow, deny=deny, escalate=escalate)
```

- [ ] **Step 4: Run, verify PASS** — `uv run pytest src/lottie/governance/tests/test_policy.py -v` → all pass.

- [ ] **Step 5: Gates** — `uv run mypy --strict src/lottie/governance && uv run ruff check src/lottie/governance` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/governance/policy.py src/lottie/governance/tests/test_policy.py
git commit -m "feat(governance): capability policy engine (allow/deny/escalate, load + gate)"
```

---

### Task 2: Enforce policy in `BaseAgent.run` + audit the block

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Test: `src/lottie/core/tests/test_base_agent_policy.py`

- [ ] **Step 1: Failing test**

`src/lottie/core/tests/test_base_agent_policy.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.policy import (
    NullPolicyGate,
    PolicyDenied,
    PolicyEscalation,
    PolicyGate,
)
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Spy(BaseAgent[_In, _Out]):
    def __init__(self, llm: object, audit: object) -> None:
        super().__init__(llm, audit=audit)  # type: ignore[arg-type]
        self.ran = False

    def _execute(self, data: _In) -> _Out:
        self.ran = True
        return _Out(a=f"ok:{data.q}")


def _agent(tmp_path: Path) -> _Spy:
    return _Spy(MockLLMProvider(["x"]), SqliteAuditLogger(tmp_path))


def test_denied_run_blocks_before_execute_and_audits(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.set_policy(PolicyGate(["shell"], allow=set(), deny={"shell"}, escalate=set()))
    with pytest.raises(PolicyDenied):
        agent.run(_In(q="hi"))
    assert agent.ran is False  # _execute never reached
    rows = SqliteAuditLogger(tmp_path).query()
    assert len(rows) == 1 and rows[0].status == "denied" and rows[0].output_sha256 is None


def test_escalated_run_audits_escalated(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    agent.set_policy(PolicyGate(["fs"], allow=set(), deny=set(), escalate={"fs"}))
    with pytest.raises(PolicyEscalation):
        agent.run(_In(q="hi"))
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "escalated"


def test_default_null_gate_runs_normally(tmp_path: Path) -> None:
    agent = _agent(tmp_path)  # no set_policy → NullPolicyGate default
    out = agent.run(_In(q="hi"))
    assert out.a == "ok:hi" and agent.ran is True
    rows = SqliteAuditLogger(tmp_path).query()
    assert rows[0].status == "ok"


def test_default_policy_is_null_gate(tmp_path: Path) -> None:
    assert isinstance(_agent(tmp_path)._policy, NullPolicyGate)
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/core/tests/test_base_agent_policy.py -v` → no `set_policy` / not enforced.

- [ ] **Step 3: Implement in `src/lottie/core/base_agent.py`**

Add imports:
```python
from datetime import UTC, datetime
...
from lottie.governance.policy import NullPolicyGate, PolicyEscalation, PolicyGate, PolicyViolation
```
In `__init__`, after `self._audit = ...`, add:
```python
        self._policy: PolicyGate = NullPolicyGate()
```
Add a setter (e.g. right after `__init__`):
```python
    def set_policy(self, gate: PolicyGate) -> None:
        """Attach a policy gate (called by instantiate_agent for CLI/serve runs)."""
        self._policy = gate
```
Add the pre-check at the very top of `run` (before the `_audit_depth` line):
```python
    def run(self, data: InputT) -> OutputT:
        """Policy pre-check, then instrumented run + audit (best-effort)."""
        try:
            self._policy.check()
        except PolicyViolation as exc:
            self._write_policy_block(data, exc)
            raise
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
```
Add `_write_policy_block` (next to `_write_audit`):
```python
    def _write_policy_block(self, data: InputT, exc: PolicyViolation) -> None:
        status = "escalated" if isinstance(exc, PolicyEscalation) else "denied"
        try:
            self._audit.log(
                AuditRecord(
                    ts=datetime.now(UTC).isoformat(),
                    agent=self.name,
                    provider=self.provider,
                    status=status,
                    root=True,
                    input_sha256=hash_model(data),
                    output_sha256=None,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    error=str(exc),
                )
            )
        except Exception as e:  # never let auditing convert/suppress the policy block
            warnings.warn(f"policy-block audit failed: {e}", stacklevel=2)
```

- [ ] **Step 4: Run new tests + whole suite** — `uv run pytest src/lottie/core/tests/test_base_agent_policy.py -v` (4 pass), then `uv run pytest -q` (whole suite green — every existing agent defaults to `NullPolicyGate`, so no new blocks).

- [ ] **Step 5: Gates** — `uv run mypy --strict src && uv run ruff check` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/core/base_agent.py src/lottie/core/tests/test_base_agent_policy.py
git commit -m "feat(core): enforce capability policy at BaseAgent.run; audit blocked runs"
```

---

### Task 3: Attach the gate in `instantiate_agent` (CLI + serve enforcement)

**Files:**
- Modify: `src/lottie/project/discovery.py`
- Test: `src/lottie/project/tests/test_instantiate_policy.py` (or the existing discovery test module — create a new file to avoid churn)

- [ ] **Step 1: Failing test**

`src/lottie/project/tests/test_instantiate_policy.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.governance.policy import PolicyDenied
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


def _build(demo: Path):  # type: ignore[no-untyped-def]
    cfg = load_agent_config(demo / "agents" / "echo")
    cls = load_agent_class(demo, "echo")
    return instantiate_agent(cls, llm=MockLLMProvider(["hi"]), root=demo, config=cfg), cfg


def test_instantiate_attaches_policy_that_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    # Give the echo agent a capability and deny it in base.yaml.
    cfg_path = demo / "agents" / "echo" / "config.yaml"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace(
            "capabilities: []", "capabilities:\n  - shell"
        ),
        encoding="utf-8",
    )
    (demo / "policies" / "base.yaml").write_text("name: base\ndeny: [shell]\n", encoding="utf-8")
    agent, _ = _build(demo)
    from agents.echo.schema import EchoInput  # type: ignore[import-not-found]

    with pytest.raises(PolicyDenied):
        agent.run(EchoInput(query="hi"))


def test_instantiate_empty_policy_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)  # default base.yaml is empty
    agent, _ = _build(demo)
    from agents.echo.schema import EchoInput  # type: ignore[import-not-found]

    out = agent.run(EchoInput(query="hi"))
    assert out is not None
```

> NOTE: confirm the generated `echo` agent's input class name + field via `lottie create agent echo`
> output / `agents/echo/schema.py` (it has been `EchoInput(query: str)` historically). If the
> generated names differ, adjust the import/field in the test to match the real scaffold — do not
> change the production code to fit a guessed name. The conftest `_disable_audit` fixture keeps these
> runs from writing a real audit.db; the policy check still fires (it does not depend on audit).

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest src/lottie/project/tests/test_instantiate_policy.py -v` → `PolicyDenied` not raised (gate not attached yet).

- [ ] **Step 3: Implement in `src/lottie/project/discovery.py`**

Add `from lottie.governance.policy import build_policy_gate` with the other imports. Change the tail of `instantiate_agent` so it attaches the gate to the constructed agent on BOTH paths:
```python
    if hasattr(agent_cls, "from_project"):
        agent: BaseAgent[BaseModel, BaseModel] = agent_cls.from_project(
            llm=llm, root=root, config=config, enable_benchmarks=enable_benchmarks
        )
    else:
        agent = agent_cls(llm=llm, enable_benchmarks=enable_benchmarks)
    agent.set_policy(
        build_policy_gate(root, policies=config.policies, capabilities=config.capabilities)
    )
    return agent
```
(Drop the old `# type: ignore[no-any-return]` since we now assign-then-return; if mypy complains about the `from_project` return type, keep a localized `# type: ignore` on that assignment only.)

- [ ] **Step 4: Run new tests + whole suite** — `uv run pytest src/lottie/project/tests/test_instantiate_policy.py -v` (2 pass), then `uv run pytest -q` (whole suite green; existing projects ship empty `base.yaml` → no blocks).

- [ ] **Step 5: Gates** — `uv run mypy --strict src && uv run ruff check` → clean.

- [ ] **Step 6: Commit**
```bash
git add src/lottie/project/discovery.py src/lottie/project/tests/test_instantiate_policy.py
git commit -m "feat(project): instantiate_agent attaches the capability policy gate"
```

---

## Self-review checklist (controller, before finishing)

- [ ] Spec coverage: capabilities evaluated vs merged declared policies; deny→`PolicyDenied`, escalate→`PolicyEscalation` (deny>escalate>allow); empty/missing rules behave per spec (empty=allow-all, missing=`PolicyConfigError`); blocked runs audited `denied`/`escalated` and never reach `_execute`; gate attached by `instantiate_agent`; `NullPolicyGate` default unchanged.
- [ ] `governance.policy` imports no `lottie.core` / `lottie.project` (acyclic; primitives in `build_policy_gate`).
- [ ] Whole suite green (existing empty `base.yaml` ⇒ no new blocks); no test weakened.
- [ ] Type names consistent: `Policy`, `PolicyGate`, `NullPolicyGate`, `PolicyViolation`/`PolicyDenied`/`PolicyEscalation`/`PolicyConfigError`, `load_policy`, `build_policy_gate`, `set_policy`.
- [ ] `uv run pytest -q`, `uv run mypy --strict src`, `uv run ruff check` green.
- [ ] Do NOT push — finish via finishing-a-development-branch, wait for the user.
```
