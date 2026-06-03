# lottie serve — Serving Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/lottie/serve/` — a transport-agnostic `AgentService` that lists and runs agents by name, gated by a pluggable identity `SecurityGate`, with a transport-agnostic error hierarchy.

**Architecture:** A thin service over the existing `project.config` + `project.discovery` + `llm.build_provider` run path (same flow as `cli/run.py`), factored so future Phase-4 transports (REST/MCP) share one code path. Sync-only; no web/MCP deps. Every run flows `payload → gate.check_input → validate → agent.run → gate.check_output → RunResult`.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, `MockLLMProvider`, mypy --strict, ruff. No new dependencies.

---

## File structure

| File | Responsibility |
|---|---|
| `src/lottie/serve/__init__.py` | Public exports |
| `src/lottie/serve/schema.py` | `AgentInfo`, `RunResult` (Pydantic, no logic) |
| `src/lottie/serve/security.py` | `SecurityGate` — identity `check_input`/`check_output`, injectable |
| `src/lottie/serve/service.py` | `ServeError` hierarchy + `AgentService` |
| `src/lottie/serve/tests/__init__.py` | Test package marker |
| `src/lottie/serve/tests/test_schema.py` | Schema defaults |
| `src/lottie/serve/tests/test_security.py` | Identity gate contract |
| `src/lottie/serve/tests/test_service.py` | `list_agents` + `run_agent` (MockLLM only) |

Reference APIs (already exist — do not modify):
- `lottie.project.discovery.discover_agents(root) -> list[UnitInfo]` where `UnitInfo` has `.name: str`, `.provider: str | None` (import-free; provider resolution already guarded).
- `lottie.project.discovery.load_input_model(root, name) -> type[BaseModel]`, `load_agent_class(root, name) -> type[BaseAgent[BaseModel, BaseModel]]` (import user code).
- `lottie.project.config.load_agent_config(unit_dir) -> AgentConfig` with `.provider: str`.
- `lottie.llm.build_provider(name: str) -> LLMProvider`.
- `agent.run(data) -> BaseModel`; `agent.last_metrics: RunMetrics | None` with `.latency_ms: float`, `.input_tokens: int`, `.output_tokens: int`, `.cost_usd: float`.

---

## Task 1: Schemas (`AgentInfo`, `RunResult`)

**Files:**
- Create: `src/lottie/serve/__init__.py` (empty for now)
- Create: `src/lottie/serve/schema.py`
- Create: `src/lottie/serve/tests/__init__.py` (empty)
- Test: `src/lottie/serve/tests/test_schema.py`

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p src/lottie/serve/tests
: > src/lottie/serve/__init__.py
: > src/lottie/serve/tests/__init__.py
```

- [ ] **Step 2: Write the failing test**

`src/lottie/serve/tests/test_schema.py`:

```python
from __future__ import annotations

from lottie.serve.schema import AgentInfo, RunResult


def test_agent_info_defaults() -> None:
    info = AgentInfo(name="echo")
    assert info.name == "echo"
    assert info.provider is None


def test_run_result_defaults_and_output() -> None:
    result = RunResult(agent="echo", output={"result": "hi", "n": 3})
    assert result.agent == "echo"
    assert result.output == {"result": "hi", "n": 3}
    assert result.latency_ms == 0.0
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cost_usd == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest src/lottie/serve/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.serve.schema'`

- [ ] **Step 4: Write minimal implementation**

`src/lottie/serve/schema.py`:

```python
"""Serving-core schemas — transport-agnostic agent metadata and run output."""

from __future__ import annotations

from pydantic import BaseModel


class AgentInfo(BaseModel):
    """One discovered agent, import-free (name + configured provider)."""

    name: str
    provider: str | None = None


class RunResult(BaseModel):
    """Result of one agent run: output payload plus per-run metrics."""

    agent: str
    output: dict[str, object]  # output.model_dump(); dict (not Any) keeps mypy honest
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest src/lottie/serve/tests/test_schema.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/__init__.py src/lottie/serve/schema.py src/lottie/serve/tests/__init__.py src/lottie/serve/tests/test_schema.py
git commit -m "feat(serve): add AgentInfo and RunResult schemas"
```

---

## Task 2: SecurityGate (identity, pluggable)

**Files:**
- Create: `src/lottie/serve/security.py`
- Test: `src/lottie/serve/tests/test_security.py`

- [ ] **Step 1: Write the failing test**

`src/lottie/serve/tests/test_security.py`:

```python
from __future__ import annotations

from lottie.serve.security import SecurityGate


def test_check_input_is_identity() -> None:
    gate = SecurityGate()
    assert gate.check_input('{"a": 1}') == '{"a": 1}'


def test_check_output_is_identity() -> None:
    gate = SecurityGate()
    assert gate.check_output('{"result": "ok"}') == '{"result": "ok"}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/serve/tests/test_security.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.serve.security'`

- [ ] **Step 3: Write minimal implementation**

`src/lottie/serve/security.py`:

```python
"""Single security chokepoint for external content entering/leaving a run.

Identity for now. The real InputSanitizerSkill / OutputValidationSkill /
SecretDetectionSkill (Phase 1) swap in via constructor injection without
changing any call site. See CLAUDE.md rules 8 and 9.
"""

from __future__ import annotations


class SecurityGate:
    """Identity gate. Subclass / replace to perform real scanning."""

    def check_input(self, text: str) -> str:
        # TODO(phase1): route through InputSanitizerSkill
        return text

    def check_output(self, text: str) -> str:
        # TODO(phase1): route through OutputValidationSkill + SecretDetectionSkill
        return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/serve/tests/test_security.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lottie/serve/security.py src/lottie/serve/tests/test_security.py
git commit -m "feat(serve): add identity SecurityGate seam"
```

---

## Task 3: AgentService — errors + `list_agents`

**Files:**
- Create: `src/lottie/serve/service.py`
- Test: `src/lottie/serve/tests/test_service.py`

- [ ] **Step 1: Write the failing test**

`src/lottie/serve/tests/test_service.py` (this file grows in Task 4 — write the full helper block now):

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.serve.service import AgentService

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a real project with one generated `echo` agent on disk."""
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def test_list_agents_returns_name_and_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    svc = AgentService(demo)
    infos = svc.list_agents()
    assert [i.name for i in infos] == ["echo"]
    assert infos[0].provider is not None
    assert "anthropic" in infos[0].provider


def test_list_agents_does_not_import_user_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    broken = demo / "agents" / "broken"
    broken.mkdir()
    (broken / "agent.py").write_text("this is !!! not valid python", encoding="utf-8")
    svc = AgentService(demo)
    names = {i.name for i in svc.list_agents()}
    assert {"echo", "broken"} <= names  # broken lists despite unimportable agent.py
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/serve/tests/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.serve.service'`

- [ ] **Step 3: Write minimal implementation**

`src/lottie/serve/service.py`:

```python
"""Transport-agnostic serving core: list and run agents by name."""

from __future__ import annotations

from pathlib import Path

from lottie.project.discovery import discover_agents
from lottie.serve.schema import AgentInfo
from lottie.serve.security import SecurityGate


class ServeError(Exception):
    """Base for serving-core errors. Transport-agnostic — no typer."""


class AgentNotFoundError(ServeError):
    """No agents/<name>/agent.py exists."""


class InvalidInputError(ServeError):
    """The payload failed the agent's Input validation."""


class AgentExecutionError(ServeError):
    """The agent raised while running."""


class AgentService:
    """Lists and runs agents under a project root, gated by a SecurityGate."""

    def __init__(self, root: Path, *, gate: SecurityGate | None = None) -> None:
        self._root = root
        self._gate = gate or SecurityGate()

    def list_agents(self) -> list[AgentInfo]:
        """One AgentInfo per discovered agent. Import-free."""
        return [
            AgentInfo(name=unit.name, provider=unit.provider)
            for unit in discover_agents(self._root)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/serve/tests/test_service.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lottie/serve/service.py src/lottie/serve/tests/test_service.py
git commit -m "feat(serve): add AgentService with errors and list_agents"
```

---

## Task 4: AgentService — `run_agent`

**Files:**
- Modify: `src/lottie/serve/service.py` (add imports + `run_agent` method)
- Test: `src/lottie/serve/tests/test_service.py` (append run_agent tests)

- [ ] **Step 1: Write the failing tests**

Append to `src/lottie/serve/tests/test_service.py`. Add these imports to the top import block:

```python
from collections.abc import Mapping

from lottie.llm import LLMResponse, Message, MockLLMProvider
from lottie.llm.base import LLMProvider
from lottie.serve.security import SecurityGate
from lottie.serve.service import (
    AgentExecutionError,
    AgentNotFoundError,
    AgentService,
    InvalidInputError,
)
```

Then append these test cases and helpers:

```python
class _BoomProvider(LLMProvider):
    """Provider whose complete() always raises — to force an execution error."""

    @property
    def model(self) -> str:
        return "boom/boom"

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("boom")


class _SpyGate(SecurityGate):
    """Records the order of gate calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def check_input(self, text: str) -> str:
        self.calls.append("in")
        return super().check_input(text)

    def check_output(self, text: str) -> str:
        self.calls.append("out")
        return super().check_output(text)


def _mock_provider(monkeypatch: pytest.MonkeyPatch, response: str = "hello world") -> None:
    """Patch build_provider in the service module to return a MockLLMProvider."""
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider([response]),
    )


def test_run_agent_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    svc = AgentService(demo)
    result = svc.run_agent("echo", {"query": "hi"})
    assert result.agent == "echo"
    assert result.output == {"result": "hello world"}
    assert result.latency_ms >= 0.0  # metrics populated from last_metrics


def test_run_agent_unknown_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    svc = AgentService(demo)
    with pytest.raises(AgentNotFoundError):
        svc.run_agent("nope", {"query": "hi"})


def test_run_agent_invalid_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    svc = AgentService(demo)
    with pytest.raises(InvalidInputError):
        svc.run_agent("echo", {"wrong": "field"})  # echo Input requires `query`


def test_run_agent_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lottie.serve.service.build_provider", lambda name: _BoomProvider()
    )
    svc = AgentService(demo)
    with pytest.raises(AgentExecutionError):
        svc.run_agent("echo", {"query": "hi"})


def test_run_agent_gate_called_input_then_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    gate = _SpyGate()
    svc = AgentService(demo, gate=gate)
    svc.run_agent("echo", {"query": "hi"})
    assert gate.calls == ["in", "out"]


def test_run_agent_provider_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    captured: list[str] = []

    def fake_build(name: str) -> MockLLMProvider:
        captured.append(name)
        return MockLLMProvider(["ok"])

    monkeypatch.setattr("lottie.serve.service.build_provider", fake_build)
    svc = AgentService(demo)

    svc.run_agent("echo", {"query": "hi"}, provider="openai/gpt-4o")
    assert captured == ["openai/gpt-4o"]

    captured.clear()
    svc.run_agent("echo", {"query": "hi"})  # falls back to config provider
    assert captured[0] is not None
    assert "anthropic" in captured[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/lottie/serve/tests/test_service.py -v`
Expected: FAIL — `AttributeError: 'AgentService' object has no attribute 'run_agent'` (and `ImportError` for the new symbols if run before the impl import resolves)

- [ ] **Step 3: Write the implementation**

Edit `src/lottie/serve/service.py`. Replace the import block and add the `run_agent` method.

New import block (top of file, after the module docstring):

```python
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from lottie.llm import build_provider
from lottie.project.config import load_agent_config
from lottie.project.discovery import (
    discover_agents,
    load_agent_class,
    load_input_model,
)
from lottie.serve.schema import AgentInfo, RunResult
from lottie.serve.security import SecurityGate
```

Add this method to `AgentService` (after `list_agents`):

```python
    def run_agent(
        self,
        name: str,
        payload: Mapping[str, object],
        *,
        provider: str | None = None,
    ) -> RunResult:
        """Run one agent: gate input, validate, run, gate output, return metrics."""
        unit_dir = self._root / "agents" / name
        if not (unit_dir / "agent.py").is_file():
            raise AgentNotFoundError(f"agent '{name}' not found")

        self._gate.check_input(json.dumps(payload))

        cfg = load_agent_config(unit_dir)
        llm = build_provider(provider or cfg.provider)

        input_model = load_input_model(self._root, name)
        try:
            data = input_model.model_validate(payload)
        except ValidationError as exc:
            raise InvalidInputError(f"invalid input for '{name}': {exc}") from exc

        agent = load_agent_class(self._root, name)(llm=llm)
        try:
            output = agent.run(data)
        except Exception as exc:  # noqa: BLE001 — any agent failure → one typed error
            raise AgentExecutionError(f"agent '{name}' failed: {exc}") from exc

        self._gate.check_output(output.model_dump_json())

        m = agent.last_metrics
        return RunResult(
            agent=name,
            output=output.model_dump(),
            latency_ms=m.latency_ms if m else 0.0,
            input_tokens=m.input_tokens if m else 0,
            output_tokens=m.output_tokens if m else 0,
            cost_usd=m.cost_usd if m else 0.0,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/serve/tests/test_service.py -v`
Expected: PASS (8 passed — 2 from Task 3 + 6 new)

- [ ] **Step 5: Commit**

```bash
git add src/lottie/serve/service.py src/lottie/serve/tests/test_service.py
git commit -m "feat(serve): add AgentService.run_agent with gate and metrics"
```

---

## Task 5: Public exports + final verification

**Files:**
- Modify: `src/lottie/serve/__init__.py`
- Test: `src/lottie/serve/tests/test_service.py` (add one export-surface test)

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/serve/tests/test_service.py`:

```python
def test_public_exports() -> None:
    import lottie.serve as serve

    for symbol in (
        "AgentInfo",
        "RunResult",
        "SecurityGate",
        "AgentService",
        "ServeError",
        "AgentNotFoundError",
        "InvalidInputError",
        "AgentExecutionError",
    ):
        assert hasattr(serve, symbol), symbol
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/serve/tests/test_service.py::test_public_exports -v`
Expected: FAIL — `AssertionError: AgentInfo`

- [ ] **Step 3: Write the implementation**

`src/lottie/serve/__init__.py`:

```python
"""Lottie serving core — transport-agnostic agent list/run service."""

from __future__ import annotations

from lottie.serve.schema import AgentInfo, RunResult
from lottie.serve.security import SecurityGate
from lottie.serve.service import (
    AgentExecutionError,
    AgentNotFoundError,
    AgentService,
    InvalidInputError,
    ServeError,
)

__all__ = [
    "AgentExecutionError",
    "AgentInfo",
    "AgentNotFoundError",
    "AgentService",
    "InvalidInputError",
    "RunResult",
    "SecurityGate",
    "ServeError",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/serve/tests/test_service.py::test_public_exports -v`
Expected: PASS

- [ ] **Step 5: Full suite + type + lint gate**

Run:
```bash
uv run pytest -q
uv run mypy --strict src/lottie/serve
uv run ruff check src/lottie/serve
```
Expected: all serve tests pass, full suite green (was 169 → now 169 + 13 new = 182), `mypy --strict` clean, `ruff` clean.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/__init__.py src/lottie/serve/tests/test_service.py
git commit -m "feat(serve): export public serving-core surface"
```

---

## Self-review notes (resolved during planning)

- **`AgentInfo` shape:** trimmed to `name` + `provider` — the only fields `discover_agents` yields import-free. Input/output model names need importing user code and have no config source; deferred to `inspect` / a future `describe_agent`.
- **LLM-free tests:** `run_agent` builds its own provider via `build_provider`, so tests monkeypatch `lottie.serve.service.build_provider` to return a `MockLLMProvider` (or `_BoomProvider`). No real LLM, satisfies CLAUDE.md rule 5.
- **Error decoupling:** `ServeError` hierarchy carries no `typer` dependency — the core stays transport-agnostic; transports map these later.
- **Method/type consistency:** `run_agent(name, payload, *, provider=None) -> RunResult`, `list_agents() -> list[AgentInfo]`, `SecurityGate.check_input/check_output(text: str) -> str`, `RunResult.output: dict[str, object]` — names identical across all tasks.
