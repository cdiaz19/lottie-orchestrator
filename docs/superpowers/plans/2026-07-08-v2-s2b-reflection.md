# V2 S2b — Reflexive Write-Back Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a successful run, an opt-in reflection step distills the execution trajectory into durable memory lessons — through the S1 gateway (gated, provenance-stamped, audited), bounded by the run's token budget, and observable via an OpenTelemetry span. Plus a `lottie reflect` CLI for manual episodic→semantic consolidation.

**Architecture:** A pure `RunTrajectory` + `build_reflection_prompt`/`parse_reflection` (`memory/reflection.py`). A `BaseAgent._maybe_reflect(data, output)` post-run hook (opt-in `memory.reflect.enabled`): it builds the trajectory from `last_metrics`, primes a `RunContext` with the run's already-spent tokens so `self.complete()` enforces the per-run token cap (skip-when-exhausted), calls the reflection LLM, parses ADD-only deltas, and applies them via a lazily-built `MemoryAgent` gateway (`origin=REFLECTION`). Best-effort: reflection never fails the run. `memory.reflect` config + `instantiate_agent` wiring + a `lottie reflect` CLI (reuses the S1-gated `MemoryAgent` consolidation).

**Tech Stack:** Python 3.12+, Pydantic v2, `uv`, `pytest`, `mypy --strict`, `ruff`.

## Global Constraints

- **Rule 2:** typed models cross boundaries (`RunTrajectory`). **Rule 5:** unit tests use `MockLLMProvider`/`MockMemoryClient`. **Rule 6 / 7b:** `mypy --strict` (no `Any`, no `# type: ignore`) + `ruff` + `pytest --all-extras` green before push. **Rule 7:** conventional commits. **Rule 13b:** reflection writes go through the `MemoryAgent` gateway — never `self.memory.remember` directly.
- **OFF by default:** `memory.reflect.enabled` defaults False. An unchanged project reflects nothing.
- **Cost-budgeted:** the reflection LLM call routes through `self.complete()` with a `RunContext` primed to the run's spent tokens, so the existing `_enforce_token_cap` skips/aborts reflection when the run's `max_run_tokens` is already reached. Pre-check also skips before spending.
- **Best-effort / fail-open:** reflection failure (LLM error, gate rejection of a lesson, store error) NEVER fails the primary run — it is caught and warned. The primary output is already gated and returned.
- **Poisoning defense:** reflected lessons pass through the `MemoryAgent` gateway's `MemoryContentGate` like any write (a lesson that reads as an injection/secret is rejected). Provenance `origin=REFLECTION`, `source_agent=self.name`.
- **Observable:** the reflection step is wrapped in an OTel `run_span`.
- **Acyclic imports:** `memory/reflection.py` imports only `lottie.llm` (Message) + `lottie.memory.schema` (pure). `base_agent.py` imports `memory.reflection` + `memory.schema` at module level and `memory.agent` **lazily inside `_maybe_reflect`** (avoids the `core ↔ memory.agent` cycle).
- **Scope:** S2b = post-run reflection hook + config + wiring + `lottie reflect` CLI + OTel ONLY. Deltas are ADD-only (dedup folds; UPDATE/DEPRECATE by the Reflector deferred). Reflection runs on the SUCCESS path only (failed-run reflection deferred). NO distillation (S3), NO benchmark delta (S4), NO harness (S5). Reflection LLM cost is bounded by the token cap but NOT added to the cumulative cost ledger (deferred — documented).

---

## File Structure

- `src/lottie/memory/reflection.py` — **create**: `RunTrajectory`, `build_reflection_prompt`, `parse_reflection`, `REFLECT_SYSTEM_PROMPT`.
- `src/lottie/project/config.py` — **modify**: `ReflectConfig` + `MemoryConfig.reflect`.
- `src/lottie/core/base_agent.py` — **modify**: reflect state + `set_reflect` + `_maybe_reflect` + call in `run()`.
- `src/lottie/project/discovery.py` — **modify**: wire `set_reflect` in `instantiate_agent`.
- `src/lottie/cli/reflect.py` — **create**: `lottie reflect <agent>`.
- `src/lottie/cli/app.py` — **modify**: register the command.
- `CLAUDE.md` — **modify**: one-line note on reflection default-off + gateway.
- Tests: `src/lottie/memory/tests/test_reflection.py`, `src/lottie/project/tests/test_memory_config.py` (extend), `src/lottie/core/tests/test_reflection_hook.py` (create), `src/lottie/project/tests/test_memory_injection.py` (extend), `src/lottie/cli/tests/test_reflect_cli.py` (create).

---

## Task 1: `RunTrajectory` + reflection prompt/parse (pure)

**Files:**
- Create: `src/lottie/memory/reflection.py`
- Test: `src/lottie/memory/tests/test_reflection.py`

**Interfaces:**
- Consumes: `Message` (`lottie.llm`), `DeltaOp`/`MemoryDelta` (`lottie.memory.schema`).
- Produces: `RunTrajectory(task: str, outcome: str, success: bool, error: str | None = None, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0, latency_ms: float = 0.0)`; `REFLECT_SYSTEM_PROMPT: str`; `build_reflection_prompt(trajectory: RunTrajectory) -> list[Message]`; `parse_reflection(text: str) -> list[MemoryDelta]` (one non-blank line → one ADD delta, `tags=["reflection"]`).

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/test_reflection.py`:

```python
from lottie.memory.reflection import (
    RunTrajectory,
    build_reflection_prompt,
    parse_reflection,
)
from lottie.memory.schema import DeltaOp


def _traj() -> RunTrajectory:
    return RunTrajectory(
        task='{"q": "sum 2+2"}',
        outcome='{"a": "4"}',
        success=True,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=12.0,
    )


def test_prompt_has_system_and_user_with_trajectory() -> None:
    msgs = build_reflection_prompt(_traj())
    assert [m.role for m in msgs] == ["system", "user"]
    assert "sum 2+2" in msgs[1].content
    assert "success" in msgs[1].content.lower()


def test_parse_reflection_one_add_per_line() -> None:
    deltas = parse_reflection("check units before returning\n\nprefer int division here\n")
    assert len(deltas) == 2
    assert all(d.op is DeltaOp.ADD for d in deltas)
    assert deltas[0].content == "check units before returning"
    assert deltas[0].tags == ["reflection"]


def test_parse_reflection_empty_is_no_deltas() -> None:
    assert parse_reflection("   \n\n") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_reflection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.memory.reflection'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/memory/reflection.py`:

```python
"""Reflection primitives: distill a run trajectory into memory lessons.

Pure — the LLM call itself lives on BaseAgent (so it counts against the run's token
budget); this module only builds the prompt and parses the result. Imports only
lottie.llm (Message) + lottie.memory.schema, so it stays acyclic.
"""

from __future__ import annotations

from pydantic import BaseModel

from lottie.llm import Message
from lottie.memory.schema import DeltaOp, MemoryDelta

REFLECT_SYSTEM_PROMPT = (
    "You are a reflection step run after an agent finished a task. Read the execution "
    "trajectory and distill at most a few DURABLE, reusable lessons that would help a "
    "future run of this agent do better. Each lesson: one line, standalone, imperative, "
    "no numbering or bullets. If there is no durable lesson, output nothing."
)


class RunTrajectory(BaseModel):
    """A minimal record of one execution, fed to the Reflector."""

    task: str
    outcome: str
    success: bool
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


def build_reflection_prompt(trajectory: RunTrajectory) -> list[Message]:
    """Render the trajectory into a system+user message pair for the reflection call."""
    body = (
        f"task: {trajectory.task}\n"
        f"success: {trajectory.success}\n"
        f"outcome: {trajectory.outcome}\n"
        f"error: {trajectory.error or 'none'}\n"
        f"tokens: {trajectory.input_tokens + trajectory.output_tokens}  "
        f"cost_usd: {trajectory.cost_usd}"
    )
    return [
        Message(role="system", content=REFLECT_SYSTEM_PROMPT),
        Message(role="user", content=body),
    ]


def parse_reflection(text: str) -> list[MemoryDelta]:
    """One non-blank output line → one ADD delta tagged 'reflection'."""
    return [
        MemoryDelta(op=DeltaOp.ADD, content=line.strip(), tags=["reflection"])
        for line in text.splitlines()
        if line.strip()
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/memory/tests/test_reflection.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/memory/reflection.py src/lottie/memory/tests/test_reflection.py
git commit -m "feat(memory): RunTrajectory + reflection prompt/parse (V2 S2b)"
```

---

## Task 2: `ReflectConfig` + `MemoryConfig.reflect`

**Files:**
- Modify: `src/lottie/project/config.py`
- Test: `src/lottie/project/tests/test_memory_config.py`

**Interfaces:**
- Produces: `ReflectConfig(enabled: bool = False)`; `MemoryConfig.reflect: ReflectConfig = ReflectConfig()`.

- [ ] **Step 1: Write the failing test**

Add to `src/lottie/project/tests/test_memory_config.py`:

```python
def test_memory_config_reflect_defaults_off() -> None:
    from lottie.project.config import AgentConfig, ReflectConfig

    cfg = AgentConfig(provider="mock")
    assert isinstance(cfg.memory.reflect, ReflectConfig)
    assert cfg.memory.reflect.enabled is False


def test_memory_config_reflect_from_dict() -> None:
    from lottie.project.config import AgentConfig

    cfg = AgentConfig.model_validate(
        {"provider": "mock", "memory": {"enabled": True, "reflect": {"enabled": True}}}
    )
    assert cfg.memory.reflect.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/project/tests/test_memory_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'ReflectConfig'`.

- [ ] **Step 3: Write minimal implementation**

Edit `src/lottie/project/config.py`. Add `ReflectConfig` before `MemoryConfig` (near `RecallConfig`):

```python
class ReflectConfig(BaseModel):
    """Per-agent post-run reflection config. Disabled by default."""

    enabled: bool = False
```

Add the field to `MemoryConfig` (after `recall`):

```python
    reflect: ReflectConfig = ReflectConfig()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/project/tests/test_memory_config.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/project/config.py src/lottie/project/tests/test_memory_config.py
git commit -m "feat(config): memory.reflect config (V2 S2b)"
```

---

## Task 3: BaseAgent reflection hook

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Test: `src/lottie/core/tests/test_reflection_hook.py`

**Interfaces:**
- Consumes: `self.last_metrics`, `self.memory`, `self._audit`, `self.complete`, `self._max_run_tokens`; `RunContext` (`lottie.core.metrics`); `run_span` (`lottie.governance.otel`); `RunTrajectory`/`build_reflection_prompt`/`parse_reflection` (`lottie.memory.reflection`); `MemoryOrigin` (`lottie.memory.schema`); `TokenCapExceeded` (already imported), `TurnLimitExceeded` (already defined in this file); `MemoryAgent` (lazy import).
- Produces: state `self._reflect_enabled: bool = False`, `self._reflect_namespace: str = ""`; setter `set_reflect(self, *, enabled: bool, namespace: str) -> None`; method `_maybe_reflect(self, data: InputT, output: OutputT) -> None`; a call to `self._maybe_reflect(data, output)` in `run()` after the output gate, before `return output`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/core/tests/test_reflection_hook.py`:

```python
from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryOrigin, MemoryQuery, MemoryTier


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Answer(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a="answer")


def test_reflection_disabled_writes_nothing() -> None:
    mem = MockMemoryClient()
    agent = _Answer(llm=MockLLMProvider(["lesson one"]), memory=mem)
    agent.run(_In(q="hi"))
    assert mem.records == []


def test_reflection_enabled_writes_lessons_with_provenance() -> None:
    mem = MockMemoryClient()
    # first canned response = _execute has no LLM call here, so the ONLY completion is reflection
    agent = _Answer(llm=MockLLMProvider(["always validate units\ncache the parsed config"]), memory=mem)
    agent.set_reflect(enabled=True, namespace="ns")
    agent.run(_In(q="hi"))
    notes = mem.recall(MemoryQuery(text="", namespace="ns", tier=MemoryTier.SEMANTIC)).hits
    contents = {h.record.content for h in notes}
    assert "always validate units" in contents
    assert "cache the parsed config" in contents
    assert all(h.record.origin is MemoryOrigin.REFLECTION for h in notes)
    assert all(h.record.source_agent == agent.name for h in notes)


def test_reflection_failure_does_not_break_run() -> None:
    # NullMemoryClient (default) raises on write; run must still return normally.
    agent = _Answer(llm=MockLLMProvider(["a lesson"]))
    agent.set_reflect(enabled=True, namespace="ns")
    out = agent.run(_In(q="hi"))
    assert out.a == "answer"


def test_reflection_skipped_when_token_cap_reached() -> None:
    mem = MockMemoryClient()
    agent = _Answer(llm=MockLLMProvider(["should not be written"]), memory=mem)
    agent.set_reflect(enabled=True, namespace="ns")
    # simulate a run that already consumed its token budget
    agent.set_run_limits(max_run_tokens=1)
    from lottie.core.metrics import RunMetrics
    from datetime import UTC, datetime
    agent.last_metrics = RunMetrics(
        name=agent.name, kind="agent", provider=None, timestamp=datetime.now(UTC),
        latency_ms=1.0, input_tokens=5, output_tokens=5, cost_usd=0.0, retry_count=0,
        success=True, version=None, error=None,
    )
    agent._maybe_reflect(_In(q="hi"), _Out(a="answer"))
    assert mem.records == []  # cap already reached -> reflection skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/core/tests/test_reflection_hook.py -q`
Expected: FAIL — `AttributeError: '_Answer' object has no attribute 'set_reflect'`.

- [ ] **Step 3: Write minimal implementation**

Edit `src/lottie/core/base_agent.py`.

Add imports (with the other `lottie.memory`/governance imports at the top):

```python
from lottie.core.metrics import RunContext
from lottie.governance.otel import run_span
from lottie.memory.reflection import (
    RunTrajectory,
    build_reflection_prompt,
    parse_reflection,
)
```

Add `MemoryOrigin` to the existing `lottie.memory.schema` import (which already brings `MemoryQuery`, `MemoryTier` from S2a).

Add reflect state in `__init__` (after the recall state from S2a):

```python
        self._reflect_enabled: bool = False
        self._reflect_namespace: str = ""
```

Add the setter (near `set_recall`):

```python
    def set_reflect(self, *, enabled: bool, namespace: str) -> None:
        """Enable post-run reflexive write-back for this agent (via instantiate_agent)."""
        self._reflect_enabled = enabled
        self._reflect_namespace = namespace
```

Add the hook method (near `_load_recall`):

```python
    def _maybe_reflect(self, data: InputT, output: OutputT) -> None:
        """Best-effort: distill the finished run into memory lessons via the gateway.

        Routes the reflection LLM call through self.complete() with a RunContext primed
        to the run's spent tokens, so the per-run token cap enforces (skip-when-exhausted).
        Never raises — reflection failure must not fail the already-successful run.
        """
        if not self._reflect_enabled:
            return
        m = self.last_metrics
        used = (m.input_tokens + m.output_tokens) if m is not None else 0
        if self._max_run_tokens is not None and used >= self._max_run_tokens:
            warnings.warn("reflection skipped: run token cap reached", stacklevel=2)
            return
        self._recall_prefix = ""  # reflection gets no recalled context of its own
        trajectory = RunTrajectory(
            task=data.model_dump_json(),
            outcome=output.model_dump_json(),
            success=True,
            input_tokens=m.input_tokens if m is not None else 0,
            output_tokens=m.output_tokens if m is not None else 0,
            cost_usd=m.cost_usd if m is not None else 0.0,
            latency_ms=m.latency_ms if m is not None else 0.0,
        )
        ctx = RunContext()
        ctx.input_tokens = used  # prime so _enforce_token_cap counts cumulatively
        ctx.cost_usd = m.cost_usd if m is not None else 0.0
        self._active_ctx = ctx
        try:
            with run_span(f"{self.name}.reflect", self.kind):
                response = self.complete(build_reflection_prompt(trajectory))
                deltas = parse_reflection(response.content)
                if deltas:
                    from lottie.memory.agent import MemoryAgent  # lazy: avoid core↔memory.agent cycle

                    gateway = MemoryAgent(llm=self.llm, memory=self.memory, audit=self._audit)
                    gateway.apply(
                        deltas,
                        namespace=self._reflect_namespace,
                        source_agent=self.name,
                        origin=MemoryOrigin.REFLECTION,
                    )
        except (TokenCapExceeded, TurnLimitExceeded) as exc:
            warnings.warn(f"reflection skipped: {exc}", stacklevel=2)
        except Exception as exc:  # best-effort — never fail the run
            warnings.warn(f"reflection failed: {exc}", stacklevel=2)
        finally:
            self._active_ctx = None
```

In `run()`, add the hook call after the output gate, before `return output`:

```python
            self._verify(data, output)
            self._security.check_output(output.model_dump_json())
            self._maybe_reflect(data, output)  # best-effort post-run reflexive write-back
            return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/core/tests/test_reflection_hook.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the core suite for regressions**

Run: `uv run pytest src/lottie/core -q`
Expected: PASS (reflection defaults disabled → existing runs unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/lottie/core/base_agent.py src/lottie/core/tests/test_reflection_hook.py
git commit -m "feat(core): post-run reflexive write-back hook, budget-bounded (V2 S2b)"
```

---

## Task 4: Wire reflect config through `instantiate_agent`

**Files:**
- Modify: `src/lottie/project/discovery.py`
- Test: `src/lottie/project/tests/test_memory_injection.py`

**Interfaces:**
- Produces: inside the existing `if config.memory.enabled:` block in `instantiate_agent`, when `config.memory.reflect.enabled`, call `agent.set_reflect(enabled=True, namespace=config.memory.namespace or agent.name)`.

- [ ] **Step 1: Write the failing test**

Add to `src/lottie/project/tests/test_memory_injection.py`:

```python
def test_reflect_wired_when_enabled(tmp_path: Path) -> None:
    agent = instantiate_agent(
        _Echo,
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": True, "reflect": {"enabled": True}}),
    )
    assert agent._reflect_enabled is True
    assert agent._reflect_namespace == agent.name


def test_reflect_off_when_memory_disabled(tmp_path: Path) -> None:
    agent = instantiate_agent(
        _Echo,
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": False, "reflect": {"enabled": True}}),
    )
    assert agent._reflect_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/project/tests/test_memory_injection.py -q`
Expected: FAIL — `assert agent._reflect_enabled is True` fails.

- [ ] **Step 3: Write minimal implementation**

Edit `src/lottie/project/discovery.py`. Extend the memory block (after the `recall` wiring from S2a):

```python
        if config.memory.reflect.enabled:
            agent.set_reflect(
                enabled=True,
                namespace=config.memory.namespace or agent.name,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/project/tests/test_memory_injection.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/lottie/project/discovery.py src/lottie/project/tests/test_memory_injection.py
git commit -m "feat(project): wire memory.reflect config into instantiate_agent (V2 S2b)"
```

---

## Task 5: `lottie reflect` CLI

**Files:**
- Create: `src/lottie/cli/reflect.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_reflect_cli.py`

**Interfaces:**
- Consumes: `find_project_root`/`load_agent_config` (`lottie.project.config`), `build_provider` (`lottie.llm`), `build_memory_client` (`lottie.memory.store`), `MemoryAgent`/`ReflectionInput` (`lottie.memory`).
- Produces: `reflect(name: str, namespace: str | None = None, limit: int = 50, provider: str | None = None) -> None` — runs the agent's `MemoryAgent` episodic→semantic consolidation for the namespace and prints a summary; registered as `lottie reflect`.

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/test_reflect_cli.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from lottie.cli.app import app

runner = CliRunner()


def _project(tmp_path: Path) -> Path:
    (tmp_path / "lottie.yaml").write_text(
        "project: t\nproviders:\n  default: mock\n", encoding="utf-8"
    )
    unit = tmp_path / "agents" / "digest"
    unit.mkdir(parents=True)
    (unit / "agent.py").write_text("# stub\n", encoding="utf-8")
    (unit / "config.yaml").write_text(
        "provider: mock\nmemory:\n  enabled: true\n  backend: mock\n", encoding="utf-8"
    )
    return tmp_path


def test_reflect_unknown_agent_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(_project(tmp_path))
    result = runner.invoke(app, ["reflect", "nope"])
    assert result.exit_code != 0


def test_reflect_runs_consolidation(tmp_path: Path, monkeypatch) -> None:
    from lottie.llm import MockLLMProvider

    monkeypatch.chdir(_project(tmp_path))
    # build_provider always returns a LiteLLMProvider (real network) — patch it in the
    # reflect module's namespace so the consolidation LLM call is a deterministic mock.
    monkeypatch.setattr(
        "lottie.cli.reflect.build_provider",
        lambda _model: MockLLMProvider(["lesson a\nlesson b"]),
    )
    result = runner.invoke(app, ["reflect", "digest", "--namespace", "ns"])
    assert result.exit_code == 0
    assert "ns" in result.stdout
```

Note: `build_provider(model)` always constructs a `LiteLLMProvider` (verified — there is no "mock" backend), so the CLI test MUST monkeypatch `lottie.cli.reflect.build_provider` as above; otherwise the consolidation makes a real network call.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_reflect_cli.py -q`
Expected: FAIL — no `reflect` command registered (`exit_code == 2`, "No such command").

- [ ] **Step 3: Write minimal implementation**

Create `src/lottie/cli/reflect.py`:

```python
"""`lottie reflect <agent>` — manual episodic→semantic memory consolidation.

Runs the agent's MemoryAgent consolidation (S1-gated: content-screened, deduped,
provenance-stamped, audited) over its memory namespace. Distinct from the automatic
per-run reflection hook — this is the batch/manual curation entry point.
"""

from __future__ import annotations

from typing import Annotated

import typer

from lottie.llm import build_provider
from lottie.memory.agent import MemoryAgent  # NOT re-exported from lottie.memory (import cycle)
from lottie.memory.schema import ReflectionInput
from lottie.memory.store import build_memory_client
from lottie.project.config import find_project_root, load_agent_config


def reflect(
    name: str,
    namespace: Annotated[
        str | None, typer.Option("--namespace", help="Memory namespace (default: agent name).")
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max episodic records to consolidate.")
    ] = 50,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override the LLM provider.")
    ] = None,
) -> None:
    """Consolidate an agent's episodic memory into durable semantic notes."""
    root = find_project_root()
    unit_dir = root / "agents" / name
    if not (unit_dir / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    cfg = load_agent_config(unit_dir)
    llm = build_provider(provider or cfg.provider)
    memory = build_memory_client(root, backend=cfg.memory.backend, path=cfg.memory.path)
    ns = namespace or cfg.memory.namespace or name

    agent = MemoryAgent(llm=llm, memory=memory)
    result = agent.run(ReflectionInput(namespace=ns, limit=limit))

    typer.echo(
        f"reflected '{ns}': consolidated {result.consolidated_count} episodic record(s) "
        f"-> {len(result.written_ids)} semantic note(s)"
    )
    for note in result.notes:
        typer.echo(f"  - {note}")
```

Edit `src/lottie/cli/app.py` — add the import with the other CLI imports and register the command with the other `app.command(...)` registrations:

```python
from lottie.cli.reflect import reflect
```
```python
app.command("reflect")(reflect)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/lottie/cli/tests/test_reflect_cli.py -q`
Expected: PASS (2 tests). If `build_provider("mock")` cannot be built in-test, apply the fixture note from Step 1.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/reflect.py src/lottie/cli/app.py src/lottie/cli/tests/test_reflect_cli.py
git commit -m "feat(cli): lottie reflect — manual memory consolidation (V2 S2b)"
```

---

## Task 6: CLAUDE.md note + full gate

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the note to CLAUDE.md**

Append to rule 13b (or add a short sentence beneath it):

```markdown
   Reflection (post-run write-back) is OFF by default (`memory.reflect.enabled`), routes its
   LLM call through the run's token budget, and writes lessons through the same gateway.
```

- [ ] **Step 2: Run the full local gate (rule 7b)**

```bash
uv sync --dev --all-extras
uv run ruff check .
uv run mypy --strict src
uv run pytest -q
```
Expected: ruff clean; mypy clean (file count +1: `reflection.py` and `cli/reflect.py`); pytest all green (996 + ~13 new S2b tests).

- [ ] **Step 3: Fix any gate failures**

If mypy flags the lazy `MemoryAgent` import or the `RunContext` field assignment, mirror the existing patterns (S1 used a lazy `MemoryAgent` import in the CLI; `RunContext` fields are plain dataclass attributes). No `Any`, no `# type: ignore`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(memory): note reflection default-off + gateway (V2 S2b)"
```

---

## Lab round (R24) — separate `lottie-lab` PR, after S2b merges

Not part of this plan's commits. After merge, add Round 24: a reflect-enabled agent whose scripted MockLLM returns two lessons; run it; assert two SEMANTIC notes land in the namespace with `origin=reflection`; assert reflection is skipped when `max_run_tokens` is already consumed; assert a run with reflection OFF writes nothing; drive `lottie reflect` and confirm consolidation output. Mirror the R20/R17 driver harness.

---

## Self-Review

**Spec coverage (epic §3.4 reflection + §1 item #1):**
- Post-run reflection hook, opt-in, distills trajectory → lessons via gateway → Task 3. ✅
- Minimal `RunTrajectory` (input/output/metrics) → Task 1. ✅
- Cost-budgeted (self.complete + primed ctx + skip-when-exhausted) → Task 3. ✅
- Best-effort (never fails the run) → Task 3 + `test_reflection_failure_does_not_break_run`. ✅
- OTel span → Task 3 `run_span`. ✅
- Provenance origin=REFLECTION via gateway → Task 3. ✅
- `memory.reflect` config + wiring → Tasks 2, 4. ✅
- `lottie reflect` CLI → Task 5. ✅
- Out of scope (distillation, benchmark delta, harness, failed-run reflection, UPDATE/DEPRECATE deltas) → none built. ✅

**Placeholder scan:** no TBD/TODO; full code in each code step; the two test-environment caveats (mock provider in Task 5, `MemoryAgent` export location) name the exact file to check rather than leaving it vague. ✅

**Type consistency:** `set_reflect(*, enabled, namespace)` identical across Task 3 def, Task 4 call, tests. `RunTrajectory`/`build_reflection_prompt`/`parse_reflection` identical Task 1 ↔ Task 3. `ReflectConfig`/`MemoryConfig.reflect` identical Task 2 ↔ Task 4. `reflect(name, namespace, limit, provider)` identical Task 5 def ↔ test. ✅

**Note on scope discipline:** reflection runs on the SUCCESS path only (after the output gate); the `RunTrajectory.error` field exists for a future failed-run reflection but is always `None` here. The reflection LLM cost is bounded by the token cap but is NOT added to the cumulative cost ledger (the primed `RunContext` is discarded, not recorded) — documented deferral. The `lottie reflect` CLI and the auto-hook are deliberately two mechanisms: the CLI consolidates persisted episodic memory (existing `MemoryAgent._execute`), the hook distills the just-finished run — both write through the S1 gateway.
```
