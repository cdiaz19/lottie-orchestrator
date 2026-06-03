# `lottie benchmark` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `lottie benchmark agent <name>` runs an agent against a YAML eval suite, scores assertion-based accuracy, aggregates per-run metrics, prints a table, persists a JSON report, and supports `--compare` across configured providers.

**Architecture:** New `src/lottie/benchmark/` package holds the Pydantic schemas (`schema.py`) and the testable runner (`runner.py` — load suite, run each case, score `expect`, aggregate into a `ProviderReport`/`BenchmarkReport`). `src/lottie/cli/benchmark.py` is a thin `benchmark` sub-Typer (`agent` command) wired into `cli/app.py`. The runner's `run_suite` takes a constructed agent, so scoring/aggregation is unit-tested with `MockLLMProvider`; CLI tests mock `litellm.completion`.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, Rich, PyYAML, pytest. `mypy --strict` + `ruff` must stay clean (no `Any` without justification). Run all tools via `uv run` from the project dir `/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator`. TDD throughout.

**Reference spec:** `docs/superpowers/specs/2026-06-01-benchmark-design.md`

---

## File Structure

- `src/lottie/benchmark/__init__.py` — **create**: public exports (extended across tasks).
- `src/lottie/benchmark/schema.py` — **create**: `EvalExpect`, `EvalCase`, `EvalSuite`, `CaseResult`, `ProviderReport`, `BenchmarkReport`.
- `src/lottie/benchmark/runner.py` — **create**: `load_suite`, `_passes`, `_percentile`, `run_suite`, `benchmark`.
- `src/lottie/benchmark/tests/__init__.py` — **create**: empty.
- `src/lottie/benchmark/tests/test_schema.py` — **create**.
- `src/lottie/benchmark/tests/test_runner.py` — **create**.
- `src/lottie/cli/benchmark.py` — **create**: `benchmark_app` + `agent` command.
- `src/lottie/cli/app.py` — **modify**: register `benchmark_app`.
- `src/lottie/cli/tests/test_benchmark.py` — **create**.

Pattern references to read first: `src/lottie/core/metrics.py` (`RunMetrics`, the per-run record), `src/lottie/core/runnable.py` (`run()` sets `self.last_metrics`), `src/lottie/cli/run.py` (provider build + agent load), `src/lottie/cli/tests/test_run.py` (litellm-mock pattern), `src/lottie/cli/registry.py` (sub-Typer + Rich table style), `src/lottie/project/config.py` (`load_lottie_config`, `load_agent_config`, `find_project_root`), `src/lottie/project/discovery.py` (`load_agent_class`, `load_input_model`).

---

## Task 1: Benchmark schemas

**Files:**
- Create: `src/lottie/benchmark/__init__.py`, `src/lottie/benchmark/schema.py`, `src/lottie/benchmark/tests/__init__.py`, `src/lottie/benchmark/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

Create empty `src/lottie/benchmark/tests/__init__.py`. Then create `src/lottie/benchmark/tests/test_schema.py`:

```python
from __future__ import annotations

from lottie.benchmark.schema import (
    BenchmarkReport,
    CaseResult,
    EvalCase,
    EvalExpect,
    EvalSuite,
    ProviderReport,
)


def test_eval_expect_defaults() -> None:
    e = EvalExpect()
    assert e.equals == {}
    assert e.contains == {}


def test_eval_case_default_expect() -> None:
    c = EvalCase(name="x", input={"query": "hi"})
    assert c.expect.equals == {}
    assert c.input == {"query": "hi"}


def test_eval_suite_holds_cases() -> None:
    suite = EvalSuite(cases=[EvalCase(name="x", input={})])
    assert len(suite.cases) == 1


def test_case_result_defaults() -> None:
    r = CaseResult(name="x", passed=True, success=True)
    assert r.latency_ms == 0.0
    assert r.input_tokens == 0
    assert r.error is None


def test_reports_compose() -> None:
    pr = ProviderReport(
        provider="mock/m",
        case_count=1,
        accuracy=1.0,
        success_rate=1.0,
        latency_p50_ms=5.0,
        latency_p95_ms=5.0,
        mean_cost_usd=0.0,
        total_input_tokens=1,
        total_output_tokens=2,
        cases=[CaseResult(name="x", passed=True, success=True)],
    )
    report = BenchmarkReport(agent="echo", providers=[pr])
    assert report.agent == "echo"
    assert report.providers[0].accuracy == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/benchmark/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.benchmark'`.

- [ ] **Step 3: Write the schemas**

Create `src/lottie/benchmark/schema.py`:

```python
"""Pydantic models for the benchmark subsystem.

Pure data shapes. `EvalSuite`/`EvalCase`/`EvalExpect` describe the input eval
file; `CaseResult`/`ProviderReport`/`BenchmarkReport` describe the output report.
"""

from __future__ import annotations

from pydantic import BaseModel


class EvalExpect(BaseModel):
    """Assertions checked against an agent's output for one case."""

    equals: dict[str, object] = {}    # output field == value
    contains: dict[str, str] = {}     # substring present in str(output field)


class EvalCase(BaseModel):
    """One eval: an input payload plus what to expect from the output."""

    name: str
    input: dict[str, object]          # validated against the agent's Input model
    expect: EvalExpect = EvalExpect()


class EvalSuite(BaseModel):
    """The full eval file for an agent."""

    cases: list[EvalCase]


class CaseResult(BaseModel):
    """Outcome of running one eval case."""

    name: str
    passed: bool                      # expect matched (False if the run errored)
    success: bool                     # agent ran without raising
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class ProviderReport(BaseModel):
    """Aggregate results for one provider across the whole suite."""

    provider: str
    case_count: int
    accuracy: float                   # passed / case_count (0.0 if empty)
    success_rate: float               # success / case_count
    latency_p50_ms: float
    latency_p95_ms: float
    mean_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    cases: list[CaseResult]


class BenchmarkReport(BaseModel):
    """One report per `lottie benchmark` run, across one or more providers."""

    agent: str
    providers: list[ProviderReport]
```

Create `src/lottie/benchmark/__init__.py`:

```python
from lottie.benchmark.schema import (
    BenchmarkReport,
    CaseResult,
    EvalCase,
    EvalExpect,
    EvalSuite,
    ProviderReport,
)

__all__ = [
    "BenchmarkReport",
    "CaseResult",
    "EvalCase",
    "EvalExpect",
    "EvalSuite",
    "ProviderReport",
]
```

- [ ] **Step 4: Run tests + type-check + lint**

Run: `uv run pytest src/lottie/benchmark/tests/test_schema.py -v` → all pass.
Run: `uv run mypy --strict src/lottie/benchmark` → `Success`.
Run: `uv run ruff check src/lottie/benchmark` → `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/benchmark/__init__.py src/lottie/benchmark/schema.py src/lottie/benchmark/tests/__init__.py src/lottie/benchmark/tests/test_schema.py
git commit -m "feat(benchmark): add benchmark schemas"
```

---

## Task 2: Runner — load_suite, scoring, percentile, run_suite

**Files:**
- Create: `src/lottie/benchmark/runner.py`, `src/lottie/benchmark/tests/test_runner.py`
- Modify: `src/lottie/benchmark/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/benchmark/tests/test_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import typer
from pydantic import BaseModel

from lottie.benchmark.runner import _passes, _percentile, load_suite, run_suite
from lottie.benchmark.schema import EvalCase, EvalExpect, EvalSuite
from lottie.core import BaseAgent
from lottie.llm import Message, MockLLMProvider


class _In(BaseModel):
    query: str


class _Out(BaseModel):
    result: str


class _EchoAgent(BaseAgent[BaseModel, BaseModel]):
    """Returns the LLM's content as `result` (mirrors the scaffold echo agent).

    Typed `BaseAgent[BaseModel, BaseModel]` to match `run_suite`'s parameter
    (BaseAgent generics are invariant); narrows `data` with `isinstance`.
    """

    def _execute(self, data: BaseModel) -> BaseModel:
        assert isinstance(data, _In)
        response = self.complete([Message(role="user", content=data.query)])
        return _Out(result=response.content)


def test_passes_equals_and_contains() -> None:
    dump = {"result": "pong"}
    assert _passes(dump, EvalExpect(equals={"result": "pong"})) is True
    assert _passes(dump, EvalExpect(equals={"result": "ping"})) is False
    assert _passes(dump, EvalExpect(contains={"result": "on"})) is True
    assert _passes(dump, EvalExpect(contains={"result": "zzz"})) is False
    assert _passes(dump, EvalExpect()) is True  # empty expect = presence check


def test_percentile_nearest_rank() -> None:
    assert _percentile([], 50) == 0.0
    assert _percentile([10.0], 50) == 10.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0


def test_run_suite_scores_and_aggregates() -> None:
    # Three cases: contains-pass, equals-fail, input-validation-fail.
    suite = EvalSuite(
        cases=[
            EvalCase(name="c1", input={"query": "hi"}, expect=EvalExpect(contains={"result": "ECHO"})),
            EvalCase(name="c2", input={"query": "hi"}, expect=EvalExpect(equals={"result": "nope"})),
            EvalCase(name="c3", input={"bad": "field"}),  # missing required `query`
        ]
    )
    # Mock returns "ECHO" for the two cases that actually run.
    agent = _EchoAgent(MockLLMProvider(["ECHO", "ECHO"]), enable_benchmarks=False)

    report = run_suite(agent, suite, _In)

    assert report.provider == "mock/mock-model"
    assert report.case_count == 3
    assert [c.passed for c in report.cases] == [True, False, False]
    assert report.cases[2].success is False
    assert report.cases[2].error is not None
    assert report.accuracy == pytest.approx(1 / 3)
    assert report.success_rate == pytest.approx(2 / 3)


def test_load_suite_reads_yaml(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agents" / "echo"
    agent_dir.mkdir(parents=True)
    (agent_dir / "evals.yaml").write_text(
        "cases:\n"
        "  - name: greets\n"
        "    input: {query: hello}\n"
        "    expect:\n"
        "      contains: {result: hello}\n",
        encoding="utf-8",
    )
    suite = load_suite(tmp_path, "echo")
    assert suite.cases[0].name == "greets"
    assert suite.cases[0].expect.contains == {"result": "hello"}


def test_load_suite_missing_file(tmp_path: Path) -> None:
    (tmp_path / "agents" / "echo").mkdir(parents=True)
    with pytest.raises(typer.BadParameter):
        load_suite(tmp_path, "echo")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/benchmark/tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.benchmark.runner'`.

- [ ] **Step 3: Write `runner.py`**

Create `src/lottie/benchmark/runner.py`:

```python
"""Load an eval suite, run it through an agent, and aggregate the results."""

from __future__ import annotations

import math
from pathlib import Path

import typer
import yaml
from pydantic import BaseModel, ValidationError

from lottie.benchmark.schema import (
    BenchmarkReport,
    CaseResult,
    EvalCase,
    EvalExpect,
    EvalSuite,
    ProviderReport,
)
from lottie.core import BaseAgent
from lottie.llm import build_provider
from lottie.project.discovery import load_agent_class, load_input_model


def load_suite(root: Path, name: str) -> EvalSuite:
    """Read and validate agents/<name>/evals.yaml."""
    path = root / "agents" / name / "evals.yaml"
    if not path.is_file():
        raise typer.BadParameter(f"no evals.yaml for agent '{name}'")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(f"cannot read {path}: {exc}") from exc
    try:
        return EvalSuite.model_validate(raw)
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid evals.yaml for '{name}': {exc}") from exc


def _passes(dump: dict[str, object], expect: EvalExpect) -> bool:
    """True when every `equals` and `contains` assertion holds against `dump`."""
    for field, value in expect.equals.items():
        if dump.get(field) != value:
            return False
    for field, substring in expect.contains.items():
        if substring not in str(dump.get(field)):
            return False
    return True


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of `values`; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def run_suite(
    agent: BaseAgent[BaseModel, BaseModel],
    suite: EvalSuite,
    input_model: type[BaseModel],
) -> ProviderReport:
    """Run every case through `agent`, score it, and aggregate a ProviderReport."""
    results: list[CaseResult] = []
    for case in suite.cases:
        results.append(_run_case(agent, case, input_model))
    return _aggregate(agent.provider or "unknown", results)


def _run_case(
    agent: BaseAgent[BaseModel, BaseModel],
    case: EvalCase,
    input_model: type[BaseModel],
) -> CaseResult:
    try:
        data = input_model.model_validate(case.input)
    except ValidationError as exc:
        return CaseResult(name=case.name, passed=False, success=False, error=repr(exc))
    try:
        output = agent.run(data)
    except Exception as exc:  # noqa: BLE001 — a bad case must not abort the suite
        return CaseResult(name=case.name, passed=False, success=False, error=repr(exc))
    m = agent.last_metrics
    return CaseResult(
        name=case.name,
        passed=_passes(output.model_dump(), case.expect),
        success=True,
        latency_ms=m.latency_ms if m else 0.0,
        input_tokens=m.input_tokens if m else 0,
        output_tokens=m.output_tokens if m else 0,
        cost_usd=m.cost_usd if m else 0.0,
    )


def _aggregate(provider: str, results: list[CaseResult]) -> ProviderReport:
    n = len(results)
    passed = sum(1 for r in results if r.passed)
    succeeded = [r for r in results if r.success]
    latencies = [r.latency_ms for r in succeeded]
    costs = [r.cost_usd for r in succeeded]
    return ProviderReport(
        provider=provider,
        case_count=n,
        accuracy=passed / n if n else 0.0,
        success_rate=len(succeeded) / n if n else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        mean_cost_usd=sum(costs) / len(costs) if costs else 0.0,
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        cases=results,
    )


def benchmark(root: Path, name: str, providers: list[str]) -> BenchmarkReport:
    """Run the agent's eval suite once per provider."""
    suite = load_suite(root, name)
    input_model = load_input_model(root, name)
    agent_cls = load_agent_class(root, name)
    reports = [
        run_suite(agent_cls(llm=build_provider(p), enable_benchmarks=False), suite, input_model)
        for p in providers
    ]
    return BenchmarkReport(agent=name, providers=reports)
```

- [ ] **Step 4: Extend `__init__.py`**

Add the runner exports to `src/lottie/benchmark/__init__.py` — add this import below the schema import block:

```python
from lottie.benchmark.runner import benchmark, load_suite, run_suite
```

and add `"benchmark"`, `"load_suite"`, `"run_suite"` to `__all__` (keep it sorted; run ruff to confirm order).

- [ ] **Step 5: Run tests + type-check + lint**

Run: `uv run pytest src/lottie/benchmark/tests/ -v` → all pass.
Run: `uv run mypy --strict src/lottie/benchmark` → `Success`. (If mypy flags `agent.run(data)` return type or `model_validate`, ensure `input_model: type[BaseModel]` and `output.model_dump()` are used as written — `model_dump()` returns `dict[str, Any]`; assigning to the `_passes` param typed `dict[str, object]` is accepted by mypy since `Any` is compatible.)
Run: `uv run ruff check src/lottie/benchmark` → `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/benchmark/runner.py src/lottie/benchmark/__init__.py src/lottie/benchmark/tests/test_runner.py
git commit -m "feat(benchmark): add eval-suite runner, scoring, and aggregation"
```

---

## Task 3: CLI command `lottie benchmark agent`

**Files:**
- Create: `src/lottie/cli/benchmark.py`, `src/lottie/cli/tests/test_benchmark.py`
- Modify: `src/lottie/cli/app.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/cli/tests/test_benchmark.py`:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold_with_evals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    (demo / "agents" / "echo" / "evals.yaml").write_text(
        "cases:\n"
        "  - name: greets\n"
        "    input: {query: hello}\n"
        "    expect:\n"
        "      contains: {result: hello}\n",
        encoding="utf-8",
    )
    return demo


def _fake_completion(model: str, messages: object, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )


def test_benchmark_runs_and_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold_with_evals(tmp_path, monkeypatch)
    monkeypatch.setattr("litellm.completion", _fake_completion)

    result = runner.invoke(app, ["benchmark", "agent", "echo"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output  # provider row

    report_path = demo / ".lottie" / "benchmarks" / "echo-report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["agent"] == "echo"
    assert len(data["providers"]) == 1
    assert data["providers"][0]["accuracy"] == 1.0  # output "hello world" contains "hello"


def test_benchmark_compare_two_providers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _scaffold_with_evals(tmp_path, monkeypatch)
    monkeypatch.setattr("litellm.completion", _fake_completion)

    result = runner.invoke(app, ["benchmark", "agent", "echo", "--compare"])
    assert result.exit_code == 0, result.output
    data = json.loads((demo / ".lottie" / "benchmarks" / "echo-report.json").read_text())
    # lottie.yaml default = anthropic/..., fallback = openai/gpt-4o
    assert len(data["providers"]) == 2


def test_benchmark_missing_evals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    result = runner.invoke(app, ["benchmark", "agent", "echo"])
    assert result.exit_code != 0
    assert "evals.yaml" in result.output


def test_benchmark_unknown_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    result = runner.invoke(app, ["benchmark", "agent", "nope"])
    assert result.exit_code != 0
    assert "nope" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/cli/tests/test_benchmark.py -v`
Expected: FAIL — `benchmark` is not a registered command.

- [ ] **Step 3: Create `cli/benchmark.py` and wire it**

Create `src/lottie/cli/benchmark.py`:

```python
"""`lottie benchmark agent <name>` — run an agent's eval suite and report."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lottie.benchmark.runner import benchmark
from lottie.benchmark.schema import ProviderReport
from lottie.project.config import find_project_root, load_agent_config, load_lottie_config

benchmark_app = typer.Typer(help="Benchmark agents against eval suites.", no_args_is_help=True)


@benchmark_app.command("agent")
def benchmark_agent(
    name: str,
    compare: Annotated[
        bool, typer.Option("--compare", help="Run across all configured providers.")
    ] = False,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Benchmark a single provider.")
    ] = None,
) -> None:
    """Run agents/<name>/evals.yaml and print + persist a report."""
    root = find_project_root()
    if not (root / "agents" / name / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    providers = _resolve_providers(root, name, compare, provider)
    report = benchmark(root, name, providers)

    _print_table(report.providers)
    out = root / ".lottie" / "benchmarks" / f"{name}-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Report written to {out}")


def _resolve_providers(root: Path, name: str, compare: bool, provider: str | None) -> list[str]:
    if provider is not None:
        return [provider]
    if compare:
        cfg = load_lottie_config(root)
        seen: list[str] = []
        for p in (cfg.providers.default, cfg.providers.fallback):
            if p and p not in seen:
                seen.append(p)
        return seen
    return [load_agent_config(root / "agents" / name).provider]


def _print_table(reports: list[ProviderReport]) -> None:
    table = Table(title="Benchmark")
    for col in ("provider", "cases", "accuracy", "success", "p50 ms", "p95 ms", "mean $", "in/out tok"):
        table.add_column(col)
    for r in reports:
        table.add_row(
            r.provider,
            str(r.case_count),
            f"{r.accuracy:.0%}",
            f"{r.success_rate:.0%}",
            f"{r.latency_p50_ms:.1f}",
            f"{r.latency_p95_ms:.1f}",
            f"{r.mean_cost_usd:.4f}",
            f"{r.total_input_tokens}/{r.total_output_tokens}",
        )
    Console().print(table)
```

Then edit `src/lottie/cli/app.py`: add `from lottie.cli.benchmark import benchmark_app` with the other CLI imports, and `app.add_typer(benchmark_app, name="benchmark")` with the other `add_typer`/`command` registrations.

- [ ] **Step 4: Run tests + type-check + lint**

Run: `uv run pytest src/lottie/cli/tests/test_benchmark.py -v` → all 4 pass.
Run: `uv run mypy --strict src/lottie/cli/benchmark.py src/lottie/cli/app.py` → `Success`.
Run: `uv run ruff check src/lottie/cli` → `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/cli/benchmark.py src/lottie/cli/app.py src/lottie/cli/tests/test_benchmark.py
git commit -m "feat(cli): add lottie benchmark agent command"
```

---

## Task 4: Full gate — suite, mypy, ruff

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: PASS — all prior tests plus the new benchmark tests, zero failures.

- [ ] **Step 2: Type-check the whole tree**

Run: `uv run mypy --strict src/lottie`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/lottie`
Expected: `All checks passed!`.

- [ ] **Step 4: Smoke-check the command help**

Run: `uv run python -c "from lottie.cli import app; print('ok')"`
Expected: prints `ok` (the new `benchmark_app` import wires cleanly, no cycle).

- [ ] **Step 5: Commit any gate fixes**

```bash
git add -A
git commit -m "chore: satisfy mypy --strict and ruff for benchmark"
```

(Skip this commit if Steps 1–4 needed no changes.)

---

## Notes for the implementer

- Run every command from the project dir via `uv run`.
- `run_suite` takes a constructed agent so the scoring/aggregation logic is
  tested with `MockLLMProvider` (no real LLM). Only `benchmark()` builds a real
  provider via `build_provider`, and that path is covered by the CLI tests that
  mock `litellm.completion`.
- Construct benchmark agents with `enable_benchmarks=False` so the suite does
  not append to the dev `.lottie/benchmarks/<name>.jsonl` per-run log.
- `agent.last_metrics` is set by `BaseAgent.run` (see `core/runnable.py`); read
  it right after `agent.run(data)` for the case's latency/tokens/cost.
- A case that fails input validation or raises during the run becomes
  `passed=False, success=False` with `error` set — the suite keeps going.
- Do NOT add LLM-as-judge scoring, `lottie report`/`audit`, or threshold gating
  — all out of scope (see spec).
