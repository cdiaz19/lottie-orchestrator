"""Load an eval suite, run it through an agent, and aggregate the results."""

from __future__ import annotations

import math
from collections.abc import Mapping
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


def _passes(dump: Mapping[str, object], expect: EvalExpect) -> bool:
    """True when every `equals` and `contains` assertion holds against `dump`."""
    for field, value in expect.equals.items():
        if dump.get(field) != value:
            return False
    return all(
        substring in str(dump.get(field))
        for field, substring in expect.contains.items()
    )


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
    # Latency/cost only over succeeded runs (failed cases have no real timing);
    # token totals (below) over all results — failed cases contribute 0, so the sum stays honest.
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
