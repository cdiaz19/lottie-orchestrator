# `lottie benchmark` — Design

> Date: 2026-06-01
> Phase: 0 — Foundations (eval + benchmarking)
> Status: approved

## Goal

`lottie benchmark agent <name>` runs an agent against a YAML eval suite,
scores accuracy, aggregates the per-run metrics already captured by
`RunMetrics`, prints a table, and persists a JSON report. `--compare` runs the
same suite across the providers configured in `lottie.yaml`.

This is the eval layer from the testing matrix (real LLM when a user runs it),
and it backfills the accuracy/cost data the `list`/`inspect` columns were
deliberately omitted for.

## Scope decisions

- **Assertion-based accuracy** (not LLM-as-judge): each eval case carries an
  `expect` spec (field equality + substring `contains`) checked against the
  agent's output. `accuracy = passed / total`. Deterministic and unit-testable
  with `MockLLMProvider`.
- **Eval cases in `agents/<name>/evals.yaml`** — one file per agent, a list of
  cases. Colocated with the agent.
- **`--compare` uses `lottie.yaml` providers** (default + fallback). Report is a
  Rich table + a persisted JSON aggregate.
- **Logic split from CLI** — `src/lottie/benchmark/` holds the schema + runner
  (testable without a real LLM); `cli/benchmark.py` is a thin command wrapper.
- **A failing case never aborts the suite** — it is recorded as
  `success=False, passed=False` with the error, and the run continues.

## Module layout (`src/lottie/benchmark/`, logic) + `cli/benchmark.py` (command)

| File | Responsibility |
|---|---|
| `benchmark/schema.py` | Pydantic models (no logic) |
| `benchmark/runner.py` | `load_suite`, `run_suite`, `benchmark`, aggregation helpers |
| `benchmark/__init__.py` | Public exports |
| `cli/benchmark.py` | `benchmark` sub-Typer (`agent` command); wired into `cli/app.py` |
| `benchmark/tests/` | `test_schema.py`, `test_runner.py` |
| `cli/tests/test_benchmark.py` | End-to-end CLI tests (litellm mocked) |

## Schemas (`schema.py`)

```python
class EvalExpect(BaseModel):
    equals: dict[str, object] = {}    # output field == value
    contains: dict[str, str] = {}     # substring in str(output field)


class EvalCase(BaseModel):
    name: str
    input: dict[str, object]          # raw payload, validated against agent Input
    expect: EvalExpect = EvalExpect()


class EvalSuite(BaseModel):
    cases: list[EvalCase]


class CaseResult(BaseModel):
    name: str
    passed: bool                      # expect matched (False if the run errored)
    success: bool                     # agent ran without raising
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class ProviderReport(BaseModel):
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
    agent: str
    providers: list[ProviderReport]
```

`dict[str, object]` (not `Any`) keeps `mypy --strict` honest while allowing
YAML-parsed scalars (str/int/bool) as field values.

## Runner (`runner.py`)

### `load_suite(root: Path, name: str) -> EvalSuite`
Read `agents/<name>/evals.yaml` (via `yaml.safe_load`), `EvalSuite.model_validate`.
Missing file → `typer.BadParameter("no evals.yaml for agent '<name>'")`.
Unreadable / invalid → `typer.BadParameter` (mirrors `project.config._load_yaml_model`).

### `run_suite(agent, suite, input_model) -> ProviderReport`  (the testable core)
For each `case`:
1. `data = input_model.model_validate(case.input)` — a validation error →
   `CaseResult(passed=False, success=False, error=...)`, continue.
2. `output = agent.run(data)` inside `try/except` — any exception →
   `CaseResult(passed=False, success=False, error=repr(exc))`, continue.
3. On success, read `agent.last_metrics` for latency/tokens/cost; score with
   `_passes(output.model_dump(), case.expect)`; build `CaseResult`.

`_passes(dump, expect)`: every `k,v` in `expect.equals` satisfies
`dump.get(k) == v`, AND every `k,sub` in `expect.contains` satisfies
`sub in str(dump.get(k))`. Empty `expect` ⇒ passes (presence/no-error check).

Aggregate into `ProviderReport`: `provider = agent.provider or "unknown"`,
`accuracy = passed/n`, `success_rate = success/n`, latency p50/p95 via
`_percentile`, `mean_cost_usd`, token totals. Empty suite ⇒ zeros.

`_percentile(values: list[float], pct: float) -> float`: nearest-rank on the
sorted values (`values[ceil(pct/100 * n) - 1]`), `0.0` for an empty list.
Computed over successful cases' latencies (errored cases contribute `0.0`
metrics and are excluded from latency percentiles but counted in
`case_count`/`success_rate`).

### `benchmark(root, name, providers) -> BenchmarkReport`
`suite = load_suite(root, name)`; `input_model = load_input_model(root, name)`.
For each provider: `llm = build_provider(provider)`;
`agent = load_agent_class(root, name)(llm=llm, enable_benchmarks=False)`
(so the suite does not append to the dev `.jsonl`); `run_suite(...)`.
Return `BenchmarkReport(agent=name, providers=[...])`.

## CLI (`cli/benchmark.py`)

`benchmark_app = typer.Typer(...)`, registered `app.add_typer(benchmark_app, name="benchmark")`.

```
lottie benchmark agent <name> [--compare] [--provider TEXT]
```
- Resolve providers: `--provider` given ⇒ `[provider]`; elif `--compare` ⇒
  `[default, fallback]` from `lottie.yaml` (dedupe, drop `None`); else
  `[cfg.provider]` (the agent's configured provider).
- `report = benchmark(root, name, providers)`.
- Print a Rich table: one row per `ProviderReport` — provider, cases,
  accuracy %, success %, P50 ms, P95 ms, mean cost, in/out tokens.
- Persist `report.model_dump_json(indent=2)` to
  `.lottie/benchmarks/<name>-report.json` (create parent dirs). Echo the path.
- Unknown agent (`agents/<name>/agent.py` missing) → `typer.BadParameter`.

## Eval file format (`agents/<name>/evals.yaml`)

```yaml
cases:
  - name: greets
    input: {query: "hello"}
    expect:
      contains: {result: "hello"}
  - name: exact
    input: {query: "ping"}
    expect:
      equals: {result: "pong"}
```

## Testing

- `benchmark/tests/test_schema.py` — model defaults (empty `expect`, etc.).
- `benchmark/tests/test_runner.py` — **MockLLMProvider only, no real LLM**:
  - `run_suite` with a small concrete agent + `MockLLMProvider` replaying known
    responses: a passing `equals` case, a passing `contains` case, a failing
    case, and a case whose input fails validation → assert per-`CaseResult`
    `passed`/`success`/`error` and the aggregate `accuracy`/`success_rate`.
  - `_percentile` on known lists (incl. empty → 0.0) and p50/p95 on a multi-case
    report.
  - `load_suite` parses a written `evals.yaml`; missing file → `BadParameter`.
- `cli/tests/test_benchmark.py` — scaffold agent, write `evals.yaml`,
  `monkeypatch.setattr("litellm.completion", ...)` (as `test_run` does):
  - `benchmark agent <name>` exits 0, table mentions the provider/accuracy,
    `.lottie/benchmarks/<name>-report.json` exists and parses to a
    `BenchmarkReport` with one provider.
  - `--compare` produces two provider rows.
  - missing `evals.yaml` → non-zero exit + message.
  - unknown agent → non-zero exit.

## Out of scope

- LLM-as-judge / semantic scoring (assertion-based only).
- `lottie benchmark skill`, `lottie report performance` (trend charts),
  `lottie audit` — separate commands.
- Regression gating / thresholds (pass-fail CI on accuracy drop).
- Concurrency — cases run sequentially.
- Wiring benchmark scores back into `list`/`inspect` columns (future).
- `mypy --strict` + `ruff` stay clean; no `Any` without justification.
