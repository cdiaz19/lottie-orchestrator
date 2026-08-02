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


class MetricDelta(BaseModel):
    """One metric compared between the baseline and learning arms."""

    metric: str
    baseline: float
    learning: float
    delta: float                      # learning - baseline
    pct_change: float | None = None   # None when the baseline is zero
    higher_is_better: bool


class LearningDeltaReport(BaseModel):
    """Does learning actually help? The evidence behind the default-on decision.

    Both arms run the SAME suite on the SAME provider with all memory WRITES disabled —
    only recall differs. Writes stay off in both arms deliberately: a benchmark that
    mutated the corpus it measures would not be reproducible, and the second run would
    silently report different numbers than the first.
    """

    agent: str
    provider: str
    namespace: str
    recalled_notes: int               # how much learned context the learning arm had
    baseline: ProviderReport
    learning: ProviderReport
    deltas: list[MetricDelta]
    verdict: str                      # improved | neutral | regressed (on accuracy)
