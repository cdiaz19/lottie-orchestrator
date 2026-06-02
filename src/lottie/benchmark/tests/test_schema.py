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
