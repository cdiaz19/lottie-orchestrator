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
            EvalCase(
                name="c1",
                input={"query": "hi"},
                expect=EvalExpect(contains={"result": "ECHO"}),
            ),
            EvalCase(
                name="c2",
                input={"query": "hi"},
                expect=EvalExpect(equals={"result": "nope"}),
            ),
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


def test_run_suite_empty_suite_zeros() -> None:
    agent = _EchoAgent(MockLLMProvider(["unused"]), enable_benchmarks=False)
    report = run_suite(agent, EvalSuite(cases=[]), _In)
    assert report.case_count == 0
    assert report.accuracy == 0.0
    assert report.success_rate == 0.0
    assert report.latency_p50_ms == 0.0
    assert report.mean_cost_usd == 0.0
    assert report.cases == []


def test_load_suite_invalid_yaml(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agents" / "echo"
    agent_dir.mkdir(parents=True)
    # `cases` must be a list; a string fails EvalSuite validation.
    (agent_dir / "evals.yaml").write_text("cases: not-a-list\n", encoding="utf-8")
    with pytest.raises(typer.BadParameter):
        load_suite(tmp_path, "echo")
