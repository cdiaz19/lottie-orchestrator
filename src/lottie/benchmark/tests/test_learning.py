"""Learning-delta measurement: arm isolation, metric maths, and the verdict gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lottie.benchmark.learning import (
    _arm_config,
    _delta,
    _verdict,
    learning_delta,
    write_delta_report,
)
from lottie.benchmark.schema import CaseResult, LearningDeltaReport, ProviderReport
from lottie.llm import MockLLMProvider
from lottie.memory.schema import MemoryOrigin, MemoryRecord, MemoryTier
from lottie.memory.store import SqliteMemoryClient
from lottie.project.config import AgentConfig

SCHEMA_SRC = """
from __future__ import annotations
from pydantic import BaseModel


class ProbeInput(BaseModel):
    query: str


class ProbeOutput(BaseModel):
    result: str
"""

AGENT_SRC = """
from __future__ import annotations
from lottie.core import BaseAgent
from lottie.llm import Message

from .schema import ProbeInput, ProbeOutput


class ProbeAgent(BaseAgent[ProbeInput, ProbeOutput]):
    def _execute(self, data: ProbeInput) -> ProbeOutput:
        response = self.complete([Message(role="user", content=data.query)])
        return ProbeOutput(result=response.content)
"""


def _report(**kw: object) -> ProviderReport:
    base: dict[str, object] = {
        "provider": "mock/sim",
        "case_count": 2,
        "accuracy": 0.5,
        "success_rate": 1.0,
        "latency_p50_ms": 10.0,
        "latency_p95_ms": 20.0,
        "mean_cost_usd": 0.01,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "cases": [],
    }
    base.update(kw)
    return ProviderReport.model_validate(base)


class TestArmIsolation:
    """Both arms must disable every memory WRITE — only recall differs."""

    def _cfg(self) -> AgentConfig:
        return AgentConfig.model_validate(
            {
                "provider": "mock/sim",
                "memory": {
                    "enabled": True,
                    "recall": {"enabled": True},
                    "reflect": {"enabled": True},
                    "trajectory": {"enabled": True},
                },
            }
        )

    def test_baseline_disables_recall(self) -> None:
        assert _arm_config(self._cfg(), recall=False).memory.recall.enabled is False

    def test_learning_enables_recall(self) -> None:
        assert _arm_config(self._cfg(), recall=True).memory.recall.enabled is True

    @pytest.mark.parametrize("recall", [True, False])
    def test_reflection_is_off_in_both_arms(self, recall: bool) -> None:
        # A benchmark that wrote lessons would mutate the corpus it measures.
        assert _arm_config(self._cfg(), recall=recall).memory.reflect.enabled is False

    @pytest.mark.parametrize("recall", [True, False])
    def test_trajectory_is_off_in_both_arms(self, recall: bool) -> None:
        assert _arm_config(self._cfg(), recall=recall).memory.trajectory.enabled is False

    def test_the_original_config_is_not_mutated(self) -> None:
        cfg = self._cfg()
        _arm_config(cfg, recall=False)
        assert cfg.memory.reflect.enabled is True


class TestDeltaMaths:
    def test_delta_is_learning_minus_baseline(self) -> None:
        d = _delta("accuracy", _report(accuracy=0.5), _report(accuracy=0.8), True)
        assert d.delta == pytest.approx(0.3)

    def test_pct_change(self) -> None:
        d = _delta("accuracy", _report(accuracy=0.5), _report(accuracy=0.75), True)
        assert d.pct_change == pytest.approx(50.0)

    def test_pct_change_is_none_on_a_zero_baseline(self) -> None:
        # Guards a ZeroDivisionError on a suite where the baseline scored nothing.
        d = _delta("accuracy", _report(accuracy=0.0), _report(accuracy=0.5), True)
        assert d.pct_change is None

    def test_direction_is_recorded_per_metric(self) -> None:
        cost = _delta("mean_cost_usd", _report(), _report(), False)
        acc = _delta("accuracy", _report(), _report(), True)
        assert cost.higher_is_better is False and acc.higher_is_better is True

    def test_integer_metrics_are_handled(self) -> None:
        d = _delta(
            "total_input_tokens", _report(total_input_tokens=100),
            _report(total_input_tokens=150), False
        )
        assert d.delta == pytest.approx(50.0)


class TestVerdict:
    def test_positive_accuracy_delta_is_improved(self) -> None:
        assert _verdict(0.2) == "improved"

    def test_zero_is_neutral(self) -> None:
        assert _verdict(0.0) == "neutral"

    def test_negative_is_regressed(self) -> None:
        assert _verdict(-0.1) == "regressed"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "lottie.yaml").write_text("name: demo\n")
    agent_dir = tmp_path / "agents" / "probe"
    agent_dir.mkdir(parents=True)
    (tmp_path / "agents" / "__init__.py").write_text("")
    (agent_dir / "__init__.py").write_text("")
    (agent_dir / "agent.py").write_text(AGENT_SRC)
    (agent_dir / "schema.py").write_text(SCHEMA_SRC)
    (agent_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "provider": "mock/sim",
                "memory": {
                    "enabled": True,
                    "backend": "sqlite",
                    "path": ".lottie/memory.db",
                    "namespace": "probe",
                    "recall": {"enabled": False},
                },
            }
        )
    )
    (agent_dir / "evals.yaml").write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "name": "a",
                        "input": {"query": "q1"},
                        "expect": {"contains": {"result": "x"}},
                    },
                    {
                        "name": "b",
                        "input": {"query": "q2"},
                        "expect": {"contains": {"result": "x"}},
                    },
                ]
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "lottie.benchmark.learning.build_provider",
        lambda model: MockLLMProvider(responses=["x", "x", "x", "x"]),
    )
    return tmp_path


class TestEndToEnd:
    def test_runs_both_arms(self, project: Path) -> None:
        report = learning_delta(project, "probe", "mock/sim")
        assert report.baseline.case_count == 2
        assert report.learning.case_count == 2

    def test_reports_every_metric(self, project: Path) -> None:
        report = learning_delta(project, "probe", "mock/sim")
        assert {d.metric for d in report.deltas} == {
            "accuracy",
            "success_rate",
            "mean_cost_usd",
            "latency_p50_ms",
            "latency_p95_ms",
            "total_input_tokens",
            "total_output_tokens",
        }

    def test_records_the_namespace(self, project: Path) -> None:
        assert learning_delta(project, "probe", "mock/sim").namespace == "probe"

    def test_recalled_notes_is_zero_on_an_empty_store(self, project: Path) -> None:
        # A 'neutral' verdict with 0 notes means the experiment never ran.
        assert learning_delta(project, "probe", "mock/sim").recalled_notes == 0

    def test_recalled_notes_counts_semantic_notes(self, project: Path) -> None:
        client = SqliteMemoryClient(project / ".lottie" / "memory.db")
        for i in range(3):
            client.remember(
                MemoryRecord(
                    content=f"lesson {i}",
                    tier=MemoryTier.SEMANTIC,
                    namespace="probe",
                    origin=MemoryOrigin.REFLECTION,
                )
            )
        assert learning_delta(project, "probe", "mock/sim").recalled_notes == 3

    def test_benchmarking_writes_no_memory(self, project: Path) -> None:
        """The measurement must not mutate the corpus it measures."""
        learning_delta(project, "probe", "mock/sim")
        db = project / ".lottie" / "memory.db"
        if not db.exists():
            return  # nothing written at all is the strongest form of the guarantee
        client = SqliteMemoryClient(db)
        from lottie.memory.schema import MemoryQuery

        hits = client.recall(MemoryQuery(text="", namespace="probe", limit=100)).hits
        assert hits == []

    def test_report_is_reproducible(self, project: Path) -> None:
        """Running the benchmark twice must not change what it measures.

        Only the state-dependent metrics are compared. Latency (and the cost derived
        from it) is wall-clock and jitters between runs regardless of learned state —
        asserting on it would make this a flaky test of the machine, not of isolation.
        """
        stable = {"accuracy", "success_rate", "total_input_tokens", "total_output_tokens"}
        first = learning_delta(project, "probe", "mock/sim")
        second = learning_delta(project, "probe", "mock/sim")
        assert first.verdict == second.verdict
        assert first.recalled_notes == second.recalled_notes
        assert {d.metric: d.delta for d in first.deltas if d.metric in stable} == {
            d.metric: d.delta for d in second.deltas if d.metric in stable
        }


class TestWriteReport:
    def _report(self) -> LearningDeltaReport:
        return LearningDeltaReport(
            agent="probe",
            provider="mock/sim",
            namespace="probe",
            recalled_notes=2,
            baseline=_report(cases=[CaseResult(name="a", passed=True, success=True)]),
            learning=_report(),
            deltas=[_delta("accuracy", _report(), _report(), True)],
            verdict="neutral",
        )

    def test_writes_a_machine_readable_file(self, tmp_path: Path) -> None:
        out = write_delta_report(tmp_path, self._report())
        assert out == tmp_path / ".lottie" / "benchmarks" / "probe-learning-delta.json"

    def test_the_file_is_valid_json(self, tmp_path: Path) -> None:
        out = write_delta_report(tmp_path, self._report())
        payload = json.loads(out.read_text())
        assert payload["verdict"] == "neutral"
        assert payload["recalled_notes"] == 2

    def test_creates_the_directory(self, tmp_path: Path) -> None:
        assert write_delta_report(tmp_path, self._report()).parent.is_dir()


def test_an_unreadable_store_does_not_fail_the_benchmark(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counting recalled notes is diagnostics, not the measurement.

    If the store cannot be read the benchmark must still produce a verdict — losing the
    note count is a worse-but-usable report, whereas aborting loses the whole run.
    """

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("store on fire")

    monkeypatch.setattr("lottie.benchmark.learning.build_memory_client", _boom)
    report = learning_delta(project, "probe", "mock/sim")
    assert report.recalled_notes == 0
    assert report.verdict in {"improved", "neutral", "regressed"}
