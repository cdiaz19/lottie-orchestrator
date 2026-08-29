"""`lottie plan` — inspect recorded mesh runs (E6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli.app import app
from lottie.mesh.plan import Plan, PlanStep, hash_task, save_plan

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "lottie.yaml").write_text("project: demo\nproviders:\n  default: mock/sim\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _plan(*steps: list[str], task: str = "write it") -> Plan:
    return Plan(
        task_sha256=hash_task(task),
        steps=[PlanStep(step=i, workers=w) for i, w in enumerate(steps)],
    )


class TestList:
    def test_reports_nothing_when_no_plans_exist(self, project: Path) -> None:
        assert "no recorded plans" in runner.invoke(app, ["plan", "list", "assistant"]).output

    def test_lists_recorded_threads_with_their_step_count(self, project: Path) -> None:
        save_plan(project, "assistant", "run-1", _plan(["draft"], ["review"]))
        out = runner.invoke(app, ["plan", "list", "assistant"]).output
        assert "run-1" in out and "2 step(s)" in out

    def test_plans_are_scoped_per_agent(self, project: Path) -> None:
        save_plan(project, "assistant", "run-1", _plan(["draft"]))
        assert "no recorded plans" in runner.invoke(app, ["plan", "list", "other"]).output


class TestShow:
    def test_renders_the_recorded_steps(self, project: Path) -> None:
        save_plan(project, "assistant", "run-1", _plan(["draft"], ["review"]))
        out = runner.invoke(app, ["plan", "show", "assistant", "run-1"]).output
        assert "draft" in out and "review" in out

    def test_a_fan_out_is_labelled_parallel(self, project: Path) -> None:
        save_plan(project, "assistant", "run-1", _plan(["draft", "factcheck"]))
        assert "parallel" in runner.invoke(app, ["plan", "show", "assistant", "run-1"]).output

    def test_a_single_worker_step_is_labelled_sequential(self, project: Path) -> None:
        save_plan(project, "assistant", "run-1", _plan(["draft"]))
        assert "sequential" in runner.invoke(app, ["plan", "show", "assistant", "run-1"]).output

    def test_it_says_the_task_is_hash_only(self, project: Path) -> None:
        # An operator reasonably wonders where the task text went; say so rather than
        # leaving them to guess it was lost.
        save_plan(project, "assistant", "run-1", _plan(["draft"], task="SENSITIVE"))
        out = runner.invoke(app, ["plan", "show", "assistant", "run-1"]).output
        assert "hash only" in out and "SENSITIVE" not in out

    def test_a_missing_plan_is_a_clear_error(self, project: Path) -> None:
        result = runner.invoke(app, ["plan", "show", "assistant", "never-ran"])
        assert result.exit_code != 0 and "no recorded plan" in result.output

    def test_a_traversing_thread_id_is_refused(self, project: Path) -> None:
        assert runner.invoke(app, ["plan", "show", "assistant", "../../etc"]).exit_code != 0
