"""`lottie session` and `lottie run --session` over a real project layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from lottie.cli.app import app
from lottie.llm import MockLLMProvider

runner = CliRunner()

_SCHEMA = """
from __future__ import annotations
from pydantic import BaseModel


class StepperInput(BaseModel):
    task: str


class StepperOutput(BaseModel):
    answer: str
"""

_AGENT = """
from __future__ import annotations
from lottie.core import BaseAgent

from .schema import StepperInput, StepperOutput


class StepperAgent(BaseAgent[StepperInput, StepperOutput]):
    def _execute(self, data: StepperInput) -> StepperOutput:
        raw = self.session_progress.get("step", 0)
        step = raw if isinstance(raw, int) else 0
        self.save_progress(step=step + 1)
        return StepperOutput(answer=f"step {step + 1}")
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "lottie.yaml").write_text("name: demo\n")
    agent_dir = tmp_path / "agents" / "stepper"
    agent_dir.mkdir(parents=True)
    (tmp_path / "agents" / "__init__.py").write_text("")
    (agent_dir / "__init__.py").write_text("")
    (agent_dir / "agent.py").write_text(_AGENT)
    (agent_dir / "schema.py").write_text(_SCHEMA)
    (agent_dir / "config.yaml").write_text(yaml.safe_dump({"provider": "mock/sim"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "lottie.cli.run.build_provider", lambda model: MockLLMProvider(responses=["ok"] * 10)
    )
    return tmp_path


def _run(*args: str) -> object:
    return runner.invoke(app, ["run", "stepper", "--input", '{"task": "t"}', *args])


class TestRunWithSession:
    def test_a_run_without_a_session_creates_nothing(self, project: Path) -> None:
        _run()
        assert not (project / ".lottie" / "sessions").exists()

    def test_a_session_run_persists_progress(self, project: Path) -> None:
        result = _run("--session", "s1")
        assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
        assert (project / ".lottie" / "sessions" / "s1" / "state.json").is_file()

    def test_resuming_advances_the_progress(self, project: Path) -> None:
        _run("--session", "s1")
        result = _run("--session", "s1")
        assert "step 2" in result.output  # type: ignore[attr-defined]

    def test_separate_sessions_do_not_interfere(self, project: Path) -> None:
        _run("--session", "a")
        _run("--session", "a")
        result = _run("--session", "b")
        assert "step 1" in result.output  # type: ignore[attr-defined]

    def test_a_traversing_session_id_is_refused(self, project: Path) -> None:
        result = _run("--session", "../../etc")
        assert result.exit_code != 0  # type: ignore[attr-defined]


class TestSessionCli:
    def test_list_is_empty_initially(self, project: Path) -> None:
        assert "no sessions" in runner.invoke(app, ["session", "list"]).output

    def test_list_shows_agent_runs_and_progress(self, project: Path) -> None:
        _run("--session", "s1")
        out = runner.invoke(app, ["session", "list"]).output
        assert "s1" in out and "agent=StepperAgent" in out and "runs=1" in out

    def test_show_prints_the_state(self, project: Path) -> None:
        _run("--session", "s1")
        payload = json.loads(runner.invoke(app, ["session", "show", "s1"]).output)
        assert payload["progress"] == {"step": 1}
        assert len(payload["runs"]) == 1

    def test_show_history_is_hash_only(self, project: Path) -> None:
        _run("--session", "s1")
        payload = json.loads(runner.invoke(app, ["session", "show", "s1"]).output)
        assert payload["runs"][0]["input_sha256"] is not None
        assert "t" not in str(payload["runs"][0].get("input", ""))

    def test_show_on_a_missing_session_fails(self, project: Path) -> None:
        assert runner.invoke(app, ["session", "show", "ghost"]).exit_code != 0

    def test_delete_removes_it(self, project: Path) -> None:
        _run("--session", "s1")
        runner.invoke(app, ["session", "delete", "s1"])
        assert "no sessions" in runner.invoke(app, ["session", "list"]).output

    def test_delete_reports_a_missing_session(self, project: Path) -> None:
        assert "no session named" in runner.invoke(app, ["session", "delete", "ghost"]).output

    def test_traversal_is_refused_by_the_cli(self, project: Path) -> None:
        assert runner.invoke(app, ["session", "show", "../../etc"]).exit_code != 0
