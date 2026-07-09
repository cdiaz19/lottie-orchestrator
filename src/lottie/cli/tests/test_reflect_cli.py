from pathlib import Path

import pytest
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


def test_reflect_unknown_agent_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_project(tmp_path))
    result = runner.invoke(app, ["reflect", "nope"])
    assert result.exit_code != 0


def test_reflect_runs_consolidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
