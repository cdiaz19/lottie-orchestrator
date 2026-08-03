from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def test_doctor_passes_with_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_doctor_fails_on_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_doctor_outside_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = runner.invoke(app, ["doctor"])
    # Outside a project: not fatal, just reported.
    assert result.exit_code == 0, result.output
    assert "not in a Lottie project" in result.output


def test_doctor_warns_when_http_hardening_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # default project has an openai fallback
    monkeypatch.delenv("LOTTIE_API_KEYS", raising=False)
    monkeypatch.delenv("LOTTIE_RATE_LIMIT_PER_MIN", raising=False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "LOTTIE_API_KEYS unset" in result.output
    assert "LOTTIE_RATE_LIMIT_PER_MIN unset" in result.output


def test_doctor_no_hardening_warning_when_keys_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    monkeypatch.chdir(tmp_path / "demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LOTTIE_API_KEYS", "sk-live")
    result = runner.invoke(app, ["doctor"])
    assert "LOTTIE_API_KEYS unset" not in result.output


# --- V2 self-learning advisories (S6) ----------------------------------------


def _project_with_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg: dict[str, object]
) -> None:
    """A real scaffolded project plus one agent carrying `cfg`."""
    import yaml

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    root = tmp_path / "demo"
    monkeypatch.chdir(root)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agent_dir = root / "agents" / "probe"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "agent.py").write_text("# stub\n")
    (agent_dir / "config.yaml").write_text(
        yaml.safe_dump({"provider": "anthropic/claude-sonnet-4-6", **cfg})
    )


def test_warns_when_reflect_is_unbounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project_with_agent(
        tmp_path, monkeypatch, {"memory": {"enabled": True, "reflect": {"enabled": True}}}
    )
    assert "unbounded" in runner.invoke(app, ["doctor"]).output


def test_no_warning_when_reflect_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_agent(
        tmp_path,
        monkeypatch,
        {"max_run_tokens": 5000, "memory": {"enabled": True, "reflect": {"enabled": True}}},
    )
    assert "unbounded" not in runner.invoke(app, ["doctor"]).output


def test_warns_when_trajectory_is_on_but_memory_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_agent(
        tmp_path, monkeypatch, {"memory": {"enabled": False, "trajectory": {"enabled": True}}}
    )
    assert "no trajectories will be written" in runner.invoke(app, ["doctor"]).output


def test_warns_when_trajectories_are_never_consulted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Writing a corpus nothing ever reads is pure cost.
    _project_with_agent(
        tmp_path, monkeypatch, {"memory": {"enabled": True, "trajectory": {"enabled": True}}}
    )
    assert "never consulted" in runner.invoke(app, ["doctor"]).output


def test_no_warning_when_trajectories_feed_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_agent(
        tmp_path,
        monkeypatch,
        {
            "memory": {
                "enabled": True,
                "trajectory": {"enabled": True},
                "recall": {"enabled": True},
            }
        },
    )
    assert "never consulted" not in runner.invoke(app, ["doctor"]).output


def test_clean_project_has_no_learning_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project_with_agent(tmp_path, monkeypatch, {})
    out = runner.invoke(app, ["doctor"]).output
    assert "unbounded" not in out and "never consulted" not in out
