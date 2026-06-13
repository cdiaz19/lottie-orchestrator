from __future__ import annotations

from pathlib import Path

import typer
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _init_project(tmp_path: Path, monkeypatch) -> Path:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    return demo


def test_mesh_resume_help_lists_command(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(app, ["mesh", "--help"])
    assert result.exit_code == 0
    assert "resume" in result.output
    assert "history" in result.output


def test_mesh_resume_rejects_bad_decision(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        ["mesh", "resume", "--agent", "gate", "--thread-id", "t1", "--decision", "bogus"],
    )
    assert result.exit_code != 0
    assert "approve" in result.output.lower() or "reject" in result.output.lower()


def test_mesh_resume_approve_reports_no_checkpoint(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # No prior run in THIS process → resume can't find the thread; should error
    # helpfully (here: the `gate` agent does not exist in a fresh project, which
    # is itself a clean "agent not found" error), not crash.
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        ["mesh", "resume", "--agent", "gate", "--thread-id", "missing", "--decision", "approve"],
    )
    # non-zero exit, handled (no unhandled traceback)
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, (typer.Exit, SystemExit))


def test_mesh_resume_rejects_non_json_input(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        ["mesh", "resume", "--agent", "gate", "--thread-id", "t1", "--input", "not json"],
    )
    assert result.exit_code != 0
    assert "json" in result.output.lower()


def test_mesh_resume_rejects_non_object_input(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        ["mesh", "resume", "--agent", "gate", "--thread-id", "t1", "--input", "[1, 2]"],
    )
    assert result.exit_code != 0
    assert "object" in result.output.lower()


def test_mesh_resume_accepts_valid_object_input(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Valid --input parses cleanly; resume then fails because `gate` is absent in a
    # fresh project — proving the JSON object path ran without a parse error.
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        ["mesh", "resume", "--agent", "gate", "--thread-id", "t1", "--input", '{"k": "v"}'],
    )
    assert result.exit_code != 0
    assert "json" not in result.output.lower()  # not a parse failure
    assert result.exception is None or isinstance(result.exception, (typer.Exit, SystemExit))


def test_mesh_history_is_clean_and_informative(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(app, ["mesh", "history", "--agent", "gate", "--thread-id", "t1"])
    # Clean exit, no unhandled traceback; output mentions the cross-process limit.
    assert result.exception is None or isinstance(result.exception, (typer.Exit, SystemExit))
    assert "sqlite" in result.output.lower() or "in-memory" in result.output.lower()
