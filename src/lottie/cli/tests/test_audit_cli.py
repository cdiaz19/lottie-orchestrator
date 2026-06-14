from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.schema import AuditRecord

runner = CliRunner()


def _seed(root: Path, agent: str = "echo") -> None:
    SqliteAuditLogger(root).log(
        AuditRecord(
            ts="2026-06-13T10:00:00+00:00", agent=agent, provider="mock/x", status="ok",
            root=True, input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=1,
            output_tokens=2, cost_usd=0.0, latency_ms=1.0, error=None,
        )
    )


def _init_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    return demo


def test_audit_lists_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _init_project(tmp_path, monkeypatch)
    _seed(demo)
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "echo" in result.output


def test_audit_empty_is_friendly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init_project(tmp_path, monkeypatch)
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "no audit" in result.output.lower()


def test_audit_filters_by_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _init_project(tmp_path, monkeypatch)
    _seed(demo, agent="echo")
    _seed(demo, agent="other")
    result = runner.invoke(app, ["audit", "--agent", "other"])
    assert result.exit_code == 0
    assert "other" in result.output and "echo" not in result.output
