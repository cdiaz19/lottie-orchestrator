from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def test_http_app_serves_both_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lottie.serve.http_app import build_http_app

    demo = _project(tmp_path, monkeypatch)
    client = TestClient(build_http_app(demo))
    # OpenAI route group
    assert client.get("/v1/models").status_code == 200
    # REST route group
    rest = client.get("/v1/agents")
    assert rest.status_code == 200
    assert {a["name"] for a in rest.json()["agents"]} >= {"echo", "hello"}
