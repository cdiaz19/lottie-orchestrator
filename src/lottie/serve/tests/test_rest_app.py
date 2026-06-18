from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project with a generated `echo` agent (Input {query}, Output {result})."""
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def test_list_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.serve.rest_app import build_rest_app

    demo = _project(tmp_path, monkeypatch)
    client = TestClient(build_rest_app(demo))
    resp = client.get("/v1/agents")
    assert resp.status_code == 200
    names = {a["name"] for a in resp.json()["agents"]}
    assert {"echo", "hello"} <= names
    assert all("provider" in a for a in resp.json()["agents"])


def test_agent_detail_has_input_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lottie.serve.rest_app import build_rest_app

    demo = _project(tmp_path, monkeypatch)
    client = TestClient(build_rest_app(demo))
    resp = client.get("/v1/agents/echo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "echo"
    assert "query" in body["input_schema"]["properties"]  # echo Input has a `query` field


def test_agent_detail_unknown_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lottie.serve.rest_app import build_rest_app

    demo = _project(tmp_path, monkeypatch)
    client = TestClient(build_rest_app(demo))
    resp = client.get("/v1/agents/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found"
