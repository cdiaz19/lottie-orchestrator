from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


def _chat_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project with a chat-capable `echo` agent and the default `hello` agent."""
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    # Make echo chat-capable: query -> Input.query, Output.result -> content.
    cfg = demo / "agents" / "echo" / "config.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8")
        + "chat:\n  input_field: query\n  output_field: result\n",
        encoding="utf-8",
    )
    return demo


def test_models_lists_only_chat_capable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lottie.serve.openai_app import build_openai_app

    demo = _chat_project(tmp_path, monkeypatch)
    client = TestClient(build_openai_app(demo))
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert "echo" in ids          # chat block present
    assert "hello" not in ids     # default agent has no chat block
    assert all(m["object"] == "model" and m["owned_by"] == "lottie" for m in body["data"])


def _mock_provider(monkeypatch: pytest.MonkeyPatch, response: str = "hello world") -> None:
    from lottie.llm import MockLLMProvider

    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider([response]),
    )


def test_chat_completion_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lottie.serve.openai_app import build_openai_app

    demo = _chat_project(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    client = TestClient(build_openai_app(demo))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "echo"
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] >= 0
    assert "lottie" in body
