"""S4 HTTP hardening — API-key auth, rate limiting, pagination on the shared app."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from lottie.serve.pagination import MAX_LIMIT, page_bounds  # noqa: E402

runner = CliRunner()


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from lottie.serve.http_app import build_http_app

    demo = _project(tmp_path, monkeypatch)
    return TestClient(build_http_app(demo))


# --- pagination unit -------------------------------------------------------

class TestPageBounds:
    def test_default_limit_is_none_return_all(self) -> None:
        assert page_bounds({}) == (None, 0)  # no silent truncation

    def test_parses_and_clamps(self) -> None:
        assert page_bounds({"limit": "5", "offset": "2"}) == (5, 2)
        assert page_bounds({"limit": "0"}) == (1, 0)  # clamp up to 1
        assert page_bounds({"limit": "9999"}) == (MAX_LIMIT, 0)  # clamp to ceiling
        assert page_bounds({"offset": "-3"}) == (None, 0)  # limit absent -> all; offset clamped

    def test_garbage_limit_returns_all(self) -> None:
        assert page_bounds({"limit": "abc", "offset": "x"}) == (None, 0)


# --- pagination over HTTP --------------------------------------------------

def test_agents_pagination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)  # scaffold has 'echo' + 'hello'
    full = client.get("/v1/agents").json()["agents"]
    assert len(full) >= 2
    one = client.get("/v1/agents?limit=1").json()["agents"]
    assert len(one) == 1
    page2 = client.get("/v1/agents?limit=1&offset=1").json()["agents"]
    assert len(page2) == 1 and page2[0]["name"] != one[0]["name"]
    assert client.get("/v1/agents?offset=999").json()["agents"] == []


# --- auth ------------------------------------------------------------------

def test_auth_open_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOTTIE_API_KEYS", raising=False)
    client = _client(tmp_path, monkeypatch)
    assert client.get("/v1/agents").status_code == 200


def test_auth_required_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOTTIE_API_KEYS", "sk-alpha, sk-beta")
    client = _client(tmp_path, monkeypatch)
    # missing
    r = client.get("/v1/agents")
    assert r.status_code == 401
    assert "sk-alpha" not in r.text  # never echo a valid key
    # valid Bearer
    assert client.get("/v1/agents", headers={"Authorization": "Bearer sk-alpha"}).status_code == 200
    # valid X-API-Key
    assert client.get("/v1/agents", headers={"X-API-Key": "sk-beta"}).status_code == 200
    # wrong
    assert client.get("/v1/agents", headers={"Authorization": "Bearer nope"}).status_code == 401


# --- rate limit ------------------------------------------------------------

def test_rate_limit_unset_unlimited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOTTIE_RATE_LIMIT_PER_MIN", raising=False)
    client = _client(tmp_path, monkeypatch)
    for _ in range(10):
        assert client.get("/v1/agents").status_code == 200


def test_rate_limit_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOTTIE_RATE_LIMIT_PER_MIN", "2")
    client = _client(tmp_path, monkeypatch)
    assert client.get("/v1/agents").status_code == 200
    assert client.get("/v1/agents").status_code == 200
    r = client.get("/v1/agents")  # bucket empty
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"


def test_rate_limit_separate_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOTTIE_RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("LOTTIE_API_KEYS", "sk-a,sk-b")
    client = _client(tmp_path, monkeypatch)
    ha = {"Authorization": "Bearer sk-a"}
    hb = {"Authorization": "Bearer sk-b"}
    assert client.get("/v1/agents", headers=ha).status_code == 200
    assert client.get("/v1/agents", headers=ha).status_code == 429  # a exhausted
    assert client.get("/v1/agents", headers=hb).status_code == 200  # b independent
