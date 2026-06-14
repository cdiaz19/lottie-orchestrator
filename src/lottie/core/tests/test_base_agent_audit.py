from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.governance.audit import SqliteAuditLogger, hash_model
from lottie.llm import MockLLMProvider


class _In(BaseModel):
    q: str


class _Out(BaseModel):
    a: str


class _Echo(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(a=f"echo:{data.q}")


class _Boom(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        raise RuntimeError("boom")


class _Outer(BaseAgent[_In, _Out]):
    """Runs an inner agent inside its own _execute (mimics a mesh top-level -> worker)."""

    def __init__(self, llm: object, inner: _Echo, audit: object) -> None:
        super().__init__(llm, audit=audit)  # type: ignore[arg-type]
        self._inner = inner

    def _execute(self, data: _In) -> _Out:
        return self._inner.run(data)


def _llm() -> MockLLMProvider:
    return MockLLMProvider(["unused"])


def test_successful_run_writes_one_root_record(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Echo(_llm(), audit=logger)
    agent.run(_In(q="hi"))
    rows = logger.query()
    assert len(rows) == 1
    r = rows[0]
    assert r.status == "ok" and r.root is True
    assert r.input_sha256 == hash_model(_In(q="hi"))
    assert r.output_sha256 == hash_model(_Out(a="echo:hi"))
    assert r.latency_ms >= 0.0


def test_failed_run_writes_error_record_and_reraises(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    agent = _Boom(_llm(), audit=logger)
    with pytest.raises(RuntimeError):
        agent.run(_In(q="hi"))
    rows = logger.query()
    assert len(rows) == 1
    assert rows[0].status == "error" and rows[0].output_sha256 is None and rows[0].error


def test_nested_run_flags_inner_non_root(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    inner = _Echo(_llm(), audit=logger)
    outer = _Outer(_llm(), inner, logger)
    outer.run(_In(q="hi"))
    rows = logger.query()
    statuses = {(r.agent, r.root) for r in rows}
    assert ("_Echo", False) in statuses   # inner ran nested
    assert ("_Outer", True) in statuses   # outer is top-level


def test_audit_failure_never_breaks_a_successful_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_model: object) -> str:
        raise ValueError("hash exploded")

    monkeypatch.setattr("lottie.core.base_agent.hash_model", _boom)
    agent = _Echo(_llm(), audit=SqliteAuditLogger(tmp_path))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = agent.run(_In(q="hi"))  # must NOT raise
    assert out.a == "echo:hi"


def test_audit_failure_does_not_mask_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_model: object) -> str:
        raise ValueError("hash exploded")

    monkeypatch.setattr("lottie.core.base_agent.hash_model", _boom)
    agent = _Boom(_llm(), audit=SqliteAuditLogger(tmp_path))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="boom"):  # original error, not "hash exploded"
            agent.run(_In(q="hi"))
