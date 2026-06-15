from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from lottie.governance.audit import (
    NullAuditLogger,
    SqliteAuditLogger,
    build_audit_logger,
    hash_model,
)
from lottie.governance.schema import AuditRecord


class _M(BaseModel):
    x: int


def _rec(
    agent: str = "echo", ts: str = "2026-06-13T10:00:00+00:00", status: str = "ok"
) -> AuditRecord:
    return AuditRecord(
        ts=ts, agent=agent, provider="mock/x", status=status, root=True,
        input_sha256="a" * 64, output_sha256="b" * 64, input_tokens=1, output_tokens=2,
        cost_usd=0.0, latency_ms=1.0, error=None,
    )


def test_hash_model_is_sha256_of_json() -> None:
    import hashlib
    m = _M(x=5)
    assert hash_model(m) == hashlib.sha256(m.model_dump_json().encode()).hexdigest()


def test_log_then_query_roundtrip(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(_rec())
    rows = logger.query()
    assert len(rows) == 1 and rows[0].agent == "echo" and rows[0].status == "ok"
    assert (tmp_path / ".lottie" / "audit.db").is_file()


def test_query_is_newest_first_and_append_only(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(_rec(ts="2026-06-13T10:00:00+00:00"))
    logger.log(_rec(ts="2026-06-13T11:00:00+00:00"))
    rows = logger.query()
    assert [r.ts for r in rows] == ["2026-06-13T11:00:00+00:00", "2026-06-13T10:00:00+00:00"]


def test_query_filters_agent_since_and_limit(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(_rec(agent="a", ts="2026-06-13T09:00:00+00:00"))
    logger.log(_rec(agent="b", ts="2026-06-13T10:00:00+00:00"))
    logger.log(_rec(agent="b", ts="2026-06-13T11:00:00+00:00"))
    assert {r.agent for r in logger.query(agent="b")} == {"b"}
    assert len(logger.query(since="2026-06-13T10:00:00+00:00")) == 2
    assert len(logger.query(limit=1)) == 1


def test_query_empty_db_returns_empty(tmp_path: Path) -> None:
    assert SqliteAuditLogger(tmp_path).query() == []


def test_log_is_best_effort_on_bad_path(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    logger = SqliteAuditLogger(blocker)  # blocker/.lottie/audit.db — parent is a file
    with pytest.warns(Warning):
        logger.log(_rec())  # must NOT raise


def test_null_logger_is_noop() -> None:
    NullAuditLogger().log(_rec())  # no raise, nothing persisted


def test_build_audit_logger_respects_disable_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")
    assert isinstance(build_audit_logger(tmp_path), NullAuditLogger)
    monkeypatch.delenv("LOTTIE_DISABLE_AUDIT", raising=False)
    assert isinstance(build_audit_logger(tmp_path), SqliteAuditLogger)


def test_total_cost_sums_one_agent(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    for cost in (0.01, 0.02, 0.03):
        logger.log(
            AuditRecord(
                ts="2026-06-14T10:00:00+00:00", agent="digest", provider="mock/x",
                status="ok", root=True, input_sha256="a" * 64, output_sha256="b" * 64,
                input_tokens=1, output_tokens=2, cost_usd=cost, latency_ms=1.0, error=None,
            )
        )
    assert abs(logger.total_cost("digest") - 0.06) < 1e-9


def test_total_cost_unknown_agent_is_zero(tmp_path: Path) -> None:
    assert SqliteAuditLogger(tmp_path).total_cost("nope") == 0.0


def test_total_cost_ignores_other_agents(tmp_path: Path) -> None:
    logger = SqliteAuditLogger(tmp_path)
    logger.log(
        AuditRecord(
            ts="2026-06-14T10:00:00+00:00", agent="a", provider=None, status="ok",
            root=True, input_sha256="a" * 64, output_sha256=None, input_tokens=0,
            output_tokens=0, cost_usd=0.05, latency_ms=1.0, error=None,
        )
    )
    assert SqliteAuditLogger(tmp_path).total_cost("b") == 0.0
