"""AuditSubscriber — the observer form of the audit write (V3 S4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lottie.governance.audit import SqliteAuditLogger
from lottie.governance.subscribers import AuditSubscriber
from lottie.runtime.events import RunBlocked, RunCompleted, RunFailed, RunStarted


def _completed(**kw: object) -> RunCompleted:
    base: dict[str, object] = {
        "run_id": "r1",
        "runnable": "Demo",
        "kind": "agent",
        "root": True,
        "provider": "mock/sim",
        "input_sha256": "a" * 64,
        "output_sha256": "b" * 64,
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.01,
        "latency_ms": 12.0,
    }
    base.update(kw)
    return RunCompleted.model_validate(base)


class TestWrites:
    def test_a_completed_run_is_logged_ok(self, tmp_path: Path) -> None:
        AuditSubscriber(SqliteAuditLogger(tmp_path)).on_event(_completed())
        rows = SqliteAuditLogger(tmp_path).query()
        assert len(rows) == 1 and rows[0].status == "ok" and rows[0].root is True

    def test_a_failed_run_is_logged_error(self, tmp_path: Path) -> None:
        AuditSubscriber(SqliteAuditLogger(tmp_path)).on_event(
            RunFailed(
                run_id="r1",
                runnable="Demo",
                kind="agent",
                input_sha256="a" * 64,
                error="boom",
                latency_ms=1.0,
            )
        )
        rows = SqliteAuditLogger(tmp_path).query()
        assert rows[0].status == "error" and rows[0].error == "boom"

    def test_a_failed_run_records_no_output_hash(self, tmp_path: Path) -> None:
        AuditSubscriber(SqliteAuditLogger(tmp_path)).on_event(
            RunFailed(
                run_id="r1", runnable="D", kind="agent", input_sha256="a" * 64,
                error="boom", latency_ms=1.0,
            )
        )
        assert SqliteAuditLogger(tmp_path).query()[0].output_sha256 is None


class TestIgnores:
    def test_run_started_is_not_logged(self, tmp_path: Path) -> None:
        # The ledger records outcomes, not intentions.
        AuditSubscriber(SqliteAuditLogger(tmp_path)).on_event(
            RunStarted(run_id="r1", runnable="D", kind="agent", input_sha256="a" * 64)
        )
        assert SqliteAuditLogger(tmp_path).query() == []

    def test_run_blocked_is_not_logged_here(self, tmp_path: Path) -> None:
        # A refused run carries a governance-specific status only the gate knows, so the
        # gate audits it through the BlockAudit callback instead.
        AuditSubscriber(SqliteAuditLogger(tmp_path)).on_event(
            RunBlocked(
                run_id="r1", runnable="D", kind="agent", input_sha256="a" * 64,
                blocked_by="policy", error="denied",
            )
        )
        assert SqliteAuditLogger(tmp_path).query() == []


class TestUnbuildableRecord:
    def test_a_record_that_cannot_be_built_warns_and_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bus already isolates a raising subscriber, but saying the RECORD was the
        problem is more useful than a generic 'subscriber failed'."""

        def _boom(**kw: object) -> object:
            raise ValueError("bad record")

        monkeypatch.setattr("lottie.governance.subscribers.AuditRecord", _boom)
        with pytest.warns(UserWarning, match="audit record could not be built"):
            AuditSubscriber(SqliteAuditLogger(tmp_path)).on_event(_completed())

    def test_nothing_is_written_when_the_record_cannot_be_built(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(**kw: object) -> object:
            raise ValueError("bad record")

        monkeypatch.setattr("lottie.governance.subscribers.AuditRecord", _boom)
        with pytest.warns(UserWarning):
            AuditSubscriber(SqliteAuditLogger(tmp_path)).on_event(_completed())
        assert SqliteAuditLogger(tmp_path).query() == []
