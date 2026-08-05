"""SessionMiddleware — the guard that keeps a blocked run out of the history (V3 S5)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.session.middleware import SessionMiddleware
from lottie.session.schema import SessionRun
from lottie.session.store import SessionStore


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


def _ctx() -> ExecutionContext:
    return ExecutionContext(runnable="D", kind="agent", input=_In(task="t"), run_id="r1")


def _nxt(ctx: ExecutionContext) -> BaseModel:
    return _Out(answer="ok")


def _hasher(model: BaseModel) -> str:
    return "a" * 64


def test_a_blocked_run_is_not_recorded(tmp_path: Path) -> None:
    # `usage()` is None when the gates refused before `_execute`. Nothing ran, so the
    # session history must not imply it did.
    store = SessionStore(tmp_path)
    state = store.save(store.start("s1", "D"))
    module = SessionMiddleware(store, lambda: state, lambda s: None, lambda: None, _hasher)
    module(_ctx(), _nxt)
    assert store.require("s1").runs == []


def test_a_completed_run_is_recorded(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    state = store.save(store.start("s1", "D"))
    module = SessionMiddleware(
        store,
        lambda: state,
        lambda s: None,
        lambda: SessionRun(ts=0.0, status="ok", cost_usd=0.5),
        _hasher,
    )
    module(_ctx(), _nxt)
    rows = store.require("s1").runs
    assert len(rows) == 1 and rows[0].cost_usd == 0.5 and rows[0].input_sha256 == "a" * 64


def test_no_session_means_no_write(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    module = SessionMiddleware(
        store, lambda: None, lambda s: None, lambda: SessionRun(ts=0.0, status="ok"), _hasher
    )
    module(_ctx(), _nxt)
    assert store.list() == []
