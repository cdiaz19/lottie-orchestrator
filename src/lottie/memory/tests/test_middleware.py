"""Memory middleware — the guards that keep a blocked run from being recorded (V3 S5)."""

from __future__ import annotations

import warnings

from pydantic import BaseModel

from lottie.memory.middleware import RunUsage, TrajectoryMiddleware
from lottie.memory.schema import MemoryDelta, MemoryOrigin
from lottie.runtime.context import ExecutionContext


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


def _ctx() -> ExecutionContext:
    return ExecutionContext(
        runnable="Demo", kind="agent", input=_In(task="t"), run_id="r1"
    )


def _nxt(ctx: ExecutionContext) -> BaseModel:
    return _Out(answer="ok")


class TestNoUsageMeansNoRecord:
    """`usage()` returns None when the gates blocked the run before `_execute`.

    Nothing happened, so nothing should be written — a blocked run must not leave a
    trajectory implying it ran.
    """

    def test_a_blocked_run_writes_no_trajectory(self) -> None:
        applied: list[list[MemoryDelta]] = []

        module = TrajectoryMiddleware(
            lambda deltas, ns, origin: applied.append(deltas),
            lambda: None,  # no metrics — the run never started
            enabled=True,
            namespace="ns",
            max_chars=100,
        )
        module(_ctx(), _nxt)
        assert applied == []

    def test_a_completed_run_does_write(self) -> None:
        applied: list[list[MemoryDelta]] = []

        module = TrajectoryMiddleware(
            lambda deltas, ns, origin: applied.append(deltas),
            lambda: RunUsage(success=True, input_tokens=3),
            enabled=True,
            namespace="ns",
            max_chars=100,
        )
        module(_ctx(), _nxt)
        assert len(applied) == 1 and applied[0][0].tags == ["trajectory", "success"]

    def test_a_failed_run_is_tagged_failure(self) -> None:
        applied: list[list[MemoryDelta]] = []
        module = TrajectoryMiddleware(
            lambda deltas, ns, origin: applied.append(deltas),
            lambda: RunUsage(success=False, error="boom"),
            enabled=True,
            namespace="ns",
            max_chars=100,
        )
        module(_ctx(), _nxt)
        assert applied[0][0].tags == ["trajectory", "failure"]

    def test_a_gateway_failure_never_fails_the_run(self) -> None:
        def _boom(deltas: list[MemoryDelta], ns: str, origin: MemoryOrigin) -> None:
            raise RuntimeError("gateway down")

        module = TrajectoryMiddleware(
            _boom,
            lambda: RunUsage(success=True),
            enabled=True,
            namespace="ns",
            max_chars=100,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert module(_ctx(), _nxt) == _Out(answer="ok")
