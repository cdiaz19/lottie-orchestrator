"""Lifecycle-event emission, including the ordering invariant S2 depends on.

`RunCompleted` fires from the INNERMOST frame, so a subscriber (audit) observes the run
before any middleware post-phase (cost settle). Today that ordering is hand-maintained in
`BaseAgent.run`'s nested finally blocks and documented at `core/base_agent.py:461-466`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.events import (
    EventBus,
    RunBlocked,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
)
from lottie.runtime.middleware import Middleware, Next
from lottie.runtime.pipeline import Pipeline


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


def _hasher(model: BaseModel) -> str:
    return f"h:{model.model_dump_json()}"


def _core(data: _Input) -> _Output:
    return _Output(text=data.text.upper())


def _boom(data: _Input) -> _Output:
    raise ValueError("kaboom")


class _Recorder:
    name = "recorder"

    def __init__(self, log: list[str] | None = None) -> None:
        self.seen: list[RunEvent] = []
        self._log = log

    def on_event(self, event: RunEvent) -> None:
        self.seen.append(event)
        if self._log is not None:
            self._log.append(f"event:{type(event).__name__}")

    def only(self, model: type[RunEvent]) -> list[RunEvent]:
        return [e for e in self.seen if isinstance(e, model)]


class _Denier:
    name = "policy"
    order = 20

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        raise PermissionError("denied")


def _pipe(
    bus: EventBus, middleware: Sequence[Middleware] = (), core: object = _core
) -> Pipeline[_Input, _Output]:
    return Pipeline(
        runnable="Demo",
        kind="agent",
        core=core,  # type: ignore[arg-type]
        hasher=_hasher,
        middleware=middleware,
        bus=bus,
    )


def _bus_with_recorder(log: list[str] | None = None) -> tuple[EventBus, _Recorder]:
    bus, rec = EventBus(), _Recorder(log)
    bus.subscribe(rec)
    return bus, rec


class TestHappyPath:
    def test_emits_started_then_completed(self) -> None:
        bus, rec = _bus_with_recorder()
        _pipe(bus).execute(_Input(text="hi"))
        assert [type(e).__name__ for e in rec.seen] == ["RunStarted", "RunCompleted"]

    def test_completed_carries_the_output_hash(self) -> None:
        bus, rec = _bus_with_recorder()
        _pipe(bus).execute(_Input(text="hi"))
        completed = rec.only(RunCompleted)[0]
        assert isinstance(completed, RunCompleted)
        assert completed.output_sha256 == _hasher(_Output(text="HI"))

    def test_started_and_completed_share_one_run_id(self) -> None:
        bus, rec = _bus_with_recorder()
        _pipe(bus).execute(_Input(text="hi"))
        assert len({e.run_id for e in rec.seen}) == 1

    def test_completed_reports_latency(self) -> None:
        bus, rec = _bus_with_recorder()
        _pipe(bus).execute(_Input(text="hi"))
        completed = rec.only(RunCompleted)[0]
        assert isinstance(completed, RunCompleted)
        assert completed.latency_ms >= 0.0


class TestInnermostFrameInvariant:
    def test_completed_fires_before_any_middleware_post_phase(self) -> None:
        """The audit-before-settle guarantee, asserted directly."""
        log: list[str] = []
        bus, _ = _bus_with_recorder(log)

        class _Settler:
            name = "cost"
            order = 30

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                log.append("pre:cost")
                try:
                    return nxt(ctx)
                finally:
                    log.append("settle:cost")

        _pipe(bus, [_Settler()]).execute(_Input(text="hi"))
        assert log == ["pre:cost", "event:RunStarted", "event:RunCompleted", "settle:cost"]

    def test_started_fires_after_every_pre_phase(self) -> None:
        log: list[str] = []
        bus, _ = _bus_with_recorder(log)

        class _Gate:
            name = "gate"
            order = 10

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                log.append("pre:gate")
                return nxt(ctx)

        _pipe(bus, [_Gate()]).execute(_Input(text="hi"))
        assert log.index("pre:gate") < log.index("event:RunStarted")


class TestFailure:
    def test_core_failure_emits_run_failed(self) -> None:
        bus, rec = _bus_with_recorder()
        with pytest.raises(ValueError):
            _pipe(bus, core=_boom).execute(_Input(text="hi"))
        assert len(rec.only(RunFailed)) == 1

    def test_core_failure_emits_no_run_completed(self) -> None:
        bus, rec = _bus_with_recorder()
        with pytest.raises(ValueError):
            _pipe(bus, core=_boom).execute(_Input(text="hi"))
        assert rec.only(RunCompleted) == []


class TestBlocked:
    def test_a_gate_abort_emits_run_blocked(self) -> None:
        bus, rec = _bus_with_recorder()
        with pytest.raises(PermissionError):
            _pipe(bus, [_Denier()]).execute(_Input(text="hi"))
        assert len(rec.only(RunBlocked)) == 1

    def test_run_blocked_names_the_refusing_middleware(self) -> None:
        bus, rec = _bus_with_recorder()
        with pytest.raises(PermissionError):
            _pipe(bus, [_Denier()]).execute(_Input(text="hi"))
        blocked = rec.only(RunBlocked)[0]
        assert isinstance(blocked, RunBlocked)
        assert blocked.blocked_by == "policy"

    def test_a_blocked_run_emits_no_started_or_completed(self) -> None:
        bus, rec = _bus_with_recorder()
        with pytest.raises(PermissionError):
            _pipe(bus, [_Denier()]).execute(_Input(text="hi"))
        assert rec.only(RunStarted) == []
        assert rec.only(RunCompleted) == []

    def test_a_core_failure_is_not_reported_as_blocked(self) -> None:
        # The distinction that makes RunBlocked meaningful: the work never started.
        bus, rec = _bus_with_recorder()
        with pytest.raises(ValueError):
            _pipe(bus, core=_boom).execute(_Input(text="hi"))
        assert rec.only(RunBlocked) == []


class TestSubscriberIsolationEndToEnd:
    def test_a_broken_subscriber_cannot_fail_a_run(self) -> None:
        class _Exploder:
            name = "exploder"

            def on_event(self, event: RunEvent) -> None:
                raise RuntimeError("blew up")

        bus = EventBus()
        bus.subscribe(_Exploder())
        with pytest.warns(UserWarning):
            result = _pipe(bus).execute(_Input(text="hi"))
        assert result.text == "HI"
