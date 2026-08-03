"""Execution semantics of the middleware onion.

These are the tests S2 leans on when it claims the swap-in is behaviour-preserving:
ordering, abort, cleanup-on-exception, and reverse post-phase order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Middleware, Next
from lottie.runtime.pipeline import Pipeline


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


def _hasher(model: BaseModel) -> str:
    return hashlib.sha256(model.model_dump_json().encode()).hexdigest()


def _core(data: _Input) -> _Output:
    return _Output(text=data.text.upper())


class _Tracer:
    """Records its own pre and post phases into a shared log."""

    def __init__(self, label: str, order: int, log: list[str]) -> None:
        self.name = label
        self.order = order
        self._log = log

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._log.append(f"pre:{self.name}")
        try:
            return nxt(ctx)
        finally:
            self._log.append(f"post:{self.name}")


class _Aborter:
    """A gate that refuses: never calls `nxt`."""

    name = "aborter"

    def __init__(self, order: int) -> None:
        self.order = order

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        raise PermissionError("denied")


def _pipeline(*mw: Middleware) -> Pipeline[_Input, _Output]:
    return Pipeline(
        runnable="Demo", kind="agent", core=_core, hasher=_hasher, middleware=list(mw)
    )


def _with_core(
    core: object, middleware: Sequence[Middleware] = ()
) -> Pipeline[_Input, _Output]:
    return Pipeline(
        runnable="Demo",
        kind="agent",
        core=core,  # type: ignore[arg-type]
        hasher=_hasher,
        middleware=middleware,
    )


class TestBareExecution:
    def test_runs_the_core_with_no_middleware(self) -> None:
        assert _pipeline().execute(_Input(text="hi")).text == "HI"

    def test_returns_the_concrete_output_type(self) -> None:
        assert isinstance(_pipeline().execute(_Input(text="hi")), _Output)


class TestOrdering:
    def test_pre_phases_run_low_order_first(self) -> None:
        log: list[str] = []
        _pipeline(_Tracer("a", 10, log), _Tracer("b", 20, log)).execute(_Input(text="x"))
        assert log[:2] == ["pre:a", "pre:b"]

    def test_post_phases_run_in_reverse(self) -> None:
        log: list[str] = []
        _pipeline(_Tracer("a", 10, log), _Tracer("b", 20, log)).execute(_Input(text="x"))
        assert log[-2:] == ["post:b", "post:a"]

    def test_registration_order_does_not_matter(self) -> None:
        log: list[str] = []
        # Registered high-order first; must still run low-order first.
        _pipeline(_Tracer("b", 20, log), _Tracer("a", 10, log)).execute(_Input(text="x"))
        assert log == ["pre:a", "pre:b", "post:b", "post:a"]

    def test_full_onion_sequence(self) -> None:
        log: list[str] = []
        _pipeline(
            _Tracer("a", 10, log), _Tracer("b", 20, log), _Tracer("c", 30, log)
        ).execute(_Input(text="x"))
        assert log == ["pre:a", "pre:b", "pre:c", "post:c", "post:b", "post:a"]


class TestAbort:
    def test_a_middleware_that_does_not_call_next_aborts_the_run(self) -> None:
        with pytest.raises(PermissionError):
            _pipeline(_Aborter(20)).execute(_Input(text="x"))

    def test_abort_skips_the_core(self) -> None:
        ran: list[str] = []

        def _tracking_core(data: _Input) -> _Output:
            ran.append("core")
            return _Output(text=data.text)

        with pytest.raises(PermissionError):
            _with_core(_tracking_core, [_Aborter(20)]).execute(_Input(text="x"))
        assert ran == []

    def test_outer_post_phases_still_run_when_an_inner_middleware_aborts(self) -> None:
        # The cost-settle guarantee: a denied run still releases its reservation.
        log: list[str] = []
        with pytest.raises(PermissionError):
            _pipeline(_Tracer("outer", 10, log), _Aborter(20)).execute(_Input(text="x"))
        assert log == ["pre:outer", "post:outer"]


class TestCoreFailure:
    def test_core_exception_propagates(self) -> None:
        def _boom(data: _Input) -> _Output:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            _with_core(_boom).execute(_Input(text="x"))

    def test_post_phases_still_run_when_the_core_raises(self) -> None:
        log: list[str] = []

        def _boom(data: _Input) -> _Output:
            raise ValueError("kaboom")

        with pytest.raises(ValueError):
            _with_core(_boom, [_Tracer("a", 10, log)]).execute(_Input(text="x"))
        assert log == ["pre:a", "post:a"]


class _Grabber:
    name = "grabber"
    order = 10

    def __init__(self, seen: list[ExecutionContext]) -> None:
        self._seen = seen

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._seen.append(ctx)
        return nxt(ctx)


class TestContext:
    def test_each_run_gets_a_fresh_context(self) -> None:
        seen: list[ExecutionContext] = []
        pipe = _pipeline(_Grabber(seen))
        pipe.execute(_Input(text="x"))
        pipe.execute(_Input(text="x"))
        assert len({c.run_id for c in seen}) == 2

    def test_context_carries_the_runnable_identity(self) -> None:
        seen: list[ExecutionContext] = []
        _pipeline(_Grabber(seen)).execute(_Input(text="hi"))
        assert seen[0].runnable == "Demo"
        assert seen[0].kind == "agent"
        assert seen[0].input == _Input(text="hi")

    def test_state_written_by_a_pre_phase_is_visible_in_its_post_phase(self) -> None:
        class _Stateful:
            name = "stateful"
            order = 10

            def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
                ctx.scoped("stateful")["handle"] = "H1"
                try:
                    return nxt(ctx)
                finally:
                    assert ctx.scoped("stateful")["handle"] == "H1"

        _pipeline(_Stateful()).execute(_Input(text="x"))
