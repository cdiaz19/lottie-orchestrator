"""Dispatch-overhead budget for the middleware chain.

The V3 spec makes performance a gate for every slice, which requires a baseline. The
bound below is deliberately generous — roughly two orders of magnitude above the
measured cost — because this runs on shared CI where a tight bound would be flaky. It
is a runaway-regression guard, not a precision instrument: it catches someone adding an
I/O call or an O(n^2) walk to the hot path, which is exactly the failure mode that
matters.
"""

from __future__ import annotations

from time import perf_counter

from pydantic import BaseModel

from lottie.runtime.context import ExecutionContext
from lottie.runtime.events import EventBus, RunEvent
from lottie.runtime.middleware import Next
from lottie.runtime.pipeline import Pipeline

ITERATIONS = 1000
CHAIN_DEPTH = 10
SUBSCRIBERS = 3

# Budget: total wall time per run, chain + events, excluding the core function.
MAX_OVERHEAD_MS = 1.0


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


def _hasher(model: BaseModel) -> str:
    return "a" * 64  # constant valid-shaped digest: measures dispatch, not hashing


def _core(data: _Input) -> _Output:
    return _Output(text=data.text)


class _Passthrough:
    def __init__(self, order: int) -> None:
        self.name = f"mw{order}"
        self.order = order

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        return nxt(ctx)


class _Sink:
    name = "sink"

    def on_event(self, event: RunEvent) -> None:
        return None


def _build(depth: int, subscribers: int = 0) -> Pipeline[_Input, _Output]:
    bus = EventBus()
    for _ in range(subscribers):
        bus.subscribe(_Sink())
    return Pipeline(
        runnable="Bench",
        kind="agent",
        core=_core,
        hasher=_hasher,
        middleware=[_Passthrough((i + 1) * 10) for i in range(depth)],
        bus=bus,
    )


def _measure(pipe: Pipeline[_Input, _Output]) -> float:
    """Mean milliseconds per `execute` call."""
    data = _Input(text="benchmark")
    pipe.execute(data)  # warm up import/attribute caches
    start = perf_counter()
    for _ in range(ITERATIONS):
        pipe.execute(data)
    return ((perf_counter() - start) * 1000) / ITERATIONS


class TestDispatchOverhead:
    def test_full_chain_stays_within_budget(self) -> None:
        per_run_ms = _measure(_build(CHAIN_DEPTH, SUBSCRIBERS))
        assert per_run_ms < MAX_OVERHEAD_MS, (
            f"{CHAIN_DEPTH} middleware + {SUBSCRIBERS} subscribers cost "
            f"{per_run_ms:.4f} ms/run, budget {MAX_OVERHEAD_MS} ms"
        )

    def test_chain_cost_grows_no_worse_than_linearly(self) -> None:
        """Guards the shape of the cost, which a wall-clock bound alone cannot."""
        shallow = _measure(_build(2))
        deep = _measure(_build(20))
        # 10x the middleware must not cost more than 20x the time. Loose enough to
        # survive CI jitter, tight enough to catch quadratic dispatch.
        assert deep < shallow * 20 + MAX_OVERHEAD_MS
