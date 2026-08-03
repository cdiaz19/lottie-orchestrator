"""Execution core — compiles middleware into an onion and runs it.

`Pipeline` is what `InstrumentedRunnable.run` becomes in S2. It owns exactly two
responsibilities: order the chain, and run it. Everything else is a mounted module.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import cast

from pydantic import BaseModel

from lottie.runtime.context import (
    ExecutionContext,
    NullUsage,
    RunKind,
    UsageAccumulator,
)
from lottie.runtime.events import (
    EventBus,
    RunBlocked,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from lottie.runtime.middleware import Middleware


class Pipeline[InputT: BaseModel, OutputT: BaseModel]:
    """An ordered middleware chain wrapping a core execution function.

    `hasher` is required rather than defaulted so the kernel never has to guess at a
    hashing scheme: S2 injects `lottie.governance.audit.hash_model`, keeping event
    hashes byte-identical to the ones already in the audit ledger.
    """

    def __init__(
        self,
        *,
        runnable: str,
        kind: RunKind,
        core: Callable[[InputT], OutputT],
        hasher: Callable[[BaseModel], str],
        middleware: Sequence[Middleware] = (),
        bus: EventBus | None = None,
        usage_factory: Callable[[], UsageAccumulator] = NullUsage,
    ) -> None:
        self._runnable = runnable
        self._kind = kind
        self._core = core
        self._hasher = hasher
        self._chain = sorted(middleware, key=lambda m: m.order)
        self._bus = bus if bus is not None else EventBus()
        self._usage_factory = usage_factory

    def execute(self, data: InputT) -> OutputT:
        """Run `data` through the chain and return the core's typed output.

        The single `cast` is the only place typing is narrowed: the chain is
        heterogeneous and speaks `BaseModel`, but the core function is typed
        `Callable[[InputT], OutputT]`, so the value flowing back out is an `OutputT`
        by construction.
        """
        ctx = ExecutionContext(
            runnable=self._runnable,
            kind=self._kind,
            input=data,
            run_id=uuid.uuid4().hex,
            usage=self._usage_factory(),
        )
        entered: list[str] = []
        reached_core = False

        def step(index: int) -> Callable[[ExecutionContext], BaseModel]:
            def call(c: ExecutionContext) -> BaseModel:
                nonlocal reached_core
                if index == len(self._chain):
                    reached_core = True
                    return self._core_frame(c)
                mw = self._chain[index]
                entered.append(mw.name)
                return mw(c, step(index + 1))

            return call

        try:
            return cast(OutputT, step(0)(ctx))
        except Exception as exc:
            if not reached_core:
                # A gate refused before the work ever started — today's `_write_block`.
                self._emit_blocked(ctx, entered, exc)
            raise

    def _core_frame(self, ctx: ExecutionContext) -> BaseModel:
        """Innermost frame: run the real work, emit lifecycle events from HERE.

        Emitting here rather than from `execute` is load-bearing. It means observers see
        the completed run BEFORE any middleware post-phase, so an audit subscriber
        records the real cost before the cost middleware's `finally` settles the
        reservation — the invariant documented at `core/base_agent.py:461-466`.
        """
        input_hash = self._hasher(ctx.input)
        self._bus.emit(
            RunStarted(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                input_sha256=input_hash,
            )
        )
        start = perf_counter()
        try:
            output = self._core(cast(InputT, ctx.input))
        except Exception as exc:
            self._bus.emit(
                RunFailed(
                    run_id=ctx.run_id,
                    runnable=ctx.runnable,
                    kind=ctx.kind,
                    input_sha256=input_hash,
                    error=repr(exc),
                    latency_ms=(perf_counter() - start) * 1000,
                )
            )
            raise
        self._bus.emit(
            RunCompleted(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                input_sha256=input_hash,
                output_sha256=self._hasher(output),
                input_tokens=ctx.usage.input_tokens,
                output_tokens=ctx.usage.output_tokens,
                cost_usd=ctx.usage.cost_usd,
                latency_ms=(perf_counter() - start) * 1000,
            )
        )
        return output

    def _emit_blocked(
        self, ctx: ExecutionContext, entered: list[str], exc: Exception
    ) -> None:
        self._bus.emit(
            RunBlocked(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                input_sha256=self._hasher(ctx.input),
                blocked_by=entered[-1] if entered else "unknown",
                error=str(exc),
            )
        )
