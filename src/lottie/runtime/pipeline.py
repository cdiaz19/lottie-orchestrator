"""Execution core — compiles middleware into an onion and runs it.

`Pipeline` is what `InstrumentedRunnable.run` becomes in S2. It owns exactly two
responsibilities: order the chain, and run it. Everything else is a mounted module.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack
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
from lottie.runtime.middleware import Middleware, ScopedMiddleware, StreamCore

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class UnsafeHasherError(RuntimeError):
    """The injected hasher returned something that is not a sha256 digest.

    D6 says events carry hashes, never raw content. That guarantee is only as strong as
    the hasher, so it is VERIFIED rather than trusted: a hasher that echoes its input
    would put raw payloads on a bus the Plugin SDK (E7) opens to third parties. Caught
    by lab Round 28, which injected an echoing hasher and watched the payload leak.
    """


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
        provider: str | None = None,
        hasher: Callable[[BaseModel], str],
        middleware: Sequence[Middleware] = (),
        bus: EventBus | None = None,
        usage_factory: Callable[[], UsageAccumulator] = NullUsage,
    ) -> None:
        self._runnable = runnable
        self._kind = kind
        self._provider = provider
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
            provider=self._provider,
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

    def execute_stream(self, data: InputT, core: StreamCore) -> Generator[str, None, None]:
        """Run `core`'s deltas through the SCOPED subset of the chain.

        Only middleware offering `scope` participate — the ones whose whole effect is a
        scope. `verify`, `check_output` and `reflect` need an output value and are
        therefore absent, which matches `run_stream`'s existing behaviour exactly.

        `ExitStack` is what makes this correct: the `with` spans generator consumption,
        so a cost reservation is settled after the last delta rather than at generator
        creation, and the stack unwinds in reverse — onion post-order, for free.
        """
        ctx = ExecutionContext(
            runnable=self._runnable,
            kind=self._kind,
            input=data,
            run_id=uuid.uuid4().hex,
            usage=self._usage_factory(),
            provider=self._provider,
        )
        scoped = [m for m in self._chain if isinstance(m, ScopedMiddleware)]
        with ExitStack() as stack:
            for mw in scoped:
                stack.enter_context(mw.scope(ctx))
            input_hash = self._checked_hash(ctx.input)
            self._bus.emit(
                RunStarted(
                    run_id=ctx.run_id,
                    runnable=ctx.runnable,
                    kind=ctx.kind,
                    root=ctx.root,
                    provider=ctx.provider,
                    input_sha256=input_hash,
                )
            )
            start = perf_counter()
            try:
                yield from core(ctx)
            except GeneratorExit:
                # BaseException, NOT Exception — handled explicitly. The transport cancels
                # by closing the generator, and a cancelled stream is a PARTIAL run, not a
                # successful one. Without this branch the `finally` below would emit
                # RunCompleted and the ledger would record a cancelled request as "ok".
                ctx.scoped("pipeline")["failed"] = True
                self._bus.emit(
                    RunFailed(
                        run_id=ctx.run_id,
                        runnable=ctx.runnable,
                        kind=ctx.kind,
                        root=ctx.root,
                        provider=ctx.provider,
                        input_sha256=input_hash,
                        error="stream closed before completion",
                        latency_ms=(perf_counter() - start) * 1000,
                        input_tokens=ctx.usage.input_tokens,
                        output_tokens=ctx.usage.output_tokens,
                        cost_usd=ctx.usage.cost_usd,
                    )
                )
                raise
            except Exception as exc:
                ctx.scoped("pipeline")["failed"] = True
                self._bus.emit(
                    RunFailed(
                        run_id=ctx.run_id,
                        runnable=ctx.runnable,
                        kind=ctx.kind,
                        root=ctx.root,
                        provider=ctx.provider,
                        input_sha256=input_hash,
                        error=repr(exc),
                        latency_ms=(perf_counter() - start) * 1000,
                        input_tokens=ctx.usage.input_tokens,
                        output_tokens=ctx.usage.output_tokens,
                        cost_usd=ctx.usage.cost_usd,
                    )
                )
                raise
            finally:
                # A `finally`, not an else: the transport cancels by CLOSING the
                # generator, which raises GeneratorExit here. A cancelled stream still
                # consumed budget and still belongs in the ledger.
                if not ctx.scoped("pipeline").get("failed"):
                    self._bus.emit(
                        RunCompleted(
                            run_id=ctx.run_id,
                            runnable=ctx.runnable,
                            kind=ctx.kind,
                            root=ctx.root,
                            provider=ctx.provider,
                            input_sha256=input_hash,
                            # A stream has no single typed Output.
                            output_sha256=None,
                            input_tokens=ctx.usage.input_tokens,
                            output_tokens=ctx.usage.output_tokens,
                            cost_usd=ctx.usage.cost_usd,
                            latency_ms=(perf_counter() - start) * 1000,
                        )
                    )

    def scoped_names(self) -> list[str]:
        """Names of the middleware that participate in a streaming chain, in order."""
        return [m.name for m in self._chain if isinstance(m, ScopedMiddleware)]

    def _checked_hash(self, model: BaseModel) -> str:
        """Hash `model`, verifying the result is actually a digest.

        One regex per emission. Negligible next to the LLM call it accompanies, and it
        converts D6 from a convention the caller must honour into a property the kernel
        enforces.
        """
        digest = self._hasher(model)
        if not _SHA256_RE.match(digest):
            raise UnsafeHasherError(
                "hasher did not return a sha256 hex digest; events must never carry raw "
                "content (V3 spec D6). Got a "
                f"{len(digest)}-character value."
            )
        return digest

    def _core_frame(self, ctx: ExecutionContext) -> BaseModel:
        """Innermost frame: run the real work, emit lifecycle events from HERE.

        Emitting here rather than from `execute` is load-bearing. It means observers see
        the completed run BEFORE any middleware post-phase, so an audit subscriber
        records the real cost before the cost middleware's `finally` settles the
        reservation — the invariant documented at `core/base_agent.py:461-466`.
        """
        input_hash = self._checked_hash(ctx.input)
        self._bus.emit(
            RunStarted(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                root=ctx.root,
                provider=ctx.provider,
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
                    root=ctx.root,
                    provider=ctx.provider,
                    input_sha256=input_hash,
                    error=repr(exc),
                    latency_ms=(perf_counter() - start) * 1000,
                    input_tokens=ctx.usage.input_tokens,
                    output_tokens=ctx.usage.output_tokens,
                    cost_usd=ctx.usage.cost_usd,
                )
            )
            raise
        self._bus.emit(
            RunCompleted(
                run_id=ctx.run_id,
                runnable=ctx.runnable,
                kind=ctx.kind,
                root=ctx.root,
                provider=ctx.provider,
                input_sha256=input_hash,
                output_sha256=self._checked_hash(output),
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
                root=ctx.root,
                provider=ctx.provider,
                input_sha256=self._checked_hash(ctx.input),
                blocked_by=entered[-1] if entered else "unknown",
                error=str(exc),
            )
        )
