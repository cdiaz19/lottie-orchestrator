"""Memory middleware — recall and trajectory capture as mounted modules (V3 S5).

Owned by the memory subsystem. Each takes the memory client (or a gateway callback) and
its own configuration — no `BaseAgent` — which is what lets `core` stop importing
`memory.recall`, `memory.reflection` and `memory.schema`.

Reflection is deliberately NOT here. `_maybe_reflect` re-enters the agent's own
`complete()` with a hand-primed usage context so the per-run token cap counts
cumulatively; extracting it would need a host Protocol that is `BaseAgent` in all but
spelling. It becomes a real module once E4 owns message assembly and the run budget is
itself a module. Recorded rather than faked.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

from pydantic import BaseModel

from lottie.memory.base import MemoryClient
from lottie.memory.recall import RecalledMemory, render_as_data
from lottie.memory.reflection import RunTrajectory, clip
from lottie.memory.schema import (
    DeltaOp,
    MemoryDelta,
    MemoryOrigin,
    MemoryQuery,
    MemoryTier,
)
from lottie.runtime.context import ExecutionContext
from lottie.runtime.middleware import Next, Order


class RunUsage(BaseModel):
    """The slice of a finished run's metrics these modules record.

    Declared here rather than imported from `core.metrics`: naming what it needs is what
    keeps the memory-to-core edge from existing at all.
    """

    success: bool
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


#: Reads the just-finished run's metrics.
UsageReader = Callable[[], RunUsage | None]

#: Applies learned content through the MemoryAgent gateway (rule 13b). Injected so this
#: module never imports `memory.agent`, which would pull `core` back in.
DeltaApplier = Callable[[list[MemoryDelta], str, MemoryOrigin], None]


class RecallMiddleware:
    """Best-effort recall-as-data before `_execute`, cleared on the way out.

    Fail-OPEN, unlike the write gate: a read failure means the run proceeds without
    context, because losing recall is a degraded run while failing the run is a lost one.
    """

    name = "recall"
    order = Order.RECALL

    def __init__(
        self,
        memory: MemoryClient,
        set_prefix: Callable[[str], None],
        *,
        enabled: bool,
        namespace: str,
        limit: int,
    ) -> None:
        self._memory = memory
        self._set_prefix = set_prefix
        self._enabled = enabled
        self._namespace = namespace
        self._limit = limit

    def _load(self) -> None:
        self._set_prefix("")
        if not self._enabled:
            return
        try:
            result = self._memory.recall(
                MemoryQuery(
                    text="",
                    namespace=self._namespace,
                    tier=MemoryTier.SEMANTIC,
                    limit=self._limit,
                )
            )
            self._set_prefix(render_as_data(RecalledMemory.from_result(result)))
        except Exception as exc:  # recall is best-effort — never break the run
            warnings.warn(f"recall failed, proceeding without context: {exc}", stacklevel=2)
            self._set_prefix("")

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        self._load()
        try:
            return nxt(ctx)
        finally:
            self._set_prefix("")


class TrajectoryMiddleware:
    """Append the finished run to episodic memory through the gateway (rule 13b).

    Runs for successes AND failures — failures are the more useful half of a distillation
    corpus. Makes no LLM call, so it has no budget interaction. Never raises: a store
    failure must not fail an otherwise-good run, nor mask an already-failing one.
    """

    name = "trajectory"
    order = Order.TRAJECTORY

    def __init__(
        self,
        apply_deltas: DeltaApplier,
        usage: UsageReader,
        *,
        enabled: bool,
        namespace: str,
        max_chars: int,
    ) -> None:
        self._apply = apply_deltas
        self._usage = usage
        self._enabled = enabled
        self._namespace = namespace
        self._max_chars = max_chars

    def _persist(self, ctx: ExecutionContext, output: BaseModel | None) -> None:
        if not self._enabled:
            return
        m = self._usage()
        if m is None:  # gates blocked the run before `_execute` — nothing happened
            return
        try:
            limit = self._max_chars
            trajectory = RunTrajectory(
                task=clip(ctx.input.model_dump_json(), limit),
                outcome=clip(output.model_dump_json(), limit) if output is not None else "",
                success=m.success,
                error=m.error,
                input_tokens=m.input_tokens,
                output_tokens=m.output_tokens,
                cost_usd=m.cost_usd,
                latency_ms=m.latency_ms,
            )
            self._apply(
                [
                    MemoryDelta(
                        op=DeltaOp.ADD,
                        content=trajectory.model_dump_json(),
                        tags=["trajectory", "success" if m.success else "failure"],
                    )
                ],
                self._namespace,
                MemoryOrigin.MANUAL,
            )
        except Exception as exc:  # best-effort — never fail or mask the run
            warnings.warn(f"trajectory persistence failed: {exc}", stacklevel=2)

    def __call__(self, ctx: ExecutionContext, nxt: Next) -> BaseModel:
        output: BaseModel | None = None
        try:
            output = nxt(ctx)
            return output
        finally:
            self._persist(ctx, output)
