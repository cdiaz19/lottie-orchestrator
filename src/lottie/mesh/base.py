"""MeshAgent — a BaseAgent that orchestrates worker agents via a supervisor loop."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from lottie.core import BaseAgent
from lottie.core.metrics import RunMetrics
from lottie.llm import LLMProvider, TokenUsage
from lottie.memory.base import MemoryClient
from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.local import LocalEngine
from lottie.mesh.router import SupervisorRouter
from lottie.mesh.schema import (
    ApprovalDecision,
    MeshInput,
    MeshOutput,
    MeshRunResult,
    MeshState,
    RouteDecision,
)


class MeshAgent(BaseAgent[MeshInput, MeshOutput]):
    """Routes a task across declared worker nodes until the supervisor says FINISH."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        nodes: Mapping[str, MeshNode],
        descriptions: Mapping[str, str],
        engine: MeshEngine | None = None,
        name: str | None = None,
        memory: MemoryClient | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            llm,
            name=name,
            memory=memory,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self._nodes = dict(nodes)
        self._descriptions = dict(descriptions)
        self._engine = engine or LocalEngine()
        self._router = SupervisorRouter(self.complete)

    def _accumulate(self, metrics: RunMetrics | None) -> None:
        """Fold one worker run's tokens/cost into the active mesh run context."""
        if metrics is None or self._active_ctx is None:
            return
        self._active_ctx.add_usage(
            TokenUsage(
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
            ),
            metrics.cost_usd,
        )

    def _route_fn(self) -> RouteFn:
        def route(state: MeshState) -> RouteDecision:
            return self._router.route(state, self._descriptions)

        return route

    def _to_output(self, result: MeshRunResult) -> MeshOutput:
        return MeshOutput(
            final=result.state.final or "",
            history=result.state.history,
            status=result.status,
            thread_id=result.thread_id,
            pending=result.pending,
        )

    def _execute(self, data: MeshInput) -> MeshOutput:
        result = self._engine.run(
            MeshState(task=data.task),
            nodes=self._nodes,
            route=self._route_fn(),
            max_steps=data.max_steps,
        )
        return self._to_output(result)

    def resume(self, thread_id: str, decision: ApprovalDecision) -> MeshOutput:
        """Continue an interrupted mesh run from its checkpoint."""
        result = self._engine.resume(
            thread_id, nodes=self._nodes, route=self._route_fn(), decision=decision
        )
        return self._to_output(result)
