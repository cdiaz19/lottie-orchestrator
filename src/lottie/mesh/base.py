"""MeshAgent — a BaseAgent that orchestrates worker agents via a supervisor loop."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

from lottie.core import BaseAgent
from lottie.core.metrics import RunMetrics
from lottie.llm import LLMProvider, TokenUsage
from lottie.memory.base import MemoryClient
from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.local import LocalEngine
from lottie.mesh.plan import PlanRecorder, save_plan
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
        #: Set by instantiate_agent when a project root is known. None = no recording,
        #: so a directly-constructed mesh (tests) leaves no artefacts.
        self._plans_root: Path | None = None

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
        # The recorder wraps the router, so the plan captures the DECISIONS as they are
        # made rather than inferring them from history afterwards. Exact by construction,
        # and identical across both engines — which is why neither engine had to change.
        recorder = PlanRecorder(self._route_fn())
        result = self._engine.run(
            MeshState(task=data.task),
            nodes=self._nodes,
            route=recorder,
            max_steps=data.max_steps,
        )
        self._save_plan(recorder, result, data.task)
        return self._to_output(result)

    def _save_plan(self, recorder: PlanRecorder, result: MeshRunResult, task: str) -> None:
        """Persist the routing decisions this run made, so it can be replayed (E6).

        Best-effort and only for completed runs: an interrupted run has not finished
        deciding, so recording it would produce a plan that replays a partial flow as if
        it were the whole one. Never raises — a recording failure must not fail a run that
        already succeeded.
        """
        if self._plans_root is None or result.status != "complete":
            return
        try:
            save_plan(
                self._plans_root, self.name, result.thread_id or "last", recorder.plan(task)
            )
        except Exception as exc:  # best-effort — never fail an already-successful run
            warnings.warn(f"plan recording failed: {exc}", stacklevel=2)

    def set_plans_root(self, root: Path | None) -> None:
        """Enable plan recording under `root/.lottie/plans/` (via instantiate_agent)."""
        self._plans_root = root

    def resume(self, thread_id: str, decision: ApprovalDecision) -> MeshOutput:
        """Continue an interrupted mesh run from its checkpoint."""
        result = self._engine.resume(
            thread_id, nodes=self._nodes, route=self._route_fn(), decision=decision
        )
        return self._to_output(result)
