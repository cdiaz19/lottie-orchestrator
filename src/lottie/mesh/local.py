"""Hand-rolled, dependency-free mesh engine (Phase 2 default)."""

from __future__ import annotations

from collections.abc import Mapping

from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.errors import MeshError, MeshStepLimitExceeded
from lottie.mesh.schema import FINISH, ApprovalDecision, MeshRunResult, MeshState


class LocalEngine(MeshEngine):
    """Deterministic in-process supervisor loop (no checkpointing / HITL)."""

    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
        thread_id: str | None = None,
    ) -> MeshRunResult:
        state = initial
        for _ in range(max_steps):
            decision = route(state)
            if decision.next == FINISH:
                last = state.history[-1].result if state.history else ""
                return MeshRunResult(state=state.model_copy(update={"final": last}))
            state = nodes[decision.next](state)
        raise MeshStepLimitExceeded(
            f"routing loop exceeded max_steps={max_steps} without FINISH"
        )

    def resume(
        self,
        thread_id: str,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        decision: ApprovalDecision,
    ) -> MeshRunResult:
        raise MeshError(
            "HITL resume requires LangGraphEngine; install lottie-orchestrator[mesh]"
        )
