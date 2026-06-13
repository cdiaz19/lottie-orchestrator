"""Hand-rolled, dependency-free mesh engine (Phase 2 default)."""

from __future__ import annotations

from collections.abc import Mapping

from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.errors import MeshStepLimitExceeded
from lottie.mesh.schema import FINISH, MeshState


class LocalEngine(MeshEngine):
    """Deterministic in-process supervisor loop."""

    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
    ) -> MeshState:
        state = initial
        for _ in range(max_steps):
            decision = route(state)
            if decision.next == FINISH:
                last = state.history[-1].result if state.history else ""
                return state.model_copy(update={"final": last})
            state = nodes[decision.next](state)
        raise MeshStepLimitExceeded(
            f"routing loop exceeded max_steps={max_steps} without FINISH"
        )
