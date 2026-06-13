"""MeshEngine ABC. LocalEngine is the v1 default; a LangGraphEngine adapter
lands in Phase 3 behind this same interface (keep `run` engine-agnostic)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from lottie.mesh.schema import ApprovalDecision, MeshRunResult, MeshState, RouteDecision

MeshNode = Callable[[MeshState], MeshState]
RouteFn = Callable[[MeshState], RouteDecision]


class MeshEngine(ABC):
    """Drives a supervisor→worker loop over typed state, with pause/resume."""

    @abstractmethod
    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
        thread_id: str | None = None,
    ) -> MeshRunResult:
        """Run until FINISH, max_steps, or an interrupt. Returns a MeshRunResult."""

    @abstractmethod
    def resume(
        self,
        thread_id: str,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        decision: ApprovalDecision,
    ) -> MeshRunResult:
        """Continue an interrupted run from its checkpoint."""
