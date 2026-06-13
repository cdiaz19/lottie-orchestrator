"""MeshEngine ABC. LocalEngine is the v1 default; a LangGraphEngine adapter
lands in Phase 3 behind this same interface (keep `run` engine-agnostic)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

from lottie.mesh.schema import MeshState, RouteDecision

MeshNode = Callable[[MeshState], MeshState]
RouteFn = Callable[[MeshState], RouteDecision]


class MeshEngine(ABC):
    """Drives a supervisor→worker loop over typed state."""

    @abstractmethod
    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
    ) -> MeshState:
        """Loop: route → dispatch chosen node → repeat until FINISH or max_steps."""
