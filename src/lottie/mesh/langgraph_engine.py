"""LangGraph-backed mesh engine. ALL langgraph imports are confined to this module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lottie.mesh.checkpoint import build_checkpointer
from lottie.mesh.engine import MeshEngine, MeshNode, RouteFn
from lottie.mesh.errors import MeshError
from lottie.mesh.schema import FINISH, ApprovalDecision, MeshRunResult, MeshState

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


class LangGraphEngine(MeshEngine):
    """Runs the supervisor→worker mesh on a compiled LangGraph StateGraph."""

    def __init__(
        self,
        *,
        checkpoint: str = "memory",
        root: Path | None = None,
        interrupt_before: list[str] | None = None,
    ) -> None:
        if not _HAS_LANGGRAPH:
            raise MeshError("LangGraphEngine requires: pip install lottie-orchestrator[mesh]")
        self._checkpoint = checkpoint
        self._root = root
        self._interrupt_before = list(interrupt_before or [])
        self._saver: Any = None

    def _build(self, nodes: Mapping[str, MeshNode], route: RouteFn) -> Any:
        builder = StateGraph(MeshState)

        def supervisor(state: MeshState) -> dict[str, Any]:
            return {}

        def choose(state: MeshState) -> Any:
            decision = route(state)
            if decision.parallel:
                return [Send(name, state) for name in decision.parallel]
            return END if decision.next == FINISH else decision.next

        builder.add_node("supervisor", supervisor)
        for name, node in nodes.items():
            builder.add_node(name, self._wrap(node))
        builder.add_edge(START, "supervisor")
        path_map: dict[Any, str] = {name: name for name in nodes}
        path_map[END] = END
        builder.add_conditional_edges("supervisor", choose, path_map)
        for name in nodes:
            builder.add_edge(name, "supervisor")

        if self._saver is None:
            self._saver = build_checkpointer(self._checkpoint, self._root)
        return builder.compile(
            checkpointer=self._saver, interrupt_before=self._interrupt_before or None
        )

    @staticmethod
    def _wrap(node: MeshNode) -> Any:
        def lg_node(state: MeshState) -> dict[str, Any]:
            new_state = node(state)
            delta = new_state.history[len(state.history):]
            update: dict[str, Any] = {"history": delta}
            # Only write `final` when set: parallel branches share the single-value
            # `final` channel, so emitting None from every branch would collide.
            if new_state.final is not None:
                update["final"] = new_state.final
            return update

        return lg_node

    def run(
        self,
        initial: MeshState,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        max_steps: int,
        thread_id: str | None = None,
    ) -> MeshRunResult:
        graph = self._build(nodes, route)
        tid = thread_id or "default"
        config = {"configurable": {"thread_id": tid}, "recursion_limit": max_steps * 3}
        graph.invoke(initial, config)
        snap = graph.get_state(config)
        state = MeshState.model_validate(snap.values)
        last = state.history[-1].result if state.history else ""
        ordered = sorted(state.history, key=lambda s: (s.step, s.worker))
        return MeshRunResult(
            state=state.model_copy(update={"history": ordered, "final": state.final or last}),
            status="complete",
            thread_id=tid,
        )

    def history(
        self,
        thread_id: str,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
    ) -> list[MeshState]:
        """Return the MeshState at each persisted checkpoint for a thread (newest first)."""
        graph = self._build(nodes, route)
        config = {"configurable": {"thread_id": thread_id}}
        states: list[MeshState] = []
        for snap in graph.get_state_history(config):
            if not snap.values.get("task"):
                continue
            states.append(MeshState.model_validate(snap.values))
        return states

    def resume(
        self,
        thread_id: str,
        *,
        nodes: Mapping[str, MeshNode],
        route: RouteFn,
        decision: ApprovalDecision,
    ) -> MeshRunResult:
        raise NotImplementedError  # implemented in a later task (HITL)
