"""Checkpointer factory for LangGraphEngine. langgraph imports are guarded."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lottie.mesh.errors import MeshError

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(kind: str = "memory", root: Path | None = None) -> BaseCheckpointSaver[Any]:
    """Return a MemorySaver (default) or a SqliteSaver under .lottie/mesh/."""
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise MeshError(
            "LangGraph mesh features require: pip install lottie-orchestrator[mesh]"
        ) from exc

    if kind == "memory":
        return MemorySaver()
    if kind == "sqlite":  # pragma: no cover - not exercised in the hermetic suite
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        base = (root or Path.cwd()) / ".lottie" / "mesh"
        base.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(base / "checkpoints.db"), check_same_thread=False)
        return SqliteSaver(conn)
    raise MeshError(f"unknown checkpointer kind {kind!r}")
