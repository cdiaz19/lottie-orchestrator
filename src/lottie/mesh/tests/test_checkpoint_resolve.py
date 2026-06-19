from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")


def test_resolve_precedence_arg_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.mesh.langgraph_engine import _resolve_checkpoint

    monkeypatch.setenv("LOTTIE_MESH_CHECKPOINT", "sqlite")
    assert _resolve_checkpoint("memory") == "memory"  # explicit arg beats env


def test_resolve_env_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from lottie.mesh.langgraph_engine import _resolve_checkpoint

    monkeypatch.setenv("LOTTIE_MESH_CHECKPOINT", "sqlite")
    assert _resolve_checkpoint(None) == "sqlite"
    monkeypatch.delenv("LOTTIE_MESH_CHECKPOINT", raising=False)
    assert _resolve_checkpoint(None) == "memory"


def test_build_sqlite_checkpointer(tmp_path: Path) -> None:
    from lottie.mesh.checkpoint import build_checkpointer

    saver = build_checkpointer("sqlite", tmp_path)
    assert saver is not None
    assert (tmp_path / ".lottie" / "mesh").is_dir()
