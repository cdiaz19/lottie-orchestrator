"""Shared pytest fixtures for the whole suite.

CLI tests assert on Typer/rich error output. rich wraps into a box sized to the
terminal width, and with no tty (CI) it defaults to 80 columns — which splits
multi-word phrases like "not empty" across the box border and breaks substring
assertions. Forcing a wide width makes CLI output deterministic everywhere.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture(autouse=True)
def _disable_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-off audit in tests; audit tests opt in by injecting a SqliteAuditLogger."""
    monkeypatch.setenv("LOTTIE_DISABLE_AUDIT", "1")


@pytest.fixture(autouse=True)
def _isolate_unit_import_state() -> object:
    """Undo the global import-state mutation that ``lottie run``/``init`` perform.

    ``lottie.project.discovery`` inserts a project root at ``sys.path[0]`` and
    purges cached ``agents``/``skills`` modules so each CLI invocation imports the
    right user code. Run against a scaffolded fixture project (whose ``skills/`` is
    empty), that leaves a foreign root on ``sys.path`` and the real ``skills.*``
    modules evicted — so a later test that imports a real reference unit
    (``skills.chunker``, ``agents.research`` …) resolves against the wrong root and
    errors at setup. Snapshot ``sys.path`` and drop ``agents``/``skills`` modules
    after every test so the next import re-resolves cleanly from the repo root.
    """
    path_before = list(sys.path)
    yield
    if sys.path != path_before:
        sys.path[:] = path_before
    for name in [
        m
        for m in sys.modules
        if m in {"agents", "skills"} or m.startswith(("agents.", "skills."))
    ]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_mesh_checkpoint() -> object:
    """`lottie serve --port` setdefaults LOTTIE_MESH_CHECKPOINT=sqlite in production code,
    which monkeypatch can't revert. Snapshot + restore it so a serve-invoking test can't
    flip later hermetic mesh tests onto the sqlite checkpointer (which would write/accumulate
    .lottie/mesh/checkpoints.db and cross-contaminate runs)."""
    import os

    before = os.environ.get("LOTTIE_MESH_CHECKPOINT")
    yield
    if before is None:
        os.environ.pop("LOTTIE_MESH_CHECKPOINT", None)
    else:
        os.environ["LOTTIE_MESH_CHECKPOINT"] = before
