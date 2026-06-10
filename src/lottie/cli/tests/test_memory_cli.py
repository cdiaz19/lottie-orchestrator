"""Tests for `lottie memory` CLI sub-commands (knowledge-graph queries).

These commands query the *knowledge dependency graph* — distinct from the
runtime agent memory subsystem in ``lottie.memory``.  All tests use
``--root <tmp_path>`` so they are fully isolated from the real project tree.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(
    root: Path,
    layer: str,
    doc_id: str,
    depends_on: tuple[str, ...] = (),
    last_verified: str = "2026-05-01",
) -> None:
    """Write a minimal knowledge doc with YAML frontmatter under ``root``."""
    d = root / "knowledge" / layer
    d.mkdir(parents=True, exist_ok=True)
    deps_yaml = "[" + ", ".join(depends_on) + "]"
    fm = (
        f"---\n"
        f"id: {doc_id}\n"
        f"layer: {layer}\n"
        f"status: curated\n"
        f"last_verified: {last_verified}\n"
        f"depends_on: {deps_yaml}\n"
        f"tags: []\n"
        f"---\n"
        f"body for {doc_id}\n"
    )
    filename = doc_id.replace("/", "_") + ".md"
    (d / filename).write_text(fm, encoding="utf-8")


def _memory(root: Path, *args: str) -> Any:
    return runner.invoke(app, ["memory", *args, "--root", str(root)])


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


def test_graph_empty_root_prints_empty_message(tmp_path: Path) -> None:
    """graph on an empty tree prints 'Knowledge graph is empty.'"""
    result = _memory(tmp_path, "graph")
    assert result.exit_code == 0, result.output
    assert "knowledge graph is empty" in result.output.lower()


def test_graph_shows_edges_and_counts(tmp_path: Path) -> None:
    """graph lists edges and a node/edge summary."""
    _write(tmp_path, "global", "doc/a")
    _write(tmp_path, "global", "doc/b", depends_on=("doc/a",))
    _write(tmp_path, "global", "doc/c", depends_on=("doc/b",))

    result = _memory(tmp_path, "graph")
    assert result.exit_code == 0, result.output
    # Edges doc/a -> doc/b and doc/b -> doc/c should appear
    assert "doc/a" in result.output
    assert "doc/b" in result.output
    # Summary counts
    output_lower = result.output.lower()
    assert "node" in output_lower
    assert "edge" in output_lower


# ---------------------------------------------------------------------------
# impact
# ---------------------------------------------------------------------------


def test_impact_chain(tmp_path: Path) -> None:
    """impact on the root of a chain returns all downstream dependents."""
    _write(tmp_path, "global", "doc/a")
    _write(tmp_path, "global", "doc/b", depends_on=("doc/a",))
    _write(tmp_path, "global", "doc/c", depends_on=("doc/b",))

    result = _memory(tmp_path, "impact", "doc/a")
    assert result.exit_code == 0, result.output
    assert "doc/b" in result.output
    assert "doc/c" in result.output


def test_impact_leaf_shows_nothing_depends(tmp_path: Path) -> None:
    """impact on a leaf node (nothing depends on it) prints 'Nothing depends on'."""
    _write(tmp_path, "global", "doc/a")
    _write(tmp_path, "global", "doc/b", depends_on=("doc/a",))

    result = _memory(tmp_path, "impact", "doc/b")
    assert result.exit_code == 0, result.output
    assert "nothing depends on" in result.output.lower()


def test_impact_unknown_id_exit_zero(tmp_path: Path) -> None:
    """impact with an unknown id exits 0 and prints a 'no document' message."""
    _write(tmp_path, "global", "doc/a")

    result = _memory(tmp_path, "impact", "nope/x")
    assert result.exit_code == 0, result.output
    assert "no document" in result.output.lower() or "nope/x" in result.output


# ---------------------------------------------------------------------------
# audit — no cycle
# ---------------------------------------------------------------------------


def test_audit_acyclic_exits_zero(tmp_path: Path) -> None:
    """audit on an acyclic graph exits 0 and reports no cycles."""
    _write(tmp_path, "global", "doc/a")
    _write(tmp_path, "global", "doc/b", depends_on=("doc/a",))

    result = _memory(tmp_path, "audit")
    assert result.exit_code == 0, result.output
    assert "no cycle" in result.output.lower()


def test_audit_reports_orphans(tmp_path: Path) -> None:
    """audit identifies orphan nodes (degree 0)."""
    _write(tmp_path, "global", "orphan/x")  # no edges

    result = _memory(tmp_path, "audit")
    assert result.exit_code == 0, result.output
    assert "orphan" in result.output.lower()


# ---------------------------------------------------------------------------
# audit — with cycle
# ---------------------------------------------------------------------------


def test_audit_cycle_exits_nonzero(tmp_path: Path) -> None:
    """audit with a cyclic graph exits non-zero."""
    _write(tmp_path, "global", "doc/x", depends_on=("doc/y",))
    _write(tmp_path, "global", "doc/y", depends_on=("doc/x",))

    result = _memory(tmp_path, "audit")
    assert result.exit_code != 0, result.output


def test_audit_cycle_output_mentions_cycle(tmp_path: Path) -> None:
    """audit with a cycle prints a message mentioning the cycle."""
    _write(tmp_path, "global", "doc/x", depends_on=("doc/y",))
    _write(tmp_path, "global", "doc/y", depends_on=("doc/x",))

    result = _memory(tmp_path, "audit")
    assert "cycle" in result.output.lower()


# ---------------------------------------------------------------------------
# audit — stale
# ---------------------------------------------------------------------------


def test_audit_stale_doc_reported(tmp_path: Path) -> None:
    """audit reports a doc with last_verified in 2020 as stale (always > 90 days ago).

    The fresh doc uses today's date so it is always 0 days old and never stale,
    regardless of when the suite runs.
    """
    today = datetime.now(UTC).date().isoformat()
    _write(tmp_path, "global", "doc/old", last_verified="2020-01-01")
    _write(tmp_path, "global", "doc/fresh", last_verified=today)

    result = _memory(tmp_path, "audit")
    assert result.exit_code == 0, result.output
    assert "doc/old" in result.output
    assert "stale" in result.output.lower()
    # doc/fresh must NOT appear in the stale section (it may appear as an orphan)
    stale_section = result.output.lower().split("stale", maxsplit=1)[-1]
    assert "doc/fresh" not in stale_section


def test_audit_stale_days_custom(tmp_path: Path) -> None:
    """audit --stale-days 1 flags a doc from 2020 as stale."""
    _write(tmp_path, "global", "doc/old", last_verified="2020-01-01")

    result = _memory(tmp_path, "audit", "--stale-days", "1")
    assert result.exit_code == 0, result.output
    assert "doc/old" in result.output
