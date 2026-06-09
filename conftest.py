"""Shared pytest fixtures for the whole suite.

CLI tests assert on Typer/rich error output. rich wraps into a box sized to the
terminal width, and with no tty (CI) it defaults to 80 columns — which splits
multi-word phrases like "not empty" across the box border and breaks substring
assertions. Forcing a wide width makes CLI output deterministic everywhere.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "200")
