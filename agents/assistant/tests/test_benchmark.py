"""Hermetic benchmark run for the assistant mesh (mock provider, no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.benchmark.runner import benchmark
from lottie.cli import app
from lottie.llm import MockLLMProvider

_DOC = """\
---
id: kb/ma
layer: global
scope: project
tags: [ai]
status: curated
last_verified: "2025-01-01"
depends_on: []
---

Multi-agent systems coordinate specialized agents via typed messages.
"""


def _seeded_provider(_name: str) -> MockLLMProvider:
    return MockLLMProvider(
        [
            "research",
            "Multi-agent systems coordinate agents.",
            "Summary.\n- a\n- b",
            "critic",
            "Looks good; add an example.",
            "FINISH",
        ]
    )


def test_benchmark_assistant_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOTTIE_EMBEDDING_MODEL", "mock/embed")
    monkeypatch.setenv("LOTTIE_VECTOR_STORE", "memory")

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    kb = demo / "knowledge" / "global"
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "ma.md").write_text(_DOC, encoding="utf-8")

    repo = Path(__file__).resolve().parents[3]
    for unit in ("assistant", "critic", "research"):
        shutil.copytree(repo / "agents" / unit, demo / "agents" / unit, dirs_exist_ok=True)
    for skill in ("retrieval", "summarizer"):
        shutil.copytree(repo / "skills" / skill, demo / "skills" / skill, dirs_exist_ok=True)

    monkeypatch.setattr("lottie.benchmark.runner.build_provider", _seeded_provider)

    report = benchmark(demo, "assistant", ["mock/x"])
    assert report.agent == "assistant"
    assert report.providers and report.providers[0].cases
    case = report.providers[0].cases[0]
    assert case.success, f"benchmark case errored: {case.error}"
    assert case.passed
