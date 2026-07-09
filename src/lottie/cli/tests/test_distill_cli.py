from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli.app import app
from lottie.llm import MockLLMProvider
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryOrigin, MemoryRecord, MemoryTier

runner = CliRunner()


def _project(tmp_path: Path) -> Path:
    (tmp_path / "lottie.yaml").write_text(
        "project: t\nproviders:\n  default: mock\n", encoding="utf-8"
    )
    unit = tmp_path / "agents" / "digest"
    unit.mkdir(parents=True)
    (unit / "agent.py").write_text("# stub\n", encoding="utf-8")
    (unit / "config.yaml").write_text(
        "provider: mock\nmemory:\n  enabled: true\n  backend: mock\n", encoding="utf-8"
    )
    return tmp_path


def _seeded_memory() -> MockMemoryClient:
    mem = MockMemoryClient()
    mem.remember(
        MemoryRecord(
            content="always cite the source", tier=MemoryTier.SEMANTIC, namespace="ns",
            origin=MemoryOrigin.REFLECTION, source_agent="digest", run_id="r1",
        )
    )
    return mem


def test_distill_writes_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        "lottie.cli.distill.build_provider",
        lambda _m: MockLLMProvider(["Answer about {topic}, always cite the source."]),
    )
    monkeypatch.setattr(
        "lottie.cli.distill.build_memory_client", lambda *_a, **_k: _seeded_memory()
    )
    result = runner.invoke(app, ["distill", "digest", "--namespace", "ns", "--skill-name", "cited"])
    assert result.exit_code == 0, result.stdout
    draft = root / "skills" / "draft" / "cited"
    assert (draft / "template.yaml").is_file()
    assert not (draft / "skill.py").exists()  # draft is not callable


def test_distill_no_notes_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setattr("lottie.cli.distill.build_provider", lambda _m: MockLLMProvider(["x"]))
    monkeypatch.setattr(
        "lottie.cli.distill.build_memory_client", lambda *_a, **_k: MockMemoryClient()
    )
    result = runner.invoke(app, ["distill", "digest", "--namespace", "ns"])
    assert result.exit_code == 1


def test_distill_secret_in_template_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setattr(
        "lottie.cli.distill.build_provider",
        lambda _m: MockLLMProvider(["use key AKIAIOSFODNN7EXAMPLE for {topic}"]),
    )
    monkeypatch.setattr(
        "lottie.cli.distill.build_memory_client", lambda *_a, **_k: _seeded_memory()
    )
    result = runner.invoke(app, ["distill", "digest", "--namespace", "ns"])
    assert result.exit_code == 2
    # security property: a secret-bearing template must NEVER land on disk
    assert not (tmp_path / "skills" / "draft" / "digest_distilled").exists()
