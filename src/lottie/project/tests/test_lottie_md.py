from __future__ import annotations

from pathlib import Path

from lottie.project.lottie_md import sync

_TEMPLATE = """# demo

## Agents
_None yet._

## Skills
_None yet._
"""


def _project(tmp_path: Path) -> Path:
    (tmp_path / "lottie.yaml").write_text("project: demo\n", encoding="utf-8")
    (tmp_path / "LOTTIE.md").write_text(_TEMPLATE, encoding="utf-8")
    agent_dir = tmp_path / "agents" / "researcher"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.py").write_text("# agent\n", encoding="utf-8")
    skill_dir = tmp_path / "skills" / "web_search"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.py").write_text("# skill\n", encoding="utf-8")
    return tmp_path


def test_sync_writes_entries(tmp_path: Path) -> None:
    root = _project(tmp_path)
    sync(root)
    md = (root / "LOTTIE.md").read_text()
    assert "- **ResearcherAgent** — `agents/researcher/`" in md
    assert "- **WebSearchSkill** — `skills/web_search/`" in md
    assert "_None yet_" not in md


def test_sync_is_idempotent(tmp_path: Path) -> None:
    root = _project(tmp_path)
    sync(root)
    first = (root / "LOTTIE.md").read_text()
    sync(root)
    assert (root / "LOTTIE.md").read_text() == first


def test_sync_restores_placeholder_when_empty(tmp_path: Path) -> None:
    (tmp_path / "lottie.yaml").write_text("project: demo\n", encoding="utf-8")
    (tmp_path / "LOTTIE.md").write_text(_TEMPLATE, encoding="utf-8")
    sync(tmp_path)
    md = (tmp_path / "LOTTIE.md").read_text()
    assert md.count("_None yet_") == 2
