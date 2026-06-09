from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from lottie.cli import app
from lottie.llm import MockLLMProvider

runner = CliRunner()


def _init_demo(tmp_path: Path) -> Path:
    result = runner.invoke(app, ["init", "demo"])
    assert result.exit_code == 0, result.output
    return tmp_path / "demo"


def test_create_agent_scaffolds_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    result = runner.invoke(app, ["create", "agent", "web_search"])
    assert result.exit_code == 0, result.output

    base = demo / "agents" / "web_search"
    for rel in [
        "__init__.py",
        "AGENT.md",
        "agent.py",
        "schema.py",
        "config.yaml",
        "prompts.py",
        "tests/__init__.py",
        "tests/test_web_search.py",
    ]:
        assert (base / rel).is_file(), f"missing {rel}"
    assert "class WebSearchAgent(BaseAgent" in (base / "agent.py").read_text()
    assert "Created agent at" in result.output
    assert "Next:" in result.output


def test_create_skill_scaffolds_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    result = runner.invoke(app, ["create", "skill", "cleaner"])
    assert result.exit_code == 0, result.output

    base = demo / "skills" / "cleaner"
    for rel in [
        "__init__.py",
        "SKILL.md",
        "skill.py",
        "schema.py",
        "tests/__init__.py",
        "tests/test_cleaner.py",
    ]:
        assert (base / rel).is_file(), f"missing {rel}"
    assert "class CleanerSkill(BaseSkill" in (base / "skill.py").read_text()


def test_create_refuses_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no lottie.yaml here
    result = runner.invoke(app, ["create", "agent", "foo"])
    assert result.exit_code != 0
    assert not (tmp_path / "agents").exists()


def test_create_refuses_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "dup"]).exit_code == 0
    result = runner.invoke(app, ["create", "agent", "dup"])
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "not empty" in output


def test_create_refuses_when_name_matches_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    (demo / "agents" / "afile").write_text("keep")
    result = runner.invoke(app, ["create", "agent", "afile"])
    assert result.exit_code != 0
    assert (demo / "agents" / "afile").read_text() == "keep"


@pytest.mark.parametrize(
    "bad", ["Web", "web-search", "a/b", "", ".", "..", "../x", "1foo", "class"]
)
def test_create_rejects_bad_names(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    result = runner.invoke(app, ["create", "agent", bad])
    assert result.exit_code != 0


def test_create_updates_lottie_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0

    md = (demo / "LOTTIE.md").read_text()
    agents_section, skills_section = md.split("## Skills")
    # Agent registered under ## Agents, placeholder replaced.
    assert "- **ResearcherAgent** — `agents/researcher/`" in agents_section
    assert "_None yet" not in agents_section
    # Skill registered under ## Skills, placeholder replaced.
    assert "- **CleanerSkill** — `skills/cleaner/`" in skills_section
    assert "_None yet" not in skills_section


def test_generated_python_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "cleaner"]).exit_code == 0

    py_files = list((demo / "agents").rglob("*.py")) + list((demo / "skills").rglob("*.py"))
    assert py_files, "no generated .py files found"
    for f in py_files:
        py_compile.compile(str(f), doraise=True)


def test_generated_project_tests_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "researcher"]).exit_code == 0
    assert runner.invoke(app, ["create", "skill", "web_search"]).exit_code == 0

    # The generated agent (3) + skill (3) tests must pass out of the box.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(demo)],
        cwd=demo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


_DESC_AGENT_PLAN = {
    "class_name": "GreeterAgent",
    "input_fields": [{"name": "who", "type": "str", "description": "name"}],
    "output_fields": [{"name": "greeting", "type": "str", "description": "msg"}],
    "system_prompt": "Greet warmly.",
    "run_body": (
        "from lottie.llm import Message\n"
        "response = self.complete([Message(role='system', content=SYSTEM_PROMPT),"
        " Message(role='user', content=data.who)])\n"
        "return GreeterAgentOutput(greeting=response.content)"
    ),
    "tools": [],
}


def test_create_agent_from_desc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    fake_llm = MockLLMProvider([json.dumps(_DESC_AGENT_PLAN)])
    with patch("lottie.cli.create.build_provider", return_value=fake_llm):
        result = runner.invoke(
            app, ["create", "agent", "greeter", "--from-desc", "greets people"]
        )
    assert result.exit_code == 0, result.output
    agent_py = (demo / "agents" / "greeter" / "agent.py").read_text()
    assert "class GreeterAgent(BaseAgent" in agent_py
    assert "who: str" in (demo / "agents" / "greeter" / "schema.py").read_text()


_DESC_SKILL_PLAN = {
    "class_name": "ShoutSkill",
    "input_fields": [{"name": "text", "type": "str", "description": "in"}],
    "output_fields": [{"name": "result", "type": "str", "description": "out"}],
    "system_prompt": "",
    "run_body": "return ShoutSkillOutput(result=data.text.upper())",
    "tools": [],
}


def test_create_skill_from_desc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    demo = _init_demo(tmp_path)
    monkeypatch.chdir(demo)
    fake_llm = MockLLMProvider([json.dumps(_DESC_SKILL_PLAN)])
    with patch("lottie.cli.create.build_provider", return_value=fake_llm):
        result = runner.invoke(
            app, ["create", "skill", "shout", "--from-desc", "uppercases text"]
        )
    assert result.exit_code == 0, result.output
    skill_py = (demo / "skills" / "shout" / "skill.py").read_text()
    assert "class ShoutSkill(BaseSkill" in skill_py
    assert "text: str" in (demo / "skills" / "shout" / "schema.py").read_text()
