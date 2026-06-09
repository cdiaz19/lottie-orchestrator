from __future__ import annotations

import json
from pathlib import Path

import pytest

from lottie.llm import MockLLMProvider
from lottie.scaffold.generator import generate_from_desc

_GOOD_AGENT = {
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

_SECRET_SKILL = {
    "class_name": "LeakSkill",
    "input_fields": [{"name": "text", "type": "str", "description": "in"}],
    "output_fields": [{"name": "result", "type": "str", "description": "out"}],
    "system_prompt": "",
    "run_body": 'key = "AKIAIOSFODNN7EXAMPLE"\nreturn LeakSkillOutput(result=key)',
    "tools": [],
}


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lottie.yaml").write_text("project: demo\n", encoding="utf-8")
    (tmp_path / "agents").mkdir()
    (tmp_path / "skills").mkdir()
    return tmp_path


def test_clean_agent_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, monkeypatch)
    llm = MockLLMProvider([json.dumps(_GOOD_AGENT)])
    result = generate_from_desc("agent", "greeter", "greets people", llm)
    assert result.passed is True
    assert (root / "agents" / "greeter" / "agent.py").is_file()


def test_secret_plan_is_rejected_and_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, monkeypatch)
    # Same plan twice: initial attempt + the one retry both leak.
    llm = MockLLMProvider([json.dumps(_SECRET_SKILL), json.dumps(_SECRET_SKILL)])
    with pytest.raises(Exception):  # noqa: B017
        generate_from_desc("skill", "leak", "leaks a key", llm)
    assert not (root / "skills" / "leak").exists()
