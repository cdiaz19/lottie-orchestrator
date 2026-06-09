from __future__ import annotations

import json

import pytest

from lottie.llm import MockLLMProvider
from lottie.scaffold.scaffolder_agent import ScaffolderAgent
from lottie.scaffold.schema import ScaffoldRequest

_PLAN = {
    "class_name": "GreeterAgent",
    "input_fields": [{"name": "who", "type": "str", "description": "name"}],
    "output_fields": [{"name": "greeting", "type": "str", "description": "msg"}],
    "system_prompt": "Greet warmly.",
    "run_body": 'return GreeterAgentOutput(greeting="hi")',
    "tools": [],
}


def test_returns_validated_plan() -> None:
    agent = ScaffolderAgent(llm=MockLLMProvider([json.dumps(_PLAN)]))
    plan = agent.run(ScaffoldRequest(kind="agent", name="greeter", description="greets"))
    assert plan.class_name == "GreeterAgent"
    assert plan.input_fields[0].name == "who"


def test_strips_code_fences() -> None:
    fenced = "```json\n" + json.dumps(_PLAN) + "\n```"
    agent = ScaffolderAgent(llm=MockLLMProvider([fenced]))
    plan = agent.run(ScaffoldRequest(kind="agent", name="greeter", description="greets"))
    assert plan.class_name == "GreeterAgent"


def test_repair_feedback_appears_in_prompt() -> None:
    mock = MockLLMProvider([json.dumps(_PLAN)])
    agent = ScaffolderAgent(llm=mock)
    agent.run(
        ScaffoldRequest(
            kind="agent", name="greeter", description="greets", repair_feedback="mypy: bad type"
        )
    )
    sent = mock.calls[0][-1].content
    assert "mypy: bad type" in sent


def test_invalid_json_raises() -> None:
    agent = ScaffolderAgent(llm=MockLLMProvider(["not json at all"]))
    with pytest.raises(ValueError, match="plan"):
        agent.run(ScaffoldRequest(kind="agent", name="greeter", description="greets"))
