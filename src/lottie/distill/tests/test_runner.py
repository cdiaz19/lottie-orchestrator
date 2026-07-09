from collections.abc import Mapping

import pytest

from lottie.distill.runner import TemplateRunnerSkill
from lottie.distill.schema import DistilledSkillSpec, TemplateInput
from lottie.llm import LLMResponse, Message, MockLLMProvider


def _spec() -> DistilledSkillSpec:
    return DistilledSkillSpec(name="summ", template="Summarize {topic}.", slots=["topic"])


def test_runner_fills_and_completes() -> None:
    skill = TemplateRunnerSkill(MockLLMProvider(["a summary"]), _spec())
    out = skill.run(TemplateInput(slots={"topic": "otters"}))
    assert out.content == "a summary"


def test_runner_missing_slot_fails_closed() -> None:
    skill = TemplateRunnerSkill(MockLLMProvider(["x"]), _spec())
    with pytest.raises(KeyError):
        skill.run(TemplateInput(slots={}))


def test_runner_sends_filled_prompt() -> None:
    class _Capture(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__(["ok"])
            self.seen = ""

        def complete(
            self, messages: list[Message], model_params: Mapping[str, object] | None = None
        ) -> LLMResponse:
            self.seen = messages[-1].content
            return super().complete(messages, model_params)

    llm = _Capture()
    TemplateRunnerSkill(llm, _spec()).run(TemplateInput(slots={"topic": "otters"}))
    assert llm.seen == "Summarize otters."
