"""TemplateRunnerSkill — the single generic executor for distilled template skills.

Fills a DistilledSkillSpec's template from typed slot inputs and calls the LLM. No
LLM-authored code runs (D2): every distilled skill is data (a template) executed here.
"""

from __future__ import annotations

from lottie.core import BaseSkill
from lottie.distill.generate import fill_template
from lottie.distill.schema import DistilledSkillSpec, TemplateInput, TemplateOutput
from lottie.llm import LLMProvider, Message


class TemplateRunnerSkill(BaseSkill[TemplateInput, TemplateOutput]):
    """Execute a distilled template: fill slots, complete, return the text."""

    capability_name = "template_runner"

    def __init__(
        self, llm: LLMProvider, spec: DistilledSkillSpec, *, name: str | None = None
    ) -> None:
        super().__init__(name=name)
        self._llm = llm
        self._spec = spec

    def _execute(self, data: TemplateInput) -> TemplateOutput:
        prompt = fill_template(self._spec.template, data.slots)
        response = self._llm.complete([Message(role="user", content=prompt)])
        return TemplateOutput(content=response.content)
