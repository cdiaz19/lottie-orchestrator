"""ScaffolderAgent — turns a natural-language description into a ScaffoldPlan.

The agent only produces the validated plan via the injected LLMProvider; rendering,
the rule-13 gate, and writing live in `generator.generate_from_desc`. Tests inject
MockLLMProvider (CLAUDE.md rule 5).
"""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message
from lottie.scaffold.generator import _class_name
from lottie.scaffold.scaffolder_prompts import SYSTEM_PROMPT, build_user_prompt
from lottie.scaffold.schema import ScaffoldPlan, ScaffoldRequest


class ScaffolderAgent(BaseAgent[ScaffoldRequest, ScaffoldPlan]):
    """Generate a structured ScaffoldPlan from a description."""

    def _execute(self, data: ScaffoldRequest) -> ScaffoldPlan:
        class_name = _class_name(data.name, data.kind)
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=build_user_prompt(data, class_name)),
            ]
        )
        return _parse_plan(response.content)


def _parse_plan(content: str) -> ScaffoldPlan:
    """Parse the LLM response into a ScaffoldPlan.

    Extracts the outermost ``{...}`` object, which tolerates ```json fences,
    trailing whitespace, and prose before/after the JSON in one step. If no object
    is present the original text is validated so genuinely-bad output still fails.
    """
    text = content.strip()
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        text = text[first : last + 1]
    try:
        return ScaffoldPlan.model_validate_json(text)
    except ValueError as exc:
        raise ValueError(f"ScaffolderAgent produced an invalid plan: {exc}") from exc
