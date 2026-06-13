"""CriticAgent — single-LLM-call reviewer used as a mesh worker."""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message

from .prompts import SYSTEM_PROMPT
from .schema import CriticInput, CriticOutput


class CriticAgent(BaseAgent[CriticInput, CriticOutput]):
    """Reviews a draft and returns a terse critique."""

    def _execute(self, data: CriticInput) -> CriticOutput:
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=data.text),
            ]
        )
        return CriticOutput(review=response.content)
