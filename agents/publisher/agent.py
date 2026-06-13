"""PublisherAgent — single-LLM-call finalizer used as a mesh worker."""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message

from .prompts import SYSTEM_PROMPT
from .schema import PublisherInput, PublisherOutput


class PublisherAgent(BaseAgent[PublisherInput, PublisherOutput]):
    """Publishes/finalizes a draft and returns the release-ready text."""

    def _execute(self, data: PublisherInput) -> PublisherOutput:
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=data.text),
            ]
        )
        return PublisherOutput(published=response.content)
