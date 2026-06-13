from __future__ import annotations

from agents.publisher.agent import PublisherAgent
from agents.publisher.schema import PublisherInput
from lottie.llm import MockLLMProvider


def test_publisher_finalizes_text() -> None:
    agent = PublisherAgent(MockLLMProvider(["Published: multi-agent overview."]))
    out = agent.run(PublisherInput(text="A reviewed draft about multi-agent systems."))
    assert out.published
    assert agent.last_metrics is not None
