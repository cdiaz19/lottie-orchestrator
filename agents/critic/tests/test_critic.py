from __future__ import annotations

from agents.critic.agent import CriticAgent
from agents.critic.schema import CriticInput
from lottie.llm import MockLLMProvider


def test_critic_reviews_text() -> None:
    agent = CriticAgent(MockLLMProvider(["Concise and accurate; tighten the intro."]))
    out = agent.run(CriticInput(text="A long draft about multi-agent systems."))
    assert out.review
    assert agent.last_metrics is not None
