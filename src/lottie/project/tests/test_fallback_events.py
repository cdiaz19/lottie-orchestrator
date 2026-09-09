"""A provider fallback reaches the event bus (E5 follow-up).

E5 shipped with a stated deviation: its design said a fallback would emit on the runtime
event bus, and it did not, because the provider is constructed before the agent that owns
the bus. E7 made the bus a PUBLIC extension point, which made the gap worth closing — a
telemetry plugin should be able to alert on "we fell back to the secondary model".
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from lottie.core.base_agent import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.llm.base import LLMProvider, LLMResponse, Message
from lottie.llm.routing import RoutedProvider
from lottie.project.config import AgentConfig
from lottie.project.discovery import instantiate_agent
from lottie.runtime.events import ProviderFallback, RunEvent


class RateLimitError(Exception):
    """Named after litellm's taxonomy — `is_transient` classifies by name."""


class _In(BaseModel):
    task: str


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(answer=self.complete([Message(role="user", content=data.task)]).content)


class _Failing(LLMProvider):
    @property
    def model(self) -> str:
        return "primary/model"

    def complete(
        self, messages: list[Message], model_params: Mapping[str, object] | None = None
    ) -> LLMResponse:
        raise RateLimitError("429")



class _Collector:
    name = "collector"

    def __init__(self) -> None:
        self.seen: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.seen.append(event)


def _agent(
    tmp_path: Path, llm: LLMProvider, *plugins: object
) -> _Agent:
    agent = instantiate_agent(
        _Agent,  # type: ignore[arg-type]
        llm=llm,
        root=tmp_path,
        config=AgentConfig.model_validate({"provider": "mock/sim"}),
        enable_benchmarks=False,
    )
    if plugins:
        agent.set_plugins(list(plugins))  # type: ignore[arg-type]
    return cast(_Agent, agent)


def _routed() -> RoutedProvider:
    return RoutedProvider([_Failing(), MockLLMProvider(responses=["from the fallback"])])


class TestFallbackReachesTheBus:
    def test_a_fallback_emits_an_event(self, tmp_path: Path) -> None:
        collector = _Collector()
        agent = _agent(tmp_path, _routed(), collector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.run(_In(task="hi"))
        assert any(isinstance(e, ProviderFallback) for e in collector.seen)

    def test_the_event_names_both_models(self, tmp_path: Path) -> None:
        collector = _Collector()
        agent = _agent(tmp_path, _routed(), collector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.run(_In(task="hi"))
        event = next(e for e in collector.seen if isinstance(e, ProviderFallback))
        assert event.failed_model == "primary/model"
        assert event.fallback_model.startswith("mock")

    def test_the_reason_is_the_exception_TYPE_not_its_message(self, tmp_path: Path) -> None:
        """An error string can carry a prompt fragment or a key.

        Events are hash-and-scalar only, so the reason names the exception class rather
        than repeating whatever the provider put in the message.
        """
        collector = _Collector()
        agent = _agent(tmp_path, _routed(), collector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent.run(_In(task="hi"))
        event = next(e for e in collector.seen if isinstance(e, ProviderFallback))
        assert event.reason == "RateLimitError" and "429" not in event.model_dump_json()

    def test_the_run_still_succeeds_on_the_fallback(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, _routed())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert agent.run(_In(task="hi")).answer == "from the fallback"

    def test_a_fallback_still_warns_as_well(self, tmp_path: Path) -> None:
        # Two traces, not one: the event for machines, the warning for whoever is
        # watching the terminal. A silent fallback is the dangerous kind.
        agent = _agent(tmp_path, _routed())
        with pytest.warns(UserWarning, match="falling back"):
            agent.run(_In(task="hi"))


class TestNoFallbackConfigured:
    def test_a_plain_provider_emits_no_fallback_event(self, tmp_path: Path) -> None:
        collector = _Collector()
        agent = _agent(tmp_path, MockLLMProvider(responses=["fine"]), collector)
        agent.run(_In(task="hi"))
        assert not any(isinstance(e, ProviderFallback) for e in collector.seen)

    def test_a_plain_provider_is_not_wired_at_all(self, tmp_path: Path) -> None:
        # A project with no fallback should pay nothing for the feature.
        agent = _agent(tmp_path, MockLLMProvider(responses=["fine"]))
        assert agent.run(_In(task="hi")).answer == "fine"


class TestStableBus:
    def test_the_bus_is_reused_across_runs(self, tmp_path: Path) -> None:
        """The change that made this possible: one bus per AGENT, not per run.

        The subscriber set is identical on every run, so this is behaviourally the same —
        but a per-run bus gave the provider nothing stable to emit onto.
        """
        agent = _agent(tmp_path, MockLLMProvider(responses=["a", "b"]))
        agent.run(_In(task="one"))
        first = agent.event_bus()
        agent.run(_In(task="two"))
        assert agent.event_bus() is first

    def test_plugins_still_receive_on_every_run(self, tmp_path: Path) -> None:
        collector = _Collector()
        agent = _agent(tmp_path, MockLLMProvider(responses=["a", "b"]), collector)
        agent.run(_In(task="one"))
        agent.run(_In(task="two"))
        assert len(collector.seen) == 4  # started + completed, twice
