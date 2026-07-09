import warnings
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from lottie.core import BaseAgent
from lottie.llm import MockLLMProvider
from lottie.memory.base import NullMemoryClient
from lottie.memory.store import SqliteMemoryClient
from lottie.project.config import AgentConfig
from lottie.project.discovery import instantiate_agent


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    text: str


class _Echo(BaseAgent[_Input, _Output]):
    def _execute(self, data: _Input) -> _Output:
        return _Output(text=data.text)


def _cfg(**kw: object) -> AgentConfig:
    return AgentConfig.model_validate({"provider": "mock", **kw})


def test_memory_disabled_keeps_null_client(tmp_path: Path) -> None:
    agent = instantiate_agent(
        cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(),
    )
    assert isinstance(agent.memory, NullMemoryClient)


def test_memory_enabled_injects_sqlite(tmp_path: Path) -> None:
    agent = instantiate_agent(
        cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": True, "backend": "sqlite"}),
    )
    assert isinstance(agent.memory, SqliteMemoryClient)


def test_set_memory_setter() -> None:
    agent = _Echo(llm=MockLLMProvider(["x"]))
    store = SqliteMemoryClient(Path("/tmp/does-not-persist-here.db"))
    agent.set_memory(store)
    assert agent.memory is store


def test_recall_wired_when_enabled(tmp_path: Path) -> None:
    agent = instantiate_agent(
        cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": True, "recall": {"enabled": True, "limit": 2}}),
    )
    assert agent._recall_enabled is True
    assert agent._recall_limit == 2
    assert agent._recall_namespace == agent.name  # defaulted to agent name


def test_recall_namespace_explicit(tmp_path: Path) -> None:
    agent = instantiate_agent(
        cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": True, "namespace": "lessons", "recall": {"enabled": True}}),
    )
    assert agent._recall_namespace == "lessons"


def test_recall_off_when_memory_disabled(tmp_path: Path) -> None:
    agent = instantiate_agent(
        cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": False, "recall": {"enabled": True}}),
    )
    assert agent._recall_enabled is False


def test_reflect_wired_when_enabled(tmp_path: Path) -> None:
    agent = instantiate_agent(
        cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": True, "reflect": {"enabled": True}}),
    )
    assert agent._reflect_enabled is True
    assert agent._reflect_namespace == agent.name


def test_reflect_off_when_memory_disabled(tmp_path: Path) -> None:
    agent = instantiate_agent(
        cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
        llm=MockLLMProvider(["x"]),
        root=tmp_path,
        config=_cfg(memory={"enabled": False, "reflect": {"enabled": True}}),
    )
    assert agent._reflect_enabled is False


def test_reflect_without_token_cap_warns(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instantiate_agent(
            cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
            llm=MockLLMProvider(["x"]),
            root=tmp_path,
            config=_cfg(memory={"enabled": True, "reflect": {"enabled": True}}),
        )
    assert any("max_run_tokens" in str(w.message) for w in caught)


def test_reflect_with_token_cap_no_warn(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instantiate_agent(
            cast(type[BaseAgent[BaseModel, BaseModel]], _Echo),
            llm=MockLLMProvider(["x"]),
            root=tmp_path,
            config=_cfg(
                max_run_tokens=1000, memory={"enabled": True, "reflect": {"enabled": True}}
            ),
        )
    assert not any("max_run_tokens" in str(w.message) for w in caught)
