from pathlib import Path

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
        _Echo, llm=MockLLMProvider(["x"]), root=tmp_path, config=_cfg()
    )
    assert isinstance(agent.memory, NullMemoryClient)


def test_memory_enabled_injects_sqlite(tmp_path: Path) -> None:
    agent = instantiate_agent(
        _Echo,
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
