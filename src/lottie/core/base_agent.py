"""BaseAgent — LLM-backed, role-driven unit that reasons and decides.

Agents call skills as tools and reason via an injected `LLMProvider`. Every
`run` is auto-instrumented; token and cost usage is captured transparently as
long as LLM calls go through `self.complete`.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from lottie.core.metrics import Kind
from lottie.core.runnable import InstrumentedRunnable
from lottie.llm import LLMProvider, LLMResponse, Message
from lottie.memory.base import MemoryClient, NullMemoryClient


class BaseAgent[InputT: BaseModel, OutputT: BaseModel](InstrumentedRunnable[InputT, OutputT]):
    """Extend this for every agent. Implement only `_execute`."""

    kind: ClassVar[Kind] = "agent"

    def __init__(
        self,
        llm: LLMProvider,
        *,
        name: str | None = None,
        memory: MemoryClient | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self.llm = llm
        self.memory: MemoryClient = memory or NullMemoryClient()

    @property
    def provider(self) -> str | None:
        return self.llm.model

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        """Run an LLM completion, accumulating tokens/cost into the active run."""
        response = self.llm.complete(messages, model_params)
        if self._active_ctx is not None:
            self._active_ctx.add_usage(response.usage, response.cost_usd)
        return response

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT: ...
