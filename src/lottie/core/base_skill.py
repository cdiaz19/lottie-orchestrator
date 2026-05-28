"""BaseSkill — stateless, deterministic, typed capability.

Skills execute decisions made by agents. No LLM is required (though one may be
used internally). Same input always produces predictable output, so they are unit-testable
without mocks.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from lottie.core.metrics import Kind
from lottie.core.runnable import InstrumentedRunnable


class BaseSkill[InputT: BaseModel, OutputT: BaseModel](InstrumentedRunnable[InputT, OutputT]):
    """Extend this for every skill. Implement only `_execute`."""

    kind: ClassVar[Kind] = "skill"

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT: ...
