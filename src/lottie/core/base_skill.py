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
from lottie.governance.capability import active_capability_gate


class BaseSkill[InputT: BaseModel, OutputT: BaseModel](InstrumentedRunnable[InputT, OutputT]):
    """Extend this for every skill. Implement only `_execute`."""

    kind: ClassVar[Kind] = "skill"

    #: Capability name matched against an agent's declared `capabilities` (rule 11).
    #: Defaults to the class name minus a trailing "Skill", lowercased
    #: (`RetrievalSkill` -> "retrieval"). Override to set an explicit name.
    capability_name: ClassVar[str | None] = None

    @classmethod
    def resolved_capability_name(cls) -> str:
        """The name checked against the calling agent's declared capabilities."""
        if cls.capability_name is not None:
            return cls.capability_name
        name = cls.__name__
        if name.endswith("Skill"):
            name = name[: -len("Skill")]
        return name.lower()

    def run(self, data: InputT) -> OutputT:
        """Enforce rule 11 (fail-closed), then run.

        Reads the capability gate the calling agent activated around its `_execute`
        window. Default `NullCapabilityGate` (no active agent / undeclared caps) allows
        all, so direct skill construction and unit tests are unenforced.
        """
        active_capability_gate().check(self.resolved_capability_name())
        return super().run(data)

    @abstractmethod
    def _execute(self, data: InputT) -> OutputT: ...
