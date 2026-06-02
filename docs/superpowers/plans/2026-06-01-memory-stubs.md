# Memory Stubs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the memory subsystem's interface, schemas, and test doubles (the memory analogue of `LLMProvider`/`MockLLMProvider`), plus a `self.memory` injection point on `BaseAgent` — no real persistence.

**Architecture:** New `src/lottie/memory/` package mirroring `src/lottie/llm/`: `schema.py` (Pydantic models), `base.py` (`MemoryClient` ABC + `NullMemoryClient` + exceptions, depends only on schema), `mock.py` (`MockMemoryClient` in-memory double), `agent.py` (`MemoryAgent` BaseAgent subclass + `MockMemoryAgent`). `BaseAgent` gains an optional `memory` param defaulting to `NullMemoryClient`. Import direction is one-way (`core` → `memory.base` → `memory.schema`; `memory.agent` → `core`) to avoid a cycle.

**Tech Stack:** Python 3.12, Pydantic v2, pytest. `mypy --strict` and `ruff` must stay clean (no `Any` without justification). Run all tools via `uv run` from the project dir `/Users/cdiaz19/Documents/trae_projects/lottie-orchestrator`. TDD throughout.

**Reference spec:** `docs/superpowers/specs/2026-06-01-memory-stubs-design.md`

---

## File Structure

- `src/lottie/memory/__init__.py` — **create**: public exports; imports `schema`/`base`/`mock` before `agent`.
- `src/lottie/memory/schema.py` — **create**: `MemoryTier`, `MemoryRecord`, `MemoryQuery`, `MemoryHit`, `RecallResult`, `ReflectionInput`, `ReflectionResult`.
- `src/lottie/memory/base.py` — **create**: `MemoryError`, `MemoryNotConfiguredError`, `MemoryClient` ABC, `NullMemoryClient`.
- `src/lottie/memory/mock.py` — **create**: `MockMemoryClient`.
- `src/lottie/memory/agent.py` — **create**: `REFLECT_SYSTEM_PROMPT`, `MemoryAgent`, `MockMemoryAgent`.
- `src/lottie/memory/tests/__init__.py` — **create**: empty.
- `src/lottie/memory/tests/test_schema.py` — **create**.
- `src/lottie/memory/tests/test_mock_client.py` — **create**.
- `src/lottie/memory/tests/test_memory_agent.py` — **create**.
- `src/lottie/core/base_agent.py` — **modify**: add `memory` param + `self.memory`.
- `src/lottie/core/tests/test_base_agent.py` — **modify**: add memory-injection tests.

Pattern references the implementer should read first: `src/lottie/llm/base.py` (ABC + Pydantic shape), `src/lottie/llm/mock.py` (test-double shape with a `.calls`/`.records` capture list), `src/lottie/llm/__init__.py` (export style), `src/lottie/core/base_agent.py` (the class being extended).

---

## Task 1: Memory schemas

**Files:**
- Create: `src/lottie/memory/__init__.py` (temporary minimal export, expanded in later tasks), `src/lottie/memory/schema.py`, `src/lottie/memory/tests/__init__.py`, `src/lottie/memory/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/__init__.py` as an empty file. Then create `src/lottie/memory/tests/test_schema.py`:

```python
from __future__ import annotations

from lottie.memory.schema import (
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    MemoryTier,
    RecallResult,
    ReflectionInput,
    ReflectionResult,
)


def test_tier_values() -> None:
    assert MemoryTier.WORKING.value == "working"
    assert MemoryTier.EPISODIC.value == "episodic"
    assert MemoryTier.SEMANTIC.value == "semantic"
    assert MemoryTier.PROCEDURAL.value == "procedural"


def test_record_defaults() -> None:
    rec = MemoryRecord(content="hello", namespace="demo")
    assert rec.tier is MemoryTier.EPISODIC
    assert rec.tags == []
    assert rec.metadata == {}
    assert rec.memory_id is None


def test_record_defaults_are_independent() -> None:
    a = MemoryRecord(content="a", namespace="demo")
    b = MemoryRecord(content="b", namespace="demo")
    a.tags.append("x")
    a.metadata["k"] = "v"
    assert b.tags == []
    assert b.metadata == {}


def test_query_defaults() -> None:
    q = MemoryQuery(text="find", namespace="demo")
    assert q.tier is None
    assert q.tags == []
    assert q.limit == 10


def test_hit_and_recall_result() -> None:
    rec = MemoryRecord(content="hello", namespace="demo")
    hit = MemoryHit(record=rec, score=1.0)
    result = RecallResult(hits=[hit])
    assert result.hits[0].record.content == "hello"
    assert result.hits[0].score == 1.0
    assert RecallResult().hits == []


def test_reflection_models() -> None:
    assert ReflectionInput(namespace="demo").limit == 50
    out = ReflectionResult()
    assert out.notes == []
    assert out.consolidated_count == 0
    assert out.written_ids == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.memory'`.

- [ ] **Step 3: Write the schemas**

Create `src/lottie/memory/schema.py`:

```python
"""Pydantic models for the memory subsystem.

Pure data shapes — no logic, no imports beyond pydantic/stdlib. `base.py`,
`mock.py`, and `agent.py` all build on these.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class MemoryTier(str, Enum):
    """The four memory tiers (T0–T3)."""

    WORKING = "working"        # T0 — in-context, not persisted
    EPISODIC = "episodic"      # T1 — append-only event log
    SEMANTIC = "semantic"      # T2 — consolidated knowledge
    PROCEDURAL = "procedural"  # T3 — config/rules


class MemoryRecord(BaseModel):
    """A single stored memory."""

    content: str
    tier: MemoryTier = MemoryTier.EPISODIC
    namespace: str
    tags: list[str] = []
    metadata: dict[str, str] = {}
    memory_id: str | None = None  # assigned by MemoryClient.remember


class MemoryQuery(BaseModel):
    """A retrieval request against a namespace."""

    text: str
    namespace: str
    tier: MemoryTier | None = None  # None = any tier
    tags: list[str] = []            # match-any
    limit: int = 10


class MemoryHit(BaseModel):
    """One recalled record with its relevance score."""

    record: MemoryRecord
    score: float


class RecallResult(BaseModel):
    """Ordered hits for a query."""

    hits: list[MemoryHit] = []


class ReflectionInput(BaseModel):
    """Input to MemoryAgent: consolidate recent episodic memory."""

    namespace: str
    limit: int = 50


class ReflectionResult(BaseModel):
    """Output of MemoryAgent consolidation."""

    notes: list[str] = []
    consolidated_count: int = 0
    written_ids: list[str] = []
```

Note: Pydantic v2 deep-copies mutable field defaults per instance, so the bare
`[]` / `{}` defaults are safe (proven by `test_record_defaults_are_independent`).

Create `src/lottie/memory/__init__.py` with the schema exports only for now (later tasks extend it):

```python
from lottie.memory.schema import (
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    MemoryTier,
    RecallResult,
    ReflectionInput,
    ReflectionResult,
)

__all__ = [
    "MemoryHit",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryTier",
    "RecallResult",
    "ReflectionInput",
    "ReflectionResult",
]
```

- [ ] **Step 4: Run tests + type-check + lint**

Run: `uv run pytest src/lottie/memory/tests/test_schema.py -v` → all pass.
Run: `uv run mypy --strict src/lottie/memory/schema.py` → `Success`.
Run: `uv run ruff check src/lottie/memory` → `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/memory/__init__.py src/lottie/memory/schema.py src/lottie/memory/tests/__init__.py src/lottie/memory/tests/test_schema.py
git commit -m "feat(memory): add memory schemas"
```

---

## Task 2: MemoryClient ABC, NullMemoryClient, MockMemoryClient

**Files:**
- Create: `src/lottie/memory/base.py`, `src/lottie/memory/mock.py`, `src/lottie/memory/tests/test_mock_client.py`
- Modify: `src/lottie/memory/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/test_mock_client.py`:

```python
from __future__ import annotations

import pytest

from lottie.memory.base import MemoryNotConfiguredError, NullMemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryQuery, MemoryRecord, MemoryTier


def _rec(
    content: str,
    *,
    namespace: str = "demo",
    tier: MemoryTier = MemoryTier.EPISODIC,
    tags: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        content=content, namespace=namespace, tier=tier, tags=tags or []
    )


def test_remember_assigns_id_and_sets_field() -> None:
    client = MockMemoryClient()
    mid = client.remember(_rec("hello"))
    assert mid == "demo-0"
    assert client.records[0].memory_id == "demo-0"
    assert client.remember(_rec("again")) == "demo-1"


def test_recall_substring_match() -> None:
    client = MockMemoryClient()
    client.remember(_rec("the cat sat"))
    client.remember(_rec("a dog barked"))
    hits = client.recall(MemoryQuery(text="cat", namespace="demo")).hits
    assert len(hits) == 1
    assert hits[0].record.content == "the cat sat"
    assert hits[0].score == 1.0


def test_recall_empty_text_matches_all() -> None:
    client = MockMemoryClient()
    client.remember(_rec("one"))
    client.remember(_rec("two"))
    hits = client.recall(MemoryQuery(text="", namespace="demo")).hits
    assert len(hits) == 2


def test_recall_filters_namespace_tier_tags() -> None:
    client = MockMemoryClient()
    client.remember(_rec("keep", namespace="a", tier=MemoryTier.EPISODIC, tags=["x"]))
    client.remember(_rec("other ns", namespace="b"))
    client.remember(_rec("wrong tier", namespace="a", tier=MemoryTier.SEMANTIC))
    client.remember(_rec("wrong tag", namespace="a", tags=["y"]))
    q = MemoryQuery(text="", namespace="a", tier=MemoryTier.EPISODIC, tags=["x"])
    hits = client.recall(q).hits
    assert [h.record.content for h in hits] == ["keep"]


def test_recall_limit_truncates() -> None:
    client = MockMemoryClient()
    for i in range(5):
        client.remember(_rec(f"item {i}"))
    hits = client.recall(MemoryQuery(text="", namespace="demo", limit=2)).hits
    assert len(hits) == 2


def test_forget_returns_true_then_false() -> None:
    client = MockMemoryClient()
    mid = client.remember(_rec("bye"))
    assert client.forget(mid) is True
    assert client.forget(mid) is False
    assert client.records == []


def test_seeded_records_get_ids() -> None:
    client = MockMemoryClient(records=[_rec("seed")])
    assert client.records[0].memory_id == "demo-0"


def test_null_client_raises_on_all_ops() -> None:
    client = NullMemoryClient()
    with pytest.raises(MemoryNotConfiguredError):
        client.remember(_rec("x"))
    with pytest.raises(MemoryNotConfiguredError):
        client.recall(MemoryQuery(text="", namespace="demo"))
    with pytest.raises(MemoryNotConfiguredError):
        client.forget("demo-0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_mock_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.memory.base'`.

- [ ] **Step 3: Write `base.py` and `mock.py`**

Create `src/lottie/memory/base.py`:

```python
"""Provider-agnostic memory interface.

All agent memory access goes through `MemoryClient` (injected as
`self.memory` by `BaseAgent`); agent code never imports a store SDK directly.
This module depends only on `schema.py` so `lottie.core` can import it without
creating an import cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lottie.memory.schema import MemoryQuery, MemoryRecord, RecallResult


class MemoryError(Exception):
    """Base class for memory subsystem errors."""


class MemoryNotConfiguredError(MemoryError):
    """Raised when an agent uses memory without a configured client."""


class MemoryClient(ABC):
    """Abstract memory store. Swap implementations via config."""

    @abstractmethod
    def remember(self, record: MemoryRecord) -> str:
        """Persist `record`; return its assigned `memory_id`."""

    @abstractmethod
    def recall(self, query: MemoryQuery) -> RecallResult:
        """Return records matching `query`, ranked by relevance."""

    @abstractmethod
    def forget(self, memory_id: str) -> bool:
        """Remove the record with `memory_id`; return True if one was removed."""


class NullMemoryClient(MemoryClient):
    """Default client for agents without memory configured. Fails loud."""

    _MSG = "memory not enabled for this agent — set memory.enabled in config.yaml"

    def remember(self, record: MemoryRecord) -> str:
        raise MemoryNotConfiguredError(self._MSG)

    def recall(self, query: MemoryQuery) -> RecallResult:
        raise MemoryNotConfiguredError(self._MSG)

    def forget(self, memory_id: str) -> bool:
        raise MemoryNotConfiguredError(self._MSG)
```

Create `src/lottie/memory/mock.py`:

```python
"""In-memory `MemoryClient` for tests.

Stores records in a plain list, assigns deterministic ids, and does naive
substring/tag matching on recall — enough for agent integration tests without
a real store. Unit tests must never touch a real store (CLAUDE.md rule 5).
"""

from __future__ import annotations

from lottie.memory.base import MemoryClient
from lottie.memory.schema import MemoryHit, MemoryQuery, MemoryRecord, RecallResult


class MockMemoryClient(MemoryClient):
    """Deterministic in-memory store. Inspect `self.records` in tests."""

    def __init__(self, records: list[MemoryRecord] | None = None) -> None:
        self.records: list[MemoryRecord] = []
        self._counter = 0
        for record in records or []:
            self.remember(record)

    def remember(self, record: MemoryRecord) -> str:
        memory_id = f"{record.namespace}-{self._counter}"
        self._counter += 1
        self.records.append(record.model_copy(update={"memory_id": memory_id}))
        return memory_id

    def recall(self, query: MemoryQuery) -> RecallResult:
        text = query.text.lower()
        tags = set(query.tags)
        hits: list[MemoryHit] = []
        for record in self.records:
            if record.namespace != query.namespace:
                continue
            if query.tier is not None and record.tier != query.tier:
                continue
            if tags and not (tags & set(record.tags)):
                continue
            if text and text not in record.content.lower():
                continue
            hits.append(MemoryHit(record=record, score=1.0))
        return RecallResult(hits=hits[: query.limit])

    def forget(self, memory_id: str) -> bool:
        for i, record in enumerate(self.records):
            if record.memory_id == memory_id:
                del self.records[i]
                return True
        return False
```

- [ ] **Step 4: Extend `__init__.py`**

Replace `src/lottie/memory/__init__.py` with:

```python
from lottie.memory.base import (
    MemoryClient,
    MemoryError,
    MemoryNotConfiguredError,
    NullMemoryClient,
)
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import (
    MemoryHit,
    MemoryQuery,
    MemoryRecord,
    MemoryTier,
    RecallResult,
    ReflectionInput,
    ReflectionResult,
)

__all__ = [
    "MemoryClient",
    "MemoryError",
    "MemoryHit",
    "MemoryNotConfiguredError",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryTier",
    "MockMemoryClient",
    "NullMemoryClient",
    "RecallResult",
    "ReflectionInput",
    "ReflectionResult",
]
```

(`agent.py` exports are added in Task 4.)

- [ ] **Step 5: Run tests + type-check + lint**

Run: `uv run pytest src/lottie/memory/tests/ -v` → all pass.
Run: `uv run mypy --strict src/lottie/memory` → `Success`.
Run: `uv run ruff check src/lottie/memory` → `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/memory/base.py src/lottie/memory/mock.py src/lottie/memory/__init__.py src/lottie/memory/tests/test_mock_client.py
git commit -m "feat(memory): add MemoryClient ABC, NullMemoryClient, MockMemoryClient"
```

---

## Task 3: Wire `self.memory` into BaseAgent

**Files:**
- Modify: `src/lottie/core/base_agent.py`
- Modify: `src/lottie/core/tests/test_base_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/core/tests/test_base_agent.py`. First check the file's existing imports/helpers: it already defines `FakeProvider`, `_In(BaseModel)` with field `x: int`, and `_Out(BaseModel)` with field `text: str`, and imports `BaseAgent`. Reuse them. You will need a concrete agent subclass — if the file already defines one (e.g. `_Agent`), reuse it; otherwise define this minimal one near the other helpers:

```python
class _MemAgent(BaseAgent[_In, _Out]):
    def _execute(self, data: _In) -> _Out:
        return _Out(text=str(data.x))
```

Then add these tests (put the imports at the top of the file with the others):

```python
from lottie.memory import MemoryNotConfiguredError, MockMemoryClient
from lottie.memory.base import NullMemoryClient
from lottie.memory.schema import MemoryQuery, MemoryRecord


def test_base_agent_default_memory_is_null() -> None:
    agent = _MemAgent(FakeProvider())
    assert isinstance(agent.memory, NullMemoryClient)
    with pytest.raises(MemoryNotConfiguredError):
        agent.memory.recall(MemoryQuery(text="", namespace="demo"))


def test_base_agent_uses_injected_memory() -> None:
    client = MockMemoryClient()
    agent = _MemAgent(FakeProvider(), memory=client)
    assert agent.memory is client
    agent.memory.remember(MemoryRecord(content="hi", namespace="demo"))
    assert client.records[0].content == "hi"
```

If `pytest` is not already imported in this test file, add `import pytest` at the top.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/core/tests/test_base_agent.py -k memory -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'memory'` (and the default-memory test fails on missing `agent.memory`).

- [ ] **Step 3: Modify `base_agent.py`**

In `src/lottie/core/base_agent.py`, add the import (with the other `from lottie...` imports, after the `from lottie.llm import ...` line):

```python
from lottie.memory.base import MemoryClient, NullMemoryClient
```

(Import from `lottie.memory.base`, NOT `lottie.memory` — `base.py` depends only on `schema.py`, so this avoids a `core ↔ memory` import cycle.)

Then change the `__init__` signature and body. Replace:

```python
    def __init__(
        self,
        llm: LLMProvider,
        *,
        name: str | None = None,
        enable_benchmarks: bool | None = None,
        benchmarks_root: Path | None = None,
    ) -> None:
        super().__init__(
            name=name,
            enable_benchmarks=enable_benchmarks,
            benchmarks_root=benchmarks_root,
        )
        self.llm = llm
```

with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/lottie/core/tests/test_base_agent.py -v` → all pass, including every pre-existing test (the change is additive).
Run: `uv run mypy --strict src/lottie/core/base_agent.py` → `Success`.
Run: `uv run ruff check src/lottie/core` → `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/lottie/core/base_agent.py src/lottie/core/tests/test_base_agent.py
git commit -m "feat(core): inject self.memory into BaseAgent (defaults to NullMemoryClient)"
```

---

## Task 4: MemoryAgent + MockMemoryAgent

**Files:**
- Create: `src/lottie/memory/agent.py`, `src/lottie/memory/tests/test_memory_agent.py`
- Modify: `src/lottie/memory/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `src/lottie/memory/tests/test_memory_agent.py`:

```python
from __future__ import annotations

from lottie.memory.agent import MemoryAgent, MockMemoryAgent
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import MemoryRecord, MemoryTier, ReflectionInput


def _episodic(content: str) -> MemoryRecord:
    return MemoryRecord(content=content, namespace="demo", tier=MemoryTier.EPISODIC)


def test_mock_memory_agent_consolidates_and_writes_back() -> None:
    client = MockMemoryClient(records=[_episodic("user asked X"), _episodic("user asked Y")])
    agent = MockMemoryAgent(responses=["note one\nnote two"], memory=client)

    result = agent.run(ReflectionInput(namespace="demo"))

    assert result.notes == ["note one", "note two"]
    assert result.consolidated_count == 2
    assert len(result.written_ids) == 2
    # Two new SEMANTIC records written back, tagged reflection.
    semantic = [r for r in client.records if r.tier is MemoryTier.SEMANTIC]
    assert [r.content for r in semantic] == ["note one", "note two"]
    assert all(r.tags == ["reflection"] for r in semantic)


def test_mock_memory_agent_blank_lines_ignored() -> None:
    client = MockMemoryClient(records=[_episodic("e")])
    agent = MockMemoryAgent(responses=["  alpha  \n\n   \nbeta\n"], memory=client)
    result = agent.run(ReflectionInput(namespace="demo"))
    assert result.notes == ["alpha", "beta"]


def test_mock_memory_agent_defaults_are_usable() -> None:
    # No args: canned response + fresh empty client. Nothing to consolidate.
    agent = MockMemoryAgent()
    result = agent.run(ReflectionInput(namespace="demo"))
    assert result.consolidated_count == 0
    assert result.notes == ["note one", "note two"]


def test_memory_agent_is_base_agent_subclass() -> None:
    from lottie.core import BaseAgent

    assert issubclass(MemoryAgent, BaseAgent)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/lottie/memory/tests/test_memory_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.memory.agent'`.

- [ ] **Step 3: Write `agent.py`**

Create `src/lottie/memory/agent.py`:

```python
"""MemoryAgent — LLM-driven consolidation of episodic memory into semantic notes.

`MemoryAgent` reads recent episodic records via `self.memory`, asks the injected
LLM to consolidate them, and writes the resulting notes back as SEMANTIC
records. `MockMemoryAgent` prewires it with mock dependencies for tests.
"""

from __future__ import annotations

from lottie.core import BaseAgent
from lottie.llm import Message, MockLLMProvider
from lottie.memory.base import MemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.schema import (
    MemoryQuery,
    MemoryRecord,
    MemoryTier,
    ReflectionInput,
    ReflectionResult,
)

REFLECT_SYSTEM_PROMPT = (
    "You consolidate an agent's recent episodic memory into durable notes. "
    "Read the log below and produce concise, standalone semantic notes — one "
    "per line, no numbering or bullets."
)


class MemoryAgent(BaseAgent[ReflectionInput, ReflectionResult]):
    """Consolidates recent episodic memory into semantic notes via the LLM."""

    def _execute(self, data: ReflectionInput) -> ReflectionResult:
        recalled = self.memory.recall(
            MemoryQuery(
                text="",
                namespace=data.namespace,
                tier=MemoryTier.EPISODIC,
                limit=data.limit,
            )
        )
        episodic = [hit.record.content for hit in recalled.hits]
        response = self.complete(
            [
                Message(role="system", content=REFLECT_SYSTEM_PROMPT),
                Message(role="user", content="\n".join(episodic)),
            ]
        )
        notes = [line.strip() for line in response.content.splitlines() if line.strip()]
        written = [
            self.memory.remember(
                MemoryRecord(
                    content=note,
                    tier=MemoryTier.SEMANTIC,
                    namespace=data.namespace,
                    tags=["reflection"],
                )
            )
            for note in notes
        ]
        return ReflectionResult(
            notes=notes,
            consolidated_count=len(episodic),
            written_ids=written,
        )


class MockMemoryAgent(MemoryAgent):
    """MemoryAgent prewired with a mock LLM + mock client for tests."""

    def __init__(
        self,
        responses: list[str] | None = None,
        memory: MemoryClient | None = None,
    ) -> None:
        super().__init__(
            llm=MockLLMProvider(responses or ["note one\nnote two"]),
            memory=memory or MockMemoryClient(),
        )
```

- [ ] **Step 4: Extend `__init__.py`**

In `src/lottie/memory/__init__.py`, add the agent import (place it after the `from lottie.memory.mock import ...` line — `schema`/`base`/`mock` must be imported before `agent` so the one-way import chain resolves cleanly):

```python
from lottie.memory.agent import MemoryAgent, MockMemoryAgent
```

and add `"MemoryAgent"` and `"MockMemoryAgent"` to `__all__` (keep it alphabetically sorted to satisfy ruff: `MemoryAgent` goes after `MemoryError`-block entries per existing order — place `"MemoryAgent"` and `"MockMemoryAgent"` so the list stays sorted; run ruff to confirm).

- [ ] **Step 5: Run tests + type-check + lint**

Run: `uv run pytest src/lottie/memory/tests/ -v` → all pass.
Run: `uv run mypy --strict src/lottie/memory` → `Success`.
Run: `uv run ruff check src/lottie/memory` → `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/lottie/memory/agent.py src/lottie/memory/__init__.py src/lottie/memory/tests/test_memory_agent.py
git commit -m "feat(memory): add MemoryAgent and MockMemoryAgent"
```

---

## Task 5: Full gate — suite, mypy, ruff

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: PASS — all prior tests plus the new memory + base_agent tests, zero failures.

- [ ] **Step 2: Type-check the whole package**

Run: `uv run mypy --strict src/lottie`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/lottie`
Expected: `All checks passed!`.

- [ ] **Step 4: Verify no import cycle**

Run: `uv run python -c "import lottie.core; import lottie.memory; from lottie.memory import MemoryAgent; print('ok')"`
Expected: prints `ok` with no `ImportError`/`partially initialized module` error.

- [ ] **Step 5: Commit any gate fixes**

```bash
git add -A
git commit -m "chore: satisfy mypy --strict and ruff for memory stubs"
```

(Skip this commit if Steps 1–4 needed no changes.)

---

## Notes for the implementer

- Run every command from the project dir, via `uv run`.
- Import direction is load-bearing: `lottie.core.base_agent` imports from
  `lottie.memory.base` (submodule, not the package); `lottie.memory.agent`
  imports from `lottie.core`. Never make `base.py` or `schema.py` import from
  `lottie.core`, and never make `base_agent.py` import the `lottie.memory`
  package (only `lottie.memory.base`).
- `MockMemoryClient.remember` copies the record (`model_copy`) before storing,
  so the caller's instance isn't mutated; the stored copy carries the id.
- Do NOT add real SQLite/ChromaDB, timestamps, embeddings, or config-driven
  client construction — all explicitly out of scope (see spec).
- Mock recall scores are always `1.0` (no real ranking) — intentional for the stub.
