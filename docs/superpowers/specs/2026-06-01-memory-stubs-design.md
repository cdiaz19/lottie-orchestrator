# Memory Stubs — Design

> Date: 2026-06-01
> Phase: 0 — Foundations (Memory harness stub)
> Status: approved

## Goal

Stand up the memory subsystem's interface, schemas, and test doubles — the
memory analogue of `LLMProvider` + `MockLLMProvider` — so agents can be written
against `self.memory` before any real store exists. No SQLite, no ChromaDB.

Delivers: `src/lottie/memory/` with all Pydantic schemas, the `MemoryClient`
ABC + `NullMemoryClient` default + `MockMemoryClient` in-memory double, a
`MemoryAgent` (BaseAgent subclass owning `reflect()`-style consolidation) +
`MockMemoryAgent`, and a backward-compatible `self.memory` injection point on
`BaseAgent`.

## Scope decisions

- **Client = storage interface; Agent = LLM harness.** `MemoryClient` is the
  storage contract (remember/recall/forget). `MemoryAgent` is a `BaseAgent`
  subclass that performs LLM-driven consolidation over what the client returns.
- **Three client ops, tier as a field.** `remember` / `recall` / `forget`. Tier
  is a field on `MemoryRecord` / `MemoryQuery`, not a method-per-tier. No
  speculative per-tier API.
- **`MemoryAgent` is a real `BaseAgent` subclass driven by the injected LLM.**
  `MockMemoryAgent` prewires it with `MockLLMProvider` + `MockMemoryClient`.
- **`self.memory` is wired into `BaseAgent` now**, defaulting to a
  `NullMemoryClient` that raises on use — additive and backward-compatible.
- **No real persistence.** `MockMemoryClient` is in-memory only. Real
  SQLite/ChromaDB tiers and `created_at` timestamps are deferred to a later task.

## Module layout (`src/lottie/memory/`, mirrors `src/lottie/llm/`)

| File | Responsibility |
|---|---|
| `schema.py` | All Pydantic v2 models (no logic) |
| `base.py` | `MemoryClient` ABC, `NullMemoryClient`, `MemoryError`/`MemoryNotConfiguredError`. Depends only on `schema.py`. |
| `mock.py` | `MockMemoryClient` — in-memory double |
| `agent.py` | `MemoryAgent` (+ `MockMemoryAgent`). Imports `lottie.core.BaseAgent`. |
| `__init__.py` | Public exports; imports `schema`/`base`/`mock` **before** `agent` |
| `tests/` | Colocated unit tests (matches `llm/tests`, `core/tests`) |

## Schemas (`schema.py`)

```python
class MemoryTier(str, Enum):
    WORKING = "working"        # T0 — in-context
    EPISODIC = "episodic"      # T1 — append-only log
    SEMANTIC = "semantic"      # T2 — consolidated knowledge
    PROCEDURAL = "procedural"  # T3 — config/rules


class MemoryRecord(BaseModel):
    content: str
    tier: MemoryTier = MemoryTier.EPISODIC
    namespace: str
    tags: list[str] = []
    metadata: dict[str, str] = {}
    memory_id: str | None = None   # assigned by remember()


class MemoryQuery(BaseModel):
    text: str
    namespace: str
    tier: MemoryTier | None = None   # None = any tier
    tags: list[str] = []             # match-any
    limit: int = 10


class MemoryHit(BaseModel):
    record: MemoryRecord
    score: float


class RecallResult(BaseModel):
    hits: list[MemoryHit] = []


class ReflectionInput(BaseModel):
    namespace: str
    limit: int = 50


class ReflectionResult(BaseModel):
    notes: list[str] = []
    consolidated_count: int = 0
    written_ids: list[str] = []
```

`metadata` is `dict[str, str]` (not `dict[str, Any]`) to satisfy
`mypy --strict` rule 6. No timestamp field in the stub — episodic ordering is
insertion order in the mock; a real `created_at` arrives with the SQLite store.

## `MemoryClient` ABC + `NullMemoryClient` (`base.py`)

```python
class MemoryStoreError(Exception): ...   # not "MemoryError" — that shadows a builtin
class MemoryNotConfiguredError(MemoryStoreError): ...


class MemoryClient(ABC):
    @abstractmethod
    def remember(self, record: MemoryRecord) -> str: ...        # returns memory_id
    @abstractmethod
    def recall(self, query: MemoryQuery) -> RecallResult: ...
    @abstractmethod
    def forget(self, memory_id: str) -> bool: ...               # True if removed


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

Raise-on-use (not silent no-op) so a misconfigured agent fails clearly. Agents
that never touch `self.memory` are unaffected.

## `MockMemoryClient` (`mock.py`)

In-memory `list[MemoryRecord]`, deterministic, no real store.

- `__init__(self, records: list[MemoryRecord] | None = None)` — seed optional
  records (each gets an id via `remember`). Exposes `self.records` for assertions.
- `remember`: `memory_id = f"{record.namespace}-{self._counter}"`, increment
  counter, copy the record with that id set (`model_copy(update=...)`), append,
  return the id.
- `recall`: from `self.records`, keep those where `record.namespace ==
  query.namespace`, AND (`query.tier is None` or `record.tier == query.tier`),
  AND (`not query.tags` or `set(query.tags) & set(record.tags)`), AND
  (`query.text.lower() in record.content.lower()` — empty text matches all).
  Each match → `MemoryHit(record=record, score=1.0)`. Truncate to `query.limit`.
  Return `RecallResult(hits=...)`.
- `forget`: drop the record whose `memory_id` matches; return `True` if one was
  removed, else `False`.

## `MemoryAgent` + `MockMemoryAgent` (`agent.py`)

```python
class MemoryAgent(BaseAgent[ReflectionInput, ReflectionResult]):
    """Consolidates recent episodic memory into semantic notes via the LLM."""

    def _execute(self, data: ReflectionInput) -> ReflectionResult:
        recalled = self.memory.recall(
            MemoryQuery(text="", namespace=data.namespace,
                        tier=MemoryTier.EPISODIC, limit=data.limit)
        )
        episodic = [hit.record.content for hit in recalled.hits]
        messages = [
            Message(role="system", content=REFLECT_SYSTEM_PROMPT),
            Message(role="user", content="\n".join(episodic)),
        ]
        response = self.complete(messages)
        notes = [line.strip() for line in response.content.splitlines() if line.strip()]
        written = [
            self.memory.remember(
                MemoryRecord(content=note, tier=MemoryTier.SEMANTIC,
                             namespace=data.namespace, tags=["reflection"])
            )
            for note in notes
        ]
        return ReflectionResult(
            notes=notes, consolidated_count=len(episodic), written_ids=written
        )


class MockMemoryAgent(MemoryAgent):
    """MemoryAgent prewired with mock LLM + mock client for tests."""

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

`REFLECT_SYSTEM_PROMPT` is a module-level constant (short instruction to
summarize episodic logs into durable notes, one per line).

## BaseAgent wiring (`core/base_agent.py`)

Add a `memory` parameter, defaulting to a `NullMemoryClient`:

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
    super().__init__(...)            # unchanged
    self.llm = llm
    self.memory: MemoryClient = memory or NullMemoryClient()
```

**Import-cycle avoidance:** import from the submodule, not the package —
`from lottie.memory.base import MemoryClient, NullMemoryClient`. `base.py`
imports only `schema.py`, so there is no `core ↔ memory` cycle. `agent.py`
imports `lottie.core.BaseAgent` (one direction only). `memory/__init__.py` must
import `schema`/`base`/`mock` before `agent`.

## Public exports (`memory/__init__.py`)

`MemoryTier`, `MemoryRecord`, `MemoryQuery`, `MemoryHit`, `RecallResult`,
`ReflectionInput`, `ReflectionResult`, `MemoryClient`, `NullMemoryClient`,
`MockMemoryClient`, `MemoryStoreError`, `MemoryNotConfiguredError`,
`MemoryAgent`, `MockMemoryAgent`.

## Testing (TDD, no real LLM)

Colocated under `src/lottie/memory/tests/`:

- `test_schema.py` — each model constructs with defaults; `MemoryTier` values;
  `metadata`/`tags` default independently per instance.
- `test_mock_client.py` — `remember` returns id and sets `memory_id`; `recall`
  filters by namespace, tier, tags (match-any), substring (incl. empty-text =
  all); `limit` truncates; `forget` returns True then False; `NullMemoryClient`
  raises `MemoryNotConfiguredError` on all three ops.
- `test_memory_agent.py` — `MockMemoryAgent` seeded with episodic records: `run`
  recalls them, the canned LLM response is split into notes, each note is
  written back as a SEMANTIC record (assert via the client's `.records`),
  `ReflectionResult.notes`/`consolidated_count`/`written_ids` are correct.
  Verifies no real LLM (mock only).

Extend `src/lottie/core/tests/test_base_agent.py`:
- A `BaseAgent` built without `memory` has `self.memory` as a `NullMemoryClient`
  (and calling it raises `MemoryNotConfiguredError`).
- A `BaseAgent` built with an injected `MockMemoryClient` uses it.
- All existing `test_base_agent` cases still pass (additive change).

## Out of scope

- Real SQLite (episodic) / ChromaDB (semantic) / YAML (procedural) stores.
- `created_at` / timestamps / TTL / eviction.
- Reading the agent `config.yaml` `memory:` block to auto-construct a client
  (wiring `lottie run` to build a real client) — later task.
- Embedding/vector similarity scoring (mock uses substring match, score 1.0).
- `mypy --strict` and `ruff` must stay clean; no `Any` without justification.
