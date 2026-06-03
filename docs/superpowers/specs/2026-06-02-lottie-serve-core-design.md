# `lottie serve` — Serving Core Design

> Date: 2026-06-02
> Phase: 0 — Foundations (transport-agnostic serving core)
> Status: approved

## Goal

Stand up the transport-agnostic serving core that any future Phase-4 transport
(MCP, OpenAI-compat, REST, WebSocket) plugs into: an `AgentService` that lists
and runs agents by name, with a `SecurityGate` chokepoint on every run. No web
or MCP dependencies, no `lottie serve` CLI command yet. This is the in-process
engine; transports are deferred to Phase 4.

Delivers `src/lottie/serve/` with `AgentService`, a transport-agnostic error
hierarchy, a pluggable identity `SecurityGate`, and the `AgentInfo` / `RunResult`
schemas — all unit-testable with `MockLLMProvider`.

## Scope decisions

- **Serving core only.** No FastAPI / Starlette / MCP SDK / OpenAI-compat /
  WebSocket. No `lottie serve` CLI command. Transport choice is a Phase-4
  decision and is explicitly out of this slice.
- **Sync-only.** `AgentService.run_agent` is plain synchronous; `BaseAgent`
  stays sync-first. Phase-4 async transports wrap the call in a threadpool. No
  `anyio`, no async tests in this slice.
- **SecurityGate = identity, pluggable.** `check_input` / `check_output` pass
  their text through unchanged for now, but are the single chokepoint every run
  flows through. The real `InputSanitizerSkill` / `OutputValidationSkill` /
  `SecretDetectionSkill` (Phase 1) swap in later via constructor injection — zero
  call-site change. Honors CLAUDE.md rules 8/9 structurally before those skills
  exist.
- **Transport-agnostic errors.** A `ServeError` hierarchy (no `typer`) so the
  core never couples to the CLI. Transports map these to HTTP/MCP status later.
- **Reuses the existing run path.** Same `project.config` + `project.discovery`
  + `llm.build_provider` flow as `cli/run.py`, factored into a service so both
  CLI and future transports share one code path.

## Module layout (`src/lottie/serve/`, mirrors `benchmark/` + `memory/`)

| File | Responsibility |
|---|---|
| `schema.py` | Pydantic v2 models (`AgentInfo`, `RunResult`) — no logic |
| `security.py` | `SecurityGate` — identity `check_input` / `check_output`, injectable |
| `service.py` | `AgentService` + `ServeError` hierarchy |
| `__init__.py` | Public exports |
| `tests/` | Colocated unit tests (`MockLLMProvider` only, no real LLM) |

## Schemas (`schema.py`)

```python
class AgentInfo(BaseModel):
    name: str
    provider: str | None = None       # configured provider (may be None)


class RunResult(BaseModel):
    agent: str
    output: dict[str, object]         # output.model_dump() — dict, not Any (rule 6)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
```

`AgentInfo` carries only what `discovery.UnitInfo` yields import-free (name +
configured provider). Input/output model names and descriptions require importing
user code (and have no config-level source), so they belong to a future
`describe_agent` / the existing `inspect` command — not the import-free listing.

`dict[str, object]` (not `Any`) keeps `mypy --strict` honest while carrying an
arbitrary agent output payload. Metrics fields are read from `agent.last_metrics`
so future transports return rich responses for free; they default to zeros when
no metrics were captured.

## Errors (`service.py`)

```python
class ServeError(Exception):
    """Base for serving-core errors. Transport-agnostic — no typer."""

class AgentNotFoundError(ServeError): ...      # no agents/<name>/agent.py
class InvalidInputError(ServeError): ...        # payload fails Input validation
class AgentExecutionError(ServeError): ...       # agent.run raised
```

Deliberately not `typer.BadParameter`: the core must not depend on the CLI.
Transports (and a future `lottie serve` command) catch these and map to HTTP
4xx/5xx or MCP error responses.

## `SecurityGate` (`security.py`)

```python
class SecurityGate:
    """Single chokepoint for external content entering/leaving an agent run.

    Identity for now. The real InputSanitizerSkill / OutputValidationSkill /
    SecretDetectionSkill (Phase 1) swap in via constructor injection without
    changing any call site. See CLAUDE.md rules 8 and 9.
    """

    def check_input(self, text: str) -> str:
        return text   # TODO(phase1): route through InputSanitizerSkill

    def check_output(self, text: str) -> str:
        return text   # TODO(phase1): route through OutputValidationSkill + SecretDetectionSkill
```

The gate operates on serialized text (the raw external payload JSON in, the
serialized output JSON out) because the Phase-1 security skills scan text. The
identity implementation returns its argument unchanged; a subclass / replacement
performs real scanning and may raise on a violation.

## `AgentService` (`service.py`)

```python
class AgentService:
    def __init__(self, root: Path, *, gate: SecurityGate | None = None) -> None:
        self._root = root
        self._gate = gate or SecurityGate()
        # discover metadata once at init (import-free) for list_agents()
```

### `list_agents() -> list[AgentInfo]`
Import-free. Reuse `project.discovery.discover_agents` (metadata scan, no user
code import) to build one `AgentInfo(name, provider)` per discovered agent.
Discovery already guards provider resolution (`_provider_of` swallows a bad
`config.yaml` → `None`), so a broken agent lists with `provider=None` and never
crashes the listing.

### `run_agent(name, payload, *, provider=None) -> RunResult`
`payload: Mapping[str, object]` (covariant — accepts `dict[str, str]` etc.).

1. `unit_dir = root / "agents" / name`; missing `agent.py` → `AgentNotFoundError`.
2. `self._gate.check_input(json.dumps(payload))` — gate the raw external payload.
3. `cfg = load_agent_config(unit_dir)`; `llm = build_provider(provider or cfg.provider)`.
4. `input_model = load_input_model(root, name)`;
   `data = input_model.model_validate(payload)` — `ValidationError` →
   `InvalidInputError`.
5. `agent = load_agent_class(root, name)(llm=llm)`;
   `output = agent.run(data)` in `try/except Exception` → `AgentExecutionError`
   (chain the original via `from exc`).
6. `output_json = output.model_dump_json()`;
   `self._gate.check_output(output_json)` — gate the serialized output.
7. Read `agent.last_metrics` (if present) for latency/tokens/cost; build and
   return `RunResult(agent=name, output=output.model_dump(), ...)`.

No provider construction or class import happens until `run_agent` is called —
`list_agents` stays import-free so a broken agent can be listed but only fails
when actually run.

## Public exports (`serve/__init__.py`)

`AgentInfo`, `RunResult`, `SecurityGate`, `AgentService`, `ServeError`,
`AgentNotFoundError`, `InvalidInputError`, `AgentExecutionError`.

## Testing (TDD, no real LLM)

Colocated under `src/lottie/serve/tests/`, `MockLLMProvider` only:

- `test_schema.py` — `AgentInfo` / `RunResult` construct with defaults; metrics
  default to zero; `output` accepts an arbitrary dict.
- `test_security.py` — `SecurityGate().check_input` / `check_output` return their
  argument unchanged (identity contract).
- `test_service.py` — scaffold an agent into a tmp project (reuse the generator
  or write minimal `agent.py` + `config.yaml`), then:
  - `list_agents()` returns `AgentInfo(name, provider)` per agent **without
    importing user code** (assert via a deliberately import-broken second agent
    that still lists, with `provider` resolved or `None`).
  - `run_agent` happy path with a `MockLLMProvider`-backed agent → `RunResult`
    with the expected `output` dict and metrics populated from `last_metrics`.
  - unknown name → `AgentNotFoundError`.
  - payload failing Input validation → `InvalidInputError`.
  - an agent whose `run` raises → `AgentExecutionError` (and the cause is chained).
  - a spy `SecurityGate` subclass asserts `check_input` **and** `check_output`
    are each called exactly once per successful run, input before output.
  - `--provider`-style override: `run_agent(..., provider=...)` builds the
    overriding provider (assert via a stub `build_provider` or distinct mock).

## Out of scope

- Any transport: MCP, OpenAI-compat (`/v1/chat/completions`), REST, WebSocket.
- The `lottie serve` CLI command (wiring + `--port`).
- Async / `anyio` / concurrency — sync-only core.
- Real security skills (`InputSanitizerSkill`, `OutputValidationSkill`,
  `SecretDetectionSkill`) — the gate is identity until Phase 1.
- Auth, rate limiting, streaming, multi-agent orchestration / routing.
- Threading `model_params` + `registry` paths into the run (still deferred).
- `mypy --strict` + `ruff` stay clean; no `Any` without justification.
