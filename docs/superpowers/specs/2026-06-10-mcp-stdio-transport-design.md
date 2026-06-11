# MCP stdio Transport — Design

> Date: 2026-06-10
> Phase: 4 — Integration Layer (first transport, slice 1 of N)
> Status: approved
> Builds on: `2026-06-02-lottie-serve-core-design.md` (the transport-agnostic serving core)

## Goal

Build the first transport on top of the existing serving core (`AgentService`):
an **MCP stdio server** that exposes each discovered agent as a typed MCP tool.
A host (Claude Code, Cursor, Codex) launches `lottie serve` as a subprocess and
talks MCP over stdin/stdout; its LLM sees one tool per agent — e.g.
`research(query: str)` — with a real JSON schema, and calls it directly.

This proves Lottie's core pitch ("drops into any MCP host") with the smallest
possible slice: a pure wrapper over `AgentService`, no changes to the core, no
web/ASGI/port/auth surface.

## Decisions (locked via brainstorming)

| # | Decision | Rationale |
|---|---|---|
| 1 | **MCP first** (before REST / OpenAI-compat) | Flagship, differentiated integration; proves the "drops into Claude Code/Cursor/Codex" pitch. |
| 2 | **stdio binding** (not HTTP) | Exactly how MCP hosts launch servers (subprocess over stdin/stdout). Smallest slice; no port/ASGI/auth. `--port`/HTTP deferred to the REST slice. |
| 3 | **One typed tool per agent** | Host LLM sees real schema'd tools, not an opaque `run_agent(name, payload)`. Schema comes free from each agent's `Input` Pydantic model (`.model_json_schema()`). |
| 4 | **Return = output structured + metrics text** | `structuredContent = output` (clean, typed, what the LLM consumes); latency/tokens/cost as a trailing human-readable `TextContent` line — governance visible without polluting the structured view. |
| 5 | **Low-level `mcp.server.Server`** (not FastMCP) | Tools are *dynamic* (one per discovered agent, schema from a runtime Pydantic model). FastMCP's `@tool` introspects *static* function signatures; the low-level Server lets us set `inputSchema` directly. |
| 6 | **Optional `[serve]` extra** for the `mcp` dep | Mirrors the existing `chroma` extra; keeps the base install lean. `lottie serve` guards the import with a friendly install hint. |

## Module layout

New code only; the serving core (`service.py`, `security.py`, `schema.py`) is
unchanged.

```
src/lottie/serve/
  mcp_server.py          — build_mcp_server(root) -> Server; serve_stdio(root) -> None
  tests/test_mcp_server.py
src/lottie/cli/
  serve.py               — `lottie serve` Typer command (stdio)
```

`mcp_server.py` builds a low-level `mcp.server.Server` backed by a single
`AgentService(root)`. The MCP layer is a pure wrapper: every tool call routes
through `AgentService.run_agent`, so the `SecurityGate` chokepoint (CLAUDE.md
rules 8/9) is inherited for free, and the transport-agnostic `ServeError`
hierarchy is the only error contract the wrapper depends on.

## Tool registration (server build time)

`build_mcp_server(root)`:

1. `discover_agents(root)` — import-free metadata scan.
2. For each agent, **import** it to obtain its schema and class:
   `load_input_model(root, name)` + `load_agent_class(root, name)`.
   - On **any** exception (broken `config.yaml`, `schema.py`, `agent.py`,
     missing `Input`): **log a warning and skip** that agent. A broken agent
     never crashes the server — it simply does not appear as a tool.
   - Build a registry: `name -> (input_model, description)`.
3. `@server.list_tools()` returns one `mcp.types.Tool` per healthy agent:
   - `name = <agent name>`
   - `description =` first line of the agent's system prompt
     (`load_system_prompt`), falling back to a generic
     `"Run the <name> agent."` when absent.
   - `inputSchema = input_model.model_json_schema()`.

The import happens once at server build. `discover_agents` stays the import-free
listing path; serving a live agent imports it anyway, so importing at boot is
consistent (the serve-core spec only kept *listing* import-free).

### Tool naming

Agent directory names are already validated at creation time (`lottie create`
rejects keywords / leading underscore / non-identifier names), so an agent name
is a safe MCP tool name as-is. No additional sanitization in this slice.

## Run flow & return mapping

`@server.call_tool(name, arguments)`:

```python
result = await anyio.to_thread.run_sync(
    lambda: service.run_agent(name, arguments)
)
```

- **Threadpool-wraps the sync core** (`AgentService.run_agent` is synchronous;
  the serve-core spec's "transports threadpool-wrap" decision). No async churn
  in `BaseAgent` or the core.
- `arguments` is the MCP tool-call argument dict; passed straight as the payload.
  `run_agent` re-validates it via the agent's `Input` Pydantic model — the core
  stays the validation authority even if a host skips `inputSchema` checks.

On success, return:
- `structuredContent = result.output` (the agent's `output.model_dump()` dict).
- A trailing `TextContent` line:
  `[lottie] {latency_ms:.0f}ms · {input_tokens}/{output_tokens} tok · ${cost_usd:.4f}`.

(The exact `call_tool` return shape — a `list[ContentBlock]`, or the
`(content, structuredContent)` tuple the installed SDK version expects — is
confirmed against the pinned `mcp` version during implementation; the contract
here is "output as structured content + one metrics text line.")

## Error mapping

`AgentService` raises typed `ServeError`s. `call_tool` catches `ServeError` and
surfaces it as an MCP **tool error** (`isError`) carrying the message:

| Core error | Cause | MCP result |
|---|---|---|
| `InvalidInputError` | payload fails `Input` validation | tool error (`isError`), message = validation detail |
| `AgentExecutionError` | agent's `run` raised | tool error (`isError`) |
| `AgentLoadError` | config/module load failed at call time | tool error (`isError`); defensive — healthy agents were filtered at build |
| `AgentNotFoundError` | no such agent | tool error (`isError`); defensive — tools are pre-registered |

Tool errors (`isError`) are the right channel rather than protocol-level
exceptions: a failed agent run is a tool-execution failure the host LLM can see
and react to, not a transport fault.

## CLI command & dependency

`src/lottie/cli/serve.py` — `lottie serve`:

1. Guard the `mcp` import. If absent →
   `typer.BadParameter("lottie serve needs the MCP SDK. Install: pip install lottie-orchestrator[serve]")`.
2. `root = find_project_root()` (reuse `project.config`).
3. `serve_stdio(root)` — runs the MCP stdio event loop until the host closes
   stdin.

`--port` / HTTP binding is **not** added here; it arrives with the REST slice
when an ASGI app exists to share. Until then `lottie serve` means "stdio MCP
server." (This intentionally diverges from the CLAUDE.md `lottie serve --port
8080` line, which describes the eventual all-transports state; CLAUDE.md's CLI
table is updated to note `serve` is stdio-only for now.)

### Dependency

`pyproject.toml` `[project.optional-dependencies]`:

```toml
serve = ["mcp>=<pinned>"]
```

Mirrors the existing `chroma` extra. `mcp` is **not** a base dependency. The
`mypy` override list gains `mcp.*` only if the SDK ships without type stubs
(confirmed during implementation; prefer no override if `mcp` is typed).

## Testing (TDD, no real LLM)

Colocated under `src/lottie/serve/tests/test_mcp_server.py`, `MockLLMProvider`
only. Reuse the generator (or write a minimal `agent.py` + `config.yaml` +
`schema.py`) to scaffold a tmp project with a MockLLM-backed agent.

- **list_tools** → one `Tool` per healthy agent; `tool.name == agent name`;
  `tool.inputSchema == Input.model_json_schema()`.
- **broken agent skipped** → a second agent with an import-broken `agent.py`
  (or missing `Input`) does **not** appear in `list_tools`; the server still
  builds and the healthy agent still lists.
- **call_tool happy path** → `structuredContent == output dict`; a metrics
  `TextContent` line is present and contains the latency/tokens/cost.
- **call_tool invalid arguments** → `isError` result (maps `InvalidInputError`).
- **call_tool agent raises** → `isError` result (agent whose `run` throws →
  `AgentExecutionError`).
- **CLI** `lottie serve`:
  - `mcp` import monkeypatched to fail → clean `BadParameter` with the install
    hint (no traceback).
  - happy construction: `serve_stdio` monkeypatched so the test asserts the
    server is built for the resolved root **without** blocking on the stdio loop.

Drive the server in tests via the MCP SDK's in-memory client/session helper
(e.g. a connected server/client pair) where practical, or by invoking the
registered `list_tools` / `call_tool` handlers directly — whichever the pinned
`mcp` version supports cleanly. No subprocess, no real stdio in unit tests.

`mypy --strict` + `ruff` stay clean. No `Any` without justification. The `mcp`
SDK types are used directly; if a handler signature forces an untyped boundary,
it is annotated and justified inline.

## Out of scope (this slice)

- HTTP / SSE / Streamable-HTTP binding and `lottie serve --port`.
- REST and OpenAI-compat (`/v1/chat/completions`) transports.
- Auth, rate limiting, streaming/partial results.
- Real security skills (`InputSanitizerSkill` / `OutputValidationSkill` /
  `SecretDetectionSkill`) — the `SecurityGate` stays identity until wired.
- MCP **resources** and **prompts** (tools only this slice).
- Multi-agent routing / supervisor orchestration.
- Threading `model_params` / `registry` into the run (still deferred from the
  serve-core slice).
- Hot-reload of agents while the server runs (server snapshots the agent set at
  build; restart to pick up new agents).

## Follow-ups (deferred)

- REST + OpenAI-compat slice → shared ASGI app → `lottie serve --port`,
  reconciling the CLAUDE.md `serve --port 8080` line.
- MCP HTTP/SSE binding off the same `Server` object.
- Per-agent tool descriptions richer than the system-prompt first line
  (input/output field docs as tool annotations).
- Audit-log the metrics emitted per tool call (Phase 3 governance).
- `cli/run.py` already noted to delegate to `AgentService.run_agent`; the MCP
  path is the second consumer of that single run path.
