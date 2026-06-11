# MCP stdio Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose each Lottie agent as a typed MCP tool over a stdio server (`lottie serve`), built as a pure wrapper over the existing `AgentService`.

**Architecture:** New `src/lottie/serve/mcp_server.py` builds a low-level `mcp.server.lowlevel.Server`. At build time it discovers agents, imports each to read its `Input` JSON schema, and registers one MCP `Tool` per healthy agent (broken agents are logged and skipped). A `call_tool` handler threadpool-wraps the sync `AgentService.run_agent`, returns the agent output as `structuredContent` plus a metrics text line, and maps every `ServeError` to an MCP `isError` result. A thin `lottie serve` CLI command runs the stdio loop. `mcp` is an optional `[serve]` dependency, imported lazily so the base install never needs it.

**Tech Stack:** Python 3.12, `mcp>=1.27.2` (official MCP SDK, low-level Server API), `anyio` (ships with `mcp`), Typer, pytest + pytest-asyncio (`asyncio_mode = "auto"` already set).

---

## Spec reference

`docs/superpowers/specs/2026-06-10-mcp-stdio-transport-design.md`

## Verified API facts (do not re-derive)

- Imports: `from mcp.server.lowlevel import Server`, `from mcp import types`, `from mcp.server.stdio import stdio_server`, `from mcp.shared.memory import create_connected_server_and_client_session` (in-memory test client).
- `Server("lottie")` — constructor takes the server name.
- `@server.list_tools()` decorates an **async** func returning `list[types.Tool]`.
- `types.Tool(name=str, description=str|None, inputSchema=dict[str, Any])`.
- `@server.call_tool(validate_input=False)` decorates an **async** func `(name: str, arguments: dict) -> ...`. With `validate_input=False` the SDK does **not** pre-validate against `inputSchema` — `AgentService.run_agent`'s Pydantic validation is the single authority (spec decision).
- A `call_tool` handler may return either a `tuple[Iterable[types.ContentBlock], dict]` (→ `content` + `structuredContent`) **or** a `types.CallToolResult` directly. Any raised exception is auto-wrapped to an `isError` result by the SDK.
- `types.TextContent(type="text", text=...)` is a `ContentBlock`.
- `types.CallToolResult(content=[...], isError=True)` — for the explicit error path.
- stdio loop: `async with stdio_server() as (read, write): await server.run(read, write, server.create_initialization_options())`.
- Client (tests): `async with create_connected_server_and_client_session(server) as client:` then `await client.list_tools()` → `.tools`; `await client.call_tool(name, args)` → result with `.isError`, `.content`, `.structuredContent`.
- `mcp` ships `py.typed` → **no** mypy override needed.
- Generated `echo` agent: `Input` requires field `query`; output is `{"result": <text>}`; `SYSTEM_PROMPT` first line is `You are EchoAgent, a Lottie agent.`
- Existing test seam: monkeypatch `lottie.serve.service.build_provider` to return a `MockLLMProvider` (patch is visible inside the anyio worker thread).

---

## File structure

| File | Responsibility |
|---|---|
| `src/lottie/serve/mcp_server.py` (create) | `build_mcp_server(root)`, `serve_stdio(root)`, tool registration, run/error mapping. Imports `mcp` at module top — **never** imported by `serve/__init__.py`. |
| `src/lottie/cli/serve.py` (create) | `serve()` Typer command: lazy-import guard → friendly hint, then `serve_stdio(find_project_root())`. |
| `src/lottie/cli/app.py` (modify) | Register `app.command("serve")(serve)`. |
| `pyproject.toml` (modify) | Add `serve = ["mcp>=1.27.2"]` optional extra. |
| `src/lottie/serve/tests/test_mcp_server.py` (create) | Unit tests for build/list/call/error (in-memory MCP client). |
| `src/lottie/cli/tests/test_serve.py` (create) | CLI command tests (import guard + happy wiring). |
| `CLAUDE.md` (modify) | Note `lottie serve` is stdio-MCP for now; `--port` deferred. |

> **Critical:** `src/lottie/serve/__init__.py` stays unchanged. It must NOT import `mcp_server` — doing so would pull `mcp` into every `import lottie.serve`, breaking the base install without the `[serve]` extra.

---

### Task 1: Add the `mcp` optional dependency

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)

- [ ] **Step 1: Add the `serve` extra**

In `pyproject.toml`, under `[project.optional-dependencies]` (which already has `chroma = [...]`), add:

```toml
serve = ["mcp>=1.27.2"]
```

- [ ] **Step 2: Install it into the dev environment**

Run: `uv sync --extra serve --extra chroma`
Expected: resolves and installs `mcp` (and its deps: `anyio`, `httpx`, `pydantic`, `jsonschema`, …). No errors.

- [ ] **Step 3: Verify the import works**

Run: `.venv/bin/python -c "from mcp.server.lowlevel import Server; from mcp import types; from mcp.server.stdio import stdio_server; print('mcp ok')"`
Expected: prints `mcp ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(serve): add mcp SDK as optional [serve] extra"
```

---

### Task 2: `build_mcp_server` — tool registration (list_tools)

**Files:**
- Create: `src/lottie/serve/mcp_server.py`
- Test: `src/lottie/serve/tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing tests (shared scaffold + list_tools)**

Create `src/lottie/serve/tests/test_mcp_server.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.llm import MockLLMProvider

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a real project with one generated `echo` agent on disk."""
    from lottie.cli import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    demo = tmp_path / "demo"
    monkeypatch.chdir(demo)
    assert runner.invoke(app, ["create", "agent", "echo"]).exit_code == 0
    return demo


def _mock_provider(monkeypatch: pytest.MonkeyPatch, response: str = "hello world") -> None:
    """Patch build_provider in the service module to return a MockLLMProvider."""
    monkeypatch.setattr(
        "lottie.serve.service.build_provider",
        lambda name: MockLLMProvider([response]),
    )


async def test_list_tools_one_per_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.project.discovery import load_input_model
    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.list_tools()

    names = {t.name for t in result.tools}
    assert {"echo", "hello"} <= names  # `init` ships a hello agent; we added echo
    echo = next(t for t in result.tools if t.name == "echo")
    assert echo.inputSchema == load_input_model(demo, "echo").model_json_schema()
    assert echo.description  # non-empty (first system-prompt line)


async def test_broken_agent_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    broken = demo / "agents" / "broken"
    broken.mkdir()
    (broken / "agent.py").write_text("!!! not valid python", encoding="utf-8")

    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        result = await client.list_tools()

    names = {t.name for t in result.tools}
    assert "broken" not in names  # unimportable agent skipped, not fatal
    assert "echo" in names        # healthy agents still register
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest src/lottie/serve/tests/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lottie.serve.mcp_server'`

- [ ] **Step 3: Write the minimal implementation (build_mcp_server + list_tools)**

Create `src/lottie/serve/mcp_server.py`:

```python
"""MCP stdio transport: expose each agent as a typed MCP tool.

Pure wrapper over AgentService. Imports the `mcp` SDK at module top, so this
module is imported lazily (never from serve/__init__.py) — the base install
does not require the optional [serve] extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from lottie.project.discovery import (
    discover_agents,
    load_input_model,
    load_system_prompt,
)
from lottie.serve.service import AgentService, ServeError

logger = logging.getLogger(__name__)


def _tool_description(root: Path, name: str) -> str:
    """First line of the agent's system prompt, or a generic fallback."""
    prompt = load_system_prompt(root, name)
    if prompt and prompt.strip():
        return prompt.strip().splitlines()[0]
    return f"Run the {name} agent."


def build_mcp_server(root: Path, *, service: AgentService | None = None) -> Server:
    """Build an MCP Server exposing one typed tool per healthy agent under `root`."""
    svc = service or AgentService(root)

    tools: dict[str, types.Tool] = {}
    for unit in discover_agents(root):
        try:
            input_model = load_input_model(root, unit.name)
            description = _tool_description(root, unit.name)
        except Exception as exc:  # noqa: BLE001 — a broken agent is skipped, not fatal
            logger.warning("skipping agent %r: %s", unit.name, exc)
            continue
        tools[unit.name] = types.Tool(
            name=unit.name,
            description=description,
            inputSchema=input_model.model_json_schema(),
        )

    server: Server = Server("lottie")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return list(tools.values())

    return server
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest src/lottie/serve/tests/test_mcp_server.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Lint + typecheck**

Run: `.venv/bin/ruff check src/lottie/serve/mcp_server.py && .venv/bin/mypy --strict src/lottie/serve/mcp_server.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/mcp_server.py src/lottie/serve/tests/test_mcp_server.py
git commit -m "feat(serve): register one MCP tool per agent (list_tools)"
```

---

### Task 3: `call_tool` — run + return mapping + error mapping

**Files:**
- Modify: `src/lottie/serve/mcp_server.py`
- Test: `src/lottie/serve/tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/lottie/serve/tests/test_mcp_server.py`:

```python
from collections.abc import Mapping  # noqa: E402  (top-of-file group is fine too)

from lottie.llm import LLMResponse, Message  # noqa: E402
from lottie.llm.base import LLMProvider  # noqa: E402


class _BoomProvider(LLMProvider):
    """Provider whose complete() always raises — to force an execution error."""

    @property
    def model(self) -> str:
        return "boom/boom"

    def complete(
        self,
        messages: list[Message],
        model_params: Mapping[str, object] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("boom")


async def test_call_tool_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        res = await client.call_tool("echo", {"query": "hi"})

    assert res.isError is False
    assert res.structuredContent == {"result": "hello world"}
    texts = [c.text for c in res.content if c.type == "text"]
    assert any("[lottie]" in t for t in texts)  # metrics line present


async def test_call_tool_invalid_input_is_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    _mock_provider(monkeypatch)
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        res = await client.call_tool("echo", {"wrong": "field"})

    assert res.isError is True  # InvalidInputError → isError


async def test_call_tool_execution_error_is_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from lottie.serve.mcp_server import build_mcp_server

    demo = _scaffold(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "lottie.serve.service.build_provider", lambda name: _BoomProvider()
    )
    server = build_mcp_server(demo)
    async with create_connected_server_and_client_session(server) as client:
        res = await client.call_tool("echo", {"query": "hi"})

    assert res.isError is True  # AgentExecutionError → isError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest src/lottie/serve/tests/test_mcp_server.py -k call_tool -v`
Expected: FAIL — the server has no `call_tool` handler, so the calls error differently than asserted / the client raises.

- [ ] **Step 3: Add the call_tool handler**

In `src/lottie/serve/mcp_server.py`, inside `build_mcp_server`, **after** the `_list_tools` definition and **before** `return server`, add:

```python
    @server.call_tool(validate_input=False)
    async def _call_tool(
        name: str, arguments: dict[str, object]
    ) -> tuple[list[types.ContentBlock], dict[str, object]] | types.CallToolResult:
        try:
            result = await anyio.to_thread.run_sync(
                lambda: svc.run_agent(name, arguments)
            )
        except ServeError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))],
                isError=True,
            )
        metrics = types.TextContent(
            type="text",
            text=(
                f"[lottie] {result.latency_ms:.0f}ms · "
                f"{result.input_tokens}/{result.output_tokens} tok · "
                f"${result.cost_usd:.4f}"
            ),
        )
        return ([metrics], result.output)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest src/lottie/serve/tests/test_mcp_server.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint + typecheck**

Run: `.venv/bin/ruff check src/lottie/serve/mcp_server.py && .venv/bin/mypy --strict src/lottie/serve/mcp_server.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/mcp_server.py src/lottie/serve/tests/test_mcp_server.py
git commit -m "feat(serve): MCP call_tool — run agent, return output + metrics, map errors"
```

---

### Task 4: `serve_stdio` — the stdio run loop

**Files:**
- Modify: `src/lottie/serve/mcp_server.py`
- Test: `src/lottie/serve/tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing test**

Append to `src/lottie/serve/tests/test_mcp_server.py`:

```python
def test_serve_stdio_runs_built_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """serve_stdio builds the server for `root` and hands it to the run loop."""
    import lottie.serve.mcp_server as mod

    demo = _scaffold(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    def _fake_run(server: object) -> None:
        captured["server"] = server

    # Replace the anyio-driven loop so the test never blocks on stdio.
    monkeypatch.setattr(mod, "_run_stdio_blocking", _fake_run)
    mod.serve_stdio(demo)

    assert "server" in captured  # a built Server was passed to the run loop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest src/lottie/serve/tests/test_mcp_server.py::test_serve_stdio_runs_built_server -v`
Expected: FAIL — `AttributeError: module 'lottie.serve.mcp_server' has no attribute '_run_stdio_blocking'` / `serve_stdio` not defined.

- [ ] **Step 3: Add `serve_stdio` and the run loop**

In `src/lottie/serve/mcp_server.py`, append at module level (after `build_mcp_server`):

```python
async def _run_stdio(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def _run_stdio_blocking(server: Server) -> None:
    """Drive the async stdio loop to completion (seam: patched in tests)."""
    anyio.run(_run_stdio, server)


def serve_stdio(root: Path) -> None:
    """Build the MCP server for `root` and serve it over stdio until stdin closes."""
    server = build_mcp_server(root)
    _run_stdio_blocking(server)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest src/lottie/serve/tests/test_mcp_server.py::test_serve_stdio_runs_built_server -v`
Expected: PASS

- [ ] **Step 5: Lint + typecheck**

Run: `.venv/bin/ruff check src/lottie/serve/mcp_server.py && .venv/bin/mypy --strict src/lottie/serve/mcp_server.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/lottie/serve/mcp_server.py src/lottie/serve/tests/test_mcp_server.py
git commit -m "feat(serve): serve_stdio run loop over the MCP server"
```

---

### Task 5: `lottie serve` CLI command

**Files:**
- Create: `src/lottie/cli/serve.py`
- Modify: `src/lottie/cli/app.py`
- Test: `src/lottie/cli/tests/test_serve.py`

- [ ] **Step 1: Write the failing tests**

Create `src/lottie/cli/tests/test_serve.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lottie.cli import app

runner = CliRunner()


def _scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "demo"]).exit_code == 0
    return tmp_path / "demo"


def test_serve_missing_mcp_shows_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy `from lottie.serve.mcp_server import serve_stdio` to fail,
    # simulating an environment without the [serve] extra installed.
    monkeypatch.setitem(sys.modules, "lottie.serve.mcp_server", None)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "lottie-orchestrator[serve]" in result.output


def test_serve_invokes_serve_stdio_with_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = _scaffold(tmp_path, monkeypatch)
    monkeypatch.chdir(demo)
    captured: dict[str, Path] = {}

    # Patch the function the command imports; never enter the real stdio loop.
    monkeypatch.setattr(
        "lottie.serve.mcp_server.serve_stdio",
        lambda root: captured.__setitem__("root", root),
    )
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert captured["root"] == demo
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest src/lottie/cli/tests/test_serve.py -v`
Expected: FAIL — `serve` is not a registered command (`Usage: ... No such command 'serve'`), exit_code is for unknown command but the assertions on output/root fail.

- [ ] **Step 3: Create the command**

Create `src/lottie/cli/serve.py`:

```python
"""`lottie serve` — run the MCP stdio server for the current project."""

from __future__ import annotations

import typer

from lottie.project.config import find_project_root


def serve() -> None:
    """Serve the project's agents as MCP tools over stdio."""
    try:
        from lottie.serve.mcp_server import serve_stdio
    except ImportError as exc:
        raise typer.BadParameter(
            "lottie serve needs the MCP SDK. "
            "Install: pip install lottie-orchestrator[serve]"
        ) from exc
    root = find_project_root()
    serve_stdio(root)
```

- [ ] **Step 4: Register it in the CLI app**

In `src/lottie/cli/app.py`, add the import alongside the other command imports:

```python
from lottie.cli.serve import serve
```

and register it with the other `app.command(...)` lines:

```python
app.command("serve")(serve)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest src/lottie/cli/tests/test_serve.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Lint + typecheck**

Run: `.venv/bin/ruff check src/lottie/cli/serve.py src/lottie/cli/app.py && .venv/bin/mypy --strict src/lottie/cli/serve.py src/lottie/cli/app.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/lottie/cli/serve.py src/lottie/cli/app.py src/lottie/cli/tests/test_serve.py
git commit -m "feat(cli): lottie serve — run the MCP stdio server"
```

---

### Task 6: Docs + full-suite gate

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the CLI docs**

In `CLAUDE.md`, in the `## CLI commands` block, replace the existing serve line:

```bash
lottie serve --port 8080               # start MCP + OpenAI-compat + REST
```

with:

```bash
lottie serve                           # start the MCP stdio server (one tool per agent)
# --port / HTTP, OpenAI-compat, and REST transports land in later Phase-4 slices
```

- [ ] **Step 2: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (the 209 existing + 8 new = 217), no failures.

- [ ] **Step 3: Run the full type + lint gate (the CI gate)**

Run: `.venv/bin/mypy --strict src && .venv/bin/ruff check .`
Expected: `Success: no issues found` from mypy; ruff reports `All checks passed!`

- [ ] **Step 4: Smoke-test the command surface (no real LLM)**

Run: `.venv/bin/python -c "from lottie.cli import app; from typer.testing import CliRunner; r=CliRunner().invoke(app, ['--help']); print('serve' in r.output)"`
Expected: prints `True` (the command is wired into help output)

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: lottie serve is the MCP stdio transport (Phase 4 slice 1)"
```

---

## Self-review notes

- **Spec coverage:** stdio binding (Task 4/5), one typed tool per agent (Task 2), output-structured + metrics-text return (Task 3), `ServeError`→`isError` mapping (Task 3), low-level Server (Task 2), optional `[serve]` extra + lazy import + install hint (Tasks 1, 5), broken-agent skip (Task 2), `validate_input=False` keeps `run_agent` the validation authority (Task 3), CLAUDE.md note (Task 6). Out-of-scope items (HTTP/`--port`, REST, OpenAI-compat, auth, streaming, resources/prompts, real security skills) are not implemented — correct.
- **`serve/__init__.py` untouched** on purpose: keeps `mcp` out of the base import path (the build-time crit the serve-core final review surfaced about foreign deps leaking).
- **Test count** is approximate; the gate asserts "all pass," not an exact number.
