"""`lottie reflect <agent>` — manual episodic→semantic memory consolidation.

Runs the agent's MemoryAgent consolidation (S1-gated: content-screened, deduped,
provenance-stamped, audited) over its memory namespace. Distinct from the automatic
per-run reflection hook — this is the batch/manual curation entry point.
"""

from __future__ import annotations

from typing import Annotated

import typer

from lottie.llm import build_provider
from lottie.memory.agent import MemoryAgent  # NOT re-exported from lottie.memory (import cycle)
from lottie.memory.schema import ReflectionInput
from lottie.memory.store import build_memory_client
from lottie.project.config import find_project_root, load_agent_config


def reflect(
    name: str,
    namespace: Annotated[
        str | None, typer.Option("--namespace", help="Memory namespace (default: agent name).")
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max episodic records to consolidate.")
    ] = 50,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override the LLM provider.")
    ] = None,
) -> None:
    """Consolidate an agent's episodic memory into durable semantic notes."""
    root = find_project_root()
    unit_dir = root / "agents" / name
    if not (unit_dir / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    cfg = load_agent_config(unit_dir)
    llm = build_provider(provider or cfg.provider)
    memory = build_memory_client(root, backend=cfg.memory.backend, path=cfg.memory.path)
    ns = namespace or cfg.memory.namespace or name

    agent = MemoryAgent(llm=llm, memory=memory)
    result = agent.run(ReflectionInput(namespace=ns, limit=limit))

    typer.echo(
        f"reflected '{ns}': consolidated {result.consolidated_count} episodic record(s) "
        f"-> {len(result.written_ids)} semantic note(s)"
    )
    for note in result.notes:
        typer.echo(f"  - {note}")
