"""`lottie distill <agent>` — author a reusable skill template from successful runs.

Reads the agent's EPISODIC trajectories (written by the V2 S3a hook), asks the LLM to
extract the shared pattern as a parameterized prompt template, and writes it to
`skills/draft/<name>/` after a security screen.

The output is a template, never Python — nothing authored here is ever imported or
executed, only rendered by `TemplateRunnerSkill`. Promotion to a registered skill is a
human decision (`lottie distill review`, S3c).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from lottie.distill.author import DistillParseError, build_distill_prompt, parse_distilled
from lottie.distill.schema import DistillProvenance
from lottie.distill.store import DraftRejected, bump_minor, existing_version, write_draft
from lottie.llm import build_provider
from lottie.memory.reflection import RunTrajectory
from lottie.memory.schema import MemoryQuery, MemoryTier
from lottie.memory.store import build_memory_client
from lottie.project.config import AgentConfig, find_project_root, load_agent_config

distill_app = typer.Typer(help="Distil an agent's successful runs into reusable skills.")


def _trajectories(
    root: Path, cfg: AgentConfig, namespace: str, limit: int
) -> list[RunTrajectory]:
    """Load successful trajectories from the agent's episodic tier."""
    memory = build_memory_client(root, backend=cfg.memory.backend, path=cfg.memory.path)
    hits = memory.recall(
        MemoryQuery(
            text="",
            namespace=namespace,
            tier=MemoryTier.EPISODIC,
            tags=["success"],
            limit=limit,
        )
    ).hits
    out: list[RunTrajectory] = []
    for hit in hits:
        try:
            out.append(RunTrajectory.model_validate_json(hit.record.content))
        except Exception:  # a non-trajectory episodic record — skip, never fail the run
            continue
    return out


@distill_app.command("run")
def distill(
    name: str,
    skill_name: Annotated[
        str | None, typer.Option("--name", help="Draft skill name (default: <agent>_distilled).")
    ] = None,
    namespace: Annotated[
        str | None, typer.Option("--namespace", help="Memory namespace (default: agent name).")
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max successful trajectories to distil from.")
    ] = 20,
    provider: Annotated[
        str | None, typer.Option("--provider", help="Override the LLM provider.")
    ] = None,
) -> None:
    """Distil an agent's successful trajectories into a draft skill template."""
    root = find_project_root()
    unit_dir = root / "agents" / name
    if not (unit_dir / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    cfg = load_agent_config(unit_dir)
    ns = namespace or cfg.memory.namespace or name
    trajectories = _trajectories(root, cfg, ns, limit)
    if not trajectories:
        raise typer.BadParameter(
            f"no successful trajectories in namespace '{ns}'. "
            "Enable memory.trajectory for this agent and run it first."
        )

    target_name = skill_name or f"{name}_distilled"
    prior = existing_version(root, target_name)
    version = bump_minor(prior) if prior else "0.1.0"

    llm = build_provider(provider or cfg.provider)
    response = llm.complete(build_distill_prompt(name, trajectories))
    try:
        skill = parse_distilled(response.content, name=target_name, version=version)
    except DistillParseError as exc:
        raise typer.BadParameter(f"distillation failed: {exc}") from exc

    prov = DistillProvenance(
        source_agent=name,
        trajectory_count=len(trajectories),
        version=version,
        run_ids=[],
    )
    try:
        target = write_draft(root, skill, prov)
    except DraftRejected as exc:
        raise typer.BadParameter(f"draft rejected by the security gate: {exc}") from exc

    typer.echo(
        f"distilled '{target_name}' v{version} from {len(trajectories)} run(s) -> "
        f"{target.relative_to(root)}"
    )
    typer.echo(f"  slots: {', '.join(sorted(skill.slot_names())) or '(none)'}")
    typer.echo("  DRAFT — review and promote with `lottie distill review`")


@distill_app.command("show")
def show(name: str) -> None:
    """Print a distilled draft's template as JSON."""
    from lottie.distill.store import load_draft

    root = find_project_root()
    skill, prov = load_draft(root, name)
    typer.echo(json.dumps({"skill": skill.model_dump(), "provenance": prov.model_dump()}, indent=2))


@distill_app.command("list")
def list_command() -> None:
    """List distilled drafts awaiting review."""
    from lottie.distill.store import list_drafts

    root = find_project_root()
    names = list_drafts(root)
    if not names:
        typer.echo("no distilled drafts")
        return
    for draft in names:
        version = existing_version(root, draft) or "?"
        typer.echo(f"{draft}  v{version}  (draft)")
