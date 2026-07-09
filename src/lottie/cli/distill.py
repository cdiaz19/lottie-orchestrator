"""`lottie distill <agent>` — synthesize a draft template-skill from learned memory.

Recalls the agent's SEMANTIC notes, asks the LLM to synthesize ONE reusable prompt
template, secret-scans it (templates are legitimately instructional, so injection
review is the human promotion gate in S3b), and writes a NON-callable draft under
skills/draft/<name>/. Promotion to a runnable skill is `lottie distill review` (S3b).
"""

from __future__ import annotations

from typing import Annotated

import typer

from lottie.distill.generate import build_distill_prompt, extract_slots
from lottie.distill.io import save_draft
from lottie.distill.schema import DistilledSkillSpec, DistillProvenance
from lottie.llm import build_provider
from lottie.memory.schema import MemoryQuery, MemoryTier
from lottie.memory.store import build_memory_client
from lottie.project.config import find_project_root, load_agent_config
from lottie.security import SecretDetectionSkill


def distill(
    name: str,
    namespace: Annotated[str | None, typer.Option("--namespace")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
    skill_name: Annotated[str | None, typer.Option("--skill-name")] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
) -> None:
    """Distill an agent's learned lessons into a draft template-skill."""
    root = find_project_root()
    unit_dir = root / "agents" / name
    if not (unit_dir / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    cfg = load_agent_config(unit_dir)
    llm = build_provider(provider or cfg.provider)
    memory = build_memory_client(root, backend=cfg.memory.backend, path=cfg.memory.path)
    ns = namespace or cfg.memory.namespace or name

    hits = memory.recall(
        MemoryQuery(text="", namespace=ns, tier=MemoryTier.SEMANTIC, limit=limit)
    ).hits
    notes = [h.record.content for h in hits]
    if not notes:
        typer.echo(f"no semantic notes in namespace '{ns}' to distill", err=True)
        raise typer.Exit(code=1)

    template = llm.complete(build_distill_prompt(notes)).content.strip()
    if SecretDetectionSkill().scan_text(template, source="distill"):
        typer.echo("distillation rejected: secret detected in generated template", err=True)
        raise typer.Exit(code=2)

    sname = skill_name or f"{name}_distilled"
    run_ids = sorted({h.record.run_id for h in hits if h.record.run_id})
    spec = DistilledSkillSpec(name=sname, template=template, slots=extract_slots(template))
    provenance = DistillProvenance(source_agent=name, namespace=ns, source_run_ids=run_ids)
    draft_dir = save_draft(root, spec, provenance)

    typer.echo(
        f"distilled draft skill '{sname}' -> {draft_dir} (slots: {spec.slots or 'none'}). "
        "Review and promote to enable it."
    )
