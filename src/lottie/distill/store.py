"""Read and write distilled-skill drafts under `skills/draft/<name>/`.

Agents write only to draft (rule 12's pattern, mirrored from knowledge). Promotion to a
registered skill is a human decision — S3c.

Every write is screened first. A distilled template is LLM-authored content derived from
trajectories that themselves originated in untrusted input, so it is exactly the kind of
payload rule 10 exists for: an unscreened template could carry injected instructions that
execute on every future invocation of the skill.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from lottie.distill.schema import DistilledSkill, DistillProvenance
from lottie.security.content_gate import ContentGate, ContentRejected


class DraftRejected(ContentRejected):
    """A distilled draft failed its write-time security screen. Carries no content."""


class DraftNotFound(FileNotFoundError):
    """No draft exists for the requested skill name."""


def draft_gate() -> ContentGate:
    """The screen every distilled draft passes before touching disk."""
    return ContentGate(source="distill-draft", error=DraftRejected, label="draft write")


def draft_dir(root: Path, name: str) -> Path:
    return root / "skills" / "draft" / name


def bump_minor(version: str) -> str:
    """0.1.0 -> 0.2.0. Re-distilling an existing skill produces a new minor version."""
    major, minor, _patch = (int(p) for p in version.split("."))
    return f"{major}.{minor + 1}.0"


def existing_version(root: Path, name: str) -> str | None:
    """The version already on disk for `name`, or None when there is no draft."""
    path = draft_dir(root, name) / "template.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    version = data.get("version")
    return str(version) if version else None


def _render_skill_md(skill: DistilledSkill, prov: DistillProvenance) -> str:
    rows = "\n".join(
        f"| `{s.name}` | {'yes' if s.required else 'no'} | {s.description} |" for s in skill.slots
    ) or "| _(none)_ | | |"
    return (
        f"# {skill.name} (distilled, v{skill.version})\n\n"
        f"> **DRAFT — not registered.** Distilled from {prov.trajectory_count} successful "
        f"run(s) of `{prov.source_agent}`. Executed by `TemplateRunnerSkill`; this is a "
        f"prompt template, not code. Promote with `lottie distill review`.\n\n"
        f"## What it does\n\n{skill.description}\n\n"
        f"## Slots\n\n| Name | Required | Description |\n|---|---|---|\n{rows}\n\n"
        f"## Provenance\n\n"
        f"- Source agent: `{prov.source_agent}`\n"
        f"- Trajectories: {prov.trajectory_count}\n"
        f"- Version: {skill.version}\n"
    )


def write_draft(root: Path, skill: DistilledSkill, prov: DistillProvenance) -> Path:
    """Screen and write a draft to `skills/draft/<name>/`. Returns the directory.

    The screen covers the description, system prompt, and user template together — an
    injection split across two fields would evade a per-field check.
    """
    draft_gate().check("\n".join([skill.description, skill.system_prompt, skill.user_template]))

    target = draft_dir(root, skill.name)
    target.mkdir(parents=True, exist_ok=True)
    stamped = prov.model_copy(update={"created_at": time.time(), "version": skill.version})

    (target / "template.yaml").write_text(
        yaml.safe_dump(skill.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (target / "provenance.yaml").write_text(
        yaml.safe_dump(stamped.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (target / "SKILL.md").write_text(_render_skill_md(skill, stamped), encoding="utf-8")
    return target


def load_draft(root: Path, name: str) -> tuple[DistilledSkill, DistillProvenance]:
    """Read a draft back off disk."""
    target = draft_dir(root, name)
    template = target / "template.yaml"
    if not template.is_file():
        raise DraftNotFound(f"no distilled draft named {name!r}")
    skill = DistilledSkill.model_validate(
        yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    )
    prov_path = target / "provenance.yaml"
    prov = DistillProvenance.model_validate(
        yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
        if prov_path.is_file()
        else {"source_agent": "unknown", "trajectory_count": 0, "version": skill.version}
    )
    return skill, prov


def list_drafts(root: Path) -> list[str]:
    """Names of every distilled draft, sorted."""
    base = root / "skills" / "draft"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "template.yaml").is_file())
