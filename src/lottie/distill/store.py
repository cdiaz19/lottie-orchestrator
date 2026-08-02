"""Read and write distilled-skill drafts under `skills/draft/<name>/`.

Agents write only to draft (rule 12's pattern, mirrored from knowledge). Promotion to a
registered skill is a human decision — S3c.

Every write is screened first. A distilled template is LLM-authored content derived from
trajectories that themselves originated in untrusted input, so it is exactly the kind of
payload rule 10 exists for: an unscreened template could carry injected instructions that
execute on every future invocation of the skill.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import yaml

from lottie.distill.schema import DistilledSkill, DistillProvenance, PromotionRecord
from lottie.security.content_gate import ContentGate, ContentRejected


class DraftRejected(ContentRejected):
    """A distilled draft failed its write-time security screen. Carries no content."""


class DraftNotFound(FileNotFoundError):
    """No draft exists for the requested skill name."""


def draft_gate() -> ContentGate:
    """The screen every distilled draft passes before touching disk."""
    return ContentGate(source="distill-draft", error=DraftRejected, label="draft write")


class InvalidSkillName(ValueError):
    """A skill name that would escape the drafts directory, or is otherwise unusable."""


#: Same shape as `DistilledSkill.name`. Enforced here too because the CLI reaches these
#: path helpers with a RAW `--name` before any pydantic model validates it.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")


def safe_name(name: str) -> str:
    """Validate a skill name before it is ever joined onto a path.

    `Path("/root/skills/draft") / "../../etc"` silently escapes, so `lottie distill show
    ../../something` would read and YAML-parse an arbitrary file. Credit to PR #35, which
    guarded this on the CLI; validating at the path chokepoint instead covers every caller.
    """
    if not _NAME_RE.match(name):
        raise InvalidSkillName(
            f"invalid skill name {name!r}: must match {_NAME_RE.pattern}"
        )
    return name


def draft_dir(root: Path, name: str) -> Path:
    return root / "skills" / "draft" / safe_name(name)


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


class NotPromotable(ValueError):
    """A draft cannot be promoted as requested."""


def promoted_dir(root: Path, name: str) -> Path:
    """Promoted distilled skills live beside — never inside — hand-written Python skills.

    A separate directory keeps discovery unambiguous and makes it structurally obvious
    that these are data, not modules (rule 13c).
    """
    return root / "skills" / "distilled" / safe_name(name)


def promote(root: Path, name: str, *, capability: str, reviewer: str) -> Path:
    """Human-approve a draft: re-screen, stamp the decision, move it to `skills/distilled/`.

    The content is screened AGAIN here, not just at authoring. A draft is a file on disk
    that a human or a process may have edited between `distill` and `review`; trusting the
    authoring-time screen would mean a promoted skill was never checked in the state it
    actually ships in.

    `capability` is supplied by the reviewer, never by the model — that is the whole point
    of the gate. The promoted skill carries it, so an agent must declare it (rule 11) in
    addition to `distilled`.
    """
    skill, prov = load_draft(root, name)
    if not capability:
        raise NotPromotable("a capability must be declared at promotion")

    approved = skill.model_copy(update={"capability": capability})
    draft_gate().check(
        "\n".join([approved.description, approved.system_prompt, approved.user_template])
    )

    target = promoted_dir(root, name)
    target.mkdir(parents=True, exist_ok=True)
    record = PromotionRecord(
        skill_name=name,
        capability=capability,
        reviewer=reviewer,
        source_version=skill.version,
        approved_at=time.time(),
    )
    (target / "template.yaml").write_text(
        yaml.safe_dump(approved.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "provenance.yaml").write_text(
        yaml.safe_dump(prov.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "promotion.yaml").write_text(
        yaml.safe_dump(record.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "SKILL.md").write_text(_render_skill_md(approved, prov), encoding="utf-8")
    shutil.rmtree(draft_dir(root, name))
    return target


def reject(root: Path, name: str) -> None:
    """Discard a draft. Rejection removes it — nothing unreviewed lingers as if approved."""
    target = draft_dir(root, name)
    if not (target / "template.yaml").is_file():
        raise DraftNotFound(f"no distilled draft named {name!r}")
    shutil.rmtree(target)


def load_promoted(root: Path, name: str) -> tuple[DistilledSkill, PromotionRecord]:
    """Read a promoted distilled skill and the decision that promoted it."""
    target = promoted_dir(root, name)
    template = target / "template.yaml"
    if not template.is_file():
        raise DraftNotFound(f"no promoted distilled skill named {name!r}")
    skill = DistilledSkill.model_validate(
        yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    )
    record = PromotionRecord.model_validate(
        yaml.safe_load((target / "promotion.yaml").read_text(encoding="utf-8")) or {}
    )
    return skill, record


def list_promoted(root: Path) -> list[str]:
    """Names of every promoted distilled skill, sorted."""
    base = root / "skills" / "distilled"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "template.yaml").is_file())
