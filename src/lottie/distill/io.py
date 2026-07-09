"""On-disk persistence for distilled draft skills.

A draft lives in `skills/draft/<name>/` as `template.yaml` + `provenance.yaml` +
`SKILL.md`. It has NO `skill.py`, so `discover_skills` never treats it as callable —
promotion (S3b) is what makes a distilled skill runnable.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lottie.distill.schema import DistilledSkillSpec, DistillProvenance


def save_draft(root: Path, spec: DistilledSkillSpec, provenance: DistillProvenance) -> Path:
    """Write a distilled draft under `<root>/skills/draft/<spec.name>/`; return the dir."""
    skill_dir = Path(root) / "skills" / "draft" / spec.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "template.yaml").write_text(
        yaml.safe_dump(spec.model_dump(), sort_keys=True), encoding="utf-8"
    )
    (skill_dir / "provenance.yaml").write_text(
        yaml.safe_dump(provenance.model_dump(), sort_keys=True), encoding="utf-8"
    )
    (skill_dir / "SKILL.md").write_text(
        f"# {spec.name} (distilled draft)\n\n"
        f"Parameterized prompt-template skill, version {spec.version}.\n\n"
        f"**Slots:** {', '.join(spec.slots) or '(none)'}\n\n"
        f"**Provenance:** distilled from agent `{provenance.source_agent}`, "
        f"namespace `{provenance.namespace}`.\n\n"
        "Draft — not callable until a human promotes it (`lottie distill review`).\n",
        encoding="utf-8",
    )
    return skill_dir


def load_spec(skill_dir: Path) -> DistilledSkillSpec:
    """Load the `DistilledSkillSpec` from a draft/promoted skill directory."""
    raw = yaml.safe_load((Path(skill_dir) / "template.yaml").read_text(encoding="utf-8"))
    return DistilledSkillSpec.model_validate(raw)
