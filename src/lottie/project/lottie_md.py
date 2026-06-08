"""Regenerate the LOTTIE.md registry sections from on-disk discovery.

`sync` rebuilds the `## Agents` and `## Skills` sections wholesale from what is
actually present, so `lottie status` keeps LOTTIE.md current. Filesystem-only
(via discovery) — never imports user code — and idempotent.
"""

from __future__ import annotations

from pathlib import Path

from lottie.project.discovery import UnitInfo, discover_agents, discover_skills
from lottie.scaffold.generator import Kind, _class_name

_PLACEHOLDER = "_None yet_"
_AGENTS_HEADING = "## Agents"
_SKILLS_HEADING = "## Skills"


def sync(root: Path) -> None:
    """Rewrite LOTTIE.md's Agents/Skills sections from discovery. No-op if absent."""
    md = root / "LOTTIE.md"
    if not md.exists():
        return
    text = md.read_text(encoding="utf-8")
    text = _replace_section(text, _AGENTS_HEADING, _entries(discover_agents(root), "agent"))
    text = _replace_section(text, _SKILLS_HEADING, _entries(discover_skills(root), "skill"))
    md.write_text(text, encoding="utf-8")


def _entries(units: list[UnitInfo], kind: Kind) -> list[str]:
    if not units:
        return [_PLACEHOLDER]
    return [f"- **{_class_name(u.name, kind)}** — `{kind}s/{u.name}/`" for u in units]


def _replace_section(text: str, heading: str, body: list[str]) -> str:
    """Replace lines from `heading` until the next `## ` (or EOF) with heading+body.

    Assumes LOTTIE.md's structure: a `# title`, then the `## Agents` / `## Skills`
    registry sections, which are the trailing content. Content after the final
    section (a footer) or a `## ` inside a fenced code block is not modelled, so
    sync is meant only for this tool-managed file, not arbitrary Markdown.
    """
    lines = text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        return text
    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1
    rebuilt = [heading, "", *body, ""]
    new_lines = [*lines[:start], *rebuilt, *lines[end:]]
    return "\n".join(new_lines).rstrip("\n") + "\n"
