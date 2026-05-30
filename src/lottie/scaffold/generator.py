"""Scaffold a complete agent or skill module from Jinja2 templates."""

from __future__ import annotations

import keyword
from pathlib import Path
from typing import Literal

import typer
from jinja2 import TemplateError

from lottie.scaffold.renderer import TemplateRendererSkill
from lottie.scaffold.schema import RenderContext, RenderInput

Kind = Literal["agent", "skill"]


def _validate_name(name: str) -> None:
    """Name must be a single lowercase snake_case Python identifier."""
    if (
        name in ("", ".", "..")
        or name != name.strip()
        or "/" in name
        or "\\" in name
        or not name.isidentifier()
        or not name.islower()
        or name.startswith("_")
        or keyword.iskeyword(name)
    ):
        raise typer.BadParameter(
            f"{name!r} is not a valid name — use a lowercase snake_case identifier "
            "(e.g. web_search)."
        )


def _class_name(name: str, kind: Kind) -> str:
    """web_search + skill -> WebSearchSkill."""
    pascal = "".join(part.capitalize() for part in name.split("_"))
    return f"{pascal}{kind.capitalize()}"


# (output relpath template, template name); "" template -> empty file.
_AGENT_PLAN: list[tuple[str, str]] = [
    ("__init__.py", ""),
    ("AGENT.md", "agent/AGENT.md.j2"),
    ("agent.py", "agent/agent.py.j2"),
    ("schema.py", "agent/schema.py.j2"),
    ("config.yaml", "agent/config.yaml.j2"),
    ("prompts.py", "agent/prompts.py.j2"),
    ("tests/__init__.py", ""),
    ("tests/test_{name}.py", "agent/test.py.j2"),
]
_SKILL_PLAN: list[tuple[str, str]] = [
    ("__init__.py", ""),
    ("SKILL.md", "skill/SKILL.md.j2"),
    ("skill.py", "skill/skill.py.j2"),
    ("schema.py", "skill/schema.py.j2"),
    ("tests/__init__.py", ""),
    ("tests/test_{name}.py", "skill/test.py.j2"),
]
_PLANS: dict[Kind, tuple[str, list[tuple[str, str]]]] = {
    "agent": ("agents", _AGENT_PLAN),
    "skill": ("skills", _SKILL_PLAN),
}


def generate(kind: Kind, name: str) -> Path:
    """Scaffold a complete `kind` module named `name`; return its directory."""
    _validate_name(name)
    class_name = _class_name(name, kind)
    root = _project_root()
    parent_dir, plan = _PLANS[kind]
    target = root / parent_dir / name
    _guard(target)

    context = RenderContext(name=name, class_name=class_name)
    renderer = TemplateRendererSkill(enable_benchmarks=False)
    files = _render_plan(plan, context, name, renderer)
    _write(target, files)
    _update_lottie_md(root, kind, class_name, f"{parent_dir}/{name}/")
    return target


def _project_root() -> Path:
    cwd = Path.cwd()
    if not (cwd / "lottie.yaml").exists():
        raise typer.BadParameter("not a Lottie project — run `lottie init` first.")
    return cwd


def _guard(target: Path) -> None:
    if target.exists() and not target.is_dir():
        raise typer.BadParameter(f"{target} already exists as a file — choose another name.")
    if target.is_dir() and any(target.iterdir()):
        raise typer.BadParameter(f"{target} already exists and is not empty — choose another name.")


def _render_plan(
    plan: list[tuple[str, str]],
    context: RenderContext,
    name: str,
    renderer: TemplateRendererSkill,
) -> dict[str, str]:
    """Render every template up front so a failure writes nothing."""
    files: dict[str, str] = {}
    for relpath, template in plan:
        out = relpath.format(name=name)
        if not template:
            files[out] = ""
            continue
        try:
            files[out] = renderer.run(RenderInput(template=template, context=context)).content
        except TemplateError as exc:
            raise typer.BadParameter(f"failed to render {template}: {exc}") from exc
    return files


def _write(target: Path, files: dict[str, str]) -> None:
    for relpath, content in files.items():
        path = target / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _update_lottie_md(root: Path, kind: Kind, class_name: str, location: str) -> None:
    """Register the new unit under its LOTTIE.md section. No-op if the file is absent."""
    md = root / "LOTTIE.md"
    if not md.exists():
        return
    heading = "## Agents" if kind == "agent" else "## Skills"
    entry = f"- **{class_name}** — `{location}`"
    lines = md.read_text(encoding="utf-8").splitlines()
    try:
        h = lines.index(heading)
    except ValueError:
        return
    k = h + 1
    while k < len(lines) and lines[k].strip() == "":  # skip blank lines after heading
        k += 1
    if k < len(lines) and lines[k].lstrip().startswith("_None yet"):
        lines[k] = entry  # replace the placeholder
    else:
        lines.insert(k, entry)  # prepend above the existing list
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
