"""`lottie init` — scaffold a new Lottie project skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lottie.cli import templates
from lottie.project.lottie_md import sync
from lottie.scaffold.hello import HELLO_FILES


def init(
    name: str,
    here: Annotated[
        bool, typer.Option("--here", help="Scaffold into the current directory.")
    ] = False,
) -> None:
    """Scaffold a new Lottie project skeleton."""
    _validate_name(name)
    target = Path.cwd() if here else Path.cwd() / name
    _guard(target, here)
    _scaffold(target, name)
    typer.echo(f"Created Lottie project at {target}")
    if here:
        typer.echo("Next: lottie create agent <name>")
    else:
        typer.echo(f"Next: cd {name} && lottie create agent <name>")


def _validate_name(name: str) -> None:
    """Project name must be a single directory segment — no path escapes."""
    if name in ("", ".", "..") or name != Path(name).name:
        raise typer.BadParameter(
            f"{name!r} is not a valid project name — "
            "use a single directory name without path separators."
        )


def _guard(target: Path, here: bool) -> None:
    """Refuse to scaffold over an existing project — fail before any write."""
    if here:
        # --here intentionally allows a populated cwd (git repos, monorepos);
        # it refuses only when a lottie.yaml is already present.
        if (target / "lottie.yaml").exists():
            raise typer.BadParameter(
                f"{target} already contains a lottie.yaml — refusing to overwrite."
            )
        return
    if target.exists() and not target.is_dir():
        raise typer.BadParameter(
            f"{target} already exists as a file — choose another name."
        )
    if target.is_dir() and any(target.iterdir()):
        raise typer.BadParameter(
            f"{target} already exists and is not empty — choose another name."
        )


def _scaffold(target: Path, name: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "lottie.yaml").write_text(templates.LOTTIE_YAML.format(name=name), encoding="utf-8")
    (target / "LOTTIE.md").write_text(templates.LOTTIE_MD.format(name=name), encoding="utf-8")
    (target / ".gitignore").write_text(templates.GITIGNORE, encoding="utf-8")

    (target / "agents").mkdir(exist_ok=True)
    (target / "agents" / "__init__.py").write_text(templates.AGENTS_INIT, encoding="utf-8")
    (target / "skills").mkdir(exist_ok=True)
    (target / "skills" / "__init__.py").write_text(templates.SKILLS_INIT, encoding="utf-8")

    (target / "policies").mkdir(exist_ok=True)
    (target / "policies" / "base.yaml").write_text(templates.POLICY_BASE, encoding="utf-8")

    for layer in templates.KNOWLEDGE_LAYERS:
        layer_dir = target / "knowledge" / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / ".gitkeep").write_text("", encoding="utf-8")

    hello_dir = target / "agents" / "hello"
    for relpath, content in HELLO_FILES.items():
        path = hello_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    sync(target)
