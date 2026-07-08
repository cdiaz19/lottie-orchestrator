"""`lottie doctor` — environment health checks (no live network)."""

from __future__ import annotations

import importlib.util
import os
import sys

import typer
from rich.console import Console

from lottie.project.config import find_project_root, load_lottie_config

# Provider prefix -> required env var. None means no key needed (e.g. local).
_PROVIDER_ENV: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": None,
}

# (label, ok, detail)
Check = tuple[str, bool, str]


def doctor() -> None:
    """Check environment health — Python, deps, project, API keys."""
    checks: list[Check] = []
    warnings: list[str] = []

    py_ok = sys.version_info >= (3, 12)
    checks.append(("Python >= 3.12", py_ok, f"{sys.version_info.major}.{sys.version_info.minor}"))

    for dep in ("litellm", "jinja2", "pydantic", "yaml"):
        ok = importlib.util.find_spec(dep) is not None
        checks.append((f"dep: {dep}", ok, "installed" if ok else "MISSING"))

    project_checks, project_warnings = _project_checks()
    checks.extend(project_checks)
    warnings.extend(project_warnings)
    warnings.extend(_hardening_warnings())

    _render(Console(), checks, warnings)
    if any(not ok for _, ok, _ in checks):
        raise typer.Exit(1)


def _project_checks() -> tuple[list[Check], list[str]]:
    checks: list[Check] = []
    warnings: list[str] = []
    try:
        root = find_project_root()
    except typer.BadParameter:
        warnings.append("not in a Lottie project — skipping project checks")
        return checks, warnings
    checks.append(("Lottie project", True, str(root)))
    cfg = load_lottie_config(root)
    models = [cfg.providers.default]
    if cfg.providers.fallback:
        models.append(cfg.providers.fallback)
    for model in models:
        prefix = model.split("/")[0]
        if prefix not in _PROVIDER_ENV:
            warnings.append(f"unknown provider '{prefix}' — set its API key manually")
            continue
        env = _PROVIDER_ENV[prefix]
        if env is None:
            checks.append((f"key: {prefix}", True, "no key needed"))
        else:
            present = bool(os.environ.get(env))
            checks.append((f"key: {env}", present, "set" if present else "MISSING"))
    return checks, warnings


def _hardening_warnings() -> list[str]:
    """Advisory checks for the v1 HTTP-hardening config (all opt-in; unset = off)."""
    warnings: list[str] = []
    if not os.environ.get("LOTTIE_API_KEYS"):
        warnings.append(
            "LOTTIE_API_KEYS unset — the HTTP transport (lottie serve --port) accepts "
            "unauthenticated requests. Set it to require an API key in production."
        )
    if not os.environ.get("LOTTIE_RATE_LIMIT_PER_MIN"):
        warnings.append(
            "LOTTIE_RATE_LIMIT_PER_MIN unset — HTTP rate limiting is disabled."
        )
    return warnings


def _render(console: Console, checks: list[Check], warnings: list[str]) -> None:
    for label, ok, detail in checks:
        mark = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"{mark}  {label}  ({detail})")
    for warning in warnings:
        console.print(f"[yellow]WARN[/yellow]  {warning}")
