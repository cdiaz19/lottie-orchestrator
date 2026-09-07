"""`lottie modules` — show what actually wraps an agent's runs (V3 S6).

The chain is otherwise invisible: an operator can read `config.yaml` and infer what
*should* be mounted, but inference is exactly how a disabled security gate goes unnoticed.
This prints what is really there, in execution order.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lottie.core.middleware import KNOWN_MODULES, build_chain
from lottie.llm import MockLLMProvider
from lottie.project.config import AgentConfig, find_project_root, load_agent_config
from lottie.project.discovery import discover_agents, instantiate_agent, load_agent_class


def modules(
    name: Annotated[
        str | None, typer.Argument(help="Agent to inspect. Omit to list every agent.")
    ] = None,
) -> None:
    """Show the runtime modules mounted on an agent, in chain order."""
    root = find_project_root()
    names = [name] if name else sorted(u.name for u in discover_agents(root))
    if name and not (root / "agents" / name / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    console = Console()
    for agent_name in names:
        cfg = load_agent_config(root / "agents" / agent_name)

        # Config-derived facts print FIRST, before any attempt to instantiate. A broken
        # plugin makes instantiation fail, and that is exactly when an operator most needs
        # to see what is configured — a diagnostic that only works when things are fine is
        # not a diagnostic.
        _report_config(console, cfg)

        try:
            agent = instantiate_agent(
                load_agent_class(root, agent_name),
                # A mock provider: this command inspects composition, never runs the agent,
                # and must not require an API key to answer "what is mounted?".
                llm=MockLLMProvider(responses=["unused"]),
                root=root,
                config=cfg,
                enable_benchmarks=False,
            )
        except Exception as exc:
            console.print(f"[red]{agent_name}[/red]: cannot inspect — {exc}")
            continue

        disabled = {n for n, m in cfg.modules.items() if not m.enabled}
        mounted = agent.mounted_modules()
        chain = {m.name: m.order for m in build_chain(agent, frozenset(disabled))}

        table = Table(title=f"{agent_name} — {len(mounted)} module(s) mounted")
        table.add_column("Order", justify="right")
        table.add_column("Module")
        table.add_column("State")
        for module_name in mounted:
            table.add_row(str(chain[module_name]), module_name, "[green]on[/green]")
        for module_name in sorted(disabled):
            table.add_row("—", module_name, "[yellow]disabled[/yellow]")
        console.print(table)




def _report_config(console: Console, cfg: AgentConfig) -> None:
    """Print what the config declares, independent of whether the agent can be built."""
    if cfg.plugins:
        # Third-party code in this process. An operator should never have to read source
        # to find out what is loaded, or on what terms.
        console.print(
            f"  [yellow]plugins[/yellow] ({len(cfg.plugins)}): "
            + ", ".join(p.module for p in cfg.plugins)
        )
        console.print(
            "  [dim]plugins observe only — they cannot intercept a run, and events carry "
            "hashes, never raw content. They are NOT sandboxed.[/dim]"
        )

    unknown = sorted(set(cfg.modules) - set(KNOWN_MODULES))
    if unknown:
        console.print(
            f"[yellow]WARN[/yellow]  unknown module name(s) in config: "
            f"{', '.join(unknown)} — this config line does nothing"
        )
