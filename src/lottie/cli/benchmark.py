"""`lottie benchmark agent <name>` — run an agent's eval suite and report."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lottie.benchmark.learning import learning_delta, write_delta_report
from lottie.benchmark.runner import benchmark
from lottie.benchmark.schema import LearningDeltaReport, ProviderReport
from lottie.project.config import find_project_root, load_agent_config, load_lottie_config

# Wide fixed width so long provider names (e.g. "anthropic/claude-sonnet-4-6")
# aren't truncated by Rich in headless/CI terminals (default width 80 truncates).
_REPORT_WIDTH = 200

benchmark_app = typer.Typer(
    help="Benchmark agents against eval suites.", no_args_is_help=True
)


@benchmark_app.command("agent")
def benchmark_agent(
    name: str,
    compare: Annotated[
        bool,
        typer.Option("--compare", help="Run across all configured providers."),
    ] = False,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Benchmark a single provider."),
    ] = None,
    learning: Annotated[
        bool,
        typer.Option(
            "--learning-delta",
            help="Run the suite with recall off, then on, and report the difference.",
        ),
    ] = False,
) -> None:
    """Run agents/<name>/evals.yaml and print + persist a report."""
    root = find_project_root()
    if not (root / "agents" / name / "agent.py").is_file():
        raise typer.BadParameter(f"agent '{name}' not found")

    if learning:
        if compare:
            raise typer.BadParameter(
                "--learning-delta compares recall on vs off on ONE provider; "
                "--compare varies the provider instead. Use one or the other."
            )
        _run_learning_delta(root, name, provider)
        return

    providers = _resolve_providers(root, name, compare, provider)
    report = benchmark(root, name, providers)

    _print_table(report.providers)
    out = root / ".lottie" / "benchmarks" / f"{name}-report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Report written to {out}")


def _resolve_providers(
    root: Path, name: str, compare: bool, provider: str | None
) -> list[str]:
    if provider is not None:
        return [provider]
    if compare:
        cfg = load_lottie_config(root)
        seen: list[str] = []
        for p in (cfg.providers.default, cfg.providers.fallback):
            if p and p not in seen:
                seen.append(p)
        return seen
    return [load_agent_config(root / "agents" / name).provider]


def _print_table(reports: list[ProviderReport]) -> None:
    table = Table(title="Benchmark")
    for col in (
        "provider",
        "cases",
        "accuracy",
        "success",
        "p50 ms",
        "p95 ms",
        "mean $",
        "in/out tok",
    ):
        table.add_column(col)
    for r in reports:
        table.add_row(
            r.provider,
            str(r.case_count),
            f"{r.accuracy:.0%}",
            f"{r.success_rate:.0%}",
            f"{r.latency_p50_ms:.1f}",
            f"{r.latency_p95_ms:.1f}",
            f"{r.mean_cost_usd:.4f}",
            f"{r.total_input_tokens}/{r.total_output_tokens}",
        )
    Console(width=_REPORT_WIDTH).print(table)


def _run_learning_delta(root: Path, name: str, provider: str | None) -> None:
    """Run both arms, print the comparison, and persist the machine-readable report."""
    chosen = provider or load_agent_config(root / "agents" / name).provider
    report = learning_delta(root, name, chosen)
    _print_delta_table(report)
    out = write_delta_report(root, report)
    typer.echo(f"Learning-delta report written to {out}")


def _print_delta_table(report: LearningDeltaReport) -> None:
    table = Table(
        title=(
            f"{report.agent} — learning delta ({report.provider}) · "
            f"verdict: {report.verdict} · {report.recalled_notes} note(s) recalled"
        ),
        width=_REPORT_WIDTH,
    )
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Learning", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Change", justify="right")
    for d in report.deltas:
        # An improvement is a rise for accuracy and a fall for cost/latency/tokens.
        good = d.delta > 0 if d.higher_is_better else d.delta < 0
        marker = "" if d.delta == 0 else ("[green]" if good else "[red]")
        close = "" if d.delta == 0 else ("[/green]" if good else "[/red]")
        pct = "—" if d.pct_change is None else f"{d.pct_change:+.1f}%"
        table.add_row(
            d.metric,
            f"{d.baseline:.4g}",
            f"{d.learning:.4g}",
            f"{marker}{d.delta:+.4g}{close}",
            pct,
        )
    Console(width=_REPORT_WIDTH).print(table)
