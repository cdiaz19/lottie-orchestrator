"""Per-run instrumentation primitives shared by BaseAgent and BaseSkill.

`RunMetrics` is one record per `run()` call (distinct from the aggregate
benchmark report in the spec). Records append to
`.lottie/benchmarks/<name>.jsonl` in dev mode.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lottie.llm import TokenUsage

Kind = Literal["agent", "skill"]


class RunMetrics(BaseModel):
    """Metrics captured for a single agent or skill run."""

    name: str
    kind: Kind
    provider: str | None
    timestamp: datetime
    latency_ms: float
    success: bool
    version: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retry_count: int = 0
    error: str | None = None


@dataclass
class RunContext:
    """Mutable accumulator filled during a run (tokens, cost, retries)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retry_count: int = 0
    metrics: RunMetrics | None = field(default=None)

    def add_usage(self, usage: TokenUsage, cost_usd: float = 0.0) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost_usd += cost_usd


@lru_cache(maxsize=8)
def git_version(cwd: Path | None = None) -> str | None:
    """Short git commit hash tying metrics to code, or None outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def append_metrics(
    record: RunMetrics,
    root: Path,
    enabled: bool = True,
) -> Path | None:
    """Append `record` as one JSON line to the benchmarks file.

    Returns the file path, or None when benchmarking is disabled.
    """
    if not enabled:
        return None
    path = root / ".lottie" / "benchmarks" / f"{record.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
    return path
