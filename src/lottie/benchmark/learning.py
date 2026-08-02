"""Measure whether learning actually helps — the evidence behind the default-on decision.

Runs an agent's eval suite twice on the same provider: once with recall OFF (baseline) and
once with recall ON (learning), then reports per-metric deltas.

Isolation
---------
Both arms disable every memory WRITE — trajectory persistence and reflection. Only recall
differs. This is deliberate and load-bearing: a benchmark that mutated the corpus it
measures would not be reproducible, and running it twice would silently report different
numbers. The measurement is read-only with respect to learned state.

The baseline achieves a clean namespace by disabling recall rather than by copying the
store to a scratch namespace. Same isolation, no chance of corrupting real memory, and
nothing to clean up if the run dies partway.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from lottie.benchmark.runner import load_suite, run_suite
from lottie.benchmark.schema import (
    LearningDeltaReport,
    MetricDelta,
    ProviderReport,
)
from lottie.llm import build_provider
from lottie.memory.schema import MemoryQuery, MemoryTier
from lottie.memory.store import build_memory_client
from lottie.project.config import AgentConfig, load_agent_config
from lottie.project.discovery import instantiate_agent, load_agent_class, load_input_model

#: (attribute on ProviderReport, higher-is-better). Accuracy is the primary metric;
#: the rest are reported so a quality gain bought with a large cost regression is visible
#: rather than hidden behind a single number.
_METRICS: list[tuple[str, bool]] = [
    ("accuracy", True),
    ("success_rate", True),
    ("mean_cost_usd", False),
    ("latency_p50_ms", False),
    ("latency_p95_ms", False),
    ("total_input_tokens", False),
    ("total_output_tokens", False),
]


def _arm_config(config: AgentConfig, *, recall: bool) -> AgentConfig:
    """Config for one arm: recall as requested, every memory WRITE off."""
    memory = config.memory.model_copy(
        update={
            "recall": config.memory.recall.model_copy(update={"enabled": recall}),
            "reflect": config.memory.reflect.model_copy(update={"enabled": False}),
            "trajectory": config.memory.trajectory.model_copy(update={"enabled": False}),
        }
    )
    return config.model_copy(update={"memory": memory})


def _delta(
    metric: str, baseline: ProviderReport, learning: ProviderReport, better_up: bool
) -> MetricDelta:
    base = float(getattr(baseline, metric))
    learned = float(getattr(learning, metric))
    return MetricDelta(
        metric=metric,
        baseline=base,
        learning=learned,
        delta=learned - base,
        pct_change=((learned - base) / base * 100) if base else None,
        higher_is_better=better_up,
    )


def _verdict(accuracy_delta: float) -> str:
    """The gate is accuracy: anything else is context, not a pass/fail."""
    if accuracy_delta > 0:
        return "improved"
    return "neutral" if accuracy_delta == 0 else "regressed"


def _recalled_notes(root: Path, config: AgentConfig, namespace: str) -> int:
    """How much learned context the learning arm actually had.

    Reported because a 'neutral' verdict means something very different when the store
    was empty (the experiment never ran) than when it held fifty notes (learning did not
    help). Best-effort — an unreadable store must not fail the benchmark.
    """
    if not config.memory.enabled:
        return 0
    try:
        client = build_memory_client(
            root, backend=config.memory.backend, path=config.memory.path
        )
        return len(
            client.recall(
                MemoryQuery(
                    text="", namespace=namespace, tier=MemoryTier.SEMANTIC, limit=1000
                )
            ).hits
        )
    except Exception:
        return 0


def learning_delta(root: Path, name: str, provider: str) -> LearningDeltaReport:
    """Run the suite twice — recall off, then recall on — and compare."""
    suite = load_suite(root, name)
    input_model: type[BaseModel] = load_input_model(root, name)
    agent_cls = load_agent_class(root, name)
    config = load_agent_config(root / "agents" / name)
    namespace = config.memory.namespace or name

    arms: dict[str, ProviderReport] = {}
    for arm, recall_on in (("baseline", False), ("learning", True)):
        agent = instantiate_agent(
            agent_cls,
            llm=build_provider(provider),
            root=root,
            config=_arm_config(config, recall=recall_on),
            enable_benchmarks=False,
        )
        arms[arm] = run_suite(agent, suite, input_model)

    deltas = [_delta(m, arms["baseline"], arms["learning"], up) for m, up in _METRICS]
    accuracy = next(d.delta for d in deltas if d.metric == "accuracy")
    return LearningDeltaReport(
        agent=name,
        provider=provider,
        namespace=namespace,
        recalled_notes=_recalled_notes(root, config, namespace),
        baseline=arms["baseline"],
        learning=arms["learning"],
        deltas=deltas,
        verdict=_verdict(accuracy),
    )


def write_delta_report(root: Path, report: LearningDeltaReport) -> Path:
    """Persist the machine-readable report that gates the default-on decision."""
    out = root / ".lottie" / "benchmarks" / f"{report.agent}-learning-delta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return out
