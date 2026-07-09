"""Reflection primitives: distill a run trajectory into memory lessons.

Pure — the LLM call itself lives on BaseAgent (so it counts against the run's token
budget); this module only builds the prompt and parses the result. Imports only
lottie.llm (Message) + lottie.memory.schema, so it stays acyclic.
"""

from __future__ import annotations

from pydantic import BaseModel

from lottie.llm import Message
from lottie.memory.schema import DeltaOp, MemoryDelta

REFLECT_SYSTEM_PROMPT = (
    "You are a reflection step run after an agent finished a task. Read the execution "
    "trajectory and distill at most a few DURABLE, reusable lessons that would help a "
    "future run of this agent do better. Each lesson: one line, standalone, imperative, "
    "no numbering or bullets. If there is no durable lesson, output nothing."
)


class RunTrajectory(BaseModel):
    """A minimal record of one execution, fed to the Reflector."""

    task: str
    outcome: str
    success: bool
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


def build_reflection_prompt(trajectory: RunTrajectory) -> list[Message]:
    """Render the trajectory into a system+user message pair for the reflection call."""
    body = (
        f"task: {trajectory.task}\n"
        f"success: {trajectory.success}\n"
        f"outcome: {trajectory.outcome}\n"
        f"error: {trajectory.error or 'none'}\n"
        f"tokens: {trajectory.input_tokens + trajectory.output_tokens}  "
        f"cost_usd: {trajectory.cost_usd}"
    )
    return [
        Message(role="system", content=REFLECT_SYSTEM_PROMPT),
        Message(role="user", content=body),
    ]


def parse_reflection(text: str) -> list[MemoryDelta]:
    """One non-blank output line → one ADD delta tagged 'reflection'."""
    return [
        MemoryDelta(op=DeltaOp.ADD, content=line.strip(), tags=["reflection"])
        for line in text.splitlines()
        if line.strip()
    ]
