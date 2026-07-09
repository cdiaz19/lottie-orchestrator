"""Pure helpers for distilling lessons into a template (no LLM call here).

The LLM call lives in the CLI so it uses the project's provider. This module builds
the prompt and does deterministic slot extraction/filling.
"""

from __future__ import annotations

import re

from lottie.llm import Message

DISTILL_SYSTEM_PROMPT = (
    "You synthesize ONE reusable prompt template from an agent's learned lessons. "
    "Output ONLY the template text — no commentary, no code fences. Use {slot} "
    "placeholders in snake_case for the parts that vary per invocation."
)

_SLOT_RE = re.compile(r"{(\w+)}")


def build_distill_prompt(notes: list[str]) -> list[Message]:
    """System+user prompt asking the LLM to synthesize a template from `notes`."""
    body = "Learned lessons:\n" + "\n".join(f"- {n}" for n in notes)
    return [
        Message(role="system", content=DISTILL_SYSTEM_PROMPT),
        Message(role="user", content=body),
    ]


def extract_slots(template: str) -> list[str]:
    """Sorted unique `{word}` slot names found in `template`."""
    return sorted(set(_SLOT_RE.findall(template)))


def fill_template(template: str, slots: dict[str, str]) -> str:
    """Replace each `{word}` with `slots[word]`; raise KeyError listing missing slots."""
    missing = [name for name in extract_slots(template) if name not in slots]
    if missing:
        raise KeyError(f"missing slots: {', '.join(missing)}")
    return _SLOT_RE.sub(lambda m: slots[m.group(1)], template)
