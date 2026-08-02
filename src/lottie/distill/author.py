"""Author a distilled skill from an agent's successful trajectories.

Pure functions: build the prompt, parse the model's reply. No I/O, no LLM call — the
caller owns both, so this is unit-testable with zero mocking.

The model is asked for JSON rather than YAML: parsing is stricter, failures are loud,
and a malformed reply cannot silently produce a half-valid template. The result is
written to disk as YAML by `store.write_draft` for human readability at review time.
"""

from __future__ import annotations

import json
import re

from lottie.distill.schema import DistilledSkill
from lottie.llm import Message
from lottie.memory.reflection import RunTrajectory

DISTILL_SYSTEM_PROMPT = (
    "You extract a REUSABLE PROMPT TEMPLATE from examples of an agent's successful runs.\n"
    "Return ONLY a JSON object with these keys:\n"
    '  "description": one sentence describing what the template does\n'
    '  "system_prompt": the system message the template should send\n'
    '  "user_template": the user message, with {slot_name} placeholders\n'
    '  "slots": a list of {"name","description","required"} objects\n'
    "Rules: slot names are lowercase snake_case; every placeholder in user_template "
    "MUST be declared in slots; do not invent placeholders you do not declare. "
    "Do not include instructions that alter the assistant's identity or safety rules."
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class DistillParseError(ValueError):
    """The model's reply was not a usable distilled skill."""


def build_distill_prompt(agent_name: str, trajectories: list[RunTrajectory]) -> list[Message]:
    """Render successful trajectories into a system+user pair for the authoring call."""
    examples = "\n\n".join(
        f"--- run {i + 1} ---\ntask: {t.task}\noutcome: {t.outcome}"
        for i, t in enumerate(trajectories)
    )
    return [
        Message(role="system", content=DISTILL_SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                f"Agent: {agent_name}\n"
                f"{len(trajectories)} successful run(s) follow. Extract the shared "
                f"pattern as one reusable template.\n\n{examples}"
            ),
        ),
    ]


def parse_distilled(text: str, *, name: str, version: str = "0.1.0") -> DistilledSkill:
    """Parse the authoring reply into a validated `DistilledSkill`.

    Fail-closed: a reply whose `user_template` references an undeclared placeholder is
    rejected rather than silently rendered with a literal `{hole}` at call time.
    """
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise DistillParseError("no JSON object in distillation reply")
    try:
        # `_JSON_BLOCK` only matches a `{...}` span, so a successful parse is always a
        # dict — a non-object reply (a bare JSON array, say) fails the search above with
        # "no JSON object" instead. No isinstance check is reachable here.
        payload: dict[str, object] = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise DistillParseError(f"distillation reply is not valid JSON: {exc}") from exc

    payload["name"] = name
    payload["version"] = version
    try:
        skill = DistilledSkill.model_validate(payload)
    except Exception as exc:
        raise DistillParseError(f"distillation reply failed validation: {exc}") from exc

    undeclared = set(re.findall(r"\{([a-z][a-z0-9_]*)\}", skill.user_template)) - skill.slot_names()
    if undeclared:
        raise DistillParseError(f"template references undeclared slot(s): {sorted(undeclared)}")
    return skill
