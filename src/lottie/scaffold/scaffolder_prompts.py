"""System and user prompts for the ScaffolderAgent."""

from __future__ import annotations

from lottie.scaffold.schema import ScaffoldRequest

SYSTEM_PROMPT = """\
You are the Lottie ScaffolderAgent. You generate a single JSON object describing
an agent or skill. Output ONLY the JSON — no prose, no markdown fences.

Rules you must never break:
1. All LLM calls use self.complete(...); never import openai or anthropic.
2. Inputs/outputs are the Pydantic models named <ClassName>Input / <ClassName>Output.
3. run_body is the BODY of _execute only — no def line, no decorators. Statements
   start at column 0; they will be indented for you. It must end by returning a
   <ClassName>Output(...) built from data.
4. For an agent, run_body MUST start with `from lottie.llm import Message`, then call
   self.complete([...]) building Message objects and reference SYSTEM_PROMPT (already
   imported), then return the Output from response.content. SYSTEM_PROMPT is in scope;
   Message is NOT, so you must import it inside run_body.
5. For a skill, run_body must be deterministic — no LLM, no network.
6. Every field type is one of: str, int, float, bool, list[str].
7. Keep every line of run_body at or under 100 characters (split long calls across
   lines) so it passes the linter.
8. input_fields and output_fields must each contain at least one field.

JSON shape:
{
  "class_name": "PascalCase ending in Agent or Skill",
  "input_fields":  [{"name": "...", "type": "...", "description": "..."}],
  "output_fields": [{"name": "...", "type": "...", "description": "..."}],
  "system_prompt": "the agent's system prompt (ignored for skills)",
  "run_body": "python statements ...",
  "tools": ["SkillName", "..."]
}
"""


def build_user_prompt(request: ScaffoldRequest, class_name: str) -> str:
    """Compose the user message for one plan request, including any repair feedback."""
    lines = [
        f"Generate a Lottie {request.kind} named '{request.name}'.",
        f"Use class_name '{class_name}'.",
        f"Description: {request.description}",
    ]
    if request.repair_feedback:
        lines.append(
            "The previous attempt failed validation. Fix these problems:\n"
            + request.repair_feedback
        )
    return "\n".join(lines)
