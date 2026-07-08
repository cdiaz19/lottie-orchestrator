# S5 Design — HITL edited_input-on-approve

- Date: 2026-07-08 · Epic: v1.0.0 S5 · Lab R19 · Closes: edited_input accepted-not-applied.

## Goal
Today `ApprovalDecision.edited_input` is accepted but ignored on approve (langgraph_engine.resume
line ~184: "intentionally not applied"). Apply it — the human-edited values reach the resumed
worker — with typed, fail-closed validation.

## Grounding
- Mesh workers operate on `MeshState` (`task: str`, `history: Annotated[list, operator.add]`,
  `final: str|None`). Editable string fields = `task`, `final`.
- `resume()` (langgraph_engine): on reject it records a StepResult via `update_state(as_node)`;
  on approve it `graph.invoke(None, config)` to continue the interrupted node.
- `ApprovalDecision.edited_input: dict[str,str]`.
- serve: `service.resume_agent` converts ResumeDecision→ApprovalDecision; rest resume route maps
  typed errors. `InvalidInputError` already → 400 in the run route.

## Design
On approve WITH non-empty `edited_input`, before `graph.invoke(None, config)`:
1. **Validate (fail-closed):** every key must be an editable `MeshState` field
   (`task`/`final` — string fields only; `history` is reducer-managed, not editable). Unknown or
   non-editable key → `EditedInputError`. Then `MeshState.model_validate({**snap.values,
   **edited_input})` — any type/shape violation → `EditedInputError`.
2. **Apply:** `graph.update_state(config, dict(edited_input))` so the checkpoint state carries the
   edits; the resumed worker then runs on the edited state. (Uses `update_state`, the same
   mechanism the reject path already uses — the codebase idiom for LangGraph state mutation;
   equivalent to `Command(update=...)`.)
3. `graph.invoke(None, config)` as today.

`EditedInputError(MeshError)` in mesh/errors. `service.resume_agent` catches it → raises
`InvalidInputError` (→ 400). rest resume route gains `except InvalidInputError → 400
invalid_request`. Empty edited_input → unchanged (plain approve, back-compat).

## Tests (grow from 931)
- Engine: approve+edit(task) → resumed worker sees the edited task in its output/final; approve
  with empty edit → unchanged; edit with unknown key → EditedInputError; edit with wrong type
  (history) → EditedInputError.
- Service/serve: resume with a bad edit → 400 invalid_request (REST); good edit → 200 with edited
  result. Existing resume tests stay green.

## Files
- mesh/errors.py (+EditedInputError), mesh/langgraph_engine.py (apply), serve/service.py (map),
  serve/rest_app.py (400). Tests alongside.

## Risks
- Only top-level string MeshState fields are editable (task/final). Editing history/structured
  worker inputs is out of scope (documented). LocalEngine has no HITL (unchanged).
