# S5 Plan — HITL edited_input

Design: 2026-07-08-hitl-edited-input-design.md. TDD, mypy+ruff per task. Baseline 931.

- T1 mesh/errors.py: EditedInputError(MeshError) + export.
- T2 langgraph_engine.resume: on approve+edited_input → validate (editable fields task/final; model_validate) fail-closed → EditedInputError; else update_state(edited_input) before invoke(None). Tests (engine, [mesh] extra).
- T3 service.resume_agent: except EditedInputError → InvalidInputError. rest_app resume route: except InvalidInputError → 400. Tests.
- T4 Full gate + review.

## Lab R19
Downstream: run mesh to interrupt, resume approve+edited_input → worker acts on edited value; bad edit → 400; empty edit → unchanged.
