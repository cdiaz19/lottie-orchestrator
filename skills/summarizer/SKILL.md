# SummarizerSkill

## What it does

LLM-backed summarisation: accepts arbitrary text and returns a concise prose
summary plus a capped list of bullet-point highlights.  Deterministic given a
fixed provider and model; uses the injected `LLMProvider` — no vendor SDK is
imported.

## Input

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | `str` | yes | — | The text to summarise |
| `max_points` | `int` | no | `5` | Maximum bullet points to return (must be ≥ 1) |

## Output

| Field | Type | Description |
|---|---|---|
| `summary` | `str` | Concise prose paragraph summarising the text |
| `points` | `list[str]` | Up to `max_points` key highlights as plain strings (no bullet marker) |

## Parsing contract

The skill asks the model to return a prose paragraph and `- ` bullets, but
parsing is **order-independent**: every non-empty line is classified on its own.

- Lines matching `[-*•]` or `N.`/`N)` (1–2 digit markers only, to avoid
  false positives on prose like "100. Some sentence.") are collected as
  `points`; the bullet marker is stripped.
- All other non-empty lines are prose lines.
- `summary` = all prose lines joined with a single space (leading AND trailing
  prose are both included, nothing is dropped).
- `points` is always capped to `max_points`.
- **Fallback (all-bullets):** if the response contains no prose lines,
  `summary` is set to the full stripped response text so it is never empty
  when the model returned content.
- **No bullets at all:** `points=[]`, `summary` = full stripped response.

## LLM contract

- **Provider**: injected `LLMProvider` (constructor arg `llm`).
- **No vendor SDK**: `lottie.llm` abstraction only (CLAUDE.md rule 1).
- **Token tracking**: `InstrumentedRunnable` records `last_metrics` after each
  run; input/output token counts from the LLM response are NOT accumulated into
  the run context (skills do not have a `RunContext` accumulator — only agents
  do via `BaseAgent.complete`).  Token data is available via
  `LLMResponse.usage` if callers need it.

## Side effects

Calls the injected `LLMProvider.complete` once per `run` invocation.

## Examples

### Basic summarisation
```python
from lottie.llm import MockLLMProvider
from skills.summarizer.schema import SummarizerInput
from skills.summarizer.skill import SummarizerSkill

mock = MockLLMProvider(["Lottie is a framework.\n- Typed schemas\n- Provider-agnostic"])
skill = SummarizerSkill(mock)
out = skill.run(SummarizerInput(text="...", max_points=3))
# out.summary == "Lottie is a framework."
# out.points  == ["Typed schemas", "Provider-agnostic"]
```

### Plain-prose response (no bullets)
```python
mock = MockLLMProvider(["Lottie orchestrates agents with typed schemas."])
skill = SummarizerSkill(mock)
out = skill.run(SummarizerInput(text="..."))
# out.summary == "Lottie orchestrates agents with typed schemas."
# out.points  == []
```
