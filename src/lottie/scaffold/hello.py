"""Bundled, runnable starter agent written by `lottie init`.

HELLO_FILES maps a path (relative to agents/hello/) to its literal content. The
agent calls the configured LLM provider to greet the user, so `lottie run hello`
works end-to-end immediately and its MockLLM tests pass out of the box.
"""

from __future__ import annotations

_AGENT_PY = '''\
"""HelloAgent — a runnable starter agent created by `lottie init`."""
from __future__ import annotations
from lottie.core import BaseAgent
from lottie.llm import Message
from .prompts import SYSTEM_PROMPT
from .schema import HelloInput, HelloOutput


class HelloAgent(BaseAgent[HelloInput, HelloOutput]):
    """Greets the user via the configured LLM provider."""

    def _execute(self, data: HelloInput) -> HelloOutput:
        response = self.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=f"Greet {data.name}."),
            ]
        )
        return HelloOutput(greeting=response.content)
'''

_SCHEMA_PY = '''\
"""Typed input/output models for HelloAgent."""
from __future__ import annotations
from pydantic import BaseModel


class HelloInput(BaseModel):
    """Input for HelloAgent."""
    name: str = "world"


class HelloOutput(BaseModel):
    """Output from HelloAgent."""
    greeting: str
'''

_PROMPTS_PY = '''\
"""Prompt templates for HelloAgent."""
from __future__ import annotations

SYSTEM_PROMPT = """\\
You are HelloAgent, a friendly Lottie starter agent.
Greet the user warmly in one short sentence.
"""
'''

_CONFIG_YAML = """\
provider: anthropic/claude-sonnet-4-6
model_params:
  temperature: 0.3
  max_tokens: 256
capabilities: []
policies:
  - base
memory:
  enabled: false
  namespace: hello
"""

_AGENT_MD = """\
# HelloAgent

## Role
Greets the user — the runnable starter agent created by `lottie init`.

## Input
| Field | Type | Description |
|---|---|---|
| name | str | Who to greet (defaults to "world") |

## Output
| Field | Type | Description |
|---|---|---|
| greeting | str | The generated greeting |

## Provider
Default: anthropic/claude-sonnet-4-6

## Tools (Skills used)
_None yet._

## Policies
- base

## Examples
### Example 1
Input: `{"name": "Ada"}`
Output: `{"greeting": "Hello, Ada!"}`
"""

_TEST_PY = '''\
"""Integration tests for HelloAgent (MockLLMProvider — no real LLM)."""
from __future__ import annotations
from lottie.llm import MockLLMProvider
from agents.hello.agent import HelloAgent
from agents.hello.schema import HelloInput, HelloOutput


def test_hello_greets_via_llm() -> None:
    agent = HelloAgent(llm=MockLLMProvider(["Hello, Ada!"]))
    result = agent.run(HelloInput(name="Ada"))
    assert isinstance(result, HelloOutput)
    assert result.greeting == "Hello, Ada!"


def test_hello_makes_one_llm_call() -> None:
    mock = MockLLMProvider(["hi"])
    HelloAgent(llm=mock).run(HelloInput(name="Ada"))
    assert len(mock.calls) == 1


def test_hello_defaults_to_world() -> None:
    agent = HelloAgent(llm=MockLLMProvider(["Hello, world!"]))
    result = agent.run(HelloInput())
    assert result.greeting == "Hello, world!"
'''

HELLO_FILES: dict[str, str] = {
    "__init__.py": "",
    "AGENT.md": _AGENT_MD,
    "agent.py": _AGENT_PY,
    "schema.py": _SCHEMA_PY,
    "config.yaml": _CONFIG_YAML,
    "prompts.py": _PROMPTS_PY,
    "tests/__init__.py": "",
    "tests/test_hello.py": _TEST_PY,
}
