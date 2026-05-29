"""Scaffold file contents for `lottie init`.

Static string constants; the two that need the project name use
`str.format(name=...)`. Kept separate from command logic so the future
`lottie create` generator can reuse them.
"""

from __future__ import annotations

KNOWLEDGE_LAYERS: list[str] = ["global", "platform", "project", "memory", "draft"]

LOTTIE_YAML = """\
project: {name}
providers:
  default: anthropic/claude-sonnet-4-6
  fallback: openai/gpt-4o
policies:
  - base
registry:
  agents: agents/
  skills: skills/
"""

LOTTIE_MD = """\
# {name}

> A Lottie project. This file is read automatically by all AI tools.

## Agents
_None yet — scaffold one with `lottie create agent <name>`._

## Skills
_None yet — scaffold one with `lottie create skill <name>`._
"""

GITIGNORE = """\
# Lottie runtime
.lottie/

# Private AI context
.private-journey/

# Personal Claude Code settings
.claude/settings.local.json

# Python
__pycache__/
.venv/
"""

POLICY_BASE = """\
# Base governance policy. Rules: allow / deny / escalate.
name: base
allow: []
deny: []
escalate: []
"""

AGENTS_INIT = '"""Auto-discovers and registers all agents in this project."""\n'

SKILLS_INIT = '"""Auto-discovers and registers all skills in this project."""\n'
