"""Capability policy engine: load allow/deny/escalate rules and evaluate them
against an agent's declared capabilities at the run chokepoint.

Imports only stdlib + pydantic + yaml so governance stays free of core/project deps.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel


class Policy(BaseModel):
    """One governance policy file: allow/deny/escalate over capability names."""

    name: str = ""
    allow: list[str] = []
    deny: list[str] = []
    escalate: list[str] = []


class PolicyViolation(Exception):
    """Base for a blocked run."""


class PolicyDenied(PolicyViolation):
    """A declared capability is denied (or not in a non-empty allow-list)."""


class PolicyEscalation(PolicyViolation):
    """A declared capability requires human approval (escalate). Blocked for now."""


class PolicyConfigError(Exception):
    """A declared policy file is missing or malformed."""


def load_policy(root: Path, name: str) -> Policy:
    """Read root/policies/<name>.yaml. Empty file -> empty Policy; missing -> error."""
    path = Path(root) / "policies" / f"{name}.yaml"
    if not path.is_file():
        raise PolicyConfigError(f"declared policy {name!r} not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return Policy(name=name)
    if not isinstance(raw, dict):
        raise PolicyConfigError(f"policy {name!r} must be a mapping, got {type(raw).__name__}")
    return Policy(
        name=str(raw.get("name", name)),
        allow=list(raw.get("allow") or []),
        deny=list(raw.get("deny") or []),
        escalate=list(raw.get("escalate") or []),
    )


class PolicyGate:
    """Evaluates an agent's declared capabilities against merged policy rules."""

    def __init__(
        self,
        capabilities: Iterable[str],
        *,
        allow: set[str],
        deny: set[str],
        escalate: set[str],
    ) -> None:
        self._caps = sorted(capabilities)
        self._allow = allow
        self._deny = deny
        self._escalate = escalate

    def check(self) -> None:
        """Raise on a violation; else return. Precedence: deny > escalate > allow-whitelist."""
        for cap in self._caps:
            if cap in self._deny:
                raise PolicyDenied(f"capability {cap!r} denied by policy")
        for cap in self._caps:
            if cap in self._escalate:
                raise PolicyEscalation(f"capability {cap!r} requires approval (escalate)")
        if self._allow:
            for cap in self._caps:
                if cap not in self._allow:
                    raise PolicyDenied(f"capability {cap!r} not in policy allow-list")


class NullPolicyGate(PolicyGate):
    """No-op gate -- the BaseAgent default and the 'no policies declared' result."""

    def __init__(self) -> None:
        super().__init__([], allow=set(), deny=set(), escalate=set())

    def check(self) -> None:
        return


def build_policy_gate(
    root: Path, *, policies: list[str], capabilities: list[str]
) -> PolicyGate:
    """Merge every declared policy file (union) and bind capabilities.

    Returns NullPolicyGate when no policies are declared.
    """
    if not policies:
        return NullPolicyGate()
    allow: set[str] = set()
    deny: set[str] = set()
    escalate: set[str] = set()
    for name in policies:
        p = load_policy(root, name)
        allow |= set(p.allow)
        deny |= set(p.deny)
        escalate |= set(p.escalate)
    return PolicyGate(capabilities, allow=allow, deny=deny, escalate=escalate)
