"""The kernel must not import the subsystems that mount onto it.

`src/lottie/core/__init__.py` eagerly imports `base_agent`. Once S2 makes `BaseAgent`
import the kernel, a kernel -> core import becomes a circular import at package-init
time. This test is the structural guarantee behind the dependency reversal in V3 spec
section 5.1 — it fails loudly the moment someone reintroduces the coupling.

This is not hypothetical: V2 S5b hit exactly this cycle
(core -> session -> security -> core.__init__ -> base_agent) and had to break it with a
lazy import.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = ("lottie.core", "lottie.governance", "lottie.memory", "lottie.security", "lottie.llm")

RUNTIME_DIR = Path(__file__).resolve().parent.parent


def _kernel_modules() -> list[Path]:
    """Every .py file in the kernel package, excluding its own tests."""
    return [p for p in RUNTIME_DIR.rglob("*.py") if "tests" not in p.parts]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


class TestKernelImportHygiene:
    def test_finds_the_kernel_modules(self) -> None:
        # Guard against the glob silently matching nothing and the suite passing vacuously.
        names = {p.name for p in _kernel_modules()}
        assert "context.py" in names

    def test_no_kernel_module_imports_a_subsystem(self) -> None:
        offenders: list[str] = []
        for path in _kernel_modules():
            for imported in _imported_modules(path):
                if any(imported == f or imported.startswith(f + ".") for f in FORBIDDEN):
                    offenders.append(f"{path.name} imports {imported}")
        assert offenders == []
