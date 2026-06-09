"""SchemaValidatorSkill — type-check and lint generated files (rule-13 steps 4-5).

Runs `mypy --strict` then `ruff check` over the given paths as subprocesses and
reports a single pass/fail plus the combined diagnostics. Used by the code-write
gate, which passes absolute paths (cwd-independent); mypy still resolves a generated
unit's package and relative imports by walking up its __init__.py parents.
"""

from __future__ import annotations

import subprocess
import sys

from lottie.core import BaseSkill
from lottie.security.schema import ValidateInput, ValidateOutput


class SchemaValidatorSkill(BaseSkill[ValidateInput, ValidateOutput]):
    """Validate files with mypy --strict and ruff check."""

    def _execute(self, data: ValidateInput) -> ValidateOutput:
        if not data.paths:
            return ValidateOutput(passed=True, diagnostics="")
        mypy = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "mypy", "--strict", *data.paths],
            capture_output=True,
            text=True,
        )
        ruff = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "ruff", "check", *data.paths],
            capture_output=True,
            text=True,
        )
        passed = mypy.returncode == 0 and ruff.returncode == 0
        diagnostics = "".join(
            [mypy.stdout, mypy.stderr, ruff.stdout, ruff.stderr]
        ).strip()
        return ValidateOutput(passed=passed, diagnostics=diagnostics)
