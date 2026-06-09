"""Security layer — input/output gates and the code-write pipeline."""

from lottie.security.code_scanner import CodeSecurityScanSkill
from lottie.security.schema import (
    GateResult,
    ScanInput,
    ScanOutput,
    SecurityFinding,
    ValidateInput,
    ValidateOutput,
)
from lottie.security.secret_detector import SecretDetectionSkill
from lottie.security.validator import SchemaValidatorSkill
from lottie.security.write_gate import guard_and_write

__all__ = [
    "CodeSecurityScanSkill",
    "GateResult",
    "ScanInput",
    "ScanOutput",
    "SchemaValidatorSkill",
    "SecretDetectionSkill",
    "SecurityFinding",
    "ValidateInput",
    "ValidateOutput",
    "guard_and_write",
]
