"""Structural validation module — mechanical checks on generated code."""

from agentguard.validation.types import (
    AutoFix,
    CheckResult,
    ValidationError,
    ValidationReport,
)
from agentguard.validation.validator import Validator

__all__ = [
    "AutoFix",
    "CheckResult",
    "ValidationError",
    "ValidationReport",
    "Validator",
]
