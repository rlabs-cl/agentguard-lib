"""Validation types — shared dataclasses for all validation checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    """Severity level for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """A single validation error."""

    check: str
    file_path: str
    line: int | None = None
    column: int | None = None
    message: str = ""
    severity: Severity = Severity.ERROR
    code: str | None = None  # e.g. "E501", "F401"

    def __str__(self) -> str:
        loc = self.file_path
        if self.line:
            loc += f":{self.line}"
            if self.column:
                loc += f":{self.column}"
        return f"[{self.check}] {loc}: {self.message}"


@dataclass
class AutoFix:
    """A trivial fix that was applied automatically."""

    check: str
    file_path: str
    description: str
    before: str = ""
    after: str = ""

    def __str__(self) -> str:
        return f"[autofix:{self.check}] {self.file_path}: {self.description}"


@dataclass
class CheckResult:
    """Result of a single validation check."""

    check: str  # "syntax", "lint", "types", "imports", "structure"
    passed: bool
    details: str = ""
    errors: list[ValidationError] = field(default_factory=list)
    duration_ms: int = 0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{self.check}] {status} ({len(self.errors)} errors, {self.duration_ms}ms)"


@dataclass
class ValidationReport:
    """Aggregated result of all validation checks."""

    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    auto_fixed: list[AutoFix] = field(default_factory=list)
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def blocking_errors(self) -> list[ValidationError]:
        """Errors that must be fixed (severity == ERROR)."""
        return [e for e in self.errors if e.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationError]:
        """Non-blocking warnings."""
        return [e for e in self.errors if e.severity == Severity.WARNING]

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        parts = [
            f"Validation: {status}",
            f"{len(self.errors)} errors",
            f"{len(self.auto_fixed)} auto-fixed",
        ]
        return " | ".join(parts)
