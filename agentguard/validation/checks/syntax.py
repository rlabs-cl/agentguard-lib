"""Syntax check — verify Python code parses correctly via ast.parse()."""

from __future__ import annotations

import ast
import time

from agentguard.validation.types import CheckResult, Severity, ValidationError


def check_syntax(files: dict[str, str]) -> CheckResult:
    """Check that all Python files parse without syntax errors.

    Args:
        files: Dict of {file_path: file_content}.

    Returns:
        CheckResult with syntax errors if any.
    """
    start = time.perf_counter()
    errors: list[ValidationError] = []

    for file_path, content in files.items():
        if not file_path.endswith(".py"):
            continue

        try:
            ast.parse(content, filename=file_path)
        except SyntaxError as e:
            errors.append(
                ValidationError(
                    check="syntax",
                    file_path=file_path,
                    line=e.lineno,
                    column=e.offset,
                    message=e.msg or "SyntaxError",
                    severity=Severity.ERROR,
                )
            )

    duration_ms = int((time.perf_counter() - start) * 1000)
    return CheckResult(
        check="syntax",
        passed=len(errors) == 0,
        details=f"Parsed {sum(1 for f in files if f.endswith('.py'))} Python files",
        errors=errors,
        duration_ms=duration_ms,
    )
