"""Lint check — run Ruff on generated Python code."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from agentguard.validation.types import CheckResult, Severity, ValidationError

logger = logging.getLogger(__name__)


def check_lint(files: dict[str, str]) -> CheckResult:
    """Run Ruff linter on all Python files.

    Writes files to a temp directory, runs `ruff check --output-format=json`,
    and parses the results.

    Args:
        files: Dict of {file_path: file_content}.

    Returns:
        CheckResult with lint errors/warnings.
    """
    start = time.perf_counter()
    py_files = {k: v for k, v in files.items() if k.endswith(".py")}

    if not py_files:
        return CheckResult(
            check="lint",
            passed=True,
            details="No Python files to lint",
            duration_ms=0,
        )

    errors: list[ValidationError] = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write files to temp dir
            for file_path, content in py_files.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")

            # Run ruff
            result = subprocess.run(
                [
                    "ruff", "check",
                    "--output-format=json",
                    "--select=E,F,W",  # Errors, pyflakes, warnings
                    "--no-fix",
                    str(tmpdir_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse JSON output (ruff outputs JSON array even on exit code 1)
            output = result.stdout.strip()
            if output:
                findings = json.loads(output)
                for finding in findings:
                    # Map file path back to original
                    abs_path = finding.get("filename", "")
                    rel_path = abs_path.replace(str(tmpdir_path) + "\\", "").replace(
                        str(tmpdir_path) + "/", ""
                    )

                    severity = Severity.WARNING
                    code = finding.get("code", "")
                    if code.startswith("E") or code.startswith("F"):
                        severity = Severity.ERROR

                    errors.append(
                        ValidationError(
                            check="lint",
                            file_path=rel_path,
                            line=finding.get("location", {}).get("row"),
                            column=finding.get("location", {}).get("column"),
                            message=finding.get("message", ""),
                            severity=severity,
                            code=code,
                        )
                    )

    except FileNotFoundError:
        logger.warning("Ruff not found; skipping lint check")
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            check="lint",
            passed=True,
            details="Ruff not installed — lint check skipped",
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Ruff timed out after 30s")
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            check="lint",
            passed=False,
            details="Ruff timed out",
            errors=[
                ValidationError(
                    check="lint",
                    file_path="<all>",
                    message="Ruff timed out after 30 seconds",
                    severity=Severity.ERROR,
                )
            ],
            duration_ms=duration_ms,
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse Ruff output: %s", e)

    duration_ms = int((time.perf_counter() - start) * 1000)
    blocking = [e for e in errors if e.severity == Severity.ERROR]
    return CheckResult(
        check="lint",
        passed=len(blocking) == 0,
        details=f"Ruff found {len(errors)} issues ({len(blocking)} errors)",
        errors=errors,
        duration_ms=duration_ms,
    )
