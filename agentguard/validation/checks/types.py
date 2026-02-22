"""Type check — run mypy on generated Python code."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from agentguard.validation.types import CheckResult, Severity, ValidationError

logger = logging.getLogger(__name__)


def check_types(files: dict[str, str]) -> CheckResult:
    """Run mypy type checker on all Python files.

    Writes files to a temp directory, runs `mypy --output=json` and
    parses the results.

    Args:
        files: Dict of {file_path: file_content}.

    Returns:
        CheckResult with type errors.
    """
    start = time.perf_counter()
    py_files = {k: v for k, v in files.items() if k.endswith(".py")}

    if not py_files:
        return CheckResult(
            check="types",
            passed=True,
            details="No Python files to type-check",
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

            # Write minimal mypy config
            mypy_ini = tmpdir_path / "mypy.ini"
            mypy_ini.write_text(
                "[mypy]\n"
                "ignore_missing_imports = True\n"
                "no_error_summary = True\n"
                "show_error_codes = True\n",
                encoding="utf-8",
            )

            # Collect file paths to check
            files_to_check = [str(tmpdir_path / f) for f in py_files]

            # Run mypy
            result = subprocess.run(
                [
                    "mypy",
                    "--config-file", str(mypy_ini),
                    "--no-color-output",
                    *files_to_check,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse mypy output (line format: file:line: severity: message [code])
            for line in result.stdout.strip().splitlines():
                parsed = _parse_mypy_line(line, str(tmpdir_path))
                if parsed:
                    errors.append(parsed)

    except FileNotFoundError:
        logger.warning("mypy not found; skipping type check")
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            check="types",
            passed=True,
            details="mypy not installed — type check skipped",
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired:
        logger.warning("mypy timed out after 60s")
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            check="types",
            passed=False,
            details="mypy timed out",
            errors=[
                ValidationError(
                    check="types",
                    file_path="<all>",
                    message="mypy timed out after 60 seconds",
                    severity=Severity.ERROR,
                )
            ],
            duration_ms=duration_ms,
        )

    duration_ms = int((time.perf_counter() - start) * 1000)
    blocking = [e for e in errors if e.severity == Severity.ERROR]
    return CheckResult(
        check="types",
        passed=len(blocking) == 0,
        details=f"mypy found {len(errors)} issues ({len(blocking)} errors)",
        errors=errors,
        duration_ms=duration_ms,
    )


def _parse_mypy_line(line: str, tmpdir: str) -> ValidationError | None:
    """Parse a single mypy output line into a ValidationError."""
    # Format: path:line: severity: message  [code]
    if ": error:" not in line and ": warning:" not in line and ": note:" not in line:
        return None

    try:
        # Split on ': ' to get path:line, severity, message
        parts = line.split(": ", 2)
        if len(parts) < 3:
            return None

        file_loc = parts[0]
        severity_str = parts[1].strip()
        message = parts[2].strip()

        # Parse file:line
        loc_parts = file_loc.rsplit(":", 1)
        file_path = loc_parts[0].replace(tmpdir + "\\", "").replace(tmpdir + "/", "")
        line_num = int(loc_parts[1]) if len(loc_parts) > 1 else None

        # Parse code from [code] at end of message
        code = None
        if message.endswith("]"):
            bracket = message.rfind("[")
            if bracket != -1:
                code = message[bracket + 1 : -1]
                message = message[:bracket].strip()

        severity = Severity.ERROR if severity_str == "error" else Severity.WARNING

        return ValidationError(
            check="types",
            file_path=file_path,
            line=line_num,
            message=message,
            severity=severity,
            code=code,
        )
    except (ValueError, IndexError):
        return None
