"""Structure check — verify generated files match archetype expectations."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agentguard.validation.types import CheckResult, Severity, ValidationError

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype


def check_structure(
    files: dict[str, str],
    archetype: Archetype | None = None,
) -> CheckResult:
    """Check that generated file structure matches archetype expectations.

    Verifies:
    - Expected files exist
    - Expected directories are represented
    - No unexpected top-level files

    Args:
        files: Dict of {file_path: file_content}.
        archetype: Archetype defining expected structure (optional).

    Returns:
        CheckResult with structure violations.
    """
    start = time.perf_counter()

    if archetype is None:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CheckResult(
            check="structure",
            passed=True,
            details="No archetype provided — structure check skipped",
            duration_ms=duration_ms,
        )

    errors: list[ValidationError] = []
    generated_paths = set(files.keys())
    generated_dirs = _extract_dirs(generated_paths)

    structure = archetype.structure

    # Check expected files (using glob-like matching with {project_name} placeholder)
    expected_files = structure.get("expected_files", [])
    for pattern in expected_files:
        if not _pattern_matches_any(pattern, generated_paths):
            errors.append(
                ValidationError(
                    check="structure",
                    file_path=pattern,
                    message=f"Expected file '{pattern}' not found in generated output",
                    severity=Severity.WARNING,  # Warning, not error — LLM might use slightly different structure
                )
            )

    # Check expected directories
    expected_dirs = structure.get("expected_dirs", [])
    for pattern in expected_dirs:
        if not _pattern_matches_any_dir(pattern, generated_dirs):
            errors.append(
                ValidationError(
                    check="structure",
                    file_path=pattern,
                    message=f"Expected directory '{pattern}' not found in generated output",
                    severity=Severity.WARNING,
                )
            )

    # Check that we have at least some Python files
    py_files = [f for f in generated_paths if f.endswith(".py")]
    if not py_files:
        errors.append(
            ValidationError(
                check="structure",
                file_path="<project>",
                message="No Python files found in generated output",
                severity=Severity.ERROR,
            )
        )

    # Check for a main entry point
    has_main = any(
        f.endswith("main.py") or f.endswith("app.py") or f.endswith("__main__.py")
        for f in generated_paths
    )
    if not has_main and archetype.tech_stack.language == "python":
        errors.append(
            ValidationError(
                check="structure",
                file_path="<project>",
                message="No main entry point found (main.py, app.py, or __main__.py)",
                severity=Severity.WARNING,
            )
        )

    duration_ms = int((time.perf_counter() - start) * 1000)
    blocking = [e for e in errors if e.severity == Severity.ERROR]
    return CheckResult(
        check="structure",
        passed=len(blocking) == 0,
        details=f"Checked {len(generated_paths)} files against archetype '{archetype.id}'",
        errors=errors,
        duration_ms=duration_ms,
    )


def _extract_dirs(file_paths: set[str]) -> set[str]:
    """Extract all directory paths from file paths."""
    dirs: set[str] = set()
    for path in file_paths:
        parts = path.replace("\\", "/").split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]) + "/")
    return dirs


def _pattern_matches_any(pattern: str, paths: set[str]) -> bool:
    """Check if a pattern (possibly with {project_name} placeholder) matches any path."""
    # Normalize
    pattern = pattern.replace("\\", "/")

    # Direct match
    if pattern in paths:
        return True

    # If pattern has {project_name}, try matching with any segment
    if "{project_name}" in pattern:
        # Replace placeholder with wildcard matching
        prefix, suffix = pattern.split("{project_name}", 1)
        for path in paths:
            normalized = path.replace("\\", "/")
            if normalized.startswith(prefix) and normalized.endswith(suffix):
                return True

    # Also check if pattern is a basename match
    pattern_name = pattern.split("/")[-1]
    for path in paths:
        if path.replace("\\", "/").endswith("/" + pattern_name) or path == pattern_name:
            return True

    return False


def _pattern_matches_any_dir(pattern: str, dirs: set[str]) -> bool:
    """Check if a directory pattern matches any directory."""
    pattern = pattern.replace("\\", "/")
    if not pattern.endswith("/"):
        pattern += "/"

    if pattern in dirs:
        return True

    # With {project_name} placeholder
    if "{project_name}" in pattern:
        prefix, suffix = pattern.split("{project_name}", 1)
        for d in dirs:
            if d.startswith(prefix) and d.endswith(suffix):
                return True

    # Check basename match
    parts = pattern.rstrip("/").split("/")
    dir_name = parts[-1]
    return any(d.rstrip("/").endswith(dir_name) for d in dirs)
