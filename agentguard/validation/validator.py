"""Validator — orchestrates all structural checks on generated code."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentguard.validation.autofix import autofix
from agentguard.validation.checks.imports import check_imports
from agentguard.validation.checks.lint import check_lint
from agentguard.validation.checks.structure import check_structure
from agentguard.validation.checks.syntax import check_syntax
from agentguard.validation.checks.types import check_types
from agentguard.validation.types import (
    AutoFix,
    CheckResult,
    ValidationError,
    ValidationReport,
)

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype

logger = logging.getLogger(__name__)

# Map of check name → checker function
_CHECK_REGISTRY: dict[str, Any] = {
    "syntax": check_syntax,
    "lint": check_lint,
    "types": check_types,
    "imports": check_imports,
    # "structure" is special — needs archetype arg
}


class Validator:
    """Runs structural checks on generated code.

    Usage::

        validator = Validator(archetype=arch)
        report = validator.check(files)
        if not report.passed:
            for err in report.blocking_errors:
                print(err)

        # Or with autofix
        report = validator.check(files, autofix=True)
        fixed_files = ...  # get from validator.last_fixed_files
    """

    def __init__(self, archetype: Archetype | None = None) -> None:
        self._archetype = archetype
        self._last_fixed_files: dict[str, str] | None = None

    def check(
        self,
        code: str | dict[str, str],
        checks: list[str] | None = None,
        do_autofix: bool = True,
    ) -> ValidationReport:
        """Run validation checks on code.

        Args:
            code: A single file string or dict of {file_path: content}.
            checks: Subset of checks to run. Default: all from archetype or all available.
            do_autofix: Whether to auto-fix trivial issues first.

        Returns:
            ValidationReport with pass/fail and all issues found.
        """
        # Normalize input to dict
        files = {"<input>.py": code} if isinstance(code, str) else dict(code)

        self._last_fixed_files = None

        # Determine which checks to run
        if checks is None:
            if self._archetype and self._archetype.validation:
                checks = list(self._archetype.validation.checks)
            else:
                checks = ["syntax", "lint", "types", "imports", "structure"]

        # Auto-fix first (if enabled)
        auto_fixed: list[AutoFix] = []
        if do_autofix:
            files, auto_fixed = autofix(files)
            self._last_fixed_files = files
            if auto_fixed:
                logger.info("Auto-fixed %d issues", len(auto_fixed))

        # Run checks in order (fast → slow)
        check_order = ["syntax", "lint", "imports", "structure", "types"]
        ordered_checks = [c for c in check_order if c in checks]
        # Add any remaining checks not in our predefined order
        ordered_checks.extend(c for c in checks if c not in ordered_checks)

        all_results: list[CheckResult] = []
        all_errors: list[ValidationError] = []
        stop_early = False

        for check_name in ordered_checks:
            if stop_early:
                break

            result = self._run_check(check_name, files)
            all_results.append(result)
            all_errors.extend(result.errors)

            # Stop early if syntax fails — other checks will be meaningless
            if check_name == "syntax" and not result.passed:
                logger.info("Syntax check failed — skipping remaining checks")
                stop_early = True

        passed = all(r.passed for r in all_results)

        report = ValidationReport(
            passed=passed,
            checks=all_results,
            auto_fixed=auto_fixed,
            errors=all_errors,
        )

        logger.info("Validation %s: %s", "PASSED" if passed else "FAILED", report)
        return report

    def _run_check(self, check_name: str, files: dict[str, str]) -> CheckResult:
        """Run a single named check."""
        if check_name == "structure":
            return check_structure(files, archetype=self._archetype)

        checker = _CHECK_REGISTRY.get(check_name)
        if checker is None:
            logger.warning("Unknown check: %s", check_name)
            return CheckResult(
                check=check_name,
                passed=True,
                details=f"Unknown check '{check_name}' — skipped",
            )

        return checker(files)  # type: ignore[no-any-return]

    @property
    def last_fixed_files(self) -> dict[str, str] | None:
        """The files after auto-fix was applied (if any)."""
        return self._last_fixed_files
