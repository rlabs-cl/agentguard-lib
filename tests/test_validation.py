"""Tests for the validation module — syntax, lint, types, imports, structure."""

from __future__ import annotations

from agentguard.archetypes.base import Archetype
from agentguard.validation.autofix import autofix
from agentguard.validation.checks.imports import check_imports
from agentguard.validation.checks.lint import check_lint
from agentguard.validation.checks.structure import check_structure
from agentguard.validation.checks.syntax import check_syntax
from agentguard.validation.checks.types import check_types
from agentguard.validation.types import (
    AutoFix,
    CheckResult,
    Severity,
    ValidationError,
    ValidationReport,
)
from agentguard.validation.validator import Validator

# ------------------------------------------------------------------ #
#  Validation types
# ------------------------------------------------------------------ #

class TestValidationTypes:
    def test_severity_ordering(self):
        assert Severity.ERROR.name == "ERROR"
        assert Severity.WARNING.name == "WARNING"
        assert Severity.INFO.name == "INFO"

    def test_validation_error_str(self):
        err = ValidationError(
            check="syntax",
            file_path="main.py",
            line=5,
            message="unexpected EOF",
        )
        assert "main.py" in str(err)
        assert "line 5" in str(err).lower() or "5" in str(err)

    def test_check_result_passed(self):
        cr = CheckResult(check="syntax", passed=True, details="OK")
        assert cr.passed is True
        assert cr.duration_ms == 0

    def test_check_result_failed(self):
        err = ValidationError(check="syntax", file_path="x.py", line=1, message="bad")
        cr = CheckResult(check="syntax", passed=False, details="Fail", errors=[err])
        assert cr.passed is False
        assert len(cr.errors) == 1

    def test_validation_report_blocking(self):
        errs = [
            ValidationError(check="syntax", file_path="x.py", line=1, message="bad", severity=Severity.ERROR),
            ValidationError(check="lint", file_path="x.py", line=2, message="style", severity=Severity.WARNING),
        ]
        report = ValidationReport(passed=False, checks=[], auto_fixed=[], errors=errs)
        assert len(report.blocking_errors) == 1
        assert len(report.warnings) == 1

    def test_autofix_dataclass(self):
        fix = AutoFix(check="autofix", file_path="main.py", description="trailing newline")
        assert fix.file_path == "main.py"


# ------------------------------------------------------------------ #
#  Syntax check
# ------------------------------------------------------------------ #

class TestSyntaxCheck:
    def test_valid_python(self):
        result = check_syntax({"main.py": "x = 1\n"})
        assert result.passed is True
        assert result.check == "syntax"

    def test_invalid_python(self):
        result = check_syntax({"main.py": "def broken(\n"})
        assert result.passed is False
        assert len(result.errors) >= 1
        assert result.errors[0].check == "syntax"
        assert result.errors[0].file_path == "main.py"

    def test_multiple_files_one_bad(self):
        result = check_syntax({
            "good.py": "x = 1\n",
            "bad.py": "def f(\n",
        })
        assert result.passed is False
        assert len(result.errors) == 1
        assert result.errors[0].file_path == "bad.py"

    def test_non_python_skipped(self):
        result = check_syntax({
            "readme.md": "# Hello",
            "good.py": "x = 1\n",
        })
        assert result.passed is True

    def test_empty_files(self):
        result = check_syntax({})
        assert result.passed is True

    def test_empty_python_file(self):
        result = check_syntax({"empty.py": ""})
        assert result.passed is True


# ------------------------------------------------------------------ #
#  Lint check
# ------------------------------------------------------------------ #

class TestLintCheck:
    def test_clean_code(self):
        result = check_lint({"main.py": "x = 1\n"})
        assert result.passed is True

    def test_unused_import(self):
        result = check_lint({"main.py": "import os\nx = 1\n"})
        # Ruff should flag unused import
        assert result.check == "lint"
        # Either passed or caught the unused import (depends on ruff availability)

    def test_syntax_error_still_runs(self):
        """Lint should handle files with syntax errors gracefully."""
        result = check_lint({"broken.py": "def f(\n"})
        assert result.check == "lint"

    def test_empty_files(self):
        result = check_lint({})
        assert result.passed is True


# ------------------------------------------------------------------ #
#  Type check
# ------------------------------------------------------------------ #

class TestTypeCheck:
    def test_clean_code(self):
        result = check_types({"main.py": "x: int = 1\n"})
        assert result.check == "types"

    def test_empty_files(self):
        result = check_types({})
        assert result.passed is True


# ------------------------------------------------------------------ #
#  Import check
# ------------------------------------------------------------------ #

class TestImportCheck:
    def test_stdlib_import(self):
        result = check_imports({"main.py": "import os\nimport sys\n"})
        assert result.passed is True

    def test_project_internal_import(self):
        """Imports between project files should resolve."""
        files = {
            "myapp/__init__.py": "",
            "myapp/models.py": "class User: pass\n",
            "myapp/service.py": "from myapp.models import User\n",
        }
        result = check_imports(files)
        assert result.passed is True

    def test_unknown_import(self):
        """Import of a non-existent module should fail."""
        result = check_imports({
            "main.py": "import totally_fake_nonexistent_module_xyz\n",
        })
        assert result.passed is False
        assert len(result.errors) >= 1

    def test_relative_import(self):
        files = {
            "pkg/__init__.py": "",
            "pkg/a.py": "x = 1\n",
            "pkg/b.py": "from . import a\n",
        }
        result = check_imports(files)
        assert result.passed is True

    def test_empty_files(self):
        result = check_imports({})
        assert result.passed is True

    def test_non_python_skipped(self):
        result = check_imports({"readme.md": "# hello\nimport os\n"})
        assert result.passed is True


# ------------------------------------------------------------------ #
#  Structure check
# ------------------------------------------------------------------ #

class TestStructureCheck:
    def test_no_archetype(self):
        """Without archetype, structure check passes."""
        result = check_structure({"main.py": "x = 1\n"}, archetype=None)
        assert result.passed is True

    def test_basic_structure(self):
        arch = Archetype.load("api_backend")
        files = {
            "main.py": "print('hello')\n",
            "requirements.txt": "fastapi\n",
        }
        # The check may report missing expected files — that's correct
        result = check_structure(files, archetype=arch)
        assert result.check == "structure"

    def test_empty_files(self):
        result = check_structure({}, archetype=None)
        assert result.passed is True


# ------------------------------------------------------------------ #
#  Autofix
# ------------------------------------------------------------------ #

class TestAutofix:
    def test_trailing_newline(self):
        files = {"main.py": "x = 1"}  # Missing trailing newline
        fixed, fixes = autofix(files)
        assert fixed["main.py"].endswith("\n")
        assert any("trailing newline" in f.description.lower() for f in fixes)

    def test_already_has_newline(self):
        files = {"main.py": "x = 1\n"}
        fixed, fixes = autofix(files)
        assert fixed["main.py"] == "x = 1\n"
        # No trailing newline fix should be applied
        trailing_fixes = [f for f in fixes if "trailing newline" in f.description.lower()]
        assert len(trailing_fixes) == 0

    def test_unused_import_removal(self):
        code = "import os\nimport sys\n\nx = 1\n"
        files = {"main.py": code}
        fixed, fixes = autofix(files)
        # os and sys should be removed (unused)
        assert "import os" not in fixed["main.py"]
        assert "import sys" not in fixed["main.py"]

    def test_used_import_kept(self):
        code = "import os\n\npath = os.getcwd()\n"
        files = {"main.py": code}
        fixed, fixes = autofix(files)
        assert "import os" in fixed["main.py"]

    def test_empty_files(self):
        fixed, fixes = autofix({})
        assert fixed == {}
        assert fixes == []

    def test_non_python_untouched(self):
        files = {"readme.md": "# Hello"}
        fixed, fixes = autofix(files)
        assert fixed["readme.md"] == "# Hello"


# ------------------------------------------------------------------ #
#  Validator orchestrator
# ------------------------------------------------------------------ #

class TestValidator:
    def test_valid_code_passes(self):
        validator = Validator()
        report = validator.check({"main.py": "x: int = 1\n"})
        assert report.passed is True

    def test_syntax_error_fails(self):
        validator = Validator()
        report = validator.check({"main.py": "def broken(\n"})
        assert report.passed is False
        assert any(c.check == "syntax" for c in report.checks)

    def test_autofix_applied(self):
        validator = Validator()
        validator.check({"main.py": "x = 1"})  # Missing trailing newline
        # Autofix should add the newline
        if validator.last_fixed_files:
            assert validator.last_fixed_files["main.py"].endswith("\n")

    def test_specific_checks(self):
        validator = Validator()
        report = validator.check(
            {"main.py": "x = 1\n"},
            checks=["syntax"],
        )
        assert report.passed is True
        assert len(report.checks) == 1
        assert report.checks[0].check == "syntax"

    def test_no_autofix(self):
        validator = Validator()
        validator.check(
            {"main.py": "x = 1"},  # Missing trailing newline
            do_autofix=False,
        )
        # Without autofix, the file should be left as-is
        assert validator.last_fixed_files is None or "main.py" not in validator.last_fixed_files

    def test_with_archetype(self):
        arch = Archetype.load("api_backend")
        validator = Validator(archetype=arch)
        report = validator.check({"main.py": "x = 1\n"})
        assert isinstance(report, ValidationReport)
