"""Tests for the challenge module — SelfChallenger, GroundingChecker, types."""

from __future__ import annotations

import pytest

from agentguard.challenge.challenger import (
    SelfChallenger,
    _parse_criteria_results,
    _parse_grounding_section,
)
from agentguard.challenge.grounding import GroundingChecker, GroundingReport
from agentguard.challenge.types import ChallengeResult, CriterionResult

# ------------------------------------------------------------------ #
#  Challenge types
# ------------------------------------------------------------------ #

class TestChallengeTypes:
    def test_criterion_result_pass(self):
        cr = CriterionResult(criterion="No secrets", passed=True, explanation="OK")
        assert "PASS" in str(cr)

    def test_criterion_result_fail(self):
        cr = CriterionResult(criterion="No secrets", passed=False, explanation="Found hardcoded key")
        assert "FAIL" in str(cr)

    def test_challenge_result_passed(self):
        result = ChallengeResult(
            passed=True,
            criteria_results=[
                CriterionResult("C1", True, "OK"),
                CriterionResult("C2", True, "OK"),
            ],
        )
        assert result.passed is True
        assert len(result.failed_criteria) == 0
        assert "PASSED" in str(result)

    def test_challenge_result_failed(self):
        result = ChallengeResult(
            passed=False,
            criteria_results=[
                CriterionResult("C1", True, "OK"),
                CriterionResult("C2", False, "Bad"),
            ],
        )
        assert len(result.failed_criteria) == 1
        assert "FAILED" in str(result)

    def test_challenge_result_with_violations(self):
        result = ChallengeResult(
            passed=False,
            grounding_violations=["Used fake_module"],
        )
        assert "grounding" in str(result).lower()


# ------------------------------------------------------------------ #
#  Response parsing
# ------------------------------------------------------------------ #

class TestResponseParsing:
    def test_parse_criteria_all_pass(self):
        text = (
            "CRITERION 1: PASS: Looks good\n"
            "CRITERION 2: PASS: All clear\n"
        )
        results = _parse_criteria_results(text, ["Check A", "Check B"])
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_parse_criteria_mixed(self):
        text = (
            "CRITERION 1: PASS: Good\n"
            "CRITERION 2: FAIL: Missing error handling\n"
            "CRITERION 3: PASS: Present\n"
        )
        results = _parse_criteria_results(text, ["A", "B", "C"])
        assert results[0].passed is True
        assert results[1].passed is False
        assert results[2].passed is True

    def test_parse_criteria_unparseable(self):
        """Unparseable criteria default to FAIL."""
        text = "This response contains no verdict markers at all"
        results = _parse_criteria_results(text, ["Check A"])
        assert len(results) == 1
        assert results[0].passed is False
        assert "parse" in results[0].explanation.lower()

    def test_parse_grounding_violations(self):
        text = (
            "GROUNDING:\n"
            "- VIOLATION: used fake_api_module\n"
            "- VIOLATION: referenced imaginary.helper\n"
            "- ASSUMPTION: database is PostgreSQL\n"
        )
        violations, assumptions = _parse_grounding_section(text)
        assert len(violations) == 2
        assert "fake_api_module" in violations[0]
        assert len(assumptions) == 1
        assert "PostgreSQL" in assumptions[0]

    def test_parse_grounding_none(self):
        text = (
            "GROUNDING:\n"
            "- NONE\n"
            "- NONE\n"
        )
        violations, assumptions = _parse_grounding_section(text)
        assert len(violations) == 0
        assert len(assumptions) == 0


# ------------------------------------------------------------------ #
#  SelfChallenger — prompt rendering
# ------------------------------------------------------------------ #

class TestSelfChallenger:
    def test_render_challenge_prompt(self):
        """SelfChallenger should render a structured challenge prompt."""
        challenger = SelfChallenger()
        messages = challenger.render_challenge_prompt(
            output="def hello(): pass",
            criteria=["Endpoints match spec", "No hardcoded secrets"],
            task_description="Implement auth module",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Endpoints match spec" in messages[1]["content"]
        assert "No hardcoded secrets" in messages[1]["content"]

    def test_build_feedback(self):
        results = [
            CriterionResult("C1", True, "OK"),
            CriterionResult("C2", False, "Missing"),
        ]
        feedback = SelfChallenger.build_feedback(
            results,
            violations=["Used fake_mod"],
            assumptions=["DB is Postgres"],
        )
        assert "FAILED CRITERIA" in feedback
        assert "C2" in feedback
        assert "GROUNDING VIOLATIONS" in feedback
        assert "ASSUMPTIONS" in feedback


# ------------------------------------------------------------------ #
#  GroundingChecker
# ------------------------------------------------------------------ #

class TestGroundingChecker:
    def test_stdlib_imports_grounded(self):
        checker = GroundingChecker()
        report = checker.check_files({"main.py": "import os\nimport sys\n"})
        assert report.passed is True

    def test_project_internal_imports_grounded(self):
        checker = GroundingChecker()
        files = {
            "myapp/__init__.py": "",
            "myapp/service.py": "from myapp import models\n",
            "myapp/models.py": "class User: pass\n",
        }
        report = checker.check_files(files)
        assert report.passed is True

    def test_unknown_imports_flagged(self):
        checker = GroundingChecker()
        report = checker.check_files({
            "main.py": "import totally_fake_nonexistent_xyz_module\n",
        })
        assert report.passed is False
        assert len(report.violations) >= 1

    def test_known_modules_override(self):
        checker = GroundingChecker(known_modules={"custom_sdk"})
        report = checker.check_files({
            "main.py": "import custom_sdk\n",
        })
        assert report.passed is True

    def test_relative_imports_always_grounded(self):
        checker = GroundingChecker()
        files = {
            "pkg/__init__.py": "",
            "pkg/a.py": "from . import something\n",
        }
        report = checker.check_files(files)
        assert report.passed is True

    def test_common_third_party_grounded(self):
        checker = GroundingChecker()
        report = checker.check_files({
            "main.py": "import fastapi\nimport pydantic\n",
        })
        assert report.passed is True

    def test_syntax_error_skipped(self):
        checker = GroundingChecker()
        report = checker.check_files({"broken.py": "def f(\n"})
        assert report.passed is True  # Can't analyze, skip

    def test_non_python_skipped(self):
        checker = GroundingChecker()
        report = checker.check_files({"readme.md": "import fake\n"})
        assert report.passed is True

    def test_single_file(self):
        checker = GroundingChecker()
        report = checker.check_single("import os\n")
        assert report.passed is True

    def test_grounding_report_str(self):
        report = GroundingReport()
        assert "PASSED" in str(report)
        report = GroundingReport(violations=["used fake_mod"])
        assert "FAILED" in str(report)
