"""Tests for the challenge module — SelfChallenger, GroundingChecker, types."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentguard.challenge.challenger import (
    SelfChallenger,
    _parse_criteria_results,
    _parse_grounding_section,
)
from agentguard.challenge.grounding import GroundingChecker, GroundingReport
from agentguard.challenge.types import ChallengeResult, CriterionResult
from tests.conftest import MockLLMProvider

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

    def test_challenge_result_cost_default(self):
        result = ChallengeResult(passed=True)
        assert result.cost.total_cost == Decimal("0")


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
#  SelfChallenger
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
class TestSelfChallenger:
    async def test_challenge_passes_when_all_criteria_pass(self):
        """When the LLM responds with all PASS, the challenge should pass."""
        response = (
            "CRITERION 1: PASS: Endpoints match\n"
            "CRITERION 2: PASS: No secrets found\n"
            "\n"
            "GROUNDING:\n"
            "- NONE\n"
            "- NONE\n"
        )
        llm = MockLLMProvider(responses=[response])
        challenger = SelfChallenger(llm=llm)

        result = await challenger.challenge(
            output="def hello(): pass",
            criteria=["Endpoints match spec", "No hardcoded secrets"],
            auto_rework=False,
        )
        assert result.passed is True
        assert len(result.criteria_results) == 2
        assert all(c.passed for c in result.criteria_results)

    async def test_challenge_fails_and_reworks(self):
        """When criteria fail, challenger should attempt rework."""
        fail_response = (
            "CRITERION 1: FAIL: Missing endpoint\n"
            "\n"
            "GROUNDING:\n"
            "- NONE\n"
            "- NONE\n"
        )
        rework_response = "def hello(): return 'fixed'"
        pass_response = (
            "CRITERION 1: PASS: Now correct\n"
            "\n"
            "GROUNDING:\n"
            "- NONE\n"
            "- NONE\n"
        )
        llm = MockLLMProvider(responses=[fail_response, rework_response, pass_response])
        challenger = SelfChallenger(llm=llm)

        result = await challenger.challenge(
            output="def hello(): pass",
            criteria=["Has correct endpoint"],
            max_retries=3,
            auto_rework=True,
        )
        assert result.passed is True
        assert result.attempt == 2
        assert llm.call_count == 3  # eval + rework + eval

    async def test_challenge_with_grounding_violations(self):
        response = (
            "CRITERION 1: PASS: OK\n"
            "\n"
            "GROUNDING:\n"
            "- VIOLATION: Used imaginary_api module\n"
            "- NONE\n"
        )
        llm = MockLLMProvider(responses=[response])
        challenger = SelfChallenger(llm=llm)

        result = await challenger.evaluate_only(
            output="import imaginary_api",
            criteria=["Code is correct"],
        )
        assert result.passed is False  # Grounding violation causes failure
        assert len(result.grounding_violations) == 1

    async def test_rework_standalone(self):
        llm = MockLLMProvider(responses=["def fixed(): return 42\n"])
        challenger = SelfChallenger(llm=llm)

        result = await challenger.rework(
            output="def broken(): pass",
            feedback="Function should return 42",
        )
        assert "fixed" in result

    async def test_max_retries_exhausted(self):
        """When all retries fail, return last result with passed=False."""
        fail = "CRITERION 1: FAIL: Still wrong\nGROUNDING:\n- NONE\n- NONE\n"
        rework = "def still_broken(): pass"
        llm = MockLLMProvider(responses=[fail, rework, fail])
        challenger = SelfChallenger(llm=llm)

        result = await challenger.challenge(
            output="def broken(): pass",
            criteria=["Must work"],
            max_retries=2,
        )
        assert result.passed is False


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
