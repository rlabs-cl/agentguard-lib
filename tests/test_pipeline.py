"""Tests for the Pipeline module."""

from __future__ import annotations

import json

import pytest

from agentguard.challenge.types import ChallengeResult
from agentguard.pipeline import Pipeline
from agentguard.validation.types import ValidationReport


class TestValidationReport:
    def test_passed(self):
        report = ValidationReport(passed=True, checks=[], auto_fixed=[], errors=[])
        assert "PASSED" in str(report)

    def test_failed(self):
        from agentguard.validation.types import ValidationError
        err = ValidationError(check="syntax", file_path="x.py", line=1, message="bad")
        report = ValidationReport(passed=False, checks=[], auto_fixed=[], errors=[err])
        assert "FAILED" in str(report)


class TestChallengeResult:
    def test_passed(self):
        result = ChallengeResult(passed=True)
        assert "PASSED" in str(result)


class TestPipelineInit:
    def test_init_with_strings(self):
        """Pipeline should accept string args and resolve them."""
        pipe = Pipeline(
            archetype="api_backend",
            llm="anthropic/claude-sonnet-4-20250514",
        )
        assert pipe.archetype.id == "api_backend"

    def test_init_with_objects(self, api_backend_archetype, mock_llm):
        pipe = Pipeline(
            archetype=api_backend_archetype,
            llm=mock_llm,
        )
        assert pipe.archetype.id == "api_backend"


@pytest.mark.asyncio
class TestPipelineGenerate:
    async def test_full_generate(self, api_backend_archetype, mock_llm_factory):
        """Integration: run the full pipeline with mock LLM."""
        skeleton_json = json.dumps([
            {"path": "main.py", "purpose": "Entry point"},
        ])
        contract_code = 'def start():\n    """Start."""\n    raise NotImplementedError\n'
        wired_code = 'def start():\n    """Start."""\n    raise NotImplementedError\n'
        logic_code = 'def start():\n    """Start."""\n    print("started")\n'

        llm = mock_llm_factory([skeleton_json, contract_code, wired_code, logic_code])

        pipe = Pipeline(
            archetype=api_backend_archetype,
            llm=llm,
            trace_store=None,
        )

        result = await pipe.generate(
            "A simple app",
            skip_challenge=True,
        )

        assert len(result.files) >= 1
        assert "main.py" in result.files

    async def test_validate_valid_code(self, api_backend_archetype, mock_llm):
        pipe = Pipeline(archetype=api_backend_archetype, llm=mock_llm)
        report = await pipe.validate({"app.py": "x = 1\n"})
        assert report.passed is True

    async def test_validate_invalid_code(self, api_backend_archetype, mock_llm):
        pipe = Pipeline(archetype=api_backend_archetype, llm=mock_llm)
        report = await pipe.validate({"app.py": "def broken(\n"})
        assert report.passed is False
        assert len(report.errors) >= 1

    async def test_challenge_no_criteria(self, api_backend_archetype, mock_llm):
        """Challenge with empty criteria should pass immediately."""
        pipe = Pipeline(archetype=api_backend_archetype, llm=mock_llm)
        result = await pipe.challenge("x = 1", criteria=[])
        assert result.passed is True


@pytest.mark.asyncio
class TestPipelineSteps:
    async def test_skeleton_step(self, api_backend_archetype, mock_llm_factory):
        skeleton_json = json.dumps([
            {"path": "main.py", "purpose": "Entry"},
            {"path": "models.py", "purpose": "Models"},
        ])
        llm = mock_llm_factory([skeleton_json])

        pipe = Pipeline(archetype=api_backend_archetype, llm=llm)
        skeleton = await pipe.skeleton("A simple API")

        assert len(skeleton.files) == 2
