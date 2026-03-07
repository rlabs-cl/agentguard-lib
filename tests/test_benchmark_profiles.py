"""Tests for the dynamic benchmark profile system.

Three evaluation paths are tested:
1. Named profile  — archetype specifies ``benchmark.profile = "code"`` (or any registered profile)
2. Inline criteria — archetype specifies ``benchmark.criteria`` → LLM-as-judge
3. Generic fallback — archetype has no profile and no criteria
"""

from __future__ import annotations

import pytest

from agentguard.benchmark.criteria_evaluator import (
    CriteriaBasedEvaluator,
    _files_to_text,
    _parse_judge_response,
)
from agentguard.benchmark.profiles import (
    BenchmarkProfile,
    get_profile,
    list_profiles,
    register_profile,
)
from agentguard.benchmark.profiles.builtin import (
    _archetype_evaluate,
    _code_evaluate,
    _documentation_evaluate,
    _generic_evaluate,
)
from agentguard.benchmark.types import (
    BenchmarkConfig,
    BenchmarkCriterion,
    BenchmarkSpec,
    Complexity,
)

# ══════════════════════════════════════════════════════════════════
#  Profile registry
# ══════════════════════════════════════════════════════════════════


class TestProfileRegistry:
    def test_builtin_profiles_registered(self) -> None:
        profiles = list_profiles()
        assert "code" in profiles
        assert "generic" in profiles
        assert "documentation" in profiles
        assert "archetype" in profiles

    def test_get_existing_profile(self) -> None:
        p = get_profile("code")
        assert p is not None
        assert p.name == "code"
        assert callable(p.evaluate)

    def test_get_missing_profile(self) -> None:
        assert get_profile("does_not_exist_xyz") is None

    def test_register_custom_profile(self) -> None:
        def _eval(spec, files, et, ot):
            from agentguard.benchmark.types import DimensionScore, ReadinessScore
            dims = [DimensionScore("custom", 0.9, True)]
            r = ReadinessScore("custom", 0.9, True, dims)
            return r, r

        register_profile(BenchmarkProfile(name="test_custom_xyz", description="test", evaluate=_eval))
        assert get_profile("test_custom_xyz") is not None

    def test_list_profiles_sorted(self) -> None:
        profiles = list_profiles()
        assert profiles == sorted(profiles)


# ══════════════════════════════════════════════════════════════════
#  Path 1 — Named profile (code)
# ══════════════════════════════════════════════════════════════════


class TestNamedProfileCode:
    """code profile delegates to the existing AST evaluator."""

    def test_code_profile_with_python_files(self) -> None:
        good_code = {
            "app.py": (
                '"""Module."""\n'
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n\n"
                "@dataclass\n"
                "class Task:\n"
                '    """Task entity."""\n'
                "    title: str\n"
                "    done: bool = False\n\n"
                "def complete(task: Task) -> Task:\n"
                '    """Mark task as done."""\n'
                "    return Task(title=task.title, done=True)\n"
            ),
        }
        enterprise, operational = _code_evaluate("build a task manager", good_code, 0.6, 0.6)
        assert enterprise.category == "enterprise"
        assert operational.category == "operational"
        assert len(enterprise.dimensions) == 7
        assert len(operational.dimensions) == 6

    def test_code_profile_empty_files(self) -> None:
        enterprise, operational = _code_evaluate("hello", {}, 0.6, 0.6)
        assert enterprise.overall_score < 0.3
        assert not enterprise.passed


# ══════════════════════════════════════════════════════════════════
#  Path 2 — Inline criteria (LLM-as-judge)
# ══════════════════════════════════════════════════════════════════


class TestCriteriaBasedEvaluator:
    """Inline criteria path using a mock LLM judge."""

    def _make_criteria(self) -> list[BenchmarkCriterion]:
        return [
            BenchmarkCriterion(
                name="completeness",
                description="Output covers all aspects of the spec",
                rubric="0–3: Missing main aspects. 4–6: Partial. 7–10: Complete.",
                weight=1.0,
            ),
            BenchmarkCriterion(
                name="clarity",
                description="Output is clear and readable",
                rubric="0–3: Confusing. 4–6: Acceptable. 7–10: Clear.",
                weight=1.0,
            ),
        ]

    @pytest.mark.asyncio
    async def test_evaluates_with_llm_judge(self, mock_llm) -> None:
        criteria = self._make_criteria()
        evaluator = CriteriaBasedEvaluator(criteria, threshold=0.6)
        files = {"output.md": "# Analysis\n\nThis covers all required aspects.\n\nCode:\n```\ndef solve(): pass\n```\n"}
        enterprise, operational = await evaluator.evaluate("Analyse the spec", files, mock_llm)

        assert enterprise.category == "criteria"
        assert enterprise is operational  # same object — criteria doesn't split categories
        assert len(enterprise.dimensions) == 2
        assert enterprise.dimensions[0].dimension == "completeness"
        assert enterprise.dimensions[1].dimension == "clarity"

    @pytest.mark.asyncio
    async def test_empty_criteria_returns_zero(self, mock_llm) -> None:
        evaluator = CriteriaBasedEvaluator([], threshold=0.6)
        files = {"out.txt": "hello"}
        enterprise, operational = await evaluator.evaluate("spec", files, mock_llm)
        assert enterprise.overall_score == 0.0
        assert not enterprise.passed

    @pytest.mark.asyncio
    async def test_zero_weight_criterion_excluded(self, mock_llm) -> None:
        criteria = [
            BenchmarkCriterion(name="active", description="d", rubric="r", weight=1.0),
            BenchmarkCriterion(name="inactive", description="d", rubric="r", weight=0.0),
        ]
        evaluator = CriteriaBasedEvaluator(criteria, threshold=0.6)
        files = {"out.txt": "content"}
        enterprise, _ = await evaluator.evaluate("spec", files, mock_llm)
        # Only 1 dimension — the inactive one should be excluded
        assert len(enterprise.dimensions) == 1
        assert enterprise.dimensions[0].dimension == "active"

    @pytest.mark.asyncio
    async def test_weighted_aggregate(self, mock_llm_scores) -> None:
        """Weighted average is computed correctly."""
        criteria = [
            BenchmarkCriterion(name="a", description="d", rubric="r", weight=2.0),
            BenchmarkCriterion(name="b", description="d", rubric="r", weight=1.0),
        ]
        evaluator = CriteriaBasedEvaluator(criteria, threshold=0.6)
        files = {"out.txt": "content"}
        # mock_llm_scores returns 8/10 for "a" and 5/10 for "b"
        enterprise, _ = await evaluator.evaluate("spec", files, mock_llm_scores)
        # weighted avg = (0.8*2 + 0.5*1) / 3 = 2.1/3 = 0.7
        assert enterprise.overall_score == pytest.approx(0.7, abs=0.01)


# ══════════════════════════════════════════════════════════════════
#  Path 3 — Generic fallback
# ══════════════════════════════════════════════════════════════════


class TestGenericProfile:
    """Generic fallback profile used when no profile or criteria are defined."""

    def test_generic_scores_content_present(self) -> None:
        files = {
            "output.txt": "This is long content that covers many words from the specification.\n" * 20,
        }
        enterprise, operational = _generic_evaluate(
            "build something with words from the specification",
            files, 0.6, 0.6,
        )
        assert enterprise.overall_score > 0.3
        assert enterprise.category == "generic"
        assert enterprise is operational  # generic doesn't split

    def test_generic_empty_output_scores_low(self) -> None:
        enterprise, _ = _generic_evaluate("spec", {}, 0.6, 0.6)
        assert enterprise.overall_score < 0.3

    def test_generic_spec_coverage(self) -> None:
        spec = "build a fastapi authentication service with jwt tokens"
        files = {"api.py": "fastapi authentication service jwt tokens implementation here\n" * 5}
        enterprise, _ = _generic_evaluate(spec, files, 0.6, 0.6)
        # Should score well on spec_coverage dimension
        coverage_dim = next(d for d in enterprise.dimensions if d.dimension == "spec_coverage")
        assert coverage_dim.score > 0.5

    def test_generic_has_three_dimensions(self) -> None:
        files = {"f.txt": "hello world"}
        enterprise, _ = _generic_evaluate("spec", files, 0.6, 0.6)
        assert len(enterprise.dimensions) == 3
        names = {d.dimension for d in enterprise.dimensions}
        assert names == {"content_present", "structure", "spec_coverage"}


# ══════════════════════════════════════════════════════════════════
#  Documentation profile
# ══════════════════════════════════════════════════════════════════


class TestDocumentationProfile:
    def test_well_structured_docs_score_higher(self) -> None:
        good_docs = {
            "README.md": (
                "# Project Title\n\n"
                "## Overview\n\nThis is a well-structured README.\n\n"
                "## Installation\n\n```bash\npip install mylib\n```\n\n"
                "## Usage\n\n```python\nimport mylib\n```\n\n"
                "## Contributing\n\nPlease open a PR.\n\n"
                "## License\n\nMIT\n"
            ),
        }
        poor_docs = {"README.md": "hello this is my project it does stuff"}

        good_enterprise, _ = _documentation_evaluate("create a README", good_docs, 0.6, 0.6)
        poor_enterprise, _ = _documentation_evaluate("create a README", poor_docs, 0.6, 0.6)
        assert good_enterprise.overall_score > poor_enterprise.overall_score

    def test_documentation_category(self) -> None:
        files = {"doc.md": "# Title\n\nContent.\n"}
        enterprise, _ = _documentation_evaluate("spec", files, 0.6, 0.6)
        assert enterprise.category == "documentation"


# ══════════════════════════════════════════════════════════════════
#  Archetype profile
# ══════════════════════════════════════════════════════════════════


class TestArchetypeProfile:
    def test_valid_yaml_scores_higher(self) -> None:
        valid_yaml = {
            "my_archetype.yaml": "id: my_arch\nname: My Archetype\ndescription: Test archetype\n",
        }
        invalid_yaml = {
            "broke.yaml": "{ invalid: yaml: content: }",
        }
        good_e, _ = _archetype_evaluate("create an archetype", valid_yaml, 0.6, 0.6)
        bad_e, _ = _archetype_evaluate("create an archetype", invalid_yaml, 0.6, 0.6)
        assert good_e.overall_score > bad_e.overall_score

    def test_no_yaml_files_scores_zero(self) -> None:
        files = {"something.py": "x = 1"}
        enterprise, _ = _archetype_evaluate("spec", files, 0.6, 0.6)
        assert enterprise.overall_score == 0.0

    def test_archetype_category(self) -> None:
        files = {"arch.yaml": "id: test\nname: Test\n"}
        enterprise, _ = _archetype_evaluate("spec", files, 0.6, 0.6)
        assert enterprise.category == "archetype"


# ══════════════════════════════════════════════════════════════════
#  Judge response parser
# ══════════════════════════════════════════════════════════════════


class TestParseJudgeResponse:
    def test_standard_format(self) -> None:
        text = "Score: 8/10\nReason: Output is clear and complete."
        score, reason = _parse_judge_response(text)
        assert score == pytest.approx(0.8)
        assert "clear" in reason.lower()

    def test_decimal_score(self) -> None:
        text = "Score: 7.5/10\nReason: Good but missing edge cases."
        score, reason = _parse_judge_response(text)
        assert score == pytest.approx(0.75)

    def test_malformed_response_defaults_to_05(self) -> None:
        score, reason = _parse_judge_response("I think it's okay")
        assert score == pytest.approx(0.5)
        assert reason == "No reason provided."

    def test_score_clamped_to_10(self) -> None:
        score, _ = _parse_judge_response("Score: 10/10\nReason: Perfect.")
        assert score == pytest.approx(1.0)

    def test_score_clamped_minimum(self) -> None:
        score, _ = _parse_judge_response("Score: 0/10\nReason: Empty output.")
        assert score == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════
#  Files-to-text helper
# ══════════════════════════════════════════════════════════════════


class TestFilesToText:
    def test_single_file(self) -> None:
        text = _files_to_text({"app.py": "x = 1\n"})
        assert "=== app.py ===" in text
        assert "x = 1" in text

    def test_multiple_files(self) -> None:
        files = {"a.py": "a = 1\n", "b.py": "b = 2\n"}
        text = _files_to_text(files)
        assert "=== a.py ===" in text
        assert "=== b.py ===" in text

    def test_empty_files(self) -> None:
        text = _files_to_text({})
        assert text == ""


# ══════════════════════════════════════════════════════════════════
#  BenchmarkConfig.validate() relaxation
# ══════════════════════════════════════════════════════════════════


class TestBenchmarkConfigValidation:
    def test_partial_complexities_allowed_when_relaxed(self) -> None:
        config = BenchmarkConfig(
            specs=[BenchmarkSpec(Complexity.TRIVIAL, "test", "general")],
            require_all_complexities=False,
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_partial_complexities_rejected_by_default(self) -> None:
        config = BenchmarkConfig(
            specs=[BenchmarkSpec(Complexity.TRIVIAL, "test", "general")],
        )
        errors = config.validate()
        assert any("Missing complexity" in e for e in errors)

    def test_all_complexities_still_valid_with_flag(self) -> None:
        from agentguard.benchmark.types import ALL_COMPLEXITIES
        specs = [BenchmarkSpec(c, f"test {c}", "general") for c in ALL_COMPLEXITIES]
        config = BenchmarkConfig(specs=specs, require_all_complexities=False)
        assert len(config.validate()) == 0


# ══════════════════════════════════════════════════════════════════
#  BenchmarkCriterion dataclass
# ══════════════════════════════════════════════════════════════════


class TestBenchmarkCriterion:
    def test_default_weight(self) -> None:
        c = BenchmarkCriterion(
            name="completeness",
            description="covers all aspects",
            rubric="0=nothing, 1=everything",
        )
        assert c.weight == 1.0

    def test_custom_weight(self) -> None:
        c = BenchmarkCriterion(
            name="quality", description="d", rubric="r", weight=2.5,
        )
        assert c.weight == 2.5


# ══════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════


class _MockLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    class _cost:
        total_cost = 0.001

    class _tokens:
        total_tokens = 0

    cost = _cost()
    tokens = _tokens()


class _MockLLM:
    """LLM mock that always returns Score: 7/10."""

    async def generate(self, messages):
        return _MockLLMResponse("Score: 7/10\nReason: Mock judge response.")


class _MockLLMScores:
    """LLM mock that returns 8/10 for first call, 5/10 for second."""

    def __init__(self):
        self._call_count = 0

    async def generate(self, messages):
        self._call_count += 1
        score = 8 if self._call_count == 1 else 5
        return _MockLLMResponse(f"Score: {score}/10\nReason: Mock response {self._call_count}.")


@pytest.fixture
def mock_llm():
    return _MockLLM()


@pytest.fixture
def mock_llm_scores():
    return _MockLLMScores()
