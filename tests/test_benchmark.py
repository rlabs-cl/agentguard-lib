"""Tests for the benchmark module — types, catalog, evaluator, report, runner."""

from __future__ import annotations

import json

import pytest

# ══════════════════════════════════════════════════════════════════
#  Types
# ══════════════════════════════════════════════════════════════════

from agentguard.benchmark.types import (
    ALL_COMPLEXITIES,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkSpec,
    Complexity,
    ComplexityRun,
    DimensionScore,
    EnterpriseCheck,
    OperationalCheck,
    ReadinessScore,
    RunResult,
)


class TestComplexity:
    def test_all_five_levels(self) -> None:
        assert len(ALL_COMPLEXITIES) == 5
        assert Complexity.TRIVIAL in ALL_COMPLEXITIES
        assert Complexity.ENTERPRISE in ALL_COMPLEXITIES

    def test_string_values(self) -> None:
        assert Complexity.TRIVIAL == "trivial"
        assert Complexity.ENTERPRISE == "enterprise"


class TestEnterpriseCheck:
    def test_seven_dimensions(self) -> None:
        dims = list(EnterpriseCheck)
        assert len(dims) == 7
        assert EnterpriseCheck.TYPE_SAFETY in dims
        assert EnterpriseCheck.TESTABILITY in dims


class TestOperationalCheck:
    def test_six_dimensions(self) -> None:
        dims = list(OperationalCheck)
        assert len(dims) == 6
        assert OperationalCheck.DEBUGGABILITY in dims
        assert OperationalCheck.TEAM_ONBOARDING in dims


class TestDimensionScore:
    def test_to_dict(self) -> None:
        score = DimensionScore(
            dimension="type_safety",
            score=0.75,
            passed=True,
            findings=["Good type annotations"],
        )
        d = score.to_dict()
        assert d["dimension"] == "type_safety"
        assert d["score"] == 0.75
        assert d["passed"] is True
        assert len(d["findings"]) == 1

    def test_frozen(self) -> None:
        score = DimensionScore(dimension="x", score=0.5, passed=True)
        with pytest.raises(AttributeError):
            score.score = 0.9  # type: ignore[misc]


class TestReadinessScore:
    def test_to_dict(self) -> None:
        score = ReadinessScore(
            category="enterprise",
            overall_score=0.65,
            passed=True,
            dimensions=[
                DimensionScore("a", 0.7, True),
                DimensionScore("b", 0.6, True),
            ],
        )
        d = score.to_dict()
        assert d["category"] == "enterprise"
        assert len(d["dimensions"]) == 2


class TestRunResult:
    def _make_run_result(self, ent_score: float = 0.7, op_score: float = 0.6) -> RunResult:
        return RunResult(
            enterprise=ReadinessScore("enterprise", ent_score, ent_score >= 0.6),
            operational=ReadinessScore("operational", op_score, op_score >= 0.6),
            files_generated=5,
            total_lines=100,
            total_tokens=1000,
            cost_usd=0.05,
            duration_ms=2000,
        )

    def test_combined_score(self) -> None:
        r = self._make_run_result(0.8, 0.6)
        assert r.combined_score == pytest.approx(0.7)

    def test_to_dict_roundtrip(self) -> None:
        r = self._make_run_result()
        d = r.to_dict()
        assert d["files_generated"] == 5
        assert "enterprise" in d
        assert "operational" in d


class TestComplexityRun:
    def test_improvement_calculated(self) -> None:
        control = RunResult(
            enterprise=ReadinessScore("enterprise", 0.4, False),
            operational=ReadinessScore("operational", 0.3, False),
        )
        treatment = RunResult(
            enterprise=ReadinessScore("enterprise", 0.7, True),
            operational=ReadinessScore("operational", 0.6, True),
        )
        run = ComplexityRun(
            complexity=Complexity.MEDIUM,
            spec="Build a todo app",
            control=control,
            treatment=treatment,
        )
        assert run.improvement == pytest.approx(0.3)


class TestBenchmarkConfig:
    def test_validate_empty_model(self) -> None:
        config = BenchmarkConfig(model="")
        errors = config.validate()
        assert any("Model" in e for e in errors)

    def test_validate_missing_complexity_levels(self) -> None:
        config = BenchmarkConfig(
            model="anthropic/claude-sonnet-4-20250514",
            specs=[BenchmarkSpec(Complexity.TRIVIAL, "test", "general")],
        )
        errors = config.validate()
        assert any("Missing complexity" in e for e in errors)

    def test_validate_all_levels_covered(self) -> None:
        specs = [BenchmarkSpec(c, f"test {c}", "general") for c in ALL_COMPLEXITIES]
        config = BenchmarkConfig(model="anthropic/claude-sonnet-4-20250514", specs=specs)
        errors = config.validate()
        assert len(errors) == 0


class TestBenchmarkReport:
    def _make_report(self) -> BenchmarkReport:
        control = RunResult(
            enterprise=ReadinessScore("enterprise", 0.4, False),
            operational=ReadinessScore("operational", 0.3, False),
            cost_usd=0.01,
        )
        treatment = RunResult(
            enterprise=ReadinessScore("enterprise", 0.7, True),
            operational=ReadinessScore("operational", 0.6, True),
            cost_usd=0.03,
        )
        runs = [
            ComplexityRun(
                complexity=c,
                spec=f"Spec for {c}",
                control=control,
                treatment=treatment,
            )
            for c in ALL_COMPLEXITIES
        ]
        report = BenchmarkReport(
            archetype_id="api_backend",
            archetype_hash="abc123",
            model="anthropic/claude-sonnet-4-20250514",
            runs=runs,
        )
        report.compute_aggregates()
        return report

    def test_compute_aggregates(self) -> None:
        report = self._make_report()
        assert report.enterprise_avg == pytest.approx(0.7)
        assert report.operational_avg == pytest.approx(0.6)
        assert report.improvement_avg == pytest.approx(0.3)
        assert report.total_cost_usd == pytest.approx(0.2)  # 5 * (0.01 + 0.03)

    def test_json_roundtrip(self) -> None:
        report = self._make_report()
        json_str = report.to_json()
        restored = BenchmarkReport.from_json(json_str)
        assert restored.archetype_id == "api_backend"
        assert len(restored.runs) == 5
        assert restored.enterprise_avg == pytest.approx(0.7)

    def test_sign_and_verify(self) -> None:
        report = self._make_report()
        report.sign("my-secret")
        assert report.signature
        assert report.verify("my-secret")
        assert not report.verify("wrong-secret")

    def test_to_dict_has_all_fields(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        assert "version" in d
        assert "archetype_id" in d
        assert "runs" in d
        assert "overall_passed" in d


# ══════════════════════════════════════════════════════════════════
#  Catalog
# ══════════════════════════════════════════════════════════════════

from agentguard.benchmark.catalog import (
    BENCHMARK_CATALOG,
    get_default_specs,
    get_specs_for_category,
)


class TestCatalog:
    def test_all_standard_categories_present(self) -> None:
        expected = {"backend", "cli", "frontend", "fullstack", "library", "script", "data", "devops", "general"}
        assert expected.issubset(set(BENCHMARK_CATALOG.keys()))

    def test_alias_categories(self) -> None:
        assert "ml" in BENCHMARK_CATALOG
        assert "infra" in BENCHMARK_CATALOG
        assert "mobile" in BENCHMARK_CATALOG

    def test_each_category_has_5_complexity_levels(self) -> None:
        for cat, specs_by_level in BENCHMARK_CATALOG.items():
            for level in ALL_COMPLEXITIES:
                assert level in specs_by_level, f"{cat} missing {level}"
                assert len(specs_by_level[level]) >= 1, f"{cat}/{level} has no specs"

    def test_get_specs_for_category_fallback(self) -> None:
        # Unknown category should fall back to general
        specs = get_specs_for_category("unknown_category")
        assert specs is BENCHMARK_CATALOG["general"]

    def test_get_default_specs(self) -> None:
        defaults = get_default_specs("backend")
        assert len(defaults) == 5  # One per complexity level
        complexities = {s.complexity for s in defaults}
        assert complexities == set(ALL_COMPLEXITIES)


# ══════════════════════════════════════════════════════════════════
#  Evaluator
# ══════════════════════════════════════════════════════════════════

from agentguard.benchmark.evaluator import evaluate_enterprise, evaluate_operational


class TestEvaluateEnterprise:
    def test_empty_files(self) -> None:
        score = evaluate_enterprise({})
        assert score.category == "enterprise"
        assert score.overall_score < 0.3  # Low but not necessarily zero (base scores)
        assert not score.passed

    def test_well_typed_code_scores_higher(self) -> None:
        good_code = {
            "app.py": (
                '"""Application module."""\n'
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n"
                "from typing import Optional\n\n"
                "@dataclass\n"
                "class User:\n"
                '    """A user entity."""\n'
                "    name: str\n"
                "    email: str\n\n"
                "def get_user(user_id: int) -> User:\n"
                '    """Get user by ID."""\n'
                "    return User(name='test', email='test@test.com')\n"
            ),
        }
        bad_code = {
            "app.py": (
                "def get_user(user_id):\n"
                "    return {'name': 'test'}\n"
            ),
        }
        good_score = evaluate_enterprise(good_code)
        bad_score = evaluate_enterprise(bad_code)
        assert good_score.overall_score > bad_score.overall_score

    def test_seven_enterprise_dimensions(self) -> None:
        code = {"app.py": "x = 1\n"}
        score = evaluate_enterprise(code)
        assert len(score.dimensions) == 7

    def test_multi_file_modularity(self) -> None:
        single = {"app.py": "def main(): pass\n"}
        multi = {
            "models.py": "class User: pass\n",
            "routes.py": "def get(): pass\n",
            "services.py": "def process(): pass\n",
            "utils.py": "def helper(): pass\n",
        }
        s_score = evaluate_enterprise(single)
        m_score = evaluate_enterprise(multi)
        # Get modularity dim from each
        s_mod = next(d for d in s_score.dimensions if d.dimension == "modularity")
        m_mod = next(d for d in m_score.dimensions if d.dimension == "modularity")
        assert m_mod.score > s_mod.score


class TestEvaluateOperational:
    def test_empty_files(self) -> None:
        score = evaluate_operational({})
        assert score.category == "operational"
        assert score.overall_score < 0.3  # Low but not necessarily zero (base scores)
        assert not score.passed

    def test_six_operational_dimensions(self) -> None:
        code = {"app.py": "x = 1\n"}
        score = evaluate_operational(code)
        assert len(score.dimensions) == 6

    def test_code_with_tests_scores_higher(self) -> None:
        no_tests = {"app.py": "def main(): pass\n"}
        with_tests = {
            "app.py": "def main(): pass\n",
            "test_app.py": (
                "import pytest\n\n"
                "@pytest.fixture\n"
                "def client():\n"
                "    return object()\n\n"
                "def test_main(client):\n"
                "    assert True\n\n"
                "def test_edge_case():\n"
                "    with pytest.raises(ValueError):\n"
                "        raise ValueError('test')\n"
            ),
        }
        no_score = evaluate_operational(no_tests)
        with_score = evaluate_operational(with_tests)
        # test_surface dimension should be higher
        no_ts = next(d for d in no_score.dimensions if d.dimension == "test_surface")
        with_ts = next(d for d in with_score.dimensions if d.dimension == "test_surface")
        assert with_ts.score > no_ts.score

    def test_debuggability_custom_exceptions(self) -> None:
        code = {
            "errors.py": (
                "class AppError(Exception):\n"
                '    """Base application error."""\n'
                "    pass\n\n"
                "class NotFoundError(AppError):\n"
                "    pass\n"
            ),
            "app.py": (
                "from errors import NotFoundError\n"
                "import logging\n"
                "logger = logging.getLogger(__name__)\n\n"
                "def get_item(item_id: int) -> dict:\n"
                "    logger.info('Getting item %d', item_id)\n"
                "    raise NotFoundError(f'Item {item_id} not found') from None\n"
            ),
        }
        score = evaluate_operational(code)
        debug = next(d for d in score.dimensions if d.dimension == "debuggability")
        assert debug.score > 0.3


# ══════════════════════════════════════════════════════════════════
#  Report formatter
# ══════════════════════════════════════════════════════════════════

from agentguard.benchmark.report import format_report_compact, format_report_markdown


class TestReportFormatter:
    def _make_report(self) -> BenchmarkReport:
        control = RunResult(
            enterprise=ReadinessScore("enterprise", 0.4, False),
            operational=ReadinessScore("operational", 0.3, False),
            cost_usd=0.01,
        )
        treatment = RunResult(
            enterprise=ReadinessScore("enterprise", 0.7, True),
            operational=ReadinessScore("operational", 0.6, True),
            cost_usd=0.03,
        )
        runs = [
            ComplexityRun(
                complexity=c,
                spec=f"Build a test for {c.value} complexity",
                control=control,
                treatment=treatment,
            )
            for c in ALL_COMPLEXITIES
        ]
        report = BenchmarkReport(
            archetype_id="api_backend",
            model="anthropic/claude-sonnet-4-20250514",
            runs=runs,
            overall_passed=True,
            created_at="2025-01-01T00:00:00Z",
        )
        report.compute_aggregates()
        return report

    def test_markdown_contains_key_sections(self) -> None:
        report = self._make_report()
        md = format_report_markdown(report)
        assert "# AgentGuard Benchmark Report" in md
        assert "PASSED" in md
        assert "api_backend" in md
        assert "Summary" in md
        assert "Trivial" in md
        assert "Enterprise" in md

    def test_compact_format(self) -> None:
        report = self._make_report()
        line = format_report_compact(report)
        assert "PASS" in line
        assert "api_backend" in line
        assert "5 runs" in line


# ══════════════════════════════════════════════════════════════════
#  Runner (unit-level tests — no actual LLM calls)
# ══════════════════════════════════════════════════════════════════

from agentguard.benchmark.runner import _parse_file_blocks


class TestParseFileBlocks:
    def test_standard_format(self) -> None:
        content = """
```app.py
print("hello")
```

```utils/helpers.py
def helper():
    return 42
```
"""
        files = _parse_file_blocks(content)
        assert "app.py" in files
        assert "utils/helpers.py" in files
        assert "42" in files["utils/helpers.py"]

    def test_language_prefix(self) -> None:
        content = """
```python app.py
import sys
print(sys.argv)
```
"""
        files = _parse_file_blocks(content)
        assert "app.py" in files

    def test_angle_brackets(self) -> None:
        content = """
```<src/main.py>
def main():
    pass
```
"""
        files = _parse_file_blocks(content)
        assert "src/main.py" in files

    def test_no_blocks_returns_empty(self) -> None:
        files = _parse_file_blocks("Just some text without code blocks")
        assert files == {}
