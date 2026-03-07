"""Built-in benchmark profiles.

Importing this module registers all built-in profiles into the global registry.
It is imported automatically by ``agentguard.benchmark.profiles.__init__``.
"""

from __future__ import annotations

from agentguard.benchmark.evaluator import evaluate_enterprise, evaluate_operational
from agentguard.benchmark.profiles.registry import BenchmarkProfile, register_profile
from agentguard.benchmark.types import DimensionScore, ReadinessScore

# ══════════════════════════════════════════════════════════════════
#  code — wraps the AST heuristic evaluator (default for code archetypes)
# ══════════════════════════════════════════════════════════════════


def _code_evaluate(
    spec: str,
    files: dict[str, str],
    enterprise_threshold: float,
    operational_threshold: float,
) -> tuple[ReadinessScore, ReadinessScore]:
    return (
        evaluate_enterprise(files, enterprise_threshold),
        evaluate_operational(files, operational_threshold),
    )


# ══════════════════════════════════════════════════════════════════
#  documentation — heading structure, length, code examples
# ══════════════════════════════════════════════════════════════════


def _documentation_evaluate(
    spec: str,
    files: dict[str, str],
    enterprise_threshold: float,
    operational_threshold: float,
) -> tuple[ReadinessScore, ReadinessScore]:
    all_content = "\n".join(files.values())
    lines = all_content.splitlines()

    headings = [ln for ln in lines if ln.startswith("#")]
    heading_score = min(1.0, len(headings) / 5)

    word_count = len(all_content.split())
    length_score = min(1.0, word_count / 200)

    code_blocks = all_content.count("```")
    example_score = min(1.0, code_blocks / 4)

    dims = [
        DimensionScore("heading_structure", heading_score, heading_score >= enterprise_threshold),
        DimensionScore("content_length", length_score, length_score >= enterprise_threshold),
        DimensionScore("code_examples", example_score, example_score >= enterprise_threshold),
    ]
    overall = sum(d.score for d in dims) / len(dims)
    readiness = ReadinessScore("documentation", overall, overall >= enterprise_threshold, dims)
    return readiness, readiness


# ══════════════════════════════════════════════════════════════════
#  archetype — YAML validity + required field checks
# ══════════════════════════════════════════════════════════════════


def _archetype_evaluate(
    spec: str,
    files: dict[str, str],
    enterprise_threshold: float,
    operational_threshold: float,
) -> tuple[ReadinessScore, ReadinessScore]:
    try:
        import yaml as _yaml
    except ImportError:
        return _generic_evaluate(spec, files, enterprise_threshold, operational_threshold)

    yaml_files = {p: c for p, c in files.items() if p.endswith((".yaml", ".yml"))}

    if not yaml_files:
        dims = [DimensionScore("yaml_present", 0.0, False, ["No YAML files found"])]
        score = ReadinessScore("archetype", 0.0, False, dims)
        return score, score

    valid_count = 0
    has_required_fields = 0

    for content in yaml_files.values():
        try:
            data = _yaml.safe_load(content)
            if isinstance(data, dict):
                valid_count += 1
                if "id" in data and "name" in data:
                    has_required_fields += 1
        except Exception:
            pass

    total = len(yaml_files)
    validity_score = valid_count / total
    fields_score = has_required_fields / max(valid_count, 1) if valid_count > 0 else 0.0

    dims = [
        DimensionScore("yaml_validity", validity_score, validity_score >= enterprise_threshold),
        DimensionScore("required_fields", fields_score, fields_score >= enterprise_threshold),
    ]
    overall = sum(d.score for d in dims) / len(dims)
    readiness = ReadinessScore("archetype", overall, overall >= enterprise_threshold, dims)
    return readiness, readiness


# ══════════════════════════════════════════════════════════════════
#  generic — universal fallback: presence, structure, spec coverage
# ══════════════════════════════════════════════════════════════════


def _generic_evaluate(
    spec: str,
    files: dict[str, str],
    enterprise_threshold: float,
    operational_threshold: float,
) -> tuple[ReadinessScore, ReadinessScore]:
    total_chars = sum(len(c) for c in files.values())
    num_files = len(files)

    # Output has meaningful content
    content_score = min(1.0, total_chars / 500)

    # Multiple files signal intentional structure
    structure_score = min(1.0, num_files / 3) if num_files > 0 else 0.0

    # Keywords from spec appear in the output
    spec_words = [w for w in spec.lower().split() if len(w) > 4]
    if spec_words:
        output_lower = " ".join(files.values()).lower()
        hits = sum(1 for w in spec_words if w in output_lower)
        coverage_score = min(1.0, hits / len(spec_words))
    else:
        coverage_score = 0.5

    dims = [
        DimensionScore("content_present", content_score, content_score >= enterprise_threshold),
        DimensionScore("structure", structure_score, structure_score >= enterprise_threshold),
        DimensionScore("spec_coverage", coverage_score, coverage_score >= enterprise_threshold),
    ]
    overall = sum(d.score for d in dims) / len(dims)
    readiness = ReadinessScore("generic", overall, overall >= enterprise_threshold, dims)
    return readiness, readiness


# ══════════════════════════════════════════════════════════════════
#  Register all built-in profiles
# ══════════════════════════════════════════════════════════════════

register_profile(BenchmarkProfile(
    name="code",
    description="Python code evaluation using AST-based enterprise + operational checkers.",
    evaluate=_code_evaluate,
))

register_profile(BenchmarkProfile(
    name="documentation",
    description="Heuristic evaluation for documentation and prose output.",
    evaluate=_documentation_evaluate,
))

register_profile(BenchmarkProfile(
    name="archetype",
    description="Evaluation for archetype YAML output: validity, required fields.",
    evaluate=_archetype_evaluate,
))

register_profile(BenchmarkProfile(
    name="generic",
    description="Universal fallback: content presence, structure, spec-keyword coverage.",
    evaluate=_generic_evaluate,
))
