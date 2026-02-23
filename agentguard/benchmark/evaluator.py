"""Evaluator — scores generated code across enterprise and operational readiness.

Enterprise readiness (7 dimensions):
  type_safety, modularity, maintainability, accessibility,
  performance, observability, testability

Operational readiness (6 dimensions):
  debuggability, feature_extensibility, cloud_scalability,
  api_migration_cost, test_surface, team_onboarding

Each dimension yields a 0.0 – 1.0 score based on static-analysis heuristics
applied to the generated file set.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable

from agentguard.benchmark.types import (
    DimensionScore,
    EnterpriseCheck,
    OperationalCheck,
    ReadinessScore,
)

# Type alias for checker functions (compatible with Python 3.11+).
_Checker = Callable[[dict[str, str]], DimensionScore]


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════


def evaluate_enterprise(
    files: dict[str, str],
    threshold: float = 0.6,
) -> ReadinessScore:
    """Score enterprise readiness across all 7 dimensions."""
    checkers: dict[str, _Checker] = {
        EnterpriseCheck.TYPE_SAFETY: _check_type_safety,
        EnterpriseCheck.MODULARITY: _check_modularity,
        EnterpriseCheck.MAINTAINABILITY: _check_maintainability,
        EnterpriseCheck.ACCESSIBILITY: _check_accessibility,
        EnterpriseCheck.PERFORMANCE: _check_performance,
        EnterpriseCheck.OBSERVABILITY: _check_observability,
        EnterpriseCheck.TESTABILITY: _check_testability,
    }
    return _evaluate_category("enterprise", checkers, files, threshold)


def evaluate_operational(
    files: dict[str, str],
    threshold: float = 0.6,
) -> ReadinessScore:
    """Score operational readiness across all 6 dimensions."""
    checkers: dict[str, _Checker] = {
        OperationalCheck.DEBUGGABILITY: _check_debuggability,
        OperationalCheck.FEATURE_EXTENSIBILITY: _check_feature_extensibility,
        OperationalCheck.CLOUD_SCALABILITY: _check_cloud_scalability,
        OperationalCheck.API_MIGRATION_COST: _check_api_migration_cost,
        OperationalCheck.TEST_SURFACE: _check_test_surface,
        OperationalCheck.TEAM_ONBOARDING: _check_team_onboarding,
    }
    return _evaluate_category("operational", checkers, files, threshold)


# ══════════════════════════════════════════════════════════════════
#  SHARED ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════════


def _py_files(files: dict[str, str]) -> dict[str, str]:
    """Filter to Python source files only."""
    return {p: c for p, c in files.items() if p.endswith(".py")}


def _total_lines(files: dict[str, str]) -> int:
    return sum(c.count("\n") + 1 for c in files.values())


def _parse_trees(py_files: dict[str, str]) -> dict[str, ast.Module | None]:
    """Parse Python files, returning None for files that fail to parse."""
    trees: dict[str, ast.Module | None] = {}
    for path, content in py_files.items():
        try:
            trees[path] = ast.parse(content, filename=path)
        except SyntaxError:
            trees[path] = None
    return trees


def _count_functions(tree: ast.Module) -> int:
    return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))


def _count_classes(tree: ast.Module) -> int:
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def _has_pattern(content: str, pattern: str) -> bool:
    return bool(re.search(pattern, content, re.IGNORECASE))


def _ratio_with_cap(numerator: float, denominator: float, cap: float = 1.0) -> float:
    """Ratio capped at `cap`, returns 0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return min(numerator / denominator, cap)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score(value: float, dimension: str, findings: list[str], threshold: float = 0.6) -> DimensionScore:
    s = _clamp(value)
    return DimensionScore(
        dimension=dimension,
        score=s,
        passed=s >= threshold,
        findings=findings,
    )


# ══════════════════════════════════════════════════════════════════
#  ENTERPRISE CHECKS
# ══════════════════════════════════════════════════════════════════


def _check_type_safety(files: dict[str, str]) -> DimensionScore:
    """Type annotations, dataclasses/TypedDict/Pydantic, typed returns."""
    pf = _py_files(files)
    trees = _parse_trees(pf)
    findings: list[str] = []
    if not trees:
        return _score(0.0, "type_safety", ["No Python files found"])

    total_funcs = 0
    annotated_funcs = 0
    has_type_imports = False

    for path, tree in trees.items():
        if tree is None:
            findings.append(f"{path}: syntax error — could not parse")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_funcs += 1
                if node.returns is not None:
                    annotated_funcs += 1
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in ("typing", "typing_extensions", "pydantic", "dataclasses"):
                    has_type_imports = True

    annotation_ratio = _ratio_with_cap(annotated_funcs, total_funcs)
    base = annotation_ratio * 0.7

    # Bonus: type-related imports (pydantic, dataclasses, typing)
    if has_type_imports:
        base += 0.15
        findings.append("Uses typing / pydantic / dataclasses imports")

    # Bonus: any file has `from __future__ import annotations`
    future_annotations = any(
        _has_pattern(c, r"from __future__ import annotations") for c in pf.values()
    )
    if future_annotations:
        base += 0.15
        findings.append("Uses `from __future__ import annotations`")

    findings.insert(0, f"{annotated_funcs}/{total_funcs} functions have return-type annotations")
    return _score(base, "type_safety", findings)


def _check_modularity(files: dict[str, str]) -> DimensionScore:
    """Multi-file structure, separation of concerns, reasonable file sizes."""
    pf = _py_files(files)
    findings: list[str] = []
    if not pf:
        return _score(0.0, "modularity", ["No Python files found"])

    n_files = len(pf)
    findings.append(f"{n_files} Python file(s)")

    # Multi-file bonus (>1 = 0.3, >3 = 0.5, >5 = 0.7)
    if n_files == 1:
        file_score = 0.2
    elif n_files <= 3:
        file_score = 0.4
    elif n_files <= 5:
        file_score = 0.6
    else:
        file_score = 0.8

    # File size variance — penalise god files (any single file >60% of lines)
    total = _total_lines(pf)
    max_file_lines = max((c.count("\n") + 1) for c in pf.values())
    max_ratio = max_file_lines / total if total else 1.0
    if max_ratio > 0.6 and n_files > 1:
        file_score -= 0.15
        findings.append(f"Largest file is {max_ratio:.0%} of total — consider splitting")

    # Bonus: __init__.py presence (package structure)
    if any(p.endswith("__init__.py") for p in pf):
        file_score += 0.1
        findings.append("Has __init__.py (package structure)")

    # Bonus: separate models/schemas/routes files
    concern_files = {"model", "schema", "route", "service", "util", "config", "handler"}
    found = [kw for kw in concern_files if any(kw in p.lower() for p in pf)]
    if found:
        file_score += min(len(found) * 0.05, 0.2)
        findings.append(f"Concern separation: {', '.join(found)}")

    return _score(file_score, "modularity", findings)


def _check_maintainability(files: dict[str, str]) -> DimensionScore:
    """Docstrings, comments, naming conventions, function length."""
    pf = _py_files(files)
    trees = _parse_trees(pf)
    findings: list[str] = []
    if not trees:
        return _score(0.0, "maintainability", ["No Python files found"])

    total_funcs = 0
    documented_funcs = 0
    long_funcs = 0  # functions > 50 lines

    for path, tree in trees.items():
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_funcs += 1
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                ):
                    documented_funcs += 1
                # Check function length
                if hasattr(node, "end_lineno") and node.end_lineno and node.lineno:
                    length = node.end_lineno - node.lineno
                    if length > 50:
                        long_funcs += 1

    doc_ratio = _ratio_with_cap(documented_funcs, total_funcs)
    base = doc_ratio * 0.5
    findings.append(f"{documented_funcs}/{total_funcs} functions have docstrings")

    # Penalise long functions
    if long_funcs and total_funcs:
        penalty = min(long_funcs / total_funcs * 0.3, 0.3)
        base -= penalty
        findings.append(f"{long_funcs} function(s) exceed 50 lines")

    # Bonus: module-level docstrings
    mod_docs = sum(
        1 for tree in trees.values()
        if tree and tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    )
    if mod_docs:
        base += 0.15
        findings.append(f"{mod_docs} file(s) have module docstrings")

    # Bonus: snake_case function names (convention compliance)
    snake_funcs = 0
    for tree in trees.values():
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if re.match(r"^[a-z_][a-z0-9_]*$", node.name) or node.name.startswith("_"):
                    snake_funcs += 1
    if total_funcs:
        snake_ratio = snake_funcs / total_funcs
        if snake_ratio > 0.8:
            base += 0.15
            findings.append("Consistent snake_case naming")

    # Bonus: constants / config separated
    for content in pf.values():
        if _has_pattern(content, r"^[A-Z_]{3,}\s*="):
            base += 0.1
            findings.append("Uses UPPER_CASE constants")
            break

    return _score(base, "maintainability", findings)


def _check_accessibility(files: dict[str, str]) -> DimensionScore:
    """i18n readiness, error messages quality, API documentation hints."""
    findings: list[str] = []
    all_content = "\n".join(files.values())

    base = 0.3  # Baseline for having any code

    # Check for string externalization patterns
    if _has_pattern(all_content, r"gettext|ngettext|_\(|i18n|locale"):
        base += 0.2
        findings.append("i18n / locale patterns detected")

    # Check for descriptive error messages (not bare `raise Exception`)
    bare_raises = len(re.findall(r"raise\s+Exception\s*\(", all_content))
    specific_raises = len(re.findall(r"raise\s+(?!Exception)\w+Error", all_content))
    if specific_raises > bare_raises:
        base += 0.15
        findings.append("Uses specific exception types over bare Exception")
    elif bare_raises:
        findings.append(f"{bare_raises} bare Exception raise(s) — prefer specific types")

    # Check for HTTP status code descriptions in APIs
    if _has_pattern(all_content, r"status_code|HTTPStatus|status\.HTTP"):
        base += 0.1
        findings.append("Uses explicit HTTP status codes")

    # Check for docstrings / API doc patterns (OpenAPI hints)
    if _has_pattern(all_content, r"@app\.\w+|@router\.\w+"):
        if _has_pattern(all_content, r'summary\s*=|description\s*=|response_model'):
            base += 0.15
            findings.append("API endpoints have documentation metadata")

    # Check for ARIA / a11y patterns in frontend code
    html_tsx = {p: c for p, c in files.items() if any(p.endswith(e) for e in (".tsx", ".jsx", ".html", ".vue"))}
    if html_tsx:
        a11y_patterns = ["aria-", "role=", "tabIndex", "alt=", "sr-only", "aria-label"]
        found_a11y = [p for p in a11y_patterns if any(p in c for c in html_tsx.values())]
        if found_a11y:
            base += 0.2
            findings.append(f"Accessibility: {', '.join(found_a11y)}")

    if not findings:
        findings.append("No specific accessibility patterns detected")

    return _score(base, "accessibility", findings)


def _check_performance(files: dict[str, str]) -> DimensionScore:
    """Async patterns, caching, pagination, connection pooling."""
    pf = _py_files(files)
    findings: list[str] = []
    all_content = "\n".join(files.values())

    base = 0.3

    # Async patterns
    if _has_pattern(all_content, r"async\s+def|await\s+|asyncio"):
        base += 0.15
        findings.append("Uses async/await patterns")

    # Caching
    if _has_pattern(all_content, r"@cache|lru_cache|@cached|redis|memcache|TTLCache"):
        base += 0.15
        findings.append("Caching mechanism detected")

    # Pagination
    if _has_pattern(all_content, r"offset|limit|page_size|paginate|skip.*take"):
        base += 0.1
        findings.append("Pagination patterns detected")

    # Connection pooling
    if _has_pattern(all_content, r"pool_size|create_pool|ConnectionPool|max_connections"):
        base += 0.1
        findings.append("Connection pooling detected")

    # Lazy loading / generators
    if _has_pattern(all_content, r"yield\s|\.stream\(|lazy|__iter__"):
        base += 0.1
        findings.append("Lazy loading / streaming patterns")

    # Batch processing
    if _has_pattern(all_content, r"batch|bulk|chunk"):
        base += 0.1
        findings.append("Batch processing patterns")

    if not findings:
        findings.append("No specific performance patterns detected")

    return _score(base, "performance", findings)


def _check_observability(files: dict[str, str]) -> DimensionScore:
    """Logging, metrics, tracing, health checks."""
    all_content = "\n".join(files.values())
    findings: list[str] = []

    base = 0.1

    # Structured logging
    if _has_pattern(all_content, r"logging\.getLogger|structlog|logger\.\w+"):
        base += 0.2
        findings.append("Structured logging present")

    # Log levels
    log_levels = ["debug", "info", "warning", "error", "critical"]
    found_levels = [lv for lv in log_levels if _has_pattern(all_content, rf"logger\.{lv}|logging\.{lv}")]
    if len(found_levels) >= 2:
        base += 0.1
        findings.append(f"Multiple log levels: {', '.join(found_levels)}")

    # Metrics / instrumentation
    if _has_pattern(all_content, r"prometheus|metrics|counter|histogram|gauge|statsd|datadog"):
        base += 0.15
        findings.append("Metrics instrumentation detected")

    # Distributed tracing
    if _has_pattern(all_content, r"opentelemetry|trace_id|span|jaeger|zipkin|tracing"):
        base += 0.15
        findings.append("Distributed tracing detected")

    # Health endpoints
    if _has_pattern(all_content, r"/health|/readiness|/liveness|healthcheck"):
        base += 0.15
        findings.append("Health check endpoint(s) present")

    # Error reporting
    if _has_pattern(all_content, r"sentry|bugsnag|rollbar|error_handler|exception_handler"):
        base += 0.1
        findings.append("Error reporting integration")

    if not findings:
        findings.append("No observability patterns detected")

    return _score(base, "observability", findings)


def _check_testability(files: dict[str, str]) -> DimensionScore:
    """Test files, dependency injection, abstract interfaces, mocking readiness."""
    pf = _py_files(files)
    all_content = "\n".join(files.values())
    findings: list[str] = []

    base = 0.2

    # Test files present
    test_files = [p for p in files if "test" in p.lower() or p.startswith("test_")]
    if test_files:
        base += 0.25
        findings.append(f"{len(test_files)} test file(s) present")

    # pytest / unittest patterns
    if _has_pattern(all_content, r"import pytest|from pytest|unittest\.TestCase|def test_"):
        base += 0.1
        findings.append("Test framework usage detected")

    # Dependency injection patterns (constructor injection)
    trees = _parse_trees(pf)
    di_count = 0
    for tree in trees.values():
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in ast.walk(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                        # Count parameters (excluding self)
                        params = len(item.args.args) - 1
                        if params >= 1:
                            di_count += 1
    if di_count:
        base += 0.15
        findings.append(f"{di_count} class(es) use constructor injection")

    # Abstract base classes / protocols
    if _has_pattern(all_content, r"ABC|abstractmethod|Protocol|@abstractmethod"):
        base += 0.15
        findings.append("Abstract interfaces / Protocols defined")

    # Fixtures / factories
    if _has_pattern(all_content, r"@pytest\.fixture|factory|conftest"):
        base += 0.1
        findings.append("Test fixtures / factories detected")

    if not findings:
        findings.append("No testability patterns detected")

    return _score(base, "testability", findings)


# ══════════════════════════════════════════════════════════════════
#  OPERATIONAL CHECKS
# ══════════════════════════════════════════════════════════════════


def _check_debuggability(files: dict[str, str]) -> DimensionScore:
    """Structured errors, descriptive messages, traceback context."""
    pf = _py_files(files)
    all_content = "\n".join(files.values())
    findings: list[str] = []

    base = 0.2

    # Custom exception classes
    custom_exc = len(re.findall(r"class\s+\w+(?:Error|Exception)\s*\(", all_content))
    if custom_exc:
        base += 0.2
        findings.append(f"{custom_exc} custom exception class(es)")

    # try/except with specific exceptions
    specific_catches = len(re.findall(r"except\s+(?!Exception\b|BaseException\b)\w+", all_content))
    bare_catches = len(re.findall(r"except\s*:|except\s+Exception\b", all_content))
    if specific_catches > bare_catches:
        base += 0.15
        findings.append("Prefers specific exception handling")
    elif bare_catches:
        findings.append(f"{bare_catches} bare except/Exception catch(es)")

    # Logging with context
    ctx_logs = len(re.findall(r"logger\.\w+\(.*%[sd]|logger\.\w+\(.*\{.*\}", all_content))
    if ctx_logs:
        base += 0.15
        findings.append(f"{ctx_logs} contextual log statement(s)")

    # Error context (raise from, chained exceptions)
    if _has_pattern(all_content, r"raise\s+\w+.*\bfrom\b"):
        base += 0.15
        findings.append("Uses exception chaining (raise ... from ...)")

    # repr / __str__ on data classes
    if _has_pattern(all_content, r"def __repr__|def __str__|@dataclass"):
        base += 0.1
        findings.append("Debug-friendly representations (__repr__ / dataclass)")

    if not findings:
        findings.append("No specific debuggability patterns detected")

    return _score(base, "debuggability", findings)


def _check_feature_extensibility(files: dict[str, str]) -> DimensionScore:
    """Plugin points, strategy pattern, configuration-driven behavior."""
    all_content = "\n".join(files.values())
    findings: list[str] = []

    base = 0.2

    # Abstract base / Protocol for extensibility
    if _has_pattern(all_content, r"ABC|Protocol|@abstractmethod|Abstract\w+"):
        base += 0.2
        findings.append("Abstract interfaces for extensibility")

    # Registry / plugin patterns
    if _has_pattern(all_content, r"register|registry|plugin|strategy|handler_map"):
        base += 0.15
        findings.append("Registry / plugin / strategy pattern")

    # Config-driven behavior
    if _has_pattern(all_content, r"config\[|settings\.|getenv|\.env|load_config"):
        base += 0.15
        findings.append("Configuration-driven behavior")

    # Hooks / events / signals
    if _has_pattern(all_content, r"on_\w+|hook|signal|event_handler|subscriber|emit"):
        base += 0.15
        findings.append("Event hooks / signals pattern")

    # Decorator-based extension
    if _has_pattern(all_content, r"def\s+\w+\(.*func.*\)|@\w+\.register"):
        base += 0.1
        findings.append("Decorator-based extension points")

    if not findings:
        findings.append("No extensibility patterns detected")

    return _score(base, "feature_extensibility", findings)


def _check_cloud_scalability(files: dict[str, str]) -> DimensionScore:
    """Stateless design, env-var config, health checks, container readiness."""
    all_content = "\n".join(files.values())
    findings: list[str] = []

    base = 0.2

    # Environment variable configuration (12-factor)
    env_refs = len(re.findall(r"os\.environ|os\.getenv|environ\.get|settings_from_env|dotenv", all_content))
    if env_refs:
        base += 0.15
        findings.append(f"Environment-based configuration ({env_refs} references)")

    # Health / readiness endpoints
    if _has_pattern(all_content, r"/health|/ready|/liveness|/readiness|healthcheck"):
        base += 0.15
        findings.append("Health/readiness endpoints")

    # Docker / container hints
    docker_files = [p for p in files if "dockerfile" in p.lower() or "docker-compose" in p.lower()]
    if docker_files:
        base += 0.15
        findings.append("Dockerfile / docker-compose present")

    # Graceful shutdown
    if _has_pattern(all_content, r"signal\.signal|SIGTERM|SIGINT|graceful.*shutdown|lifespan"):
        base += 0.15
        findings.append("Graceful shutdown handling")

    # Stateless design (no global mutable state)
    global_state = len(re.findall(r"^[A-Z_]+\s*=\s*\{\}|^[A-Z_]+\s*=\s*\[\]", all_content, re.MULTILINE))
    if global_state == 0:
        base += 0.1
        findings.append("No obvious global mutable state")
    else:
        findings.append(f"{global_state} global mutable state reference(s)")

    # Connection pooling / resource management
    if _has_pattern(all_content, r"pool|async with|contextmanager|__aenter__"):
        base += 0.1
        findings.append("Resource management patterns")

    if not findings:
        findings.append("No cloud scalability patterns detected")

    return _score(base, "cloud_scalability", findings)


def _check_api_migration_cost(files: dict[str, str]) -> DimensionScore:
    """API versioning, backward compatibility, deprecation markers."""
    all_content = "\n".join(files.values())
    findings: list[str] = []

    base = 0.3  # Higher baseline — absence of bad patterns is good

    # API versioning
    if _has_pattern(all_content, r"/v\d+/|/api/v\d|version.*header|api_version"):
        base += 0.2
        findings.append("API versioning present")

    # Deprecation warnings
    if _has_pattern(all_content, r"@deprecated|DeprecationWarning|warnings\.warn"):
        base += 0.15
        findings.append("Deprecation warnings used")

    # Serialization schemas (Pydantic / marshmallow / attrs)
    if _has_pattern(all_content, r"BaseModel|Schema|@attr|TypedDict"):
        base += 0.15
        findings.append("Explicit serialization schemas (migration-friendly)")

    # Optional fields / backward compat defaults
    if _has_pattern(all_content, r"Optional\[|default=|Field\(.*default"):
        base += 0.1
        findings.append("Optional/default fields for backward compatibility")

    # Clear public API (__all__ exports)
    if _has_pattern(all_content, r"__all__\s*="):
        base += 0.1
        findings.append("Explicit __all__ exports")

    if not findings:
        findings.append("No API migration patterns detected")

    return _score(base, "api_migration_cost", findings)


def _check_test_surface(files: dict[str, str]) -> DimensionScore:
    """Amount and diversity of tests — unit, integration, edge-case coverage."""
    findings: list[str] = []

    test_files = {p: c for p, c in files.items() if "test" in p.lower()}
    test_funcs = 0
    for content in test_files.values():
        test_funcs += len(re.findall(r"def test_\w+|async def test_\w+", content))

    if not test_files:
        return _score(0.1, "test_surface", ["No test files found"])

    base = 0.3
    findings.append(f"{len(test_files)} test file(s), {test_funcs} test function(s)")

    # Ratio of test lines to source lines
    source_files = {p: c for p, c in files.items() if "test" not in p.lower() and p.endswith(".py")}
    source_lines = _total_lines(source_files) if source_files else 0
    test_lines = _total_lines(test_files)
    if source_lines:
        ratio = test_lines / source_lines
        if ratio >= 0.5:
            base += 0.2
            findings.append(f"Test/source line ratio: {ratio:.1f}")
        elif ratio >= 0.2:
            base += 0.1
            findings.append(f"Test/source line ratio: {ratio:.1f}")

    # Diversity: pytest fixtures, parametrize, mocking
    test_content = "\n".join(test_files.values())
    diverse = []
    if _has_pattern(test_content, r"@pytest\.fixture"):
        diverse.append("fixtures")
    if _has_pattern(test_content, r"@pytest\.mark\.parametrize"):
        diverse.append("parametrize")
    if _has_pattern(test_content, r"mock|patch|MagicMock|mocker"):
        diverse.append("mocking")
    if _has_pattern(test_content, r"assert.*raises|pytest\.raises"):
        diverse.append("error testing")

    if diverse:
        base += min(len(diverse) * 0.1, 0.3)
        findings.append(f"Test patterns: {', '.join(diverse)}")

    return _score(base, "test_surface", findings)


def _check_team_onboarding(files: dict[str, str]) -> DimensionScore:
    """README, setup instructions, consistent structure, clear entry points."""
    findings: list[str] = []

    base = 0.2

    # README / documentation files
    doc_files = [
        p for p in files
        if any(p.lower().endswith(e) for e in (".md", ".rst", ".txt"))
        and any(kw in p.lower() for kw in ("readme", "contributing", "docs", "guide"))
    ]
    if doc_files:
        base += 0.2
        findings.append(f"Documentation: {', '.join(doc_files)}")

    # Setup / configuration files
    setup_files = [
        p for p in files
        if any(kw in p.lower() for kw in (
            "pyproject.toml", "setup.py", "setup.cfg",
            "requirements.txt", "makefile", "justfile",
        ))
    ]
    if setup_files:
        base += 0.15
        findings.append(f"Build/setup files: {', '.join(setup_files)}")

    # Clear entry point
    has_main = any(
        p.endswith(("main.py", "app.py", "__main__.py", "cli.py"))
        for p in files
    )
    if has_main:
        base += 0.15
        findings.append("Clear main entry point")

    # Module docstrings count (already counted in maintainability, but relevant here too)
    pf = _py_files(files)
    trees = _parse_trees(pf)
    modules_with_docs = sum(
        1 for tree in trees.values()
        if tree and tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    )
    if modules_with_docs:
        base += 0.1
        findings.append(f"{modules_with_docs} file(s) with module-level docstrings")

    # Consistent naming (all snake_case files)
    py_names = [p.split("/")[-1].replace(".py", "") for p in pf]
    snake_names = [n for n in py_names if re.match(r"^[a-z_][a-z0-9_]*$", n)]
    if py_names and len(snake_names) == len(py_names):
        base += 0.1
        findings.append("Consistent snake_case file naming")

    if not findings:
        findings.append("No onboarding-friendly patterns detected")

    return _score(base, "team_onboarding", findings)


# ══════════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════════


def _evaluate_category(
    category: str,
    checkers: dict[str, Any],
    files: dict[str, str],
    threshold: float,
) -> ReadinessScore:
    """Run all checkers and aggregate into a ReadinessScore."""
    dimensions: list[DimensionScore] = []
    for _name, checker in checkers.items():
        dim_score = checker(files)
        dimensions.append(dim_score)

    if dimensions:
        overall = sum(d.score for d in dimensions) / len(dimensions)
    else:
        overall = 0.0

    return ReadinessScore(
        category=category,
        overall_score=overall,
        passed=overall >= threshold,
        dimensions=dimensions,
    )
