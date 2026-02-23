"""MCP tool definitions for AgentGuard.

Each function is registered as an MCP tool on the ``FastMCP`` server.
The functions themselves are thin wrappers that call into the
``Pipeline``, ``Validator``, and ``SelfChallenger`` classes.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


async def agentguard_generate(
    spec: str,
    archetype: str = "api_backend",
    llm: str = "anthropic/claude-sonnet-4-20250514",
) -> str:
    """Generate production-quality code from a spec using top-down generation,
    structural validation, and self-challenge.

    Returns a JSON object with generated files and trace summary.
    """
    from agentguard.pipeline import Pipeline

    pipe = Pipeline(archetype=archetype, llm=llm)
    result = await pipe.generate(spec)

    files = result.files
    cost = float(result.total_cost.total_cost)
    return json.dumps(
        {
            "files": files,
            "file_count": len(files),
            "total_cost_usd": f"${cost:.4f}",
        },
        indent=2,
    )


async def agentguard_validate(
    files: dict[str, str],
    archetype: str = "api_backend",
) -> str:
    """Run structural validation on code: syntax, lint, types, imports, structure.

    Returns a JSON pass/fail report with details.
    """
    from agentguard.archetypes.base import Archetype
    from agentguard.validation.validator import Validator

    arch = Archetype.load(archetype)
    validator = Validator(archetype=arch)
    report = validator.check(files)

    return json.dumps(
        {
            "passed": report.passed,
            "checks": [
                {
                    "check": c.check,
                    "passed": c.passed,
                    "details": c.details,
                }
                for c in report.checks
            ],
            "errors": [
                {
                    "check": e.check,
                    "file": e.file_path,
                    "line": e.line,
                    "message": e.message,
                }
                for e in report.errors
            ],
            "auto_fixes": len(report.auto_fixed),
        },
        indent=2,
    )


async def agentguard_challenge(
    code: str,
    criteria: list[str] | None = None,
    llm: str = "anthropic/claude-sonnet-4-20250514",
) -> str:
    """LLM-based self-review: checks code against quality criteria.

    Returns issues found and suggested fixes.
    """
    from agentguard.challenge.challenger import SelfChallenger
    from agentguard.llm.factory import create_llm_provider

    provider = create_llm_provider(llm)
    challenger = SelfChallenger(llm=provider)

    result = await challenger.challenge(
        output=code,
        criteria=criteria or [],
        task_description="Code review via MCP",
    )

    return json.dumps(
        {
            "passed": result.passed,
            "criteria_results": [
                {
                    "criterion": c.criterion,
                    "passed": c.passed,
                    "explanation": c.explanation,
                }
                for c in result.criteria_results
            ],
            "assumptions": result.assumptions,
            "grounding_violations": result.grounding_violations,
            "feedback": result.feedback,
        },
        indent=2,
    )


async def agentguard_list_archetypes() -> str:
    """List all available project archetypes with their descriptions."""
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    archetypes = []
    for arch_id in registry.list_available():
        entry = registry.get_entry(arch_id)
        archetypes.append(
            {
                "id": entry.archetype.id,
                "name": entry.archetype.name,
                "description": entry.archetype.description,
                "trust_level": entry.trust_level.value,
                "content_hash": entry.content_hash,
            }
        )
    return json.dumps(archetypes, indent=2)


async def agentguard_get_archetype(name: str) -> str:
    """Get detailed configuration for a specific archetype.

    Includes tech stack, validation rules, and challenge criteria.
    """
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    entry = registry.get_entry(name)
    arch = entry.archetype

    return json.dumps(
        {
            "id": arch.id,
            "name": arch.name,
            "description": arch.description,
            "version": arch.version,
            "maturity": getattr(arch, "maturity", "production"),
            "trust_level": entry.trust_level.value,
            "content_hash": entry.content_hash,
            "tech_stack": {
                "language": arch.tech_stack.language,
                "framework": arch.tech_stack.framework,
                "database": arch.tech_stack.database,
                "testing": arch.tech_stack.testing,
            },
            "pipeline_levels": arch.pipeline.levels,
            "validation_checks": arch.validation.checks,
            "challenge_criteria": arch.self_challenge.criteria,
            "infrastructure_files": getattr(arch, "infrastructure_files", []),
        },
        indent=2,
    )


async def agentguard_trace_summary(trace_id: str | None = None) -> str:
    """Get a summary of a generation trace: LLM calls, cost, validation results.

    If trace_id is omitted, returns info about the last trace (if available).
    """
    return json.dumps(
        {
            "note": "Trace lookup requires a trace store. "
            "Use the HTTP server with --trace-store for persistent traces.",
            "trace_id": trace_id,
        },
        indent=2,
    )


async def agentguard_benchmark(
    archetype: str = "api_backend",
    model: str = "anthropic/claude-sonnet-4-20250514",
    category: str | None = None,
    budget: float = 10.0,
) -> str:
    """Run a comparative benchmark for an archetype.

    Generates code WITH and WITHOUT AgentGuard across 5 complexity levels,
    evaluating enterprise and operational readiness. Returns the full
    benchmark report as JSON.
    """
    from agentguard.benchmark.catalog import get_default_specs
    from agentguard.benchmark.runner import BenchmarkRunner
    from agentguard.benchmark.types import BenchmarkConfig

    cat = category or archetype
    specs = get_default_specs(cat)
    config = BenchmarkConfig(model=model, specs=specs, budget_ceiling_usd=budget)
    runner = BenchmarkRunner(archetype=archetype, config=config)
    report = await runner.run()
    return report.to_json()
