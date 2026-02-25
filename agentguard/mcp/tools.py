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
    llm: str = "",  # ignored — the calling agent uses its own LLM
) -> str:
    """Return structured generation instructions for the calling agent.

    Instead of calling an internal LLM, this returns the prompts and
    workflow the calling agent should follow to generate code itself using
    its own configured LLM.  The agent always does the thinking.

    Returns a JSON object with next-step instructions for the full pipeline:
    skeleton → contracts_and_wiring → logic → validate.
    """
    return json.dumps(
        {
            "tool": "generate",
            "description": (
                "Structured generation workflow for you (the calling agent) to execute "
                "using your own LLM. Do NOT delegate to any external model — you generate "
                "all code yourself by following these steps in order."
            ),
            "spec": spec,
            "archetype": archetype,
            "steps": [
                {
                    "step": 1,
                    "action": "Call `skeleton` with the spec and archetype to get the file tree.",
                },
                {
                    "step": 2,
                    "action": (
                        "Call `contracts_and_wiring` with the skeleton JSON to get "
                        "typed stubs and import wiring instructions per file."
                    ),
                },
                {
                    "step": 3,
                    "action": (
                        "For each non-trivial function, call `logic` to get implementation "
                        "instructions, then write the full function body yourself."
                    ),
                },
                {
                    "step": 4,
                    "action": "Call `validate` on all generated files to catch syntax/import issues.",
                },
                {
                    "step": 5,
                    "action": (
                        "Call `get_challenge_criteria` for the archetype, then self-review "
                        "your generated code against each criterion and fix any issues."
                    ),
                },
            ],
            "next_step": "Call `skeleton` now with spec and archetype to begin.",
        },
        indent=2,
    )


async def agentguard_validate(
    files: dict[str, str],
    archetype: str = "api_backend",
) -> str:
    """Return a structured validation prompt for the calling agent.

    The agent performs the actual review using its own LLM and knowledge of
    the language declared in the archetype.  No internal LLM or external tool
    is invoked here — the archetype's own configuration drives what to check.

    Returns a JSON object containing:
    - tech_stack: the language, framework, linter, and type_checker from the archetype
    - prerequisites: derived from the archetype's tech_stack — the agent determines
      what local tooling is needed based on those values
    - checks: the validation checks the archetype requires, with the archetype's
      own lint rules and type strictness settings
    - challenge_criteria: the archetype's self_challenge criteria
    - expected_structure: dirs and files the archetype expects
    - response_format: exact JSON shape the agent must return
    - files: the code to review
    """
    from agentguard.archetypes.base import Archetype

    try:
        arch = Archetype.load(archetype)
        tech_stack = {
            "language": arch.tech_stack.language,
            "framework": arch.tech_stack.framework,
            "database": arch.tech_stack.database,
            "testing": arch.tech_stack.testing,
            "linter": arch.tech_stack.linter,
            "type_checker": arch.tech_stack.type_checker,
        }
        checks = arch.validation.checks
        lint_rules = arch.validation.lint_rules
        type_strictness = arch.validation.type_strictness
        expected_dirs = arch.structure.get("expected_dirs", [])
        expected_files = arch.structure.get("expected_files", [])
        challenge_criteria = arch.self_challenge.criteria
    except Exception:
        tech_stack = {"language": archetype, "framework": "", "linter": "", "type_checker": ""}
        checks = ["syntax", "lint", "types", "imports", "structure"]
        lint_rules = "default"
        type_strictness = "basic"
        expected_dirs = []
        expected_files = []
        challenge_criteria = []

    return json.dumps(
        {
            "tool": "validate",
            "description": (
                "Validation task for you (the calling agent). "
                "Review the provided files against the archetype's declared checks "
                "using your own LLM knowledge of the language. "
                "YOU perform every check — do not delegate to another model."
            ),
            "archetype": archetype,
            "tech_stack": tech_stack,
            "prerequisites": {
                "note": (
                    "Based on the tech_stack above, determine what local tooling "
                    "is required to run this project (runtime, package manager, "
                    f"linter: {tech_stack.get('linter')}, "
                    f"type checker: {tech_stack.get('type_checker')}, "
                    f"test runner: {tech_stack.get('testing')}). "
                    "If those tools are available, run them and include the output. "
                    "If not, perform the equivalent checks yourself using your language knowledge "
                    "and note which tools were unavailable."
                ),
            },
            "checks_required": checks,
            "lint_rules": lint_rules,
            "type_strictness": type_strictness,
            "expected_structure": {
                "dirs": expected_dirs,
                "files": expected_files,
            },
            "challenge_criteria": challenge_criteria,
            "scoring": {
                "scale": {
                    "0": "Critical failure — blocking issue, must be fixed before shipping",
                    "1": "Warning — issue present but project can run",
                    "2": "Acceptable — minor imperfection, non-blocking",
                    "3": "Clean — fully satisfies the check",
                },
                "blocking_threshold": (
                    "A check is blocking when score <= 1. "
                    "Overall passed = no blocking check scored below 2."
                ),
            },
            "response_format": {
                "description": "Return this exact JSON structure after completing your review.",
                "schema": {
                    "passed": "boolean — true only if no blocking check scored below 2",
                    "blocking_failures": "integer — count of blocking checks that scored 0 or 1",
                    "checks_results": [
                        {
                            "check": "string — matches an entry in checks_required",
                            "score": "integer 0-3",
                            "level": "one of: critical_fail | warning | acceptable | clean",
                            "tool_output": "string | null — raw output if a native tool was run",
                            "findings": "string — what you found, be specific (file and line if relevant)",
                            "fix_suggestion": "string | null — concrete fix if score < 3, else null",
                        }
                    ],
                    "criteria_results": [
                        {
                            "criterion": "string — the exact criterion text from challenge_criteria",
                            "passed": "boolean",
                            "explanation": "string",
                        }
                    ],
                    "overall_notes": "string — summary of the most important findings",
                },
            },
            "files": files,
        },
        indent=2,
    )


async def agentguard_challenge(
    code: str,
    criteria: list[str] | None = None,
    llm: str = "",  # ignored — the calling agent uses its own LLM
) -> str:
    """Return a self-review prompt for the calling agent.

    Instead of calling an internal LLM, this packages the code and criteria
    into a structured review task that the calling agent executes itself using
    its own configured LLM.  The agent always does the thinking.

    Returns a JSON object with the code and criteria for the agent to review.
    """
    review_criteria = criteria or [
        "No hardcoded secrets, credentials, or environment-specific values",
        "All imports are used and resolvable",
        "Error handling present on all I/O and external calls",
        "Functions have type annotations",
        "No dead code or TODO stubs left in production paths",
        "Consistent naming conventions throughout",
    ]
    return json.dumps(
        {
            "tool": "challenge",
            "description": (
                "Self-review task for you (the calling agent). Review the provided code "
                "against each criterion using your own judgment. Do NOT delegate to any "
                "external model — you perform the review."
            ),
            "instructions": (
                "For each criterion below, assess the code and respond with: "
                "PASS or FAIL, a one-sentence explanation, and a suggested fix if FAIL. "
                "Then provide an overall verdict and a summary of required changes."
            ),
            "criteria": review_criteria,
            "code_to_review": code,
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
    model: str = "",
    category: str | None = None,
    budget: float = 10.0,
) -> str:
    """Run a comparative benchmark for an archetype.

    Generates code WITH and WITHOUT AgentGuard across 5 complexity levels,
    evaluating enterprise and operational readiness. Returns the full
    benchmark report as JSON.

    Note: ``model`` is optional — if omitted, falls back to the model
    string configured on the pipeline / server.  When calling from an
    agent that already has an LLM, prefer the agent-native
    ``benchmark`` + ``benchmark_evaluate`` tools instead.
    """
    from agentguard.benchmark.catalog import get_default_specs
    from agentguard.benchmark.runner import BenchmarkRunner
    from agentguard.benchmark.types import BenchmarkConfig

    cat = category or archetype
    specs = get_default_specs(cat)
    config = BenchmarkConfig(specs=specs, model=model, budget_ceiling_usd=budget)
    runner = BenchmarkRunner(archetype=archetype, config=config, llm=model or None)
    report = await runner.run()
    return report.to_json()
