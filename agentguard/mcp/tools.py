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
    """Return the archetype config and files for the calling agent to derive
    and execute its own validation criteria.

    The agent reads the archetype's declared tech_stack, validation checks,
    lint rules, type strictness, expected structure, and self_challenge criteria,
    then determines what those mean for that specific stack and evaluates the files.
    No criteria are prescribed here — the agent generates them from the archetype config.
    """
    from agentguard.archetypes.base import Archetype

    try:
        arch = Archetype.load(archetype)
        config = {
            "tech_stack": {
                "language": arch.tech_stack.language,
                "framework": arch.tech_stack.framework,
                "database": arch.tech_stack.database,
                "testing": arch.tech_stack.testing,
                "linter": arch.tech_stack.linter,
                "type_checker": arch.tech_stack.type_checker,
            },
            "validation": {
                "checks": arch.validation.checks,
                "lint_rules": arch.validation.lint_rules,
                "type_strictness": arch.validation.type_strictness,
            },
            "structure": {
                "expected_dirs": arch.structure.get("expected_dirs", []),
                "expected_files": arch.structure.get("expected_files", []),
            },
            "self_challenge_criteria": arch.self_challenge.criteria,
        }
    except Exception:
        config = {
            "tech_stack": {"language": archetype},
            "validation": {"checks": ["syntax", "lint", "types", "imports", "structure"]},
            "structure": {},
            "self_challenge_criteria": [],
        }

    return json.dumps(
        {
            "tool": "validate",
            "description": (
                "Validation task for you (the calling agent). "
                "Using the archetype config below as your specification, derive the "
                "appropriate validation criteria for this specific tech stack, then "
                "evaluate the provided files against them. "
                "YOU decide what each check means for this language and framework — "
                "the archetype config is your source of truth."
            ),
            "instructions": [
                "Read the archetype config (tech_stack, validation, structure, self_challenge_criteria).",
                "For each entry in validation.checks, determine what that check means for "
                "this specific stack (e.g. 'lint' for a ruff+python project means different "
                "rules than for an eslint+typescript project — you decide based on the config).",
                "Evaluate the files against the criteria you derived.",
                "For structure checks, compare the file paths against expected_dirs and expected_files.",
                "For self_challenge_criteria, evaluate each criterion against the files.",
                "If any native tooling (linter, type checker, test runner) can be run locally, "
                "do so and include the raw output in your response.",
                "Return your results in the response_format below.",
            ],
            "archetype_config": config,
            "scoring": {
                "0": "Critical failure — blocking, must be fixed before shipping",
                "1": "Warning — issue present but project can run",
                "2": "Acceptable — minor imperfection, non-blocking",
                "3": "Clean — fully satisfies the check",
                "blocking_rule": "passed=false if any check with a blocking nature scores 0 or 1",
            },
            "response_format": {
                "passed": "boolean",
                "blocking_failures": "integer",
                "checks_results": [
                    {
                        "check": "string — from validation.checks",
                        "criteria_derived": "string — the criteria you determined for this stack",
                        "score": "integer 0-3",
                        "level": "critical_fail | warning | acceptable | clean",
                        "tool_output": "string | null — raw output if a native tool was run",
                        "findings": "string — specific file and line if relevant",
                        "fix_suggestion": "string | null",
                    }
                ],
                "criteria_results": [
                    {
                        "criterion": "string — exact text from self_challenge_criteria",
                        "passed": "boolean",
                        "explanation": "string",
                    }
                ],
                "overall_notes": "string",
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
