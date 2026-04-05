"""MCP tool definitions for AgentGuard.

Each function is registered as an MCP tool on the ``FastMCP`` server.
Utility tools — validation, archetype listing, traces. Pure computation, no LLM.
"""

from __future__ import annotations

import json
import logging

from agentguard.mcp.agent_tools import _CONFIDENTIALITY_DIRECTIVE

logger = logging.getLogger(__name__)


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
    except KeyError:
        available = Archetype.list_available()
        return json.dumps(
            {
                "error": f"Archetype '{archetype}' not found.",
                "available": available,
                "hint": (
                    "Install a marketplace archetype with: agentguard install <slug>. "
                    "If the MCP server was already running when you installed it, "
                    "call the reload_archetypes tool first."
                ),
            },
            indent=2,
        )

    return json.dumps(
        {
            "_confidentiality": _CONFIDENTIALITY_DIRECTIVE,
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


async def agentguard_reload_archetypes() -> str:
    """Reload user-installed archetypes from ~/.agentguard/archetypes/.

    Call this after running ``agentguard install <slug>`` in a terminal so that
    the MCP server picks up the newly installed archetype without restarting.
    Built-in archetypes are never removed.
    """
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    reloaded = registry.reload_user_archetypes()
    all_ids = registry.list_available()
    return json.dumps(
        {
            "status": "ok",
            "user_archetypes_reloaded": reloaded,
            "all_available": all_ids,
        },
        indent=2,
    )


async def agentguard_trace_summary(trace_id: str | None = None) -> str:
    """Get a summary of a generation trace: LLM calls, cost, validation results.

    PREREQUISITE: Traces are only stored when the AgentGuard HTTP server is
    started with the ``--trace-store`` flag (e.g. ``agentguard serve --trace-store``).
    Without that flag, no traces are persisted and this tool always returns an empty result.

    If trace_id is omitted, returns info about the last trace (if available).
    """
    return json.dumps(
        {
            "note": (
                "Trace lookup requires the HTTP server to be running with "
                "--trace-store enabled. Start the server with: "
                "  agentguard serve --trace-store\n"
                "Then re-run your workflow and call trace_summary again."
            ),
            "trace_id": trace_id,
        },
        indent=2,
    )
