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
    """List all available project archetypes (local + marketplace).

    Returns local (built-in and installed) archetypes first, then appends
    marketplace archetypes that are not yet installed locally.  Marketplace
    items are marked with ``"source": "marketplace"`` and include their slug
    for installation or direct use.
    """
    import os
    import urllib.error
    import urllib.request

    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    archetypes = []
    local_ids: set[str] = set()

    for arch_id in registry.list_available():
        entry = registry.get_entry(arch_id)
        local_ids.add(arch_id)
        # Also track the hyphenated form so we can deduplicate later
        local_ids.add(arch_id.replace("_", "-"))
        archetypes.append(
            {
                "id": entry.archetype.id,
                "name": entry.archetype.name,
                "description": entry.archetype.description,
                "trust_level": entry.trust_level.value,
                "content_hash": entry.content_hash,
                "source": "local",
            }
        )

    # ── Append marketplace archetypes not installed locally ──
    base_url = os.environ.get(
        "AGENTGUARD_API_URL",
        os.environ.get("AGENTGUARD_PLATFORM_URL", "https://api.agentguard.rlabs.cl"),
    ).rstrip("/")

    try:
        # Paginate through all marketplace results (API uses 1-based page param)
        page = 1
        page_size = 20
        max_pages = 5  # safety limit
        while page <= max_pages:
            url = f"{base_url}/api/marketplace/archetypes?page={page}&limit={page_size}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                slug = item.get("slug", "")
                underscore_id = slug.replace("-", "_")
                # Skip if already available locally
                if slug in local_ids or underscore_id in local_ids:
                    continue
                archetypes.append(
                    {
                        "id": slug,
                        "name": item.get("name", ""),
                        "description": item.get("description", ""),
                        "trust_level": item.get("trust_level", "community"),
                        "content_hash": item.get("content_hash", ""),
                        "source": "marketplace",
                        "category": item.get("category", ""),
                    }
                )

            total = data.get("total", 0)
            if page * page_size >= total:
                break
            page += 1

    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("Could not fetch marketplace archetypes: %s", exc)
        # Non-fatal — we still return local archetypes

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


async def agentguard_search_marketplace(
    query: str = "",
    category: str = "",
) -> str:
    """Search the AgentGuard marketplace for available archetypes.

    Returns published archetypes from the online marketplace, including those
    not yet installed locally.  Useful for discovering new archetypes.

    Parameters
    ----------
    query : str, optional
        Free-text search (name, description, tags).
    category : str, optional
        Filter by category (e.g. ``engineering``, ``product-design``,
        ``technical-specs``, ``marketing-gtm``, ``data-analytics``,
        ``process-operations``, ``strategy-management``, ``academic-writing``).
    """
    import os
    import urllib.error
    import urllib.parse
    import urllib.request

    base_url = os.environ.get(
        "AGENTGUARD_API_URL",
        os.environ.get("AGENTGUARD_PLATFORM_URL", "https://api.agentguard.rlabs.cl"),
    ).rstrip("/")

    params: dict[str, str] = {}
    if query:
        params["q"] = query
    if category:
        params["category"] = category

    qs = urllib.parse.urlencode(params) if params else ""
    url = f"{base_url}/api/marketplace/archetypes"
    if qs:
        url = f"{url}?{qs}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        return json.dumps(
            {"error": f"Could not reach marketplace: {exc}", "items": []},
            indent=2,
        )

    # Build a compact response for the LLM
    items = data.get("items", [])
    results = []
    for item in items:
        results.append(
            {
                "slug": item.get("slug", ""),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "category": item.get("category", ""),
                "tags": item.get("tags", []),
                "price_cents": item.get("price_cents", 0),
                "currency": item.get("currency", "clp"),
                "downloads": item.get("downloads", 0),
                "rating_avg": item.get("rating_avg", 0),
                "trust_level": item.get("trust_level", "community"),
            }
        )

    return json.dumps(
        {"total": len(results), "items": results},
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
