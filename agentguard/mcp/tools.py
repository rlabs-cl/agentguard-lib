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


async def agentguard_search_marketplace(
    query: str | None = None,
    category: str | None = None,
    sort: str = "popular",
    page: int = 1,
    page_size: int = 20,
) -> str:
    """Search and browse published marketplace archetypes.

    Requires AGENTGUARD_API_KEY (or ~/.agentguard/config.yaml) to be configured.
    Returns items with slug, name, description, price, tags and licensed status.
    """
    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import load_config

    config = load_config()
    if not config.api_key:
        return json.dumps(
            {
                "error": "AGENTGUARD_API_KEY not configured.",
                "hint": (
                    "Set AGENTGUARD_API_KEY as an env var or add api_key to "
                    "~/.agentguard/config.yaml"
                ),
            },
            indent=2,
        )

    client = PlatformClient(config)
    try:
        data = await client.search_marketplace(
            query=query,
            category=category,
            sort=sort,
            page=page,
            page_size=page_size,
        )
        items = data.get("items", [])
        result = {
            "total": data.get("total", len(items)),
            "page": data.get("page", page),
            "page_size": data.get("page_size", page_size),
            "items": [
                {
                    "slug": item.get("slug"),
                    "name": item.get("name"),
                    "description": item.get("description", ""),
                    "category": item.get("category", ""),
                    "price_cents": item.get("price_cents"),
                    "currency": item.get("currency", "clp"),
                    "tags": item.get("tags", []),
                    "version": item.get("version", ""),
                    "downloads": item.get("downloads", 0),
                }
                for item in items
            ],
        }
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)
    finally:
        await client.close()


async def agentguard_install_archetype(slug: str) -> str:
    """Download and install a marketplace archetype to ~/.agentguard/archetypes/.

    Verifies license, downloads the YAML with integrity check, saves it locally,
    and reloads the registry so the archetype is immediately available.
    Requires AGENTGUARD_API_KEY (or ~/.agentguard/config.yaml) to be configured.
    """
    from pathlib import Path

    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import load_config

    config = load_config()
    if not config.api_key:
        return json.dumps(
            {
                "error": "AGENTGUARD_API_KEY not configured.",
                "hint": (
                    "Set AGENTGUARD_API_KEY as an env var or add api_key to "
                    "~/.agentguard/config.yaml"
                ),
            },
            indent=2,
        )

    client = PlatformClient(config)
    try:
        # 0. Auto-fetch encryption_salt if missing (needed for decryption)
        if not config.encryption_salt:
            try:
                info = await client.validate_api_key()
                salt = info.get("encryption_salt")
                if salt:
                    config.encryption_salt = salt
                    from agentguard.platform.config import save_config
                    save_config(config)
            except Exception:
                pass

        # 1. Check license
        license_info = await client.check_license(slug)
        if not license_info.get("licensed"):
            return json.dumps(
                {
                    "error": f"Not licensed for archetype '{slug}'.",
                    "reason": license_info.get("reason", ""),
                    "hint": f"Purchase '{slug}' at https://agentguard.rlabs.cl",
                },
                indent=2,
            )

        # 2. Download (includes integrity hash check)
        data = await client.download_archetype(slug)
        yaml_content: str = data["yaml_content"]
        content_hash: str = data.get("content_hash", "")
        name: str = data.get("name", slug)
        version: str = data.get("version", "")

        # 3. Save to ~/.agentguard/archetypes/{slug}.yaml
        dest_dir = Path.home() / ".agentguard" / "archetypes"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{slug}.yaml"
        dest.write_text(yaml_content, encoding="utf-8")

        # 4. Reload registry so it's available immediately
        from agentguard.archetypes.registry import get_archetype_registry
        registry = get_archetype_registry()
        reloaded = registry.reload_user_archetypes()

        return json.dumps(
            {
                "status": "installed",
                "slug": slug,
                "name": name,
                "version": version,
                "content_hash": content_hash[:16] + "…" if content_hash else "",
                "path": str(dest),
                "archetypes_now_available": registry.list_available(),
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc), "slug": slug}, indent=2)
    finally:
        await client.close()


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
