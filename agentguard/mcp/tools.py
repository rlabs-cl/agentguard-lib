"""MCP tool definitions for AgentGuard.

Each function is registered as an MCP tool on the ``FastMCP`` server.
Utility tools — validation, archetype listing, traces. Pure computation, no LLM.
"""

from __future__ import annotations

import json
import logging

from agentguard._http import make_request
from agentguard.mcp.agent_tools import _confidentiality_directive_for

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
                    "The archetype was not found in the built-in registry. "
                    "Try using the archetype name directly in a pipeline tool "
                    "(skeleton, contracts_and_wiring, etc.) — marketplace archetypes "
                    "are auto-downloaded when used. If the archetype doesn't exist "
                    "at all, use `list_archetypes` or `search_marketplace` to find "
                    "the correct slug. Both hyphens and underscores are accepted."
                ),
            },
            indent=2,
        )

    return json.dumps(
        {
            "_confidentiality": _confidentiality_directive_for(arch.confidentiality_policy),
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
            req = make_request(url)
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

    result = {
        "usage_hint": (
            "To USE any archetype, pass its 'id' as the `archetype` parameter to "
            "skeleton, contracts_and_wiring, validate, or any pipeline tool. "
            "Marketplace archetypes are auto-downloaded — no install command needed. "
            "source='local' means ready to use; source='marketplace' requires "
            "AGENTGUARD_API_KEY and the user must own/have purchased it."
        ),
        "archetypes": archetypes,
    }
    return json.dumps(result, indent=2)


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
        req = make_request(url)
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


async def agentguard_my_archetypes() -> str:
    """List archetypes the current user can actually use, with ownership/access details.

    For each archetype returns:
    - id, name, description
    - access: "built-in" | "authored" | "purchased" | "available_remote"
    - location: "local" (loaded in memory) | "cached" (on disk) | "remote" (needs download)
    - ready: true if the archetype can be used immediately without any download

    Requires AGENTGUARD_API_KEY for marketplace access info.
    Built-in archetypes are always listed regardless of API key.

    The response includes a markdown table for easy reading.
    """
    import os
    import urllib.error
    import urllib.request
    from pathlib import Path

    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    results: list[dict] = []
    seen_ids: set[str] = set()

    # ── 1. Built-in archetypes (always available) ──
    for arch_id in registry.list_available():
        entry = registry.get_entry(arch_id)
        seen_ids.add(arch_id)
        seen_ids.add(arch_id.replace("_", "-"))
        is_official = entry.trust_level.value == "official"
        results.append({
            "id": arch_id,
            "name": entry.archetype.name,
            "description": entry.archetype.description,
            "access": "built-in" if is_official else "authored/purchased",
            "location": "local",
            "ready": True,
        })

    # ── 2. Check disk cache for .agx/.yaml not yet in registry ──
    user_dir = Path.home() / ".agentguard" / "archetypes"
    if user_dir.exists():
        for f in sorted(user_dir.iterdir()):
            if f.suffix not in (".yaml", ".agx"):
                continue
            slug = f.stem
            underscore = slug.replace("-", "_")
            if slug in seen_ids or underscore in seen_ids:
                continue
            seen_ids.add(slug)
            seen_ids.add(underscore)
            results.append({
                "id": slug,
                "name": slug.replace("-", " ").replace("_", " ").title(),
                "description": f"Cached on disk ({f.suffix})",
                "access": "purchased" if f.suffix == ".agx" else "authored/purchased",
                "location": "cached",
                "ready": True,
            })

    # ── 3. Check marketplace for user's purchased/authored archetypes ──
    api_key = os.environ.get("AGENTGUARD_API_KEY", "")
    marketplace_note: str | None = None

    if api_key:
        base_url = os.environ.get(
            "AGENTGUARD_API_URL",
            os.environ.get("AGENTGUARD_PLATFORM_URL", "https://api.agentguard.rlabs.cl"),
        ).rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # ── Strategy: try the dedicated /my-archetypes endpoint first (1 call).
        # Falls back to the N-call approach if the endpoint doesn't exist yet
        # (backwards compat with older platform versions).
        used_fast_path = False

        try:
            my_url = f"{base_url}/api/engine/my-archetypes"
            my_req = make_request(my_url, headers=headers)
            with urllib.request.urlopen(my_req, timeout=15) as resp:
                my_data = json.loads(resp.read())

            # Fast path succeeded — parse the response
            for item in my_data.get("archetypes", []):
                slug = item.get("slug", "")
                underscore = slug.replace("-", "_")
                if slug in seen_ids or underscore in seen_ids:
                    continue
                seen_ids.add(slug)
                seen_ids.add(underscore)
                results.append({
                    "id": slug,
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "access": item.get("access", "purchased"),
                    "location": "remote",
                    "ready": False,
                    "category": item.get("category", ""),
                })
            used_fast_path = True

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                pass  # Endpoint not deployed yet — fall back to legacy
            elif exc.code in (401, 403):
                # Distinguish genuine auth failure from CDN / bot-protection
                # challenges (e.g. Cloudflare 1010). The latter usually ships
                # an HTML body with a CF error code — treat those as transient
                # and leave the real reason in the note so the user can act.
                body_snippet = ""
                try:
                    body_snippet = exc.read(2048).decode("utf-8", errors="replace")
                except Exception:
                    pass
                if "cloudflare" in body_snippet.lower() or "error code: 101" in body_snippet.lower():
                    marketplace_note = (
                        "Marketplace blocked by an upstream CDN rule — check "
                        "the request User-Agent or whitelist this client."
                    )
                else:
                    marketplace_note = (
                        "API key is invalid or expired. Check AGENTGUARD_API_KEY in MCP config."
                    )
                used_fast_path = True  # Don't retry with legacy
            else:
                logger.warning("my-archetypes endpoint error: %s", exc)
        except (urllib.error.URLError, OSError, ValueError):
            pass  # Network error — fall back to legacy

        # ── Legacy fallback: check marketplace + individual license calls ──
        if not used_fast_path:
            user_id: str | None = None
            try:
                validate_url = f"{base_url}/api/engine/validate"
                req = make_request(validate_url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    user_data = json.loads(resp.read())
                    user_id = user_data.get("user_id")
            except Exception:
                pass

            try:
                authored_items: list[dict] = []
                needs_license_check: list[dict] = []

                page = 1
                while page <= 5:
                    url = f"{base_url}/api/marketplace/archetypes?page={page}&limit=50"
                    req = make_request(url)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = json.loads(resp.read())

                    items = data.get("items", [])
                    if not items:
                        break

                    for item in items:
                        slug = item.get("slug", "")
                        underscore = slug.replace("-", "_")
                        if slug in seen_ids or underscore in seen_ids:
                            continue
                        author_id = item.get("author_id", "")
                        if user_id and author_id == user_id:
                            authored_items.append(item)
                        else:
                            needs_license_check.append(item)

                    total = data.get("total", 0)
                    if page * 50 >= total:
                        break
                    page += 1

                for item in authored_items:
                    slug = item.get("slug", "")
                    seen_ids.add(slug)
                    seen_ids.add(slug.replace("-", "_"))
                    results.append({
                        "id": slug,
                        "name": item.get("name", ""),
                        "description": item.get("description", ""),
                        "access": "authored",
                        "location": "remote",
                        "ready": False,
                        "category": item.get("category", ""),
                    })

                # Cap at 15 license checks to keep response time reasonable
                checked = 0
                for item in needs_license_check:
                    if checked >= 15:
                        marketplace_note = (
                            f"License checked for {checked} of {len(needs_license_check)} "
                            "non-authored marketplace archetypes. Some may not appear. "
                            "Use search_marketplace to check specific ones."
                        )
                        break
                    slug = item.get("slug", "")
                    try:
                        lic_url = f"{base_url}/api/engine/license/{slug}"
                        lic_req = make_request(lic_url, headers=headers)
                        with urllib.request.urlopen(lic_req, timeout=5) as lic_resp:
                            lic_data = json.loads(lic_resp.read())
                        checked += 1
                        if not lic_data.get("licensed", False):
                            continue
                        seen_ids.add(slug)
                        seen_ids.add(slug.replace("-", "_"))
                        results.append({
                            "id": slug,
                            "name": item.get("name", ""),
                            "description": item.get("description", ""),
                            "access": "purchased" if lic_data.get("reason") != "free" else "free",
                            "location": "remote",
                            "ready": False,
                            "category": item.get("category", ""),
                        })
                    except Exception:
                        checked += 1

            except (urllib.error.URLError, OSError, ValueError):
                pass  # Non-fatal
    else:
        marketplace_note = (
            "AGENTGUARD_API_KEY not set — only showing built-in and locally cached archetypes. "
            "Set the API key in MCP config to see marketplace archetypes you've authored or purchased."
        )

    # ── 4. Build markdown table ──
    table_lines = [
        "| Archetype | Access | Location | Ready | Description |",
        "|-----------|--------|----------|-------|-------------|",
    ]
    for r in results:
        ready_icon = "Yes" if r["ready"] else "Download needed"
        desc = (r["description"] or "")[:60]
        table_lines.append(
            f"| `{r['id']}` | {r['access']} | {r['location']} | {ready_icon} | {desc} |"
        )

    response: dict[str, Any] = {
        "summary": f"{len(results)} archetypes accessible to this user",
        "table": "\n".join(table_lines),
        "archetypes": results,
        "usage_hint": (
            "To use any archetype listed here, pass its 'id' as the `archetype` "
            "parameter to any pipeline tool (skeleton, validate, etc.). "
            "Archetypes with location='remote' will be auto-downloaded on first use."
        ),
    }
    if marketplace_note:
        response["note"] = marketplace_note

    return json.dumps(response, indent=2)


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
