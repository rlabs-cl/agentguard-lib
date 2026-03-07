"""MCP-native tool definitions — designed for LLM agents (no API key needed).

These tools expose AgentGuard's structured workflow to the calling LLM agent.
Instead of making their own LLM calls, they return rendered prompts, criteria,
and structure that the agent uses to guide its own generation.

This is the correct paradigm for MCP: the tool provides the *structure*,
the agent (who IS an LLM) does the *thinking*.

v1.1 Changes:
- ``skeleton`` now emits file tiers (config/foundation/feature) and an
  interface-summary block so sub-agents don't need to re-read shared files.
- ``contracts_and_wiring`` merges L2+L3 into a single call (saves ~15K tokens).
- Legacy ``contracts`` / ``wiring`` still work but emit a deprecation hint.
- ``digest`` generates a compact project-level summary for self-challenge.
- ``get_challenge_criteria`` now embeds criteria from the archetype config
  (cached — no need to call separately after ``get_archetype``).

v1.2 Changes:
- ``debug`` returns a structured debugging protocol (data_sources →
  hypothesis_protocol → fix_protocol → escalation_criteria) loaded from the
  target archetype's ``debug_config`` block.  Builtin archetypes:
  ``debug_backend`` (Python/FastAPI) and ``debug_frontend`` (React/TypeScript).
  Falls back to a sensible default protocol when the archetype has no
  ``debug_config``.
- ``migrate`` returns a structured migration plan including a source-file
  digest, concern checklist, incompatibility signals, and step_order — all
  loaded from the target archetype's ``migration_config`` block.
"""

from __future__ import annotations

import functools as _functools
import json
import logging
import time as _time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from agentguard.mcp.usage_tracker import get_tracker as _get_tracker

logger = logging.getLogger(__name__)

_AsyncFn = TypeVar("_AsyncFn", bound=Callable[..., Awaitable[Any]])


def _track_mcp_tool(fn: _AsyncFn) -> _AsyncFn:
    """Decorator: records one ``mcp_tool`` analytics event per tool call.

    Extracts the ``archetype`` / ``target_archetype`` kwarg as the slug.
    ``benchmark_evaluate`` is a terminal event — its buffer is force-flushed
    immediately after the event is recorded.
    """
    _tool_name = fn.__name__.replace("agentguard_", "")
    _is_terminal = _tool_name == "benchmark_evaluate"

    @_functools.wraps(fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        _archetype: str | None = kwargs.get("archetype") or kwargs.get("target_archetype")
        _t0 = _time.perf_counter()
        _ok = True
        try:
            return await fn(*args, **kwargs)
        except Exception:
            _ok = False
            raise
        finally:
            _tr = _get_tracker()
            if _archetype and _archetype in _marketplace_cache:
                _tr.track(_tool_name, _archetype, _ok, int((_time.perf_counter() - _t0) * 1000))
            if _is_terminal:
                _tr.force_flush()

    return _wrapper  # type: ignore[return-value]


# ── helpers ────────────────────────────────────────────────────────

# In-process cache for marketplace archetypes (avoids repeated API calls per session)
_marketplace_cache: dict[str, Any] = {}


def _load_arch(archetype: str) -> Any:
    """Load an archetype by ID.

    Resolution order:
    1. Built-in registry  (no auth required)
    2. In-process cache   (already fetched this session)
    3. Marketplace API    (requires AGENTGUARD_API_KEY + ownership/purchase)
    """
    from agentguard.archetypes.base import Archetype

    try:
        return Archetype.load(archetype)
    except KeyError:
        pass

    if archetype in _marketplace_cache:
        return _marketplace_cache[archetype]

    return _fetch_from_marketplace(archetype)


def _fetch_from_marketplace(archetype_id: str) -> Any:
    """Download a marketplace archetype via the platform API and cache it for this session.

    Requires:
    - ``AGENTGUARD_API_KEY`` env var set to a valid ``ag_`` key
    - The calling user must be the archetype author, owner, or have purchased it

    Raises:
        KeyError:        Archetype not found in marketplace or API key not set
        PermissionError: Invalid API key (401) or not licensed/purchased (403)
        RuntimeError:    Unexpected API error
    """
    import json as _json
    import os
    import urllib.error
    import urllib.request

    from agentguard.archetypes.registry import get_archetype_registry
    from agentguard.archetypes.schema import TrustLevel

    api_key = os.environ.get("AGENTGUARD_API_KEY", "")
    if not api_key:
        raise KeyError(
            f"Archetype '{archetype_id}' is not a built-in. "
            "Set the AGENTGUARD_API_KEY environment variable to load marketplace archetypes."
        )

    base_url = os.environ.get(
        "AGENTGUARD_API_URL", "https://api.agentguard.dev"
    ).rstrip("/")
    url = f"{base_url}/api/marketplace/archetypes/{archetype_id}/download"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = _json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise PermissionError(
                "Invalid or expired API key — cannot load marketplace archetype. "
                "Check the AGENTGUARD_API_KEY environment variable."
            ) from exc
        if exc.code == 403:
            raise PermissionError(
                f"Access denied for archetype '{archetype_id}'. "
                "You must be the author or have purchased it in the AgentGuard marketplace."
            ) from exc
        if exc.code == 404:
            raise KeyError(
                f"Archetype '{archetype_id}' not found in the marketplace."
            ) from exc
        raise RuntimeError(
            f"Marketplace API error {exc.code} while loading archetype "
            f"'{archetype_id}': {exc.reason}"
        ) from exc

    yaml_content: str = data["yaml_content"]
    content_hash: str | None = data.get("content_hash")
    trust_level_str: str = data.get("trust_level", "community")

    registry = get_archetype_registry()
    trust_level = (
        TrustLevel(trust_level_str)
        if trust_level_str in TrustLevel.__members__
        else TrustLevel.community
    )

    if content_hash:
        entry = registry.register_remote(
            archetype_id, yaml_content, content_hash, trust_level=trust_level
        )
    else:
        # No hash provided — register without integrity verification
        entry = registry.register_validated(yaml_content, trust_level=trust_level)

    _marketplace_cache[archetype_id] = entry.archetype
    logger.info(
        "Loaded marketplace archetype '%s' (trust=%s)", archetype_id, trust_level_str
    )
    return entry.archetype


def _get_prompt(template_id: str) -> Any:
    from agentguard.prompts.registry import get_prompt_registry
    return get_prompt_registry().get(template_id)


def _maturity_infrastructure(arch: Any) -> list[dict[str, str]]:
    """Return mandatory infrastructure files for the archetype's maturity."""
    maturity = getattr(arch, "maturity", "production")
    infra: list[str] = getattr(arch, "infrastructure_files", [])
    if not infra:
        return []

    # Tags embedded in comments: "# production+", "# enterprise"
    production_plus: list[str] = []
    enterprise_only: list[str] = []
    for f in infra:
        clean = f.split("#")[0].strip()
        tag = f.split("#")[1].strip().lower() if "#" in f else "production+"
        if "enterprise" in tag:
            enterprise_only.append(clean)
        else:
            production_plus.append(clean)

    result: list[dict[str, str]] = []
    if maturity in ("production", "enterprise"):
        for p in production_plus:
            result.append({
                "path": p,
                "tier": "infrastructure",
                "purpose": f"Mandatory infrastructure ({p.split('/')[-1]})",
            })
    if maturity == "enterprise":
        for p in enterprise_only:
            result.append({
                "path": p,
                "tier": "infrastructure",
                "purpose": f"Enterprise infrastructure ({p.split('/')[-1]})",
            })
    return result


# ── 1. skeleton (enhanced) ─────────────────────────────────────────

@_track_mcp_tool
async def agentguard_skeleton(
    spec: str,
    archetype: str = "api_backend",
    maturity: str | None = None,
) -> str:
    """Get the rendered L1 skeleton prompt for a project specification.

    Returns the system + user prompt that guides file-tree generation.
    The calling LLM should follow these instructions to produce the skeleton.

    Enhanced in v1.1:
    - Emits ``file_tiers`` guidance (config/foundation/feature).
    - Emits ``interface_summary_hint`` so sub-agents get a compact context
      instead of re-reading full files.
    - Emits ``infrastructure_files`` that MUST be included based on maturity.
    """
    arch = _load_arch(archetype)
    effective_maturity = maturity or getattr(arch, "maturity", "production")

    template = _get_prompt("skeleton")
    messages = template.render(
        spec=spec,
        archetype_name=arch.name,
        language=arch.tech_stack.language,
        framework=arch.tech_stack.framework,
        expected_structure=arch.get_expected_structure_text(),
    )

    infra = _maturity_infrastructure(arch)

    return json.dumps(
        {
            "level": "L1 — Skeleton",
            "description": "Generate the file tree with one-line responsibilities for each file.",
            "archetype": arch.id,
            "maturity": effective_maturity,
            "tech_stack": {
                "language": arch.tech_stack.language,
                "framework": arch.tech_stack.framework,
            },
            "instructions": [m.content for m in messages],
            "expected_output_format": (
                'JSON array: [{"path": "...", "purpose": "...", "tier": "config|foundation|feature"}]'
            ),
            "file_tiers": {
                "config": "Build config, package.json, tsconfig — generate directly, skip L2/L3.",
                "foundation": "Shared types, utils, constants — needs contracts only (no wiring).",
                "feature": "Components, pages, routes — full contracts + wiring pipeline.",
                "infrastructure": "Mandatory files from archetype (error boundaries, logger, etc.).",
            },
            "infrastructure_files": infra,
            "interface_summary_hint": (
                "After generating foundation files (types, context, utils), build a compact "
                "summary of their exports (type names, function signatures, context API). "
                "Pass this summary to sub-agents instead of having them re-read full files. "
                "Example: 'Service{id,name,duration,price}, useApp()→{services,addService,...}'"
            ),
            "next_step": "Once you have the skeleton, call `contracts_and_wiring` with it.",
        },
        indent=2,
    )


# ── 2. contracts_and_wiring (merged L2+L3) ─────────────────────────

@_track_mcp_tool
async def agentguard_contracts_and_wiring(
    spec: str,
    skeleton_json: str,
    archetype: str = "api_backend",
) -> str:
    """Get merged L2+L3 prompts: typed stubs WITH import wiring in one pass.

    This replaces the separate ``contracts`` → ``wiring`` flow, saving ~15K
    tokens by eliminating the intermediate JSON shuttle.

    Only processes files with tier ``foundation`` or ``feature``.
    Config-tier files should be generated directly without this tool.
    """
    arch = _load_arch(archetype)
    contracts_template = _get_prompt("contracts")
    wiring_template = _get_prompt("wiring")
    skeleton_files = json.loads(skeleton_json)

    file_prompts = []
    for entry in skeleton_files:
        path = entry["path"]
        purpose = entry["purpose"]
        tier = entry.get("tier", "feature")

        # Config files skip the pipeline entirely
        if tier == "config":
            continue

        # Build the contracts prompt
        contract_messages = contracts_template.render(
            spec=spec,
            file_path=path,
            file_purpose=purpose,
            language=arch.tech_stack.language,
            skeleton_files=skeleton_files,
            reference_patterns="",
        )

        prompt_entry: dict[str, Any] = {
            "file": path,
            "purpose": purpose,
            "tier": tier,
            "contracts_instructions": [m.content for m in contract_messages],
        }

        # Foundation/infrastructure files get contracts only
        if tier in ("foundation", "infrastructure"):
            prompt_entry["wiring_instructions"] = None
            prompt_entry["note"] = (
                "Foundation file — generate stubs with exports. No wiring needed."
            )
        else:
            # Feature files also get wiring instructions
            wiring_messages = wiring_template.render(
                file_path=path,
                language=arch.tech_stack.language,
                file_contracts="(generate contracts first, then wire imports)",
                other_files=[
                    {"path": f["path"], "contracts": f"(see {f['path']} contracts)"}
                    for f in skeleton_files
                    if f["path"] != path and f.get("tier") != "config"
                ],
            )
            prompt_entry["wiring_instructions"] = [m.content for m in wiring_messages]
            prompt_entry["note"] = (
                "Feature file — generate stubs, then wire imports from foundation files."
            )

        file_prompts.append(prompt_entry)

    return json.dumps(
        {
            "level": "L2+L3 — Contracts & Wiring (merged)",
            "description": (
                "For each file, generate typed function/class stubs with imports "
                "already wired. Foundation files get stubs only. Feature files get "
                "stubs + import wiring in one pass."
            ),
            "file_count": len(file_prompts),
            "files": file_prompts,
            "anti_patterns": [
                "Do NOT use `null!` or `as any` — use safe type narrowing.",
                "Do NOT use default exports — use named exports for grep-ability.",
                "Context hooks MUST throw on null: `if (!ctx) throw new Error(...)`.",
                "Data layer functions SHOULD be async-compatible.",
            ],
            "next_step": (
                "Once all stubs are done, call `logic` for each function "
                "that needs implementation."
            ),
        },
        indent=2,
    )


# ── 3. Legacy contracts (backward-compatible) ──────────────────────

@_track_mcp_tool
async def agentguard_contracts(
    spec: str,
    skeleton_json: str,
    archetype: str = "api_backend",
) -> str:
    """Get the rendered L2 contracts prompt for each file in the skeleton.

    .. deprecated:: 1.1
        Use ``contracts_and_wiring`` instead for a merged L2+L3 flow.
    """
    arch = _load_arch(archetype)
    template = _get_prompt("contracts")
    skeleton_files = json.loads(skeleton_json)

    file_prompts = []
    for entry in skeleton_files:
        path = entry["path"]
        purpose = entry["purpose"]
        messages = template.render(
            spec=spec,
            file_path=path,
            file_purpose=purpose,
            language=arch.tech_stack.language,
            skeleton_files=skeleton_files,
            reference_patterns="",
        )
        file_prompts.append({
            "file": path,
            "purpose": purpose,
            "instructions": [m.content for m in messages],
        })

    return json.dumps(
        {
            "level": "L2 — Contracts",
            "description": (
                "For each file, generate typed function/class stubs with "
                "docstrings. Every function body should be `raise NotImplementedError`."
            ),
            "file_count": len(file_prompts),
            "files": file_prompts,
            "deprecation": (
                "Consider using `contracts_and_wiring` for a merged L2+L3 "
                "flow (saves ~15K tokens)."
            ),
            "next_step": "Once all stubs are done, call the `wiring` tool.",
        },
        indent=2,
    )


# ── 4. Legacy wiring (backward-compatible) ─────────────────────────

@_track_mcp_tool
async def agentguard_wiring(
    contracts_json: str,
    archetype: str = "api_backend",
) -> str:
    """Get the rendered L3 wiring prompt for connecting files.

    .. deprecated:: 1.1
        Use ``contracts_and_wiring`` instead for a merged L2+L3 flow.
    """
    arch = _load_arch(archetype)
    template = _get_prompt("wiring")
    contracts = json.loads(contracts_json)

    file_prompts = []
    for file_path, code in contracts.items():
        other_files = [
            {"path": p, "contracts": c}
            for p, c in contracts.items()
            if p != file_path
        ]
        messages = template.render(
            file_path=file_path,
            language=arch.tech_stack.language,
            file_contracts=code,
            other_files=other_files,
        )
        file_prompts.append({
            "file": file_path,
            "instructions": [m.content for m in messages],
        })

    return json.dumps(
        {
            "level": "L3 — Wiring",
            "description": (
                "Wire imports and call chains between files. Keep all function "
                "signatures intact, add correct imports, no business logic yet."
            ),
            "file_count": len(file_prompts),
            "files": file_prompts,
            "deprecation": (
                "Consider using `contracts_and_wiring` for a merged L2+L3 "
                "flow (saves ~15K tokens)."
            ),
            "next_step": "Once wiring is done, call the `logic` tool for each function.",
        },
        indent=2,
    )


# ── 5. logic (unchanged) ───────────────────────────────────────────

@_track_mcp_tool
async def agentguard_logic(
    file_path: str,
    file_code: str,
    function_name: str,
    archetype: str = "api_backend",
) -> str:
    """Get the rendered L4 logic prompt for implementing a single function.

    Returns the prompt that guides the implementation of one function body,
    replacing the ``raise NotImplementedError`` with real logic.
    """
    arch = _load_arch(archetype)
    template = _get_prompt("logic")

    messages = template.render(
        file_path=file_path,
        language=arch.tech_stack.language,
        file_code=file_code,
        function_name=function_name,
        function_signature="(see code above)",
        function_docstring="(see code above)",
        dependencies=[],
        reference_patterns="",
    )

    return json.dumps(
        {
            "level": "L4 — Logic",
            "description": f"Implement the body of `{function_name}` in `{file_path}`.",
            "instructions": [m.content for m in messages],
            "next_step": (
                "Repeat for each NotImplementedError function. "
                "Then call `validate` to check the result."
            ),
        },
        indent=2,
    )


# ── 6. get_challenge_criteria (enhanced — criteria cached) ──────────

@_track_mcp_tool
async def agentguard_get_challenge_criteria(
    archetype: str = "api_backend",
    extra_criteria: list[str] | None = None,
) -> str:
    """Get the self-challenge criteria and review prompt for an archetype.

    Returns the criteria list and the rendered challenge prompt template
    so the calling LLM can perform the self-review itself.

    v1.1: Criteria are embedded in the archetype YAML — this tool simply
    retrieves them (cached). No extra MCP call needed after ``get_archetype``.
    """
    arch = _load_arch(archetype)

    criteria = list(arch.self_challenge.criteria)
    if extra_criteria:
        criteria.extend(extra_criteria)

    criteria_lines = []
    for i, c in enumerate(criteria, 1):
        criteria_lines.append(
            f"CRITERION {i}: {c}\nRESULT: [PASS/FAIL]\nEXPLANATION: ..."
        )

    return json.dumps(
        {
            "level": "Self-Challenge",
            "description": (
                "Review your generated code against these criteria. "
                "Be strict — if in doubt, FAIL the criterion."
            ),
            "archetype": arch.id,
            "maturity": getattr(arch, "maturity", "production"),
            "criteria": criteria,
            "criteria_count": len(criteria),
            "review_format": {
                "per_criterion": (
                    "CRITERION N: <text>\\nRESULT: PASS or FAIL\\nEXPLANATION: ..."
                ),
                "grounding": (
                    "Check: Did you invent any API/function/class not in the spec? "
                    "List violations or write NONE."
                ),
                "assumptions": (
                    "List every assumption not explicitly in the spec, "
                    "or write NONE."
                ),
            },
            "instructions": (
                "For each criterion, evaluate PASS or FAIL with explanation. "
                "Then check grounding (invented APIs) and list assumptions. "
                "If any criterion fails, rework the code before finalizing."
            ),
            "tip": (
                "Use the `digest` tool to build a compact project summary "
                "instead of re-reading every file. Saves ~8K tokens."
            ),
        },
        indent=2,
    )


# ── 7. digest (NEW — compact project summary for review) ───────────

@_track_mcp_tool
async def agentguard_digest(
    files_json: str,
    archetype: str = "api_backend",
) -> str:
    """Generate a compact project digest for efficient self-challenge review.

    Instead of reading every file in full (~2K lines), the digest extracts
    only exports, function signatures, import graphs, and key patterns
    (~200 lines). An LLM can evaluate all challenge criteria from this
    digest without re-reading full source.

    Args:
        files_json: JSON object mapping file paths to their full source code.
        archetype: Archetype ID for context.

    Returns:
        Structured digest with per-file summaries and cross-cutting analysis.
    """
    arch = _load_arch(archetype)
    files: dict[str, str] = json.loads(files_json)

    file_digests: list[dict[str, Any]] = []
    all_exports: dict[str, list[str]] = {}
    total_lines = 0

    for path, code in sorted(files.items()):
        lines = code.split("\n")
        total_lines += len(lines)
        line_count = len(lines)

        # Extract imports
        imports = [
            ln.strip() for ln in lines
            if ln.strip().startswith(("import ", "from "))
        ]

        # Extract exports / public API
        exports: list[str] = []
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith((
                "export ", "def ", "class ", "async def ",
            )):
                sig = stripped.split("{")[0].split(":")[0].strip() \
                    if "{" in stripped or ":" in stripped else stripped
                exports.append(sig)
            elif "export default" in stripped:
                exports.append(stripped)
        all_exports[path] = exports

        # Detect patterns
        patterns: list[str] = []
        code_lower = code.lower()
        if "aria-" in code or "role=" in code:
            patterns.append("a11y: ARIA attributes found")
        if "dangerouslysetinnerhtml" in code_lower:
            patterns.append("⚠ XSS risk: dangerouslySetInnerHTML")
        if "as any" in code or "null!" in code:
            patterns.append("⚠ unsafe type assertion")
        if "useeffect" in code_lower:
            patterns.append("side-effects: useEffect")
        if "localstorage" in code_lower:
            patterns.append("persistence: localStorage")
        if "try" in code and "catch" in code:
            patterns.append("error-handling: try/catch")
        if "errorboundary" in code_lower:
            patterns.append("error-boundary present")

        file_digests.append({
            "path": path,
            "lines": line_count,
            "imports_count": len(imports),
            "exports": exports[:15],  # Cap to avoid bloat
            "patterns": patterns,
            "over_150_lines": line_count > 150,
        })

    # Cross-cutting analysis
    cross_cutting = {
        "total_files": len(files),
        "total_lines": total_lines,
        "avg_lines_per_file": round(total_lines / max(len(files), 1)),
        "files_over_150_lines": [
            d["path"] for d in file_digests if d["over_150_lines"]
        ],
        "has_error_boundary": any(
            "error-boundary" in " ".join(d["patterns"]) for d in file_digests
        ),
        "has_a11y": any(
            "a11y" in " ".join(d["patterns"]) for d in file_digests
        ),
        "has_xss_risk": any(
            "XSS" in " ".join(d["patterns"]) for d in file_digests
        ),
        "has_unsafe_types": any(
            "unsafe" in " ".join(d["patterns"]) for d in file_digests
        ),
    }

    return json.dumps(
        {
            "level": "Project Digest",
            "description": (
                "Compact summary for self-challenge review. "
                f"Condensed {total_lines} source lines into a structured digest."
            ),
            "archetype": arch.id,
            "files": file_digests,
            "cross_cutting": cross_cutting,
            "usage": (
                "Pass the challenge criteria and this digest to your self-review. "
                "Only read full files if a specific criterion needs deeper inspection."
            ),
        },
        indent=2,
    )


# ── 8. debug (agent-native) ────────────────────────────────────────

@_track_mcp_tool
async def agentguard_debug(
    symptom: str,
    archetype: str = "debug_backend",
    sources: dict[str, str] | None = None,
) -> str:
    """Return a structured debugging protocol for the calling agent to execute.

    Loads the archetype's ``debug_config`` (data_sources, hypothesis_protocol,
    fix_protocol, escalation_criteria) and packages it with the reported symptom
    and any evidence the caller has already collected.  The agent follows the
    returned protocol — consulting sources, forming hypotheses, selecting the
    most likely root cause, proposing a minimal fix, or escalating when the
    criteria are met.

    No API key needed — YOU (the agent) do the debugging.
    """
    try:
        arch = _load_arch(archetype)
        dbg = arch.debug_config
        debug_config: dict[str, Any] = {
            "data_sources": dbg.data_sources,
            "hypothesis_protocol": dbg.hypothesis_protocol,
            "fix_protocol": dbg.fix_protocol,
            "escalation_criteria": dbg.escalation_criteria,
        }
    except Exception:
        debug_config = {
            "data_sources": [
                "application logs",
                "stack traces / error messages",
                "environment variables and config",
                "recent git diff",
                "relevant test output",
            ],
            "hypothesis_protocol": [
                "Restate the symptom in one sentence.",
                "Identify the system boundary where the failure manifests.",
                "Trace the call path from entry point to failure site.",
                "List ≤5 hypotheses ordered by likelihood.",
                "For each hypothesis, state the evidence that would confirm or refute it.",
                "Select the highest-confidence hypothesis and proceed to fix_protocol.",
            ],
            "fix_protocol": [
                "Make the minimal change that addresses the root cause.",
                "Verify that existing tests still pass.",
                "Add a regression test that would have caught this bug.",
                "Describe the reproduction steps and the fix in a short comment.",
            ],
            "escalation_criteria": [
                "Root cause is indeterminate after exhausting all data_sources.",
                "Fix requires a schema migration or breaking API change.",
                "Issue originates in a third-party dependency.",
                "Failure is non-deterministic or environment-specific.",
            ],
        }

    return json.dumps(
        {
            "tool": "debug",
            "description": (
                "Structured debugging protocol for you (the calling agent) to execute. "
                "Follow the steps below to diagnose the reported symptom and produce a fix "
                "or an escalation report.  Do NOT delegate to any external model — you do "
                "the debugging yourself."
            ),
            "symptom": symptom,
            "provided_sources": sources or {},
            "debug_config": debug_config,
            "instructions": [
                "Review all keys in `provided_sources` for initial evidence.",
                "Work through `debug_config.data_sources` in order — if a source is not "
                "available, note it as unavailable and continue.",
                "Follow `debug_config.hypothesis_protocol` step by step.",
                "Before proposing a fix, check each item in `debug_config.escalation_criteria`. "
                "If any criterion is met, STOP and return an escalation report instead.",
                "If no escalation is triggered, follow `debug_config.fix_protocol` to produce "
                "a minimal, tested fix.",
            ],
            "response_format": {
                "outcome": "fix | escalation",
                "root_cause": "string — one sentence",
                "confidence": "high | medium | low",
                "hypotheses_considered": [
                    {
                        "hypothesis": "string",
                        "evidence": "string",
                        "verdict": "confirmed | refuted | inconclusive",
                    }
                ],
                "fix": {
                    "description": "string | null",
                    "files_changed": [{"path": "string", "change": "string"}],
                    "regression_test": "string | null",
                },
                "escalation": {
                    "reason": "string | null",
                    "matched_criterion": "string | null",
                    "recommended_next_step": "string | null",
                },
                "data_sources_consulted": ["string"],
                "data_sources_unavailable": ["string"],
            },
        },
        indent=2,
    )


# ── 9. migrate (agent-native) ──────────────────────────────────────

@_track_mcp_tool
async def agentguard_migrate(
    source_files: dict[str, str],
    target_archetype: str = "api_backend",
    spec: str = "",
) -> str:
    """Return a structured migration plan for the calling agent to execute.

    Loads the target archetype's ``migration_config`` (risk_areas,
    concern_protocol, incompatibility_signals, step_order) and produces a
    migration analysis: a digest of what the source files contain, a concern
    checklist the agent must answer, incompatibility flags, and an ordered
    migration plan.  The agent follows the protocol and either produces a
    migration or reports blockers.

    No API key needed — YOU (the agent) do the migration work.
    """
    try:
        arch = _load_arch(target_archetype)
        mig = arch.migration_config
        migration_config: dict[str, Any] = {
            "risk_areas": mig.risk_areas,
            "concern_protocol": mig.concern_protocol,
            "incompatibility_signals": mig.incompatibility_signals,
            "step_order": mig.step_order,
        }
        target_stack: dict[str, Any] = {
            "language": arch.tech_stack.language,
            "framework": arch.tech_stack.framework,
            "testing": arch.tech_stack.testing,
            "linter": arch.tech_stack.linter,
            "type_checker": arch.tech_stack.type_checker,
        }
    except Exception:
        migration_config = {
            "risk_areas": [
                "data access patterns and ORM usage",
                "authentication and authorization logic",
                "external API integrations",
                "configuration and environment variable surface",
                "runtime dependencies and version constraints",
            ],
            "concern_protocol": [
                "Does the source project have passing tests that document current behaviour?",
                "Are there any undocumented side effects (cron jobs, background tasks, webhooks)?",
                "Are there database schema migrations that need to run before the new code?",
                "What is the rollback plan if the new version breaks in production?",
            ],
            "incompatibility_signals": [
                "Global mutable state that spans request boundaries",
                "Synchronous blocking I/O in an async target framework",
                "Direct SQL strings in non-ORM source mixed with ORM target",
                "Hardcoded host/port constants that conflict with target config model",
            ],
            "step_order": [
                "Audit source — build digest, identify risk areas",
                "Answer concern_protocol questions before proceeding",
                "Flag and resolve incompatibility_signals",
                "Scaffold target structure using the `skeleton` tool",
                "Port logic module by module, smallest first",
                "Re-run source tests against migrated code",
                "Update configuration and environment variables",
                "Write migration-specific tests for changed behaviour",
            ],
        }
        target_stack = {"language": target_archetype}

    # Produce a compact digest of the source files for the agent
    source_summary: list[dict[str, Any]] = []
    for path, content in (source_files or {}).items():
        lines = content.splitlines()
        source_summary.append(
            {
                "path": path,
                "lines": len(lines),
                "preview": "\n".join(lines[:8]) + ("\n..." if len(lines) > 8 else ""),
            }
        )

    return json.dumps(
        {
            "tool": "migrate",
            "description": (
                "Structured migration protocol for you (the calling agent) to execute. "
                "Analyse the source files, answer the concern_protocol checklist, flag any "
                "incompatibility_signals, then follow the step_order to produce the migrated "
                "codebase targeting the archetype below.  Do NOT delegate to any external "
                "model — you do the migration work yourself."
            ),
            "target_archetype": target_archetype,
            "target_stack": target_stack,
            "spec": spec,
            "source_files_digest": source_summary,
            "migration_config": migration_config,
            "instructions": [
                "Read source_files_digest to understand what exists.",
                "For each item in migration_config.incompatibility_signals, check whether it "
                "appears in the source files.  Note any that are present as blockers.",
                "Work through migration_config.concern_protocol — answer each question. "
                "If any answer reveals a blocker, STOP and return a blocker report.",
                "If no blockers, follow migration_config.step_order in sequence.",
                "For the scaffold step, call the `skeleton` tool with the spec and target_archetype.",
                "Port logic incrementally, file by file, confirming tests pass at each step.",
                "Return the migrated files and a change summary in the response_format below.",
            ],
            "response_format": {
                "outcome": "completed | blocked",
                "concern_answers": [
                    {"question": "string", "answer": "string", "is_blocker": "boolean"}
                ],
                "incompatibilities_found": [
                    {"signal": "string", "location": "string", "resolution": "string | null"}
                ],
                "migrated_files": {"<path>": "<content>"},
                "change_summary": [
                    {
                        "file": "string",
                        "change_type": "created | modified | deleted | unchanged",
                        "notes": "string",
                    }
                ],
                "blocker_report": {
                    "reason": "string | null",
                    "unresolved_concerns": ["string"],
                    "recommended_next_step": "string | null",
                },
            },
        },
        indent=2,
    )


# ── 10. benchmark (agent-native, two-step) ────────────────────────

@_track_mcp_tool
async def agentguard_benchmark(
    archetype: str = "api_backend",
    category: str | None = None,
) -> str:
    """Get benchmark specs for comparative evaluation (no API key needed).

    Returns 5 development specifications (one per complexity level) that the
    calling agent should implement **twice** each:

    1. **Control** — generate code for the spec directly (raw LLM, no tools).
    2. **Treatment** — generate code using the AgentGuard workflow
       (skeleton → contracts_and_wiring → logic → validate).

    Once both sets are generated, call ``benchmark_evaluate`` with the
    results to get scored readiness reports.

    Args:
        archetype: Project archetype name or ID.
        category: Catalog category for spec lookup (defaults to archetype name).

    Returns:
        JSON with benchmark specs and instructions for the agent.
    """
    from agentguard.benchmark.catalog import get_default_specs

    cat = category or archetype
    specs = get_default_specs(cat)

    spec_list = [
        {
            "complexity": s.complexity.value,
            "spec": s.spec,
            "category": s.category,
        }
        for s in specs
    ]

    return json.dumps(
        {
            "tool": "agentguard_benchmark",
            "description": (
                "Comparative benchmark: generate code WITH and WITHOUT "
                "AgentGuard tools, then evaluate both with benchmark_evaluate."
            ),
            "archetype": archetype,
            "total_specs": len(spec_list),
            "specs": spec_list,
            "instructions": {
                "important": (
                    "YOU (the calling agent) must generate ALL code yourself using "
                    "your own LLM capabilities. Do NOT call the `generate` or "
                    "`challenge` MCP tools — those require a separate API key and "
                    "are only for headless pipelines with no agent. "
                    "AgentGuard MCP tools only provide STRUCTURE (prompts, skeletons, "
                    "criteria) — the thinking and code generation is always done by you."
                ),
                "control": (
                    "For each spec, write production code yourself directly from "
                    "the spec text alone — no MCP tools at all. Return the "
                    "files as a dict mapping filepath → content."
                ),
                "treatment": (
                    "For each spec, use the AgentGuard structural tools yourself: "
                    "call skeleton → contracts_and_wiring → logic → validate, "
                    "then write the code following the instructions they return. "
                    "Return the files as a dict mapping filepath → content."
                ),
                "evaluate": (
                    "Once you have control and treatment file dicts for all "
                    "specs, call `benchmark_evaluate` with the results."
                ),
            },
            "next_step": (
                "Start with the first spec. Write CONTROL code yourself (no tools), "
                "then write TREATMENT code yourself guided by AgentGuard structural "
                "tools. Repeat for each spec, then call `benchmark_evaluate`."
            ),
        },
        indent=2,
    )


@_track_mcp_tool
async def agentguard_benchmark_evaluate(
    archetype: str = "api_backend",
    results_json: str = "[]",
    archetype_yaml: str = "",
    environment: str = "",
    llm_temperature: float | None = None,
    llm_seed: int | None = None,
    spec_source: str = "catalog",
    run_by: str = "",
    notes: str = "",
) -> str:
    """Evaluate benchmark results — score control vs treatment code.

    No API key needed — scoring is pure static analysis (AST-based).

    Accepts the generated code from both control and treatment runs across
    all complexity levels, evaluates enterprise and operational readiness,
    and returns a full scored report with an environment metadata envelope.

    If ``archetype_yaml`` is provided:
    - Step 0: validates the YAML schema and returns errors immediately if invalid.
    - Extracts ``scoring_weights`` so dimensions irrelevant to the archetype
      type (weight=0.0) are rendered as N/A in the report.
    - If ``AGENTGUARD_API_KEY`` is set, auto-uploads the report to the platform
      and attaches it to the archetype draft (creating it if it doesn't exist yet).

    Args:
        archetype: The archetype id used for the benchmark (catalog name or slug).
        results_json: JSON array of objects, each with:
            - ``complexity``: "trivial" | "low" | "medium" | "high" | "enterprise"
            - ``spec``: The original spec text.
            - ``control_files``: dict of filepath → content (raw LLM output).
            - ``treatment_files``: dict of filepath → content (AgentGuard output).
        archetype_yaml: Raw YAML string of a custom archetype being tested locally.
            When provided, schema validation runs first (STEP 0). The YAML's
            ``scoring_weights`` and ``id`` are used for fitness-aware reporting
            and auto-upload slug derivation.
        environment: Calling environment tag, e.g. "vscode-copilot", "cursor",
            "custom-agent", "ci". Leave blank if unknown.
        llm_temperature: Temperature used for LLM generation, if known.
        llm_seed: Random seed used, if any.
        spec_source: Origin of the specs — "catalog", "custom", or "production".
        run_by: Identifier for who ran the benchmark (email, username, etc.).
        notes: Free-text notes about this run.

    Returns:
        JSON benchmark report with per-dimension scores, overall verdict,
        and a populated environment metadata envelope.
    """
    import os
    import platform
    import sys

    from agentguard._version import __version__ as agentguard_version
    from agentguard.benchmark.evaluator import (
        evaluate_enterprise,
        evaluate_operational,
    )
    from agentguard.benchmark.report import (
        format_report_compact,
        format_report_markdown,
    )
    from agentguard.benchmark.types import (
        BenchmarkReport,
        Complexity,
        ComplexityRun,
        RunResult,
    )

    # ── STEP 0: Validate archetype YAML if provided ───────────────
    scoring_weights: dict[str, float] = {}
    archetype_slug: str = archetype
    archetype_yaml_for_upload: str | None = None

    if archetype_yaml:
        try:
            from agentguard.archetypes.schema import validate_archetype_yaml

            validated_schema = validate_archetype_yaml(archetype_yaml)
            scoring_weights = dict(validated_schema.scoring_weights or {})
            archetype_slug = validated_schema.id
            archetype_yaml_for_upload = archetype_yaml
        except ValueError as e:
            return json.dumps(
                {
                    "tool": "agentguard_benchmark_evaluate",
                    "error": "archetype_yaml failed schema validation (STEP 0)",
                    "errors": [{"field": "yaml", "message": str(e)}],
                    "benchmark_result": None,
                },
                indent=2,
            )
        except Exception as e:  # pydantic ValidationError or unexpected
            errors = []
            if hasattr(e, "errors"):
                for err in e.errors():  # pydantic ValidationError
                    loc = " → ".join(str(x) for x in err["loc"])
                    errors.append({"field": loc, "message": err["msg"]})
            else:
                errors = [{"field": "yaml", "message": str(e)}]
            return json.dumps(
                {
                    "tool": "agentguard_benchmark_evaluate",
                    "error": "archetype_yaml failed schema validation (STEP 0)",
                    "errors": errors,
                    "benchmark_result": None,
                },
                indent=2,
            )

    # ── STEP 1–3: Score ───────────────────────────────────────────
    results = json.loads(results_json)
    runs: list[ComplexityRun] = []

    for entry in results:
        complexity = Complexity(entry["complexity"])
        spec = entry["spec"]
        control_files: dict[str, str] = entry.get("control_files", {})
        treatment_files: dict[str, str] = entry.get("treatment_files", {})

        ctrl_ent = evaluate_enterprise(control_files)
        ctrl_ops = evaluate_operational(control_files)
        ctrl_lines = sum(c.count("\n") + 1 for c in control_files.values()) if control_files else 0
        control = RunResult(
            enterprise=ctrl_ent,
            operational=ctrl_ops,
            files_generated=len(control_files),
            total_lines=ctrl_lines,
        )

        treat_ent = evaluate_enterprise(treatment_files)
        treat_ops = evaluate_operational(treatment_files)
        treat_lines = sum(c.count("\n") + 1 for c in treatment_files.values()) if treatment_files else 0
        treatment = RunResult(
            enterprise=treat_ent,
            operational=treat_ops,
            files_generated=len(treatment_files),
            total_lines=treat_lines,
        )

        runs.append(ComplexityRun(
            complexity=complexity,
            spec=spec,
            control=control,
            treatment=treatment,
        ))

    report = BenchmarkReport(
        archetype_id=archetype_slug,
        model="agent-native",
        runs=runs,
    )

    # Re-compute aggregates applying fitness weights from the archetype YAML
    # (__post_init__ already ran without weights; this corrects enterprise_avg /
    # operational_avg / overall_passed to exclude N/A dimensions like accessibility).
    if scoring_weights:
        report.compute_aggregates(scoring_weights=scoring_weights)

    # ── STEP 4: Auto-upload if API key is configured ──────────────
    upload_status: dict[str, Any] = {"attempted": False}
    api_key = os.environ.get("AGENTGUARD_API_KEY", "")
    api_url = os.environ.get("AGENTGUARD_API_URL", "https://api.agentguard.dev")

    if api_key:
        upload_status["attempted"] = True
        upload_body: dict[str, Any] = {
            **report.to_dict(),
            "archetype_id": archetype_slug,
        }
        if archetype_yaml_for_upload:
            upload_body["archetype_yaml"] = archetype_yaml_for_upload
        # Extract README from the most complex treatment run and surface it as
        # long_description so the marketplace detail page can render it.
        readme_content: str | None = None
        for entry in reversed(results):  # most complex last in catalog order
            tf: dict[str, str] = entry.get("treatment_files", {})
            for path, content in tf.items():
                if path.lower().endswith("readme.md") or path.lower() == "readme.md":
                    readme_content = content
                    break
            if readme_content:
                break
        if readme_content:
            upload_body["long_description"] = readme_content
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.put(
                    f"{api_url}/api/marketplace/archetypes/{archetype_slug}/benchmark",
                    json=upload_body,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            if resp.status_code in (200, 201):
                upload_status["success"] = True
                upload_status["platform_url"] = (
                    f"{api_url.replace('api.', 'app.') if api_url.startswith('https://api.') else api_url}"
                    f"/dashboard/archetypes"
                )
            else:
                upload_status["success"] = False
                upload_status["status_code"] = resp.status_code
                try:
                    upload_status["detail"] = resp.json().get("detail", resp.text[:200])
                except Exception:
                    upload_status["detail"] = resp.text[:200]
        except Exception as exc:
            upload_status["success"] = False
            upload_status["detail"] = str(exc)

    created_at_str = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

    return json.dumps(
        {
            "tool": "agentguard_benchmark_evaluate",
            "description": (
                f"Scored {len(runs)} complexity levels. "
                f"{'PASSED' if report.overall_passed else 'FAILED'}."
            ),
            "overall_passed": report.overall_passed,
            "summary": {
                "enterprise_avg": round(report.enterprise_avg, 3),
                "operational_avg": round(report.operational_avg, 3),
                "improvement_avg": round(report.improvement_avg, 3),
            },
            "compact": format_report_compact(report),
            "markdown": format_report_markdown(report, weights=scoring_weights or None),
            "report": report.to_dict(),
            "upload": upload_status,
            "meta": {
                "agentguard_version": agentguard_version,
                "python_version": sys.version.split()[0],
                "platform": platform.system(),
                "created_at": created_at_str,
                "environment": environment or None,
                "llm_temperature": llm_temperature,
                "llm_seed": llm_seed,
                "system_prompt_injected": None,
                "spec_source": spec_source or "catalog",
                "run_by": run_by or None,
                "notes": notes or None,
            },
        },
        indent=2,
    )

