"""AgentGuard MCP server — exposes the engine as MCP tools.

Supports stdio (local AI tools) and SSE (network) transports.
Uses the ``mcp`` library's ``FastMCP`` high-level API.

Tool categories:

1. **Agent-native tools** (no API key needed) — return structured prompts and
   criteria so the calling LLM agent does the thinking itself.  This is the
   correct paradigm for MCP: the tool provides *structure*, the agent provides
   *intelligence*.

2. **Pipeline tools** (require LLM API key) — run the full AgentGuard pipeline
   internally.  Useful for thin/non-LLM clients calling via the HTTP API.

3. **Utility tools** — validation, archetype listing, traces.  Pure computation,
   no LLM needed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _create_mcp_server() -> Any:
    """Build and configure the FastMCP server instance."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "AgentGuard",
        instructions=(
            "Quality-assured LLM code generation engine. "
            "Use the agent-native tools (skeleton→contracts_and_wiring→logic→ "
            "get_challenge_criteria+digest) to get structured prompts that "
            "guide YOUR own code generation — no API key needed. "
            "Use validate to mechanically check code you produce. "
            "The generate/challenge tools run a full internal LLM pipeline "
            "and require a separate API key — prefer the agent-native tools."
        ),
    )

    # ── Agent-native tools (no API key — the agent IS the LLM) ────

    from agentguard.mcp.agent_tools import (
        agentguard_benchmark,
        agentguard_benchmark_evaluate,
        agentguard_contracts,
        agentguard_contracts_and_wiring,
        agentguard_debug,
        agentguard_digest,
        agentguard_get_challenge_criteria,
        agentguard_logic,
        agentguard_migrate,
        agentguard_skeleton,
        agentguard_wiring,
    )

    @mcp.tool()
    async def skeleton(
        spec: str,
        archetype: str = "api_backend",
        maturity: str | None = None,
    ) -> str:
        """Get the L1 skeleton prompt: file tree with responsibilities.
        Returns structured instructions for generating the project skeleton.
        Now includes file tiers and infrastructure guidance.
        No API key needed — YOU (the agent) do the generation."""
        return await agentguard_skeleton(
            spec=spec, archetype=archetype, maturity=maturity,
        )

    @mcp.tool()
    async def contracts(
        spec: str,
        skeleton_json: str,
        archetype: str = "api_backend",
    ) -> str:
        """Get L2 contract prompts: typed function/class stubs for each file.
        Pass the L1 skeleton JSON array. Returns per-file instructions.
        No API key needed — YOU (the agent) generate the stubs."""
        return await agentguard_contracts(
            spec=spec, skeleton_json=skeleton_json, archetype=archetype,
        )

    @mcp.tool()
    async def wiring(
        contracts_json: str,
        archetype: str = "api_backend",
    ) -> str:
        """Get L3 wiring prompts: import and call-chain connections.
        Pass the L2 contracts as JSON {path: code}. Returns per-file instructions.
        No API key needed — YOU (the agent) wire the imports."""
        return await agentguard_wiring(
            contracts_json=contracts_json, archetype=archetype,
        )

    @mcp.tool()
    async def logic(
        file_path: str,
        file_code: str,
        function_name: str,
        archetype: str = "api_backend",
    ) -> str:
        """Get L4 logic prompt: implement one function body.
        Returns instructions for replacing NotImplementedError with real code.
        No API key needed — YOU (the agent) write the implementation."""
        return await agentguard_logic(
            file_path=file_path,
            file_code=file_code,
            function_name=function_name,
            archetype=archetype,
        )

    @mcp.tool()
    async def get_challenge_criteria(
        archetype: str = "api_backend",
        extra_criteria: list[str] | None = None,
    ) -> str:
        """Get self-challenge criteria and review instructions for an archetype.
        Returns the criteria list so YOU (the agent) can self-review your output.
        No API key needed."""
        return await agentguard_get_challenge_criteria(
            archetype=archetype, extra_criteria=extra_criteria,
        )

    @mcp.tool()
    async def contracts_and_wiring(
        spec: str,
        skeleton_json: str,
        archetype: str = "api_backend",
    ) -> str:
        """Get merged L2+L3 prompts: typed stubs WITH import wiring in one pass.
        Replaces separate contracts→wiring calls, saving ~15K tokens.
        Pass the L1 skeleton JSON array. Returns per-file instructions by tier.
        No API key needed — YOU (the agent) generate the stubs."""
        return await agentguard_contracts_and_wiring(
            spec=spec, skeleton_json=skeleton_json, archetype=archetype,
        )

    @mcp.tool()
    async def digest(
        files_json: str,
        archetype: str = "api_backend",
    ) -> str:
        """Generate a compact project digest for efficient self-challenge review.
        Instead of reading every file in full, extracts exports, signatures,
        import graphs and key patterns into a ~200 line summary.
        No API key needed."""
        return await agentguard_digest(
            files_json=files_json, archetype=archetype,
        )

    @mcp.tool()
    async def benchmark(
        archetype: str = "api_backend",
        category: str | None = None,
    ) -> str:
        """Get benchmark specs for comparative evaluation (no API key needed).
        Returns 5 development specifications at different complexity levels.
        Generate code for each spec WITH and WITHOUT AgentGuard tools,
        then call `benchmark_evaluate` with the results."""
        return await agentguard_benchmark(
            archetype=archetype, category=category,
        )

    @mcp.tool()
    async def benchmark_evaluate(
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
        """Score control vs treatment code from a benchmark run (no API key needed).
        Accepts generated code from both paths, runs static-analysis scoring
        across enterprise and operational readiness dimensions, and returns
        a full report with per-dimension scores, overall verdict, and an
        environment metadata envelope (agentguard_version, python_version,
        platform, environment tag, token usage delta, and optional run context).

        If archetype_yaml is provided:
        - Validates the YAML schema first (STEP 0) and returns errors if invalid.
        - Extracts scoring_weights for fitness-aware N/A rendering.
        - Auto-uploads the report to the platform if AGENTGUARD_API_KEY is set.

        Args:
            archetype: Archetype used for the benchmark.
            results_json: JSON array with complexity, spec, control_files, treatment_files.
            archetype_yaml: Raw YAML of the archetype being benchmarked (enables validation,
                fitness weights, and auto-upload to the platform).
            environment: Calling tool tag — e.g. "vscode-copilot", "cursor", "custom-agent", "ci".
            llm_temperature: LLM temperature used, if known.
            llm_seed: LLM random seed used, if any.
            spec_source: "catalog", "custom", or "production".
            run_by: Who ran this benchmark (email or username).
            notes: Free-text notes about this run.
        """
        return await agentguard_benchmark_evaluate(
            archetype=archetype,
            results_json=results_json,
            archetype_yaml=archetype_yaml,
            environment=environment,
            llm_temperature=llm_temperature,
            llm_seed=llm_seed,
            spec_source=spec_source,
            run_by=run_by,
            notes=notes,
        )

    @mcp.tool()
    async def debug(
        symptom: str,
        archetype: str = "debug_backend",
        sources: dict[str, str] | None = None,
    ) -> str:
        """Return a structured debugging protocol for you (the calling agent) to execute.
        Loads the archetype's debug_config (data_sources, hypothesis_protocol,
        fix_protocol, escalation_criteria) and packages it with the reported symptom
        and any evidence collected.  YOU follow the protocol — form hypotheses,
        select the root cause, write a minimal fix, or escalate.
        No API key needed."""
        return await agentguard_debug(
            symptom=symptom, archetype=archetype, sources=sources,
        )

    @mcp.tool()
    async def migrate(
        source_files: dict[str, str],
        target_archetype: str = "api_backend",
        spec: str = "",
    ) -> str:
        """Return a structured migration plan for you (the calling agent) to execute.
        Loads the target archetype's migration_config (risk_areas, concern_protocol,
        incompatibility_signals, step_order), digests the source files, and returns
        a protocol YOU follow: answer the concern checklist, flag incompatibilities,
        then port the code step by step.
        No API key needed."""
        return await agentguard_migrate(
            source_files=source_files, target_archetype=target_archetype, spec=spec,
        )

    # ── Utility tools (pure computation, no LLM) ──────────────────

    from agentguard.mcp.tools import (
        agentguard_challenge,
        agentguard_generate,
        agentguard_get_archetype,
        agentguard_list_archetypes,
        agentguard_trace_summary,
        agentguard_validate,
    )

    @mcp.tool()
    async def validate(
        files: dict[str, str],
        archetype: str = "api_backend",
    ) -> str:
        """Return a structured validation prompt for you (the calling agent) to execute.
        Includes language-specific criteria (scored 0-3), environment prerequisites,
        expected structure from the archetype, and the exact response format to return.
        YOU review the files and return the scored results — no internal tools invoked."""
        return await agentguard_validate(files=files, archetype=archetype)

    @mcp.tool()
    async def list_archetypes() -> str:
        """List all available project archetypes with their descriptions."""
        return await agentguard_list_archetypes()

    @mcp.tool()
    async def get_archetype(name: str) -> str:
        """Get detailed configuration for a specific archetype
        (tech stack, validation rules, challenge criteria)."""
        return await agentguard_get_archetype(name=name)

    @mcp.tool()
    async def trace_summary(trace_id: str | None = None) -> str:
        """Get summary of the last generation trace: LLM calls, cost, results."""
        return await agentguard_trace_summary(trace_id=trace_id)

    # ── Full-pipeline tools (require LLM API key) ─────────────────
    # These are for thin/non-LLM clients. When using MCP with an LLM
    # agent, prefer the agent-native tools above instead.

    @mcp.tool()
    async def generate(
        spec: str,
        archetype: str = "api_backend",
    ) -> str:
        """Return structured generation instructions for you (the calling agent) to execute.
        YOU generate all code using your own LLM by following the returned workflow:
        skeleton → contracts_and_wiring → logic → validate → get_challenge_criteria."""
        return await agentguard_generate(spec=spec, archetype=archetype)

    @mcp.tool()
    async def challenge(
        code: str,
        criteria: list[str] | None = None,
    ) -> str:
        """Return a structured self-review prompt for you (the calling agent) to execute.
        YOU review the code using your own LLM against the returned criteria."""
        return await agentguard_challenge(code=code, criteria=criteria)

    # ── Register resources ─────────────────────────────────────────

    from agentguard.mcp.resources import (
        get_archetype_resource,
        get_archetypes_resource,
    )

    @mcp.resource("agentguard://archetypes")
    def archetypes_resource() -> str:
        """List of all available archetypes."""
        return get_archetypes_resource()

    @mcp.resource("agentguard://archetype/{name}")
    def archetype_resource(name: str) -> str:
        """Full archetype definition."""
        return get_archetype_resource(name)

    return mcp


def run_mcp_server(transport: str = "stdio", port: int = 8421) -> None:
    """Start the MCP server.

    Args:
        transport: ``"stdio"`` for local AI tools, ``"sse"`` for network.
        port: Port to use for SSE transport.
    """
    mcp = _create_mcp_server()

    if transport == "sse":
        mcp.settings.port = port
        logger.info("Starting MCP server (SSE) on port %d", port)
        mcp.run(transport="sse")
    else:
        logger.info("Starting MCP server (stdio)")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server(transport="stdio")
