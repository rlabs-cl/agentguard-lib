"""AgentGuard MCP server — exposes the engine as MCP tools.

Supports stdio (local AI tools) and SSE (network) transports.
Uses the ``mcp`` library's ``FastMCP`` high-level API.

Tool categories:

1. **Agent-native tools** (no API key needed) — return structured prompts and
   criteria so the calling LLM agent does the thinking itself.  This is the
   correct paradigm for MCP: the tool provides *structure*, the agent provides
   *intelligence*.

2. **Utility tools** — validation, archetype listing, traces.  Pure computation,
   no LLM needed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _create_mcp_server() -> Any:
    """Build and configure the FastMCP server instance."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "AgentGuard",
        instructions=(
            "Quality-assured LLM code generation engine. "
            "Use the agent-native tools (skeleton -> contracts_and_wiring -> logic -> "
            "get_challenge_criteria + digest) to get structured prompts that "
            "guide YOUR own code generation -- no API key needed. "
            "Use validate to mechanically check code you produce."
        ),
    )

    # ── Stats collection helpers ─────────────────────────────────
    from agentguard.stats.collector import get_collector

    collector = get_collector()

    def _record_call(
        tool_name: str,
        start: float,
        result: str = "",
        archetype: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a successful tool call to the stats collector."""
        try:
            duration_ms = int((time.monotonic() - start) * 1000)
            input_tokens = max(1, len(json.dumps(kwargs, default=str)) // 4)
            output_tokens = max(1, len(str(result)) // 4)
            params_summary = json.dumps(
                {k: str(v)[:50] for k, v in kwargs.items()}, default=str,
            )
            collector.record_tool_call(
                tool_name=tool_name,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                archetype=archetype,
                status="success",
                parameters_summary=params_summary,
            )
        except Exception:
            logger.debug("Stats: failed to record call for %s", tool_name, exc_info=True)

    def _record_error(tool_name: str, start: float, exc: Exception) -> None:
        """Record a failed tool call to the stats collector."""
        try:
            duration_ms = int((time.monotonic() - start) * 1000)
            collector.record_tool_call(
                tool_name=tool_name,
                duration_ms=duration_ms,
                status="error",
                error_message=str(exc)[:200],
            )
        except Exception:
            logger.debug("Stats: failed to record error for %s", tool_name, exc_info=True)

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
        _start = time.monotonic()
        try:
            result = await agentguard_skeleton(
                spec=spec, archetype=archetype, maturity=maturity,
            )
            _record_call('skeleton', _start, result=result, archetype=archetype, spec=spec)
            return result
        except Exception as e:
            _record_error('skeleton', _start, e)
            raise

    @mcp.tool()
    async def contracts(
        spec: str,
        skeleton_json: str,
        archetype: str = "api_backend",
    ) -> str:
        """Get L2 contract prompts: typed function/class stubs for each file.
        Pass the L1 skeleton JSON array. Returns per-file instructions.
        No API key needed — YOU (the agent) generate the stubs."""
        _start = time.monotonic()
        try:
            result = await agentguard_contracts(
                spec=spec, skeleton_json=skeleton_json, archetype=archetype,
            )
            _record_call('contracts', _start, result=result, archetype=archetype, spec=spec)
            return result
        except Exception as e:
            _record_error('contracts', _start, e)
            raise

    @mcp.tool()
    async def wiring(
        contracts_json: str,
        archetype: str = "api_backend",
    ) -> str:
        """Get L3 wiring prompts: import and call-chain connections.
        Pass the L2 contracts as JSON {path: code}. Returns per-file instructions.
        No API key needed — YOU (the agent) wire the imports."""
        _start = time.monotonic()
        try:
            result = await agentguard_wiring(
                contracts_json=contracts_json, archetype=archetype,
            )
            _record_call('wiring', _start, result=result, archetype=archetype)
            return result
        except Exception as e:
            _record_error('wiring', _start, e)
            raise

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
        _start = time.monotonic()
        try:
            result = await agentguard_logic(
                file_path=file_path,
                file_code=file_code,
                function_name=function_name,
                archetype=archetype,
            )
            _record_call('logic', _start, result=result, archetype=archetype, file_path=file_path, function_name=function_name)
            return result
        except Exception as e:
            _record_error('logic', _start, e)
            raise

    @mcp.tool()
    async def get_challenge_criteria(
        archetype: str = "api_backend",
        extra_criteria: list[str] | None = None,
    ) -> str:
        """Get self-challenge criteria and review instructions for an archetype.
        Returns the criteria list so YOU (the agent) can self-review your output.
        No API key needed."""
        _start = time.monotonic()
        try:
            result = await agentguard_get_challenge_criteria(
                archetype=archetype, extra_criteria=extra_criteria,
            )
            _record_call('get_challenge_criteria', _start, result=result, archetype=archetype)
            return result
        except Exception as e:
            _record_error('get_challenge_criteria', _start, e)
            raise

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
        _start = time.monotonic()
        try:
            result = await agentguard_contracts_and_wiring(
                spec=spec, skeleton_json=skeleton_json, archetype=archetype,
            )
            _record_call('contracts_and_wiring', _start, result=result, archetype=archetype, spec=spec)
            return result
        except Exception as e:
            _record_error('contracts_and_wiring', _start, e)
            raise

    @mcp.tool()
    async def digest(
        files: dict[str, str] | str | None = None,
        archetype: str | None = None,
        files_json: str | None = None,
    ) -> str:
        """Generate a compact project digest for efficient self-challenge review.
        Accepts `files` as a dict (path → content) or JSON string — same format as `validate`.
        `files_json` is a deprecated alias for `files`; prefer `files` in new code.
        Extracts exports, signatures, import graphs and key patterns into a ~200 line summary.
        No API key needed."""
        _start = time.monotonic()
        try:
            result = await agentguard_digest(
                files=files, archetype=archetype, files_json=files_json,
            )
            _record_call('digest', _start, result=result, archetype=archetype)
            return result
        except Exception as e:
            _record_error('digest', _start, e)
            raise

    @mcp.tool()
    async def benchmark(
        archetype: str = "api_backend",
        category: str | None = None,
    ) -> str:
        """Get benchmark specs for comparative evaluation (no API key needed).
        Returns 5 development specifications at different complexity levels.
        Generate code for each spec WITH and WITHOUT AgentGuard tools,
        then call `benchmark_evaluate` with the results."""
        _start = time.monotonic()
        try:
            result = await agentguard_benchmark(
                archetype=archetype, category=category,
            )
            _record_call('benchmark', _start, result=result, archetype=archetype)
            return result
        except Exception as e:
            _record_error('benchmark', _start, e)
            raise

    @mcp.tool()
    async def benchmark_evaluate(
        archetype: str = "api_backend",
        results_json: str | list[Any] = "[]",
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
        _start = time.monotonic()
        try:
            result = await agentguard_benchmark_evaluate(
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
            _record_call('benchmark_evaluate', _start, result=result, archetype=archetype)
            return result
        except Exception as e:
            _record_error('benchmark_evaluate', _start, e)
            raise

    @mcp.tool()
    async def debug(
        symptom: str,
        archetype: str = "debug_backend",
        files: dict[str, str] | None = None,
        sources: dict[str, str] | None = None,
    ) -> str:
        """Return a structured debugging protocol for you (the calling agent) to execute.
        Loads the archetype's debug_config (data_sources, hypothesis_protocol,
        fix_protocol, escalation_criteria) and packages it with the reported symptom
        and any evidence collected.  YOU follow the protocol — form hypotheses,
        select the root cause, write a minimal fix, or escalate.
        Pass `files` (or legacy `sources`) as a dict mapping path → content.
        No API key needed."""
        _start = time.monotonic()
        try:
            result = await agentguard_debug(
                symptom=symptom, archetype=archetype,
                files=files, sources=sources,
            )
            _record_call('debug', _start, result=result, archetype=archetype, symptom=symptom)
            return result
        except Exception as e:
            _record_error('debug', _start, e)
            raise

    @mcp.tool()
    async def migrate(
        source_files: dict[str, str] | None = None,
        target_archetype: str = "api_backend",
        spec: str = "",
        files: dict[str, str] | None = None,
    ) -> str:
        """Return a structured migration plan for you (the calling agent) to execute.
        Loads the target archetype's migration_config (risk_areas, concern_protocol,
        incompatibility_signals, step_order), digests the source files, and returns
        a protocol YOU follow: answer the concern checklist, flag incompatibilities,
        then port the code step by step.
        Pass source files via `files` (preferred) or legacy `source_files` (dict path → content).
        No API key needed."""
        _start = time.monotonic()
        try:
            result = await agentguard_migrate(
                source_files=source_files, target_archetype=target_archetype,
                spec=spec, files=files,
            )
            _record_call('migrate', _start, result=result, archetype=target_archetype, spec=spec)
            return result
        except Exception as e:
            _record_error('migrate', _start, e)
            raise

    # ── Utility tools (pure computation, no LLM) ──────────────────

    from agentguard.mcp.tools import (
        agentguard_get_archetype,
        agentguard_list_archetypes,
        agentguard_reload_archetypes,
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
        _start = time.monotonic()
        try:
            result = await agentguard_validate(files=files, archetype=archetype)
            _record_call('validate', _start, result=result, archetype=archetype)
            return result
        except Exception as e:
            _record_error('validate', _start, e)
            raise

    @mcp.tool()
    async def list_archetypes() -> str:
        """List all available project archetypes with their descriptions."""
        _start = time.monotonic()
        try:
            result = await agentguard_list_archetypes()
            _record_call('list_archetypes', _start, result=result)
            return result
        except Exception as e:
            _record_error('list_archetypes', _start, e)
            raise

    @mcp.tool()
    async def get_archetype(name: str) -> str:
        """Get detailed configuration for a specific archetype
        (tech stack, validation rules, challenge criteria)."""
        _start = time.monotonic()
        try:
            result = await agentguard_get_archetype(name=name)
            _record_call('get_archetype', _start, result=result, name=name)
            return result
        except Exception as e:
            _record_error('get_archetype', _start, e)
            raise

    @mcp.tool()
    async def reload_archetypes() -> str:
        """Reload user-installed archetypes from ~/.agentguard/archetypes/.

        Call this after running 'agentguard install <slug>' so the MCP server
        picks up the new archetype without restarting."""
        _start = time.monotonic()
        try:
            result = await agentguard_reload_archetypes()
            _record_call('reload_archetypes', _start, result=result)
            return result
        except Exception as e:
            _record_error('reload_archetypes', _start, e)
            raise

    @mcp.tool()
    async def trace_summary(trace_id: str | None = None) -> str:
        """Get summary of the last generation trace: LLM calls, cost, results.
        PREREQUISITE: requires the AgentGuard HTTP server started with --trace-store
        (e.g. `agentguard serve --trace-store`). Without that flag no traces are
        persisted and this tool returns an empty result."""
        _start = time.monotonic()
        try:
            result = await agentguard_trace_summary(trace_id=trace_id)
            _record_call('trace_summary', _start, result=result)
            return result
        except Exception as e:
            _record_error('trace_summary', _start, e)
            raise

    # ── Documentation + update tools ──────────────────────────────

    from agentguard.mcp.docs_tool import get_docs
    from agentguard.mcp.updater import do_update, get_update_notice

    @mcp.tool()
    async def docs(topic: str = "overview") -> str:
        """Get AgentGuard documentation on a specific topic.
        Topics: overview, installation, archetypes, creating_archetypes, pipeline,
        skeleton, contracts, wiring, logic, challenge, validation, benchmark,
        marketplace, configuration, archetype_yaml_schema.
        Pass a topic name or keyword to search."""
        _start = time.monotonic()
        try:
            result = get_docs(topic)
            _record_call('docs', _start, result=result, topic=topic)
            return result
        except Exception as e:
            _record_error('docs', _start, e)
            raise

    @mcp.tool()
    async def update_agentguard() -> str:
        """Update AgentGuard to the latest version from PyPI.
        Returns the update result and instructs to restart the MCP server."""
        _start = time.monotonic()
        try:
            result = await do_update()
            _record_call('update_agentguard', _start, result=result)
            return result
        except Exception as e:
            _record_error('update_agentguard', _start, e)
            raise

    # Check for updates on startup (non-blocking)
    notice = get_update_notice()
    if notice:
        logger.info(notice)

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

    # ── Usage statistics tools ───────────────────────────────────

    from agentguard.mcp.stats_tools import (
        agentguard_clear_usage_stats,
        agentguard_export_usage_stats,
        agentguard_get_session_history,
        agentguard_get_usage_stats,
        agentguard_report_compaction,
    )

    @mcp.tool()
    async def get_usage_stats(
        period: str = "week",
        project: str | None = None,
        group_by: str = "tool",
    ) -> str:
        """Get aggregated usage statistics for AgentGuard tools.
        Periods: today, week, month, all. Group by: tool, project, day, session."""
        return await agentguard_get_usage_stats(
            period=period, project=project, group_by=group_by,
        )

    @mcp.tool()
    async def get_session_history(limit: int = 10) -> str:
        """Get recent AgentGuard session history with usage summaries."""
        return await agentguard_get_session_history(limit=limit)

    @mcp.tool()
    async def clear_usage_stats(
        before: str | None = None,
        project: str | None = None,
        confirm: bool = False,
    ) -> str:
        """Clear AgentGuard usage statistics. Set confirm=True to execute."""
        return await agentguard_clear_usage_stats(
            before=before, project=project, confirm=confirm,
        )

    @mcp.tool()
    async def export_usage_stats(
        target: str = "file",
        period: str = "week",
        webhook_url: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """Export usage statistics to AgentGuard platform, a webhook, or a local file."""
        return await agentguard_export_usage_stats(
            target=target, period=period, webhook_url=webhook_url, file_path=file_path,
        )

    @mcp.tool()
    async def report_compaction(
        context_before_chars: int = 0,
        context_after_chars: int = 0,
    ) -> str:
        """Record a context window compaction event for usage tracking."""
        return await agentguard_report_compaction(
            context_before_chars=context_before_chars,
            context_after_chars=context_after_chars,
        )

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
