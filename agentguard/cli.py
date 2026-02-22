"""AgentGuard CLI — command-line interface."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from agentguard._version import __version__


@click.group()
@click.version_option(__version__, prog_name="agentguard")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """AgentGuard — Quality-assured LLM code generation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


@main.command()
@click.argument("spec")
@click.option(
    "-a", "--archetype", default="api_backend", show_default=True,
    help="Archetype name (e.g. api_backend).",
)
@click.option(
    "-m", "--model", default="anthropic/claude-sonnet-4-20250514", show_default=True,
    help="LLM model string (provider/model).",
)
@click.option(
    "-o", "--output", default="./output", show_default=True,
    help="Output directory for generated files.",
)
@click.option(
    "--trace-store", default=None,
    help="Directory for trace JSON files.",
)
@click.option(
    "--skip-challenge", is_flag=True,
    help="Skip self-challenge step.",
)
@click.option(
    "--skip-validation", is_flag=True,
    help="Skip structural validation.",
)
@click.option(
    "--report/--no-report", default=None,
    help="Report usage to AgentGuard platform (auto-detects from config).",
)
def generate(
    spec: str,
    archetype: str,
    model: str,
    output: str,
    trace_store: str | None,
    skip_challenge: bool,
    skip_validation: bool,
    report: bool | None,
) -> None:
    """Generate code from a natural-language spec.

    SPEC is the project description, e.g. "A user auth API with JWT tokens".
    """
    from agentguard.pipeline import Pipeline

    async def _run() -> None:
        pipe = Pipeline(
            archetype=archetype,
            llm=model,
            trace_store=trace_store or f"{output}/.traces",
            report_usage=report,
        )
        try:
            result = await pipe.generate(
                spec,
                skip_challenge=skip_challenge,
                skip_validation=skip_validation,
            )

            out_dir = Path(output)
            out_dir.mkdir(parents=True, exist_ok=True)

            for file_path, content in result.files.items():
                full_path = out_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")
                click.echo(f"  ✓ {file_path}")

            click.echo(f"\n{len(result.files)} files written to {out_dir}")
            click.echo(f"Total cost: ${result.total_cost:.4f}")

            if result.trace:
                click.echo(f"\nTrace summary:\n{result.trace.summary()}")
        finally:
            await pipe.close()

    asyncio.run(_run())


@main.command()
@click.argument("archetype_name", default="api_backend")
def info(archetype_name: str) -> None:
    """Show archetype information."""
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    try:
        arch = registry.get(archetype_name)
    except KeyError:
        click.echo(f"Unknown archetype: {archetype_name}", err=True)
        sys.exit(1)

    click.echo(f"Archetype: {arch.name}")
    click.echo(f"Description: {arch.description}")
    click.echo(f"Tech Stack: {arch.tech_stack.language} / {arch.tech_stack.framework}")
    click.echo(f"Pipeline Levels: {arch.pipeline.levels}")
    click.echo("\nExpected structure:")
    click.echo(arch.get_expected_structure_text())


@main.command(name="list")
def list_archetypes() -> None:
    """List available archetypes."""
    from agentguard.archetypes.base import Archetype

    available = Archetype.list_available()
    if not available:
        click.echo("No archetypes found.")
        return

    click.echo("Available archetypes:")
    for name in sorted(available):
        click.echo(f"  • {name}")


@main.command()
@click.argument("trace_file", type=click.Path(exists=True))
def trace(trace_file: str) -> None:
    """Display a trace file summary."""
    data = json.loads(Path(trace_file).read_text(encoding="utf-8"))
    click.echo(json.dumps(data, indent=2))


# ── HTTP server ───────────────────────────────────────────────────────


@main.command()
@click.option(
    "-h", "--host", default="127.0.0.1", show_default=True,
    help="Host to bind the server to.",
)
@click.option(
    "-p", "--port", default=8420, show_default=True, type=int,
    help="Port to listen on.",
)
@click.option(
    "--api-key", default=None, envvar="AGENTGUARD_API_KEY",
    help="Require this API key in X-Api-Key header.",
)
@click.option(
    "--trace-store", default=None,
    help="Directory for trace JSON files.",
)
@click.option(
    "--reload", is_flag=True, help="Enable auto-reload for development.",
)
def serve(host: str, port: int, api_key: str | None, trace_store: str | None, reload: bool) -> None:
    """Start the AgentGuard HTTP server.

    \b
    Examples:
        agentguard serve
        agentguard serve --host 0.0.0.0 --port 9000
        agentguard serve --api-key "my-secret-key"
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "uvicorn is required for the HTTP server.\n"
            'Install it with: pip install "agentguard[server]"',
            err=True,
        )
        sys.exit(1)

    from agentguard.server.app import create_app

    app = create_app(api_key=api_key, trace_store=trace_store)
    click.echo(f"Starting AgentGuard server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)


# ── MCP server ────────────────────────────────────────────────────────


@main.command(name="mcp-serve")
@click.option(
    "--transport", default="stdio", show_default=True,
    type=click.Choice(["stdio", "sse"]),
    help="MCP transport protocol.",
)
@click.option(
    "-p", "--port", default=8421, show_default=True, type=int,
    help="Port for SSE transport.",
)
def mcp_serve(transport: str, port: int) -> None:
    """Start the AgentGuard MCP server.

    \b
    Transports:
        stdio — for local AI tools (Claude Desktop, Cursor, Windsurf)
        sse   — for remote/network MCP clients
    """
    try:
        from agentguard.mcp.server import run_mcp_server
    except ImportError:
        click.echo(
            "The mcp package is required for MCP support.\n"
            'Install it with: pip install "agentguard[mcp]"',
            err=True,
        )
        sys.exit(1)

    click.echo(f"Starting AgentGuard MCP server (transport={transport})", err=True)
    run_mcp_server(transport=transport, port=port)


# ── Validate / Challenge CLI commands ─────────────────────────────────


@main.command(name="validate")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "-a", "--archetype", default=None,
    help="Archetype to validate against.",
)
@click.option(
    "--checks", default=None,
    help="Comma-separated list of checks to run (e.g. syntax,lint).",
)
def validate_cmd(files: tuple[str, ...], archetype: str | None, checks: str | None) -> None:
    """Validate code files against quality checks.

    FILES are paths to Python files to validate.
    """
    from agentguard.archetypes.base import Archetype
    from agentguard.validation.validator import Validator

    file_map: dict[str, str] = {}
    for fp in files:
        p = Path(fp)
        file_map[p.name] = p.read_text(encoding="utf-8")

    arch = Archetype.load(archetype) if archetype else None
    check_list = [c.strip() for c in checks.split(",")] if checks else None

    validator = Validator(archetype=arch)
    report = validator.check(file_map, checks=check_list)

    if report.passed:
        click.echo("✓ All checks passed")
    else:
        click.echo(f"✗ Validation FAILED — {len(report.blocking_errors)} blocking error(s)")

    for check in report.checks:
        status = "✓" if check.passed else "✗"
        click.echo(f"  {status} {check.check} ({check.duration_ms}ms)")

    for err in report.errors:
        click.echo(f"  ERROR: {err}", err=True)

    if report.auto_fixed:
        click.echo(f"\n  {len(report.auto_fixed)} auto-fix(es) applied")

    sys.exit(0 if report.passed else 1)


@main.command(name="challenge")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "-c", "--criteria", multiple=True,
    help="Quality criteria (can specify multiple).",
)
@click.option(
    "-m", "--model", default="anthropic/claude-sonnet-4-20250514", show_default=True,
    help="LLM for self-challenge.",
)
def challenge_cmd(file: str, criteria: tuple[str, ...], model: str) -> None:
    """Self-challenge a code file against quality criteria.

    FILE is the path to the code file to challenge.
    """
    from agentguard.challenge.challenger import SelfChallenger
    from agentguard.llm.factory import create_llm_provider

    code = Path(file).read_text(encoding="utf-8")
    llm = create_llm_provider(model)
    challenger = SelfChallenger(llm=llm)

    async def _run() -> None:
        result = await challenger.challenge(
            output=code,
            criteria=list(criteria) if criteria else [],
            task_description=f"Review of {file}",
        )

        if result.passed:
            click.echo("✓ Self-challenge PASSED")
        else:
            click.echo("✗ Self-challenge FAILED")

        for cr in result.criteria_results:
            status = "✓" if cr.passed else "✗"
            click.echo(f"  {status} {cr.criterion}: {cr.explanation}")

        if result.grounding_violations:
            click.echo(f"\n  ⚠ {len(result.grounding_violations)} grounding violation(s)")
            for v in result.grounding_violations:
                click.echo(f"    • {v}")

    asyncio.run(_run())


# ── Platform config commands ──────────────────────────────────────────


@main.group(name="config")
def config_group() -> None:
    """Manage AgentGuard platform configuration."""


@config_group.command(name="set-key")
@click.argument("api_key")
def config_set_key(api_key: str) -> None:
    """Set the platform API key.

    API_KEY is the key starting with 'ag_' from your AgentGuard dashboard.
    """
    from agentguard.platform.config import load_config, save_config

    cfg = load_config()
    cfg.api_key = api_key
    path = save_config(cfg)
    click.echo(f"✓ API key saved to {path}")
    if cfg.is_configured:
        click.echo(f"  Platform: {cfg.platform_url}")


@config_group.command(name="set-url")
@click.argument("url")
def config_set_url(url: str) -> None:
    """Set the platform API URL (for self-hosted deployments)."""
    from agentguard.platform.config import load_config, save_config

    cfg = load_config()
    cfg.platform_url = url.rstrip("/")
    path = save_config(cfg)
    click.echo(f"✓ Platform URL set to {cfg.platform_url}")
    click.echo(f"  Config: {path}")


@config_group.command(name="show")
def config_show() -> None:
    """Show current platform configuration."""
    from agentguard.platform.config import CONFIG_FILE, load_config

    cfg = load_config()
    click.echo(f"Config file: {CONFIG_FILE}")
    click.echo(f"  Platform URL:  {cfg.platform_url}")
    click.echo(f"  API key:       {'ag_****' + cfg.api_key[-4:] if cfg.api_key and len(cfg.api_key) > 8 else '(not set)'}")
    click.echo(f"  Enabled:       {cfg.enabled}")
    click.echo(f"  Batch size:    {cfg.batch_size}")
    click.echo(f"  Configured:    {cfg.is_configured}")


@config_group.command(name="disable")
def config_disable() -> None:
    """Disable platform usage reporting."""
    from agentguard.platform.config import load_config, save_config

    cfg = load_config()
    cfg.enabled = False
    path = save_config(cfg)
    click.echo(f"✓ Platform reporting disabled ({path})")


@config_group.command(name="enable")
def config_enable() -> None:
    """Enable platform usage reporting."""
    from agentguard.platform.config import load_config, save_config

    cfg = load_config()
    cfg.enabled = True
    path = save_config(cfg)
    click.echo(f"✓ Platform reporting enabled ({path})")
    if not cfg.api_key:
        click.echo("  ⚠ No API key set — run: agentguard config set-key <key>")


@config_group.command(name="test")
def config_test() -> None:
    """Test connectivity to the AgentGuard platform."""
    from agentguard.platform.config import load_config

    cfg = load_config()
    if not cfg.is_configured:
        click.echo("✗ Platform not configured (no API key)")
        click.echo("  Run: agentguard config set-key <api_key>")
        sys.exit(1)

    try:
        import httpx
    except ImportError:
        click.echo(
            '✗ httpx not installed — run: pip install "rlabs-agentguard[platform]"'
        )
        sys.exit(1)

    click.echo(f"Testing connection to {cfg.platform_url}...")
    try:
        resp = httpx.get(
            f"{cfg.platform_url}/api/health",
            timeout=cfg.timeout_seconds,
        )
        if resp.status_code == 200:
            click.echo("✓ Platform reachable (health: ok)")
        else:
            click.echo(f"✗ Platform returned HTTP {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Connection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
