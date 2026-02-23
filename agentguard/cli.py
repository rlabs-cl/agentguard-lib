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
@click.option(
    "--platform-key", default=None, envvar="AGENTGUARD_PLATFORM_KEY",
    help="Platform API key (ag_*) to validate on startup.",
)
def serve(
    host: str,
    port: int,
    api_key: str | None,
    trace_store: str | None,
    reload: bool,
    platform_key: str | None,
) -> None:
    """Start the AgentGuard HTTP server.

    \b
    Examples:
        agentguard serve
        agentguard serve --host 0.0.0.0 --port 9000
        agentguard serve --api-key "my-secret-key"
        agentguard serve --platform-key ag_xxx  # validates against platform
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

    # Validate platform API key on startup if provided
    if platform_key:
        _validate_platform_key(platform_key)

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


# ── Benchmark CLI command ──────────────────────────────────────────


@main.command(name="benchmark")
@click.option(
    "-a", "--archetype", required=True,
    help="Archetype name or path to YAML file.",
)
@click.option(
    "-m", "--model", default="anthropic/claude-sonnet-4-20250514", show_default=True,
    help="LLM model string (provider/model).",
)
@click.option(
    "-o", "--output", default=None,
    help="Output path for benchmark report JSON.",
)
@click.option(
    "--markdown", default=None,
    help="Output path for Markdown report.",
)
@click.option(
    "--budget", default=10.0, show_default=True, type=float,
    help="Maximum budget in USD for the benchmark run.",
)
@click.option(
    "--secret", default=None, envvar="AGENTGUARD_BENCHMARK_SECRET",
    help="HMAC signing secret for the report.",
)
@click.option(
    "--category", default=None,
    help="Category for spec catalog lookup (auto-detected from archetype if omitted).",
)
def benchmark_cmd(
    archetype: str,
    model: str,
    output: str | None,
    markdown: str | None,
    budget: float,
    secret: str | None,
    category: str | None,
) -> None:
    """Run a comparative benchmark for an archetype.

    Runs the same development request WITH and WITHOUT AgentGuard across
    5 complexity levels, evaluating enterprise and operational readiness.

    \b
    Examples:
        agentguard benchmark -a api_backend
        agentguard benchmark -a api_backend -m openai/gpt-4o --budget 5.0
        agentguard benchmark -a ./my-archetype.yaml -o report.json --markdown report.md
    """
    from agentguard.benchmark.catalog import get_default_specs
    from agentguard.benchmark.report import format_report_compact, format_report_markdown
    from agentguard.benchmark.runner import BenchmarkRunner
    from agentguard.benchmark.types import BenchmarkConfig

    # Resolve archetype
    arch_arg: str
    if archetype.endswith((".yaml", ".yml")):
        from agentguard.archetypes.base import Archetype as _Arch
        arch_obj = _Arch.from_file(archetype)
        arch_arg = arch_obj  # type: ignore[assignment]
        cat = category or getattr(arch_obj, "id", "general")
    else:
        arch_arg = archetype
        cat = category or archetype

    # Build default specs from catalog
    specs = get_default_specs(cat)

    config = BenchmarkConfig(
        model=model,
        specs=specs,
        budget_ceiling_usd=budget,
    )

    runner = BenchmarkRunner(
        archetype=arch_arg,
        config=config,
        signing_secret=secret or "",
    )

    async def _progress(complexity: str, step: str, detail: str) -> None:
        click.echo(f"  [{complexity}] {step}: {detail}", err=True)

    async def _run() -> None:
        report = await runner.run(progress_callback=_progress)

        # CLI summary
        click.echo("")
        click.echo(format_report_compact(report))

        # Write JSON report
        if output:
            Path(output).write_text(report.to_json(), encoding="utf-8")
            click.echo(f"\n✓ JSON report written to {output}")
        else:
            default_path = f"benchmark-{report.archetype_id}-{report.model.replace('/', '_')}.json"
            Path(default_path).write_text(report.to_json(), encoding="utf-8")
            click.echo(f"\n✓ JSON report written to {default_path}")

        # Write Markdown report
        if markdown:
            md = format_report_markdown(report)
            Path(markdown).write_text(md, encoding="utf-8")
            click.echo(f"✓ Markdown report written to {markdown}")

        sys.exit(0 if report.overall_passed else 1)

    asyncio.run(_run())


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


# ── Marketplace commands ──────────────────────────────────────────


@main.group(name="marketplace")
def marketplace_group() -> None:
    """Browse and search the AgentGuard archetype marketplace."""


@marketplace_group.command(name="search")
@click.argument("query")
@click.option(
    "-c", "--category", default=None,
    help="Filter by category.",
)
@click.option(
    "--sort", default="popular",
    type=click.Choice(["popular", "newest", "price_asc", "price_desc", "rating"]),
    show_default=True,
    help="Sort order.",
)
@click.option(
    "--page", default=1, show_default=True, type=int,
    help="Page number.",
)
def marketplace_search(query: str, category: str | None, sort: str, page: int) -> None:
    """Search the marketplace for archetypes.

    QUERY is the search term (name, description, slug).
    """
    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import load_config

    _require_httpx()

    cfg = load_config()
    client = PlatformClient(cfg)

    async def _run() -> None:
        try:
            result = await client.search_marketplace(
                query=query, category=category, sort=sort, page=page,
            )
            items = result.get("items", [])
            total = result.get("total", 0)

            if not items:
                click.echo("No archetypes found.")
                return

            click.echo(f"Found {total} archetype(s) (page {result.get('page', 1)}):\n")
            for arch in items:
                price = arch.get("price_cents", 0)
                price_str = "FREE" if price == 0 else f"${price / 100:.2f}"
                rating = arch.get("rating_avg", 0)
                downloads = arch.get("downloads", 0)
                click.echo(
                    f"  {arch['slug']:<30} {price_str:<8} "
                    f"★ {rating:.1f}  ↓ {downloads}"
                )
                if arch.get("description"):
                    click.echo(f"    {arch['description'][:80]}")
                click.echo()
        finally:
            await client.close()

    asyncio.run(_run())


@marketplace_group.command(name="list")
@click.option(
    "-c", "--category", default=None,
    help="Filter by category.",
)
@click.option(
    "--sort", default="popular",
    type=click.Choice(["popular", "newest", "price_asc", "price_desc", "rating"]),
    show_default=True,
    help="Sort order.",
)
@click.option(
    "--page", default=1, show_default=True, type=int,
    help="Page number.",
)
@click.option(
    "--page-size", default=20, show_default=True, type=int,
    help="Items per page.",
)
def marketplace_list(category: str | None, sort: str, page: int, page_size: int) -> None:
    """List all published archetypes in the marketplace."""
    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import load_config

    _require_httpx()

    cfg = load_config()
    client = PlatformClient(cfg)

    async def _run() -> None:
        try:
            result = await client.search_marketplace(
                category=category, sort=sort, page=page, page_size=page_size,
            )
            items = result.get("items", [])
            total = result.get("total", 0)

            if not items:
                click.echo("No archetypes published yet.")
                return

            click.echo(f"Marketplace archetypes ({total} total, page {result.get('page', 1)}):\n")
            for arch in items:
                price = arch.get("price_cents", 0)
                price_str = "FREE" if price == 0 else f"${price / 100:.2f}"
                rating = arch.get("rating_avg", 0)
                downloads = arch.get("downloads", 0)
                click.echo(
                    f"  {arch['slug']:<30} {price_str:<8} "
                    f"★ {rating:.1f}  ↓ {downloads}"
                )
        finally:
            await client.close()

    asyncio.run(_run())


@marketplace_group.command(name="info")
@click.argument("slug")
def marketplace_info(slug: str) -> None:
    """Show detailed information about a marketplace archetype."""
    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import load_config

    _require_httpx()

    cfg = load_config()
    client = PlatformClient(cfg)

    async def _run() -> None:
        try:
            detail = await client.get_archetype_detail(slug)
            click.echo(f"Name:        {detail.get('name', slug)}")
            click.echo(f"Slug:        {detail.get('slug', slug)}")
            click.echo(f"Author:      {detail.get('author_name', 'unknown')}")
            click.echo(f"Version:     {detail.get('version', '?')}")
            price = detail.get("price_cents", 0)
            click.echo(f"Price:       {'FREE' if price == 0 else f'${price / 100:.2f}'}")
            click.echo(f"Category:    {detail.get('category', '-')}")
            click.echo(f"Downloads:   {detail.get('downloads', 0)}")
            rating = detail.get("rating_avg", 0)
            click.echo(f"Rating:      ★ {rating:.1f} ({detail.get('rating_count', 0)} reviews)")
            click.echo(f"Tags:        {', '.join(detail.get('tags', []))}")
            click.echo(f"Description: {detail.get('description', '-')}")
            if detail.get("long_description"):
                click.echo(f"\n{detail['long_description']}")
            if detail.get("is_purchased"):
                click.echo("\n  ✓ You own this archetype")
        except Exception as e:
            click.echo(f"✗ Failed to fetch archetype: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_run())


# ── Install archetype from marketplace ────────────────────────────


@main.command(name="install")
@click.argument("slug")
@click.option(
    "--force", is_flag=True,
    help="Overwrite if already installed.",
)
def install_archetype(slug: str, force: bool) -> None:
    """Download and install a marketplace archetype.

    SLUG is the archetype identifier on the marketplace (e.g. "my-archetype").

    The archetype YAML will be saved to ~/.agentguard/archetypes/<slug>.yaml
    and registered for use with ``agentguard generate -a <slug>``.
    """
    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import load_config

    _require_httpx()

    cfg = load_config()
    if not cfg.is_configured:
        click.echo("✗ Platform not configured (no API key)")
        click.echo("  Run: agentguard config set-key <api_key>")
        sys.exit(1)

    client = PlatformClient(cfg)
    archetypes_dir = Path.home() / ".agentguard" / "archetypes"

    async def _run() -> None:
        try:
            # Check if already installed
            target = archetypes_dir / f"{slug}.yaml"
            if target.exists() and not force:
                click.echo(f"✗ Archetype '{slug}' already installed at {target}")
                click.echo("  Use --force to overwrite.")
                sys.exit(1)

            click.echo(f"Downloading archetype '{slug}'...")
            data = await client.download_archetype(slug)

            yaml_content = data["yaml_content"]
            content_hash = data.get("content_hash", "")
            trust_level = data.get("trust_level", "community")

            # Validate integrity via the registry
            from agentguard.archetypes.registry import ArchetypeRegistry
            from agentguard.archetypes.schema import TrustLevel

            registry = ArchetypeRegistry(strict=True)
            tl = TrustLevel(trust_level) if trust_level in TrustLevel.__members__.values() else TrustLevel.community
            entry = registry.register_remote(
                archetype_id=slug if "id" not in data else data.get("slug", slug),
                yaml_content=yaml_content,
                content_hash=content_hash,
                trust_level=tl,
            )

            # Persist to disk
            archetypes_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml_content, encoding="utf-8")

            click.echo(f"✓ Installed '{entry.archetype.id}' v{entry.archetype.version}")
            click.echo(f"  Location:   {target}")
            click.echo(f"  Trust:      {tl.value}")
            click.echo(f"  Hash:       {content_hash[:16]}…")
            click.echo(f"\n  Use: agentguard generate -a {entry.archetype.id} \"your spec\"")
        except Exception as e:
            click.echo(f"✗ Installation failed: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_run())


# ── Uninstall archetype ───────────────────────────────────────────


@main.command(name="uninstall")
@click.argument("slug")
def uninstall_archetype(slug: str) -> None:
    """Remove a previously installed marketplace archetype.

    SLUG is the archetype identifier (e.g. "my-archetype").
    """
    archetypes_dir = Path.home() / ".agentguard" / "archetypes"
    target = archetypes_dir / f"{slug}.yaml"

    if not target.exists():
        click.echo(f"✗ Archetype '{slug}' is not installed.")
        sys.exit(1)

    target.unlink()
    click.echo(f"✓ Uninstalled archetype '{slug}'")

    # Clear license cache entry
    try:
        from agentguard.platform.license_cache import LicenseCache

        LicenseCache().remove(slug)
    except Exception:
        pass


def _require_httpx() -> None:
    """Exit with a helpful message if httpx is not installed."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        click.echo(
            '✗ httpx not installed — run: pip install "rlabs-agentguard[platform]"',
            err=True,
        )
        sys.exit(1)


def _validate_platform_key(key: str) -> None:
    """Validate a platform API key against the platform API before starting the server."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        click.echo(
            "\u26a0 httpx not installed \u2014 skipping platform key validation",
            err=True,
        )
        return

    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import load_config

    cfg = load_config()
    cfg.api_key = key

    client = PlatformClient(cfg)

    async def _check() -> None:
        try:
            info = await client.validate_api_key()
            click.echo(
                f"✓ Platform key valid — {info.get('email', '?')} "
                f"(tier: {info.get('tier', '?')})"
            )
        except Exception as e:
            click.echo(f"✗ Platform key validation failed: {e}", err=True)
            sys.exit(1)
        finally:
            await client.close()

    asyncio.run(_check())


if __name__ == "__main__":
    main()
