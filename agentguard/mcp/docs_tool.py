"""Documentation tool — returns structured docs for LLM consumption.

Provides contextual help on AgentGuard concepts, tools, configuration,
and the archetype YAML schema.  Each topic is concise (<500 words) and
optimised for LLM agents that need quick, actionable guidance.
"""

from __future__ import annotations

DOCS_MAP: dict[str, str] = {
    # ── Core concepts ────────────────────────────────────────────────
    "overview": """\
# AgentGuard — Overview

AgentGuard is a quality-assured code-generation engine delivered as an MCP
server.  It does NOT contain its own LLM — instead it provides **structured
prompts, criteria, and validation rules** that the calling LLM agent follows
to produce higher-quality code.

Key principles:
- **Agent-native**: the MCP tools return guidance; the agent (you) does the
  thinking and code generation.
- **Archetype-driven**: every project starts from an *archetype* — a YAML
  file that defines the tech stack, directory structure, validation checks,
  and self-challenge criteria.
- **Layered pipeline**: code is built in four layers (L1-L4) — skeleton,
  contracts, wiring, logic — each layer adding more detail.
- **Self-challenge**: after generation the agent reviews its own output
  against archetype-specific criteria (grounding, security, style).
- **Static validation**: syntax, lint, type, import, and structure checks
  run without an LLM.

No API key is required for the agent-native tools.  The `generate` and
`challenge` tools exist for thin/non-LLM clients and require a separate
LLM API key.
""",

    "installation": """\
# Installation

```bash
pip install rlabs-agentguard
```

### MCP configuration (Claude Desktop / VS Code / Cursor)

Add to your MCP settings JSON:

```json
{
  "mcpServers": {
    "agentguard": {
      "command": "agentguard",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

For SSE (network) mode:

```bash
agentguard mcp --transport sse --port 8421
```

### Optional environment variables

| Variable              | Purpose                                    |
|-----------------------|--------------------------------------------|
| `AGENTGUARD_API_KEY`  | Platform features                          |
| `OPENAI_API_KEY`      | Only needed for `generate`/`challenge`     |
| `ANTHROPIC_API_KEY`   | Alternative LLM backend                    |

No API key is needed for the agent-native pipeline tools.
""",

    # ── Archetypes ───────────────────────────────────────────────────
    "archetypes": """\
# Archetypes

An archetype is a YAML configuration that defines everything AgentGuard
needs to guide code generation for a specific project type.

### What an archetype contains
- **tech_stack** — language, framework, database, testing tool, linter,
  type checker.
- **structure** — expected directories and files.
- **pipeline** — which levels are enabled, self-challenge settings.
- **validation** — which checks to run (syntax, lint, types, imports,
  structure).
- **self_challenge.criteria** — review checklist the agent applies after
  generation.
- **context_recipes** — token budgets and context inclusions per level.
- **reference_patterns** — named patterns (e.g. `fastapi_crud_pattern`).

### Built-in archetypes
- `api_backend` — Python/FastAPI REST API with PostgreSQL
- `web_app` — Full-stack web application
- `react_spa` — React single-page application
- `cli_tool` — Command-line tool
- `library` — Reusable library/package
- `script` — Standalone script
- `debug_backend` — Python/FastAPI debugging protocol
- `debug_frontend` — React/TypeScript debugging protocol

### User/community archetypes
Install from the marketplace:
```bash
agentguard install <slug>
```
Or place YAML files in `~/.agentguard/archetypes/`.
Call `reload_archetypes` after adding new ones.
""",

    "creating_archetypes": """\
# Creating Archetypes

Create a YAML file following the archetype schema and place it in
`~/.agentguard/archetypes/`.

### Minimal example

```yaml
id: "my_fastapi_app"
name: "My FastAPI App"
description: "Custom FastAPI archetype with Redis caching"
version: "1.0.0"
maturity: "production"

tech_stack:
  defaults:
    language: "python"
    framework: "fastapi"
    database: "postgresql"
    testing: "pytest"
    linter: "ruff"
    type_checker: "mypy"
  overridable: true

pipeline:
  levels: ["skeleton", "contracts", "wiring", "logic"]
  enable_self_challenge: true
  enable_structural_validation: true

structure:
  expected_dirs:
    - "src/{project_name}/"
    - "tests/"
  expected_files:
    - "src/{project_name}/__init__.py"
    - "pyproject.toml"

validation:
  checks: ["syntax", "lint", "types", "imports", "structure"]

self_challenge:
  criteria:
    - "All endpoints match the spec"
    - "No hardcoded secrets"
  grounding_check: true
  assumptions_must_declare: true
```

After saving, call `reload_archetypes` so the MCP server picks it up.
See the `archetype_yaml_schema` topic for the full field reference.
""",

    # ── Pipeline ─────────────────────────────────────────────────────
    "pipeline": """\
# Pipeline — L1 to L4

AgentGuard builds code through four progressive layers.  Each layer adds
detail on top of the previous one.

### L1 — Skeleton (`skeleton` tool)
Input: natural-language spec + archetype name.
Output: file tree with per-file responsibilities, file tiers
(config / foundation / feature), and an interface-summary block.

### L2 — Contracts (`contracts` tool)
Input: spec + L1 skeleton JSON.
Output: typed function and class stubs for every file — signatures,
docstrings, `raise NotImplementedError()` bodies.

### L3 — Wiring (`wiring` tool)
Input: L2 contracts JSON.
Output: import statements, dependency injection, call-chain connections.

### L2+L3 combined — `contracts_and_wiring` (recommended)
Merges L2 and L3 into a single call, saving ~15K tokens.  Returns
per-file instructions organised by tier.

### L4 — Logic (`logic` tool)
Input: one file + one function name.
Output: implementation instructions for that function body.
Called once per stub that needs filling in.

### Typical flow
```
skeleton → contracts_and_wiring → logic (per function)
  → get_challenge_criteria + digest → self-review
```
""",

    "skeleton": """\
# Skeleton (L1)

The `skeleton` tool returns structured instructions for generating the
project's file tree.

### Parameters
- `spec` (required) — natural-language description of what to build.
- `archetype` (default `"api_backend"`) — archetype to use.
- `maturity` (optional) — override the archetype's maturity level.

### Output
A prompt containing:
- The full list of files to create, grouped by tier:
  - **config** — pyproject.toml, Dockerfile, .env.example, etc.
  - **foundation** — core modules (models, schemas, config loaders).
  - **feature** — route handlers, services, CLI commands.
- Per-file responsibility descriptions.
- An interface-summary block so later steps know shared interfaces
  without re-reading every file.

### Usage tip
Pass the skeleton output (as a JSON array of file objects) to
`contracts_and_wiring` as the `skeleton_json` parameter.
""",

    "contracts": """\
# Contracts (L2)

The `contracts` tool produces typed function and class stubs for each
file in the skeleton.

### Parameters
- `spec` (required) — the original spec.
- `skeleton_json` (required) — the L1 skeleton output as a JSON array.
- `archetype` (default `"api_backend"`).

### Output
Per-file instructions containing:
- Class definitions with typed attributes.
- Function signatures with full type annotations.
- Docstrings describing expected behavior.
- `raise NotImplementedError()` as placeholder bodies.

### Note
For most workflows, prefer `contracts_and_wiring` which merges L2 and
L3 into one call.  The standalone `contracts` tool is kept for backward
compatibility and fine-grained control.
""",

    "wiring": """\
# Wiring (L3)

The `wiring` tool adds import statements and call-chain connections to
the L2 contracts.

### Parameters
- `contracts_json` (required) — the L2 contracts as JSON `{path: code}`.
- `archetype` (default `"api_backend"`).

### Output
Per-file instructions for:
- Adding correct import statements.
- Wiring dependency injection (e.g. FastAPI `Depends()`).
- Connecting service calls across modules.
- Ensuring circular imports are avoided.

### Note
Prefer `contracts_and_wiring` for new workflows.
""",

    "logic": """\
# Logic (L4)

The `logic` tool returns implementation instructions for a single
function body.

### Parameters
- `file_path` (required) — path of the file containing the function.
- `file_code` (required) — full source code of that file.
- `function_name` (required) — name of the function to implement.
- `archetype` (default `"api_backend"`).

### Output
Detailed instructions for replacing `raise NotImplementedError()` with
the real implementation, including:
- Step-by-step logic to follow.
- Error handling patterns from the archetype.
- References to related functions/models in the project.

### Workflow
Call `logic` once per function stub.  After all stubs are filled in,
use `get_challenge_criteria` + `digest` for self-review.
""",

    # ── Quality tools ────────────────────────────────────────────────
    "challenge": """\
# Challenge / Self-Review

AgentGuard supports two self-review workflows:

### 1. Agent-native (recommended)
1. Call `get_challenge_criteria(archetype=...)` — returns the criteria
   list from the archetype's `self_challenge.criteria`.
2. Call `digest(files=...)` — returns a compact ~200-line summary of
   the project (exports, signatures, import graph, patterns).
3. You (the agent) review the digest against the criteria and report
   pass/fail for each item.

### 2. Pipeline tool
Call `challenge(code=..., criteria=[...])` — returns a structured
review prompt.  This is for thin clients; agent-native is preferred.

### Parameters for `get_challenge_criteria`
- `archetype` (default `"api_backend"`)
- `extra_criteria` (optional list of strings) — additional items to
  check beyond the archetype defaults.

### Parameters for `digest`
- `files` — dict mapping file path to content, or a JSON string.
- `archetype` (optional) — for archetype-aware analysis.

### What the criteria typically cover
- Spec compliance (endpoints, features match the description)
- Grounding (no invented APIs, imports, or endpoints)
- Security (no hardcoded secrets, proper auth)
- Code quality (no magic numbers, DRY, async where needed)
""",

    "validation": """\
# Validation

The `validate` tool runs static analysis checks on generated code.

### Parameters
- `files` (required) — dict mapping file path to source content.
- `archetype` (default `"api_backend"`).

### Checks performed
Controlled by the archetype's `validation.checks` list:
- **syntax** — Python AST parse / JS parse validation.
- **lint** — Ruff (Python) or ESLint-equivalent rules.
- **types** — Type annotation completeness (basic or strict).
- **imports** — All imported names resolve within the project or
  known third-party packages.
- **structure** — Expected directories and files from the archetype
  are present.

### Output
Returns a structured validation prompt with:
- Language-specific criteria scored 0-3.
- Environment prerequisites.
- Expected structure from the archetype.
- The exact response format you should return with scored results.

You (the agent) review the files and return the scored results.
""",

    "marketplace": """\
# Marketplace — Community Archetypes

The AgentGuard marketplace hosts community-contributed archetypes that
extend the built-in set.

### Installing an archetype
```bash
agentguard install <slug>
```
This downloads the archetype YAML to `~/.agentguard/archetypes/`.

### After installation
Call `reload_archetypes` in the MCP server so it picks up the new
archetype without a restart.

### Publishing an archetype
1. Create your archetype YAML (see `creating_archetypes` topic).
2. Test it locally by placing it in `~/.agentguard/archetypes/`.
3. Run the full pipeline against a sample spec to verify.
4. Publish via the platform (requires `AGENTGUARD_API_KEY`):
   ```bash
   agentguard publish my_archetype.yaml
   ```

### Archetype discovery
Use `list_archetypes` to see all available archetypes (built-in +
user-installed).  Use `get_archetype(name)` for full details.
""",

    "configuration": """\
# Configuration

### Environment variables

| Variable               | Required | Purpose                                   |
|------------------------|----------|-------------------------------------------|
| `AGENTGUARD_API_KEY`   | No       | Platform features                          |
| `OPENAI_API_KEY`       | No       | Only for `generate`/`challenge` tools      |
| `ANTHROPIC_API_KEY`    | No       | Alternative LLM backend for pipeline tools |

No API key is needed for the agent-native tools (skeleton,
contracts_and_wiring, logic, get_challenge_criteria, digest, validate).

### Archetype paths
AgentGuard searches for archetypes in this order:
1. Built-in archetypes (bundled with the package).
2. `~/.agentguard/archetypes/` (user-installed / custom).

### MCP server configuration
The server is configured through the MCP settings in your IDE:
```json
{
  "mcpServers": {
    "agentguard": {
      "command": "agentguard",
      "args": ["mcp"],
      "env": {
        "AGENTGUARD_API_KEY": "ag-..."
      }
    }
  }
}
```

### Trace storage
To enable trace storage for `trace_summary`:
```bash
agentguard serve --trace-store
```
""",

    "archetype_yaml_schema": """\
# Archetype YAML Schema Reference

```yaml
# ── Required fields ──────────────────────────────────────────
id: "unique_slug"              # Unique identifier (lowercase, underscores)
name: "Human-Readable Name"    # Display name
description: "..."             # One-line description
version: "1.0.0"               # Semver
maturity: "production"         # "experimental" | "beta" | "production"

# ── Tech stack ───────────────────────────────────────────────
tech_stack:
  defaults:
    language: "python"          # Primary language
    framework: "fastapi"        # Primary framework
    database: "postgresql"      # Database (optional)
    testing: "pytest"           # Test framework
    linter: "ruff"              # Linter tool
    type_checker: "mypy"        # Type checker (optional)
  overridable: true             # Whether user can override defaults

# ── Pipeline settings ────────────────────────────────────────
pipeline:
  levels: ["skeleton", "contracts", "wiring", "logic"]
  enable_self_challenge: true
  enable_structural_validation: true
  max_self_challenge_retries: 3   # Optional, default 3

# ── Project structure ────────────────────────────────────────
structure:
  expected_dirs:
    - "src/{project_name}/"
    - "tests/"
  expected_files:
    - "src/{project_name}/__init__.py"
    - "pyproject.toml"

# ── Context recipes (token budgets per level) ────────────────
context_recipes:                  # Optional
  skeleton:
    include: ["spec", "archetype_structure"]
    max_tokens: 3000
  contracts:
    include: ["spec", "skeleton", "reference_patterns"]
    max_tokens: 6000
  wiring:
    include: ["contracts", "skeleton"]
    max_tokens: 8000
  logic:
    include: ["function_stub", "function_tests", "function_deps"]
    max_tokens: 5000

# ── Validation ───────────────────────────────────────────────
validation:
  checks: ["syntax", "lint", "types", "imports", "structure"]
  lint_rules: "ruff:default"      # Optional
  type_strictness: "basic"        # "basic" | "strict"

# ── Self-challenge criteria ──────────────────────────────────
self_challenge:
  criteria:
    - "All endpoints match the spec"
    - "No hardcoded secrets or credentials"
  grounding_check: true
  assumptions_must_declare: true

# ── Reference patterns (optional) ────────────────────────────
reference_patterns:
  - "fastapi_crud_pattern"
  - "jwt_auth_pattern"

# ── Debug config (optional, for debug archetype) ─────────────
debug_config:
  data_sources: [...]
  hypothesis_protocol: [...]
  fix_protocol: [...]
  escalation_criteria: [...]

# ── Migration config (optional) ──────────────────────────────
migration_config:
  risk_areas: [...]
  concern_protocol: [...]
  incompatibility_signals: [...]
  step_order: [...]

# ── Scoring weights (optional, for benchmark fitness) ────────
scoring_weights:
  enterprise: 0.6
  operational: 0.4
```
""",
    "usage_stats": """\
# Usage Statistics

AgentGuard tracks tool usage metrics automatically and persists them across
sessions and projects in a local SQLite database (`~/.agentguard/stats.db`).

## What's Tracked

Every tool call records:
- **Token usage** (input/output/total, estimated from text length)
- **Processing time** (wall clock milliseconds)
- **Tool name** and parameters
- **Archetype** used
- **Project** (working directory)
- **Session** (MCP server instance)
- **Context compaction** events (before/after size)

## MCP Tools

- `get_usage_stats(period, project, group_by)` — aggregated stats
  - Periods: `today`, `week`, `month`, `all`
  - Group by: `tool`, `project`, `day`, `session`
- `get_session_history(limit)` — recent sessions with summaries
- `clear_usage_stats(before, project, confirm)` — delete stats (requires confirm=true)
- `export_usage_stats(target, period, webhook_url, file_path)` — export to platform, webhook, or file
- `report_compaction(context_before_chars, context_after_chars)` — record compaction events

## Configuration

Configure stats export in your MCP server config. Get your API key from
the AgentGuard marketplace dashboard (Settings → API Keys).

```json
{
  "mcpServers": {
    "agentguard": {
      "command": "agentguard-mcp",
      "env": {
        "AGENTGUARD_API_KEY": "your-api-key-from-marketplace-dashboard",
        "AGENTGUARD_API_URL": "https://api.agentguard.rlabs.cl",
        "AGENTGUARD_STATS_WEBHOOKS": "https://your-api.com/stats,https://your-other-api.com/metrics"
      }
    }
  }
}
```

- **AGENTGUARD_API_KEY**: Your user API key from the marketplace (Settings → API Keys).
  Used for authentication when exporting to the platform.
- **AGENTGUARD_API_URL**: The AgentGuard platform URL (default: https://api.agentguard.rlabs.cl).
- **AGENTGUARD_STATS_WEBHOOKS**: Comma-separated list of webhook URLs where stats
  should also be sent. Each URL receives a POST with the same JSON payload.

With these set, you can export:
```
export_usage_stats(target="platform", period="week")   # sends to AgentGuard
export_usage_stats(target="webhooks", period="week")    # sends to all configured webhooks
export_usage_stats(target="all", period="week")         # sends to platform + all webhooks
```

To export to a one-off endpoint (not in config):
```
export_usage_stats(target="webhook", webhook_url="https://custom.endpoint.com/stats")
```

## Viewing Stats on the Platform

- **Authors**: Go to `/dashboard/usage` to see your tool usage, top archetypes, token trends
- **Admins**: Go to `/admin/usage` for platform-wide stats and leaderboard

## Storage

Stats are stored locally at `~/.agentguard/stats.db` (SQLite, WAL mode).
They persist across MCP server restarts and different projects.
To reset: `clear_usage_stats(confirm=true)` or delete the file.
""",
}

# Ordered list for the "topics" listing
_TOPIC_ORDER = [
    "overview",
    "installation",
    "archetypes",
    "creating_archetypes",
    "pipeline",
    "skeleton",
    "contracts",
    "wiring",
    "logic",
    "challenge",
    "validation",
    "marketplace",
    "configuration",
    "archetype_yaml_schema",
    "usage_stats",
]


def get_docs(topic: str) -> str:
    """Return documentation for a topic.

    If *topic* is not found, returns a list of all available topics.

    Args:
        topic: Documentation topic key (e.g. ``"overview"``, ``"pipeline"``).

    Returns:
        Formatted documentation string.
    """
    # Normalise: lowercase, strip, replace spaces/hyphens with underscores
    key = topic.strip().lower().replace("-", "_").replace(" ", "_")

    if key in DOCS_MAP:
        return DOCS_MAP[key]

    # Fuzzy: check if the key is a substring of any topic
    matches = [t for t in DOCS_MAP if key in t]
    if len(matches) == 1:
        return DOCS_MAP[matches[0]]

    # No match — return available topics
    topic_list = "\n".join(f"  - `{t}`" for t in _TOPIC_ORDER)
    hint = ""
    if matches:
        hint = "\n\nDid you mean one of: " + ", ".join(f"`{m}`" for m in matches)
    return (
        f"Topic `{topic}` not found.{hint}\n\n"
        f"Available topics:\n{topic_list}\n\n"
        "Call `docs(topic=\"<name>\")` with one of the above."
    )
