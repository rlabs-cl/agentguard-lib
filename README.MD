# AgentGuard

[![PyPI version](https://img.shields.io/pypi/v/rlabs-agentguard?color=blue)](https://pypi.org/project/rlabs-agentguard/)
[![Python](https://img.shields.io/pypi/pyversions/rlabs-agentguard)](https://pypi.org/project/rlabs-agentguard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/rlabs-cl/agentguard-lib/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/rlabs-agentguard)](https://pepy.tech/project/rlabs-agentguard)
[![CI](https://github.com/rlabs-cl/agentguard-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/rlabs-cl/agentguard-lib/actions/workflows/ci.yml)

> Agent-native quality engine for LLM code generation.
> AgentGuard provides structured guidance to your AI agent — it never calls an LLM itself.

---

## Why AgentGuard

AI agents generate code fast. But without structure, they hallucinate imports, skip error handling, and produce inconsistent architecture.

AgentGuard gives your agent a disciplined framework — without replacing it. It's the quality layer between your AI and your codebase.

## What It Does

AgentGuard is an MCP server that gives your AI agent (Claude, GPT, Gemini, etc.) a disciplined process for generating production-ready code:

1. **Skeleton** → file tree with responsibilities
2. **Contracts & Wiring** → typed stubs with import connections
3. **Logic** → function-by-function implementation
4. **Challenge** → self-review against quality criteria
5. **Validate** → static analysis (syntax, lint, types, imports)

Your agent does the thinking. AgentGuard provides the framework.

## Installation

```bash
pip install rlabs-agentguard
```

That's it. One command, no extras, no API keys needed.

## Configure Your IDE

### Claude Desktop / Claude Code

Add to your MCP config:

```json
{
  "mcpServers": {
    "agentguard": {
      "command": "agentguard-mcp"
    }
  }
}
```

### Cursor / Windsurf

Add to `.cursor/mcp.json` or equivalent:

```json
{
  "mcpServers": {
    "agentguard": {
      "command": "agentguard-mcp"
    }
  }
}
```

### Python (direct)

```bash
python -m agentguard  # starts MCP server on stdio
```

## Quick Example — What You Get

Ask your agent: *"Build a REST API for task management using AgentGuard"*

AgentGuard's `skeleton` tool returns a structured file tree:

```
src/
├── app.py              # FastAPI application entry point
├── config.py           # Environment and app configuration
├── models/
│   └── task.py         # Task SQLAlchemy model
├── routers/
│   └── tasks.py        # CRUD endpoints for /tasks
├── services/
│   └── task_service.py # Business logic layer
└── tests/
    └── test_tasks.py   # Endpoint + service tests
```

Then `contracts_and_wiring` generates typed stubs with all imports wired:

```python
# src/routers/tasks.py
from fastapi import APIRouter, Depends, HTTPException
from ..services.task_service import TaskService
from ..models.task import Task, TaskCreate, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])

@router.get("/", response_model=list[Task])
async def list_tasks(service: TaskService = Depends()) -> list[Task]: ...

@router.post("/", response_model=Task, status_code=201)
async def create_task(payload: TaskCreate, service: TaskService = Depends()) -> Task: ...
```

Your agent fills in the logic. AgentGuard ensures the structure is right from the start.

## Tools

### Agent-Native (structured guidance — no API key)

| Tool | Purpose |
|------|---------|
| `skeleton` | L1: file tree with responsibilities |
| `contracts_and_wiring` | L2+L3: typed stubs with imports (saves ~15K tokens vs separate calls) |
| `contracts` | L2 only: typed function/class stubs |
| `wiring` | L3 only: import and call-chain connections |
| `logic` | L4: implement one function body |
| `get_challenge_criteria` | Self-review criteria for an archetype |
| `digest` | Compact project summary for efficient review |
| `debug` | Structured debugging protocol |
| `migrate` | Migration plan with compatibility checks |

### Utility

| Tool | Purpose |
|------|---------|
| `validate` | Mechanical code checks (syntax, lint, types, structure) |
| `list_archetypes` | List all available project archetypes |
| `get_archetype` | Get detailed archetype configuration |
| `reload_archetypes` | Pick up newly installed archetypes |
| `trace_summary` | Get cost & token tracking summary |
| `docs` | Get AgentGuard documentation on any topic |
| `update_agentguard` | Update to the latest version from PyPI |

## Built-In Archetypes

| Archetype | Tech Stack |
|-----------|-----------|
| `api_backend` | Python + FastAPI (production) |
| `library` | Python reusable package (production) |
| `cli_tool` | Python CLI with subcommands |
| `react_spa` | TypeScript + React SPA (production) |
| `web_app` | Python + TypeScript full-stack (production) |
| `script` | Python one-off automation |
| `debug_backend` | Python/FastAPI debugging protocol |
| `debug_frontend` | React/TypeScript debugging protocol |

> **+ 61 community archetypes on the [AgentGuard Marketplace](https://agentguard.rlabs.cl/marketplace)** — browse, install, and publish your own.

## Marketplace

Install community archetypes:

```bash
# From the AgentGuard marketplace (agentguard.rlabs.cl)
# Use the reload_archetypes tool after installing
```

## How It Works

AgentGuard is **agent-native**: every tool returns structured prompts and criteria that your AI agent processes. The tool never calls an external LLM.

```
Your Agent (Claude, GPT, etc.)
    │
    ├── calls skeleton(spec, archetype) ─────→ returns L1 file tree prompt
    ├── calls contracts_and_wiring(spec, skeleton) → returns L2+L3 stubs prompt
    ├── calls logic(file, function) ─────────→ returns L4 implementation prompt
    └── calls get_challenge_criteria() ──────→ returns review criteria
    └── calls validate(files) ───────────────→ returns static analysis results
```

The agent reads the prompt, generates the code, validates it, and loops back if criteria aren't met. AgentGuard provides the structure — your agent provides the intelligence.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check agentguard/
```

## Latest Release

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up your dev environment
- Creating custom archetypes
- Submitting pull requests

Looking for a place to start? Check out issues labeled [`good first issue`](https://github.com/rlabs-cl/agentguard-lib/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## License

MIT — see [LICENSE](LICENSE).
