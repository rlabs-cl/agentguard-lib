# AgentGuard

[![CI](https://github.com/rlabs-cl/agentguard-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/rlabs-cl/agentguard-lib/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rlabs-agentguard?color=blue)](https://pypi.org/project/rlabs-agentguard/)
[![Python](https://img.shields.io/pypi/pyversions/rlabs-agentguard)](https://pypi.org/project/rlabs-agentguard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/rlabs-cl/agentguard-lib/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/rlabs-agentguard)](https://pypi.org/project/rlabs-agentguard/)
[![Typed](https://img.shields.io/badge/typing-typed-blue)](https://github.com/rlabs-cl/agentguard-lib)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

> A quality-assurance engine for LLM-generated code.
> Python engine + HTTP protocol + MCP server + thin SDKs for any language.

---

## What It Does

AgentGuard sits between your AI coding agent and the LLM, ensuring that every piece of generated code is:

- **Structurally sound** — Parses, lints, type-checks before any human sees it
- **Properly scoped** — Project archetypes prevent over/under-engineering
- **Built top-down** — Skeleton → contracts → wiring → logic (general to particular)
- **Self-verified** — The LLM reviews its own output against explicit criteria
- **Cost-tracked** — Every token, every dollar, every model comparison — visible

---

## Installation

Requires **Python 3.11+**.

```bash
# Core library (Anthropic + OpenAI providers included)
pip install rlabs-agentguard

# With HTTP server (FastAPI + Uvicorn)
pip install "rlabs-agentguard[server]"

# With MCP server (for Claude Desktop, Cursor, Windsurf, Cline)
pip install "rlabs-agentguard[mcp]"

# With all optional providers and transports
pip install "rlabs-agentguard[all]"
```

### Optional LLM providers

```bash
pip install "rlabs-agentguard[litellm]"    # LiteLLM router (Ollama, Together, etc.)
pip install "rlabs-agentguard[google]"     # Google Gemini
```

### Verify installation

```bash
agentguard --version
agentguard list          # Show available archetypes
agentguard info api_backend   # Show archetype details
```

---

## How It Works

AgentGuard uses a **top-down generation pipeline** that builds code from architecture to implementation, not the other way around:

```
L1 Skeleton      →  What files exist and what each one does
L2 Contracts     →  Typed function/class stubs (signatures, no bodies)
L3 Wiring        →  Import statements and call-chain connections
L4 Logic         →  Actual function implementations
   Validate      →  Syntax, lint, types, imports — mechanical checks
   Challenge     →  LLM self-reviews against 30+ criteria per archetype
```

Each level constrains the next. The LLM can't hallucinate imports at L4 because L3 already defined them. It can't invent APIs because L2 already declared the signatures. This is why MCP-generated code has better architecture — it was designed before it was implemented.

### Archetypes

An **archetype** is a project blueprint that configures the entire pipeline. It defines:

- **Tech stack** — language, framework, test runner, linter
- **Expected file structure** — what files should exist and where
- **Validation rules** — what checks to run (syntax, lint, types, imports)
- **Challenge criteria** — what the self-review evaluates (30+ criteria for `react_spa`)
- **Maturity level** — `starter` (minimal) or `production` (full infrastructure)
- **Infrastructure files** — mandatory files the pipeline must generate (ErrorBoundary, logger, constants, etc.)

| Archetype | Use When | Language | Maturity |
|-----------|----------|----------|----------|
| `script` | One-off automation, data processing | Python | starter |
| `cli_tool` | CLI with subcommands, flags, help text | Python | starter |
| `api_backend` | REST API with routes, models, auth | Python | production |
| `web_app` | Full-stack app (React + API) | Python + TS | production |
| `library` | Reusable package with public API | Python | production |
| `react_spa` | Client-side SPA with routing, state, i18n | TypeScript | production |

Pick the archetype that matches your project. Production archetypes generate more infrastructure (error boundaries, logging, code-splitting, constants) — this is intentional.

```bash
# See what an archetype expects
agentguard info react_spa
```

---

## Usage

There are **four ways** to use AgentGuard, depending on your setup:

### 1. CLI — Generate from the command line

The simplest way. No code needed.

```bash
# Generate a project from a spec
agentguard generate "A user auth API with JWT tokens, registration, and login" \
  --archetype api_backend \
  --model anthropic/claude-sonnet-4-20250514 \
  --output ./my-api

# Validate existing code files
agentguard validate src/main.py src/models.py --archetype api_backend

# Self-challenge a file against quality criteria
agentguard challenge src/main.py --criteria "No hardcoded secrets" --criteria "Error handling on all I/O"
```

#### CLI Commands Reference

| Command | What It Does |
|---------|-------------|
| `agentguard generate SPEC` | Generate a full project from a natural-language spec |
| `agentguard validate FILES...` | Run structural checks on code files |
| `agentguard challenge FILE` | Self-challenge a file against quality criteria |
| `agentguard benchmark` | Run a comparative benchmark (WITH vs WITHOUT AgentGuard) |
| `agentguard serve` | Start the HTTP API server (default port 8420) |
| `agentguard mcp-serve` | Start the MCP server (stdio or SSE transport) |
| `agentguard list` | List available archetypes |
| `agentguard info ARCHETYPE` | Show archetype details (tech stack, structure, rules) |
| `agentguard trace TRACE_FILE` | Display a trace file summary |

All commands support `--help` for full option details. Use `-v` for debug logging.

### 2. Python Library — Direct import

For building custom agents or integrating into existing Python workflows.

```python
from agentguard import Pipeline, Archetype

# Load an archetype and create a pipeline
arch = Archetype.load("api_backend")
pipe = Pipeline(archetype=arch, llm="anthropic/claude-sonnet-4-20250514")

# Generate code (returns files, trace, and cost)
result = await pipe.generate(
    spec="A user authentication API with JWT tokens, registration, and login",
)

# Write files to disk
for file_path, content in result.files.items():
    Path(file_path).write_text(content)

# Inspect what happened
print(result.trace.summary())
# → 12 LLM calls | $0.34 total | 3 structural fixes | 1 self-challenge rework
```

#### Using individual modules

You don't have to use the full pipeline. Each module works standalone:

```python
from agentguard.validation.validator import Validator
from agentguard.challenge.challenger import SelfChallenger
from agentguard.archetypes.base import Archetype

# Validate code without generating it
validator = Validator(archetype=Archetype.load("api_backend"))
report = validator.check({"main.py": code_string})
print(report.passed)  # True/False

# Challenge code against custom criteria
challenger = SelfChallenger(llm=create_llm_provider("anthropic/claude-sonnet-4-20250514"))
result = await challenger.challenge(
    output=code_string,
    criteria=["No SQL injection", "All endpoints authenticated"],
)
```

#### Supported LLMs

```python
Pipeline(llm="anthropic/claude-sonnet-4-20250514")   # Anthropic (built-in)
Pipeline(llm="openai/gpt-4o")                # OpenAI (built-in)
Pipeline(llm="google/gemini-2.0-flash")       # Google (pip install "agentguard[google]")
Pipeline(llm="litellm/ollama/llama3")         # Any LiteLLM model (pip install "agentguard[litellm]")
```

### 3. HTTP Server — For non-Python agents

Run AgentGuard as a service and call it from TypeScript, Go, Rust, or any language with HTTP.

```bash
# Start the server
agentguard serve --host 0.0.0.0 --port 8420

# Optional: require an API key
agentguard serve --api-key "my-secret-key"

# Optional: save traces to disk
agentguard serve --trace-store ./traces
```

Then call from any language:

```typescript
// TypeScript SDK (thin wrapper over HTTP)
import { AgentGuard } from "@agentguard/sdk";

const ag = new AgentGuard({ url: "http://localhost:8420" });
const result = await ag.generate({
  spec: "A user auth API with JWT tokens",
  archetype: "api_backend",
  llm: "anthropic/claude-sonnet-4-20250514",
});
```

```bash
# Or raw HTTP from any language
curl -X POST http://localhost:8420/generate \
  -H "Content-Type: application/json" \
  -d '{"spec": "A user auth API", "archetype": "api_backend"}'
```

### 4. MCP Server — For AI coding tools (recommended)

This is the most powerful integration. Your AI tool (Claude Desktop, Cursor, Windsurf, Cline) gains access to AgentGuard's tools directly. The LLM itself uses the tools during generation — no human in the loop.

**Step 1: Install with MCP support**

```bash
pip install "agentguard[mcp]"
```

**Step 2: Add to your AI tool's config**

```jsonc
// Claude Desktop: ~/.claude/claude_desktop_config.json
// Cursor:         .cursor/mcp.json
// Windsurf:       ~/.codeium/windsurf/mcp_config.json
// Cline:          .vscode/cline_mcp_settings.json
{
  "mcpServers": {
    "agentguard": {
      "command": "agentguard",
      "args": ["mcp-serve"]
    }
  }
}
```

**Step 3: Ask your AI tool to build something**

The LLM will automatically discover and use AgentGuard's tools. A typical generation flow looks like:

```
You:  "Build a whitelabel ecommerce SPA with i18n, seller onboarding,
       promo engine, and checkout"

LLM calls: skeleton(spec=..., archetype="react_spa")
       →  Returns file tree with tiers and responsibilities

LLM calls: contracts_and_wiring(spec=..., skeleton_json=...)
       →  Returns typed stubs + import wiring for every file

LLM:   Generates all files following the stubs and wiring

LLM calls: get_challenge_criteria(archetype="react_spa")
       →  Returns 36 quality criteria to self-review against

LLM:   Reviews its own output, reports pass/fail per criterion
```

No API key is needed for the agent-native tools — the host LLM does all the generation, guided by AgentGuard's structured prompts. This is the key insight: **AgentGuard doesn't replace the LLM, it gives the LLM a disciplined process to follow.**

#### MCP Tools Reference

The MCP server exposes **15 tools** in three categories:

**Agent-native tools (no API key needed — the host LLM does the work):**

| Tool | Step | What It Returns |
|------|------|-----------------|
| `skeleton` | L1 | File tree with responsibilities, tiers (config/foundation/feature), and infrastructure file requirements |
| `contracts_and_wiring` | L2+L3 | Typed function stubs + import wiring per file, merged in one pass (saves ~15K tokens vs separate calls) |
| `contracts` | L2 | Typed stubs only (use `contracts_and_wiring` instead for most cases) |
| `wiring` | L3 | Import connections only (use `contracts_and_wiring` instead for most cases) |
| `logic` | L4 | Instructions for implementing one function body — call once per `NotImplementedError` stub |
| `get_challenge_criteria` | Review | Archetype-specific quality checklist (30+ criteria for `react_spa`) with review format instructions |
| `digest` | Review | Compact project summary (~200 lines) for efficient self-challenge without re-reading every file |
| `validate` | Check | Structural validation: syntax, lint, types, imports — returns pass/fail with details |
| `list_archetypes` | Info | Names and descriptions of all available archetypes |
| `get_archetype` | Info | Full archetype config: tech stack, validation rules, challenge criteria, infrastructure files |
| `trace_summary` | Info | Summary of the last generation: LLM calls, tokens, cost |

**Benchmark tools (no API key needed — pure static analysis):**

| Tool | What It Does |
|------|-------------|
| `benchmark` | Returns 5 benchmark specs (one per complexity level) for comparative evaluation — the agent generates code WITH and WITHOUT AgentGuard tools |
| `benchmark_evaluate` | Scores control vs treatment code across 13 readiness dimensions (7 enterprise + 6 operational) using AST-based static analysis |

**Full-pipeline tools (require a separate LLM API key configured on the server):**

| Tool | What It Does |
|------|-------------|
| `generate` | Runs the entire L1→L2→L3→L4→validate→challenge pipeline using AgentGuard's internal LLM |
| `challenge` | LLM-based self-review using AgentGuard's internal LLM |

> **When to use which:** If your MCP host is already an LLM (Claude Desktop, Cursor, etc.), use the agent-native tools — they're free and the host LLM does better work when it follows the structured prompts itself. Use `generate`/`challenge` only if your MCP client is a thin script without its own LLM.

#### SSE Transport (for remote MCP clients)

```bash
# Default: stdio (for local AI tools)
agentguard mcp-serve

# SSE transport (for network/remote clients)
agentguard mcp-serve --transport sse --port 8421
```

---

## Works With Any Agent Framework

AgentGuard integrates with your existing tooling — it's not a framework, it's infrastructure:

| Framework | Integration |
|-----------|-------------|
| **LangGraph** | Python nodes for each pipeline step |
| **CrewAI** | Python tools for generation + validation |
| **OpenHands** | Python micro-agent integration |
| **Raw Python** | No framework needed — direct library import |
| **TypeScript / Go / Rust / Any** | HTTP server + thin SDK |
| **Claude Desktop / Cursor / Windsurf / Cline** | MCP server — zero integration code |

---

## Core Modules

| Module | What It Does | Use Standalone? |
|--------|-------------|:---------------:|
| **Top-Down Generator** | L1 skeleton → L2 contracts → L3 wiring → L4 logic | ✅ |
| **Structural Validator** | Syntax, lint, types, imports — zero-cost mechanical checks | ✅ |
| **Self-Challenger** | LLM reviews its own output against acceptance criteria | ✅ |
| **Context Recipes** | Right context, right amount, right time — anti-hallucination | ✅ |
| **Archetypes** | Project blueprints that configure the entire pipeline | ✅ |
| **Tracing** | Every LLM call tracked with cost, tokens, and quality metrics | ✅ |
| **Benchmark** | Comparative quality assessment: WITH vs WITHOUT AgentGuard | ✅ |

Every module works independently. Use the full pipeline or pick individual pieces.

---

## Benchmark System

AgentGuard includes a built-in **comparative benchmark** that measures the quality difference between code generated WITH and WITHOUT the AgentGuard pipeline. It generates the same project both ways, then scores both outputs across **13 readiness dimensions** using pure static analysis (AST-based — no LLM needed for scoring).

### How It Works

1. **5 complexity levels** — trivial, low, medium, high, enterprise
2. **Control run** — generate code from the spec directly (raw LLM, no AgentGuard tools)
3. **Treatment run** — generate code using the full AgentGuard workflow (skeleton → contracts → wiring → logic → validate)
4. **Score both** — 7 enterprise dimensions + 6 operational dimensions, each 0.0–1.0
5. **Report** — per-complexity scores, improvement deltas, and overall pass/fail verdict

### Readiness Dimensions

**Enterprise readiness (7 dimensions):**
type safety, modularity, maintainability, accessibility, performance, observability, testability

**Operational readiness (6 dimensions):**
debuggability, feature extensibility, cloud scalability, API migration cost, test surface, team onboarding

### CLI Usage

```bash
# Run a benchmark for an archetype (requires LLM API key for code generation)
agentguard benchmark -a api_backend
agentguard benchmark -a api_backend -m openai/gpt-4o --budget 5.0

# Custom archetype file, with report output
agentguard benchmark -a ./my-archetype.yaml -o report.json --markdown report.md

# Sign the report for marketplace publishing
agentguard benchmark -a api_backend --secret "my-signing-key"
```

### Python API

```python
from agentguard.benchmark import BenchmarkRunner, BenchmarkConfig
from agentguard.benchmark.catalog import get_default_specs

config = BenchmarkConfig(
    specs=get_default_specs("backend"),
    model="anthropic/claude-sonnet-4-20250514",   # metadata only
)

# Pass LLM directly — no separate API key config needed
runner = BenchmarkRunner(
    archetype="api_backend",
    config=config,
    llm="anthropic/claude-sonnet-4-20250514",
)
report = await runner.run()

# Sign and export
report.sign(secret="my-signing-key")
print(report.to_json())
```

### MCP / Agent-Native Usage (no API key needed)

When the host is already an LLM (Claude Desktop, Cursor, etc.), the benchmark runs as a **two-step agent-native flow**:

1. **`benchmark`** — returns 5 specs. The agent generates code for each spec twice: once raw (control), once with AgentGuard tools (treatment).
2. **`benchmark_evaluate`** — the agent passes both code sets. Scoring is pure static analysis — AST-based, zero LLM calls.

```
LLM calls: benchmark(archetype="api_backend")
       →  Returns 5 specs at different complexity levels

LLM:   For each spec, generates CONTROL code (no tools) and
       TREATMENT code (using skeleton → contracts_and_wiring → logic)

LLM calls: benchmark_evaluate(archetype="api_backend", results_json=...)
       →  Returns scored report with 13 dimensions and overall verdict
```

### Framework Integrations

All framework integrations pass through the host LLM automatically:

```python
# CrewAI — benchmark tool available as a CrewAI tool
from agentguard.integrations.crewai import agentguard_benchmark

# LangGraph — benchmark node in a StateGraph
from agentguard.integrations.langgraph import agentguard_benchmark_node

# OpenHands — benchmark method on the micro-agent
agent._benchmark(archetype="api_backend")
```

### HMAC-Signed Reports

Benchmark reports are HMAC-SHA256 signed for tamper-proof verification — required for archetype marketplace publishing:

```python
report.sign(secret="my-signing-key")
assert report.verify(secret="my-signing-key")
```

---

## Benchmarks: MCP vs No-MCP Code Generation

We ran controlled comparisons generating the same project **with** and **without** the MCP pipeline, using the same LLM (Claude) in both cases. The pipeline doesn't make the LLM smarter — it makes it more **disciplined**.

### Test Projects

| Project | Spec | Domain Complexity |
|---------|------|-------------------|
| **Health Agenda** | Patient scheduling + medication tracking + alerts | Medium (3 domains) |
| **Whitelabel Ecommerce** | i18n, seller onboarding, promo engine, pricing, search, checkout | High (8+ domains) |

### Build Metrics

| Metric | Health MCP | Health No-MCP | Ecom MCP | Ecom No-MCP |
|--------|:----------:|:-------------:|:--------:|:-----------:|
| Files | 23 | 14 | 38 | 30 |
| Lines of code | 1,907 | 998 | 5,548 | 3,324 |
| TypeScript errors | 0 | 0 | 0 | 0 |
| Vite build errors | 0 | 0 | 0 | 0 |
| Code-split chunks | — | — | 16 | 1 |

### Self-Challenge Results (Ecommerce — 36 Criteria)

| Result | MCP | No-MCP |
|--------|:---:|:------:|
| **PASS** | **24/36 (67%)** | **23/36 (64%)** |
| **FAIL** | 12/36 | 13/36 |

Both versions share 9 common failures (magic numbers, DRY violations, inline styles, etc.). The key difference is in *what each version fails at*:

- **MCP passed, No-MCP failed:** async-compatible data layer, ErrorBoundary exists, loading/error states, fuller i18n coverage
- **No-MCP passed, MCP failed:** better context splitting (3 focused contexts vs 1 god-context)

### Enterprise Readiness

| Criterion | MCP | No-MCP |
|-----------|:---:|:------:|
| Type safety | 8/10 | 7/10 |
| Modularity | 8/10 | 5/10 |
| Maintainability | 6/10 | 5/10 |
| Accessibility | 5/10 | 4/10 |
| i18n readiness | 6/10 | 5/10 |
| Performance | 8/10 | 5/10 |
| Observability | 4/10 | 2/10 |
| Testability | 5/10 | 4/10 |
| **Overall** | **6.3/10** | **4.6/10** |

### Operational Readiness

| Dimension | MCP | No-MCP | Details |
|-----------|:---:|:------:|---------|
| **Debuggability** | 8/10 | 4/10 | MCP has structured logger, ErrorBoundary, pure reducer (action-traceable). No-MCP has no logging, no error boundary, opaque `useState` callbacks. |
| **Feature extensibility** | 7/10 | 5/10 | MCP's 6-layer architecture (types → utils → contexts → hooks → components → pages) with injectable function signatures. No-MCP has data-layer coupling — `validatePromo` imports seed at module scope. |
| **Cloud scalability** | 8/10 | 4/10 | MCP code-splits into 16 chunks (lazy per page), has centralized logger for Sentry/Datadog swap, constants file for feature flags. No-MCP ships a 240KB monolithic bundle, has zero logging, no error isolation. |
| **API migration cost** | 6/10 | 3/10 | MCP utils take data as arguments (`searchProducts(products, query)`) — injectable. No-MCP bakes `PRODUCTS.find()` into cart context computed values. |
| **Test surface** | 8/10 | 5/10 | MCP has 14+ pure functions testable without React rendering, plus an exportable reducer. No-MCP has 9+ but several have module-level seed imports baked in. |
| **Team onboarding** | 7/10 | 6/10 | MCP's layered DAG lets devs own a layer. No-MCP's flatter structure is simpler but offers less parallel work boundaries. |

### What the MCP Pipeline Generates That No-MCP Skips

| Infrastructure | MCP | No-MCP | Why It Matters |
|----------------|:---:|:------:|----------------|
| ErrorBoundary | ✅ | ❌ | Without it, one page crash white-screens the whole app |
| Structured logger | ✅ | ❌ | Swap one file to connect Sentry/Datadog/CloudWatch |
| Code-splitting | ✅ | ❌ | 207KB initial load vs 240KB; independent chunk cache invalidation |
| Async hook (`useAsync`) | ✅ | ❌ | Loading/error states handled; ready for real API calls |
| Toast notification system | ✅ | ❌ | User feedback for every state mutation |
| Constants file | ✅ | ❌ | Natural home for feature flags and env-var extraction |
| Route constants | ✅ | ❌ | Change a URL in one place, not grep across files |

### Key Insight

> **The MCP pipeline's value isn't in the features it builds — both versions deliver the same checkout, search, and onboarding flows.** The value is in the *invisible infrastructure* it systematically generates: error boundaries, structured logging, code-splitting, pure utility extraction, injectable function signatures, and centralized constants.
>
> These are exactly the things that matter when you go from "it works on my laptop" to "it runs in production at scale." A solo dev building a prototype gets there faster without MCP. But the moment you need a second developer, a staging environment, or a Sentry integration, MCP's infrastructure pays for itself.

### The Gap Narrows With Complexity

| Metric | Health (MCP / No-MCP) | Ecommerce (MCP / No-MCP) |
|--------|:---------------------:|:------------------------:|
| Line ratio | 1.9× | 1.7× |
| Enterprise score | 7.5 / 4.5 | 6.3 / 4.6 |
| First-compile errors | 0 / 0 | 0 / 0 |

As projects grow more complex, the No-MCP agent produces proportionally more code (it can't avoid complexity). But the MCP pipeline's disciplined structure still delivers measurably higher enterprise quality and significantly better operational readiness.

## Demo Projects

The benchmarks above were produced from the following projects, all generated by AgentGuard's MCP pipeline (and a no-MCP baseline for comparison). You can regenerate them yourself:

```bash
# Generate the ecommerce SPA via MCP tools
agentguard generate --archetype react_spa \
  --spec "Whitelabel ecommerce SPA with i18n, seller onboarding, promo engine, pricing, search, checkout" \
  --model claude-sonnet-4-20250514

# Then validate and self-challenge the output
agentguard validate ./output --archetype react_spa
agentguard challenge ./output --archetype react_spa
```

| Project | Description |
|---------|-------------|
| Chess | Interactive chess game — MCP pipeline demo |
| Health Agenda (MCP) | Patient scheduling + medication tracking + alerts — MCP-generated |
| Health Agenda (No-MCP) | Same spec — direct generation baseline |
| Ecommerce (MCP) | Whitelabel ecommerce SPA — MCP-generated (38 files, 5,548 lines) |
| Ecommerce (No-MCP) | Same spec — direct generation baseline (30 files, 3,324 lines) |

## License

MIT — see [LICENSE](LICENSE).
