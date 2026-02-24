# AgentGuard Benchmark System

Comparative static-analysis benchmark that measures what the AgentGuard pipeline
actually adds over raw LLM output — no hype, no inflated baselines.

---

## How it works

Each benchmark run evaluates the same spec **twice**:

| Path | Description |
|---|---|
| **Control** | Raw LLM output — code generated directly from the spec, no AgentGuard tools |
| **Treatment** | AgentGuard pipeline output — skeleton → contracts_and_wiring → logic → validate |

Both outputs are fed through the same static-analysis evaluator. Scores are compared
per-dimension and per-complexity level, producing a delta (Δ) that shows honest
improvement or regression.

---

## Complexity levels

Every benchmark covers five levels of development complexity:

| Level | Description | Typical scope |
|---|---|---|
| `trivial` | Single-file, no deps, < 3 functions | Hello world, echo endpoint |
| `low` | Single-file with deps, 3–5 functions | Todo list, temperature converter |
| `medium` | Multi-file, external deps, 5–15 functions | Bookstore API, URL shortener |
| `high` | Complex architecture, auth/DB/state, 15+ functions | Task manager with JWT + RBAC |
| `enterprise` | Full system — observability, scaling, infra | Multi-tenant platform with WebSockets |

---

## Scoring dimensions

Scores are **0.0 – 1.0**, earned purely from static analysis of the generated code.
There are no free baseline points — code that doesn't demonstrate a pattern scores 0
for that check.

### Enterprise readiness (7 dimensions)

| Dimension | What is measured |
|---|---|
| `type_safety` | Return-type annotations, Pydantic/dataclass/TypedDict usage, `from __future__ import annotations` |
| `modularity` | Number of files, separation of concerns (models/routes/services), absence of god files |
| `maintainability` | Docstrings on functions and modules, function length, snake_case naming, UPPER_CASE constants |
| `accessibility` | i18n/locale patterns, specific exception types, HTTP status codes, API doc metadata |
| `performance` | async/await, caching (lru_cache/Redis), pagination, connection pooling, lazy loading |
| `observability` | Structured logging, multiple log levels, metrics/Prometheus, distributed tracing, health endpoints |
| `testability` | Test files present, pytest/unittest usage, constructor injection, ABC/Protocol, fixtures |

### Operational readiness (6 dimensions)

| Dimension | What is measured |
|---|---|
| `debuggability` | Custom exception classes, specific exception handling, contextual log statements, exception chaining, `__repr__` |
| `feature_extensibility` | Abstract interfaces, registry/plugin/strategy pattern, config-driven behavior, event hooks, decorator extension points |
| `cloud_scalability` | Environment-variable config (12-factor), health/readiness endpoints, Dockerfile, graceful shutdown, resource management |
| `api_migration_cost` | API versioning (`/v1/`), deprecation warnings, explicit serialization schemas, optional/default fields, `__all__` exports |
| `test_surface` | Test file count, test function count, test-to-source line ratio, pytest fixtures/parametrize/mocking/error testing |
| `team_onboarding` | README/docs, `pyproject.toml`/requirements, clear entry point (`main.py`/`app.py`), module docstrings, consistent file naming |

---

## Scoring rules

- Every dimension starts at **0.0**. Points are added only when a pattern is detected.
- Each bonus is additive and capped at 1.0 via `_clamp()`.
- Pass threshold per dimension: **0.60** (configurable via `BenchmarkConfig`).
- `test_surface` is the only dimension with a non-zero starting point (0.30) — but only
  after confirming test files exist. No files → early return at 0.0.

### Why no free baselines?

Earlier versions gave 10–30% starting credit for "having any code at all." This
compressed the control–treatment delta and made every result look unrealistically good.
All inflated baselines have been removed so scores reflect only what the code actually
demonstrates.

---

## Output format

### Compact summary (one line)
```
[PASS] api_backend | enterprise=0.541 operational=0.487 Δ=+0.312 cost=$0.0000 (5 runs)
```

### Markdown report (per complexity level)

```markdown
### High

> Build a REST API for a task management system with user authentication (JWT)…

|             | Control (raw LLM) | Treatment (AgentGuard) | Improvement |
|-------------|------------------:|-----------------------:|------------:|
| Enterprise  | 0.071             | 0.541                  | +0.470      |
| Operational | 0.060             | 0.487                  | +0.427      |
| Combined    | 0.066             | 0.514                  | **+0.449**  |
| Files       | 1                 | 8                      |             |
| Lines       | 120               | 640                    |             |
```

#### Per-dimension comparison (collapsible)

```markdown
<details><summary>Enterprise Dimensions — Control 0.071 vs Treatment 0.541</summary>

| Dimension       | Ctrl  | Treat | Δ      | Pass? | Winner | Findings (treatment)                  |
|-----------------|------:|------:|-------:|:-----:|:------:|---------------------------------------|
| type_safety     | 0.000 | 0.700 | +0.700 |  ✅   | treat  | 7/10 functions have return-type ann…  |
| modularity      | 0.200 | 0.650 | +0.450 |  ✅   | treat  | 6 Python file(s); concern separation  |
| maintainability | 0.100 | 0.550 | +0.450 |  ❌   | treat  | 4/8 functions have docstrings         |
| accessibility   | 0.000 | 0.250 | +0.250 |  ❌   | treat  | Uses specific exception types         |
| performance     | 0.150 | 0.450 | +0.300 |  ❌   | treat  | async/await patterns detected         |
| observability   | 0.000 | 0.400 | +0.400 |  ❌   | treat  | Structured logging present            |
| testability     | 0.000 | 0.700 | +0.700 |  ✅   | treat  | 2 test file(s); constructor injection |

</details>
```

#### Column guide

| Column | Meaning |
|---|---|
| `Ctrl` | Score earned by raw LLM output on this dimension (0.0–1.0) |
| `Treat` | Score earned by AgentGuard output on this dimension (0.0–1.0) |
| `Δ` | `Treat − Ctrl` — positive means AgentGuard improved this criterion |
| `Pass?` | ✅ if `Treat ≥ 0.60` threshold, ❌ otherwise |
| `Winner` | `treat` / `ctrl` / `tie` — who scored higher regardless of threshold |
| `Findings` | Top 2 reasons behind the treatment score from the static analyzer |

A dimension can show `❌` (below pass threshold) while still showing `treat` as winner —
meaning AgentGuard improved the criterion but not enough to cross 0.60. This distinction
matters: an improvement from 0.10 → 0.45 is real progress even if it doesn't pass.

---

## Overall pass criteria

A benchmark **passes** when all three conditions are met (defaults, all configurable):

| Condition | Default threshold |
|---|---|
| `enterprise_avg` (treatment) | ≥ 0.60 |
| `operational_avg` (treatment) | ≥ 0.60 |
| `improvement_avg` (Δ across all runs) | ≥ 0.05 |

---

## Running via MCP (agent-native, no API key)

```
1. Call benchmark(archetype="api_backend") → get 5 specs
2. For each spec:
   a. Generate CONTROL files (raw, no AgentGuard tools)
   b. Generate TREATMENT files (skeleton → contracts_and_wiring → logic → validate)
3. Call benchmark_evaluate(results_json=[...]) → get scored report
```

The `results_json` array format:
```json
[
  {
    "complexity": "trivial",
    "spec": "Create a REST API with a single GET /hello endpoint…",
    "control_files":   { "main.py": "..." },
    "treatment_files": { "src/app.py": "...", "src/routes/hello.py": "...", "tests/test_hello.py": "..." }
  }
]
```

## Running programmatically

```python
from agentguard.benchmark.catalog import get_default_specs
from agentguard.benchmark.runner import BenchmarkRunner
from agentguard.benchmark.types import BenchmarkConfig

config = BenchmarkConfig(
    specs=get_default_specs("api_backend"),
    model="anthropic/claude-sonnet-4-20250514",
)
runner = BenchmarkRunner(archetype="api_backend", config=config)
report = await runner.run()
print(report.to_json())
```

---

## Environment & run metadata

### What the report captures automatically

These fields are recorded in every `BenchmarkReport` and surfaced in the JSON output:

| Field | Where | Description |
|---|---|---|
| `model` | `BenchmarkReport` | LLM model string used for both control and treatment (e.g. `anthropic/claude-sonnet-4-20250514`) |
| `archetype_id` | `BenchmarkReport` | Archetype name used (e.g. `api_backend`) |
| `archetype_hash` | `BenchmarkReport` | SHA-256 of the archetype YAML — pins the exact evaluation rules for reproducibility |
| `created_at` | `BenchmarkReport` | ISO-8601 timestamp of when the run completed |
| `signature` | `BenchmarkReport` | HMAC-SHA256 of the report payload — detects tampering |
| `total_cost_usd` | `BenchmarkReport` | Aggregated LLM spend across all control + treatment runs |
| `cost_usd` | `RunResult` (per run) | LLM spend for each individual control or treatment run |
| `total_tokens` | `RunResult` (per run) | Token count consumed per run (control and treatment separately) |
| `duration_ms` | `RunResult` (per run) | Wall-clock time in milliseconds for each run |
| `files_generated` | `RunResult` (per run) | Number of files produced |
| `total_lines` | `RunResult` (per run) | Total lines of code produced |
| `error` | `RunResult` (per run) | Error message if a run failed, `null` otherwise |

### Token usage — control vs treatment

Because both paths use the same model and specs, the token delta is a direct measure of
the pipeline's overhead vs. the quality it adds:

| Metric | What it tells you |
|---|---|
| `control.total_tokens` | Tokens for a single raw prompt (one shot per spec) |
| `treatment.total_tokens` | Tokens for the full pipeline (skeleton + contracts_and_wiring + logic × N files + validate) |
| Δ tokens | Treatment typically uses **3–8× more tokens** than control — this is the cost of structured generation |
| Score Δ / token Δ | Efficiency ratio: how much quality improvement per extra token spent |

Treatment uses more tokens because the pipeline makes multiple LLM calls (one per step,
one per file). The benchmark report exposes both sides so customers can make an informed
cost-vs-quality tradeoff decision.

### What is NOT captured automatically (you should add this)

The following context is not tracked by the library but is critical for meaningful
comparison across runs. Pass these as metadata alongside the report:

| Field | Why it matters | How to capture |
|---|---|---|
| **Calling environment** | Results differ between VS Code Copilot, Cursor, a custom agent, or a CI pipeline — the surrounding context window and system prompts are different | Tag manually: `"environment": "vscode-copilot"` / `"cursor"` / `"custom-agent"` / `"ci"` |
| **AgentGuard version** | Scoring heuristics change between releases — a version bump can shift scores even with identical code | `import agentguard; agentguard.__version__` → embed in report metadata |
| **LLM temperature / seed** | Higher temperature = more variance between runs; seed pins reproducibility | Log these from your LLM provider config |
| **System prompt / context** | Agents with richer system prompts (e.g. Cursor rules, `.github/copilot-instructions.md`) may produce better control code, compressing the delta | Note whether any extra context was injected |
| **Retry count** | If a run was retried due to timeout or error, the successful attempt may not be representative | Count retries and flag runs that needed them |
| **Pipeline step breakdown** | Treatment time is spent across skeleton → contracts → logic → validate — knowing which step dominates helps optimize | Record per-step latency if using the programmatic runner |
| **Validation pass/fail per step** | Whether the treatment code passed `agentguard_validate` matters for interpreting its score | Log the `validate` result alongside the treatment files |
| **Python version / OS** | AST parsing behavior can differ across Python versions | `sys.version` |
| **Spec source** | Was it from the built-in catalog, a custom spec, or a real production task? | Tag as `"spec_source": "catalog"` / `"custom"` / `"production"` |

### Recommended metadata envelope

When sharing or storing a benchmark result, wrap the report JSON with this envelope:

```json
{
  "meta": {
    "agentguard_version": "0.3.0",
    "environment": "vscode-copilot",
    "python_version": "3.12.2",
    "llm_temperature": 0.2,
    "llm_seed": null,
    "system_prompt_injected": false,
    "spec_source": "catalog",
    "run_by": "ramon@rlabs.cl",
    "notes": "First unbiased run after baseline fix"
  },
  "report": { ... }
}
```

---

## Key files

| File | Role |
|---|---|
| [evaluator.py](evaluator.py) | Static-analysis scoring — all 13 dimension checkers |
| [report.py](report.py) | Markdown + compact formatters, side-by-side dimension comparison |
| [runner.py](runner.py) | Orchestrates control + treatment runs, calls evaluator, builds report |
| [catalog.py](catalog.py) | Predefined specs per archetype category and complexity level |
| [types.py](types.py) | All dataclasses: `DimensionScore`, `ReadinessScore`, `RunResult`, `ComplexityRun`, `BenchmarkReport` |
