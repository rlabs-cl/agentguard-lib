# AgentGuard Examples

This directory contains example projects that demonstrate how to use AgentGuard
in different scenarios.

## Examples

### `basic_generation.py` — Basic Code Generation

Shows the simplest possible usage: generate a project from a spec.

```bash
python examples/basic_generation.py
```

### `validation_workflow.py` — Validate & Fix Code

Demonstrates using the validator to check code, review issues,
and apply auto-fixes.

```bash
python examples/validation_workflow.py
```

### `langgraph_pipeline.py` — LangGraph Integration

Shows how to compose AgentGuard nodes in a LangGraph StateGraph
to build an autonomous code-generation agent.

```bash
pip install langgraph
python examples/langgraph_pipeline.py
```

## Benchmark

Run a comparative benchmark that measures code quality WITH vs WITHOUT AgentGuard:

```bash
# CLI — runs 5 complexity levels, scores 13 readiness dimensions
agentguard benchmark -a api_backend

# With a different model and budget cap
agentguard benchmark -a api_backend -m openai/gpt-4o --budget 5.0

# Export reports
agentguard benchmark -a api_backend -o report.json --markdown report.md
```

Or use the Python API directly:

```python
from agentguard.benchmark import BenchmarkRunner, BenchmarkConfig
from agentguard.benchmark.catalog import get_default_specs

config = BenchmarkConfig(
    specs=get_default_specs("backend"),
    model="anthropic/claude-sonnet-4-20250514",
)
runner = BenchmarkRunner(archetype="api_backend", config=config, llm="anthropic/claude-sonnet-4-20250514")
report = await runner.run()
print(report.to_json())
```

For MCP agents (no API key needed), use the two-step `benchmark` → `benchmark_evaluate` flow.
See the [README](../README.MD#benchmark-system) for details.
