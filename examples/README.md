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
