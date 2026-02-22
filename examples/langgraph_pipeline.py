#!/usr/bin/env python3
"""LangGraph integration example.

Builds a simple LangGraph StateGraph that:
1. Generates code via AgentGuard
2. Validates the result
3. Optionally re-generates on failure

Requires: ``pip install langgraph``

Usage::

    export ANTHROPIC_API_KEY=sk-...
    python examples/langgraph_pipeline.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> None:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        print("This example requires langgraph.  Install with:")
        print("  pip install langgraph")
        sys.exit(1)

    from agentguard.integrations.langgraph import (
        agentguard_generate_node,
        agentguard_validate_node,
    )

    # State is a plain dict
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("generate", agentguard_generate_node)
    graph.add_node("validate", agentguard_validate_node)

    # Edges
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")

    def decide(state: dict[str, Any]) -> str:
        if state.get("validation_passed"):
            return "done"
        attempts = state.get("_attempts", 0)
        if attempts >= 2:
            return "done"
        state["_attempts"] = attempts + 1
        return "retry"

    graph.add_conditional_edges("validate", decide, {"done": END, "retry": "generate"})

    compiled = graph.compile()

    print("╭──────────────────────────────────────────╮")
    print("│  AgentGuard — LangGraph Pipeline Example  │")
    print("╰──────────────────────────────────────────╯")
    print()

    initial_state = {
        "spec": (
            "Build a simple in-memory key-value store with "
            "get/set/delete operations and optional TTL support."
        ),
        "archetype": "script",
    }

    print(f"Spec: {initial_state['spec']}")
    print("Running pipeline…")
    print()

    result = await compiled.ainvoke(initial_state)

    print(f"Validation passed: {result.get('validation_passed')}")
    print(f"Files generated:   {len(result.get('files', {}))}")
    for path in result.get("files", {}):
        print(f"  • {path}")


if __name__ == "__main__":
    asyncio.run(main())
