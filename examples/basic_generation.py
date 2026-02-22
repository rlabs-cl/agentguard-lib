#!/usr/bin/env python3
"""Basic AgentGuard generation example.

Generates a simple Python script from a natural-language specification
using the ``script`` archetype, then writes the output files to disk.

Usage::

    export ANTHROPIC_API_KEY=sk-...
    python examples/basic_generation.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentguard.pipeline import Pipeline


async def main() -> None:
    spec = (
        "Write a Python script that reads a CSV file, calculates "
        "descriptive statistics (mean, median, std-dev) for each "
        "numeric column, and prints a summary table to stdout."
    )

    print("╭─────────────────────────────────────────╮")
    print("│  AgentGuard — Basic Generation Example  │")
    print("╰─────────────────────────────────────────╯")
    print()
    print(f"Spec: {spec[:80]}…")
    print()

    pipe = Pipeline(archetype="script")
    result = await pipe.generate(spec)

    # Write generated files
    output_dir = Path("output/basic_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, content in result.files.items():
        target = output_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"  ✔ {path} ({len(content)} chars)")

    print()
    print(f"Files written to: {output_dir}")
    print(f"Total cost:       ${result.total_cost}")
    print(f"Trace ID:         {result.trace.trace_id if result.trace else 'N/A'}")

    # Write trace
    if result.trace:
        trace_path = output_dir / "trace.json"
        trace_path.write_text(
            json.dumps(result.trace.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Trace saved:      {trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
