#!/usr/bin/env python3
"""Validation workflow example.

Demonstrates using the AgentGuard validator to check a set of files
for syntax, lint, type, import, and structural issues, then printing
the results.

Usage::

    python examples/validation_workflow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentguard.validation.validator import Validator


def main() -> None:
    # Some sample code with deliberate issues
    files = {
        "main.py": (
            "import os\n"
            "import json  # unused\n"
            "\n"
            "def greet(name):\n"
            '    return f"Hello, {name}!"\n'
            "\n"
            'if __name__ == "__main__":\n'
            '    print(greet("World"))\n'
        ),
        "utils.py": (
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def broken_func(\n"
            "    # missing closing paren\n"
        ),
    }

    print("╭─────────────────────────────────────────╮")
    print("│  AgentGuard — Validation Workflow        │")
    print("╰─────────────────────────────────────────╯")
    print()

    validator = Validator()
    report = validator.check(files)

    print(f"Overall result:  {'✔ PASSED' if report.passed else '✘ FAILED'}")
    print(f"Errors:          {len(report.errors)}")
    print(f"Warnings:        {len(report.warnings)}")
    print(f"Auto-fixes:      {len(report.auto_fixed)}")
    print()

    if report.errors:
        print("─── Errors ───")
        for err in report.errors:
            print(f"  ✘ {err}")
        print()

    if report.warnings:
        print("─── Warnings ───")
        for warn in report.warnings:
            print(f"  ⚠ {warn}")
        print()

    if report.auto_fixed:
        print("─── Auto-fixes Applied ───")
        for fix in report.auto_fixed:
            print(f"  ✔ {fix}")
        print()

    # Machine-readable output
    result = {
        "passed": report.passed,
        "errors": [str(e) for e in report.errors],
        "warnings": [str(w) for w in report.warnings],
        "auto_fixed": [str(f) for f in report.auto_fixed],
    }
    print("JSON output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
