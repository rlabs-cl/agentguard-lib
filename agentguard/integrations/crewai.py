"""CrewAI integration — AgentGuard tools for CrewAI agents.

Provides ``@tool``-decorated functions that CrewAI agents can call.

Example::

    from crewai import Agent
    from agentguard.integrations.crewai import (
        agentguard_generate,
        agentguard_validate,
        agentguard_challenge,
    )

    dev = Agent(
        role="Python developer",
        tools=[agentguard_generate, agentguard_validate, agentguard_challenge],
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _run_async(coro: Any) -> Any:
    """Helper to run an async function from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def agentguard_generate(spec: str, archetype: str = "api_backend") -> str:
    """Generate a complete project from a natural-language specification.

    Args:
        spec: The natural language specification of what to build.
        archetype: The project archetype (e.g. ``"api_backend"``, ``"cli_tool"``).

    Returns:
        JSON string mapping file paths to their content.
    """
    from agentguard.pipeline import Pipeline

    async def _run() -> dict[str, str]:
        pipe = Pipeline(archetype=archetype, llm="anthropic/claude-sonnet-4-20250514")
        result = await pipe.generate(spec)
        return result.files

    files = _run_async(_run())
    return json.dumps(files, indent=2)


def agentguard_validate(files_json: str, archetype: str | None = None) -> str:
    """Validate generated code for syntax, style, and structural correctness.

    Args:
        files_json: JSON string mapping file paths to contents.
        archetype: Optional archetype name for context-aware validation.

    Returns:
        JSON string with validation report.
    """
    from agentguard.archetypes.base import Archetype
    from agentguard.validation.validator import Validator

    files = json.loads(files_json)
    arch = Archetype.load(archetype) if archetype else None
    validator = Validator(archetype=arch)
    report = validator.check(files)

    return json.dumps(
        {
            "passed": report.passed,
            "errors": [str(e) for e in report.errors],
            "warnings": [str(w) for w in report.warnings],
            "auto_fixed": len(report.auto_fixed),
        },
        indent=2,
    )


def agentguard_challenge(
    files_json: str,
    criteria_json: str = "[]",
) -> str:
    """Self-challenge code against quality criteria using an LLM.

    Args:
        files_json: JSON string mapping file paths to contents.
        criteria_json: JSON list of quality criteria strings.

    Returns:
        JSON string with challenge results.
    """
    from agentguard.challenge.challenger import SelfChallenger
    from agentguard.llm.factory import create_llm_provider

    files = json.loads(files_json)
    criteria = json.loads(criteria_json) if criteria_json else []

    code = "\n\n".join(f"# {p}\n{c}" for p, c in files.items())
    llm = create_llm_provider("anthropic/claude-sonnet-4-20250514")
    challenger = SelfChallenger(llm=llm)

    async def _run() -> dict[str, Any]:
        result = await challenger.challenge(
            output=code,
            criteria=criteria,
            task_description="Code review via CrewAI",
        )
        return {
            "passed": result.passed,
            "feedback": result.feedback,
            "grounding_violations": result.grounding_violations,
        }

    return json.dumps(_run_async(_run()), indent=2)


def agentguard_benchmark(
    archetype: str = "api_backend",
    model: str = "anthropic/claude-sonnet-4-20250514",
    category: str | None = None,
    budget: float = 10.0,
) -> str:
    """Run a comparative benchmark for an archetype.

    Generates code WITH and WITHOUT AgentGuard across 5 complexity levels,
    evaluating enterprise and operational readiness.

    Args:
        archetype: The project archetype name.
        model: LLM model string.
        category: Catalog category (defaults to archetype name).
        budget: Maximum budget in USD.

    Returns:
        JSON string with the benchmark report.
    """
    from agentguard.benchmark.catalog import get_default_specs
    from agentguard.benchmark.runner import BenchmarkRunner
    from agentguard.benchmark.types import BenchmarkConfig

    specs = get_default_specs(category or archetype)
    config = BenchmarkConfig(model=model, specs=specs, budget_ceiling_usd=budget)
    runner = BenchmarkRunner(archetype=archetype, config=config)

    report = _run_async(runner.run())
    return report.to_json()
