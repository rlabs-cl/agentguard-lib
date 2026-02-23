"""OpenHands integration — AgentGuard micro-agent for OpenHands.

Provides a micro-agent description and helper that OpenHands can invoke
to generate, validate, and challenge code using AgentGuard.

Example::

    from agentguard.integrations.openhands import AgentGuardMicroAgent

    agent = AgentGuardMicroAgent()
    result = await agent.run(
        instruction="Build a REST API for a todo list",
        archetype="api_backend",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MICRO_AGENT_DESCRIPTION = """\
AgentGuard micro-agent for OpenHands.

I can help you generate, validate, self-challenge, and benchmark code
using the AgentGuard quality-assurance pipeline.  I support these actions:

- **generate**: Generate a full project from a specification
- **validate**: Check code for syntax, lint, type, and structural issues
- **challenge**: Self-challenge code against quality criteria via LLM
- **benchmark**: Comparative benchmark (with vs without AgentGuard)

Provide an ``action`` and appropriate parameters.
"""


@dataclass
class MicroAgentResult:
    """Result of a micro-agent run."""

    action: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "action": self.action,
                "success": self.success,
                "data": self.data,
                "error": self.error,
            },
            indent=2,
        )


class AgentGuardMicroAgent:
    """OpenHands-compatible micro-agent wrapping AgentGuard pipeline."""

    name = "agentguard"
    description = MICRO_AGENT_DESCRIPTION

    def __init__(
        self,
        llm: str = "anthropic/claude-sonnet-4-20250514",
    ) -> None:
        self.llm = llm

    async def run(
        self,
        instruction: str,
        action: str = "generate",
        archetype: str = "api_backend",
        files: dict[str, str] | None = None,
        criteria: list[str] | None = None,
        benchmark_budget: float = 10.0,
    ) -> MicroAgentResult:
        """Execute a micro-agent action.

        Args:
            instruction: Natural-language instruction or spec.
            action: One of ``"generate"``, ``"validate"``, ``"challenge"``, ``"benchmark"``.
            archetype: Archetype name for generation / validation / benchmark.
            files: Pre-existing files for validate / challenge.
            criteria: Quality criteria for challenge.
            benchmark_budget: Max budget (USD) for benchmark runs.

        Returns:
            MicroAgentResult with action outcome.
        """
        try:
            if action == "generate":
                return await self._generate(instruction, archetype)
            elif action == "validate":
                return await self._validate(files or {}, archetype)
            elif action == "challenge":
                return await self._challenge(files or {}, criteria or [])
            elif action == "benchmark":
                return await self._benchmark(archetype, benchmark_budget)
            else:
                return MicroAgentResult(
                    action=action,
                    success=False,
                    error=f"Unknown action: {action}. Use generate/validate/challenge/benchmark.",
                )
        except Exception as exc:
            logger.exception("Micro-agent %s failed", action)
            return MicroAgentResult(action=action, success=False, error=str(exc))

    async def _generate(
        self, spec: str, archetype: str
    ) -> MicroAgentResult:
        from agentguard.pipeline import Pipeline

        pipe = Pipeline(archetype=archetype, llm=self.llm)
        result = await pipe.generate(spec)
        return MicroAgentResult(
            action="generate",
            success=True,
            data={
                "files": result.files,
                "total_cost": str(result.total_cost),
            },
        )

    async def _validate(
        self, files: dict[str, str], archetype: str
    ) -> MicroAgentResult:
        from agentguard.archetypes.base import Archetype
        from agentguard.validation.validator import Validator

        arch = Archetype.load(archetype) if archetype else None
        validator = Validator(archetype=arch)
        report = validator.check(files)
        return MicroAgentResult(
            action="validate",
            success=report.passed,
            data={
                "passed": report.passed,
                "errors": [str(e) for e in report.errors],
                "warnings": [str(w) for w in report.warnings],
                "auto_fixed": len(report.auto_fixed),
            },
        )

    async def _challenge(
        self, files: dict[str, str], criteria: list[str]
    ) -> MicroAgentResult:
        from agentguard.challenge.challenger import SelfChallenger
        from agentguard.llm.factory import create_llm_provider

        code = "\n\n".join(f"# {p}\n{c}" for p, c in files.items())
        llm = create_llm_provider(self.llm)
        challenger = SelfChallenger(llm=llm)
        result = await challenger.challenge(
            output=code,
            criteria=criteria,
            task_description="Code review via OpenHands",
        )
        return MicroAgentResult(
            action="challenge",
            success=result.passed,
            data={
                "passed": result.passed,
                "feedback": result.feedback,
                "grounding_violations": result.grounding_violations,
            },
        )

    async def _benchmark(
        self, archetype: str, budget: float,
    ) -> MicroAgentResult:
        from agentguard.benchmark.catalog import get_default_specs
        from agentguard.benchmark.runner import BenchmarkRunner
        from agentguard.benchmark.types import BenchmarkConfig

        specs = get_default_specs(archetype)
        config = BenchmarkConfig(
            model=self.llm, specs=specs, budget_ceiling_usd=budget,
        )
        runner = BenchmarkRunner(archetype=archetype, config=config)
        report = await runner.run()
        return MicroAgentResult(
            action="benchmark",
            success=report.overall_passed,
            data=report.to_dict(),
        )
