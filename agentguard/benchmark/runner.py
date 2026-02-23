"""BenchmarkRunner — orchestrates comparative control/treatment runs.

Control = raw LLM generation (single prompt, no pipeline).
Treatment = full AgentGuard pipeline (skeleton → contracts → wiring → logic +
            validation + self-challenge).

For each benchmark spec the runner:
  1. Generates code via the control path (raw LLM).
  2. Generates code via the treatment path (Pipeline).
  3. Evaluates both outputs with the enterprise + operational evaluator.
  4. Records per-complexity results and computes deltas.
  5. Produces a signed BenchmarkReport.
"""

from __future__ import annotations

import logging
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from agentguard.benchmark.evaluator import evaluate_enterprise, evaluate_operational
from agentguard.benchmark.types import (
    BenchmarkConfig,
    BenchmarkReport,
    ComplexityRun,
    ReadinessScore,
    RunResult,
)

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.llm.base import LLMProvider

_log = logging.getLogger(__name__)

# Type alias for the progress callback.
_ProgressCB = Callable[..., object]

# Minimal system prompt for the control (raw LLM) run.
_CONTROL_SYSTEM = (
    "You are an expert software developer. Generate production-quality code "
    "for the following specification. Return the code as a set of files. "
    "Use the format:\n\n"
    "```<filepath>\n<content>\n```\n\n"
    "for each file. Include all necessary imports and boilerplate."
)


class BenchmarkRunner:
    """Run a comparative benchmark (control vs treatment) for an archetype.

    Usage::

        from agentguard.benchmark import BenchmarkRunner
        from agentguard.benchmark.catalog import get_default_specs

        config = BenchmarkConfig(
            model="anthropic/claude-sonnet-4-20250514",
            specs=get_default_specs("backend"),
        )
        runner = BenchmarkRunner(archetype="api_backend", config=config)
        report = await runner.run()
        report.sign(secret="my-signing-key")
        print(report.to_json())
    """

    def __init__(
        self,
        archetype: Archetype | str,
        config: BenchmarkConfig,
        *,
        signing_secret: str = "",
    ) -> None:
        from agentguard.archetypes.registry import get_archetype_registry
        from agentguard.llm.factory import create_llm_provider

        # Resolve archetype
        if isinstance(archetype, str):
            registry = get_archetype_registry()
            self._archetype: Archetype = registry.get(archetype)
        else:
            self._archetype = archetype

        # Validate config
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid benchmark config: {'; '.join(errors)}")

        self._config = config
        self._llm: LLMProvider = create_llm_provider(config.model)
        self._secret = signing_secret
        self._total_cost: float = 0.0

    async def run(self, *, progress_callback: _ProgressCB | None = None) -> BenchmarkReport:
        """Execute the full benchmark across all configured specs.

        Args:
            progress_callback: Optional async callable(complexity, step, detail)
                for reporting progress to callers (CLI, MCP, etc.).

        Returns:
            A BenchmarkReport with all run results and aggregates.
        """
        runs: list[ComplexityRun] = []
        self._total_cost = 0.0

        for i, spec in enumerate(self._config.specs):
            _log.info(
                "Benchmark [%d/%d] complexity=%s spec=%s",
                i + 1, len(self._config.specs),
                spec.complexity.value,
                spec.spec[:60],
            )

            if progress_callback:
                await progress_callback(
                    spec.complexity.value,
                    "starting",
                    f"Running {spec.complexity.value} benchmark ({i + 1}/{len(self._config.specs)})",
                )

            # Budget guard
            if self._total_cost >= self._config.budget_ceiling_usd:
                _log.warning(
                    "Budget ceiling (%.2f USD) reached after %d runs — stopping",
                    self._config.budget_ceiling_usd,
                    len(runs),
                )
                break

            # ── Control run ────────────────────────────────────
            if progress_callback:
                await progress_callback(spec.complexity.value, "control", "Raw LLM generation (no AgentGuard)")
            control = await self._run_control(spec.spec)

            # ── Treatment run ──────────────────────────────────
            if progress_callback:
                await progress_callback(spec.complexity.value, "treatment", "AgentGuard pipeline generation")
            treatment = await self._run_treatment(spec.spec)

            complexity_run = ComplexityRun(
                complexity=spec.complexity,
                spec=spec.spec,
                control=control,
                treatment=treatment,
            )
            runs.append(complexity_run)

            self._total_cost += control.cost_usd + treatment.cost_usd

            _log.info(
                "  → control=%.3f  treatment=%.3f  Δ=%.3f",
                control.combined_score,
                treatment.combined_score,
                complexity_run.improvement,
            )

        # Build report
        report = self._build_report(runs)

        if progress_callback:
            await progress_callback(
                "all", "complete",
                f"Benchmark complete: {'PASSED' if report.overall_passed else 'FAILED'} "
                f"(Δ={report.improvement_avg:.3f}, cost=${report.total_cost_usd:.4f})",
            )

        return report

    # ══════════════════════════════════════════════════════════
    #  Control run — raw LLM, no pipeline
    # ══════════════════════════════════════════════════════════

    async def _run_control(self, spec: str) -> RunResult:
        """Generate code with a raw LLM call (no AgentGuard)."""
        from agentguard.llm.types import Message

        t0 = _time.perf_counter()
        try:
            response = await self._llm.generate(
                messages=[
                    Message(role="system", content=_CONTROL_SYSTEM),
                    Message(role="user", content=spec),
                ],
            )
            duration_ms = int((_time.perf_counter() - t0) * 1000)

            # Parse file blocks from the LLM response
            files = _parse_file_blocks(response.content)
            if not files:
                # Treat the whole response as a single file
                files = {"main.py": response.content}

            # Evaluate
            enterprise = evaluate_enterprise(files, self._config.enterprise_threshold)
            operational = evaluate_operational(files, self._config.operational_threshold)

            total_lines = sum(c.count("\n") + 1 for c in files.values())
            cost = float(response.cost.total_cost)

            return RunResult(
                enterprise=enterprise,
                operational=operational,
                files_generated=len(files),
                total_lines=total_lines,
                total_tokens=response.tokens.total_tokens,
                cost_usd=cost,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((_time.perf_counter() - t0) * 1000)
            _log.error("Control run failed: %s", exc)
            return _error_result(str(exc), duration_ms)

    # ══════════════════════════════════════════════════════════
    #  Treatment run — full AgentGuard pipeline
    # ══════════════════════════════════════════════════════════

    async def _run_treatment(self, spec: str) -> RunResult:
        """Generate code through the full AgentGuard pipeline."""
        from agentguard.pipeline import Pipeline

        t0 = _time.perf_counter()
        pipe = Pipeline(
            archetype=self._archetype,
            llm=self._llm,
            report_usage=False,  # Don't report benchmark runs to platform
        )
        try:
            result = await pipe.generate(spec)
            duration_ms = int((_time.perf_counter() - t0) * 1000)

            files = result.files

            # Evaluate
            enterprise = evaluate_enterprise(files, self._config.enterprise_threshold)
            operational = evaluate_operational(files, self._config.operational_threshold)

            total_lines = sum(c.count("\n") + 1 for c in files.values())
            cost = float(result.total_cost.total_cost)

            return RunResult(
                enterprise=enterprise,
                operational=operational,
                files_generated=len(files),
                total_lines=total_lines,
                total_tokens=0,  # Pipeline doesn't expose aggregate tokens directly
                cost_usd=cost,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((_time.perf_counter() - t0) * 1000)
            _log.error("Treatment run failed: %s", exc)
            return _error_result(str(exc), duration_ms)
        finally:
            await pipe.close()

    # ══════════════════════════════════════════════════════════
    #  Report building
    # ══════════════════════════════════════════════════════════

    def _build_report(self, runs: list[ComplexityRun]) -> BenchmarkReport:
        """Build and optionally sign the benchmark report."""
        # Compute archetype hash
        try:
            import yaml  # noqa: F401

            from agentguard.archetypes.schema import compute_content_hash  # noqa: F401

            # Hash the archetype YAML if available
            arch_hash = getattr(self._archetype, "_yaml_content_hash", "")
            if not arch_hash:
                arch_hash = ""
        except Exception:
            arch_hash = ""

        report = BenchmarkReport(
            archetype_id=self._archetype.id,
            archetype_hash=arch_hash,
            model=self._config.model,
            runs=runs,
            created_at=datetime.now(UTC).isoformat(),
        )
        report.compute_aggregates()

        # Determine pass/fail
        report.overall_passed = (
            report.enterprise_avg >= self._config.enterprise_threshold
            and report.operational_avg >= self._config.operational_threshold
            and report.improvement_avg >= self._config.improvement_threshold
            and len(runs) == len(self._config.specs)  # All levels completed
        )

        # Sign if secret provided
        if self._secret:
            report.sign(self._secret)

        return report


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════


def _error_result(error: str, duration_ms: int) -> RunResult:
    """Create a zero-score RunResult for an errored run."""
    empty_enterprise = ReadinessScore(
        category="enterprise", overall_score=0.0, passed=False, dimensions=[],
    )
    empty_operational = ReadinessScore(
        category="operational", overall_score=0.0, passed=False, dimensions=[],
    )
    return RunResult(
        enterprise=empty_enterprise,
        operational=empty_operational,
        duration_ms=duration_ms,
        error=error,
    )


def _parse_file_blocks(content: str) -> dict[str, str]:
    """Parse ```filepath ... ``` code blocks from LLM output.

    Supports formats:
      ```python filepath
      ```filepath
      ```<filepath>
    """
    import re

    pattern = re.compile(
        r"```(?:\w+\s+)?<?([^\s>`]+)>?\s*\n(.*?)```",
        re.DOTALL,
    )
    files: dict[str, str] = {}
    for match in pattern.finditer(content):
        fpath = match.group(1).strip()
        fcontent = match.group(2)
        if fpath and not fpath.startswith("#"):
            files[fpath] = fcontent
    return files
