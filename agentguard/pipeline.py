"""Pipeline — main entry point for AgentGuard code generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agentguard.archetypes.registry import get_archetype_registry
from agentguard.challenge.challenger import SelfChallenger
from agentguard.challenge.types import ChallengeResult
from agentguard.llm.factory import create_llm_provider
from agentguard.prompts.registry import get_prompt_registry
from agentguard.topdown.generator import TopDownGenerator
from agentguard.topdown.types import (
    ContractsResult,
    GenerationResult,
    LogicResult,
    SkeletonResult,
    WiringResult,
)
from agentguard.tracing.tracer import Tracer
from agentguard.validation.validator import Validator

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.llm.base import LLMProvider
    from agentguard.platform.client import PlatformClient
    from agentguard.platform.config import PlatformConfig
    from agentguard.prompts.template import PromptTemplate
    from agentguard.validation.types import ValidationReport

_log = logging.getLogger(__name__)
logger = _log  # backward compat alias


class Pipeline:
    """The main entry point. Orchestrates the full quality pipeline.

    Usage::

        pipe = Pipeline(archetype="api_backend", llm="anthropic/claude-sonnet-4-20250514")
        result = await pipe.generate("A user auth API with JWT tokens")
        for path, content in result.files.items():
            Path(path).write_text(content)
    """

    def __init__(
        self,
        archetype: Archetype | str,
        llm: LLMProvider | str,
        challenge_llm: LLMProvider | str | None = None,
        trace_store: str | Path | None = None,
        prompt_overrides: dict[str, PromptTemplate] | None = None,
        report_usage: bool | None = None,
        platform_config: PlatformConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            archetype: Archetype instance or name (e.g. "api_backend").
            llm: LLMProvider instance or model string (e.g. "anthropic/claude-sonnet-4-20250514").
            challenge_llm: Optional separate LLM for self-challenge (cheaper model).
            trace_store: Directory for trace JSON files. None disables persistence.
            prompt_overrides: Override builtin prompt templates by name.
            report_usage: Whether to report usage to the AgentGuard platform.
                None = auto-detect from config, True = force enable, False = disable.
            platform_config: Explicit platform config. If None, loads from
                ``~/.agentguard/config.yaml`` or environment variables.
        """
        # Resolve archetype
        if isinstance(archetype, str):
            registry = get_archetype_registry()
            self._archetype = registry.get(archetype)
        else:
            self._archetype = archetype
            # Verify the archetype is registered (integrity check)
            registry = get_archetype_registry()
            if registry.is_registered(archetype.id):
                entry = registry.get_entry(archetype.id)
                _log.debug(
                    "Pipeline using registered archetype '%s' [%s] hash=%s…",
                    archetype.id, entry.trust_level.value, entry.content_hash[:12],
                )
            else:
                _log.warning(
                    "Pipeline using UNREGISTERED archetype '%s' — "
                    "integrity cannot be verified. Use Archetype.load() "
                    "or register via the registry for integrity guarantees.",
                    archetype.id,
                )

        # Resolve main LLM
        if isinstance(llm, str):
            self._llm = create_llm_provider(llm)
        else:
            self._llm = llm

        # Resolve challenge LLM (falls back to main LLM)
        if challenge_llm is not None:
            if isinstance(challenge_llm, str):
                self._challenge_llm = create_llm_provider(challenge_llm)
            else:
                self._challenge_llm = challenge_llm
        else:
            self._challenge_llm = self._llm

        # Trace store
        store_path = Path(trace_store) if trace_store else None
        self._tracer = Tracer(store_dir=store_path)

        # Validator and SelfChallenger
        self._validator = Validator(archetype=self._archetype)
        self._challenger = SelfChallenger(llm=self._challenge_llm)

        # Apply prompt overrides
        if prompt_overrides:
            prompt_registry = get_prompt_registry()
            for name, template in prompt_overrides.items():
                template.id = name
                prompt_registry.register(template)

        # Platform usage reporting
        self._platform: PlatformClient | None = None
        if report_usage is not False:
            try:
                from agentguard.platform.client import PlatformClient as _PC
                from agentguard.platform.config import load_config as _load_cfg

                cfg = platform_config or _load_cfg()
                if report_usage is True:
                    cfg.enabled = True
                if cfg.is_configured:
                    self._platform = _PC(cfg)
                    logger.info("Platform reporting enabled → %s", cfg.platform_url)
            except ImportError:
                if report_usage is True:
                    logger.warning(
                        "Platform reporting requested but httpx is not installed. "
                        'Install with: pip install "rlabs-agentguard[platform]"'
                    )
            except Exception:
                logger.debug("Platform client init failed — reporting disabled", exc_info=True)

        logger.info(
            "Pipeline initialized: archetype=%s, llm=%s",
            self._archetype.name,
            self._llm.provider_name,
        )

    async def generate(
        self,
        spec: str,
        *,
        skip_challenge: bool = False,
        skip_validation: bool = False,
        parallel_l4: bool = True,
        max_challenge_retries: int | None = None,
    ) -> GenerationResult:
        """Run the full generation pipeline.

        Args:
            spec: Natural-language project specification.
            skip_challenge: Skip self-challenge step (faster, less safe).
            skip_validation: Skip structural validation step.
            parallel_l4: Whether to parallelize L4 function generation.
            max_challenge_retries: Override archetype's default retry count.

        Returns:
            GenerationResult with all generated files, trace, and cost info.
        """
        import time as _time

        pipeline_start = _time.perf_counter()

        # Start background flush if platform reporting is active
        if self._platform:
            self._platform.start_background_flush()

        # Start a trace for the full generation
        self._tracer.new_trace(
            archetype=self._archetype.name,
            spec=spec[:200],
        )

        # Create top-down generator
        generator = TopDownGenerator(
            archetype=self._archetype,
            llm=self._llm,
            tracer=self._tracer,
            parallel_logic=parallel_l4,
        )

        # Run the 4-level generation
        gen_start = _time.perf_counter()
        result = await generator.generate(spec)
        gen_duration_ms = int((_time.perf_counter() - gen_start) * 1000)

        # Report generation event to platform
        if self._platform:
            summary = result.trace.summary() if result.trace else None
            await self._platform.track(
                self._platform.build_generation_event(
                    archetype=self._archetype.id,
                    model=f"{self._llm.provider_name}/{getattr(self._llm, 'model', 'unknown')}",
                    input_tokens=summary.total_tokens.prompt_tokens if summary else 0,
                    output_tokens=summary.total_tokens.completion_tokens if summary else 0,
                    cost=float(summary.total_cost.total_cost) if summary else 0.0,
                    duration_ms=gen_duration_ms,
                    success=True,
                    files_count=len(result.files),
                )
            )

        # Structural validation with auto-fix
        if not skip_validation:
            val_start = _time.perf_counter()
            report = await self.validate(result.files)
            val_duration_ms = int((_time.perf_counter() - val_start) * 1000)

            if report.auto_fixed:
                # Apply auto-fixed files back into the result
                fixed_files = self._validator.last_fixed_files or result.files
                result = GenerationResult(
                    skeleton=result.skeleton,
                    contracts=result.contracts,
                    wiring=result.wiring,
                    logic=LogicResult(files=fixed_files, wiring=result.wiring),
                    trace=result.trace,
                    total_cost=result.total_cost,
                )
            if not report.passed:
                logger.warning(
                    "Validation FAILED with %d blocking errors",
                    len(report.blocking_errors),
                )

            # Report validation event
            if self._platform:
                await self._platform.track(
                    self._platform.build_validation_event(
                        archetype=self._archetype.id,
                        duration_ms=val_duration_ms,
                        passed=report.passed,
                        fixes=len(report.auto_fixed) if report.auto_fixed else 0,
                        errors=len(report.blocking_errors),
                    )
                )

        # Self-challenge with archetype criteria
        if not skip_challenge:
            criteria = self._get_challenge_criteria()
            if criteria:
                chal_start = _time.perf_counter()
                challenge_result = await self.challenge(
                    code="\n\n".join(
                        f"# {path}\n{content}"
                        for path, content in result.files.items()
                    ),
                    criteria=criteria,
                    max_retries=max_challenge_retries,
                )
                chal_duration_ms = int((_time.perf_counter() - chal_start) * 1000)

                if challenge_result.rework_output:
                    logger.info(
                        "Self-challenge reworked output (attempt %d)",
                        challenge_result.attempt,
                    )

                # Report challenge event
                if self._platform:
                    await self._platform.track(
                        self._platform.build_challenge_event(
                            archetype=self._archetype.id,
                            model=f"{self._challenge_llm.provider_name}/{getattr(self._challenge_llm, 'model', 'unknown')}",
                            input_tokens=0,  # Challenge doesn't expose token counts directly
                            output_tokens=0,
                            cost=0.0,
                            duration_ms=chal_duration_ms,
                            passed=challenge_result.passed,
                            rework_attempts=getattr(challenge_result, "attempt", 0),
                        )
                    )

        # Finish and persist trace
        self._tracer.finish()

        # Final platform flush
        if self._platform:
            await self._platform.flush()

        pipeline_duration_ms = int((_time.perf_counter() - pipeline_start) * 1000)
        logger.info(
            "Pipeline complete: %d files generated, cost=$%.4f, duration=%dms",
            len(result.files),
            result.total_cost,
            pipeline_duration_ms,
        )
        return result

    # --- Individual steps (composable) ---

    async def skeleton(self, spec: str) -> SkeletonResult:
        """Run L1 only: generate file tree from spec."""
        self._tracer.new_trace(
            archetype=self._archetype.name,
            spec=spec[:200],
        )
        gen = TopDownGenerator(
            archetype=self._archetype,
            llm=self._llm,
            tracer=self._tracer,
        )
        result = await gen.skeleton(spec)
        self._tracer.finish()
        return result

    async def contracts(self, spec: str, skeleton: SkeletonResult) -> ContractsResult:
        """Run L2 only: generate typed stubs from skeleton."""
        gen = TopDownGenerator(
            archetype=self._archetype,
            llm=self._llm,
            tracer=self._tracer,
        )
        return await gen.contracts(spec, skeleton)

    async def wiring(self, contracts: ContractsResult) -> WiringResult:
        """Run L3 only: wire imports and call chains."""
        gen = TopDownGenerator(
            archetype=self._archetype,
            llm=self._llm,
            tracer=self._tracer,
        )
        return await gen.wiring(contracts)

    async def logic(self, wiring: WiringResult) -> LogicResult:
        """Run L4 only: implement function bodies."""
        gen = TopDownGenerator(
            archetype=self._archetype,
            llm=self._llm,
            tracer=self._tracer,
        )
        return await gen.logic(wiring)

    # --- Standalone operations ---

    async def validate(self, code: dict[str, str]) -> ValidationReport:
        """Validate generated code structurally.

        Runs syntax, lint, type, import, and structure checks. Auto-fixes
        trivial issues (formatting, unused imports, trailing newlines).

        Args:
            code: Dict of {file_path: file_content}.

        Returns:
            ValidationReport with pass/fail, check results, and auto-fixes.
        """
        report = self._validator.check(code)
        logger.info("Validation: %s", report)
        return report

    async def challenge(
        self,
        code: str,
        criteria: list[str] | None = None,
        max_retries: int | None = None,
    ) -> ChallengeResult:
        """Run self-challenge on code using the challenge LLM.

        Evaluates the code against acceptance criteria from the archetype
        (or custom criteria) using the LLM as an adversarial reviewer.

        Args:
            code: Source code to challenge.
            criteria: Specific criteria to check (defaults to archetype criteria).
            max_retries: Max rework attempts (defaults to archetype config).

        Returns:
            ChallengeResult with per-criterion verdicts and metadata.
        """
        if criteria is None:
            criteria = self._get_challenge_criteria()
        if not criteria:
            logger.info("Self-challenge: no criteria configured, skipping")
            return ChallengeResult(passed=True)

        retries = max_retries if max_retries is not None else 3
        result = await self._challenger.challenge(
            output=code,
            criteria=criteria,
            task_description=f"Code generation for {self._archetype.name} project",
            max_retries=retries,
        )
        logger.info("Self-challenge: %s", result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_challenge_criteria(self) -> list[str]:
        """Extract challenge criteria from the archetype config."""
        arch = self._archetype
        if hasattr(arch, "self_challenge") and arch.self_challenge:
            criteria = getattr(arch.self_challenge, "criteria", None)
            if criteria:
                return list(criteria)
        return []

    @property
    def tracer(self) -> Tracer:
        """Access the pipeline's tracer for inspection."""
        return self._tracer

    @property
    def archetype(self) -> Archetype:
        """Access the loaded archetype."""
        return self._archetype

    @property
    def platform(self) -> PlatformClient | None:
        """Access the platform client (None if not configured)."""
        return self._platform

    async def close(self) -> None:
        """Clean up resources (flush and close platform client)."""
        if self._platform:
            await self._platform.close()
            self._platform = None
