"""TopDownGenerator — orchestrates the 4-level generation pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentguard.llm.types import CostEstimate
from agentguard.topdown.contracts import generate_contracts
from agentguard.topdown.logic import generate_logic
from agentguard.topdown.skeleton import generate_skeleton
from agentguard.topdown.types import (
    ContractsResult,
    GenerationResult,
    LogicResult,
    SkeletonResult,
    WiringResult,
)
from agentguard.topdown.wiring import generate_wiring

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.llm.base import LLMProvider
    from agentguard.tracing.tracer import Tracer

logger = logging.getLogger(__name__)


class TopDownGenerator:
    """Orchestrates 4-level top-down code generation.

    Levels:
        L1 Skeleton  — file tree + purposes
        L2 Contracts — typed stubs per file
        L3 Wiring   — imports and call chains
        L4 Logic    — function body implementation

    Each level constrains the context for the next, preventing
    hallucination and ensuring structural coherence.
    """

    def __init__(
        self,
        archetype: Archetype,
        llm: LLMProvider,
        tracer: Tracer,
        parallel_logic: bool = True,
    ) -> None:
        self._archetype = archetype
        self._llm = llm
        self._tracer = tracer
        self._parallel_logic = parallel_logic

    async def generate(self, spec: str) -> GenerationResult:
        """Run all 4 levels sequentially and return the full result.

        Args:
            spec: Natural-language project specification.

        Returns:
            GenerationResult with skeleton through logic, plus trace and cost.
        """
        logger.info("Starting top-down generation for archetype=%s", self._archetype.name)

        skeleton = await self.skeleton(spec)
        contracts = await self.contracts(spec, skeleton)
        wiring = await self.wiring(contracts)
        logic = await self.logic(wiring)

        # Aggregate total cost
        trace = self._tracer.current_trace
        total_cost = trace.summary().total_cost if trace else CostEstimate.zero()

        result = GenerationResult(
            skeleton=skeleton,
            contracts=contracts,
            wiring=wiring,
            logic=logic,
            trace=trace,
            total_cost=total_cost,
            validation_fixes=0,
            challenge_reworks=0,
        )

        logger.info(
            "Top-down generation complete: %d files, total_cost=$%.4f",
            len(result.files),
            total_cost,
        )
        return result

    async def skeleton(self, spec: str) -> SkeletonResult:
        """L1: Generate file tree."""
        logger.info("L1: Generating skeleton…")
        return await generate_skeleton(spec, self._archetype, self._llm, self._tracer)

    async def contracts(self, spec: str, skeleton: SkeletonResult) -> ContractsResult:
        """L2: Generate typed stubs for each file."""
        logger.info("L2: Generating contracts…")
        return await generate_contracts(spec, skeleton, self._archetype, self._llm, self._tracer)

    async def wiring(self, contracts: ContractsResult) -> WiringResult:
        """L3: Wire imports and call chains."""
        logger.info("L3: Generating wiring…")
        return await generate_wiring(contracts, self._archetype, self._llm, self._tracer)

    async def logic(self, wiring: WiringResult) -> LogicResult:
        """L4: Implement function bodies."""
        logger.info("L4: Generating logic…")
        return await generate_logic(
            wiring, self._archetype, self._llm, self._tracer, parallel=self._parallel_logic
        )
