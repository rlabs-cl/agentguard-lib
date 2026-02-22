"""SSE streaming support for long-running generation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from sse_starlette.sse import EventSourceResponse

from agentguard.server.schemas import (
    SSECompleteEvent,
    SSELevelEvent,
    SSEValidationEvent,
    TraceSummaryResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from starlette.requests import Request

logger = logging.getLogger(__name__)


async def generate_sse_stream(
    request: Request,
    pipeline: Any,
    spec: str,
    *,
    skip_challenge: bool = False,
    skip_validation: bool = False,
    parallel_l4: bool = True,
    max_challenge_retries: int = 3,
) -> EventSourceResponse:
    """Run the generation pipeline and stream level-by-level SSE events.

    Returns an ``EventSourceResponse`` that the client can consume as
    a standard Server-Sent Events stream.
    """

    async def _event_generator() -> AsyncGenerator[dict[str, str], None]:
        try:
            # Import here to avoid circular imports
            from agentguard.topdown.generator import TopDownGenerator

            generator = TopDownGenerator(
                archetype=pipeline._archetype,
                llm=pipeline._llm,
                tracer=pipeline._tracer,
                parallel_logic=parallel_l4,
            )

            pipeline._tracer.new_trace(
                archetype=pipeline._archetype.name,
                spec=spec[:200],
            )

            # L1: Skeleton
            skeleton = await generator.skeleton(spec)
            yield {
                "event": "level_complete",
                "data": SSELevelEvent(
                    level="skeleton",
                    files=[f.path for f in skeleton.files],
                ).model_dump_json(),
            }

            # L2: Contracts
            contracts = await generator.contracts(spec, skeleton)
            yield {
                "event": "level_complete",
                "data": SSELevelEvent(
                    level="contracts",
                    files=list(contracts.files.keys()),
                ).model_dump_json(),
            }

            # L3: Wiring
            wiring = await generator.wiring(contracts)
            yield {
                "event": "level_complete",
                "data": SSELevelEvent(
                    level="wiring",
                    files=list(wiring.files.keys()),
                ).model_dump_json(),
            }

            # L4: Logic
            logic = await generator.logic(wiring)
            yield {
                "event": "level_complete",
                "data": SSELevelEvent(
                    level="logic",
                    files=list(logic.files.keys()),
                ).model_dump_json(),
            }

            # Validation
            if not skip_validation:
                report = pipeline._validator.check(logic.files)
                yield {
                    "event": "validation",
                    "data": SSEValidationEvent(
                        passed=report.passed,
                        errors=len(report.errors),
                        fixes=len(report.auto_fixed),
                    ).model_dump_json(),
                }

            # Final complete event
            pipeline._tracer.finish()

            yield {
                "event": "complete",
                "data": SSECompleteEvent(
                    files=logic.files,
                    trace=TraceSummaryResponse(
                        levels_completed=["skeleton", "contracts", "wiring", "logic"],
                    ),
                ).model_dump_json(),
            }

        except Exception as exc:
            logger.exception("SSE generation failed")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}),
            }

    return EventSourceResponse(_event_generator())
