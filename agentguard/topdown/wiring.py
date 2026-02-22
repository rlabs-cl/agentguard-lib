"""L3 Wiring — wire imports and call chains between files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentguard.llm.types import GenerationConfig
from agentguard.prompts.registry import get_prompt_registry
from agentguard.topdown.types import ContractsResult, WiringResult
from agentguard.tracing.trace import SpanType

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.llm.base import LLMProvider
    from agentguard.tracing.tracer import Tracer

logger = logging.getLogger(__name__)


async def generate_wiring(
    contracts: ContractsResult,
    archetype: Archetype,
    llm: LLMProvider,
    tracer: Tracer,
) -> WiringResult:
    """L3: Wire imports and call chains between files.

    Takes the L2 contracts (typed stubs) and adds correct import statements
    and inter-module call chains. Business logic stays as NotImplementedError.

    Args:
        contracts: L2 contracts result.
        archetype: Project archetype.
        llm: LLM provider.
        tracer: Tracer for recording spans.

    Returns:
        WiringResult with wired code for each file.
    """
    prompt_registry = get_prompt_registry()
    template = prompt_registry.get("wiring")
    files: dict[str, str] = {}

    with tracer.span("L3_wiring", SpanType.LEVEL) as _level_span:
        for file_path, file_code in contracts.files.items():
            # Build context: other files and their contracts
            other_files = [
                {"path": p, "contracts": c}
                for p, c in contracts.files.items()
                if p != file_path
            ]

            messages = template.render(
                language=archetype.tech_stack.language,
                file_path=file_path,
                file_contracts=file_code,
                other_files=other_files,
            )

            with tracer.span(f"llm_wiring_{file_path}", SpanType.LLM_CALL) as llm_span:
                response = await llm.generate(
                    messages,
                    config=GenerationConfig(temperature=0.0, max_tokens=4096),
                )
                tracer.record_llm_response(llm_span, response)

            code = _clean_code_response(response.content)
            files[file_path] = code
            logger.info("L3 wiring: wired imports for %s", file_path)

    logger.info("L3 wiring: %d files wired", len(files))
    return WiringResult(files=files, contracts=contracts)


def _clean_code_response(content: str) -> str:
    """Strip markdown fences and whitespace from LLM code output."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"
