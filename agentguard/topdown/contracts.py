"""L2 Contracts — generate typed stubs for each file."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentguard.llm.types import GenerationConfig
from agentguard.prompts.registry import get_prompt_registry
from agentguard.topdown.types import ContractsResult, SkeletonResult
from agentguard.tracing.trace import SpanType

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.llm.base import LLMProvider
    from agentguard.tracing.tracer import Tracer

logger = logging.getLogger(__name__)


async def generate_contracts(
    spec: str,
    skeleton: SkeletonResult,
    archetype: Archetype,
    llm: LLMProvider,
    tracer: Tracer,
) -> ContractsResult:
    """L2: Generate typed function/class stubs for each file.

    For each file in the skeleton, asks the LLM to create parseable code
    with typed signatures and `raise NotImplementedError` bodies.

    Args:
        spec: Natural language project specification.
        skeleton: L1 skeleton result.
        archetype: Project archetype.
        llm: LLM provider.
        tracer: Tracer for recording spans.

    Returns:
        ContractsResult with code stubs for each file.
    """
    prompt_registry = get_prompt_registry()
    template = prompt_registry.get("contracts")
    files: dict[str, str] = {}

    with tracer.span("L2_contracts", SpanType.LEVEL) as _level_span:
        for entry in skeleton.files:
            # Skip non-code files
            if not _is_code_file(entry.path):
                continue

            messages = template.render(
                language=archetype.tech_stack.language,
                file_path=entry.path,
                spec=spec,
                file_purpose=entry.purpose,
                skeleton_files=skeleton.files,
                reference_patterns="",  # TODO: load from archetype
            )

            with tracer.span(f"llm_contracts_{entry.path}", SpanType.LLM_CALL) as llm_span:
                response = await llm.generate(
                    messages,
                    config=GenerationConfig(temperature=0.0, max_tokens=4096),
                )
                tracer.record_llm_response(llm_span, response)

            code = _clean_code_response(response.content)
            files[entry.path] = code
            logger.info("L2 contracts: generated stubs for %s", entry.path)

    logger.info("L2 contracts: %d code files generated", len(files))
    return ContractsResult(files=files, skeleton=skeleton)


def _is_code_file(path: str) -> bool:
    """Check if a file path is a code file (vs config/docs)."""
    code_extensions = {".py", ".ts", ".js", ".go", ".rs", ".java", ".kt"}
    return any(path.endswith(ext) for ext in code_extensions)


def _clean_code_response(content: str) -> str:
    """Strip markdown fences and whitespace from LLM code output."""
    text = content.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```python or similar)
        if lines:
            lines = lines[1:]
        # Remove last line if it's a closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    return text.strip() + "\n"
