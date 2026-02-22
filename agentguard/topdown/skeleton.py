"""L1 Skeleton — generate the file tree with responsibilities."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agentguard.llm.types import GenerationConfig
from agentguard.prompts.registry import get_prompt_registry
from agentguard.topdown.types import FileEntry, SkeletonResult
from agentguard.tracing.trace import SpanType

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.llm.base import LLMProvider
    from agentguard.tracing.tracer import Tracer

logger = logging.getLogger(__name__)


async def generate_skeleton(
    spec: str,
    archetype: Archetype,
    llm: LLMProvider,
    tracer: Tracer,
) -> SkeletonResult:
    """L1: Generate file tree with one-line responsibilities.

    Args:
        spec: Natural language project specification.
        archetype: Project archetype (configures structure expectations).
        llm: LLM provider to use.
        tracer: Tracer for recording spans.

    Returns:
        SkeletonResult with list of FileEntry objects.
    """
    prompt_registry = get_prompt_registry()
    template = prompt_registry.get("skeleton")

    messages = template.render(
        spec=spec,
        archetype_name=archetype.name,
        language=archetype.tech_stack.language,
        framework=archetype.tech_stack.framework,
        expected_structure=archetype.get_expected_structure_text(),
    )

    with tracer.span("L1_skeleton", SpanType.LEVEL) as _level_span, tracer.span("llm_skeleton", SpanType.LLM_CALL) as llm_span:
        response = await llm.generate(
            messages,
            config=GenerationConfig(temperature=0.0, max_tokens=4096),
        )
        tracer.record_llm_response(llm_span, response)

    # Parse the JSON response into FileEntry objects
    files = _parse_skeleton_response(response.content)

    logger.info("L1 skeleton: %d files generated", len(files))
    return SkeletonResult(files=files)


def _parse_skeleton_response(content: str) -> list[FileEntry]:
    """Parse LLM output into a list of FileEntry objects.

    Handles both clean JSON and JSON wrapped in markdown fences.
    """
    text = content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse skeleton JSON: %s. Raw: %s", e, text[:200])
        # Fallback: try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            data = json.loads(text[start : end + 1])
        else:
            raise ValueError(f"Could not parse skeleton response as JSON: {text[:200]}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array from skeleton, got: {type(data)}")

    return [
        FileEntry(
            path=item.get("path", item.get("file", "")) or "",
            purpose=item.get("purpose", item.get("description", "")) or "",
        )
        for item in data
        if isinstance(item, dict)
    ]
