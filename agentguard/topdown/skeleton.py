"""L1 Skeleton — render the skeleton prompt for the calling agent."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agentguard.prompts.registry import get_prompt_registry
from agentguard.topdown.types import FileEntry, SkeletonResult

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype

logger = logging.getLogger(__name__)


def render_skeleton_prompt(
    spec: str,
    archetype: Archetype,
) -> list[dict[str, str]]:
    """Render the L1 skeleton prompt messages.

    Returns the rendered messages for the calling agent to use
    with its own LLM. Does not call any LLM itself.

    Args:
        spec: Natural language project specification.
        archetype: Project archetype (configures structure expectations).

    Returns:
        List of message dicts (role/content) for the skeleton prompt.
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

    return messages


def parse_skeleton_response(content: str) -> SkeletonResult:
    """Parse LLM output into a SkeletonResult.

    Handles both clean JSON and JSON wrapped in markdown fences.
    """
    files = _parse_skeleton_response(content)
    logger.info("L1 skeleton: %d files parsed", len(files))
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
