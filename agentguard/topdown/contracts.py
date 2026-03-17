"""L2 Contracts — render contract prompts for the calling agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentguard.prompts.registry import get_prompt_registry

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.topdown.types import SkeletonResult

logger = logging.getLogger(__name__)


def render_contracts_prompt(
    spec: str,
    skeleton: SkeletonResult,
    archetype: Archetype,
) -> dict[str, list[Any]]:
    """Render L2 contract prompts for each code file.

    Returns a dict mapping file path to rendered messages for the calling
    agent to use with its own LLM. Does not call any LLM itself.

    Args:
        spec: Natural language project specification.
        skeleton: L1 skeleton result.
        archetype: Project archetype.

    Returns:
        Dict of {file_path: messages} for each code file.
    """
    prompt_registry = get_prompt_registry()
    template = prompt_registry.get("contracts")
    prompts: dict[str, list[Any]] = {}

    for entry in skeleton.files:
        if not _is_code_file(entry.path):
            continue

        messages = template.render(
            language=archetype.tech_stack.language,
            file_path=entry.path,
            spec=spec,
            file_purpose=entry.purpose,
            skeleton_files=skeleton.files,
            reference_patterns="",
        )
        prompts[entry.path] = messages

    logger.info("L2 contracts: rendered prompts for %d code files", len(prompts))
    return prompts


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
