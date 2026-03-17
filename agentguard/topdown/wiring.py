"""L3 Wiring — render wiring prompts for the calling agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentguard.prompts.registry import get_prompt_registry

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype

logger = logging.getLogger(__name__)


def render_wiring_prompt(
    contracts_files: dict[str, str],
    archetype: Archetype,
) -> dict[str, list[dict[str, str]]]:
    """Render L3 wiring prompts for each file.

    Returns a dict mapping file path to rendered messages for the calling
    agent to use with its own LLM. Does not call any LLM itself.

    Args:
        contracts_files: Dict of {file_path: code} from L2 contracts.
        archetype: Project archetype.

    Returns:
        Dict of {file_path: messages} for each file.
    """
    prompt_registry = get_prompt_registry()
    template = prompt_registry.get("wiring")
    prompts: dict[str, list[dict[str, str]]] = {}

    for file_path, file_code in contracts_files.items():
        other_files = [
            {"path": p, "contracts": c}
            for p, c in contracts_files.items()
            if p != file_path
        ]

        messages = template.render(
            language=archetype.tech_stack.language,
            file_path=file_path,
            file_contracts=file_code,
            other_files=other_files,
        )
        prompts[file_path] = messages

    logger.info("L3 wiring: rendered prompts for %d files", len(prompts))
    return prompts


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
