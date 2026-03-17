"""TopDownGenerator — prompt renderer for the 4-level generation pipeline.

This class no longer calls any LLM directly. It renders prompts that the
calling agent uses with its own LLM. Kept for backward compatibility and
as a convenient orchestration layer for prompt rendering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentguard.topdown.contracts import render_contracts_prompt
from agentguard.topdown.logic import render_logic_prompt
from agentguard.topdown.skeleton import render_skeleton_prompt
from agentguard.topdown.wiring import render_wiring_prompt

if TYPE_CHECKING:
    from agentguard.archetypes.base import Archetype
    from agentguard.topdown.types import SkeletonResult

logger = logging.getLogger(__name__)


class TopDownGenerator:
    """Renders prompts for 4-level top-down code generation.

    Levels:
        L1 Skeleton  -- file tree + purposes
        L2 Contracts -- typed stubs per file
        L3 Wiring   -- imports and call chains
        L4 Logic    -- function body implementation

    Each level constrains the context for the next, preventing
    hallucination and ensuring structural coherence.

    Note: This class no longer calls any LLM. It renders prompts that the
    calling agent executes with its own LLM.
    """

    def __init__(self, archetype: Archetype) -> None:
        self._archetype = archetype

    def skeleton_prompt(self, spec: str) -> list[dict[str, str]]:
        """L1: Render skeleton prompt messages."""
        return render_skeleton_prompt(spec, self._archetype)

    def contracts_prompt(
        self, spec: str, skeleton: SkeletonResult,
    ) -> dict[str, list[dict[str, str]]]:
        """L2: Render contract prompts for each file."""
        return render_contracts_prompt(spec, skeleton, self._archetype)

    def wiring_prompt(
        self, contracts_files: dict[str, str],
    ) -> dict[str, list[dict[str, str]]]:
        """L3: Render wiring prompts for each file."""
        return render_wiring_prompt(contracts_files, self._archetype)

    def logic_prompt(
        self,
        file_path: str,
        file_code: str,
        function_name: str,
        dependency_files: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """L4: Render logic prompt for a single function."""
        return render_logic_prompt(
            file_path, file_code, function_name, self._archetype, dependency_files,
        )
