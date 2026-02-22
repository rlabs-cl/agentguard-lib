"""Hierarchical summarizer — compress large artifacts to fit token budgets.

Uses a cheap/fast LLM (e.g. Gemini Flash) to produce summaries when
context items exceed their allocated budget.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentguard.llm.types import GenerationConfig, Message

if TYPE_CHECKING:
    from agentguard.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = """\
You are a technical summarizer. Produce a concise summary of the provided \
content, preserving key technical details (function signatures, class names, \
data types, API endpoints). Remove examples, verbose explanations, and \
boilerplate.\
"""

_SUMMARIZE_USER = """\
Summarize the following content to approximately {target_tokens} tokens.
{focus_instruction}

Content:
{content}

Output ONLY the summary.\
"""


class HierarchicalSummarizer:
    """Compress large artifacts to fit token budgets using an LLM.

    Uses a cheap/fast model for summarization to minimise cost.
    Falls back to simple truncation if no LLM is available.

    Usage::

        summarizer = HierarchicalSummarizer(llm=cheap_model)
        short = await summarizer.summarize(
            long_text,
            target_tokens=500,
            focus="authentication flow",
        )
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        config: GenerationConfig | None = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            llm: LLM to use for summarization (should be cheap/fast).
                 If None, falls back to simple truncation.
            config: Generation config overrides for summarization calls.
        """
        self._llm = llm
        self._config = config or GenerationConfig(temperature=0.0, max_tokens=2048)

    async def summarize(
        self,
        content: str,
        target_tokens: int = 500,
        focus: str | None = None,
    ) -> str:
        """Summarize *content* to approximately *target_tokens*.

        Args:
            content: The text to summarize.
            target_tokens: Approximate target length in tokens.
            focus: Optional focus area (e.g. "authentication module").

        Returns:
            Summarized text.
        """
        if self._llm is None:
            # No LLM available — fall back to truncation
            return self._truncate(content, target_tokens)

        focus_instruction = ""
        if focus:
            focus_instruction = f"Focus on: {focus}"

        user_prompt = _SUMMARIZE_USER.format(
            target_tokens=target_tokens,
            focus_instruction=focus_instruction,
            content=content,
        )

        messages = [
            Message(role="system", content=_SUMMARIZE_SYSTEM),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = await self._llm.generate(messages, self._config)
            logger.debug(
                "Summarized %d chars → %d chars (cost: $%s)",
                len(content),
                len(response.content),
                response.cost.total_cost,
            )
            return response.content.strip()
        except Exception:
            logger.warning("LLM summarization failed, falling back to truncation")
            return self._truncate(content, target_tokens)

    @staticmethod
    def _truncate(content: str, target_tokens: int) -> str:
        """Simple truncation fallback (no LLM needed)."""
        # ~4 chars per token
        target_chars = target_tokens * 4
        if len(content) <= target_chars:
            return content
        return content[:target_chars] + "\n\n[... summarized by truncation ...]"
