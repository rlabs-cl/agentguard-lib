"""Token window — lightweight token counting and budget management.

Uses a simple heuristic by default (1 token ≈ 4 characters) which is
accurate enough for budget management.  Can be upgraded to use tiktoken
or a provider-specific tokenizer if precision is required.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Average characters per token for common models.
# GPT-family ≈ 4.0, Claude ≈ 3.5–4.0.  We use 4.0 as a safe default.
_DEFAULT_CHARS_PER_TOKEN = 4.0


class TokenWindow:
    """Lightweight token counter and budget manager.

    Usage::

        window = TokenWindow()
        tokens = window.count_tokens("Hello world")     # → 3
        fits = window.fits("some long text…", budget=100)  # → True/False
        trimmed = window.trim("some long text…", max_tokens=50)
    """

    def __init__(
        self,
        chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
        tokenizer: object | None = None,
    ) -> None:
        """Initialize the token window.

        Args:
            chars_per_token: Characters-per-token ratio for heuristic counting.
            tokenizer: Optional precise tokenizer object with an ``encode(text)``
                       method that returns a list of token ids.  If provided,
                       exact counts are used instead of the heuristic.
        """
        self._chars_per_token = chars_per_token
        self._tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        Uses the precise tokenizer if available, otherwise falls back
        to a character-based heuristic.
        """
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Tokenizer failed, falling back to heuristic")
        return max(1, math.ceil(len(text) / self._chars_per_token))

    def fits(self, text: str, budget: int) -> bool:
        """Check whether *text* fits within *budget* tokens."""
        return self.count_tokens(text) <= budget

    def trim(self, text: str, max_tokens: int, suffix: str = "\n[…trimmed]") -> str:
        """Trim *text* to fit within *max_tokens*.

        If the text already fits, returns it unchanged.  Otherwise
        truncates at a character boundary and appends *suffix*.
        """
        if self.fits(text, max_tokens):
            return text

        # Reserve space for suffix
        suffix_tokens = self.count_tokens(suffix)
        target_tokens = max(max_tokens - suffix_tokens, 1)
        target_chars = int(target_tokens * self._chars_per_token)

        return text[:target_chars] + suffix

    def remaining(self, used: int, budget: int) -> int:
        """Calculate remaining tokens in a budget."""
        return max(budget - used, 0)

    def split_budget(
        self,
        names: list[str],
        weights: dict[str, float] | None = None,
        total_budget: int = 5000,
    ) -> dict[str, int]:
        """Split a token budget across named items by weight.

        Args:
            names: Names of context items.
            weights: Weight per item (defaults to equal).
            total_budget: Total tokens to distribute.

        Returns:
            Dict of {name: allocated_tokens}.
        """
        if not names:
            return {}

        if weights is None:
            weights = {n: 1.0 for n in names}

        total_weight = sum(weights.get(n, 1.0) for n in names)
        if total_weight == 0:
            total_weight = 1.0

        allocations: dict[str, int] = {}
        for name in names:
            w = weights.get(name, 1.0)
            allocations[name] = max(1, int(total_budget * w / total_weight))

        return allocations
