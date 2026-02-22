"""Context recipes — define exactly what the LLM sees for each task.

Context recipes are the core anti-hallucination mechanism: they control
exactly which artifacts are included in an LLM prompt, enforced by a
token budget.  Each generation level has its own recipe specifying
relevant context pieces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentguard.context.window import TokenWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContextRecipe:
    """Defines what goes into the LLM's context for a specific task.

    Attributes:
        id: Unique recipe identifier (e.g. "l1_skeleton").
        include: Names of context items to include (e.g. "spec", "skeleton").
        exclude: Names to explicitly exclude.
        max_tokens: Token budget for the assembled context.
        summarize_if_over: Summarize oversized items instead of truncating.
        priority: Ordered list of item names — higher-priority items are
                  kept in full, lower-priority items are summarized first.
    """

    id: str
    include: list[str]
    exclude: list[str] = field(default_factory=list)
    max_tokens: int = 5000
    summarize_if_over: bool = True
    priority: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.priority:
            self.priority = list(self.include)


@dataclass(frozen=True)
class ContextBundle:
    """Immutable context snapshot delivered to the LLM.

    Attributes:
        items: Named context pieces (e.g. {"spec": "Build an auth API…"}).
        token_count: Actual token count of all items combined.
        token_budget: The budget we aimed for.
        was_summarized: Names of items that were summarized to fit budget.
        recipe_id: Which recipe produced this bundle.
    """

    items: dict[str, str]
    token_count: int
    token_budget: int
    was_summarized: list[str]
    recipe_id: str

    def as_text(self, separator: str = "\n\n---\n\n") -> str:
        """Concatenate all context items into a single string."""
        parts: list[str] = []
        for name, content in self.items.items():
            parts.append(f"### {name}\n{content}")
        return separator.join(parts)

    def __str__(self) -> str:
        summarized = f", summarized: {self.was_summarized}" if self.was_summarized else ""
        return (
            f"ContextBundle(recipe={self.recipe_id!r}, "
            f"tokens={self.token_count}/{self.token_budget}"
            f"{summarized})"
        )


# ---------------------------------------------------------------------------
# Built-in recipes for each generation level
# ---------------------------------------------------------------------------

BUILTIN_RECIPES: dict[str, ContextRecipe] = {
    "l1_skeleton": ContextRecipe(
        id="l1_skeleton",
        include=["spec", "archetype_structure"],
        max_tokens=3000,
        priority=["spec", "archetype_structure"],
    ),
    "l2_contracts": ContextRecipe(
        id="l2_contracts",
        include=["spec", "skeleton", "reference_patterns"],
        max_tokens=6000,
        priority=["skeleton", "spec", "reference_patterns"],
    ),
    "l3_wiring": ContextRecipe(
        id="l3_wiring",
        include=["contracts", "skeleton"],
        max_tokens=8000,
        priority=["contracts", "skeleton"],
    ),
    "l4_logic": ContextRecipe(
        id="l4_logic",
        include=["function_stub", "test_cases", "dependency_signatures", "reference_pattern"],
        max_tokens=5000,
        priority=["function_stub", "test_cases", "dependency_signatures", "reference_pattern"],
    ),
    "challenge": ContextRecipe(
        id="challenge",
        include=["output", "criteria", "context_summary"],
        max_tokens=3000,
        priority=["output", "criteria", "context_summary"],
    ),
}


# ---------------------------------------------------------------------------
# Context Engine
# ---------------------------------------------------------------------------


class ContextEngine:
    """Assembles context bundles for LLM prompts.

    The engine takes a recipe plus raw context items, applies the token budget,
    and optionally summarizes oversized items using an ``HierarchicalSummarizer``.

    Usage::

        engine = ContextEngine()
        bundle = await engine.assemble(
            recipe=BUILTIN_RECIPES["l2_contracts"],
            items={
                "spec": "Build a user auth API with JWT...",
                "skeleton": json.dumps(skeleton_files),
                "reference_patterns": patterns_text,
            },
        )
    """

    def __init__(
        self,
        summarizer: Any | None = None,  # HierarchicalSummarizer (optional)
        window: TokenWindow | None = None,
    ) -> None:
        self._summarizer = summarizer
        self._window = window or TokenWindow()

    async def assemble(
        self,
        recipe: ContextRecipe | str,
        items: dict[str, str],
    ) -> ContextBundle:
        """Assemble a context bundle from raw items using a recipe.

        Args:
            recipe: A ContextRecipe or the id of a builtin recipe.
            items: Dict of named context pieces.

        Returns:
            A ContextBundle constrained to the recipe's token budget.
        """
        if isinstance(recipe, str):
            recipe = self._resolve_recipe(recipe)

        # Filter items: only include what the recipe requests, minus exclusions
        filtered: dict[str, str] = {}
        for name in recipe.include:
            if name in recipe.exclude:
                continue
            if name in items:
                filtered[name] = items[name]
            else:
                logger.debug("Recipe %s requested item %r but not provided", recipe.id, name)

        # Calculate token counts per item
        item_tokens = {
            name: self._window.count_tokens(content)
            for name, content in filtered.items()
        }
        total = sum(item_tokens.values())

        was_summarized: list[str] = []

        # If over budget, trim/summarize from lowest-priority items first
        if total > recipe.max_tokens and recipe.summarize_if_over:
            filtered, was_summarized = await self._fit_to_budget(
                filtered, item_tokens, recipe,
            )
            # Recalculate totals
            item_tokens = {
                name: self._window.count_tokens(content)
                for name, content in filtered.items()
            }
            total = sum(item_tokens.values())

        return ContextBundle(
            items=filtered,
            token_count=total,
            token_budget=recipe.max_tokens,
            was_summarized=was_summarized,
            recipe_id=recipe.id,
        )

    def get_recipe(self, recipe_id: str) -> ContextRecipe:
        """Look up a builtin recipe by id."""
        return self._resolve_recipe(recipe_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_recipe(self, recipe_id: str) -> ContextRecipe:
        if recipe_id in BUILTIN_RECIPES:
            return BUILTIN_RECIPES[recipe_id]
        raise KeyError(f"Unknown recipe: {recipe_id!r}. Available: {list(BUILTIN_RECIPES)}")

    async def _fit_to_budget(
        self,
        items: dict[str, str],
        item_tokens: dict[str, int],
        recipe: ContextRecipe,
    ) -> tuple[dict[str, str], list[str]]:
        """Reduce context to fit within token budget.

        Strategy:
        1. Items are ordered by priority (highest first).
        2. Walk from lowest priority up, summarizing or truncating
           until total fits.

        Returns:
            (trimmed_items, list of summarized item names)
        """
        budget = recipe.max_tokens
        total = sum(item_tokens.values())
        was_summarized: list[str] = []

        # Order by priority (reversed — process lowest priority first)
        priority_order = list(recipe.priority)
        # Items not in priority list go to the end (lowest priority)
        for name in items:
            if name not in priority_order:
                priority_order.append(name)
        priority_order.reverse()

        result = dict(items)

        for name in priority_order:
            if total <= budget:
                break
            if name not in result:
                continue

            content = result[name]
            tokens = item_tokens.get(name, 0)
            excess = total - budget

            if self._summarizer is not None:
                # Summarize to a target that frees enough room
                target_tokens = max(tokens - excess, 1)
                try:
                    summarized = await self._summarizer.summarize(
                        content,
                        target_tokens=target_tokens,
                    )
                    old_tokens = tokens
                    new_tokens = self._window.count_tokens(summarized)
                    result[name] = summarized
                    total -= (old_tokens - new_tokens)
                    was_summarized.append(name)
                    logger.info(
                        "Summarized %r: %d → %d tokens",
                        name, old_tokens, new_tokens,
                    )
                    continue
                except Exception:
                    logger.warning("Summarizer failed for %r, falling back to truncation", name)

            # Fallback: truncate to target length
            target_tokens = max(tokens - excess, 1)
            target_chars = int(target_tokens * self._window._chars_per_token)
            if target_chars < len(content):
                truncated = content[:target_chars] + "\n\n[... truncated to fit token budget ...]"
                old_tokens = tokens
                new_tokens = self._window.count_tokens(truncated)
                result[name] = truncated
                total -= (old_tokens - new_tokens)
                was_summarized.append(name)

        return result, was_summarized
