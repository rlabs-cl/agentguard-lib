"""Tests for the context module — recipes, token window, summarizer."""

from __future__ import annotations

import pytest

from agentguard.context.recipe import (
    BUILTIN_RECIPES,
    ContextBundle,
    ContextEngine,
    ContextRecipe,
)
from agentguard.context.summarizer import HierarchicalSummarizer
from agentguard.context.window import TokenWindow
from tests.conftest import MockLLMProvider

# ------------------------------------------------------------------ #
#  TokenWindow
# ------------------------------------------------------------------ #

class TestTokenWindow:
    def test_count_tokens_heuristic(self):
        window = TokenWindow()
        tokens = window.count_tokens("Hello world")
        assert isinstance(tokens, int)
        assert tokens >= 1

    def test_count_tokens_scales_with_length(self):
        window = TokenWindow()
        short = window.count_tokens("Hi")
        long = window.count_tokens("A" * 400)
        assert long > short

    def test_fits_within_budget(self):
        window = TokenWindow()
        assert window.fits("short text", budget=100) is True
        assert window.fits("A" * 10000, budget=10) is False

    def test_trim(self):
        window = TokenWindow()
        long_text = "A" * 10000
        trimmed = window.trim(long_text, max_tokens=100)
        assert len(trimmed) < len(long_text)
        assert "trimmed" in trimmed.lower()

    def test_trim_already_fits(self):
        window = TokenWindow()
        text = "short"
        trimmed = window.trim(text, max_tokens=100)
        assert trimmed == text

    def test_remaining(self):
        window = TokenWindow()
        assert window.remaining(30, 100) == 70
        assert window.remaining(150, 100) == 0

    def test_split_budget_equal(self):
        window = TokenWindow()
        budget = window.split_budget(["a", "b", "c"], total_budget=300)
        assert sum(budget.values()) <= 300
        assert all(v >= 1 for v in budget.values())

    def test_split_budget_weighted(self):
        window = TokenWindow()
        budget = window.split_budget(
            ["a", "b"],
            weights={"a": 3.0, "b": 1.0},
            total_budget=400,
        )
        assert budget["a"] > budget["b"]

    def test_split_budget_empty(self):
        window = TokenWindow()
        assert window.split_budget([], total_budget=100) == {}


# ------------------------------------------------------------------ #
#  ContextRecipe
# ------------------------------------------------------------------ #

class TestContextRecipe:
    def test_create_recipe(self):
        recipe = ContextRecipe(
            id="test",
            include=["spec", "skeleton"],
            max_tokens=3000,
        )
        assert recipe.id == "test"
        assert recipe.max_tokens == 3000
        # Priority defaults to include order
        assert recipe.priority == ["spec", "skeleton"]

    def test_builtin_recipes_exist(self):
        assert "l1_skeleton" in BUILTIN_RECIPES
        assert "l2_contracts" in BUILTIN_RECIPES
        assert "l3_wiring" in BUILTIN_RECIPES
        assert "l4_logic" in BUILTIN_RECIPES
        assert "challenge" in BUILTIN_RECIPES

    def test_l1_recipe_config(self):
        r = BUILTIN_RECIPES["l1_skeleton"]
        assert "spec" in r.include
        assert r.max_tokens == 3000


# ------------------------------------------------------------------ #
#  ContextBundle
# ------------------------------------------------------------------ #

class TestContextBundle:
    def test_as_text(self):
        bundle = ContextBundle(
            items={"spec": "Build an API", "skeleton": "main.py\n"},
            token_count=100,
            token_budget=3000,
            was_summarized=[],
            recipe_id="test",
        )
        text = bundle.as_text()
        assert "spec" in text
        assert "Build an API" in text
        assert "skeleton" in text

    def test_str(self):
        bundle = ContextBundle(
            items={"spec": "test"},
            token_count=50,
            token_budget=3000,
            was_summarized=[],
            recipe_id="l1",
        )
        s = str(bundle)
        assert "50/3000" in s
        assert "l1" in s

    def test_str_with_summarized(self):
        bundle = ContextBundle(
            items={"spec": "test"},
            token_count=50,
            token_budget=3000,
            was_summarized=["patterns"],
            recipe_id="l2",
        )
        s = str(bundle)
        assert "patterns" in s


# ------------------------------------------------------------------ #
#  ContextEngine
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
class TestContextEngine:
    async def test_assemble_basic(self):
        engine = ContextEngine()
        bundle = await engine.assemble(
            recipe="l1_skeleton",
            items={
                "spec": "Build a user auth API with JWT",
                "archetype_structure": "main.py\nroutes/\nmodels/\n",
            },
        )
        assert bundle.recipe_id == "l1_skeleton"
        assert "spec" in bundle.items
        assert bundle.token_count > 0

    async def test_assemble_with_recipe_object(self):
        engine = ContextEngine()
        recipe = ContextRecipe(
            id="custom",
            include=["a", "b"],
            max_tokens=1000,
        )
        bundle = await engine.assemble(
            recipe=recipe,
            items={"a": "content a", "b": "content b"},
        )
        assert bundle.recipe_id == "custom"
        assert len(bundle.items) == 2

    async def test_missing_items_skipped(self):
        engine = ContextEngine()
        recipe = ContextRecipe(id="test", include=["a", "missing"], max_tokens=1000)
        bundle = await engine.assemble(
            recipe=recipe,
            items={"a": "present"},
        )
        assert "a" in bundle.items
        assert "missing" not in bundle.items

    async def test_over_budget_truncation(self):
        """When over budget and no summarizer, items should be truncated."""
        engine = ContextEngine()
        recipe = ContextRecipe(
            id="small",
            include=["big", "small"],
            max_tokens=200,
            summarize_if_over=True,
            priority=["small", "big"],  # small is higher priority
        )
        bundle = await engine.assemble(
            recipe=recipe,
            items={
                "big": "A" * 5000,
                "small": "tiny",
            },
        )
        # The big item should have been truncated significantly
        assert len(bundle.items["big"]) < 5000
        assert len(bundle.was_summarized) >= 1
        assert "truncated" in bundle.items["big"].lower()

    async def test_unknown_recipe_raises(self):
        engine = ContextEngine()
        with pytest.raises(KeyError):
            await engine.assemble(
                recipe="nonexistent_recipe_xyz",
                items={},
            )

    async def test_get_recipe(self):
        engine = ContextEngine()
        recipe = engine.get_recipe("l1_skeleton")
        assert recipe.id == "l1_skeleton"

    async def test_excluded_items(self):
        engine = ContextEngine()
        recipe = ContextRecipe(
            id="test",
            include=["a", "b"],
            exclude=["b"],
            max_tokens=1000,
        )
        bundle = await engine.assemble(
            recipe=recipe,
            items={"a": "hello", "b": "excluded"},
        )
        assert "a" in bundle.items
        assert "b" not in bundle.items


# ------------------------------------------------------------------ #
#  HierarchicalSummarizer
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
class TestHierarchicalSummarizer:
    async def test_truncation_fallback(self):
        """Without an LLM, should fall back to truncation."""
        summarizer = HierarchicalSummarizer(llm=None)
        long_text = "A" * 10000
        result = await summarizer.summarize(long_text, target_tokens=50)
        assert len(result) < len(long_text)
        assert "truncation" in result.lower()

    async def test_short_text_unchanged(self):
        summarizer = HierarchicalSummarizer(llm=None)
        short_text = "Hello world"
        result = await summarizer.summarize(short_text, target_tokens=500)
        assert result == short_text

    async def test_with_llm(self):
        """With an LLM, should call it for summarization."""
        llm = MockLLMProvider(responses=["This is a summary of the content."])
        summarizer = HierarchicalSummarizer(llm=llm)
        long_text = "A very long text " * 500
        result = await summarizer.summarize(long_text, target_tokens=50)
        assert result == "This is a summary of the content."
        assert llm.call_count == 1

    async def test_with_focus(self):
        llm = MockLLMProvider(responses=["Auth summary"])
        summarizer = HierarchicalSummarizer(llm=llm)
        result = await summarizer.summarize(
            "Long text about auth and more",
            target_tokens=50,
            focus="authentication",
        )
        assert result == "Auth summary"
        # Check that focus was included in the prompt
        last_call = llm.calls[0]
        user_msg = last_call[-1].content
        assert "authentication" in user_msg
