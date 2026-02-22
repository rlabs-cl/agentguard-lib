"""LLM pricing table and cost computation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model."""

    input_per_1k: Decimal
    output_per_1k: Decimal

    @property
    def input_per_token(self) -> Decimal:
        return self.input_per_1k / Decimal("1000")

    @property
    def output_per_token(self) -> Decimal:
        return self.output_per_1k / Decimal("1000")


# Built-in pricing table — updated periodically.
# Format: "provider/model" → ModelPricing
# Prices in USD per 1K tokens.
PRICING_TABLE: dict[str, ModelPricing] = {
    # Anthropic
    "anthropic/claude-sonnet-4-20250514": ModelPricing(
        input_per_1k=Decimal("0.003"),
        output_per_1k=Decimal("0.015"),
    ),
    "anthropic/claude-haiku-3-20250722": ModelPricing(
        input_per_1k=Decimal("0.0008"),
        output_per_1k=Decimal("0.004"),
    ),
    "anthropic/claude-opus-4-20250514": ModelPricing(
        input_per_1k=Decimal("0.015"),
        output_per_1k=Decimal("0.075"),
    ),
    # OpenAI
    "openai/gpt-4o": ModelPricing(
        input_per_1k=Decimal("0.0025"),
        output_per_1k=Decimal("0.01"),
    ),
    "openai/gpt-4o-mini": ModelPricing(
        input_per_1k=Decimal("0.00015"),
        output_per_1k=Decimal("0.0006"),
    ),
    "openai/o1": ModelPricing(
        input_per_1k=Decimal("0.015"),
        output_per_1k=Decimal("0.06"),
    ),
    "openai/o3-mini": ModelPricing(
        input_per_1k=Decimal("0.0011"),
        output_per_1k=Decimal("0.0044"),
    ),
    # Google
    "google/gemini-2.0-flash": ModelPricing(
        input_per_1k=Decimal("0.0001"),
        output_per_1k=Decimal("0.0004"),
    ),
    "google/gemini-2.5-pro": ModelPricing(
        input_per_1k=Decimal("0.00125"),
        output_per_1k=Decimal("0.01"),
    ),
}

# Fallback pricing for unknown models (conservative estimate).
_FALLBACK_PRICING = ModelPricing(
    input_per_1k=Decimal("0.005"),
    output_per_1k=Decimal("0.015"),
)


def get_model_pricing(model_string: str) -> ModelPricing:
    """Get pricing for a model.

    Args:
        model_string: Full model identifier, e.g. "anthropic/claude-sonnet-4-20250514".

    Returns:
        ModelPricing for the model, or fallback if unknown.
    """
    return PRICING_TABLE.get(model_string, _FALLBACK_PRICING)
