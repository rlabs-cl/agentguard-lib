"""Core types for the LLM module."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class TokenUsage:
    """Token usage for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class CostEstimate:
    """Cost estimate for a single LLM call."""

    input_cost: Decimal
    output_cost: Decimal
    currency: str = "USD"

    @property
    def total_cost(self) -> Decimal:
        return self.input_cost + self.output_cost

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            currency=self.currency,
        )

    @classmethod
    def zero(cls) -> CostEstimate:
        return cls(input_cost=Decimal("0"), output_cost=Decimal("0"))


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    provider: str
    tokens: TokenUsage
    cost: CostEstimate
    latency_ms: int


@dataclass
class GenerationConfig:
    """Configuration for an LLM generation call."""

    temperature: float = 0.0
    max_tokens: int = 4096
    stop_sequences: list[str] = field(default_factory=list)
    json_mode: bool = False
