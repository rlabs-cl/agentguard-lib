"""conftest.py — shared test fixtures for AgentGuard."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentguard.archetypes.base import Archetype
from agentguard.llm.base import LLMProvider
from agentguard.llm.types import (
    CostEstimate,
    GenerationConfig,
    LLMResponse,
    Message,
    TokenUsage,
)


class MockLLMProvider(LLMProvider):
    """A mock LLM provider for testing.

    Stores a queue of responses. Each call to generate() pops the next response.
    If no responses are queued, returns a default response.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(model="test-model", default_config=GenerationConfig())
        self._responses: list[str] = list(responses) if responses else []
        self._calls: list[list[Message]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(self, messages: list[Message], config: GenerationConfig | None = None) -> LLMResponse:
        self._calls.append(messages)
        content = self._responses.pop(0) if self._responses else "mock response"
        return LLMResponse(
            content=content,
            model="test-model",
            provider="mock",
            tokens=TokenUsage(prompt_tokens=100, completion_tokens=50),
            cost=CostEstimate(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
            latency_ms=10.0,
        )

    @property
    def calls(self) -> list[list[Message]]:
        return self._calls

    @property
    def call_count(self) -> int:
        return len(self._calls)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Provide a mock LLM with no pre-set responses."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_factory():
    """Provide a factory for mock LLMs with specific responses."""
    def _factory(responses: list[str]) -> MockLLMProvider:
        return MockLLMProvider(responses=responses)
    return _factory


@pytest.fixture
def api_backend_archetype() -> Archetype:
    """Load the builtin api_backend archetype."""
    return Archetype.load("api_backend")
