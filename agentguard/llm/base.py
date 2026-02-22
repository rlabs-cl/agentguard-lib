"""Abstract base class for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentguard.llm.types import GenerationConfig, LLMResponse, Message


class LLMProvider(ABC):
    """Abstract base for all LLM providers.

    Every provider must implement `generate()` which takes a list of messages
    and returns a typed LLMResponse with token counts and cost estimates.
    """

    def __init__(self, model: str, default_config: GenerationConfig | None = None) -> None:
        self.model = model
        self.default_config = default_config or GenerationConfig()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g. 'anthropic', 'openai')."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate a completion from the given messages.

        Args:
            messages: Conversation messages (system, user, assistant).
            config: Override default generation config for this call.

        Returns:
            Typed response with content, token usage, and cost.
        """

    def _resolve_config(self, config: GenerationConfig | None) -> GenerationConfig:
        """Merge call-level config with defaults."""
        if config is not None:
            return config
        return self.default_config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
