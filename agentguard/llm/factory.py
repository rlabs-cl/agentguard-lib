"""Factory for creating LLM providers from string identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentguard.llm.types import GenerationConfig

if TYPE_CHECKING:
    from agentguard.llm.base import LLMProvider


def create_llm_provider(
    model_string: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> LLMProvider:
    """Create an LLM provider from a string like 'anthropic/claude-sonnet-4-20250514'.

    Format: "provider/model-name"

    Supported providers:
        - anthropic/ → AnthropicProvider
        - openai/   → OpenAIProvider
        - google/   → GeminiProvider
        - litellm/  → LiteLLMProvider

    Args:
        model_string: Provider and model, e.g. "anthropic/claude-sonnet-4-20250514".
        api_key: Optional API key (falls back to environment variable).
        temperature: Default temperature for generation.
        max_tokens: Default max tokens for generation.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider is unknown or format is invalid.
    """
    if "/" not in model_string:
        raise ValueError(
            f"Invalid model string '{model_string}'. "
            "Expected format: 'provider/model-name' (e.g. 'anthropic/claude-sonnet-4-20250514')."
        )

    provider, model = model_string.split("/", 1)
    provider = provider.lower().strip()
    model = model.strip()

    config = GenerationConfig(temperature=temperature, max_tokens=max_tokens)

    if provider == "anthropic":
        from agentguard.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model, api_key=api_key, default_config=config)

    if provider == "openai":
        from agentguard.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model, api_key=api_key, default_config=config)

    if provider == "google":
        from agentguard.llm.gemini_provider import GeminiProvider

        return GeminiProvider(model=model, api_key=api_key, default_config=config)

    if provider == "litellm":
        from agentguard.llm.litellm_provider import LiteLLMProvider

        return LiteLLMProvider(model=model, api_key=api_key, default_config=config)

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        f"Supported: anthropic, openai, google, litellm. Got: '{model_string}'."
    )
