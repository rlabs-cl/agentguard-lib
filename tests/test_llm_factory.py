"""Tests for the LLM factory module."""

import pytest

from agentguard.llm.anthropic_provider import AnthropicProvider
from agentguard.llm.factory import create_llm_provider
from agentguard.llm.openai_provider import OpenAIProvider


class TestFactory:
    def test_anthropic_creation(self):
        provider = create_llm_provider("anthropic/claude-sonnet-4-20250514", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)

    def test_openai_creation(self):
        provider = create_llm_provider("openai/gpt-4o", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider("unknown/model", api_key="test-key")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid model string"):
            create_llm_provider("no-slash-here", api_key="test-key")

    def test_custom_config(self):
        provider = create_llm_provider(
            "anthropic/claude-sonnet-4-20250514",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2000,
        )
        assert provider.default_config.temperature == 0.5
        assert provider.default_config.max_tokens == 2000
