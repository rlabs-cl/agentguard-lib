"""Anthropic Claude LLM provider."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import anthropic

from agentguard.llm.base import LLMProvider
from agentguard.llm.types import (
    CostEstimate,
    GenerationConfig,
    LLMResponse,
    Message,
    TokenUsage,
)
from agentguard.tracing.cost import get_model_pricing


class AnthropicProvider(LLMProvider):
    """LLM provider for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        default_config: GenerationConfig | None = None,
    ) -> None:
        super().__init__(model=model, default_config=default_config)
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        cfg = self._resolve_config(config)

        # Anthropic requires system message to be separate
        system_text = ""
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        start = time.perf_counter_ns()
        optional_kwargs: dict[str, Any] = {}
        if system_text:
            optional_kwargs["system"] = system_text
        if cfg.stop_sequences:
            optional_kwargs["stop_sequences"] = cfg.stop_sequences
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            messages=api_messages,  # type: ignore[arg-type]
            **optional_kwargs,
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        tokens = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

        pricing = get_model_pricing(f"anthropic/{self.model}")
        cost = CostEstimate(
            input_cost=Decimal(str(tokens.prompt_tokens)) * pricing.input_per_token,
            output_cost=Decimal(str(tokens.completion_tokens)) * pricing.output_per_token,
        )

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            tokens=tokens,
            cost=cost,
            latency_ms=int(latency_ms),
        )
