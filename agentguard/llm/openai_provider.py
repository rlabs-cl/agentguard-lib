"""OpenAI LLM provider."""

from __future__ import annotations

import time
from decimal import Decimal

import openai

from agentguard.llm.base import LLMProvider
from agentguard.llm.types import (
    CostEstimate,
    GenerationConfig,
    LLMResponse,
    Message,
    TokenUsage,
)
from agentguard.tracing.cost import get_model_pricing


class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI models (GPT-4o, o1, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        default_config: GenerationConfig | None = None,
    ) -> None:
        super().__init__(model=model, default_config=default_config)
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        cfg = self._resolve_config(config)

        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        start = time.perf_counter_ns()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop_sequences or None,
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        content = response.choices[0].message.content or ""

        usage = response.usage
        tokens = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        pricing = get_model_pricing(f"openai/{self.model}")
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
