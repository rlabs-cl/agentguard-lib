"""LiteLLM universal LLM provider.

Wraps the ``litellm`` library which supports 100+ LLM providers through a
unified interface. This acts as a catch-all for providers not directly
supported by AgentGuard.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from agentguard.llm.base import LLMProvider
from agentguard.llm.types import (
    CostEstimate,
    GenerationConfig,
    LLMResponse,
    Message,
    TokenUsage,
)
from agentguard.tracing.cost import get_model_pricing


class LiteLLMProvider(LLMProvider):
    """Universal LLM provider using LiteLLM.

    Supports 100+ LLMs through a single interface::

        pip install litellm

    Model strings follow LiteLLM conventions:
        - ``"gpt-4o"`` (OpenAI)
        - ``"claude-sonnet-4-20250514"`` (Anthropic)
        - ``"gemini/gemini-2.0-flash"`` (Google)
        - ``"bedrock/anthropic.claude-3"`` (AWS Bedrock)
        - etc.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        default_config: GenerationConfig | None = None,
    ) -> None:
        super().__init__(model=model, default_config=default_config)
        try:
            import litellm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "litellm is required for the LiteLLM provider. "
                'Install with: pip install "agentguard[litellm]"'
            ) from exc
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "litellm"

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        import litellm

        cfg = self._resolve_config(config)

        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        if cfg.stop_sequences:
            kwargs["stop"] = cfg.stop_sequences
        if self._api_key:
            kwargs["api_key"] = self._api_key

        start = time.perf_counter_ns()
        response = await litellm.acompletion(**kwargs)
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        content = response.choices[0].message.content or ""

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        tokens = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Try litellm's cost calculation first, fall back to our pricing table
        try:
            total_cost = litellm.completion_cost(completion_response=response)
            cost = CostEstimate(
                input_cost=Decimal(str(total_cost / 2)),  # rough split
                output_cost=Decimal(str(total_cost / 2)),
            )
        except Exception:
            pricing = get_model_pricing(f"litellm/{self.model}")
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
