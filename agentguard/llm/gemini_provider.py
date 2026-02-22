"""Google Gemini LLM provider."""

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


class GeminiProvider(LLMProvider):
    """LLM provider for Google Gemini models.

    Requires the ``google-genai`` package::

        pip install google-genai

    Uses the Gemini API (``google.genai``).
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        default_config: GenerationConfig | None = None,
    ) -> None:
        super().__init__(model=model, default_config=default_config)
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Gemini provider. "
                'Install with: pip install "agentguard[google]"'
            ) from exc

        self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "google"

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        cfg = self._resolve_config(config)

        # Build system instruction and contents
        system_text = ""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        gen_config: dict[str, Any] = {
            "max_output_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        if cfg.stop_sequences:
            gen_config["stop_sequences"] = cfg.stop_sequences

        start = time.perf_counter_ns()

        # Use the synchronous client in a thread to keep the interface async
        import asyncio

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self.model,
            contents=contents,
            config={
                "system_instruction": system_text if system_text else None,
                "generation_config": gen_config,
            },
        )
        latency_ms = (time.perf_counter_ns() - start) // 1_000_000

        content = response.text or ""

        # Extract token usage from response metadata
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )

        tokens = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        pricing = get_model_pricing(f"google/{self.model}")
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
