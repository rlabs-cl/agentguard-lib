"""Tests for the LLM types module."""

import pytest

from agentguard.llm.types import CostEstimate, LLMResponse, Message, TokenUsage


class TestMessage:
    def test_create(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"


class TestTokenUsage:
    def test_total_tokens(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_zero_tokens(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        assert usage.total_tokens == 0


class TestCostEstimate:
    def test_total_cost(self):
        cost = CostEstimate(input_cost=0.003, output_cost=0.006)
        assert float(cost.total_cost) == pytest.approx(0.009)

    def test_add(self):
        a = CostEstimate(input_cost=0.001, output_cost=0.002)
        b = CostEstimate(input_cost=0.003, output_cost=0.004)
        c = a + b
        assert float(c.input_cost) == pytest.approx(0.004)
        assert float(c.output_cost) == pytest.approx(0.006)

    def test_zero(self):
        z = CostEstimate.zero()
        assert float(z.total_cost) == 0.0


class TestLLMResponse:
    def test_fields(self):
        resp = LLMResponse(
            content="hello",
            model="test/model",
            provider="test",
            tokens=TokenUsage(prompt_tokens=10, completion_tokens=5),
            cost=CostEstimate(input_cost=0.001, output_cost=0.002),
            latency_ms=50.0,
        )
        assert resp.content == "hello"
        assert resp.model == "test/model"
        assert resp.tokens.total_tokens == 15
