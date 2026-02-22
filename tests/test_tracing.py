"""Tests for the tracing module."""

from __future__ import annotations

import json
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from agentguard.llm.types import CostEstimate, LLMResponse, TokenUsage
from agentguard.tracing.cost import PRICING_TABLE, get_model_pricing
from agentguard.tracing.trace import Span, SpanType, Trace, TraceSummary
from agentguard.tracing.tracer import Tracer


class TestCostTable:
    def test_known_model(self):
        pricing = get_model_pricing("anthropic/claude-sonnet-4-20250514")
        assert pricing.input_per_1k > 0
        assert pricing.output_per_1k > 0

    def test_unknown_model_returns_fallback(self):
        pricing = get_model_pricing("unknown/model-xyz")
        assert pricing.input_per_1k > 0  # Should return fallback pricing

    def test_pricing_table_has_entries(self):
        assert len(PRICING_TABLE) > 0


class TestSpan:
    def test_create_span(self):
        span = Span(type=SpanType.LLM_CALL, name="test")
        assert span.type == SpanType.LLM_CALL
        assert span.name == "test"
        assert span.id  # UUID is generated

    def test_to_dict(self):
        span = Span(type=SpanType.PIPELINE, name="test_pipeline")
        d = span.to_dict()
        assert d["name"] == "test_pipeline"
        assert d["type"] == "pipeline"

    def test_nested_children(self):
        parent = Span(type=SpanType.PIPELINE, name="parent")
        child = Span(type=SpanType.LEVEL, name="child", parent_id=parent.id)
        parent.children.append(child)
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "child"


class TestTrace:
    def test_create_trace(self):
        trace = Trace(archetype="test", spec_summary="test spec")
        assert trace.archetype == "test"
        assert trace.id  # UUID is generated

    def test_summary(self):
        trace = Trace(archetype="test", spec_summary="test spec")
        summary = trace.summary()
        assert isinstance(summary, TraceSummary)
        assert summary.total_llm_calls == 0
        assert float(summary.total_cost.total_cost) == 0.0

    def test_to_dict(self):
        trace = Trace(archetype="test", spec_summary="hello")
        d = trace.to_dict()
        assert d["archetype"] == "test"
        assert "spans" in d


class TestTracer:
    def test_new_trace(self):
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="hello")
        assert tracer.current_trace is not None
        assert tracer.current_trace.archetype == "test"

    def test_span_context_manager(self):
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="hello")
        with tracer.span("outer", SpanType.PIPELINE) as outer:
            assert outer.name == "outer"
            with tracer.span("inner", SpanType.LEVEL) as inner:
                assert inner.parent_id == outer.id

    def test_record_llm_response(self):
        tracer = Tracer()
        tracer.new_trace(archetype="test", spec="hello")
        with tracer.span("llm_call", SpanType.LLM_CALL) as span:
            response = LLMResponse(
                content="hello",
                model="test-model",
                provider="mock",
                tokens=TokenUsage(prompt_tokens=100, completion_tokens=50),
                cost=CostEstimate(input_cost=Decimal("0.001"), output_cost=Decimal("0.002")),
                latency_ms=25.0,
            )
            tracer.record_llm_response(span, response)
        assert span.model == "mock/test-model"
        assert span.tokens.total_tokens == 150
        assert float(span.cost.total_cost) == pytest.approx(0.003)

    def test_finish_persists_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = Tracer(store_dir=Path(tmpdir))
            tracer.new_trace(archetype="test", spec="hello")
            tracer.finish()

            # Should have written a JSON file
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            data = json.loads(files[0].read_text())
            assert data["archetype"] == "test"
