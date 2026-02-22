"""Tracer — records spans during pipeline execution."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentguard.tracing.trace import Span, SpanType, Trace

if TYPE_CHECKING:
    from collections.abc import Generator

    from agentguard.llm.types import LLMResponse


class Tracer:
    """Records all operations during a pipeline run.

    Usage:
        tracer = Tracer(store_dir="./traces")
        trace = tracer.new_trace(archetype="api_backend", spec="...")
        with tracer.span("L1", SpanType.LEVEL) as level_span:
            with tracer.span("llm_call_1", SpanType.LLM_CALL) as llm_span:
                response = await llm.generate(...)
                tracer.record_llm_response(llm_span, response)
        tracer.finish(trace)
    """

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self._store_dir = Path(store_dir) if store_dir else None
        self._current_trace: Trace | None = None
        self._span_stack: list[Span] = []

    def new_trace(self, archetype: str = "", spec: str = "") -> Trace:
        """Start a new trace."""
        trace = Trace(archetype=archetype, spec_summary=spec[:200])
        self._current_trace = trace
        self._span_stack = []
        return trace

    @contextmanager
    def span(self, name: str, span_type: SpanType, **metadata: Any) -> Generator[Span, None, None]:
        """Context manager that creates, starts, and finishes a span.

        Automatically nests under the current parent span.
        """
        parent_id = self._span_stack[-1].id if self._span_stack else None
        s = Span(
            parent_id=parent_id,
            type=span_type,
            name=name,
            metadata=metadata,
        )

        # Add to parent's children or to trace root
        if self._span_stack:
            self._span_stack[-1].children.append(s)
        elif self._current_trace:
            self._current_trace.add_span(s)

        self._span_stack.append(s)
        try:
            yield s
        finally:
            s.finish()
            self._span_stack.pop()

    def record_llm_response(self, span: Span, response: LLMResponse) -> None:
        """Attach LLM response data to a span."""
        span.tokens = response.tokens
        span.cost = response.cost
        span.model = f"{response.provider}/{response.model}"
        span.metadata["latency_ms"] = response.latency_ms
        span.metadata["content_length"] = len(response.content)

    def finish(self, trace: Trace | None = None) -> Trace:
        """Finish and persist the trace.

        Returns the finished trace.
        """
        t = trace or self._current_trace
        if t is None:
            raise RuntimeError("No active trace to finish")
        t.finish()

        if self._store_dir:
            self._persist(t)

        self._current_trace = None
        self._span_stack = []
        return t

    def _persist(self, trace: Trace) -> None:
        """Save trace to a JSON file."""
        if not self._store_dir:
            return
        self._store_dir.mkdir(parents=True, exist_ok=True)
        path = self._store_dir / f"{trace.id}.json"
        path.write_text(json.dumps(trace.to_dict(), indent=2, default=str), encoding="utf-8")

    @property
    def current_trace(self) -> Trace | None:
        return self._current_trace
