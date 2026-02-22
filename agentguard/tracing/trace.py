"""Trace, Span, and TraceSummary dataclasses."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agentguard.llm.types import CostEstimate, TokenUsage


class SpanType(StrEnum):
    """Type of operation recorded in a span."""

    PIPELINE = "pipeline"
    LEVEL = "level"
    LLM_CALL = "llm_call"
    VALIDATION = "validation"
    CHALLENGE = "challenge"
    REWORK = "rework"
    AUTOFIX = "autofix"
    SUMMARIZATION = "summarization"


@dataclass
class Span:
    """One operation within a trace."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    type: SpanType = SpanType.LLM_CALL
    name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    # LLM-specific fields (populated for LLM_CALL spans)
    tokens: TokenUsage | None = None
    cost: CostEstimate | None = None
    model: str | None = None

    def finish(self) -> None:
        """Mark this span as finished."""
        self.ended_at = datetime.now(UTC)

    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        if self.ended_at is None:
            return 0
        delta = self.ended_at - self.started_at
        return int(delta.total_seconds() * 1000)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage."""
        d: dict[str, Any] = {
            "id": self.id,
            "parent_id": self.parent_id,
            "type": self.type.value,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }
        if self.tokens:
            d["tokens"] = {
                "prompt": self.tokens.prompt_tokens,
                "completion": self.tokens.completion_tokens,
                "total": self.tokens.total_tokens,
            }
        if self.cost:
            d["cost"] = {
                "input": str(self.cost.input_cost),
                "output": str(self.cost.output_cost),
                "total": str(self.cost.total_cost),
            }
        if self.model:
            d["model"] = self.model
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class TraceSummary:
    """High-level summary of a trace."""

    total_llm_calls: int = 0
    total_tokens: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    total_cost: CostEstimate = field(default_factory=CostEstimate.zero)
    levels_completed: list[str] = field(default_factory=list)
    structural_fixes: int = 0
    challenge_reworks: int = 0
    grounding_violations: int = 0
    duration_ms: int = 0
    model_breakdown: dict[str, CostEstimate] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.total_llm_calls} LLM calls | "
            f"${self.total_cost.total_cost:.4f} total | "
            f"{self.structural_fixes} structural fixes | "
            f"{self.challenge_reworks} self-challenge reworks"
        )


@dataclass
class Trace:
    """Top-level trace for one pipeline run."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    archetype: str = ""
    spec_summary: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    spans: list[Span] = field(default_factory=list)

    def finish(self) -> None:
        """Mark the trace as finished."""
        self.ended_at = datetime.now(UTC)

    def add_span(self, span: Span) -> None:
        """Add a span to the trace."""
        self.spans.append(span)

    def summary(self) -> TraceSummary:
        """Compute a high-level summary of this trace."""
        all_spans = self._flatten_spans(self.spans)

        llm_spans = [s for s in all_spans if s.type == SpanType.LLM_CALL]
        level_spans = [s for s in all_spans if s.type == SpanType.LEVEL]

        total_tokens = TokenUsage(
            prompt_tokens=sum(s.tokens.prompt_tokens for s in llm_spans if s.tokens),
            completion_tokens=sum(s.tokens.completion_tokens for s in llm_spans if s.tokens),
        )

        total_cost = CostEstimate.zero()
        model_breakdown: dict[str, CostEstimate] = {}
        for s in llm_spans:
            if s.cost:
                total_cost = total_cost + s.cost
                if s.model:
                    if s.model in model_breakdown:
                        model_breakdown[s.model] = model_breakdown[s.model] + s.cost
                    else:
                        model_breakdown[s.model] = s.cost

        return TraceSummary(
            total_llm_calls=len(llm_spans),
            total_tokens=total_tokens,
            total_cost=total_cost,
            levels_completed=[s.name for s in level_spans],
            structural_fixes=sum(
                1 for s in all_spans if s.type == SpanType.AUTOFIX
            ),
            challenge_reworks=sum(
                1 for s in all_spans if s.type == SpanType.REWORK
            ),
            grounding_violations=0,  # Phase 1
            duration_ms=self._total_duration_ms(),
            model_breakdown=model_breakdown,
        )

    @property
    def duration_ms(self) -> int:
        return self._total_duration_ms()

    def _total_duration_ms(self) -> int:
        if self.ended_at is None:
            return 0
        delta = self.ended_at - self.started_at
        return int(delta.total_seconds() * 1000)

    def _flatten_spans(self, spans: list[Span]) -> list[Span]:
        """Recursively flatten span tree."""
        result: list[Span] = []
        for span in spans:
            result.append(span)
            if span.children:
                result.extend(self._flatten_spans(span.children))
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage."""
        return {
            "id": self.id,
            "archetype": self.archetype,
            "spec_summary": self.spec_summary,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "summary": str(self.summary()),
            "spans": [s.to_dict() for s in self.spans],
        }
