"""Challenge types — shared dataclasses for self-challenge."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentguard.llm.types import CostEstimate


@dataclass
class CriterionResult:
    """Result for a single challenge criterion."""

    criterion: str
    passed: bool
    explanation: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.criterion}: {self.explanation}"


@dataclass
class ChallengeResult:
    """Result of a self-challenge evaluation."""

    passed: bool
    attempt: int = 1
    criteria_results: list[CriterionResult] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    grounding_violations: list[str] = field(default_factory=list)
    feedback: str | None = None
    rework_output: str | None = None
    cost: CostEstimate = field(default_factory=CostEstimate.zero)

    @property
    def failed_criteria(self) -> list[CriterionResult]:
        return [c for c in self.criteria_results if not c.passed]

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        failed = len(self.failed_criteria)
        total = len(self.criteria_results)
        parts = [f"Challenge: {status} ({total - failed}/{total} criteria passed)"]
        if self.grounding_violations:
            parts.append(f"{len(self.grounding_violations)} grounding violations")
        if self.assumptions:
            parts.append(f"{len(self.assumptions)} assumptions declared")
        return " | ".join(parts)
