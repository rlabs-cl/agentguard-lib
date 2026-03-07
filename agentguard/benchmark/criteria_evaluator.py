"""CriteriaBasedEvaluator — LLM-as-judge for archetype-defined inline criteria.

When an archetype YAML contains a ``benchmark.criteria`` block, this evaluator
is used instead of the AST-based evaluator.  Each criterion is judged by the
LLM using an explicit rubric supplied by the archetype author.

This makes benchmarks meaningful for archetypes that generate non-Python output
(YAML, docs, papers, other archetypes) where AST analysis is irrelevant.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentguard.benchmark.types import BenchmarkCriterion, DimensionScore, ReadinessScore

if TYPE_CHECKING:
    from agentguard.llm.base import LLMProvider

_MAX_CONTENT_CHARS = 4000
_MAX_SPEC_CHARS = 500

_JUDGE_PROMPT = """\
You are evaluating AI-generated content against a specific quality criterion.

ORIGINAL SPECIFICATION:
{spec}

CRITERION: {name}
DESCRIPTION: {description}

RUBRIC (scoring guide):
{rubric}

CONTENT TO EVALUATE:
{content}

Respond with ONLY these two lines:
Score: X/10
Reason: <one sentence explaining the score>
"""


class CriteriaBasedEvaluator:
    """Evaluate generated output against author-defined criteria via LLM judge.

    Each criterion produces one :class:`~agentguard.benchmark.types.DimensionScore`.
    The overall score is the weighted average across active criteria
    (weight > 0).  Both the ``enterprise`` and ``operational`` slots of the
    returned :class:`~agentguard.benchmark.types.RunResult` receive the same
    score, because the enterprise/operational split is meaningless for
    non-code output.
    """

    def __init__(
        self,
        criteria: list[BenchmarkCriterion],
        threshold: float = 0.6,
    ) -> None:
        # Exclude criteria the author marked as inactive (weight == 0)
        self._criteria = [c for c in criteria if c.weight > 0.0]
        self._threshold = threshold

    async def evaluate(
        self,
        spec: str,
        files: dict[str, str],
        llm: LLMProvider,
    ) -> tuple[ReadinessScore, ReadinessScore]:
        """Judge output against each criterion.

        Returns ``(enterprise_score, operational_score)`` — both identical,
        since criteria-based evaluation doesn't distinguish between the two.
        """
        if not self._criteria:
            empty = ReadinessScore(
                category="criteria",
                overall_score=0.0,
                passed=False,
                dimensions=[],
            )
            return empty, empty

        content = _files_to_text(files)
        dimensions: list[DimensionScore] = []

        for criterion in self._criteria:
            dim = await self._judge_criterion(spec, content, criterion, llm)
            dimensions.append(dim)

        total_weight = sum(c.weight for c in self._criteria)
        if total_weight == 0.0:
            overall = 0.0
        else:
            overall = (
                sum(d.score * c.weight for d, c in zip(dimensions, self._criteria, strict=False))
                / total_weight
            )

        passed = overall >= self._threshold
        readiness = ReadinessScore(
            category="criteria",
            overall_score=overall,
            passed=passed,
            dimensions=dimensions,
        )
        return readiness, readiness

    async def _judge_criterion(
        self,
        spec: str,
        content: str,
        criterion: BenchmarkCriterion,
        llm: LLMProvider,
    ) -> DimensionScore:
        from agentguard.llm.types import Message

        prompt = _JUDGE_PROMPT.format(
            spec=spec[:_MAX_SPEC_CHARS],
            name=criterion.name,
            description=criterion.description,
            rubric=criterion.rubric,
            content=content[:_MAX_CONTENT_CHARS],
        )
        try:
            response = await llm.generate(
                messages=[Message(role="user", content=prompt)],
            )
            score, reason = _parse_judge_response(response.content)
        except Exception as exc:
            score = 0.0
            reason = f"Judge call failed: {exc}"

        return DimensionScore(
            dimension=criterion.name,
            score=score,
            passed=score >= self._threshold,
            findings=[reason],
        )


# ── Helpers ───────────────────────────────────────────────────────


def _files_to_text(files: dict[str, str]) -> str:
    """Concatenate generated files into a single readable block."""
    parts: list[str] = []
    for path, content in files.items():
        parts.append(f"=== {path} ===\n{content}")
    return "\n\n".join(parts)


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract (score 0–1, reason) from the LLM judge response."""
    score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
    reason_match = re.search(r"Reason:\s*(.+)", text, re.IGNORECASE)

    if score_match:
        raw = float(score_match.group(1))
        score = max(0.0, min(1.0, raw / 10.0))
    else:
        score = 0.5  # default when judge response is malformed

    reason = reason_match.group(1).strip() if reason_match else "No reason provided."
    return score, reason
