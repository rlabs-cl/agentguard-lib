"""SelfChallenger — LLM-based adversarial review of generated code."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from agentguard.challenge.types import ChallengeResult, CriterionResult
from agentguard.llm.types import CostEstimate, GenerationConfig, Message

if TYPE_CHECKING:
    from agentguard.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Challenge prompt template (embedded — also available as builtin YAML)
# ---------------------------------------------------------------------------

_CHALLENGE_SYSTEM = """\
You are a strict code reviewer. Your job is to evaluate code output against \
acceptance criteria. Be critical — if in doubt, FAIL the criterion.\
"""

_CHALLENGE_USER = """\
You completed: {task_description}

Your output:
```
{output}
```

Context provided to you:
{context_summary}

Evaluate your output against each criterion below. For each, answer PASS or FAIL
with a one-line explanation. Use the EXACT format shown:

{criteria_list}

Then answer these grounding questions:
GROUNDING:
1. Did you use any API, function, class, or module that was NOT in your provided context?
   If yes, list each one prefixed with "- VIOLATION: ".
   If no, write "- NONE".
2. List every assumption you made that was not explicitly stated in the spec.
   Prefix each with "- ASSUMPTION: ".
   If none, write "- NONE".

Be strict. If in doubt, FAIL.\
"""

_REWORK_SYSTEM = """\
You are a code generator. You will receive code that failed a quality review, \
along with specific feedback. Produce an improved version that addresses every \
concern.\
"""

_REWORK_USER = """\
Original task: {task_description}

Previous output (FAILED review):
```
{output}
```

Review feedback:
{feedback}

Context:
{context_summary}

Produce the corrected output. Output ONLY the code — no explanations, \
no markdown fences.\
"""


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

_CRITERION_RE = re.compile(
    r"^(?:CRITERION\s+\d+|[-*]\s*(?:Criterion\s+\d+[:.])?\s*)"
    r".*?(PASS|FAIL)[:\s]*(.*)$",
    re.IGNORECASE,
)

_PASS_FAIL_RE = re.compile(
    r"(PASS|FAIL)\b[:\s]*(.*)",
    re.IGNORECASE,
)


def _parse_criteria_results(
    text: str,
    criteria: list[str],
) -> list[CriterionResult]:
    """Parse the LLM's criterion-by-criterion evaluation.

    We try to pair each criterion with a PASS/FAIL line from the LLM output.
    Unmatched criteria default to FAIL with an "unparseable" explanation.
    """
    # Collect all PASS/FAIL lines in order
    verdicts: list[tuple[bool, str]] = []
    for line in text.splitlines():
        m = _PASS_FAIL_RE.search(line)
        if m:
            passed = m.group(1).upper() == "PASS"
            explanation = m.group(2).strip().rstrip(".")
            verdicts.append((passed, explanation))

    results: list[CriterionResult] = []
    for i, criterion in enumerate(criteria):
        if i < len(verdicts):
            passed, explanation = verdicts[i]
        else:
            passed, explanation = False, "Could not parse LLM response for this criterion"
        results.append(CriterionResult(
            criterion=criterion,
            passed=passed,
            explanation=explanation,
        ))
    return results


def _parse_grounding_section(text: str) -> tuple[list[str], list[str]]:
    """Extract grounding violations and assumptions from the LLM response.

    Returns:
        (violations, assumptions)
    """
    violations: list[str] = []
    assumptions: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- VIOLATION:"):
            v = stripped.removeprefix("- VIOLATION:").strip()
            if v.upper() != "NONE":
                violations.append(v)
        elif stripped.startswith("- ASSUMPTION:"):
            a = stripped.removeprefix("- ASSUMPTION:").strip()
            if a.upper() != "NONE":
                assumptions.append(a)

    return violations, assumptions


# ---------------------------------------------------------------------------
# SelfChallenger
# ---------------------------------------------------------------------------


class SelfChallenger:
    """LLM-based self-evaluation against explicit acceptance criteria.

    After structural validation passes, the challenger asks the LLM to
    critically review its own output.  If criteria fail, it can trigger
    a rework loop (up to *max_retries* attempts).

    Usage::

        challenger = SelfChallenger(llm)
        result = await challenger.challenge(
            output=code,
            criteria=["All endpoints match the spec", "No hardcoded secrets"],
            context_summary="spec + skeleton context…",
            task_description="Implement contracts for auth module",
        )
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: GenerationConfig | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or GenerationConfig(temperature=0.0, max_tokens=2048)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def challenge(
        self,
        output: str,
        criteria: list[str],
        *,
        context_summary: str = "",
        task_description: str = "Code generation",
        max_retries: int = 3,
        grounding_check: bool = True,
        auto_rework: bool = True,
    ) -> ChallengeResult:
        """Evaluate *output* against *criteria* using the LLM.

        If the challenge fails and *auto_rework* is True, triggers a
        rework loop for up to *max_retries* total attempts.

        Args:
            output: The generated code/text to evaluate.
            criteria: List of acceptance criteria strings.
            context_summary: Summary of context the LLM had when generating.
            task_description: Human-readable description of the task.
            max_retries: Maximum number of rework attempts.
            grounding_check: Whether to check for grounding violations.
            auto_rework: If True, automatically rework on failure.

        Returns:
            ChallengeResult with per-criterion verdicts and metadata.
        """
        total_cost = CostEstimate.zero()
        current_output = output
        best_result: ChallengeResult | None = None

        for attempt in range(1, max_retries + 1):
            # Run the challenge evaluation
            result = await self._evaluate(
                output=current_output,
                criteria=criteria,
                context_summary=context_summary,
                task_description=task_description,
                grounding_check=grounding_check,
                attempt=attempt,
            )
            total_cost = total_cost + result.cost
            result = ChallengeResult(
                passed=result.passed,
                attempt=attempt,
                criteria_results=result.criteria_results,
                assumptions=result.assumptions,
                grounding_violations=result.grounding_violations,
                feedback=result.feedback,
                rework_output=result.rework_output,
                cost=total_cost,
            )

            if result.passed:
                logger.info("Self-challenge PASSED on attempt %d", attempt)
                return result

            best_result = result

            if not auto_rework or attempt == max_retries:
                break

            # Rework: generate improved output incorporating feedback
            logger.info(
                "Self-challenge FAILED on attempt %d, reworking (%d/%d)",
                attempt, attempt, max_retries,
            )
            feedback = result.feedback or self._build_feedback(result)
            reworked, rework_cost = await self._rework(
                output=current_output,
                feedback=feedback,
                context_summary=context_summary,
                task_description=task_description,
            )
            total_cost = total_cost + rework_cost
            current_output = reworked

        # Exhausted retries — return best (last) result with reworked output
        assert best_result is not None
        return ChallengeResult(
            passed=False,
            attempt=best_result.attempt,
            criteria_results=best_result.criteria_results,
            assumptions=best_result.assumptions,
            grounding_violations=best_result.grounding_violations,
            feedback=best_result.feedback,
            rework_output=current_output if current_output != output else None,
            cost=total_cost,
        )

    async def evaluate_only(
        self,
        output: str,
        criteria: list[str],
        *,
        context_summary: str = "",
        task_description: str = "Code generation",
        grounding_check: bool = True,
    ) -> ChallengeResult:
        """Run a single evaluation pass (no rework loop)."""
        return await self._evaluate(
            output=output,
            criteria=criteria,
            context_summary=context_summary,
            task_description=task_description,
            grounding_check=grounding_check,
            attempt=1,
        )

    async def rework(
        self,
        output: str,
        feedback: str,
        context_summary: str = "",
        task_description: str = "Code generation",
    ) -> str:
        """Re-generate output incorporating challenge feedback."""
        reworked, _cost = await self._rework(
            output=output,
            feedback=feedback,
            context_summary=context_summary,
            task_description=task_description,
        )
        return reworked

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _evaluate(
        self,
        output: str,
        criteria: list[str],
        context_summary: str,
        task_description: str,
        grounding_check: bool,
        attempt: int,
    ) -> ChallengeResult:
        """Run one evaluation pass."""
        criteria_list = "\n".join(
            f"CRITERION {i + 1}: {c}" for i, c in enumerate(criteria)
        )
        user_prompt = _CHALLENGE_USER.format(
            task_description=task_description,
            output=output,
            context_summary=context_summary or "(no additional context)",
            criteria_list=criteria_list,
        )
        messages = [
            Message(role="system", content=_CHALLENGE_SYSTEM),
            Message(role="user", content=user_prompt),
        ]

        t0 = time.monotonic()
        response = await self._llm.generate(messages, self._config)
        elapsed = int((time.monotonic() - t0) * 1000)

        logger.debug("Challenge LLM response (%dms): %s...", elapsed, response.content[:200])

        # Parse the response
        criteria_results = _parse_criteria_results(response.content, criteria)
        violations, assumptions = _parse_grounding_section(response.content)

        all_passed = all(c.passed for c in criteria_results)
        no_violations = len(violations) == 0 if grounding_check else True
        passed = all_passed and no_violations

        feedback: str | None = None
        if not passed:
            feedback = self._build_feedback_from_parts(
                criteria_results, violations, assumptions,
            )

        return ChallengeResult(
            passed=passed,
            attempt=attempt,
            criteria_results=criteria_results,
            assumptions=assumptions,
            grounding_violations=violations,
            feedback=feedback,
            cost=response.cost,
        )

    async def _rework(
        self,
        output: str,
        feedback: str,
        context_summary: str,
        task_description: str,
    ) -> tuple[str, CostEstimate]:
        """Ask the LLM to rework the output given feedback."""
        user_prompt = _REWORK_USER.format(
            task_description=task_description,
            output=output,
            feedback=feedback,
            context_summary=context_summary or "(no additional context)",
        )
        messages = [
            Message(role="system", content=_REWORK_SYSTEM),
            Message(role="user", content=user_prompt),
        ]
        response = await self._llm.generate(messages, self._config)
        return response.content.strip(), response.cost

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_feedback(self, result: ChallengeResult) -> str:
        """Build human-readable feedback string from a failed result."""
        return self._build_feedback_from_parts(
            result.criteria_results,
            result.grounding_violations,
            result.assumptions,
        )

    @staticmethod
    def _build_feedback_from_parts(
        criteria_results: list[CriterionResult],
        violations: list[str],
        assumptions: list[str],
    ) -> str:
        parts: list[str] = []
        failed = [c for c in criteria_results if not c.passed]
        if failed:
            parts.append("FAILED CRITERIA:")
            for c in failed:
                parts.append(f"  - {c.criterion}: {c.explanation}")
        if violations:
            parts.append("GROUNDING VIOLATIONS:")
            for v in violations:
                parts.append(f"  - {v}")
        if assumptions:
            parts.append("ASSUMPTIONS (review these):")
            for a in assumptions:
                parts.append(f"  - {a}")
        return "\n".join(parts)
