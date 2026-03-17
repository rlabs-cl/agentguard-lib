"""SelfChallenger — criteria renderer for agent-driven code review.

This module no longer calls any LLM directly. It renders challenge criteria
and review prompts that the calling agent uses with its own LLM.
"""

from __future__ import annotations

import logging
import re

from agentguard.challenge.types import CriterionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Challenge prompt templates (used by the agent-native tools)
# ---------------------------------------------------------------------------

_CHALLENGE_SYSTEM = """\
You are a strict code reviewer. Your job is to evaluate code output against \
acceptance criteria. Be critical -- if in doubt, FAIL the criterion.\
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


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

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
# SelfChallenger — criteria-only renderer
# ---------------------------------------------------------------------------


class SelfChallenger:
    """Criteria renderer for self-challenge review.

    Renders challenge prompts and criteria that the calling agent uses with its
    own LLM. No longer calls any LLM directly.

    Usage::

        challenger = SelfChallenger()
        prompt = challenger.render_challenge_prompt(
            output=code,
            criteria=["All endpoints match the spec", "No hardcoded secrets"],
            context_summary="spec + skeleton context...",
            task_description="Implement contracts for auth module",
        )
        # The calling agent sends this prompt to its own LLM
    """

    def render_challenge_prompt(
        self,
        output: str,
        criteria: list[str],
        *,
        context_summary: str = "",
        task_description: str = "Code generation",
    ) -> list[dict[str, str]]:
        """Render a challenge review prompt for the calling agent.

        Args:
            output: The generated code/text to evaluate.
            criteria: List of acceptance criteria strings.
            context_summary: Summary of context the LLM had when generating.
            task_description: Human-readable description of the task.

        Returns:
            List of message dicts (role/content) for the challenge prompt.
        """
        criteria_list = "\n".join(
            f"CRITERION {i + 1}: {c}" for i, c in enumerate(criteria)
        )
        user_prompt = _CHALLENGE_USER.format(
            task_description=task_description,
            output=output,
            context_summary=context_summary or "(no additional context)",
            criteria_list=criteria_list,
        )
        return [
            {"role": "system", "content": _CHALLENGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def build_feedback(
        criteria_results: list[CriterionResult],
        violations: list[str] | None = None,
        assumptions: list[str] | None = None,
    ) -> str:
        """Build human-readable feedback string from results."""
        violations = violations or []
        assumptions = assumptions or []
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
