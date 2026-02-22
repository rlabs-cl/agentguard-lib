"""LangGraph integration — AgentGuard nodes for LangGraph workflows.

Provides pre-built LangGraph nodes for each pipeline step so that
LangGraph-based agents can use AgentGuard as part of their graph.

Example::

    from langgraph.graph import StateGraph
    from agentguard.integrations.langgraph import (
        agentguard_generate_node,
        agentguard_validate_node,
        agentguard_challenge_node,
    )

    graph = StateGraph(AgentState)
    graph.add_node("generate", agentguard_generate_node)
    graph.add_node("validate", agentguard_validate_node)
    graph.add_node("challenge", agentguard_challenge_node)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def agentguard_generate_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: run the full AgentGuard generation pipeline.

    Reads from state:
        - ``spec`` (str): Natural language specification.
        - ``archetype`` (str, optional): Archetype name. Default: ``"api_backend"``.
        - ``llm`` (str, optional): LLM model string. Default: ``"anthropic/claude-sonnet-4-20250514"``.

    Writes to state:
        - ``files`` (dict[str, str]): Generated files.
        - ``trace``: Trace summary.
        - ``generation_cost`` (float): Total generation cost.
    """
    from agentguard.pipeline import Pipeline

    spec = state["spec"]
    archetype = state.get("archetype", "api_backend")
    llm = state.get("llm", "anthropic/claude-sonnet-4-20250514")

    pipe = Pipeline(archetype=archetype, llm=llm)
    result = await pipe.generate(spec)

    return {
        **state,
        "files": result.files,
        "trace": result.trace,
        "generation_cost": float(result.total_cost.total_cost),
    }


async def agentguard_validate_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: validate generated code.

    Reads from state:
        - ``files`` (dict[str, str]): Files to validate.
        - ``archetype`` (str, optional): Archetype for context.

    Writes to state:
        - ``validation_passed`` (bool): Whether validation passed.
        - ``validation_errors`` (list): List of validation errors.
        - ``auto_fixes`` (int): Number of auto-fixes applied.
    """
    from agentguard.archetypes.base import Archetype
    from agentguard.validation.validator import Validator

    files = state["files"]
    archetype_name = state.get("archetype")

    archetype = Archetype.load(archetype_name) if archetype_name else None
    validator = Validator(archetype=archetype)
    report = validator.check(files)

    return {
        **state,
        "validation_passed": report.passed,
        "validation_errors": [str(e) for e in report.errors],
        "auto_fixes": len(report.auto_fixed),
    }


async def agentguard_challenge_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: self-challenge code against quality criteria.

    Reads from state:
        - ``files`` (dict[str, str]): Files to challenge.
        - ``criteria`` (list[str], optional): Quality criteria.
        - ``llm`` (str, optional): LLM for challenge.

    Writes to state:
        - ``challenge_passed`` (bool): Whether challenge passed.
        - ``challenge_feedback`` (str | None): Feedback text.
        - ``grounding_violations`` (list[str]): Grounding issues found.
    """
    from agentguard.challenge.challenger import SelfChallenger
    from agentguard.llm.factory import create_llm_provider

    files = state["files"]
    criteria = state.get("criteria", [])
    llm_str = state.get("llm", "anthropic/claude-sonnet-4-20250514")

    code = "\n\n".join(f"# {path}\n{content}" for path, content in files.items())
    llm = create_llm_provider(llm_str)
    challenger = SelfChallenger(llm=llm)

    result = await challenger.challenge(
        output=code,
        criteria=criteria,
        task_description="Code review via LangGraph",
    )

    return {
        **state,
        "challenge_passed": result.passed,
        "challenge_feedback": result.feedback,
        "grounding_violations": result.grounding_violations,
    }
