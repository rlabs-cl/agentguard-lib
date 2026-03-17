"""Pipeline — workflow definition for AgentGuard code generation.

This module defines the pipeline steps and their order. It no longer
executes any LLM calls directly. The calling agent follows the workflow
using the MCP tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in the generation pipeline."""

    name: str
    description: str
    tool: str
    depends_on: list[str] = field(default_factory=list)


# The canonical pipeline step definitions
PIPELINE_STEPS: list[PipelineStep] = [
    PipelineStep(
        name="skeleton",
        description="L1: Generate file tree with responsibilities",
        tool="skeleton",
    ),
    PipelineStep(
        name="contracts_and_wiring",
        description="L2+L3: Generate typed stubs with import wiring",
        tool="contracts_and_wiring",
        depends_on=["skeleton"],
    ),
    PipelineStep(
        name="logic",
        description="L4: Implement function bodies",
        tool="logic",
        depends_on=["contracts_and_wiring"],
    ),
    PipelineStep(
        name="validate",
        description="Structural validation of generated code",
        tool="validate",
        depends_on=["logic"],
    ),
    PipelineStep(
        name="challenge",
        description="Self-review against acceptance criteria",
        tool="get_challenge_criteria",
        depends_on=["validate"],
    ),
]


def get_pipeline_steps() -> list[PipelineStep]:
    """Return the canonical pipeline step definitions."""
    return list(PIPELINE_STEPS)
