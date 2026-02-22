"""Prompt management module — versioned, template-based prompts."""

from agentguard.prompts.registry import PromptRegistry
from agentguard.prompts.template import PromptTemplate

__all__ = ["PromptTemplate", "PromptRegistry"]
