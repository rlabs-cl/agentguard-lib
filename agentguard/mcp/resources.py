"""MCP resource definitions for AgentGuard.

Resources expose read-only context that AI tools can inspect
without calling a tool.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def get_archetypes_resource() -> str:
    """Return a JSON list of all available archetypes (resource)."""
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    archetypes = []
    for arch_id in registry.list_available():
        entry = registry.get_entry(arch_id)
        archetypes.append(
            {
                "id": entry.archetype.id,
                "name": entry.archetype.name,
                "description": entry.archetype.description,
                "trust_level": entry.trust_level.value,
                "content_hash": entry.content_hash,
            }
        )
    return json.dumps(archetypes, indent=2)


def get_archetype_resource(name: str) -> str:
    """Return the full YAML-like definition for a single archetype."""
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    entry = registry.get_entry(name)
    arch = entry.archetype

    return json.dumps(
        {
            "id": arch.id,
            "name": arch.name,
            "description": arch.description,
            "version": arch.version,
            "trust_level": entry.trust_level.value,
            "content_hash": entry.content_hash,
            "tech_stack": {
                "language": arch.tech_stack.language,
                "framework": arch.tech_stack.framework,
                "database": arch.tech_stack.database,
                "testing": arch.tech_stack.testing,
                "linter": arch.tech_stack.linter,
                "type_checker": arch.tech_stack.type_checker,
            },
            "pipeline": {
                "levels": arch.pipeline.levels,
                "enable_self_challenge": arch.pipeline.enable_self_challenge,
                "enable_structural_validation": arch.pipeline.enable_structural_validation,
                "max_self_challenge_retries": arch.pipeline.max_self_challenge_retries,
            },
            "structure": arch.structure,
            "validation": {
                "checks": arch.validation.checks,
                "lint_rules": arch.validation.lint_rules,
                "type_strictness": arch.validation.type_strictness,
            },
            "self_challenge": {
                "criteria": arch.self_challenge.criteria,
                "grounding_check": arch.self_challenge.grounding_check,
            },
            "reference_patterns": arch.reference_patterns,
        },
        indent=2,
    )
