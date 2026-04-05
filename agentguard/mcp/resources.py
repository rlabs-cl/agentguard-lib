"""MCP resource definitions for AgentGuard.

Resources expose read-only context that AI tools can inspect
without calling a tool.

**IP Protection:** Non-official archetypes (community, verified) have their
proprietary fields redacted from resource responses.  The LLM receives
criteria and validation config only through the pipeline tools (skeleton,
contracts_and_wiring, get_challenge_criteria, validate), which include a
confidentiality directive instructing the LLM not to share those details
with the end user.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Trust levels that may expose full archetype internals via resources.
# Community and verified archetypes are IP-protected.
_FULL_DISCLOSURE_TRUST_LEVELS = {"official"}


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
    """Return archetype definition. Non-official archetypes have IP-sensitive
    fields redacted (criteria, validation rules, reference patterns, structure).
    """
    from agentguard.archetypes.registry import get_archetype_registry

    registry = get_archetype_registry()
    entry = registry.get_entry(name)
    arch = entry.archetype
    is_full_disclosure = entry.trust_level.value in _FULL_DISCLOSURE_TRUST_LEVELS

    result: dict = {
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
    }

    if is_full_disclosure:
        # Official archetypes: full transparency
        result["structure"] = arch.structure
        result["validation"] = {
            "checks": arch.validation.checks,
            "lint_rules": arch.validation.lint_rules,
            "type_strictness": arch.validation.type_strictness,
        }
        result["self_challenge"] = {
            "criteria": arch.self_challenge.criteria,
            "grounding_check": arch.self_challenge.grounding_check,
        }
        result["reference_patterns"] = arch.reference_patterns
    else:
        # Community/verified archetypes: redact IP-sensitive fields
        result["structure"] = "[REDACTED — proprietary to archetype creator]"
        result["validation"] = {
            "checks_count": len(arch.validation.checks),
            "note": "Validation rules are proprietary. They are applied during generation via the pipeline tools.",
        }
        result["self_challenge"] = {
            "criteria_count": len(arch.self_challenge.criteria),
            "note": "Challenge criteria are proprietary. They are applied during self-challenge via get_challenge_criteria.",
        }
        result["reference_patterns"] = (
            f"{len(arch.reference_patterns)} patterns [REDACTED]"
            if arch.reference_patterns
            else []
        )

    return json.dumps(result, indent=2)
