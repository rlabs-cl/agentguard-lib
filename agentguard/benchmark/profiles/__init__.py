"""Benchmark profile registry — named evaluator bundles for archetype output types.

Profiles let archetype authors declare *how* their output should be evaluated,
rather than relying on the default Python-AST evaluator.

Built-in profiles
-----------------
code          Classic AST-based enterprise + operational readiness (default for code archetypes).
documentation Heuristic: heading structure, content length, code examples.
archetype     Heuristic: YAML validity, required archetype fields.
generic       Universal fallback: content presence, structure, spec-keyword coverage.

Custom profiles can be registered at import time:

    from agentguard.benchmark.profiles import register_profile, BenchmarkProfile

    register_profile(BenchmarkProfile(
        name="my_profile",
        description="Custom evaluator for my archetype family.",
        evaluate=my_evaluate_fn,
    ))
"""

# Register built-ins on import
from agentguard.benchmark.profiles import builtin as _builtin  # noqa: F401
from agentguard.benchmark.profiles.registry import (
    BenchmarkProfile,
    get_profile,
    list_profiles,
    register_profile,
)

__all__ = [
    "BenchmarkProfile",
    "get_profile",
    "list_profiles",
    "register_profile",
]
