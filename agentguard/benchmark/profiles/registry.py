"""Profile registry — storage and lookup for named BenchmarkProfile objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentguard.benchmark.types import ReadinessScore

# Evaluator function signature:
#   (spec_text, files, enterprise_threshold, operational_threshold)
#   -> (enterprise_score, operational_score)
_EvalFn = Callable[
    [str, dict[str, str], float, float],
    tuple["ReadinessScore", "ReadinessScore"],
]


@dataclass
class BenchmarkProfile:
    """Named evaluator bundle for a class of archetype output.

    The ``evaluate`` callable receives:
    - ``spec``: the original benchmark specification text
    - ``files``: the generated file mapping (path → content)
    - ``enterprise_threshold`` / ``operational_threshold``: pass thresholds from config

    It returns a ``(enterprise_score, operational_score)`` tuple of
    :class:`~agentguard.benchmark.types.ReadinessScore` objects.
    For non-code archives (docs, YAML, research), returning the same object
    for both scores is perfectly fine.
    """

    name: str
    description: str
    evaluate: _EvalFn


_PROFILES: dict[str, BenchmarkProfile] = {}


def register_profile(profile: BenchmarkProfile) -> None:
    """Register a profile under its name.  Overwrites silently on conflict."""
    _PROFILES[profile.name] = profile


def get_profile(name: str) -> BenchmarkProfile | None:
    """Return a profile by name, or ``None`` if not registered."""
    return _PROFILES.get(name)


def list_profiles() -> list[str]:
    """Return sorted list of all registered profile names."""
    return sorted(_PROFILES.keys())
