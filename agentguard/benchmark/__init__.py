"""AgentGuard Benchmark — comparative quality assessment for archetypes.

Runs the same development request WITH vs WITHOUT AgentGuard across
5 complexity levels, evaluating enterprise and operational readiness.
"""

from agentguard.benchmark.types import (
    ALL_COMPLEXITIES,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkSpec,
    Complexity,
    ComplexityRun,
    DimensionScore,
    EnterpriseCheck,
    OperationalCheck,
    ReadinessScore,
    RunResult,
)

__all__ = [
    "ALL_COMPLEXITIES",
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunner",
    "BenchmarkSpec",
    "Complexity",
    "ComplexityRun",
    "DimensionScore",
    "EnterpriseCheck",
    "OperationalCheck",
    "ReadinessScore",
    "RunResult",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "BenchmarkRunner":
        from agentguard.benchmark.runner import BenchmarkRunner

        return BenchmarkRunner
    raise AttributeError(f"module 'agentguard.benchmark' has no attribute {name!r}")
