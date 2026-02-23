"""Benchmark types — dataclasses for comparative archetype benchmarking."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────


class Complexity(StrEnum):
    """Five levels of benchmark specification complexity."""

    TRIVIAL = "trivial"        # Single-file, no deps, < 3 functions
    LOW = "low"                # Single-file with some deps, 3-5 functions
    MEDIUM = "medium"          # Multi-file, external deps, 5-15 functions
    HIGH = "high"              # Complex architecture, auth/DB/state, 15+ functions
    ENTERPRISE = "enterprise"  # Full system with observability, scaling, infra


ALL_COMPLEXITIES: list[Complexity] = list(Complexity)


class EnterpriseCheck(StrEnum):
    """Enterprise readiness dimensions."""

    TYPE_SAFETY = "type_safety"
    MODULARITY = "modularity"
    MAINTAINABILITY = "maintainability"
    ACCESSIBILITY = "accessibility"
    PERFORMANCE = "performance"
    OBSERVABILITY = "observability"
    TESTABILITY = "testability"


class OperationalCheck(StrEnum):
    """Operational readiness dimensions."""

    DEBUGGABILITY = "debuggability"
    FEATURE_EXTENSIBILITY = "feature_extensibility"
    CLOUD_SCALABILITY = "cloud_scalability"
    API_MIGRATION_COST = "api_migration_cost"
    TEST_SURFACE = "test_surface"
    TEAM_ONBOARDING = "team_onboarding"


# ── Scores ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single readiness dimension."""

    dimension: str
    score: float              # 0.0 – 1.0
    passed: bool
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 3),
            "passed": self.passed,
            "findings": self.findings,
        }


@dataclass(frozen=True)
class ReadinessScore:
    """Aggregate score across all dimensions of a readiness category."""

    category: str             # "enterprise" or "operational"
    overall_score: float      # 0.0 – 1.0 (mean of dimensions)
    passed: bool
    dimensions: list[DimensionScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "overall_score": round(self.overall_score, 3),
            "passed": self.passed,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


# ── Run Results ───────────────────────────────────────────────────


@dataclass
class RunResult:
    """Result of a single code-generation run (control or treatment)."""

    enterprise: ReadinessScore
    operational: ReadinessScore
    files_generated: int = 0
    total_lines: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None

    @property
    def combined_score(self) -> float:
        return (self.enterprise.overall_score + self.operational.overall_score) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "enterprise": self.enterprise.to_dict(),
            "operational": self.operational.to_dict(),
            "files_generated": self.files_generated,
            "total_lines": self.total_lines,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class ComplexityRun:
    """Paired control/treatment run for one complexity level."""

    complexity: Complexity
    spec: str
    control: RunResult         # Raw LLM (no AgentGuard)
    treatment: RunResult       # With AgentGuard pipeline
    improvement: float = 0.0   # treatment.combined - control.combined

    def __post_init__(self) -> None:
        self.improvement = self.treatment.combined_score - self.control.combined_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "spec": self.spec,
            "control": self.control.to_dict(),
            "treatment": self.treatment.to_dict(),
            "improvement": round(self.improvement, 3),
        }


# ── Benchmark Config ──────────────────────────────────────────────


@dataclass
class BenchmarkSpec:
    """A single benchmark specification at a given complexity."""

    complexity: Complexity
    spec: str                  # Natural-language dev request
    category: str = "general"  # Archetype category


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    model: str                                # e.g. "anthropic/claude-sonnet-4-20250514"
    specs: list[BenchmarkSpec] = field(default_factory=list)
    budget_ceiling_usd: float = 10.0          # Max total spend
    enterprise_threshold: float = 0.6         # Min enterprise score to pass
    operational_threshold: float = 0.6        # Min operational score to pass
    improvement_threshold: float = 0.05       # Min avg improvement to pass
    timeout_per_run_s: int = 300              # 5 min per run

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if config is valid."""
        errors: list[str] = []
        if not self.model:
            errors.append("Model is required")
        if not self.specs:
            errors.append("At least one benchmark spec is required")

        # Check all 5 complexity levels are covered
        covered = {s.complexity for s in self.specs}
        missing = set(ALL_COMPLEXITIES) - covered
        if missing:
            errors.append(
                f"Missing complexity levels: {', '.join(sorted(m.value for m in missing))}"
            )
        return errors


# ── Benchmark Report ──────────────────────────────────────────────


@dataclass
class BenchmarkReport:
    """Full benchmark report — signed, serializable."""

    version: str = "1.0.0"
    archetype_id: str = ""
    archetype_hash: str = ""       # SHA-256 of the YAML content
    model: str = ""
    runs: list[ComplexityRun] = field(default_factory=list)
    overall_passed: bool = False
    improvement_avg: float = 0.0   # Mean improvement across runs
    enterprise_avg: float = 0.0    # Mean enterprise score (treatment)
    operational_avg: float = 0.0   # Mean operational score (treatment)
    total_cost_usd: float = 0.0
    created_at: str = ""
    signature: str = ""            # HMAC-SHA256

    def compute_aggregates(self) -> None:
        """Recompute averages from runs."""
        if not self.runs:
            return
        n = len(self.runs)
        self.improvement_avg = sum(r.improvement for r in self.runs) / n
        self.enterprise_avg = sum(
            r.treatment.enterprise.overall_score for r in self.runs
        ) / n
        self.operational_avg = sum(
            r.treatment.operational.overall_score for r in self.runs
        ) / n
        self.total_cost_usd = sum(
            r.control.cost_usd + r.treatment.cost_usd for r in self.runs
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "archetype_id": self.archetype_id,
            "archetype_hash": self.archetype_hash,
            "model": self.model,
            "runs": [r.to_dict() for r in self.runs],
            "overall_passed": self.overall_passed,
            "improvement_avg": round(self.improvement_avg, 3),
            "enterprise_avg": round(self.enterprise_avg, 3),
            "operational_avg": round(self.operational_avg, 3),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "created_at": self.created_at,
            "signature": self.signature,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkReport:
        runs = [
            ComplexityRun(
                complexity=Complexity(r["complexity"]),
                spec=r["spec"],
                control=_run_result_from_dict(r["control"]),
                treatment=_run_result_from_dict(r["treatment"]),
            )
            for r in data.get("runs", [])
        ]
        return cls(
            version=data.get("version", "1.0.0"),
            archetype_id=data.get("archetype_id", ""),
            archetype_hash=data.get("archetype_hash", ""),
            model=data.get("model", ""),
            runs=runs,
            overall_passed=data.get("overall_passed", False),
            improvement_avg=data.get("improvement_avg", 0.0),
            enterprise_avg=data.get("enterprise_avg", 0.0),
            operational_avg=data.get("operational_avg", 0.0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            created_at=data.get("created_at", ""),
            signature=data.get("signature", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> BenchmarkReport:
        return cls.from_dict(json.loads(json_str))

    def sign(self, secret: str) -> None:
        """Sign the report with HMAC-SHA256(archetype_hash + secret)."""
        self.created_at = datetime.now(UTC).isoformat()
        key = (self.archetype_hash + secret).encode()
        payload = self._signable_payload()
        self.signature = hmac.new(key, payload, hashlib.sha256).hexdigest()

    def verify(self, secret: str) -> bool:
        """Verify the HMAC signature."""
        if not self.signature:
            return False
        key = (self.archetype_hash + secret).encode()
        payload = self._signable_payload()
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def _signable_payload(self) -> bytes:
        """Build the signable blob (everything except signature)."""
        data = self.to_dict()
        data.pop("signature", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


# ── Helpers ───────────────────────────────────────────────────────


def _readiness_from_dict(data: dict[str, Any]) -> ReadinessScore:
    dims = [
        DimensionScore(
            dimension=d["dimension"],
            score=d["score"],
            passed=d["passed"],
            findings=d.get("findings", []),
        )
        for d in data.get("dimensions", [])
    ]
    return ReadinessScore(
        category=data["category"],
        overall_score=data["overall_score"],
        passed=data["passed"],
        dimensions=dims,
    )


def _run_result_from_dict(data: dict[str, Any]) -> RunResult:
    return RunResult(
        enterprise=_readiness_from_dict(data["enterprise"]),
        operational=_readiness_from_dict(data["operational"]),
        files_generated=data.get("files_generated", 0),
        total_lines=data.get("total_lines", 0),
        total_tokens=data.get("total_tokens", 0),
        cost_usd=data.get("cost_usd", 0.0),
        duration_ms=data.get("duration_ms", 0),
        error=data.get("error"),
    )
