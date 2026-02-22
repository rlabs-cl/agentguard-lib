"""Pydantic request/response schemas — the JSON contract for thin SDKs.

These models define the HTTP API surface. Thin SDKs in TypeScript, Go, etc.
are generated from or mirror these exact types.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ------------------------------------------------------------------ #
#  Shared types
# ------------------------------------------------------------------ #

class TraceSummaryResponse(BaseModel):
    """Trace summary returned with generation results."""

    id: str = ""
    total_llm_calls: int = 0
    total_cost: dict[str, str] = Field(default_factory=lambda: {"total": "0", "currency": "USD"})
    structural_fixes: int = 0
    challenge_reworks: int = 0
    duration_ms: int = 0
    levels_completed: list[str] = Field(default_factory=list)


class CheckResultResponse(BaseModel):
    """Result of a single validation check."""

    check: str
    passed: bool
    details: str = ""
    duration_ms: int = 0


class AutoFixResponse(BaseModel):
    """Description of an auto-applied fix."""

    file: str
    fix: str
    line: int | None = None


class ValidationErrorResponse(BaseModel):
    """A single validation error."""

    check: str
    file_path: str
    line: int | None = None
    column: int | None = None
    message: str
    severity: str = "error"
    code: str | None = None


class CriterionResultResponse(BaseModel):
    """Result for a single challenge criterion."""

    criterion: str
    passed: bool
    explanation: str = ""


# ------------------------------------------------------------------ #
#  /v1/generate
# ------------------------------------------------------------------ #

class GenerateOptions(BaseModel):
    """Optional generation settings."""

    skip_challenge: bool = False
    skip_validation: bool = False
    parallel_l4: bool = True
    max_challenge_retries: int = 3


class GenerateRequest(BaseModel):
    """POST /v1/generate"""

    spec: str = Field(..., description="Natural language description of what to build")
    archetype: str = Field(default="api_backend", description="Project archetype")
    llm: str = Field(
        default="anthropic/claude-sonnet-4-20250514",
        description="LLM model identifier (provider/model)",
    )
    options: GenerateOptions = Field(default_factory=GenerateOptions)


class GenerateResponse(BaseModel):
    """Response from POST /v1/generate"""

    files: dict[str, str]
    trace: TraceSummaryResponse


# ------------------------------------------------------------------ #
#  /v1/validate
# ------------------------------------------------------------------ #

class ValidateRequest(BaseModel):
    """POST /v1/validate"""

    files: dict[str, str] = Field(..., description="Map of file paths to contents")
    archetype: str | None = Field(default=None, description="Archetype to validate against")
    checks: list[str] | None = Field(default=None, description="Specific checks to run")


class ValidateResponse(BaseModel):
    """Response from POST /v1/validate"""

    passed: bool
    checks: list[CheckResultResponse] = Field(default_factory=list)
    auto_fixed: list[AutoFixResponse] = Field(default_factory=list)
    errors: list[ValidationErrorResponse] = Field(default_factory=list)


# ------------------------------------------------------------------ #
#  /v1/challenge
# ------------------------------------------------------------------ #

class ChallengeRequest(BaseModel):
    """POST /v1/challenge"""

    code: str = Field(..., description="Code to challenge")
    criteria: list[str] | None = Field(default=None, description="Quality criteria")
    llm: str = Field(
        default="anthropic/claude-sonnet-4-20250514",
        description="LLM for challenge",
    )


class ChallengeResponse(BaseModel):
    """Response from POST /v1/challenge"""

    passed: bool
    attempt: int = 1
    criteria_results: list[CriterionResultResponse] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    grounding_violations: list[str] = Field(default_factory=list)
    feedback: str | None = None
    cost: dict[str, str] = Field(default_factory=lambda: {"total": "0", "currency": "USD"})


# ------------------------------------------------------------------ #
#  /v1/archetypes
# ------------------------------------------------------------------ #

class ArchetypeSummary(BaseModel):
    """Brief archetype info for listing."""

    id: str
    name: str
    description: str = ""
    trust_level: str = "official"
    content_hash: str = ""


class ArchetypeDetail(BaseModel):
    """Full archetype detail."""

    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    trust_level: str = "official"
    content_hash: str = ""
    tech_stack: dict[str, str] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    structure: dict[str, list[str]] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    self_challenge: dict[str, Any] = Field(default_factory=dict)
    reference_patterns: list[str] = Field(default_factory=list)


class ArchetypeVerifyRequest(BaseModel):
    """Request to verify an archetype's integrity."""

    archetype_id: str
    content_hash: str = ""


class ArchetypeVerifyResponse(BaseModel):
    """Result of archetype integrity verification."""

    archetype_id: str
    registered: bool
    trust_level: str | None = None
    content_hash: str | None = None
    hash_match: bool | None = None


# ------------------------------------------------------------------ #
#  /v1/traces
# ------------------------------------------------------------------ #

class TraceListItem(BaseModel):
    """Brief trace info for listing."""

    id: str
    archetype: str = ""
    spec_summary: str = ""
    started_at: str = ""
    total_cost: str = "0"
    duration_ms: int = 0


class TraceDetail(BaseModel):
    """Full trace detail."""

    id: str
    archetype: str = ""
    spec_summary: str = ""
    started_at: str = ""
    ended_at: str | None = None
    spans: list[dict[str, Any]] = Field(default_factory=list)
    summary: TraceSummaryResponse = Field(default_factory=TraceSummaryResponse)


# ------------------------------------------------------------------ #
#  SSE events (for /v1/generate/stream)
# ------------------------------------------------------------------ #

class SSELevelEvent(BaseModel):
    """SSE event emitted when a generation level completes."""

    event: str = "level_complete"
    level: str
    files: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    cost: str = "0"


class SSEValidationEvent(BaseModel):
    """SSE event emitted after validation."""

    event: str = "validation"
    passed: bool
    errors: int = 0
    fixes: int = 0


class SSEChallengeEvent(BaseModel):
    """SSE event emitted after self-challenge."""

    event: str = "challenge"
    passed: bool
    criteria_passed: int = 0
    criteria_total: int = 0
    rework: bool = False


class SSECompleteEvent(BaseModel):
    """Final SSE event with the complete result."""

    event: str = "complete"
    files: dict[str, str] = Field(default_factory=dict)
    trace: TraceSummaryResponse = Field(default_factory=TraceSummaryResponse)


# ------------------------------------------------------------------ #
#  Health
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    """GET /health"""

    status: str = "ok"
    version: str = ""


# ------------------------------------------------------------------ #
#  Error (RFC 7807)
# ------------------------------------------------------------------ #

class ProblemDetail(BaseModel):
    """RFC 7807 error response."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str = ""
    instance: str = ""
