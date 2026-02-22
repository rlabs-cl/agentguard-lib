"""Strict Pydantic schema for archetype YAML validation.

Every archetype — built-in or community — must pass this validation
before it can be loaded, published, or used in the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Enums ─────────────────────────────────────────────────────────


class Maturity(StrEnum):
    prototype = "prototype"
    production = "production"
    enterprise = "enterprise"


class TrustLevel(StrEnum):
    """How much the platform trusts this archetype."""

    official = "official"  # Built-in, shipped with the package
    verified = "verified"  # Staff-reviewed community archetype
    community = "community"  # User-published, passed schema only


VALID_LANGUAGES = frozenset({
    "python", "typescript", "javascript", "go", "rust", "java", "csharp", "ruby",
})

VALID_FRAMEWORKS = frozenset({
    "fastapi", "flask", "django", "express", "nextjs", "react", "vue", "svelte",
    "gin", "actix", "spring", "rails", "click", "typer", "none", "stdlib",
})

VALID_DATABASES = frozenset({
    "postgresql", "mysql", "sqlite", "mongodb", "redis", "dynamodb", "none",
})

VALID_TESTERS = frozenset({
    "pytest", "jest", "vitest", "mocha", "go_test", "cargo_test", "junit", "rspec", "unittest",
})

VALID_LINTERS = frozenset({
    "ruff", "eslint", "biome", "golangci-lint", "clippy", "checkstyle", "rubocop", "none",
})

VALID_TYPE_CHECKERS = frozenset({
    "mypy", "pyright", "tsc", "none",
})

VALID_PIPELINE_LEVELS = frozenset({
    "skeleton", "contracts", "wiring", "logic",
})

VALID_CHECKS = frozenset({
    "syntax", "lint", "types", "imports", "structure",
})

VALID_CATEGORIES = frozenset({
    "general", "backend", "frontend", "cli", "library", "script", "fullstack",
    "data", "ml", "devops", "mobile", "infra",
})


# ── Sub-schemas ───────────────────────────────────────────────────


class TechStackSchema(BaseModel):
    """Tech stack with constrained choices."""

    language: str = "python"
    framework: str = "fastapi"
    database: str = "postgresql"
    testing: str = "pytest"
    linter: str = "ruff"
    type_checker: str = "mypy"

    @field_validator("language")
    @classmethod
    def _valid_language(cls, v: str) -> str:
        if v not in VALID_LANGUAGES:
            raise ValueError(f"Invalid language '{v}'. Must be one of: {sorted(VALID_LANGUAGES)}")
        return v

    @field_validator("framework")
    @classmethod
    def _valid_framework(cls, v: str) -> str:
        if v not in VALID_FRAMEWORKS:
            raise ValueError(f"Invalid framework '{v}'. Must be one of: {sorted(VALID_FRAMEWORKS)}")
        return v

    @field_validator("database")
    @classmethod
    def _valid_database(cls, v: str) -> str:
        if v not in VALID_DATABASES:
            raise ValueError(f"Invalid database '{v}'. Must be one of: {sorted(VALID_DATABASES)}")
        return v

    @field_validator("testing")
    @classmethod
    def _valid_testing(cls, v: str) -> str:
        if v not in VALID_TESTERS:
            raise ValueError(f"Invalid test framework '{v}'. Must be one of: {sorted(VALID_TESTERS)}")
        return v

    @field_validator("linter")
    @classmethod
    def _valid_linter(cls, v: str) -> str:
        if v not in VALID_LINTERS:
            raise ValueError(f"Invalid linter '{v}'. Must be one of: {sorted(VALID_LINTERS)}")
        return v

    @field_validator("type_checker")
    @classmethod
    def _valid_type_checker(cls, v: str) -> str:
        if v not in VALID_TYPE_CHECKERS:
            raise ValueError(f"Invalid type checker '{v}'. Must be one of: {sorted(VALID_TYPE_CHECKERS)}")
        return v


class PipelineSchema(BaseModel):
    """Pipeline configuration."""

    levels: list[str] = Field(
        default=["skeleton", "contracts", "wiring", "logic"],
        min_length=1,
        max_length=4,
    )
    enable_self_challenge: bool = True
    enable_structural_validation: bool = True
    max_self_challenge_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("levels")
    @classmethod
    def _valid_levels(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_PIPELINE_LEVELS
        if invalid:
            raise ValueError(f"Invalid pipeline levels: {invalid}. Must be from: {sorted(VALID_PIPELINE_LEVELS)}")
        # skeleton must always be first if present
        if v and v[0] != "skeleton":
            raise ValueError("Pipeline must start with 'skeleton'")
        return v


class StructureSchema(BaseModel):
    """Expected project structure."""

    expected_dirs: list[str] = Field(default_factory=list, max_length=100)
    expected_files: list[str] = Field(default_factory=list, max_length=200)

    @field_validator("expected_dirs", "expected_files")
    @classmethod
    def _no_path_traversal(cls, v: list[str]) -> list[str]:
        for p in v:
            if ".." in p or p.startswith("/") or p.startswith("\\"):
                raise ValueError(f"Path traversal not allowed in structure: '{p}'")
            if any(c in p for c in ("<", ">", "|", "\0")):
                raise ValueError(f"Invalid characters in path: '{p}'")
        return v


class ContextRecipeSchema(BaseModel):
    """Token budget for a generation level."""

    include: list[str] = Field(default_factory=list, max_length=20)
    max_tokens: int = Field(default=4000, ge=100, le=200_000)


class ValidationSchema(BaseModel):
    """Validation checks configuration."""

    checks: list[str] = Field(
        default=["syntax", "lint", "types", "imports", "structure"],
        min_length=1,
    )
    lint_rules: str = Field(default="ruff:default", max_length=100)
    type_strictness: str = Field(default="basic", pattern=r"^(basic|strict|off)$")

    @field_validator("checks")
    @classmethod
    def _valid_checks(cls, v: list[str]) -> list[str]:
        invalid = set(v) - VALID_CHECKS
        if invalid:
            raise ValueError(f"Invalid checks: {invalid}. Must be from: {sorted(VALID_CHECKS)}")
        return v


class SelfChallengeSchema(BaseModel):
    """Self-challenge criteria."""

    criteria: list[str] = Field(default_factory=list, max_length=50)
    grounding_check: bool = True
    assumptions_must_declare: bool = True

    @field_validator("criteria")
    @classmethod
    def _no_empty_criteria(cls, v: list[str]) -> list[str]:
        for i, c in enumerate(v):
            stripped = c.strip()
            if not stripped:
                raise ValueError(f"Criterion at index {i} is empty")
            if len(stripped) > 500:
                raise ValueError(f"Criterion at index {i} exceeds 500 chars")
        return [c.strip() for c in v]


# ── Main Schema ───────────────────────────────────────────────────

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}[a-z0-9]$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?$")


class ArchetypeSchema(BaseModel):
    """Strict schema that every archetype YAML must satisfy.

    Validates all fields, types, ranges, and cross-field constraints.
    Used at YAML load time, marketplace upload, and publish.
    """

    id: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., min_length=2, max_length=255)
    description: str = Field(default="", max_length=2000)
    version: str = Field(default="1.0.0", max_length=32)
    maturity: Maturity = Maturity.production

    tech_stack: TechStackSchema = Field(default_factory=TechStackSchema)
    pipeline: PipelineSchema = Field(default_factory=PipelineSchema)
    structure: StructureSchema = Field(default_factory=StructureSchema)
    context_recipes: dict[str, ContextRecipeSchema] = Field(default_factory=dict)
    validation: ValidationSchema = Field(default_factory=ValidationSchema)
    self_challenge: SelfChallengeSchema = Field(default_factory=SelfChallengeSchema)
    reference_patterns: list[str] = Field(default_factory=list, max_length=30)
    infrastructure_files: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(
                f"Archetype ID '{v}' invalid. Must be lowercase alphanumeric + underscores, "
                f"3-64 chars, start with letter, end with letter/digit."
            )
        return v

    @field_validator("version")
    @classmethod
    def _valid_version(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError(
                f"Version '{v}' must be semver: MAJOR.MINOR.PATCH (e.g. 1.2.3)"
            )
        return v

    @field_validator("reference_patterns")
    @classmethod
    def _valid_reference_patterns(cls, v: list[str]) -> list[str]:
        pattern = re.compile(r"^[a-z][a-z0-9_]+$")
        for p in v:
            if not pattern.match(p):
                raise ValueError(
                    f"Reference pattern '{p}' invalid — must be lowercase alphanumeric + underscores"
                )
        return v

    @field_validator("infrastructure_files")
    @classmethod
    def _no_infra_path_traversal(cls, v: list[str]) -> list[str]:
        for p in v:
            if ".." in p or p.startswith("/") or p.startswith("\\"):
                raise ValueError(f"Path traversal not allowed: '{p}'")
        return v

    @model_validator(mode="after")
    def _cross_field_checks(self) -> ArchetypeSchema:
        # If pipeline has self-challenge enabled, criteria should exist
        if self.pipeline.enable_self_challenge and not self.self_challenge.criteria:
            # Warn but don't fail — built-ins with challenge enabled but
            # empty criteria will use default criteria from the prompt template
            pass

        # Context recipes should reference known pipeline levels
        known = set(self.pipeline.levels)
        unknown = set(self.context_recipes.keys()) - known
        if unknown:
            raise ValueError(
                f"Context recipes reference unknown pipeline levels: {unknown}. "
                f"Pipeline levels are: {sorted(known)}"
            )
        return self


# ── Helpers ───────────────────────────────────────────────────────


def validate_archetype_yaml(yaml_content: str) -> ArchetypeSchema:
    """Parse and validate a YAML string as an archetype.

    Args:
        yaml_content: Raw YAML text.

    Returns:
        Validated ArchetypeSchema.

    Raises:
        ValueError: If YAML is invalid or doesn't parse.
        pydantic.ValidationError: If schema validation fails.
    """
    import yaml as _yaml

    try:
        data = _yaml.safe_load(yaml_content)
    except _yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Archetype YAML must be a mapping (dict), not a scalar or list")

    return _validate_archetype_dict(data)


def _validate_archetype_dict(data: dict[str, Any]) -> ArchetypeSchema:
    """Validate a raw dict (from YAML or JSON) as an archetype."""
    # Normalize nested structures
    normalized: dict[str, Any] = {}

    # Top-level scalars
    for key in ("id", "name", "description", "version", "maturity"):
        if key in data:
            normalized[key] = data[key]

    # tech_stack — YAML uses `defaults` sub-key, schema is flat
    ts_raw = data.get("tech_stack", {})
    if isinstance(ts_raw, dict):
        defaults = ts_raw.get("defaults", ts_raw)
        normalized["tech_stack"] = defaults

    # structure — YAML uses dict with expected_dirs/expected_files
    struct_raw = data.get("structure", {})
    if isinstance(struct_raw, dict):
        normalized["structure"] = struct_raw

    # Direct pass-through for the rest
    for key in (
        "pipeline", "context_recipes", "validation",
        "self_challenge", "reference_patterns", "infrastructure_files",
    ):
        if key in data:
            normalized[key] = data[key]

    return ArchetypeSchema.model_validate(normalized)


def compute_content_hash(yaml_content: str) -> str:
    """Compute a deterministic SHA-256 hash of archetype YAML.

    Normalizes the content to ensure the same logical archetype
    always produces the same hash regardless of whitespace/ordering.

    Args:
        yaml_content: Raw YAML text.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    import yaml as _yaml

    data = _yaml.safe_load(yaml_content)
    # Canonical JSON — sorted keys, no whitespace variation
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_content_hash(yaml_content: str, expected_hash: str) -> bool:
    """Check if YAML content matches a known hash.

    Args:
        yaml_content: Raw YAML text.
        expected_hash: Expected SHA-256 hex digest.

    Returns:
        True if hashes match.
    """
    return compute_content_hash(yaml_content) == expected_hash
