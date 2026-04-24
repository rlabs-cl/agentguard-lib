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


class ConfidentialityPolicy(StrEnum):
    """Author's policy for how the archetype's internals may be shared with end users.

    - transparent:  Criteria, structure, and YAML may be reproduced verbatim in responses.
                    Intended for own archetypes where full auditability is the point.
    - attribution:  Reproduction permitted with explicit attribution to the author/archetype.
    - paraphrase:   DEFAULT. LLMs may paraphrase/explain what the archetype checks at a
                    descriptive level, but MUST NOT reproduce criterion text verbatim.
    - proprietary:  Strict no-share: describe WHAT the archetype builds and QUALITY of its
                    output only, never criteria text, counts, categories, or weights.
    """

    transparent = "transparent"
    attribution = "attribution"
    paraphrase = "paraphrase"
    proprietary = "proprietary"


class OutputKind(StrEnum):
    """What kind of artifact this archetype produces.

    - code:    Executable software (APIs, CLIs, libraries, scripts, etc.)
    - content: Structured documents, configs, specifications, manuals
    - hybrid:  Mixed — code with significant documentation, or documented configs
    """

    code = "code"
    content = "content"
    hybrid = "hybrid"


class TrustLevel(StrEnum):
    """How much the platform trusts this archetype."""

    official = "official"  # Built-in, shipped with the package
    verified = "verified"  # Staff-reviewed community archetype
    community = "community"  # User-published, passed schema only


# Programming languages — for executable code generation
VALID_TECH_LANGUAGES = frozenset({
    "python", "typescript", "javascript", "go", "rust", "java", "csharp", "ruby",
})

# Content and markup languages — for structured artifacts (docs, configs, etc.)
VALID_CONTENT_LANGUAGES = frozenset({
    "markdown", "latex", "yaml", "json", "toml", "xml", "html", "css", "sql",
})

# Combined set: AgentGuard archetypes can produce any structured artifact
VALID_LANGUAGES = VALID_TECH_LANGUAGES | VALID_CONTENT_LANGUAGES

# ── Ecosystem compatibility maps ─────────────────────────────────
# Instead of strict whitelists, we map KNOWN tools to their language
# ecosystems. Unknown tools are allowed (creative freedom), but known
# tools used with the wrong language trigger a consistency error.
#
# Values like "none" and "stdlib" always pass (universal).

_UNIVERSAL = frozenset({"none", "stdlib"})

FRAMEWORK_ECOSYSTEMS: dict[str, frozenset[str]] = {
    # Python
    "fastapi": frozenset({"python"}),
    "flask": frozenset({"python"}),
    "django": frozenset({"python"}),
    "click": frozenset({"python"}),
    "typer": frozenset({"python"}),
    "prefect": frozenset({"python"}),
    "aiokafka": frozenset({"python"}),
    "strawberry": frozenset({"python"}),
    "grpcio": frozenset({"python"}),
    "fastmcp": frozenset({"python"}),
    "langgraph": frozenset({"python"}),
    "celery": frozenset({"python"}),
    "scrapy": frozenset({"python"}),
    # JavaScript / TypeScript
    "express": frozenset({"javascript", "typescript"}),
    "nextjs": frozenset({"javascript", "typescript"}),
    "react": frozenset({"javascript", "typescript"}),
    "vue": frozenset({"javascript", "typescript"}),
    "svelte": frozenset({"javascript", "typescript"}),
    "nestjs": frozenset({"typescript"}),
    "hono": frozenset({"javascript", "typescript"}),
    # Go
    "gin": frozenset({"go"}),
    "echo": frozenset({"go"}),
    "fiber": frozenset({"go"}),
    # Rust
    "actix": frozenset({"rust"}),
    "axum": frozenset({"rust"}),
    "rocket": frozenset({"rust"}),
    # Java
    "spring": frozenset({"java"}),
    "quarkus": frozenset({"java"}),
    # Ruby
    "rails": frozenset({"ruby"}),
    "sinatra": frozenset({"ruby"}),
    # C#
    "aspnet": frozenset({"csharp"}),
    "blazor": frozenset({"csharp"}),
}

TESTER_ECOSYSTEMS: dict[str, frozenset[str]] = {
    "pytest": frozenset({"python"}),
    "unittest": frozenset({"python"}),
    "jest": frozenset({"javascript", "typescript"}),
    "vitest": frozenset({"javascript", "typescript"}),
    "mocha": frozenset({"javascript", "typescript"}),
    "go_test": frozenset({"go"}),
    "cargo_test": frozenset({"rust"}),
    "junit": frozenset({"java"}),
    "rspec": frozenset({"ruby"}),
}

LINTER_ECOSYSTEMS: dict[str, frozenset[str]] = {
    "ruff": frozenset({"python"}),
    "pylint": frozenset({"python"}),
    "flake8": frozenset({"python"}),
    "eslint": frozenset({"javascript", "typescript"}),
    "biome": frozenset({"javascript", "typescript"}),
    "golangci-lint": frozenset({"go"}),
    "clippy": frozenset({"rust"}),
    "checkstyle": frozenset({"java"}),
    "rubocop": frozenset({"ruby"}),
    # Infrastructure / spec linters (language-agnostic, work on config files)
    "tflint": frozenset({"yaml", "python", "typescript", "go"}),
    "spectral": frozenset({"yaml", "json", "python", "typescript"}),
}

TYPE_CHECKER_ECOSYSTEMS: dict[str, frozenset[str]] = {
    "mypy": frozenset({"python"}),
    "pyright": frozenset({"python"}),
    "tsc": frozenset({"typescript"}),
}


def _check_ecosystem_consistency(
    tool_name: str,
    tool_value: str,
    language: str,
    ecosystem_map: dict[str, frozenset[str]],
) -> str | None:
    """Return an error message if there's a known ecosystem incompatibility, else None."""
    if tool_value in _UNIVERSAL:
        return None
    # Content languages (markdown, latex, etc.) are documentation archetypes —
    # ecosystem checks don't apply because they don't produce executable code.
    if language in VALID_CONTENT_LANGUAGES:
        return None
    if tool_value not in ecosystem_map:
        # Unknown tool — allow it (creative freedom)
        return None
    compatible_langs = ecosystem_map[tool_value]
    if language not in compatible_langs:
        return (
            f"Ecosystem inconsistency: {tool_name}='{tool_value}' is not compatible "
            f"with language='{language}'. '{tool_value}' works with: {sorted(compatible_langs)}"
        )
    return None

VALID_PIPELINE_LEVELS = frozenset({
    "skeleton", "contracts", "wiring", "logic",
})

VALID_CHECKS = frozenset({
    "syntax", "lint", "types", "imports", "structure",
})

# ── Categories ────────────────────────────────────────────────────
# Code-oriented categories (original)
VALID_CODE_CATEGORIES = frozenset({
    "backend", "frontend", "cli", "library", "script", "fullstack",
    "data", "ml", "devops", "mobile", "infra",
})

# Content-oriented categories (new — for non-code archetypes)
VALID_CONTENT_CATEGORIES = frozenset({
    "documentation", "config", "design", "report", "tutorial",
    "specification", "template",
})

# Combined: any archetype can use any category
VALID_CATEGORIES = frozenset({"general"}) | VALID_CODE_CATEGORIES | VALID_CONTENT_CATEGORIES

VALID_DIMENSIONS = frozenset({
    # Enterprise readiness
    "type_safety", "modularity", "maintainability", "accessibility",
    "performance", "observability", "testability",
    # Operational readiness
    "debuggability", "feature_extensibility", "cloud_scalability",
    "api_migration_cost", "test_surface", "team_onboarding",
})


# ── Sub-schemas ───────────────────────────────────────────────────


class TechStackSchema(BaseModel):
    """Tech stack with ecosystem-consistency validation.

    Instead of strict whitelists, validates that tools are consistent
    with the declared language ecosystem. Unknown tools are allowed
    (creative freedom); known cross-ecosystem mismatches are rejected.
    """

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

    @model_validator(mode="after")
    def _check_ecosystem_consistency(self) -> TechStackSchema:
        """Validate that framework, tester, linter, and type_checker
        are consistent with the declared language."""
        lang = self.language
        errors: list[str] = []

        err = _check_ecosystem_consistency("framework", self.framework, lang, FRAMEWORK_ECOSYSTEMS)
        if err:
            errors.append(err)

        err = _check_ecosystem_consistency("testing", self.testing, lang, TESTER_ECOSYSTEMS)
        if err:
            errors.append(err)

        err = _check_ecosystem_consistency("linter", self.linter, lang, LINTER_ECOSYSTEMS)
        if err:
            errors.append(err)

        err = _check_ecosystem_consistency("type_checker", self.type_checker, lang, TYPE_CHECKER_ECOSYSTEMS)
        if err:
            errors.append(err)

        if errors:
            raise ValueError(
                f"Tech stack ecosystem inconsistencies:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return self


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
    confidentiality_policy: ConfidentialityPolicy = Field(
        default=ConfidentialityPolicy.paraphrase,
        description=(
            "Author's policy for how the archetype's internals may be shared with end "
            "users by the consuming LLM. Default 'paraphrase' allows descriptive "
            "explanation without verbatim reproduction; 'transparent' allows verbatim; "
            "'proprietary' forbids all reproduction. See ConfidentialityPolicy enum."
        ),
    )
    output_kind: OutputKind = OutputKind.code
    category: str = Field(default="general", max_length=64)

    tech_stack: TechStackSchema = Field(default_factory=TechStackSchema)
    pipeline: PipelineSchema = Field(default_factory=PipelineSchema)
    structure: StructureSchema = Field(default_factory=StructureSchema)
    context_recipes: dict[str, ContextRecipeSchema] = Field(default_factory=dict)
    validation: ValidationSchema = Field(default_factory=ValidationSchema)
    self_challenge: SelfChallengeSchema = Field(default_factory=SelfChallengeSchema)
    reference_patterns: list[str] = Field(default_factory=list, max_length=30)
    infrastructure_files: list[str] = Field(default_factory=list, max_length=50)
    target_output: str | None = Field(None, max_length=255)
    scoring_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension relevance weights (0.0 = N/A, 1.0 = full weight). "
                    "Missing dimensions default to 1.0.",
    )

    @field_validator("scoring_weights")
    @classmethod
    def _valid_scoring_weights(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v.keys()) - VALID_DIMENSIONS
        if unknown:
            # Suggest closest match for common typos
            msg = f"Unknown scoring_weights dimensions: {sorted(unknown)}. "
            msg += f"Valid dimensions: {sorted(VALID_DIMENSIONS)}"
            raise ValueError(msg)
        invalid_range = {k: val for k, val in v.items() if not (0.0 <= val <= 1.0)}
        if invalid_range:
            raise ValueError(
                f"scoring_weights values must be 0.0–1.0. Out of range: {invalid_range}"
            )
        return v

    @field_validator("category")
    @classmethod
    def _valid_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{v}'. Must be one of: {sorted(VALID_CATEGORIES)}"
            )
        return v

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
        import warnings

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

        # output_kind ↔ language coherence (soft check — warn, don't fail)
        lang = self.tech_stack.language
        if self.output_kind == OutputKind.content and lang in VALID_TECH_LANGUAGES:
            warnings.warn(
                f"Archetype '{self.id}' has output_kind='content' but language='{lang}' "
                f"(a tech language). Consider using a content language or output_kind='hybrid'.",
                stacklevel=2,
            )
        elif self.output_kind == OutputKind.code and lang in VALID_CONTENT_LANGUAGES:
            warnings.warn(
                f"Archetype '{self.id}' has output_kind='code' but language='{lang}' "
                f"(a content language). Consider output_kind='content' or 'hybrid'.",
                stacklevel=2,
            )

        # output_kind ↔ category coherence (soft check)
        if (
            self.output_kind == OutputKind.content
            and self.category in VALID_CODE_CATEGORIES
        ):
            warnings.warn(
                f"Archetype '{self.id}' has output_kind='content' but category='{self.category}' "
                f"(a code category). Consider a content category: {sorted(VALID_CONTENT_CATEGORIES)}.",
                stacklevel=2,
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
    for key in ("id", "name", "description", "version", "maturity", "output_kind", "category"):
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
        "target_output", "scoring_weights",
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
