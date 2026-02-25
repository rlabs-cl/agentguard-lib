"""Archetype dataclass — the project blueprint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TechStack:
    """Technology stack configuration."""

    language: str = "python"
    framework: str = "fastapi"
    database: str = "postgresql"
    testing: str = "pytest"
    linter: str = "ruff"
    type_checker: str = "mypy"


@dataclass
class PipelineConfig:
    """Pipeline behavior configuration."""

    levels: list[str] = field(default_factory=lambda: ["skeleton", "contracts", "wiring", "logic"])
    enable_self_challenge: bool = True
    enable_structural_validation: bool = True
    max_self_challenge_retries: int = 3


@dataclass
class ContextRecipeConfig:
    """Token budget per generation level."""

    include: list[str] = field(default_factory=list)
    max_tokens: int = 4000


@dataclass
class ValidationConfig:
    """Validation checks configuration."""

    checks: list[str] = field(default_factory=lambda: ["syntax", "lint", "types", "imports", "structure"])
    lint_rules: str = "ruff:default"
    type_strictness: str = "basic"


@dataclass
class SelfChallengeConfig:
    """Self-challenge criteria."""

    criteria: list[str] = field(default_factory=list)
    grounding_check: bool = True
    assumptions_must_declare: bool = True


@dataclass
class DebugConfig:
    """Debugging protocol configuration.

    Defines how the agent should approach debugging for this archetype's stack:
    what data sources to consult, how to form and rank hypotheses, how to
    validate a fix, and when to escalate rather than attempt a fix.
    Each field is a guideline for the agent — not prescriptive code.
    """

    data_sources: list[str] = field(default_factory=list)
    """Ordered list of sources the agent should consult: e.g. 'application logs',
    'database query plan', 'network traces', 'component state snapshot'."""

    hypothesis_protocol: list[str] = field(default_factory=list)
    """Steps for narrowing from symptom to root cause."""

    fix_protocol: list[str] = field(default_factory=list)
    """Steps to validate a fix before declaring it resolved."""

    escalation_criteria: list[str] = field(default_factory=list)
    """Conditions under which the agent should stop and report to the user
    rather than continuing to attempt a fix."""


@dataclass
class MigrationConfig:
    """Migration protocol configuration.

    Guides the agent when migrating an existing codebase toward this archetype.
    Defines risk categories, concern surface areas, and a decision protocol
    for when migration should pause and ask the user.
    """

    risk_areas: list[str] = field(default_factory=list)
    """Domain areas where data loss, API breaks, or behaviour changes are likely."""

    concern_protocol: list[str] = field(default_factory=list)
    """Questions the agent must answer before committing to any migration step."""

    incompatibility_signals: list[str] = field(default_factory=list)
    """Patterns in the source that indicate migration may not be feasible as-is."""

    step_order: list[str] = field(default_factory=list)
    """Recommended order of migration steps for this target archetype."""


@dataclass
class Archetype:
    """A project archetype — blueprint that configures the entire pipeline.

    Archetypes define the project type, tech stack, expected structure,
    validation rules, self-challenge criteria, and context recipes.

    The ``maturity`` field controls how much enterprise infrastructure the
    pipeline injects:

    - ``prototype`` — fast, compact, minimal files.
    - ``production`` — adds error boundaries, async data layer, constants.
    - ``enterprise`` — adds i18n, logging, split contexts, full a11y, test stubs.
    """

    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    maturity: str = "production"  # prototype | production | enterprise

    tech_stack: TechStack = field(default_factory=TechStack)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    structure: dict[str, list[str]] = field(default_factory=dict)
    context_recipes: dict[str, ContextRecipeConfig] = field(default_factory=dict)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    self_challenge: SelfChallengeConfig = field(default_factory=SelfChallengeConfig)
    reference_patterns: list[str] = field(default_factory=list)
    infrastructure_files: list[str] = field(default_factory=list)
    debug_config: DebugConfig = field(default_factory=DebugConfig)
    migration_config: MigrationConfig = field(default_factory=MigrationConfig)

    @classmethod
    def load(cls, archetype_id: str, overrides: dict[str, Any] | None = None) -> Archetype:
        """Load a built-in archetype by ID.

        Args:
            archetype_id: Built-in archetype name (e.g. "api_backend").
            overrides: Optional dict of dotted-path overrides.

        Returns:
            Configured Archetype instance.
        """
        from agentguard.archetypes.registry import get_archetype_registry

        arch = get_archetype_registry().get(archetype_id)
        if overrides:
            arch = _apply_overrides(arch, overrides)
        return arch

    @classmethod
    def from_file(cls, path: str | Path) -> Archetype:
        """Load an archetype from a YAML file with schema validation.

        The archetype is validated against the strict schema and
        registered in the global registry with ``trust_level=community``.

        Raises:
            ValueError: If YAML is invalid.
            pydantic.ValidationError: If schema validation fails.
        """
        from agentguard.archetypes.registry import get_archetype_registry
        from agentguard.archetypes.schema import TrustLevel

        p = Path(path)
        yaml_content = p.read_text(encoding="utf-8")
        registry = get_archetype_registry()
        entry = registry.register_validated(
            yaml_content, trust_level=TrustLevel.community,
        )
        return entry.archetype

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available built-in archetype IDs."""
        from agentguard.archetypes.registry import get_archetype_registry

        return get_archetype_registry().list_available()

    def get_expected_structure_text(self) -> str:
        """Format expected directory structure as text for prompts."""
        lines: list[str] = []
        for dir_path in self.structure.get("expected_dirs", []):
            lines.append(f"  {dir_path}/")
        for file_path in self.structure.get("expected_files", []):
            lines.append(f"  {file_path}")
        return "\n".join(lines) if lines else "(no structure constraints)"


def _from_dict(data: dict[str, Any]) -> Archetype:
    """Parse an archetype from a raw YAML dict."""
    tech_data = data.get("tech_stack", {}).get("defaults", {})
    tech = TechStack(
        language=tech_data.get("language", "python"),
        framework=tech_data.get("framework", "fastapi"),
        database=tech_data.get("database", "postgresql"),
        testing=tech_data.get("testing", "pytest"),
        linter=tech_data.get("linter", "ruff"),
        type_checker=tech_data.get("type_checker", "mypy"),
    )

    pipe_data = data.get("pipeline", {})
    pipeline = PipelineConfig(
        levels=pipe_data.get("levels", ["skeleton", "contracts", "wiring", "logic"]),
        enable_self_challenge=pipe_data.get("enable_self_challenge", True),
        enable_structural_validation=pipe_data.get("enable_structural_validation", True),
        max_self_challenge_retries=pipe_data.get("max_self_challenge_retries", 3),
    )

    context_data = data.get("context_recipes", {})
    recipes = {
        k: ContextRecipeConfig(
            include=v.get("include", []),
            max_tokens=v.get("max_tokens", 4000),
        )
        for k, v in context_data.items()
    }

    val_data = data.get("validation", {})
    validation = ValidationConfig(
        checks=val_data.get("checks", ["syntax", "lint", "types", "imports", "structure"]),
        lint_rules=val_data.get("lint_rules", "ruff:default"),
        type_strictness=val_data.get("type_strictness", "basic"),
    )

    chall_data = data.get("self_challenge", {})
    challenge = SelfChallengeConfig(
        criteria=chall_data.get("criteria", []),
        grounding_check=chall_data.get("grounding_check", True),
        assumptions_must_declare=chall_data.get("assumptions_must_declare", True),
    )

    dbg_data = data.get("debug_config", {})
    debug_config = DebugConfig(
        data_sources=dbg_data.get("data_sources", []),
        hypothesis_protocol=dbg_data.get("hypothesis_protocol", []),
        fix_protocol=dbg_data.get("fix_protocol", []),
        escalation_criteria=dbg_data.get("escalation_criteria", []),
    )

    mig_data = data.get("migration_config", {})
    migration_config = MigrationConfig(
        risk_areas=mig_data.get("risk_areas", []),
        concern_protocol=mig_data.get("concern_protocol", []),
        incompatibility_signals=mig_data.get("incompatibility_signals", []),
        step_order=mig_data.get("step_order", []),
    )

    return Archetype(
        id=data.get("id", "unknown"),
        name=data.get("name", "Unknown"),
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        maturity=data.get("maturity", "production"),
        tech_stack=tech,
        pipeline=pipeline,
        structure=data.get("structure", {}),
        context_recipes=recipes,
        validation=validation,
        self_challenge=challenge,
        reference_patterns=data.get("reference_patterns", []),
        infrastructure_files=data.get("infrastructure_files", []),
        debug_config=debug_config,
        migration_config=migration_config,
    )


def _apply_overrides(arch: Archetype, overrides: dict[str, Any]) -> Archetype:
    """Apply dotted-path overrides to an archetype (returns new instance)."""
    # Simple approach: convert to dict, override, convert back
    # For Phase 0, support a few common overrides
    for key, value in overrides.items():
        parts = key.split(".")
        if parts[0] == "tech_stack" and len(parts) == 2:
            setattr(arch.tech_stack, parts[1], value)
    return arch
