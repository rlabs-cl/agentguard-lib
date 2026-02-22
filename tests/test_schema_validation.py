"""Tests for archetype schema validation, content hashing, and registry integrity.

WS-1: Archetype Schema & Validation — ensures every archetype (built-in
or community) is strictly validated before load, publish, or use.
"""

from __future__ import annotations

import textwrap

import pytest
from pydantic import ValidationError

from agentguard.archetypes.registry import (
    ArchetypeRegistry,
    IntegrityError,
    reset_registry,
)
from agentguard.archetypes.schema import (
    Maturity,
    TrustLevel,
    compute_content_hash,
    validate_archetype_yaml,
    verify_content_hash,
)

# ── Fixtures ──────────────────────────────────────────────────────

MINIMAL_YAML = textwrap.dedent("""\
    id: "test_archetype"
    name: "Test Archetype"
    description: "A test archetype"
    version: "1.0.0"
    maturity: "prototype"

    tech_stack:
      defaults:
        language: "python"
        framework: "fastapi"
        database: "none"
        testing: "pytest"
        linter: "ruff"
        type_checker: "mypy"

    pipeline:
      levels: ["skeleton", "contracts", "wiring", "logic"]
      enable_self_challenge: true
      enable_structural_validation: true

    structure:
      expected_dirs: []
      expected_files:
        - "main.py"

    context_recipes:
      skeleton:
        include: ["spec"]
        max_tokens: 2000
      contracts:
        include: ["spec", "skeleton"]
        max_tokens: 4000
      wiring:
        include: ["contracts"]
        max_tokens: 4000
      logic:
        include: ["function_stub"]
        max_tokens: 4000

    validation:
      checks: ["syntax", "lint"]
      lint_rules: "ruff:default"
      type_strictness: "basic"

    self_challenge:
      criteria:
        - "Fulfills described purpose"
      grounding_check: true
      assumptions_must_declare: true
""")


# ══════════════════════════════════════════════════════════════════
#  SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════


class TestArchetypeSchema:
    """ArchetypeSchema validity checks."""

    def test_valid_minimal_yaml(self):
        schema = validate_archetype_yaml(MINIMAL_YAML)
        assert schema.id == "test_archetype"
        assert schema.name == "Test Archetype"
        assert schema.maturity == Maturity.prototype

    def test_valid_tech_stack(self):
        schema = validate_archetype_yaml(MINIMAL_YAML)
        assert schema.tech_stack.language == "python"
        assert schema.tech_stack.framework == "fastapi"
        assert schema.tech_stack.database == "none"

    def test_valid_pipeline(self):
        schema = validate_archetype_yaml(MINIMAL_YAML)
        assert schema.pipeline.levels == ["skeleton", "contracts", "wiring", "logic"]
        assert schema.pipeline.enable_self_challenge is True

    def test_valid_validation_section(self):
        schema = validate_archetype_yaml(MINIMAL_YAML)
        assert "syntax" in schema.validation.checks
        assert schema.validation.type_strictness == "basic"

    def test_valid_self_challenge(self):
        schema = validate_archetype_yaml(MINIMAL_YAML)
        assert len(schema.self_challenge.criteria) > 0
        assert schema.self_challenge.grounding_check is True

    # ── ID validation ──

    def test_invalid_id_uppercase(self):
        bad = MINIMAL_YAML.replace('id: "test_archetype"', 'id: "TestArch"')
        with pytest.raises(ValidationError, match="Archetype ID"):
            validate_archetype_yaml(bad)

    def test_invalid_id_starts_with_digit(self):
        bad = MINIMAL_YAML.replace('id: "test_archetype"', 'id: "123_test"')
        with pytest.raises(ValidationError, match="Archetype ID"):
            validate_archetype_yaml(bad)

    def test_invalid_id_too_short(self):
        bad = MINIMAL_YAML.replace('id: "test_archetype"', 'id: "a"')
        with pytest.raises(ValidationError):
            validate_archetype_yaml(bad)

    def test_invalid_id_with_spaces(self):
        bad = MINIMAL_YAML.replace('id: "test_archetype"', 'id: "test archetype"')
        with pytest.raises(ValidationError, match="Archetype ID"):
            validate_archetype_yaml(bad)

    # ── Version validation ──

    def test_invalid_version_not_semver(self):
        bad = MINIMAL_YAML.replace('version: "1.0.0"', 'version: "v1"')
        with pytest.raises(ValidationError, match="semver"):
            validate_archetype_yaml(bad)

    def test_valid_version_prerelease(self):
        good = MINIMAL_YAML.replace('version: "1.0.0"', 'version: "1.0.0-beta.1"')
        schema = validate_archetype_yaml(good)
        assert schema.version == "1.0.0-beta.1"

    # ── Tech stack validation ──

    def test_invalid_language(self):
        bad = MINIMAL_YAML.replace('language: "python"', 'language: "cobol"')
        with pytest.raises(ValidationError, match="Invalid language"):
            validate_archetype_yaml(bad)

    def test_invalid_framework(self):
        bad = MINIMAL_YAML.replace('framework: "fastapi"', 'framework: "turbo_framework"')
        with pytest.raises(ValidationError, match="Invalid framework"):
            validate_archetype_yaml(bad)

    def test_invalid_database(self):
        bad = MINIMAL_YAML.replace('database: "none"', 'database: "oracle"')
        with pytest.raises(ValidationError, match="Invalid database"):
            validate_archetype_yaml(bad)

    def test_invalid_tester(self):
        bad = MINIMAL_YAML.replace('testing: "pytest"', 'testing: "tape"')
        with pytest.raises(ValidationError, match="Invalid test framework"):
            validate_archetype_yaml(bad)

    def test_invalid_linter(self):
        bad = MINIMAL_YAML.replace('linter: "ruff"', 'linter: "superlint"')
        with pytest.raises(ValidationError, match="Invalid linter"):
            validate_archetype_yaml(bad)

    def test_invalid_type_checker(self):
        bad = MINIMAL_YAML.replace('type_checker: "mypy"', 'type_checker: "sorbet"')
        with pytest.raises(ValidationError, match="Invalid type checker"):
            validate_archetype_yaml(bad)

    # ── Pipeline validation ──

    def test_pipeline_must_start_with_skeleton(self):
        bad = MINIMAL_YAML.replace(
            'levels: ["skeleton", "contracts", "wiring", "logic"]',
            'levels: ["contracts", "wiring"]',
        )
        with pytest.raises(ValidationError, match="start with 'skeleton'"):
            validate_archetype_yaml(bad)

    def test_invalid_pipeline_level(self):
        bad = MINIMAL_YAML.replace(
            'levels: ["skeleton", "contracts", "wiring", "logic"]',
            'levels: ["skeleton", "design"]',
        )
        with pytest.raises(ValidationError, match="Invalid pipeline levels"):
            validate_archetype_yaml(bad)

    # ── Path traversal prevention ──

    def test_structure_path_traversal_dirs(self):
        bad = MINIMAL_YAML.replace(
            "expected_dirs: []",
            'expected_dirs: ["../../../etc/passwd"]',
        )
        with pytest.raises(ValidationError, match="Path traversal"):
            validate_archetype_yaml(bad)

    def test_structure_path_traversal_files(self):
        bad = MINIMAL_YAML.replace(
            '- "main.py"',
            '- "/etc/shadow"',
        )
        with pytest.raises(ValidationError, match="Path traversal"):
            validate_archetype_yaml(bad)

    # ── Cross-field: context recipes vs pipeline levels ──

    def test_context_recipe_unknown_level(self):
        """context_recipes with a key not in pipeline.levels should fail."""
        bad_yaml = textwrap.dedent("""\
            id: "test_archetype"
            name: "Test Archetype"
            version: "1.0.0"
            maturity: "prototype"
            tech_stack:
              defaults:
                language: "python"
                framework: "fastapi"
                database: "none"
                testing: "pytest"
                linter: "ruff"
                type_checker: "mypy"
            pipeline:
              levels: ["skeleton", "contracts"]
            structure:
              expected_files: ["main.py"]
            context_recipes:
              skeleton:
                include: ["spec"]
                max_tokens: 2000
              contracts:
                include: ["spec"]
                max_tokens: 2000
              design:
                include: ["spec"]
                max_tokens: 2000
            validation:
              checks: ["syntax"]
            self_challenge:
              criteria:
                - "OK"
        """)
        with pytest.raises(ValidationError, match="unknown pipeline levels"):
            validate_archetype_yaml(bad_yaml)

    # ── Invalid YAML ──

    def test_malformed_yaml_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid YAML"):
            validate_archetype_yaml("{{{{invalid yaml:")

    def test_non_dict_yaml_raises_value_error(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_archetype_yaml("- just a list item")

    # ── Maturity ──

    def test_invalid_maturity(self):
        bad = MINIMAL_YAML.replace('maturity: "prototype"', 'maturity: "draft"')
        with pytest.raises(ValidationError):
            validate_archetype_yaml(bad)

    # ── Validation section ──

    def test_invalid_check_name(self):
        bad = MINIMAL_YAML.replace(
            'checks: ["syntax", "lint"]',
            'checks: ["syntax", "fuzz"]',
        )
        with pytest.raises(ValidationError, match="Invalid checks"):
            validate_archetype_yaml(bad)

    def test_invalid_type_strictness(self):
        bad = MINIMAL_YAML.replace(
            'type_strictness: "basic"',
            'type_strictness: "extreme"',
        )
        with pytest.raises(ValidationError, match="String should match pattern"):
            validate_archetype_yaml(bad)

    # ── Self challenge ──

    def test_empty_criterion_rejected(self):
        bad = MINIMAL_YAML.replace(
            '- "Fulfills described purpose"',
            '- ""',
        )
        with pytest.raises(ValidationError, match="empty"):
            validate_archetype_yaml(bad)


# ══════════════════════════════════════════════════════════════════
#  CONTENT HASHING
# ══════════════════════════════════════════════════════════════════


class TestContentHashing:
    """Deterministic content hash computation and verification."""

    def test_hash_deterministic(self):
        h1 = compute_content_hash(MINIMAL_YAML)
        h2 = compute_content_hash(MINIMAL_YAML)
        assert h1 == h2

    def test_hash_is_sha256_hex(self):
        h = compute_content_hash(MINIMAL_YAML)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_changes_on_content_change(self):
        h1 = compute_content_hash(MINIMAL_YAML)
        modified = MINIMAL_YAML.replace('name: "Test Archetype"', 'name: "Modified"')
        h2 = compute_content_hash(modified)
        assert h1 != h2

    def test_hash_ignores_whitespace_reordering(self):
        """Same logical content with different YAML whitespace → same hash."""
        compact = textwrap.dedent("""\
            id: test_archetype
            name: Test Archetype
            description: A test archetype
            version: "1.0.0"
        """)
        spaced = textwrap.dedent("""\
            id:   test_archetype
            name:   Test Archetype
            description:   A test archetype
            version:   "1.0.0"
        """)
        assert compute_content_hash(compact) == compute_content_hash(spaced)

    def test_verify_content_hash_match(self):
        h = compute_content_hash(MINIMAL_YAML)
        assert verify_content_hash(MINIMAL_YAML, h) is True

    def test_verify_content_hash_mismatch(self):
        assert verify_content_hash(MINIMAL_YAML, "0" * 64) is False


# ══════════════════════════════════════════════════════════════════
#  BUILTIN ARCHETYPES VALIDATION
# ══════════════════════════════════════════════════════════════════


class TestBuiltinArchetypes:
    """All built-in archetypes must pass strict schema validation."""

    EXPECTED_BUILTINS = [
        "api_backend",
        "cli_tool",
        "library",
        "react_spa",
        "script",
        "web_app",
    ]

    def test_all_builtins_present(self):
        reg = ArchetypeRegistry(strict=True)
        available = reg.list_available()
        for name in self.EXPECTED_BUILTINS:
            assert name in available, f"Missing built-in: {name}"

    def test_builtins_strict_validation(self):
        """Load all builtins in strict mode — no exceptions."""
        reg = ArchetypeRegistry(strict=True)
        for name in self.EXPECTED_BUILTINS:
            entry = reg.get_entry(name)
            assert entry.trust_level == TrustLevel.official
            assert len(entry.content_hash) == 64

    def test_builtin_hashes_unique(self):
        """Each built-in has a unique content hash."""
        reg = ArchetypeRegistry(strict=True)
        hashes = [reg.get_content_hash(n) for n in self.EXPECTED_BUILTINS]
        assert len(set(hashes)) == len(hashes)


# ══════════════════════════════════════════════════════════════════
#  REGISTRY INTEGRITY
# ══════════════════════════════════════════════════════════════════


class TestRegistryIntegrity:
    """Registry with trust levels and content hashing."""

    def setup_method(self):
        reset_registry()

    def test_register_validated_succeeds(self):
        reg = ArchetypeRegistry(strict=True)
        entry = reg.register_validated(MINIMAL_YAML, trust_level=TrustLevel.community)
        assert entry.archetype.id == "test_archetype"
        assert entry.trust_level == TrustLevel.community
        assert len(entry.content_hash) == 64
        assert entry.schema is not None

    def test_register_validated_with_correct_hash(self):
        h = compute_content_hash(MINIMAL_YAML)
        reg = ArchetypeRegistry(strict=True)
        entry = reg.register_validated(
            MINIMAL_YAML, trust_level=TrustLevel.community, expected_hash=h
        )
        assert entry.content_hash == h

    def test_register_validated_wrong_hash_raises(self):
        reg = ArchetypeRegistry(strict=True)
        with pytest.raises(IntegrityError, match="hash mismatch"):
            reg.register_validated(
                MINIMAL_YAML,
                trust_level=TrustLevel.community,
                expected_hash="0" * 64,
            )

    def test_register_validated_invalid_yaml_raises(self):
        reg = ArchetypeRegistry(strict=True)
        bad_yaml = 'id: "BAD"\nname: "Bad"'
        with pytest.raises((ValueError, ValidationError)):
            reg.register_validated(bad_yaml, trust_level=TrustLevel.community)

    def test_shadow_protection_community_cannot_override_official(self):
        """Community archetype cannot override an official one."""
        reg = ArchetypeRegistry(strict=True)
        # Register as official first
        reg.register_validated(MINIMAL_YAML, trust_level=TrustLevel.official)
        # Try to override with community — should fail
        with pytest.raises(ValueError, match="Cannot override official"):
            reg.register_validated(MINIMAL_YAML, trust_level=TrustLevel.community)

    def test_register_remote_succeeds(self):
        h = compute_content_hash(MINIMAL_YAML)
        reg = ArchetypeRegistry(strict=True)
        entry = reg.register_remote(
            "test_archetype", MINIMAL_YAML, h, trust_level=TrustLevel.community
        )
        assert entry.archetype.id == "test_archetype"
        assert entry.content_hash == h

    def test_register_remote_wrong_hash_raises(self):
        reg = ArchetypeRegistry(strict=True)
        with pytest.raises(IntegrityError, match="hash mismatch"):
            reg.register_remote(
                "test_archetype", MINIMAL_YAML, "0" * 64
            )

    def test_register_remote_id_mismatch_raises(self):
        h = compute_content_hash(MINIMAL_YAML)
        reg = ArchetypeRegistry(strict=True)
        with pytest.raises(IntegrityError, match="ID mismatch"):
            reg.register_remote("wrong_id", MINIMAL_YAML, h)

    def test_is_registered(self):
        reg = ArchetypeRegistry(strict=True)
        assert reg.is_registered("api_backend") is True
        assert reg.is_registered("nonexistent_xyz") is False

    def test_get_trust_level(self):
        reg = ArchetypeRegistry(strict=True)
        assert reg.get_trust_level("api_backend") == TrustLevel.official

    def test_get_content_hash(self):
        reg = ArchetypeRegistry(strict=True)
        h = reg.get_content_hash("api_backend")
        assert len(h) == 64


class TestResetRegistry:
    """Singleton reset helper."""

    def test_reset_clears_singleton(self):
        from agentguard.archetypes.registry import get_archetype_registry

        reset_registry()
        reg1 = get_archetype_registry()
        reset_registry()
        reg2 = get_archetype_registry()
        assert reg1 is not reg2
