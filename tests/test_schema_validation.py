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
    OutputKind,
    TrustLevel,
    VALID_CATEGORIES,
    VALID_CODE_CATEGORIES,
    VALID_CONTENT_CATEGORIES,
    VALID_CONTENT_LANGUAGES,
    VALID_TECH_LANGUAGES,
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
        """Languages are still validated against VALID_LANGUAGES (finite set of targets)."""
        bad = MINIMAL_YAML.replace('language: "python"', 'language: "cobol"')
        with pytest.raises(ValidationError, match="Invalid language"):
            validate_archetype_yaml(bad)

    def test_unknown_framework_allowed(self):
        """Unknown frameworks pass — creative freedom."""
        good = MINIMAL_YAML.replace('framework: "fastapi"', 'framework: "turbo_framework"')
        schema = validate_archetype_yaml(good)
        assert schema.tech_stack.framework == "turbo_framework"

    def test_cross_ecosystem_framework_rejected(self):
        """Known framework from wrong ecosystem is rejected."""
        bad = MINIMAL_YAML.replace('framework: "fastapi"', 'framework: "express"')
        with pytest.raises(ValidationError, match="Ecosystem inconsistency"):
            validate_archetype_yaml(bad)

    def test_unknown_database_allowed(self):
        """Unknown databases pass — no whitelist restriction."""
        good = MINIMAL_YAML.replace('database: "none"', 'database: "oracle"')
        schema = validate_archetype_yaml(good)
        assert schema.tech_stack.database == "oracle"

    def test_unknown_tester_allowed(self):
        """Unknown testers pass — creative freedom."""
        good = MINIMAL_YAML.replace('testing: "pytest"', 'testing: "tape"')
        schema = validate_archetype_yaml(good)
        assert schema.tech_stack.testing == "tape"

    def test_cross_ecosystem_tester_rejected(self):
        """Known tester from wrong ecosystem is rejected."""
        bad = MINIMAL_YAML.replace('testing: "pytest"', 'testing: "jest"')
        with pytest.raises(ValidationError, match="Ecosystem inconsistency"):
            validate_archetype_yaml(bad)

    def test_unknown_linter_allowed(self):
        """Unknown linters pass — creative freedom."""
        good = MINIMAL_YAML.replace('linter: "ruff"', 'linter: "superlint"')
        schema = validate_archetype_yaml(good)
        assert schema.tech_stack.linter == "superlint"

    def test_cross_ecosystem_linter_rejected(self):
        """Known linter from wrong ecosystem is rejected."""
        bad = MINIMAL_YAML.replace('linter: "ruff"', 'linter: "eslint"')
        with pytest.raises(ValidationError, match="Ecosystem inconsistency"):
            validate_archetype_yaml(bad)

    def test_unknown_type_checker_allowed(self):
        """Unknown type checkers pass — creative freedom."""
        good = MINIMAL_YAML.replace('type_checker: "mypy"', 'type_checker: "sorbet"')
        schema = validate_archetype_yaml(good)
        assert schema.tech_stack.type_checker == "sorbet"

    def test_cross_ecosystem_type_checker_rejected(self):
        """Known type checker from wrong ecosystem is rejected."""
        bad = MINIMAL_YAML.replace('type_checker: "mypy"', 'type_checker: "tsc"')
        with pytest.raises(ValidationError, match="Ecosystem inconsistency"):
            validate_archetype_yaml(bad)

    def test_none_always_passes(self):
        """'none' is a universal value that passes for any language."""
        good = MINIMAL_YAML.replace('framework: "fastapi"', 'framework: "none"')
        good = good.replace('testing: "pytest"', 'testing: "none"')
        good = good.replace('linter: "ruff"', 'linter: "none"')
        good = good.replace('type_checker: "mypy"', 'type_checker: "none"')
        schema = validate_archetype_yaml(good)
        assert schema.tech_stack.framework == "none"

    def test_real_frameworks_pass(self):
        """Real frameworks like prefect, strawberry, langgraph should pass with python."""
        for fw in ("prefect", "strawberry", "langgraph", "aiokafka", "grpcio", "fastmcp"):
            good = MINIMAL_YAML.replace('framework: "fastapi"', f'framework: "{fw}"')
            schema = validate_archetype_yaml(good)
            assert schema.tech_stack.framework == fw

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

    # ── Output kind ──

    def test_default_output_kind_is_code(self):
        schema = validate_archetype_yaml(MINIMAL_YAML)
        assert schema.output_kind == OutputKind.code

    def test_valid_output_kind_content(self):
        good = MINIMAL_YAML + 'output_kind: "content"\n'
        schema = validate_archetype_yaml(good)
        assert schema.output_kind == OutputKind.content

    def test_valid_output_kind_hybrid(self):
        good = MINIMAL_YAML + 'output_kind: "hybrid"\n'
        schema = validate_archetype_yaml(good)
        assert schema.output_kind == OutputKind.hybrid

    def test_invalid_output_kind(self):
        bad = MINIMAL_YAML + 'output_kind: "magic"\n'
        with pytest.raises(ValidationError):
            validate_archetype_yaml(bad)

    # ── Category ──

    def test_default_category_is_general(self):
        schema = validate_archetype_yaml(MINIMAL_YAML)
        assert schema.category == "general"

    def test_valid_code_category(self):
        good = MINIMAL_YAML + 'category: "backend"\n'
        schema = validate_archetype_yaml(good)
        assert schema.category == "backend"

    def test_valid_content_category(self):
        good = MINIMAL_YAML + 'category: "documentation"\n'
        schema = validate_archetype_yaml(good)
        assert schema.category == "documentation"

    def test_invalid_category(self):
        bad = MINIMAL_YAML + 'category: "blockchain"\n'
        with pytest.raises(ValidationError, match="Invalid category"):
            validate_archetype_yaml(bad)

    # ── Language classification ──

    def test_tech_and_content_languages_disjoint(self):
        """Tech and content language sets must not overlap."""
        assert VALID_TECH_LANGUAGES & VALID_CONTENT_LANGUAGES == frozenset()

    def test_content_languages_include_markdown(self):
        assert "markdown" in VALID_CONTENT_LANGUAGES

    def test_content_categories_include_documentation(self):
        assert "documentation" in VALID_CONTENT_CATEGORIES
        assert "documentation" in VALID_CATEGORIES

    def test_code_categories_still_present(self):
        for cat in ("backend", "frontend", "cli", "library"):
            assert cat in VALID_CODE_CATEGORIES
            assert cat in VALID_CATEGORIES

    # ── Content archetype round-trip ──

    def test_content_archetype_validates(self):
        """A documentation archetype with markdown, none tooling, content kind."""
        content_yaml = textwrap.dedent("""\
            id: "docs_manual"
            name: "Documentation Manual"
            description: "End-user docs"
            version: "1.0.0"
            maturity: "production"
            output_kind: "content"
            category: "documentation"

            tech_stack:
              defaults:
                language: "markdown"
                framework: "none"
                database: "none"
                testing: "none"
                linter: "none"
                type_checker: "none"

            pipeline:
              levels: ["skeleton", "contracts", "wiring", "logic"]
              enable_self_challenge: true
              enable_structural_validation: true

            structure:
              expected_dirs: ["docs/"]
              expected_files: ["README.md"]

            validation:
              checks: ["structure"]
              lint_rules: "none"
              type_strictness: "off"

            self_challenge:
              criteria:
                - "Every section has a clear heading"
        """)
        schema = validate_archetype_yaml(content_yaml)
        assert schema.output_kind == OutputKind.content
        assert schema.category == "documentation"
        assert schema.tech_stack.language == "markdown"
        assert schema.tech_stack.framework == "none"

    def test_output_kind_language_mismatch_warns(self):
        """content kind + tech language should warn (not fail)."""
        mixed = MINIMAL_YAML + 'output_kind: "content"\n'
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema = validate_archetype_yaml(mixed)
            assert schema.output_kind == OutputKind.content
            # Should have emitted a warning about python being a tech language
            coherence_warnings = [x for x in w if "tech language" in str(x.message)]
            assert len(coherence_warnings) >= 1


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
