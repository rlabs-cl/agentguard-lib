"""Tests for WS-2: Registry as Source of Truth.

Verifies that:
- Default singleton uses strict mode
- Archetype.from_file() validates through the registry
- Legacy register() emits deprecation warning
- Pipeline logs warnings for unregistered archetypes
- HTTP API and MCP expose trust_level + content_hash
- Verify endpoint works correctly
"""

from __future__ import annotations

import warnings

import pytest

from agentguard.archetypes.registry import (
    ArchetypeRegistry,
    get_archetype_registry,
    reset_registry,
)
from agentguard.archetypes.schema import TrustLevel


class TestRegistrySingleton:
    """Default singleton must use strict mode."""

    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_default_singleton_is_strict(self):
        reg = get_archetype_registry()
        assert reg._strict is True

    def test_builtins_loaded_with_hashes(self):
        reg = get_archetype_registry()
        # Only check built-in (official) archetypes — user-installed community
        # archetypes may also be present in the registry when the developer has
        # ~/.agentguard/archetypes/ populated.
        official_ids = [
            aid for aid in reg.list_available()
            if reg.get_entry(aid).trust_level == TrustLevel.official
        ]
        assert len(official_ids) > 0, "No official archetypes loaded"
        for arch_id in official_ids:
            entry = reg.get_entry(arch_id)
            assert entry.trust_level == TrustLevel.official
            assert len(entry.content_hash) == 64

    def test_builtins_all_schema_validated(self):
        """Strict mode means all builtins have schema objects."""
        reg = get_archetype_registry()
        for arch_id in reg.list_available():
            entry = reg.get_entry(arch_id)
            assert entry.schema is not None, f"{arch_id} missing schema"


class TestLegacyRegisterDeprecation:
    """Legacy register() must emit a deprecation warning."""

    def test_register_emits_warning(self):
        from agentguard.archetypes.base import Archetype

        reg = ArchetypeRegistry(strict=True)
        arch = Archetype.load("api_backend")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.register(arch)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "register_validated" in str(w[0].message)


class TestFromFileValidation:
    """Archetype.from_file() must go through schema validation."""

    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_from_file_valid_builtin(self, tmp_path):
        """from_file with a valid YAML should succeed and register it."""
        from agentguard.archetypes.base import Archetype

        yaml_content = (
            'id: "test_from_file"\n'
            'name: "Test From File"\n'
            'version: "1.0.0"\n'
            'maturity: "prototype"\n'
            "tech_stack:\n"
            "  defaults:\n"
            '    language: "python"\n'
            '    framework: "fastapi"\n'
            '    database: "none"\n'
            '    testing: "pytest"\n'
            '    linter: "ruff"\n'
            '    type_checker: "mypy"\n'
            "pipeline:\n"
            '  levels: ["skeleton", "contracts", "wiring", "logic"]\n'
            "structure:\n"
            "  expected_files:\n"
            '    - "main.py"\n'
            "context_recipes:\n"
            "  skeleton:\n"
            '    include: ["spec"]\n'
            "    max_tokens: 2000\n"
            "  contracts:\n"
            '    include: ["spec"]\n'
            "    max_tokens: 2000\n"
            "  wiring:\n"
            '    include: ["spec"]\n'
            "    max_tokens: 2000\n"
            "  logic:\n"
            '    include: ["spec"]\n'
            "    max_tokens: 2000\n"
            "validation:\n"
            '  checks: ["syntax"]\n'
            "self_challenge:\n"
            "  criteria:\n"
            '    - "Works"\n'
        )
        f = tmp_path / "test.yaml"
        f.write_text(yaml_content, encoding="utf-8")

        arch = Archetype.from_file(f)
        assert arch.id == "test_from_file"

        # Should now be in the registry
        reg = get_archetype_registry()
        assert reg.is_registered("test_from_file")
        entry = reg.get_entry("test_from_file")
        assert entry.trust_level == TrustLevel.community
        assert len(entry.content_hash) == 64

    def test_from_file_invalid_rejects(self, tmp_path):
        """from_file with invalid YAML must raise, not silently load."""
        from agentguard.archetypes.base import Archetype

        f = tmp_path / "bad.yaml"
        f.write_text('id: "BAD"\nname: "x"', encoding="utf-8")

        with pytest.raises((ValueError, TypeError)):
            Archetype.from_file(f)


    # Note: TestVerifyEndpoint and TestArchetypeSchemaFields removed in v0.6.0
    # (HTTP server module deleted).
