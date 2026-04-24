"""Tests for the tiered confidentiality_policy feature introduced in 0.15.0."""

from __future__ import annotations

import pytest
import yaml

from agentguard.archetypes.base import Archetype
from agentguard.archetypes.schema import ConfidentialityPolicy
from agentguard.mcp.agent_tools import (
    _CONFIDENTIALITY_DIRECTIVE,
    _CONFIDENTIALITY_DIRECTIVES,
    _confidentiality_directive_for,
)


class TestConfidentialityPolicyEnum:
    def test_four_levels_exist(self) -> None:
        assert ConfidentialityPolicy.transparent.value == "transparent"
        assert ConfidentialityPolicy.attribution.value == "attribution"
        assert ConfidentialityPolicy.paraphrase.value == "paraphrase"
        assert ConfidentialityPolicy.proprietary.value == "proprietary"

    def test_exhaustive_list(self) -> None:
        values = {p.value for p in ConfidentialityPolicy}
        assert values == {"transparent", "attribution", "paraphrase", "proprietary"}


class TestDirectiveResolution:
    @pytest.mark.parametrize(
        "policy_value", ["transparent", "attribution", "paraphrase", "proprietary"]
    )
    def test_each_policy_resolves_to_unique_text(self, policy_value: str) -> None:
        text = _confidentiality_directive_for(policy_value)
        assert text == _CONFIDENTIALITY_DIRECTIVES[policy_value]
        assert len(text) > 50

    def test_all_directives_are_distinct(self) -> None:
        directives = set(_CONFIDENTIALITY_DIRECTIVES.values())
        assert len(directives) == 4

    def test_enum_value_accepted(self) -> None:
        text = _confidentiality_directive_for(ConfidentialityPolicy.transparent)
        assert text == _CONFIDENTIALITY_DIRECTIVES["transparent"]

    def test_unknown_policy_falls_back_to_paraphrase(self) -> None:
        text = _confidentiality_directive_for("unknown_value")
        assert text == _CONFIDENTIALITY_DIRECTIVES["paraphrase"]

    def test_transparent_mentions_auditability(self) -> None:
        text = _confidentiality_directive_for("transparent")
        assert "MAY reproduce" in text or "may reproduce" in text.lower()

    def test_proprietary_forbids_paraphrase(self) -> None:
        text = _confidentiality_directive_for("proprietary")
        assert "MUST NOT" in text
        assert "paraphrase" in text.lower()

    def test_paraphrase_permits_explanation_forbids_verbatim(self) -> None:
        text = _confidentiality_directive_for("paraphrase")
        assert "paraphrase" in text.lower() or "explain" in text.lower()
        assert "verbatim" in text.lower()


class TestArchetypeIntegration:
    def test_default_policy_is_paraphrase(self) -> None:
        arch = Archetype(id="test_default", name="Test")
        assert arch.confidentiality_policy == "paraphrase"

    def test_explicit_policy_applied(self) -> None:
        arch = Archetype(
            id="test_transparent",
            name="Test",
            confidentiality_policy="transparent",
        )
        assert arch.confidentiality_policy == "transparent"

    def test_from_dict_reads_policy(self) -> None:
        from agentguard.archetypes.base import _from_dict

        data = {
            "id": "explicit_proprietary",
            "name": "Explicit",
            "confidentiality_policy": "proprietary",
        }
        arch = _from_dict(data)
        assert arch.confidentiality_policy == "proprietary"

    def test_from_dict_default_when_unset(self) -> None:
        from agentguard.archetypes.base import _from_dict

        arch = _from_dict({"id": "no_policy_set", "name": "No Policy"})
        assert arch.confidentiality_policy == "paraphrase"


class TestBuiltinArchetypesAreTransparent:
    @pytest.mark.parametrize(
        "builtin_id",
        [
            "api_backend",
            "cli_tool",
            "debug_backend",
            "debug_frontend",
            "library",
            "react_spa",
            "script",
            "software_architecture",
            "web_app",
        ],
    )
    def test_builtin_declares_transparent(self, builtin_id: str) -> None:
        arch = Archetype.load(builtin_id)
        assert arch.confidentiality_policy == "transparent", (
            f"Built-in archetype '{builtin_id}' should declare "
            f"confidentiality_policy: transparent (got {arch.confidentiality_policy!r})"
        )


class TestSchemaValidation:
    def test_schema_accepts_all_four_policies(self) -> None:
        from agentguard.archetypes.schema import ArchetypeSchema

        for policy in ConfidentialityPolicy:
            schema = ArchetypeSchema(
                id="test_archetype",
                name="Test",
                confidentiality_policy=policy,
            )
            assert schema.confidentiality_policy == policy

    def test_schema_rejects_invalid_policy_value(self) -> None:
        from pydantic import ValidationError
        from agentguard.archetypes.schema import ArchetypeSchema

        with pytest.raises(ValidationError):
            ArchetypeSchema(
                id="test_archetype",
                name="Test",
                confidentiality_policy="something_not_allowed",
            )

    def test_yaml_without_policy_field_validates_as_paraphrase(self) -> None:
        from agentguard.archetypes.schema import ArchetypeSchema

        raw = {"id": "minimal_archetype", "name": "Minimal"}
        schema = ArchetypeSchema(**raw)
        assert schema.confidentiality_policy == ConfidentialityPolicy.paraphrase


class TestBackwardCompatibility:
    def test_historic_constant_still_exists(self) -> None:
        assert isinstance(_CONFIDENTIALITY_DIRECTIVE, str)
        assert len(_CONFIDENTIALITY_DIRECTIVE) > 50

    def test_historic_constant_equals_proprietary_text(self) -> None:
        """External consumers importing the old symbol get the strictest directive."""
        assert _CONFIDENTIALITY_DIRECTIVE == _CONFIDENTIALITY_DIRECTIVES["proprietary"]
