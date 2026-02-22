"""Archetype registry — loads, validates, and manages archetypes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agentguard.archetypes.base import Archetype, _from_dict
from agentguard.archetypes.schema import (
    ArchetypeSchema,
    TrustLevel,
    compute_content_hash,
    validate_archetype_yaml,
)

_BUILTIN_DIR = Path(__file__).parent / "builtin"
_log = logging.getLogger(__name__)


class RegistryEntry:
    """Wrapper that pairs an Archetype with its trust metadata."""

    __slots__ = ("archetype", "trust_level", "content_hash", "schema")

    def __init__(
        self,
        archetype: Archetype,
        trust_level: TrustLevel,
        content_hash: str,
        schema: ArchetypeSchema | None = None,
    ) -> None:
        self.archetype = archetype
        self.trust_level = trust_level
        self.content_hash = content_hash
        self.schema = schema


class ArchetypeRegistry:
    """Registry for loading, validating, and caching archetypes.

    Every archetype goes through schema validation before registration.
    The registry tracks trust level and content hash for integrity checks.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._loaded_builtins = False
        self._strict = strict  # If True, schema validation errors raise

    def get(self, archetype_id: str) -> Archetype:
        """Get an archetype by ID.

        Raises:
            KeyError: If archetype not found.
        """
        if not self._loaded_builtins:
            self._load_builtins()

        entry = self._entries.get(archetype_id)
        if entry is None:
            raise KeyError(
                f"Archetype '{archetype_id}' not found. "
                f"Available: {list(self._entries.keys())}"
            )
        return entry.archetype

    def get_entry(self, archetype_id: str) -> RegistryEntry:
        """Get the full registry entry (archetype + trust metadata).

        Raises:
            KeyError: If archetype not found.
        """
        if not self._loaded_builtins:
            self._load_builtins()

        entry = self._entries.get(archetype_id)
        if entry is None:
            raise KeyError(
                f"Archetype '{archetype_id}' not found. "
                f"Available: {list(self._entries.keys())}"
            )
        return entry

    def register(self, archetype: Archetype, *, trust_level: TrustLevel = TrustLevel.community) -> None:
        """Register a custom archetype (legacy — no schema validation).

        .. deprecated::
            Use :meth:`register_validated` instead for full schema
            validation and content hashing.
        """
        import warnings

        warnings.warn(
            "register() is deprecated — use register_validated() for schema "
            "validation and integrity hashing",
            DeprecationWarning,
            stacklevel=2,
        )
        self._entries[archetype.id] = RegistryEntry(
            archetype=archetype,
            trust_level=trust_level,
            content_hash="",
        )

    def register_validated(
        self,
        yaml_content: str,
        *,
        trust_level: TrustLevel = TrustLevel.community,
        expected_hash: str | None = None,
    ) -> RegistryEntry:
        """Register an archetype from YAML with full schema validation.

        Args:
            yaml_content: Raw YAML text of the archetype.
            trust_level: Trust classification.
            expected_hash: If provided, verify content hash matches.

        Returns:
            The RegistryEntry for the newly registered archetype.

        Raises:
            ValueError: If YAML is invalid.
            pydantic.ValidationError: If schema validation fails.
            IntegrityError: If content hash doesn't match.
        """
        # 1. Validate schema
        schema = validate_archetype_yaml(yaml_content)

        # 2. Compute and optionally verify content hash
        content_hash = compute_content_hash(yaml_content)
        if expected_hash and content_hash != expected_hash:
            raise IntegrityError(
                f"Content hash mismatch for archetype '{schema.id}': "
                f"expected {expected_hash}, got {content_hash}"
            )

        # 3. Prevent community archetypes from shadowing official ones
        existing = self._entries.get(schema.id)
        if existing and existing.trust_level == TrustLevel.official and trust_level != TrustLevel.official:
            raise ValueError(
                f"Cannot override official archetype '{schema.id}' with {trust_level.value} archetype"
            )

        # 4. Build the runtime Archetype from validated data
        data = yaml.safe_load(yaml_content)
        archetype = _from_dict(data)

        entry = RegistryEntry(
            archetype=archetype,
            trust_level=trust_level,
            content_hash=content_hash,
            schema=schema,
        )
        self._entries[archetype.id] = entry
        _log.info(
            "Registered archetype '%s' v%s [%s] hash=%s…",
            archetype.id, archetype.version, trust_level.value, content_hash[:12],
        )
        return entry

    def register_remote(
        self,
        archetype_id: str,
        yaml_content: str,
        content_hash: str,
        *,
        trust_level: TrustLevel = TrustLevel.community,
    ) -> RegistryEntry:
        """Register a marketplace archetype with mandatory hash verification.

        This is the entry point for archetypes downloaded from the platform.
        The content_hash serves as the source-of-truth integrity check.

        Args:
            archetype_id: Expected archetype ID (must match YAML content).
            yaml_content: Raw YAML from the marketplace.
            content_hash: Hash from the marketplace database.
            trust_level: Trust level assigned by the platform.

        Returns:
            The RegistryEntry.

        Raises:
            IntegrityError: If hash or ID mismatch.
            pydantic.ValidationError: If schema invalid.
        """
        entry = self.register_validated(
            yaml_content,
            trust_level=trust_level,
            expected_hash=content_hash,
        )

        if entry.archetype.id != archetype_id:
            # Roll back
            self._entries.pop(entry.archetype.id, None)
            raise IntegrityError(
                f"Archetype ID mismatch: expected '{archetype_id}', "
                f"YAML contains '{entry.archetype.id}'"
            )

        return entry

    def is_registered(self, archetype_id: str) -> bool:
        """Check if an archetype is registered."""
        if not self._loaded_builtins:
            self._load_builtins()
        return archetype_id in self._entries

    def get_trust_level(self, archetype_id: str) -> TrustLevel:
        """Get the trust level of a registered archetype."""
        return self.get_entry(archetype_id).trust_level

    def get_content_hash(self, archetype_id: str) -> str:
        """Get the content hash of a registered archetype."""
        return self.get_entry(archetype_id).content_hash

    def list_available(self) -> list[str]:
        """List available archetype IDs."""
        if not self._loaded_builtins:
            self._load_builtins()
        return sorted(self._entries.keys())

    def _load_builtins(self) -> None:
        """Load all YAML archetypes from the builtin directory."""
        self._loaded_builtins = True

        if not _BUILTIN_DIR.exists():
            return

        for yaml_file in sorted(_BUILTIN_DIR.glob("*.yaml")):
            raw = yaml_file.read_text(encoding="utf-8")
            try:
                if self._strict:
                    entry = self.register_validated(
                        raw, trust_level=TrustLevel.official
                    )
                    _log.debug("Loaded built-in archetype: %s", entry.archetype.id)
                else:
                    # Fallback for backward compat
                    data: dict[str, Any] = yaml.safe_load(raw)
                    archetype = _from_dict(data)
                    content_hash = compute_content_hash(raw)
                    self._entries[archetype.id] = RegistryEntry(
                        archetype=archetype,
                        trust_level=TrustLevel.official,
                        content_hash=content_hash,
                    )
            except Exception:
                if self._strict:
                    raise
                _log.warning("Failed to validate built-in archetype %s", yaml_file.name, exc_info=True)


class IntegrityError(Exception):
    """Raised when archetype content hash doesn't match expected value."""


# Module-level singleton
_default_registry: ArchetypeRegistry | None = None


def get_archetype_registry() -> ArchetypeRegistry:
    """Get the default archetype registry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ArchetypeRegistry(strict=True)
    return _default_registry


def reset_registry() -> None:
    """Reset the singleton registry (useful for testing)."""
    global _default_registry
    _default_registry = None
