"""Archetype encryption for on-disk IP protection.

Marketplace archetypes are stored encrypted at rest using AES-256-GCM,
keyed to the user's AGENTGUARD_API_KEY.  Without the key, the ``.agx``
file on disk is unreadable — even if copied to another machine.

Key derivation:
    PBKDF2-HMAC-SHA256(
        password = AGENTGUARD_API_KEY,
        salt     = archetype_slug (utf-8, padded to 16 bytes),
        iterations = 100_000,
    ) → 32-byte AES key

File format (``.agx``):
    salt(16) ‖ nonce(12) ‖ ciphertext ‖ tag(16)

Graceful fallback:
    If the ``cryptography`` package is not installed, the module logs a
    warning and falls back to plaintext ``.yaml`` storage.  This lets
    the library work in lightweight environments (CI, notebooks) while
    production installs get full encryption.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────

AGX_EXTENSION: Final[str] = ".agx"
YAML_EXTENSION: Final[str] = ".yaml"
_KDF_ITERATIONS: Final[int] = 100_000
_SALT_LEN: Final[int] = 16
_NONCE_LEN: Final[int] = 12
_TAG_LEN: Final[int] = 16
_KEY_LEN: Final[int] = 32  # AES-256

# ── crypto availability ──────────────────────────────────────────────

_HAS_CRYPTO: bool = False

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    _HAS_CRYPTO = True
except ImportError:
    pass


def crypto_available() -> bool:
    """Return True if encryption support is available."""
    return _HAS_CRYPTO


# ── key derivation ───────────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Read the user's API key from the environment."""
    return os.environ.get("AGENTGUARD_API_KEY")


def _derive_key(api_key: str, slug: str) -> bytes:
    """Derive a 256-bit AES key from the API key + archetype slug.

    Uses PBKDF2-HMAC-SHA256 with a deterministic salt derived from the
    slug (so the same API key + slug always produces the same key — no
    need to store the salt separately; it IS the slug).
    """
    if not _HAS_CRYPTO:
        raise RuntimeError("cryptography package not installed")

    # Deterministic 16-byte salt from the slug
    salt = hashlib.sha256(slug.encode("utf-8")).digest()[:_SALT_LEN]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_LEN,
        salt=salt,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(api_key.encode("utf-8"))


# ── encrypt / decrypt ────────────────────────────────────────────────

def encrypt_yaml(yaml_content: str, slug: str, api_key: str | None = None) -> bytes:
    """Encrypt archetype YAML content, returning the ``.agx`` binary blob.

    Args:
        yaml_content: Raw YAML string to encrypt.
        slug: Archetype slug (used in key derivation).
        api_key: API key override.  Falls back to ``AGENTGUARD_API_KEY`` env var.

    Returns:
        Binary blob: ``salt(16) ‖ nonce(12) ‖ ciphertext ‖ tag(16)``.

    Raises:
        RuntimeError: If ``cryptography`` is not installed.
        ValueError: If no API key is available.
    """
    if not _HAS_CRYPTO:
        raise RuntimeError(
            "cryptography package required for archetype encryption. "
            "Install with: pip install 'rlabs-agentguard[platform]'"
        )

    key_str = api_key or _get_api_key()
    if not key_str:
        raise ValueError(
            "AGENTGUARD_API_KEY not set — cannot encrypt archetype. "
            "Set the environment variable or pass api_key explicitly."
        )

    key = _derive_key(key_str, slug)
    salt = hashlib.sha256(slug.encode("utf-8")).digest()[:_SALT_LEN]
    nonce = os.urandom(_NONCE_LEN)

    aesgcm = AESGCM(key)
    ciphertext_with_tag = aesgcm.encrypt(
        nonce, yaml_content.encode("utf-8"), slug.encode("utf-8")  # AAD = slug
    )

    return salt + nonce + ciphertext_with_tag


def decrypt_yaml(blob: bytes, slug: str, api_key: str | None = None) -> str:
    """Decrypt an ``.agx`` blob back to YAML content.

    Args:
        blob: Raw bytes from a ``.agx`` file.
        slug: Archetype slug (used in key derivation + AAD verification).
        api_key: API key override.  Falls back to ``AGENTGUARD_API_KEY`` env var.

    Returns:
        The decrypted YAML string.

    Raises:
        RuntimeError: If ``cryptography`` is not installed.
        ValueError: If no API key, or decryption fails (wrong key, tampered data).
    """
    if not _HAS_CRYPTO:
        raise RuntimeError(
            "cryptography package required for archetype decryption. "
            "Install with: pip install 'rlabs-agentguard[platform]'"
        )

    key_str = api_key or _get_api_key()
    if not key_str:
        raise ValueError(
            "AGENTGUARD_API_KEY not set — cannot decrypt archetype. "
            "Set the environment variable or pass api_key explicitly."
        )

    min_size = _SALT_LEN + _NONCE_LEN + _TAG_LEN
    if len(blob) < min_size:
        raise ValueError(f"Invalid .agx file: too small ({len(blob)} bytes, min {min_size})")

    # Parse the binary format
    _salt = blob[:_SALT_LEN]  # not used directly — key derives salt from slug
    nonce = blob[_SALT_LEN : _SALT_LEN + _NONCE_LEN]
    ciphertext_with_tag = blob[_SALT_LEN + _NONCE_LEN :]

    key = _derive_key(key_str, slug)
    aesgcm = AESGCM(key)

    try:
        plaintext = aesgcm.decrypt(
            nonce, ciphertext_with_tag, slug.encode("utf-8")  # AAD must match
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to decrypt archetype '{slug}'. "
            "This usually means the AGENTGUARD_API_KEY doesn't match the key "
            "used when the archetype was downloaded. Try re-downloading."
        ) from exc

    return plaintext.decode("utf-8")


# ── file I/O helpers ─────────────────────────────────────────────────

def save_archetype(
    yaml_content: str,
    slug: str,
    directory: str | Path,
    *,
    api_key: str | None = None,
    force_plaintext: bool = False,
) -> Path:
    """Save an archetype to disk, encrypted if possible.

    Args:
        yaml_content: Raw YAML content.
        slug: Archetype slug.
        directory: Target directory (e.g. ``~/.agentguard/archetypes/``).
        api_key: API key override.
        force_plaintext: Skip encryption even if available (for builtins).

    Returns:
        Path to the saved file (``.agx`` if encrypted, ``.yaml`` if not).
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    can_encrypt = _HAS_CRYPTO and not force_plaintext and (_get_api_key() or api_key)

    if can_encrypt:
        blob = encrypt_yaml(yaml_content, slug, api_key=api_key)
        path = target_dir / f"{slug}{AGX_EXTENSION}"
        path.write_bytes(blob)
        logger.info("Saved encrypted archetype '%s' → %s", slug, path)
    else:
        if not force_plaintext:
            logger.warning(
                "Saving archetype '%s' in plaintext — "
                "install 'cryptography' and set AGENTGUARD_API_KEY for encryption.",
                slug,
            )
        path = target_dir / f"{slug}{YAML_EXTENSION}"
        path.write_text(yaml_content, encoding="utf-8")
        logger.info("Saved archetype '%s' → %s", slug, path)

    return path


def load_archetype(
    slug: str,
    directory: str | Path,
    *,
    api_key: str | None = None,
) -> str:
    """Load an archetype from disk, decrypting if needed.

    Tries ``.agx`` first (encrypted), falls back to ``.yaml`` (plaintext).

    Args:
        slug: Archetype slug.
        directory: Directory containing archetype files.
        api_key: API key override.

    Returns:
        The YAML content string.

    Raises:
        FileNotFoundError: If neither ``.agx`` nor ``.yaml`` exists.
    """
    target_dir = Path(directory)

    agx_path = target_dir / f"{slug}{AGX_EXTENSION}"
    yaml_path = target_dir / f"{slug}{YAML_EXTENSION}"

    if agx_path.exists():
        blob = agx_path.read_bytes()
        return decrypt_yaml(blob, slug, api_key=api_key)

    if yaml_path.exists():
        logger.debug("Loading plaintext archetype '%s' from %s", slug, yaml_path)
        return yaml_path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"Archetype '{slug}' not found in {target_dir}. "
        f"Expected {agx_path.name} or {yaml_path.name}."
    )
