"""Client-side cryptographic utilities for per-key archetype decryption."""

from __future__ import annotations

from typing import Any


def decrypt_for_key(data: dict[str, Any], salt: str, key_prefix: str) -> str:
    """Decrypt archetype content that was encrypted per-key on the server.

    Uses HKDF to derive the same AES-256 key from salt + key_prefix,
    then decrypts with AES-256-GCM.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        raise ImportError(
            "cryptography is required for encrypted archetype downloads.\n"
            'Install it with: pip install "rlabs-agentguard[platform]"'
        ) from None

    ikm = bytes.fromhex(salt)
    info = f"agentguard-archetype-{key_prefix}".encode()
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    )
    derived_key = hkdf.derive(ikm)

    nonce = bytes.fromhex(data["nonce"])
    ciphertext = bytes.fromhex(data["ciphertext"])
    aesgcm = AESGCM(derived_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")
