"""AES-256-GCM encryption / decryption for BodyLoop GLB scans.

Key derivation: SHA-256(matricule.lower().strip() + ":" + YYYYMMDD)
File format   : 12-byte nonce | 16-byte GCM tag | ciphertext
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def derive_key(matricule: str, birthdate: date) -> bytes:
    """Derive a 32-byte AES-256 key from student credentials."""
    raw = f"{matricule.lower().strip()}:{birthdate.strftime('%Y%m%d')}"
    return hashlib.sha256(raw.encode()).digest()


def encrypt_file(src: Path, dst: Path, matricule: str, birthdate: date) -> None:
    """Encrypt *src* with the student's key and write to *dst*."""
    import os

    key = derive_key(matricule, birthdate)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    plaintext = src.read_bytes()
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)  # tag is appended by library
    dst.write_bytes(nonce + ciphertext)


def decrypt_bytes(data: bytes, matricule: str, birthdate: date) -> bytes:
    """Decrypt *data* using the student's key. Raises ValueError on bad credentials."""
    if len(data) < 28:  # 12 nonce + 16 tag minimum
        raise ValueError("Encrypted data too short")
    key = derive_key(matricule, birthdate)
    aesgcm = AESGCM(key)
    nonce, ciphertext = data[:12], data[12:]
    try:
        return aesgcm.decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise ValueError("Decryption failed — wrong matricule or birthdate.") from exc


def decrypt_file(src: Path, matricule: str, birthdate: date) -> bytes:
    """Read an encrypted file and return the decrypted GLB bytes."""
    return decrypt_bytes(src.read_bytes(), matricule, birthdate)
