"""Unit tests for the crypto module."""

from datetime import date

import pytest

from bonylandmarks.crypto import decrypt_bytes, derive_key, encrypt_file


def test_derive_key_is_32_bytes() -> None:
    key = derive_key("20123456", date(2000, 5, 15))
    assert len(key) == 32


def test_derive_key_normalises_case() -> None:
    k1 = derive_key("20123456", date(2000, 5, 15))
    k2 = derive_key("20123456", date(2000, 5, 15))
    assert k1 == k2


def test_decrypt_roundtrip() -> None:
    import os
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    matricule = "20999999"
    bd = date(1999, 12, 31)
    plaintext = b"GLB_HEADER_MOCK" * 100

    key = derive_key(matricule, bd)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    encrypted = nonce + aesgcm.encrypt(nonce, plaintext, None)

    recovered = decrypt_bytes(encrypted, matricule, bd)
    assert recovered == plaintext


def test_decrypt_wrong_credentials_raises() -> None:
    import os
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    matricule = "20999999"
    bd = date(1999, 12, 31)
    plaintext = b"some data"

    key = derive_key(matricule, bd)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    encrypted = nonce + aesgcm.encrypt(nonce, plaintext, None)

    with pytest.raises(ValueError, match="Decryption failed"):
        decrypt_bytes(encrypted, "wrongid", bd)


def test_decrypt_too_short_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        decrypt_bytes(b"\x00" * 5, "abc", date(2000, 1, 1))


def test_encrypt_decrypt_file_roundtrip(tmp_path) -> None:
    from bonylandmarks.crypto import decrypt_file, encrypt_file

    src = tmp_path / "scan.glb"
    dst = tmp_path / "scan.glb.enc"
    payload = b"FAKE_GLB_CONTENT_1234567890"
    src.write_bytes(payload)

    matricule = "20111111"
    bd = date(2001, 3, 7)
    encrypt_file(src, dst, matricule, bd)

    recovered = decrypt_file(dst, matricule, bd)
    assert recovered == payload
