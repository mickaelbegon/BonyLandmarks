"""Tests for manifest.py: loading the student list and reading server.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from bonylandmarks.manifest import get_server_url, load_manifest


def test_load_manifest_returns_dev_fixture() -> None:
    """load_manifest() should return the 3 students from manifest.dev.json."""
    students = load_manifest()
    assert len(students) == 3


def test_load_manifest_sorted() -> None:
    """Returned list must be sorted by nom (ascending)."""
    students = load_manifest()
    noms = [s["nom"] for s in students]
    assert noms == sorted(noms)


def test_manifest_encrypt_decrypt_roundtrip(tmp_path: Path) -> None:
    """Fernet encryption/decryption of a manifest JSON roundtrips correctly."""
    key = Fernet.generate_key()
    f = Fernet(key)
    data = [{"nom": "TEST", "prenom": "User", "matricule": "99999999"}]
    plaintext = json.dumps(data).encode()
    encrypted = f.encrypt(plaintext)

    enc_file = tmp_path / "manifest.json.enc"
    enc_file.write_bytes(encrypted)

    decrypted = Fernet(key).decrypt(enc_file.read_bytes())
    result = json.loads(decrypted)
    assert result == data


def test_get_server_url_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_server_url() returns None when no server.json is found."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "fake_exe"))
    monkeypatch.chdir(tmp_path)
    assert get_server_url() is None


def test_get_server_url_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """get_server_url() returns the URL from server.json next to the executable."""
    server_json = tmp_path / "server.json"
    server_json.write_text(json.dumps({"url": "http://192.168.1.42:8765"}), encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "fake_exe"))
    url = get_server_url()
    assert url == "http://192.168.1.42:8765"
