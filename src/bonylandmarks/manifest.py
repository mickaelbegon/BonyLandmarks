"""Student manifest loader and server configuration reader."""

from __future__ import annotations

import importlib.resources
import json
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

_DEV_KEY = b"VGVzdEtleUZvckJvbnlMYW5kbWFya3NEZXZlbA=="

_manifest_key_env = os.environ.get("MANIFEST_KEY", "")
_MANIFEST_KEY: bytes = _manifest_key_env.encode() if _manifest_key_env else _DEV_KEY


def load_manifest() -> list[dict]:
    """Return [{"nom": ..., "prenom": ..., "matricule": ...}, ...] sorted by nom."""
    # 1. Try encrypted manifest shipped with the package
    try:
        enc_data = (
            importlib.resources.files("bonylandmarks")
            .joinpath("manifest.json.enc")
            .read_bytes()
        )
        f = Fernet(_MANIFEST_KEY)
        plaintext = f.decrypt(enc_data)
        students = json.loads(plaintext)
        return sorted(students, key=lambda s: (s["nom"], s["prenom"]))
    except (FileNotFoundError, InvalidToken, Exception):
        pass

    # 2. Fall back to dev fixture alongside this file
    try:
        dev_path = Path(__file__).parent / "manifest.dev.json"
        students = json.loads(dev_path.read_text(encoding="utf-8"))
        return sorted(students, key=lambda s: (s["nom"], s["prenom"]))
    except Exception:
        pass

    return []


def get_server_url() -> str | None:
    """Read server URL from server.json next to the executable (or cwd in dev)."""
    candidates = [
        Path(sys.executable).parent / "server.json",
        Path.cwd() / "server.json",
    ]
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            url = data.get("url", "")
            if url:
                return url.rstrip("/")
        except Exception:
            continue
    return None
