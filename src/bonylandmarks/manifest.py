"""Student manifest loader and server configuration reader."""

from __future__ import annotations

import importlib.resources
import json
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


def load_manifest_from_excel(path: Path) -> list[dict]:
    """Load the full student roster (with date_naissance) from an Excel file.

    Expected columns: Matricule | Nom | Prénom | Date de naissance
    The matricule may carry a leading apostrophe inserted by Excel — it is
    stripped automatically.

    Returns a list of dicts sorted by (nom, prenom) with keys:
        ``nom``, ``prenom``, ``matricule``, ``date_naissance`` (YYYY-MM-DD string).

    This function is intended for the **teacher's server only** — the returned
    data includes dates of birth and must never be shipped to student machines.
    Requires ``openpyxl`` (already a dependency of the teacher-server extras).
    """
    try:
        import openpyxl  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "openpyxl is required to load Excel manifests. "
            "Install it with: pip install openpyxl"
        ) from exc

    wb = openpyxl.load_workbook(path)
    ws = wb.active

    students: list[dict] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        matricule = str(row[0]).lstrip("'").strip()
        nom = str(row[1]).strip() if row[1] else ""
        prenom = str(row[2]).strip() if row[2] else ""
        dob = str(row[3]).strip() if row[3] else ""
        if matricule:
            students.append(
                {"nom": nom, "prenom": prenom, "matricule": matricule, "date_naissance": dob}
            )

    return sorted(students, key=lambda s: (s["nom"], s["prenom"]))

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
