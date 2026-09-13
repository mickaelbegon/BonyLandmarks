"""HTTP client for downloading encrypted avatar GLBs from the teacher server."""

from __future__ import annotations

from datetime import date

import httpx

from .crypto import decrypt_bytes


class TeacherServerClient:
    """Minimal client to fetch a scan from the teacher's FastAPI server."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def fetch_avatar(self, matricule: str, birthdate: date) -> bytes:
        """Download and decrypt the avatar GLB for *matricule*.

        Returns decrypted GLB bytes (contains mesh + markers).
        Raises ValueError on bad credentials, httpx.HTTPError on network issues.
        """
        url = f"{self._base}/scan/{matricule.lower()}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
            encrypted = resp.content
        return decrypt_bytes(encrypted, matricule, birthdate)
