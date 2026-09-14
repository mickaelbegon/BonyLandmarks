"""HTTP client for downloading encrypted avatar GLBs from the teacher server."""

from __future__ import annotations

import hashlib
from datetime import date

import httpx

from .crypto import decrypt_bytes


class TeacherServerClient:
    """Minimal client to fetch a scan from the teacher's FastAPI server."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def fetch_avatar(self, matricule: str, birthdate: date) -> bytes:
        """POST /auth → token → GET /scan/{matricule} → decrypt.

        Returns decrypted GLB bytes (contains mesh + markers).
        Raises ValueError on bad credentials, httpx.HTTPError on network issues.
        """
        dob_hash = hashlib.sha256(
            f"{matricule.lower().strip()}:{birthdate.strftime('%Y%m%d')}".encode()
        ).hexdigest()

        with httpx.Client(timeout=self._timeout) as client:
            # Step 1: authenticate
            auth_resp = client.post(
                f"{self._base}/auth",
                json={"matricule": matricule.lower(), "dob_hash": dob_hash},
            )
            if auth_resp.status_code == 401:
                raise ValueError("Identifiants incorrects / Wrong credentials")
            auth_resp.raise_for_status()
            token = auth_resp.json()["token"]

            # Step 2: download scan
            scan_resp = client.get(
                f"{self._base}/scan/{matricule.lower()}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if scan_resp.status_code in (401, 403):
                raise ValueError("Accès refusé / Access denied")
            scan_resp.raise_for_status()
            encrypted = scan_resp.content

        return decrypt_bytes(encrypted, matricule, birthdate)
