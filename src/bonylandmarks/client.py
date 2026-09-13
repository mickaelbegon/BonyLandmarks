"""HTTP client for downloading encrypted scans from the teacher server."""

from __future__ import annotations

from datetime import date

import httpx

from .crypto import decrypt_bytes
from .landmarks import LANDMARK_BY_CODE


class TeacherServerClient:
    """Minimal client to fetch a scan from the teacher's FastAPI server."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def fetch_scan(self, matricule: str, birthdate: date) -> tuple[bytes, dict]:
        """Download and decrypt the GLB and markers for *matricule*.

        Returns (glb_bytes, markers_dict).
        Raises ValueError on bad credentials, httpx.HTTPError on network issues.
        """
        # 1. Download encrypted GLB
        glb_url = f"{self._base}/scan/{matricule.lower()}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(glb_url)
            resp.raise_for_status()
            encrypted_glb = resp.content

        # 2. Decrypt
        glb_bytes = decrypt_bytes(encrypted_glb, matricule, birthdate)

        # 3. Download markers JSON (plain, not encrypted)
        markers_url = f"{self._base}/markers/{matricule.lower()}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(markers_url)
            resp.raise_for_status()
            markers_json = resp.json()

        return glb_bytes, markers_json
