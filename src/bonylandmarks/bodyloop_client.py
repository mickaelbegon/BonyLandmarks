"""Client HTTP pour télécharger des GLB depuis une instance BodyLoop.

API BodyLoop (OpenAPI v2, endpoints vérifiés via bodyloop_sdk) :

  POST /api/v2/authentification/token
      form: grant_type=password, username, password
      → {"access_token": str, "token_type": "bearer", "expires_in": int}

  GET  /api/v2/viatars/
      Authorization: Bearer {token}
      → {"items": [{"viatar_id": int, "parameters": {"title": str,
                    "avatar_3d": {...}|null, "mesh_3d": {...}|null}, ...}]}

  GET  /api/v2/viatars/{viatar_id}/models
      Authorization: Bearer {token}
      → {"model_names": ["avatar_3d", "mesh_3d", ...]}
         ou dict avec les noms en clés (best-effort, format non fixé)

  GET  /api/v2/viatars/{viatar_id}/models/{model_name}?scale=1000
      Authorization: Bearer {token}
      → bytes GLB (les viatars BodyLoop sont en mètres ; scale=1000 → mm)

model_name courants : "avatar_3d" (avec AutoMarkers) | "mesh_3d" (sans markers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass
class ViatarInfo:
    """Métadonnées minimales d'un viatar BodyLoop."""

    id: int
    label: str                  # title ou str(id)
    has_avatar_3d: bool = False
    has_mesh_3d: bool = False

    def __str__(self) -> str:
        return self.label


class BodyLoopClient:
    """Accès authentifié à une instance BodyLoop (OAuth2 password grant).

    Usage::

        client = BodyLoopClient("https://192.168.38.12")
        client.login("admin", "secret")
        viatars = client.list_viatars()
        glb_bytes = client.get_glb(viatars[0].id, model_name="avatar_3d")
        client.close()

    Préférer le gestionnaire de contexte::

        with BodyLoopClient("https://192.168.38.12") as client:
            client.login("admin", "secret")
            glb = client.get_glb(1, model_name="mesh_3d")
    """

    _TOKEN_PATH = "/api/v2/authentification/token"
    _VIATARS_PATH = "/api/v2/viatars/"
    _MODELS_PATH = "/api/v2/viatars/{viatar_id}/models"
    _MODEL_PATH = "/api/v2/viatars/{viatar_id}/models/{model_name}"
    _SCALE_MM = 1000  # BodyLoop stocke les viatars en mètres → ×1000 → mm

    def __init__(self, base_url: str, timeout: float = 60.0, verify_ssl: bool = False) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._verify = verify_ssl
        self._token: Optional[str] = None
        self._http: Optional[httpx.Client] = None

    # ── Connexion ──────────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> None:
        """Authentification OAuth2 password grant. Lève ValueError si refusé."""
        self._http = httpx.Client(
            base_url=self._base,
            timeout=self._timeout,
            verify=self._verify,
        )
        resp = self._http.post(
            self._TOKEN_PATH,
            data={"grant_type": "password", "username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code in (401, 403, 422):
            raise ValueError("Identifiants BodyLoop incorrects")
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        self._token = payload.get("access_token")
        if not self._token:
            raise ValueError("BodyLoop n'a pas retourné de token d'accès")
        self._http.headers["Authorization"] = f"Bearer {self._token}"

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> "BodyLoopClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── API ───────────────────────────────────────────────────────────────────

    def list_viatars(self) -> list[ViatarInfo]:
        """Liste les viatars accessibles avec disponibilité des modèles GLB."""
        self._check_auth()
        resp = self._http.get(self._VIATARS_PATH)
        resp.raise_for_status()
        data: Any = resp.json()

        # Réponse attendue : {"items": [...]} ou liste directe
        if isinstance(data, dict):
            items = data.get("items") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []

        result: list[ViatarInfo] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            vid = item.get("viatar_id")
            if vid is None:
                continue

            params = item.get("parameters") or {}
            title = params.get("title") or str(vid)

            info = ViatarInfo(
                id=int(vid),
                label=title,
                has_avatar_3d=bool(params.get("avatar_3d")),
                has_mesh_3d=bool(params.get("mesh_3d")),
            )
            result.append(info)
        return result

    def list_model_names(self, viatar_id: int) -> list[str]:
        """Retourne la liste des noms de modèles disponibles pour un viatar."""
        self._check_auth()
        url = self._MODELS_PATH.format(viatar_id=viatar_id)
        resp = self._http.get(url)
        resp.raise_for_status()
        data: Any = resp.json()

        # Soit {"model_names": [...]} soit un dict dont les clés sont les noms
        if isinstance(data, dict):
            names = data.get("model_names")
            if isinstance(names, list):
                return [str(n) for n in names]
            return [str(k) for k in data]
        if isinstance(data, list):
            return [str(n) for n in data]
        return []

    def get_glb(self, viatar_id: int, model_name: str = "avatar_3d") -> bytes:
        """Télécharge un GLB BodyLoop en millimètres.

        Parameters
        ----------
        viatar_id:
            Identifiant entier du viatar (voir :meth:`list_viatars`).
        model_name:
            ``"avatar_3d"`` (avec AutoMarkers BodyLoop) ou ``"mesh_3d"``
            (mesh pur sans markers, recommandé pour le scan partagé).

        Returns
        -------
        bytes
            Contenu brut du fichier GLB, échelle en millimètres.
        """
        self._check_auth()
        url = self._MODEL_PATH.format(viatar_id=viatar_id, model_name=model_name)
        resp = self._http.get(url, params={"scale": self._SCALE_MM})
        if resp.status_code == 404:
            raise ValueError(f"Modèle '{model_name}' non disponible pour le viatar {viatar_id}")
        resp.raise_for_status()
        return resp.content

    # ── Privé ─────────────────────────────────────────────────────────────────

    def _check_auth(self) -> None:
        if self._http is None or self._token is None:
            raise RuntimeError("Appelez login() avant d'utiliser le client BodyLoop")
