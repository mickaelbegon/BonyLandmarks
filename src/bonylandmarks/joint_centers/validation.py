"""Validation des entrées pour l'estimation des centres articulaires.

Toutes les fonctions de ce module lèvent :class:`ValueError` avec un message
explicite (nom du repère fautif + raison) plutôt que de propager silencieusement
des ``NaN`` dans les régressions.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "EPS",
    "COLLINEAR_TOL",
    "as_point",
    "as_points",
    "as_positive_float",
    "as_sex",
]

#: Tolérance absolue en dessous de laquelle une norme est considérée nulle.
EPS: float = 1e-9

#: Tolérance sur le sinus de l'angle entre deux vecteurs (critère de colinéarité).
COLLINEAR_TOL: float = 1e-6


# ── Contrôles élémentaires ────────────────────────────────────────────────────

def _check_shape(name: str, value) -> np.ndarray:
    """Convertit ``value`` en ``np.ndarray`` float de shape ``(3,)``."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: impossible de convertir en tableau numérique ({value!r})."
        ) from exc
    arr = np.squeeze(arr)
    if arr.shape != (3,):
        raise ValueError(
            f"{name}: shape attendue (3,), obtenue {np.asarray(value).shape}."
        )
    return np.array(arr, dtype=float)


def _check_finite(name: str, arr: np.ndarray) -> np.ndarray:
    """Vérifie l'absence de ``NaN`` / ``inf``."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name}: coordonnées non finies (NaN ou inf) — reçu {arr.tolist()}."
        )
    return arr


def _check_distinct(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray,
                    tol: float = EPS) -> float:
    """Vérifie que deux repères sont distincts ; retourne leur distance."""
    d = float(np.linalg.norm(a - b))
    if d <= tol:
        raise ValueError(
            f"{name_a} et {name_b} sont confondus (distance = {d:.3e}). "
            "Les deux repères doivent être distincts pour construire le repère "
            "anatomique."
        )
    return d


def _check_collinear(names: Sequence[str], pts: Sequence[np.ndarray],
                     tol: float = COLLINEAR_TOL) -> None:
    """Vérifie que trois points ne sont pas alignés.

    Critère relatif : ``||u x v|| <= tol * ||u|| * ||v||`` avec ``u = P1 - P0``
    et ``v = P2 - P0``, c.-à-d. ``sin(angle) <= tol``.
    """
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in pts)
    u, v = p1 - p0, p2 - p0
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu <= EPS or nv <= EPS:
        raise ValueError(
            f"Repères {', '.join(names)} : au moins deux points sont confondus, "
            "le plan anatomique est indéfini."
        )
    sin_angle = float(np.linalg.norm(np.cross(u, v))) / (nu * nv)
    if sin_angle <= tol:
        raise ValueError(
            f"Repères {', '.join(names)} colinéaires (sin = {sin_angle:.3e} "
            f"<= {tol:.1e}) : le plan anatomique est indéfini. Vérifiez le "
            "placement des repères."
        )


def _check_non_parallel(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray,
                        tol: float = COLLINEAR_TOL) -> None:
    """Vérifie que deux directions ne sont pas parallèles."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        raise ValueError(f"Direction nulle : {name_a if na <= EPS else name_b}.")
    sin_angle = float(np.linalg.norm(np.cross(a, b))) / (na * nb)
    if sin_angle <= tol:
        raise ValueError(
            f"Directions {name_a} et {name_b} parallèles (sin = {sin_angle:.3e}) : "
            "le repère anatomique est dégénéré."
        )


# ── API publique ──────────────────────────────────────────────────────────────

def as_point(name: str, value) -> np.ndarray:
    """Valide et normalise un repère anatomique → ``np.ndarray`` shape ``(3,)``."""
    if value is None:
        raise ValueError(f"{name}: repère manquant (None).")
    return _check_finite(name, _check_shape(name, value))


def as_points(names: Iterable[str], values: Iterable) -> tuple[np.ndarray, ...]:
    """Valide plusieurs repères d'un coup."""
    return tuple(as_point(n, v) for n, v in zip(names, values))


def as_positive_float(name: str, value) -> float:
    """Valide un scalaire strictement positif et fini (longueur, masse, taille)."""
    if value is None:
        raise ValueError(f"{name}: valeur manquante (None).")
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: valeur non numérique ({value!r}).") from exc
    if not np.isfinite(v):
        raise ValueError(f"{name}: valeur non finie ({v}).")
    if v <= 0.0:
        raise ValueError(f"{name}: doit être strictement positif, reçu {v}.")
    return v


def as_sex(value) -> float:
    """Normalise le sexe en ``0.0`` (féminin) ou ``1.0`` (masculin).

    Accepte ``0``/``1``, ``"F"``/``"M"``, ``"female"``/``"male"``,
    ``"femme"``/``"homme"``.
    """
    if value is None:
        raise ValueError("sex: valeur manquante (None).")
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("m", "male", "h", "homme", "man", "1"):
            return 1.0
        if v in ("f", "female", "femme", "w", "woman", "0"):
            return 0.0
        raise ValueError(
            f"sex: chaîne non reconnue ({value!r}). Attendu 'F'/'M' ou 0/1."
        )
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sex: valeur non numérique ({value!r}).") from exc
    if v not in (0.0, 1.0):
        raise ValueError(f"sex: attendu 0 (féminin) ou 1 (masculin), reçu {v}.")
    return v
