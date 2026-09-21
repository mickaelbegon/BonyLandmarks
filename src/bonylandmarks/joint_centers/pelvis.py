"""Estimation du centre articulaire de la hanche (HJC) par régression.

Trois régressions prédictives sont implémentées, toutes exprimées dans le
repère pelvien local construit par :func:`~.frames.build_pelvis_frame`
(origine = milieu des EIAS, axes ``AP`` / ``SI`` / ``ML``).

======================  =====================================================
Méthode                 Prédicteurs
======================  =====================================================
``bell``                largeur pelvienne PW seule (sans échelle absolue)
``harrington``          PW et profondeur pelvienne PD
``harrington_leg``      PW, PD et longueur de jambe
======================  =====================================================

**Unités** — les régressions de Harrington comportent des termes constants en
millimètres. Les fonctions bas niveau de ce module supposent donc des
coordonnées **en mm**. La fonction haut niveau
:func:`estimate_hip_joint_centers` accepte ``units="m"`` et effectue les
conversions (utile pour les fichiers GLB, exprimés en mètres).

Références
----------
- Bell AL, Pedersen DR, Brand RA (1990) A comparison of the accuracy of several
  hip center location prediction methods. J Biomech 23(6):617-621.
- Harrington ME, Zavatsky AB, Lawson SEM, Yuan Z, Theologis TN (2007)
  Prediction of the hip joint centre in adults, children, and patients with
  cerebral palsy based on magnetic resonance imaging. J Biomech 40(3):595-602.
"""
from __future__ import annotations

import numpy as np

from .frames import AnatomicalFrame, build_pelvis_frame, pelvis_dimensions
from .validation import as_point, as_positive_float

__all__ = [
    "HJC_METHODS",
    "estimate_hjc_bell",
    "estimate_hjc_harrington_pelvis",
    "estimate_hjc_harrington_leg_length",
    "estimate_hip_joint_centers",
]

#: Noms canoniques des méthodes et leurs alias acceptés.
HJC_METHODS: dict[str, tuple[str, ...]] = {
    "bell": ("bell", "brand", "bell_brand", "bell_1990", "bell_brand_1990"),
    "harrington_pelvis": (
        "harrington", "harrington_pelvis", "harrington_2007",
        "harrington_pelvis_only", "pelvis_only",
    ),
    "harrington_leg_length": (
        "harrington_leg", "harrington_leg_length", "harrington_ll",
        "leg_length",
    ),
}

#: Correspondances repère canonique → alias acceptés dans le dict ``landmarks``.
_HJC_ALIASES: dict[str, tuple[str, ...]] = {
    "RASI": ("RASI", "R.ASIS", "RASIS", "ASIS_right", "ASIS_R", "right_ASIS"),
    "LASI": ("LASI", "L.ASIS", "LASIS", "ASIS_left", "ASIS_L", "left_ASIS"),
    "RPSI": ("RPSI", "R.PSIS", "RPSIS", "PSIS_right", "PSIS_R", "right_PSIS"),
    "LPSI": ("LPSI", "L.PSIS", "LPSIS", "PSIS_left", "PSIS_L", "left_PSIS"),
}


# ── Utilitaires internes ──────────────────────────────────────────────────────

def _resolve_method(method: str) -> str:
    key = str(method).strip().lower()
    for canonical, aliases in HJC_METHODS.items():
        if key == canonical or key in aliases:
            return canonical
    known = sorted({a for aliases in HJC_METHODS.values() for a in aliases})
    raise ValueError(
        f"Méthode HJC inconnue : {method!r}. Méthodes disponibles : {known}."
    )


def _pick(landmarks: dict, canonical: str):
    for alias in _HJC_ALIASES[canonical]:
        if alias in landmarks:
            return landmarks[alias]
        for key in landmarks:
            if str(key).lower() == alias.lower():
                return landmarks[key]
    raise ValueError(
        f"Repère {canonical} absent du dictionnaire. Clés acceptées : "
        f"{list(_HJC_ALIASES[canonical])}. Clés fournies : {sorted(landmarks)}."
    )


def _assemble(frame: AnatomicalFrame, dims: dict[str, float],
              local_right: np.ndarray, local_left: np.ndarray,
              method: str, extra: dict, return_details: bool) -> dict:
    """Passe du repère local au repère global et met en forme le résultat."""
    out = {
        "right": frame.to_global(local_right),
        "left": frame.to_global(local_left),
    }
    if not return_details:
        return out
    dimensions = dict(dims)
    dimensions.update(extra)
    out.update({
        "center": {"right": out["right"], "left": out["left"]},
        "center_local": {"right": local_right, "left": local_left},
        "frame_origin": frame.origin,
        "frame_rotation": frame.rotation,
        "frame_axes": frame.axes,
        "dimensions": dimensions,
        "method": method,
    })
    return out


def _scale_landmarks(pts: tuple, units: str) -> tuple[tuple, float]:
    """Retourne les points convertis en mm et le facteur mm → unité d'origine."""
    u = str(units).strip().lower()
    if u in ("mm", "millimeter", "millimetre", "millimètre"):
        return pts, 1.0
    if u in ("m", "meter", "metre", "mètre"):
        return tuple(np.asarray(p, dtype=float) * 1000.0 for p in pts), 0.001
    if u in ("cm", "centimeter", "centimetre", "centimètre"):
        return tuple(np.asarray(p, dtype=float) * 10.0 for p in pts), 0.1
    raise ValueError(f"units doit être 'mm', 'cm' ou 'm', reçu {units!r}.")


# ── Bell / Brand ──────────────────────────────────────────────────────────────

def estimate_hjc_bell(RASI, LASI, RPSI, LPSI, return_details: bool = False):
    """HJC selon Bell, Pedersen & Brand (1990). Coordonnées **en mm**.

    Offsets purement proportionnels à la largeur pelvienne ``PW`` — la méthode
    est donc insensible aux unités (aucun terme constant) ::

        HJC_AP = -0.19 * PW
        HJC_SI = -0.30 * PW
        HJC_ML = ±0.36 * PW      (+ à droite, - à gauche)

    Parameters
    ----------
    RASI, LASI, RPSI, LPSI : array_like, shape (3,)
    return_details : bool
        Si vrai, ajoute le repère, les dimensions pelviennes et les
        coordonnées locales au résultat.

    Returns
    -------
    dict
        ``{"right": ndarray(3,), "left": ndarray(3,)}`` (+ détails si demandé).
    """
    frame = build_pelvis_frame(RASI, LASI, RPSI, LPSI)
    dims = pelvis_dimensions(RASI, LASI, RPSI, LPSI)
    pw = dims["PW"]

    ap = -0.19 * pw
    si = -0.30 * pw
    ml = 0.36 * pw
    local_right = np.array([ap, si, +ml])
    local_left = np.array([ap, si, -ml])
    return _assemble(frame, dims, local_right, local_left, "bell", {},
                     return_details)


# ── Harrington 2007 — bassin seul ─────────────────────────────────────────────

def estimate_hjc_harrington_pelvis(RASI, LASI, RPSI, LPSI,
                                   return_details: bool = False):
    """HJC selon Harrington et al. (2007), prédicteurs pelviens seuls. **mm**.

    Table 2 de la publication ::

        HJC_AP = -0.24 * PD - 9.9
        HJC_SI = -0.30 * PW - 10.9
        HJC_ML = ±(0.33 * PW + 7.3)

    Les termes constants sont en millimètres : ``RASI`` … ``LPSI`` **doivent**
    être fournis en mm.
    """
    frame = build_pelvis_frame(RASI, LASI, RPSI, LPSI)
    dims = pelvis_dimensions(RASI, LASI, RPSI, LPSI)
    pw, pd = dims["PW"], dims["PD"]

    ap = -0.24 * pd - 9.9
    si = -0.30 * pw - 10.9
    ml = 0.33 * pw + 7.3
    local_right = np.array([ap, si, +ml])
    local_left = np.array([ap, si, -ml])
    return _assemble(frame, dims, local_right, local_left,
                     "harrington_pelvis", {}, return_details)


# ── Harrington 2007 — avec longueur de jambe ──────────────────────────────────

def estimate_hjc_harrington_leg_length(RASI, LASI, RPSI, LPSI,
                                       leg_length_right,
                                       leg_length_left=None,
                                       return_details: bool = False):
    """HJC selon Harrington et al. (2007) avec longueur de jambe. **mm**.

    ::

        HJC_AP = -0.24 * PD - 9.9
        HJC_SI = -0.16 * PW - 0.04 * LL - 7.1
        HJC_ML = ±(0.28 * PD + 0.16 * PW + 7.9)

    Seul l'offset supéro-inférieur dépend de la longueur de jambe ``LL`` : les
    deux côtés ne diffèrent donc que par ``HJC_SI`` lorsque les longueurs de
    jambe sont asymétriques.

    Parameters
    ----------
    leg_length_right : float
        Longueur de jambe droite, en **mm** (EIAS → malléole médiale).
    leg_length_left : float, optional
        Longueur de jambe gauche en mm ; par défaut égale à la droite.
    """
    frame = build_pelvis_frame(RASI, LASI, RPSI, LPSI)
    dims = pelvis_dimensions(RASI, LASI, RPSI, LPSI)
    pw, pd = dims["PW"], dims["PD"]

    ll_r = as_positive_float("leg_length_right", leg_length_right)
    ll_l = ll_r if leg_length_left is None else as_positive_float(
        "leg_length_left", leg_length_left)

    ap = -0.24 * pd - 9.9
    ml = 0.28 * pd + 0.16 * pw + 7.9
    si_r = -0.16 * pw - 0.04 * ll_r - 7.1
    si_l = -0.16 * pw - 0.04 * ll_l - 7.1

    local_right = np.array([ap, si_r, +ml])
    local_left = np.array([ap, si_l, -ml])
    return _assemble(frame, dims, local_right, local_left,
                     "harrington_leg_length",
                     {"leg_length_right": ll_r, "leg_length_left": ll_l},
                     return_details)


# ── API haut niveau ───────────────────────────────────────────────────────────

def estimate_hip_joint_centers(landmarks: dict, method: str = "harrington",
                               units: str = "mm",
                               return_details: bool = False,
                               **kwargs) -> dict:
    """Estime les deux centres articulaires de hanche à partir d'un dictionnaire.

    Parameters
    ----------
    landmarks : dict
        Doit contenir ``RASI``, ``LASI``, ``RPSI``, ``LPSI`` (les alias du
        projet — ``ASIS_right``, ``PSIS_left``, … — sont également acceptés),
        chacun de shape ``(3,)``.
    method : str
        ``"bell"``, ``"harrington"`` (bassin seul) ou ``"harrington_leg"``.
    units : {"mm", "cm", "m"}
        Unité des coordonnées d'entrée. Les régressions sont appliquées en mm
        et le résultat est reconverti dans l'unité d'entrée. ``leg_length_*``
        est exprimé dans la **même** unité.
    return_details : bool
        Ajoute ``center``, ``center_local``, ``frame_origin``,
        ``frame_rotation``, ``frame_axes``, ``dimensions`` et ``method``.
    **kwargs
        ``leg_length_right`` / ``leg_length_left`` pour ``"harrington_leg"``.

    Returns
    -------
    dict
        ``{"right": ndarray(3,), "left": ndarray(3,)}`` dans le repère global
        et l'unité d'entrée.

    Examples
    --------
    >>> import numpy as np
    >>> lm = {"RASI": np.array([95., 985., -40.]),
    ...       "LASI": np.array([-95., 985., -40.]),
    ...       "RPSI": np.array([70., 985., 60.]),
    ...       "LPSI": np.array([-70., 985., 60.])}
    >>> hjc = estimate_hip_joint_centers(lm, method="harrington")
    >>> hjc["right"].shape
    (3,)
    """
    if not isinstance(landmarks, dict):
        raise ValueError(
            f"landmarks doit être un dict, reçu {type(landmarks).__name__}."
        )
    canonical = _resolve_method(method)

    pts = tuple(
        as_point(name, _pick(landmarks, name))
        for name in ("RASI", "LASI", "RPSI", "LPSI")
    )
    pts_mm, scale = _scale_landmarks(pts, units)

    if canonical == "bell":
        unknown = set(kwargs) - {"leg_length_right", "leg_length_left"}
        if unknown:
            raise ValueError(f"Arguments inattendus pour 'bell' : {sorted(unknown)}.")
        res = estimate_hjc_bell(*pts_mm, return_details=return_details)
    elif canonical == "harrington_pelvis":
        res = estimate_hjc_harrington_pelvis(*pts_mm, return_details=return_details)
    else:
        if "leg_length_right" not in kwargs:
            raise ValueError(
                "La méthode 'harrington_leg' exige leg_length_right "
                f"(en {units}, EIAS → malléole médiale)."
            )
        ll_r = as_positive_float("leg_length_right", kwargs["leg_length_right"])
        ll_l = kwargs.get("leg_length_left")
        if ll_l is not None:
            ll_l = as_positive_float("leg_length_left", ll_l) / scale
        res = estimate_hjc_harrington_leg_length(
            *pts_mm, leg_length_right=ll_r / scale, leg_length_left=ll_l,
            return_details=return_details,
        )

    if scale != 1.0:
        res = _rescale_result(res, scale)
    if return_details:
        res["units"] = str(units).lower()
    return res


def _rescale_result(res: dict, scale: float) -> dict:
    """Reconvertit un résultat calculé en mm vers l'unité d'entrée."""
    out = dict(res)
    out["right"] = res["right"] * scale
    out["left"] = res["left"] * scale
    if "center" in res:
        out["center"] = {"right": out["right"], "left": out["left"]}
        out["center_local"] = {k: v * scale for k, v in res["center_local"].items()}
        out["frame_origin"] = res["frame_origin"] * scale
        out["dimensions"] = {k: v * scale for k, v in res["dimensions"].items()}
    return out
