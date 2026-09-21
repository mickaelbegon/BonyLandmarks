"""Estimation du centre articulaire gléno-huméral (GHJC) par régression.

Deux régressions linéaires multiples, toutes deux exprimées dans le repère
scapulaire ISB local (origine AA, X antérieur, Y supérieur, Z latéral) :

========================  ===================================================
Méthode                   Prédicteurs
========================  ===================================================
``meskers``               AA, AC, AI, TS **et le processus coracoïde PC**
``sobral``                AA, AC, AI, TS + âge, sexe, taille, masse (sans PC)
========================  ===================================================

**Unités** — les deux régressions comportent des termes constants en
millimètres : les coordonnées d'entrée des fonctions bas niveau doivent être en
**mm**. ``estimate_shoulder_joint_center`` accepte ``units="m"`` / ``"cm"`` et
convertit. La taille reste en **mètres** et la masse en **kg** quelle que soit
l'unité des coordonnées.

Références
----------
- Meskers CGM, van der Helm FCT, Rozendaal LA, Rozing PM (1998) In vivo
  estimation of the glenohumeral joint rotation center from scapular bony
  landmarks by linear regression. J Biomech 31(1):93-96.
- Sobral et al. (2025) Régression du centre gléno-huméral sans processus
  coracoïde, incluant des covariables anthropométriques.
- Wu G et al. (2005) ISB recommendation ... part II. J Biomech 38(5):981-992.
"""
from __future__ import annotations

import numpy as np

from .frames import build_scapula_frame, scapula_distances
from .validation import as_point, as_positive_float, as_sex

__all__ = [
    "GHJC_METHODS",
    "estimate_ghjc_meskers",
    "estimate_ghjc_sobral",
    "estimate_shoulder_joint_center",
]

#: Noms canoniques des méthodes et leurs alias acceptés.
GHJC_METHODS: dict[str, tuple[str, ...]] = {
    "meskers": ("meskers", "meskers_1998", "meskers1998", "coracoid"),
    "sobral": ("sobral", "sobral_2025", "sobral2025"),
}

#: Correspondances repère canonique → alias acceptés dans le dict ``landmarks``.
_GHJC_ALIASES: dict[str, tuple[str, ...]] = {
    "AA": ("AA", "acromial_angle", "angulus_acromialis", "acromion_angle"),
    "AC": ("AC", "acromioclavicular_joint", "acromioclavicular", "ACJ"),
    "AI": ("AI", "scapula_inferior_angle", "angulus_inferior", "inferior_angle"),
    "TS": ("TS", "scapula_trigonum", "trigonum_spinae", "trigonum"),
    "PC": ("PC", "coracoid_process", "processus_coracoideus", "coracoid"),
}


# ── Utilitaires internes ──────────────────────────────────────────────────────

def _resolve_method(method: str) -> str:
    key = str(method).strip().lower()
    for canonical, aliases in GHJC_METHODS.items():
        if key == canonical or key in aliases:
            return canonical
    known = sorted({a for aliases in GHJC_METHODS.values() for a in aliases})
    raise ValueError(
        f"Méthode GHJC inconnue : {method!r}. Méthodes disponibles : {known}."
    )


def _pick(landmarks: dict, canonical: str, side: str):
    """Cherche un repère par alias, avec ou sans suffixe de côté."""
    candidates: list[str] = []
    for alias in _GHJC_ALIASES[canonical]:
        candidates += [alias, f"{alias}_{side}", f"{side}_{alias}",
                       f"{alias}_{side[0].upper()}"]
    lower = {str(k).lower(): k for k in landmarks}
    for cand in candidates:
        if cand in landmarks:
            return landmarks[cand]
        if cand.lower() in lower:
            return landmarks[lower[cand.lower()]]
    raise ValueError(
        f"Repère {canonical} absent du dictionnaire (côté {side}). Clés "
        f"acceptées : {list(_GHJC_ALIASES[canonical])} (avec ou sans suffixe "
        f"'_{side}'). Clés fournies : {sorted(landmarks)}."
    )


def _scale_factor(units: str) -> float:
    """Facteur unité d'entrée → mm."""
    u = str(units).strip().lower()
    if u in ("mm", "millimeter", "millimetre", "millimètre"):
        return 1.0
    if u in ("m", "meter", "metre", "mètre"):
        return 1000.0
    if u in ("cm", "centimeter", "centimetre", "centimètre"):
        return 10.0
    raise ValueError(f"units doit être 'mm', 'cm' ou 'm', reçu {units!r}.")


def _assemble(frame, gh_local: np.ndarray, dims: dict, method: str,
              side: str, return_details: bool):
    center = frame.to_global(gh_local)
    if not return_details:
        return center
    return {
        "center": center,
        "center_local": gh_local,
        "frame_origin": frame.origin,
        "frame_rotation": frame.rotation,
        "frame_axes": frame.axes,
        "dimensions": dims,
        "method": method,
        "side": side,
    }


# ── Meskers et al. 1998 ───────────────────────────────────────────────────────

def estimate_ghjc_meskers(AA, AC, AI, TS, PC, side: str = "right",
                          return_details: bool = False):
    """GHJC selon Meskers et al. (1998). Coordonnées **en mm**.

    Régression dans le repère scapulaire local (origine AA) ::

        GH_x = 18.9743 + 0.2434*PC_x + 0.2341*AI_x + 0.1590*||AI-AA||
                       + 0.0558*PC_y
        GH_y = -3.8791 - 0.1002*||AC-PC|| + 0.1732*PC_y - 0.3940*||AC-AA||
                       + 0.1205*AI_x
        GH_z = -9.2629 - 0.2403*PC_y + 1.0255*PC_z + 0.1720*||TS-PC||

    ``PC_x``, ``PC_y``, ``PC_z`` et ``AI_x`` sont les coordonnées **locales**
    (repère scapulaire) ; les ``||·||`` sont des distances, invariantes par
    changement de repère orthonormé.

    Parameters
    ----------
    AA, AC, AI, TS, PC : array_like, shape (3,)
        Angle acromial, articulation acromio-claviculaire, angle inférieur,
        trigonum spinae, processus coracoïde — en mm.
    side : {"right", "left"}
    return_details : bool

    Returns
    -------
    np.ndarray, shape (3,)
        GHJC en coordonnées globales (ou dict si ``return_details``).
    """
    frame = build_scapula_frame(AA, AI, TS, side=side)
    aa = as_point("AA", AA)
    ac = as_point("AC", AC)
    ai = as_point("AI", AI)
    ts = as_point("TS", TS)
    pc = as_point("PC", PC)

    ai_l = frame.to_local(ai)
    pc_l = frame.to_local(pc)
    d = scapula_distances(aa, ai, ts, AC=ac, PC=pc)

    gh_x = (18.9743
            + 0.2434 * pc_l[0]
            + 0.2341 * ai_l[0]
            + 0.1590 * d["L_AA_AI"]
            + 0.0558 * pc_l[1])
    gh_y = (-3.8791
            - 0.1002 * d["L_AC_PC"]
            + 0.1732 * pc_l[1]
            - 0.3940 * d["L_AA_AC"]
            + 0.1205 * ai_l[0])
    gh_z = (-9.2629
            - 0.2403 * pc_l[1]
            + 1.0255 * pc_l[2]
            + 0.1720 * d["L_TS_PC"])

    gh_local = np.array([gh_x, gh_y, gh_z])
    return _assemble(frame, gh_local, d, "meskers", side, return_details)


# ── Sobral et al. 2025 ────────────────────────────────────────────────────────

def estimate_ghjc_sobral(AA, AC, AI, TS, age, sex, height, mass,
                         side: str = "right", return_details: bool = False):
    """GHJC selon Sobral et al. (2025), sans processus coracoïde. **mm**.

    Régression dans le repère scapulaire local (origine AA), avec ``AC_x``,
    ``AC_y``, ``AC_z``, ``AI_z`` locaux, les six distances inter-repères, et
    quatre covariables : ``A`` = âge (années), ``S`` = sexe (0 = F, 1 = M),
    ``H`` = taille (m), ``W`` = masse (kg).

    Intérêt clinique : le processus coracoïde est difficile à palper de façon
    fiable ; cette régression s'en affranchit au prix de covariables
    anthropométriques.

    Parameters
    ----------
    AA, AC, AI, TS : array_like, shape (3,)
        Repères scapulaires en **mm**.
    age : float
        Âge en années.
    sex : {0, 1, "F", "M"}
        0 / "F" = féminin, 1 / "M" = masculin.
    height : float
        Taille en **mètres**.
    mass : float
        Masse en **kg**.
    side : {"right", "left"}
    return_details : bool

    Returns
    -------
    np.ndarray, shape (3,)
        GHJC en coordonnées globales (ou dict si ``return_details``).
    """
    frame = build_scapula_frame(AA, AI, TS, side=side)
    aa = as_point("AA", AA)
    ac = as_point("AC", AC)
    ai = as_point("AI", AI)
    ts = as_point("TS", TS)

    ac_l = frame.to_local(ac)
    ai_l = frame.to_local(ai)
    d = scapula_distances(aa, ai, ts, AC=ac)

    a = float(age)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"age: doit être strictement positif et fini, reçu {age!r}.")
    s = as_sex(sex)
    h = as_positive_float("height", height)
    w = as_positive_float("mass", mass)
    if h > 3.0:
        raise ValueError(
            f"height doit être en mètres (reçu {h}) — p.ex. 1.75 et non 175."
        )

    ac_x, ac_y, ac_z = float(ac_l[0]), float(ac_l[1]), float(ac_l[2])
    ai_z = float(ai_l[2])
    l_aa_ts = d["L_AA_TS"]
    l_aa_ai = d["L_AA_AI"]
    l_aa_ac = d["L_AA_AC"]
    l_ts_ai = d["L_TS_AI"]
    l_ts_ac = d["L_TS_AC"]
    l_ai_ac = d["L_AI_AC"]

    gh_x = (25.5316
            + 0.6334 * ac_x + 0.7842 * ac_y - 0.0832 * ai_z
            - 0.2673 * l_aa_ts + 0.0365 * l_aa_ai - 0.5353 * l_aa_ac
            + 0.0843 * l_ts_ai + 0.2350 * l_ts_ac - 0.1246 * l_ai_ac
            - 0.0237 * a + 2.1296 * s - 1.1900 * h + 0.0221 * w)
    gh_y = (-6.7070
            - 0.2514 * ac_x + 0.7558 * ac_y + 0.0264 * ai_z
            - 0.0620 * l_aa_ai - 1.7641 * s - 5.7094 * h)
    gh_z = (-22.5233
            + 0.5954 * ac_x + 0.0600 * ac_y + 0.1085 * ai_z
            + 0.3983 * ac_z - 0.3880 * l_aa_ac - 0.0197 * l_ts_ai
            - 0.0934 * l_ts_ac + 0.0558 * a + 0.2930 * s
            + 26.4715 * h - 0.0361 * w)

    gh_local = np.array([gh_x, gh_y, gh_z])
    dims = dict(d)
    dims.update({"age": a, "sex": s, "height": h, "mass": w})
    return _assemble(frame, gh_local, dims, "sobral", side, return_details)


# ── API haut niveau ───────────────────────────────────────────────────────────

def estimate_shoulder_joint_center(landmarks: dict, method: str = "sobral",
                                   side: str = "right", units: str = "mm",
                                   return_details: bool = False, **kwargs):
    """Estime le centre gléno-huméral à partir d'un dictionnaire de repères.

    Parameters
    ----------
    landmarks : dict
        ``AA``, ``AC``, ``AI``, ``TS`` (+ ``PC`` pour Meskers). Les alias du
        projet (``acromial_angle_right``, ``scapula_trigonum_right``, …) sont
        acceptés, avec ou sans suffixe de côté.
    method : str
        ``"sobral"`` (défaut, sans coracoïde) ou ``"meskers"`` (avec PC).
    side : {"right", "left"}
    units : {"mm", "cm", "m"}
        Unité des coordonnées d'entrée ; le résultat est rendu dans la même
        unité. ``height`` reste en mètres et ``mass`` en kg.
    return_details : bool
        Retourne un dict avec ``center``, ``center_local``, ``frame_origin``,
        ``frame_rotation``, ``frame_axes``, ``dimensions``, ``method``,
        ``side``, ``units``.
    **kwargs
        Pour ``"sobral"`` : ``age``, ``sex``, ``height``, ``mass``.

    Returns
    -------
    np.ndarray, shape (3,)
        GHJC en coordonnées globales (ou dict si ``return_details``).
    """
    if not isinstance(landmarks, dict):
        raise ValueError(
            f"landmarks doit être un dict, reçu {type(landmarks).__name__}."
        )
    canonical = _resolve_method(method)
    side = str(side).lower()
    if side not in ("right", "left"):
        raise ValueError(f"side doit être 'right' ou 'left', reçu {side!r}.")

    to_mm = _scale_factor(units)
    needed = ["AA", "AC", "AI", "TS"] + (["PC"] if canonical == "meskers" else [])
    pts = {
        name: as_point(name, _pick(landmarks, name, side)) * to_mm
        for name in needed
    }

    if canonical == "meskers":
        res = estimate_ghjc_meskers(
            pts["AA"], pts["AC"], pts["AI"], pts["TS"], pts["PC"],
            side=side, return_details=return_details,
        )
    else:
        missing = [k for k in ("age", "sex", "height", "mass") if k not in kwargs]
        if missing:
            raise ValueError(
                "La méthode 'sobral' exige les covariables "
                f"{missing} (âge en années, sexe 0/1, taille en m, masse en kg)."
            )
        res = estimate_ghjc_sobral(
            pts["AA"], pts["AC"], pts["AI"], pts["TS"],
            age=kwargs["age"], sex=kwargs["sex"],
            height=kwargs["height"], mass=kwargs["mass"],
            side=side, return_details=return_details,
        )

    from_mm = 1.0 / to_mm
    if not return_details:
        return res * from_mm if from_mm != 1.0 else res

    if from_mm != 1.0:
        res = dict(res)
        res["center"] = res["center"] * from_mm
        res["center_local"] = res["center_local"] * from_mm
        res["frame_origin"] = res["frame_origin"] * from_mm
        res["dimensions"] = {
            k: (v * from_mm if k.startswith("L_") else v)
            for k, v in res["dimensions"].items()
        }
    res["units"] = str(units).lower()
    return res
