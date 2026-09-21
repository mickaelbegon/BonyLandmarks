"""Repères anatomiques locaux (bassin, scapula) utilisés par les régressions.

Les régressions prédictives de centres articulaires (Bell, Harrington, Meskers,
Sobral) sont **définies dans un repère anatomique local**. Les appliquer
directement sur des coordonnées globales n'a aucun sens : il faut d'abord
construire le repère, exprimer les repères osseux dedans, appliquer les
équations, puis reconvertir le résultat en coordonnées globales.

Conventions
-----------
Bassin (``build_pelvis_frame``)
    Origine : milieu des EIAS. Axes ``(AP, SI, ML)`` = antérieur, supérieur,
    latéral-droit. Repère direct (det = +1).

Scapula (``build_scapula_frame``, Wu et al. 2005)
    Origine : AA (angle acromial). Axes ``(X, Y, Z)`` = antérieur, supérieur,
    latéral. Le repère gauche est le **miroir exact** du repère droit
    (det = -1) : c'est ce qui permet d'appliquer sans modification des
    régressions calibrées sur des scapulas droites.

Références
----------
- Wu G et al. (2002) ISB recommendation ... part I: ankle, hip, spine.
  J Biomech 35(4):543-548.
- Wu G et al. (2005) ISB recommendation ... part II: shoulder, elbow, wrist.
  J Biomech 38(5):981-992.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import (
    EPS,
    _check_collinear,
    _check_distinct,
    _check_non_parallel,
    as_point,
)

__all__ = [
    "AnatomicalFrame",
    "build_pelvis_frame",
    "build_scapula_frame",
    "pelvis_dimensions",
    "scapula_distances",
]


def _normalize(v: np.ndarray, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError(f"Axe {name} de norme nulle : repère anatomique dégénéré.")
    return v / n


# ── Repère générique ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnatomicalFrame:
    """Repère orthonormé local.

    Attributes
    ----------
    origin : np.ndarray, shape (3,)
        Origine du repère en coordonnées globales.
    rotation : np.ndarray, shape (3, 3)
        Les **colonnes** sont les axes du repère exprimés en coordonnées
        globales : ``rotation[:, 0]`` est le premier axe, etc. La matrice est
        orthonormée ; son déterminant vaut ``+1`` (repère direct) ou ``-1``
        (repère miroir, scapula gauche).
    axes : tuple[str, str, str]
        Noms des axes, dans l'ordre des colonnes.
    name : str
        Nom du segment ("pelvis", "scapula_right", …).
    """

    origin: np.ndarray
    rotation: np.ndarray
    axes: tuple[str, str, str]
    name: str

    def to_local(self, point) -> np.ndarray:
        """Coordonnées globales → coordonnées locales (shape ``(3,)`` ou ``(N, 3)``)."""
        p = np.asarray(point, dtype=float)
        return (p - self.origin) @ self.rotation

    def to_global(self, local) -> np.ndarray:
        """Coordonnées locales → coordonnées globales (shape ``(3,)`` ou ``(N, 3)``)."""
        p = np.asarray(local, dtype=float)
        return self.origin + p @ self.rotation.T

    @property
    def handedness(self) -> float:
        """``+1.0`` pour un repère direct, ``-1.0`` pour un repère miroir."""
        return float(np.sign(np.linalg.det(self.rotation)))

    def axis(self, name: str) -> np.ndarray:
        """Retourne l'axe nommé, en coordonnées globales."""
        try:
            idx = self.axes.index(name)
        except ValueError as exc:
            raise ValueError(
                f"Axe inconnu {name!r} ; axes disponibles : {self.axes}."
            ) from exc
        return self.rotation[:, idx]


# ── Bassin ────────────────────────────────────────────────────────────────────

def build_pelvis_frame(RASI, LASI, RPSI, LPSI) -> AnatomicalFrame:
    """Construit le repère pelvien à partir des quatre épines iliaques.

    ::

        O        = (RASI + LASI) / 2
        ML       = normalize(RASI - LASI)              # vers la droite du sujet
        anterior = normalize(ASIS_mid - PSIS_mid)
        SI       = normalize(cross(ML, anterior))      # vers le haut
        AP       = normalize(cross(SI, ML))            # vers l'avant

    L'axe ``AP`` final est ré-orthogonalisé : c'est la composante de
    ``anterior`` perpendiculaire à ``ML``, ce qui rend le repère robuste à une
    légère asymétrie des EIPS.

    Parameters
    ----------
    RASI, LASI, RPSI, LPSI : array_like, shape (3,)
        EIAS droite / gauche, EIPS droite / gauche. Unités quelconques mais
        homogènes (les régressions supposent des **mm**).

    Returns
    -------
    AnatomicalFrame
        Axes ``("AP", "SI", "ML")``.

    Raises
    ------
    ValueError
        Repère manquant, non fini, EIAS confondues, EIAS et EIPS confondues,
        ou quatre points colinéaires.
    """
    r_asi = as_point("RASI", RASI)
    l_asi = as_point("LASI", LASI)
    r_psi = as_point("RPSI", RPSI)
    l_psi = as_point("LPSI", LPSI)

    _check_distinct("RASI", r_asi, "LASI", l_asi)
    asis_mid = 0.5 * (r_asi + l_asi)
    psis_mid = 0.5 * (r_psi + l_psi)
    _check_distinct("milieu EIAS", asis_mid, "milieu EIPS", psis_mid)

    ml_raw = r_asi - l_asi
    ant_raw = asis_mid - psis_mid
    _check_non_parallel("inter-EIAS (ML)", ml_raw, "EIPS→EIAS (AP)", ant_raw)

    ml = _normalize(ml_raw, "ML")
    anterior = _normalize(ant_raw, "AP brut")
    si = _normalize(np.cross(ml, anterior), "SI")
    ap = _normalize(np.cross(si, ml), "AP")

    rotation = np.column_stack((ap, si, ml))
    return AnatomicalFrame(
        origin=asis_mid,
        rotation=rotation,
        axes=("AP", "SI", "ML"),
        name="pelvis",
    )


def pelvis_dimensions(RASI, LASI, RPSI, LPSI) -> dict[str, float]:
    """Largeur et profondeur pelviennes.

    ::

        PW = ||RASI - LASI||              # Pelvic Width
        PD = ||ASIS_mid - PSIS_mid||      # Pelvic Depth

    Returns
    -------
    dict
        ``{"PW": float, "PD": float}`` dans l'unité des repères d'entrée.
    """
    r_asi = as_point("RASI", RASI)
    l_asi = as_point("LASI", LASI)
    r_psi = as_point("RPSI", RPSI)
    l_psi = as_point("LPSI", LPSI)
    pw = _check_distinct("RASI", r_asi, "LASI", l_asi)
    asis_mid = 0.5 * (r_asi + l_asi)
    psis_mid = 0.5 * (r_psi + l_psi)
    pd = _check_distinct("milieu EIAS", asis_mid, "milieu EIPS", psis_mid)
    return {"PW": pw, "PD": pd}


# ── Scapula ───────────────────────────────────────────────────────────────────

def build_scapula_frame(AA, AI, TS, side: str = "right",
                        anterior_reference=None) -> AnatomicalFrame:
    """Construit le repère scapulaire ISB (Wu et al. 2005).

    ::

        origine = AA
        Z = normalize(AA - TS)                      # latéral
        n = cross(AI - AA, TS - AA)                 # normal au plan (AI, AA, TS)
        X = s * normalize(n)                        # antérieur
        Y = s * normalize(cross(Z, X))              # supérieur

    Le signe ``s`` vaut ``+1`` à droite et ``-1`` à gauche. En effet le produit
    vectoriel ``cross(AI - AA, TS - AA)`` pointe vers l'avant pour une scapula
    droite et vers l'arrière pour une scapula gauche (un miroir inverse
    l'orientation d'un produit vectoriel). Appliquer ``s`` aux deux axes ``X``
    et ``Y`` produit le **miroir exact** du repère droit, si bien que les
    coordonnées locales d'une scapula gauche sont numériquement identiques à
    celles de sa symétrique droite : les régressions de Meskers et Sobral,
    calibrées à droite, s'appliquent alors sans modification.

    Parameters
    ----------
    AA : array_like, shape (3,)
        Angulus acromialis (angle acromial) — origine du repère.
    AI : array_like, shape (3,)
        Angulus inferior (angle inférieur de la scapula).
    TS : array_like, shape (3,)
        Trigonum spinae scapulae.
    side : {"right", "left"}
        Côté de la scapula.
    anterior_reference : array_like, shape (3,), optional
        Direction antérieure connue (p.ex. l'axe AP du bassin). Si fournie,
        elle détermine le signe ``s`` à la place de ``side``.

    Returns
    -------
    AnatomicalFrame
        Axes ``("X", "Y", "Z")`` = antérieur, supérieur, latéral.
    """
    side = str(side).lower()
    if side not in ("right", "left"):
        raise ValueError(f"side doit être 'right' ou 'left', reçu {side!r}.")

    aa = as_point("AA", AA)
    ai = as_point("AI", AI)
    ts = as_point("TS", TS)

    _check_distinct("AA", aa, "TS", ts)
    _check_collinear(("AI", "AA", "TS"), (aa, ai, ts))

    z = _normalize(aa - ts, "Z (scapula)")
    normal = np.cross(ai - aa, ts - aa)

    if anterior_reference is not None:
        ref = as_point("anterior_reference", anterior_reference)
        dot = float(np.dot(normal, ref))
        if abs(dot) <= EPS:
            raise ValueError(
                "anterior_reference est perpendiculaire à la normale du plan "
                "scapulaire : orientation de l'axe X indéterminée."
            )
        s = 1.0 if dot > 0.0 else -1.0
    else:
        s = 1.0 if side == "right" else -1.0

    x = s * _normalize(normal, "X (scapula)")
    y = s * _normalize(np.cross(z, x), "Y (scapula)")

    rotation = np.column_stack((x, y, z))
    return AnatomicalFrame(
        origin=aa,
        rotation=rotation,
        axes=("X", "Y", "Z"),
        name=f"scapula_{side}",
    )


def scapula_distances(AA, AI, TS, AC=None, PC=None) -> dict[str, float]:
    """Distances inter-repères scapulaires (invariantes par isométrie).

    Retourne les clés ``L_AA_TS``, ``L_AA_AI``, ``L_TS_AI`` et, si ``AC`` /
    ``PC`` sont fournis, ``L_AA_AC``, ``L_TS_AC``, ``L_AI_AC``, ``L_AC_PC``,
    ``L_TS_PC``, ``L_AA_PC``.
    """
    aa = as_point("AA", AA)
    ai = as_point("AI", AI)
    ts = as_point("TS", TS)
    d: dict[str, float] = {
        "L_AA_TS": float(np.linalg.norm(aa - ts)),
        "L_AA_AI": float(np.linalg.norm(aa - ai)),
        "L_TS_AI": float(np.linalg.norm(ts - ai)),
    }
    if AC is not None:
        ac = as_point("AC", AC)
        d["L_AA_AC"] = float(np.linalg.norm(aa - ac))
        d["L_TS_AC"] = float(np.linalg.norm(ts - ac))
        d["L_AI_AC"] = float(np.linalg.norm(ai - ac))
        if PC is not None:
            pc = as_point("PC", PC)
            d["L_AC_PC"] = float(np.linalg.norm(ac - pc))
    if PC is not None:
        pc = as_point("PC", PC)
        d["L_TS_PC"] = float(np.linalg.norm(ts - pc))
        d["L_AA_PC"] = float(np.linalg.norm(aa - pc))
    return d
