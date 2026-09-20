"""Biomechanical measurements computable from surface landmark positions (3-D, mm).

Each public function takes ``ground_truth: dict[str, np.ndarray]`` and returns
a :class:`Measurement` or ``None`` when required landmarks are absent.
:func:`compute_all` aggregates every available measurement in one call.

Références
----------
- Wu G et al. (2002, 2005) ISB recommendation on definitions of JCS.
  J Biomech 35(4):543-548 & 38(5):981-992.
- Borstad JD (2006) Resting position variables at the shoulder.
  Phys Ther 86(4):549-557.  (Pectoralis Minor Index norms)
- ISAK (2001) International Standards for Anthropometric Assessment.
- Hermens HJ et al. (2000) SENIAM. J Electromyogr Kinesiol 10(5):361-374.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class Measurement:
    code: str           # identifiant machine
    label: str          # étiquette affichée
    value: float        # valeur numérique
    unit: str           # "mm", "°", "%", …
    norm_low: float | None = None   # borne basse de la norme (ou None)
    norm_high: float | None = None  # borne haute
    side: str = ""      # "D", "G", "D/G", ou ""
    category: str = ""  # "Posture", "Longueur", "ISAK", "ISB", "SENIAM"
    note: str = ""      # détail méthodologique

    @property
    def status(self) -> str:
        """'normal' / 'attention' / 'alerte' / 'info'."""
        if self.norm_low is None or self.norm_high is None:
            return "info"
        if self.norm_low <= self.value <= self.norm_high:
            return "normal"
        margin = (self.norm_high - self.norm_low) * 0.5
        if (self.norm_low - margin) <= self.value <= (self.norm_high + margin):
            return "attention"
        return "alerte"

    @property
    def value_str(self) -> str:
        if self.unit == "%":
            return f"{self.value:.1f} %"
        if self.unit == "°":
            return f"{self.value:.1f}°"
        return f"{self.value:.0f} mm"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(gt: dict, *codes: str) -> tuple[np.ndarray, ...] | None:
    """Return landmark positions for *codes*, or None if any is missing."""
    out = []
    for c in codes:
        if c not in gt:
            return None
        out.append(np.asarray(gt[c], dtype=np.float64))
    return tuple(out)


def _infer_up_axis(gt: dict) -> int:
    """Axis (0/1/2) with largest spatial range across all landmarks."""
    if not gt:
        return 1
    pts = np.array([np.asarray(v, dtype=np.float64) for v in gt.values()])
    extents = pts.max(axis=0) - pts.min(axis=0)
    return int(extents.argmax())


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def _angle_with_horizontal(vec: np.ndarray, up: int) -> float:
    """Signed angle (degrees) between *vec* and the horizontal plane."""
    v = vec.copy()
    v_up = np.zeros(3)
    v_up[up] = 1.0
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        return 0.0
    v = v / v_norm
    sin_a = float(np.dot(v, v_up))
    return float(np.degrees(np.arcsin(np.clip(sin_a, -1.0, 1.0))))


# ── Measurements ─────────────────────────────────────────────────────────────

def pectoralis_minor_index(gt: dict) -> Measurement | None:
    """Pectoralis Minor Index (%) — Borstad 2006 norm ≈ 7.65 %."""
    pts = _get(gt, "coracoid_process_right", "xiphoid_process", "suprasternal_notch")
    if pts is None:
        pts = _get(gt, "coracoid_process_left", "xiphoid_process", "suprasternal_notch")
        side = "G"
    else:
        side = "D"
    if pts is None:
        return None
    coracoid, xiphoid, jugular = pts
    # Origine approx : 40 % du vecteur xiphoid→jugular (côtes 3-5)
    origin = xiphoid + 0.4 * (jugular - xiphoid)
    length = _dist(origin, coracoid)
    thorax_height = _dist(xiphoid, jugular)
    if thorax_height < 1e-3:
        return None
    pmi = length / thorax_height * 100.0
    return Measurement(
        code="pmi", label="Pectoralis Minor Index", value=pmi,
        unit="%", norm_low=6.5, norm_high=8.8, side=side,
        category="Longueur",
        note="Borstad 2006 — norme ♂ 7.65 ± 0.6 %",
    )


def deltoid_length(gt: dict) -> Measurement | None:
    """Longueur deltoïde moyen — acromial_angle → greater_tubercle."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"acromial_angle{sfx}", f"greater_tubercle{sfx}")
        if pts is not None:
            return Measurement(
                code=f"deltoid_{side}", label="Long. deltoïde", value=_dist(*pts),
                unit="mm", side=side, category="Longueur",
                note="Acromial angle → grand tubercule",
            )
    return None


def biceps_length(gt: dict) -> Measurement | None:
    """Longueur biceps brachial — scapula_superior_angle → radiale."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"scapula_superior_angle{sfx}", f"radiale{sfx}")
        if pts is not None:
            return Measurement(
                code=f"biceps_{side}", label="Long. biceps brachial", value=_dist(*pts),
                unit="mm", side=side, category="Longueur",
                note="Angle sup. scapula → tête radiale",
            )
    return None


def upper_limb_length(gt: dict) -> Measurement | None:
    """Longueur membre supérieur — acromion → processus styloïde radial."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"acromion{sfx}", f"radial_styloid{sfx}")
        if pts is not None:
            return Measurement(
                code=f"upper_limb_{side}", label="Long. membre sup.", value=_dist(*pts),
                unit="mm", side=side, category="Longueur",
                note="Acromion → styloïde radial",
            )
    return None


def thigh_length(gt: dict) -> Measurement | None:
    """Longueur cuisse — grand trochanter → épicondyle latéral fémur."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"greater_trochanter{sfx}", f"lateral_knee{sfx}")
        if pts is not None:
            return Measurement(
                code=f"thigh_{side}", label="Long. cuisse", value=_dist(*pts),
                unit="mm", side=side, category="Longueur",
                note="Grand trochanter → épicondyle lat. fémur",
            )
    return None


def shank_length(gt: dict) -> Measurement | None:
    """Longueur jambe — épicondyle latéral → malléole latérale."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"lateral_knee{sfx}", f"lateral_malleolus{sfx}")
        if pts is not None:
            return Measurement(
                code=f"shank_{side}", label="Long. jambe", value=_dist(*pts),
                unit="mm", side=side, category="Longueur",
                note="Épicondyle lat. fémur → malléole lat.",
            )
    return None


# ── Indices posturaux ─────────────────────────────────────────────────────────

def pelvic_tilt(gt: dict) -> Measurement | None:
    """Version pelvienne sagittale (°) — ASIS/PSIS. Norme debout ≈ 7-13°."""
    up = _infer_up_axis(gt)
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"ASIS{sfx}", f"PSIS{sfx}")
        if pts is None:
            continue
        asis, psis = pts
        vec = asis - psis  # pointe vers l'avant et le haut en version ant.
        angle = _angle_with_horizontal(vec, up)
        return Measurement(
            code="pelvic_tilt", label="Tilt pelvien",
            value=round(angle, 1), unit="°",
            norm_low=7.0, norm_high=13.0, side=side,
            category="Posture",
            note="ASIS–PSIS / horizontal. + = version ant.",
        )
    return None


def pelvic_obliquity(gt: dict) -> Measurement | None:
    """Obliquité pelvienne frontale (mm) — diff. hauteur ASIS G vs D."""
    pts = _get(gt, "ASIS_left", "ASIS_right")
    if pts is None:
        return None
    up = _infer_up_axis(gt)
    left, right = pts
    delta = float(left[up] - right[up])
    return Measurement(
        code="pelvic_obliquity", label="Obliquité pelvienne",
        value=round(abs(delta), 1), unit="mm",
        norm_low=0.0, norm_high=10.0, side="G/D",
        category="Posture",
        note=f"ASIS {'gauche' if delta > 0 else 'droit'} plus haut de {abs(delta):.0f} mm",
    )


def lateral_spine_shift(gt: dict) -> Measurement | None:
    """Lateral Spine Shift Test (mm) — C7 vs mi-PSIS. Norme < 20 mm."""
    pts = _get(gt, "C7_spinous", "PSIS_left", "PSIS_right")
    if pts is None:
        return None
    up = _infer_up_axis(gt)
    c7, psis_l, psis_r = pts
    # Lateral axes = the two non-up axes; take the remaining one
    axes = [i for i in range(3) if i != up]
    mid_psis = (psis_l + psis_r) / 2.0
    delta_lat = float(c7[axes[0]] - mid_psis[axes[0]])
    return Measurement(
        code="lsst", label="Déviation latérale rachis",
        value=round(abs(delta_lat), 1), unit="mm",
        norm_low=0.0, norm_high=20.0, side="",
        category="Posture",
        note="C7 vs mi-PSIS. > 20 mm = scoliose fonctionnelle",
    )


def scapular_height_asymmetry(gt: dict) -> Measurement | None:
    """Asymétrie scapulaire hauteur (mm) — angle inférieur G vs D."""
    pts = _get(gt, "scapula_inferior_angle_left", "scapula_inferior_angle_right")
    if pts is None:
        return None
    up = _infer_up_axis(gt)
    left, right = pts
    delta = float(left[up] - right[up])
    return Measurement(
        code="scap_height", label="Asym. scapulaire hauteur",
        value=round(abs(delta), 1), unit="mm",
        norm_low=0.0, norm_high=15.0, side="G/D",
        category="Posture",
        note=f"Angle inf. scapula {'gauche' if delta > 0 else 'droit'} plus haut",
    )


def shoulder_height_asymmetry(gt: dict) -> Measurement | None:
    """Asymétrie hauteur épaules (mm) — acromion G vs D."""
    pts = _get(gt, "acromion_left", "acromion_right")
    if pts is None:
        return None
    up = _infer_up_axis(gt)
    left, right = pts
    delta = float(left[up] - right[up])
    return Measurement(
        code="shoulder_height", label="Asym. hauteur épaules",
        value=round(abs(delta), 1), unit="mm",
        norm_low=0.0, norm_high=15.0, side="G/D",
        category="Posture",
        note=f"Acromion {'gauche' if delta > 0 else 'droit'} plus haut",
    )


# ── Anthropométrie ISAK ───────────────────────────────────────────────────────

def acromiale_radiale_length(gt: dict) -> Measurement | None:
    """Longueur acromiale-radiale ISAK (mm) — site mi-acromiale-radiale."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"acromion{sfx}", f"radiale{sfx}")
        if pts is not None:
            return Measurement(
                code=f"acromiale_radiale_{side}", label="Acromiale-radiale",
                value=_dist(*pts), unit="mm",
                norm_low=300.0, norm_high=380.0, side=side,
                category="ISAK",
                note="Acromion → tête radiale (point mi-acr.-rad. = milieu)",
            )
    return None


def humerus_bicondylar(gt: dict) -> Measurement | None:
    """Diamètre bi-épicondylien de l'humérus ISAK (mm)."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"lateral_epicondyle{sfx}", f"medial_epicondyle{sfx}")
        if pts is not None:
            return Measurement(
                code=f"humerus_biep_{side}", label="Bi-épicondyle humérus",
                value=_dist(*pts), unit="mm",
                norm_low=50.0, norm_high=80.0, side=side,
                category="ISAK",
                note="Épicondyle lat. → épicondyle méd.",
            )
    return None


def femur_bicondylar(gt: dict) -> Measurement | None:
    """Diamètre bi-épicondylien du fémur ISAK (mm)."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"lateral_knee{sfx}", f"medial_knee{sfx}")
        if pts is not None:
            return Measurement(
                code=f"femur_biep_{side}", label="Bi-épicondyle fémur",
                value=_dist(*pts), unit="mm",
                norm_low=80.0, norm_high=110.0, side=side,
                category="ISAK",
                note="Épicondyle lat. → épicondyle méd. fémur",
            )
    return None


# ── Sites EMG SENIAM ──────────────────────────────────────────────────────────

def seniam_vastus_lateralis(gt: dict) -> Measurement | None:
    """Position SENIAM vaste latéral — 1/3 distal ASIS→épicondyle lat. (D)."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"ASIS{sfx}", f"lateral_knee{sfx}")
        if pts is None:
            continue
        asis, knee = pts
        site = asis + 0.333 * (knee - asis)
        d_from_asis = _dist(asis, site)
        return Measurement(
            code=f"seniam_vl_{side}", label="SENIAM VL — dist. ASIS",
            value=round(d_from_asis, 1), unit="mm",
            side=side, category="SENIAM",
            note="Site à 1/3 distal ASIS–épicondyle lat. (SENIAM)",
        )
    return None


def seniam_biceps_brachii(gt: dict) -> Measurement | None:
    """Position SENIAM biceps brachial — 1/3 distal acromion→fossa cubitale (D)."""
    for side, sfx in (("D", "_right"), ("G", "_left")):
        pts = _get(gt, f"acromion{sfx}", f"medial_epicondyle{sfx}")
        if pts is None:
            continue
        acromion, ep = pts
        site = acromion + 0.333 * (ep - acromion)
        d_from_acromion = _dist(acromion, site)
        return Measurement(
            code=f"seniam_bb_{side}", label="SENIAM Biceps — dist. acromion",
            value=round(d_from_acromion, 1), unit="mm",
            side=side, category="SENIAM",
            note="Site à 1/3 distal acromion–fosse cubitale (SENIAM)",
        )
    return None


# ── API publique ──────────────────────────────────────────────────────────────

_ALL_FNS: list[Callable] = [
    # Posture
    pelvic_tilt,
    pelvic_obliquity,
    lateral_spine_shift,
    scapular_height_asymmetry,
    shoulder_height_asymmetry,
    # Longueurs
    pectoralis_minor_index,
    deltoid_length,
    biceps_length,
    upper_limb_length,
    thigh_length,
    shank_length,
    # ISAK
    acromiale_radiale_length,
    humerus_bicondylar,
    femur_bicondylar,
    # SENIAM
    seniam_vastus_lateralis,
    seniam_biceps_brachii,
]

_CATEGORY_ORDER = ["Posture", "Longueur", "ISAK", "SENIAM"]


def compute_all(ground_truth: dict) -> list[Measurement]:
    """Return every available measurement, sorted by category."""
    results: list[Measurement] = []
    for fn in _ALL_FNS:
        try:
            m = fn(ground_truth)
            if m is not None:
                results.append(m)
        except Exception:
            pass
    return sorted(results, key=lambda m: (
        _CATEGORY_ORDER.index(m.category) if m.category in _CATEGORY_ORDER else 99,
        m.label,
    ))
