"""Anthropometric measurements exercise module for BonyLandmarks.

Twenty structured measurements computable from surface landmark positions (3-D, mm).
Each AnthroMeasure subclass implements ``compute(ground_truth) -> float | None``.

Landmark fallbacks
------------------
- ``tragus_right`` / ``tragus_left``   → ``external_acoustic_meatus_right`` / ``_left``
- ``hallux_right`` / ``hallux_left``   → ``first_metatarsal_head_right``    / ``_left``

References
----------
- Magee DJ (2014) Orthopedic Physical Assessment. 5th ed.
- Norkin CC & White DJ (1995) Measurement of Joint Motion. 3rd ed.
- Kapandji IA (2010) The Physiology of the Joints. 6th ed.
- ISAK (2001) International Standards for Anthropometric Assessment.
- Yoo WG (2013) CVA and neck pain. J Phys Ther Sci 25(5):571-572.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _get(gt: dict, *codes: str) -> Optional[tuple]:
    """Return positions for all *codes* as float64 arrays, or None if any absent."""
    out = []
    for c in codes:
        v = gt.get(c)
        if v is None:
            return None
        out.append(np.asarray(v, dtype=np.float64))
    return tuple(out)


def _infer_up_axis(gt: dict) -> int:
    """Axis index (0/1/2) with the largest spatial range across all landmarks."""
    if not gt:
        return 1
    pts = np.array([np.asarray(v, dtype=np.float64) for v in gt.values()])
    extents = pts.max(axis=0) - pts.min(axis=0)
    return int(extents.argmax())


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean 3-D distance between two points."""
    return float(np.linalg.norm(b - a))


def _horizontal_axes(up: int) -> tuple[int, int]:
    """Return (mediolateral, anteroposterior) axis indices — the two that are not *up*."""
    axes = [i for i in range(3) if i != up]
    return axes[0], axes[1]


def _project_plane(v: np.ndarray, normal_axis: int) -> np.ndarray:
    """Zero the *normal_axis* component of *v*, projecting onto the orthogonal plane."""
    v2 = v.copy()
    v2[normal_axis] = 0.0
    return v2


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (°) at vertex B formed by vectors BA and BC (full 3-D)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle3_proj(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, normal_axis: int
) -> float:
    """Angle (°) at vertex B after projecting A and C onto the plane ⊥ to *normal_axis*."""
    a2 = _project_plane(a - b, normal_axis)
    c2 = _project_plane(c - b, normal_axis)
    n1 = np.linalg.norm(a2)
    n2 = np.linalg.norm(c2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.dot(a2, c2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_horiz(vec: np.ndarray, up: int) -> float:
    """Signed elevation angle (°) of *vec* relative to the horizontal plane."""
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return 0.0
    sin_a = float(vec[up] / n)
    return float(np.degrees(np.arcsin(np.clip(sin_a, -1.0, 1.0))))


def _with_fallback(gt: dict, primary: str, fallback: str) -> Optional[np.ndarray]:
    """Return the primary landmark, or fallback if primary is absent."""
    v = gt.get(primary) or gt.get(fallback)
    return np.asarray(v, dtype=np.float64) if v is not None else None


# ── Core dataclasses ───────────────────────────────────────────────────────────

@dataclass
class AnthroMeasure:
    """Base class for all 20 anthropometric exercise measures.

    Subclasses override ``compute()`` only; all metadata lives here.
    """

    code: str
    name_fr: str
    name_en: str
    required_landmarks: list[str]
    formula_description_fr: str
    formula_description_en: str
    unit: str                        # "mm", "°", "%"
    norm_low: float | None = None
    norm_high: float | None = None
    side: str = ""                   # "right", "left", "bilateral", ""
    category: str = ""               # "posture", "longueur", "angle", "asymetrie"

    def compute(self, ground_truth: dict[str, np.ndarray]) -> float | None:
        """Return computed value, or None when required landmarks are insufficient."""
        return None

    def status_for(self, value: float) -> str:
        """Return 'normal', 'attention', 'alerte', or 'info' for *value*."""
        if self.norm_low is None or self.norm_high is None:
            return "info"
        if self.norm_low <= value <= self.norm_high:
            return "normal"
        margin = (self.norm_high - self.norm_low) * 0.5
        if (self.norm_low - margin) <= value <= (self.norm_high + margin):
            return "attention"
        return "alerte"

    def value_str(self, value: float) -> str:
        """Return a formatted display string for *value*."""
        if self.unit == "%":
            return f"{value:.1f} %"
        if self.unit == "°":
            return f"{value:.1f}°"
        return f"{value:.0f} mm"


@dataclass
class AnthroMeasureResult:
    """Pairing of an AnthroMeasure with its computed value for one subject."""

    measure: AnthroMeasure
    value: float | None

    @property
    def status(self) -> str:
        """'normal' / 'attention' / 'alerte' / 'info'."""
        if self.value is None:
            return "info"
        return self.measure.status_for(self.value)

    @property
    def value_str(self) -> str:
        """Formatted display string, or '—' when value is None."""
        if self.value is None:
            return "—"
        return self.measure.value_str(self.value)


# ── Measure 1 — Lower limb length ─────────────────────────────────────────────

@dataclass
class LowerLimbLength(AnthroMeasure):
    """3-D distance ASIS → medial malleolus (functional lower limb length)."""

    def compute(self, gt: dict) -> float | None:  # noqa: D102
        for sfx in ("_right", "_left"):
            pts = _get(gt, f"ASIS{sfx}", f"medial_malleolus{sfx}")
            if pts is not None:
                return _dist(*pts)
        return None


# ── Measure 2 — Upper limb length ─────────────────────────────────────────────

@dataclass
class UpperLimbLength(AnthroMeasure):
    """3-D distance acromion → radial styloid (functional upper limb length)."""

    def compute(self, gt: dict) -> float | None:
        for sfx in ("_right", "_left"):
            pts = _get(gt, f"acromion{sfx}", f"radial_styloid{sfx}")
            if pts is not None:
                return _dist(*pts)
        return None


# ── Measure 3 — Pelvic tilt ───────────────────────────────────────────────────

@dataclass
class PelvicTilt(AnthroMeasure):
    """Angle ASIS–PSIS relative to horizontal in the sagittal plane (°).

    Positive = anterior pelvic tilt (norm 10–12° standing).
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        for sfx in ("_right", "_left"):
            pts = _get(gt, f"ASIS{sfx}", f"PSIS{sfx}")
            if pts is not None:
                asis, psis = pts
                vec = asis - psis   # points anteriorly when asis is forward
                return abs(_angle_horiz(vec, up))
        return None


# ── Measure 4 — Q angle ───────────────────────────────────────────────────────

@dataclass
class QAngle(AnthroMeasure):
    """ASIS–patella–tibial tuberosity angle projected onto the frontal plane (°).

    Norm ≈ 10–15° (female), 8–10° (male). Elevated Q angle → patellofemoral risk.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        _, ap = _horizontal_axes(up)   # project on frontal → normal = AP axis
        for sfx in ("_right", "_left"):
            pts = _get(gt, f"ASIS{sfx}", f"patella_center{sfx}", f"tibial_tuberosity{sfx}")
            if pts is not None:
                asis, patella, tib = pts
                return _angle3_proj(asis, patella, tib, ap)
        return None


# ── Measure 5 — Knee valgus / varus ──────────────────────────────────────────

@dataclass
class KneeValgusVarus(AnthroMeasure):
    """Angle between femoral axis (GT→lat. knee) and tibial axis (lat. knee→lat. malleolus)
    in the frontal plane (°).  0° = perfect alignment; positive = valgus deviation.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        _, ap = _horizontal_axes(up)
        for sfx in ("_right", "_left"):
            pts = _get(
                gt,
                f"greater_trochanter{sfx}",
                f"lateral_knee{sfx}",
                f"lateral_malleolus{sfx}",
            )
            if pts is not None:
                troch, knee, malle = pts
                raw = _angle3_proj(troch, knee, malle, ap)
                return abs(180.0 - raw)   # deviation from straight-line alignment
        return None


# ── Measure 6 — Scapular height asymmetry ────────────────────────────────────

@dataclass
class ScapularHeightAsymmetry(AnthroMeasure):
    """Vertical height difference between left and right inferior scapular angles (mm).

    Norm < 15 mm.
    """

    def compute(self, gt: dict) -> float | None:
        pts = _get(gt, "scapula_inferior_angle_left", "scapula_inferior_angle_right")
        if pts is None:
            return None
        up = _infer_up_axis(gt)
        left, right = pts
        return abs(float(left[up] - right[up]))


# ── Measure 7 — Shoulder height asymmetry ────────────────────────────────────

@dataclass
class ShoulderHeightAsymmetry(AnthroMeasure):
    """Vertical height difference between left and right acromions (mm).

    Norm < 10 mm.
    """

    def compute(self, gt: dict) -> float | None:
        pts = _get(gt, "acromion_left", "acromion_right")
        if pts is None:
            return None
        up = _infer_up_axis(gt)
        left, right = pts
        return abs(float(left[up] - right[up]))


# ── Measure 8 — Craniovertebral angle ────────────────────────────────────────

@dataclass
class CraniovertebralAngle(AnthroMeasure):
    """Elevation angle of the line C7→tragus above horizontal (°).

    Norm ≈ 50°.  Forward-head posture: CVA < 46°.
    Fallback: external_acoustic_meatus if tragus absent.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        c7_raw = gt.get("C7_spinous")
        if c7_raw is None:
            return None
        c7 = np.asarray(c7_raw, dtype=np.float64)
        for sfx in ("_right", "_left"):
            tragus = _with_fallback(gt, f"tragus{sfx}", f"external_acoustic_meatus{sfx}")
            if tragus is not None:
                vec = tragus - c7   # from C7 upward/forward toward ear
                return abs(_angle_horiz(vec, up))
        return None


# ── Measure 9 — Cervical lateral inclination ─────────────────────────────────

@dataclass
class CervicalLateralInclination(AnthroMeasure):
    """Deviation of the tragus–C7 line from vertical in the frontal plane (°).

    Norm ≈ 0°.  Positive = head tilted laterally.
    Fallback: external_acoustic_meatus if tragus absent.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        _, ap = _horizontal_axes(up)
        c7_raw = gt.get("C7_spinous")
        if c7_raw is None:
            return None
        c7 = np.asarray(c7_raw, dtype=np.float64)
        for sfx in ("_right", "_left"):
            tragus = _with_fallback(gt, f"tragus{sfx}", f"external_acoustic_meatus{sfx}")
            if tragus is not None:
                vec = tragus - c7
                # Project onto frontal plane (remove AP depth component)
                v_front = _project_plane(vec, ap)
                n = np.linalg.norm(v_front)
                if n < 1e-9:
                    return 0.0
                v_norm = v_front / n
                v_up = np.zeros(3)
                v_up[up] = 1.0
                cos_a = float(np.dot(v_norm, v_up))
                # 0° = perfectly vertical; positive = lateral tilt
                return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return None


# ── Measure 10 — Cervical rotation asymmetry ─────────────────────────────────

@dataclass
class CervicalRotationAsymmetry(AnthroMeasure):
    """Difference in anteroposterior position of left vs right acoustic meatus rel. to C7 (mm).

    Captures head rotation: 0 = symmetric, large value = rotated posture.
    """

    def compute(self, gt: dict) -> float | None:
        pts = _get(
            gt,
            "external_acoustic_meatus_left",
            "external_acoustic_meatus_right",
            "C7_spinous",
        )
        if pts is None:
            return None
        up = _infer_up_axis(gt)
        _, ap = _horizontal_axes(up)
        left, right, c7 = pts
        left_ap = float(left[ap] - c7[ap])
        right_ap = float(right[ap] - c7[ap])
        return abs(left_ap - right_ap)


# ── Measure 11 — Carrying angle ───────────────────────────────────────────────

@dataclass
class CarryingAngle(AnthroMeasure):
    """Acromion–lat. epicondyle–radial styloid angle in the frontal plane (°).

    Norm 5–15° (physiological cubitus valgus).
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        _, ap = _horizontal_axes(up)
        for sfx in ("_right", "_left"):
            pts = _get(
                gt,
                f"acromion{sfx}",
                f"lateral_epicondyle{sfx}",
                f"radial_styloid{sfx}",
            )
            if pts is not None:
                acromion, epi, styloid = pts
                return _angle3_proj(acromion, epi, styloid, ap)
        return None


# ── Measure 12 — Elbow hyperextension ────────────────────────────────────────

@dataclass
class ElbowHyperextension(AnthroMeasure):
    """Degrees of elbow hyperextension measured in the sagittal plane (°).

    Uses olecranon–lat. epicondyle–radial styloid if olecranon available;
    falls back to acromion–lat. epicondyle–radial styloid.
    0° = neutral; positive = hyperextension.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        ml, _ = _horizontal_axes(up)   # project on sagittal → normal = ML axis
        for sfx in ("_right", "_left"):
            epi_raw = gt.get(f"lateral_epicondyle{sfx}")
            styloid_raw = gt.get(f"radial_styloid{sfx}")
            if epi_raw is None or styloid_raw is None:
                continue
            epi = np.asarray(epi_raw, dtype=np.float64)
            styloid = np.asarray(styloid_raw, dtype=np.float64)
            # Prefer olecranon; fall back to acromion as proximal reference
            prox_raw = gt.get(f"olecranon{sfx}") or gt.get(f"acromion{sfx}")
            if prox_raw is None:
                continue
            prox = np.asarray(prox_raw, dtype=np.float64)
            angle = _angle3_proj(prox, epi, styloid, ml)
            # 180° = straight arm; deviation below 180° = hyperextension in this geometry
            return max(0.0, 180.0 - angle)
        return None


# ── Measure 13 — Genu recurvatum ──────────────────────────────────────────────

@dataclass
class GenuRecurvatum(AnthroMeasure):
    """Knee hyperextension: deviation from straight in the sagittal plane (°).

    Uses greater_trochanter–lat. knee–lat. malleolus.  0° = neutral.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        ml, _ = _horizontal_axes(up)
        for sfx in ("_right", "_left"):
            pts = _get(
                gt,
                f"greater_trochanter{sfx}",
                f"lateral_knee{sfx}",
                f"lateral_malleolus{sfx}",
            )
            if pts is not None:
                troch, knee, malle = pts
                angle = _angle3_proj(troch, knee, malle, ml)
                return max(0.0, 180.0 - angle)
        return None


# ── Measure 14 — Hallux valgus angle ─────────────────────────────────────────

@dataclass
class HalluxValgusAngle(AnthroMeasure):
    """2nd metatarsal head – 1st metatarsal head – hallux angle in the transverse plane (°).

    Norm < 15°.  Fallback: first_metatarsal_head if hallux absent (angle ≈ 0°).
    Requires second_metatarsal_head landmark.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        for sfx in ("_right", "_left"):
            pts_met = _get(gt, f"second_metatarsal_head{sfx}", f"first_metatarsal_head{sfx}")
            if pts_met is None:
                continue
            met2, met1 = pts_met
            hallux = _with_fallback(gt, f"hallux{sfx}", f"first_metatarsal_head{sfx}")
            if hallux is None:
                continue
            # Vertex at first metatarsal head; project onto transverse plane (normal = up)
            a2 = _project_plane(met2 - met1, up)
            c2 = _project_plane(hallux - met1, up)
            n1 = np.linalg.norm(a2)
            n2 = np.linalg.norm(c2)
            if n1 < 1e-9 or n2 < 1e-9:
                return None
            cos_a = float(np.dot(a2, c2) / (n1 * n2))
            return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return None


# ── Measure 15 — Calcaneus valgus ────────────────────────────────────────────

@dataclass
class CalcaneusValgus(AnthroMeasure):
    """Inclination angle of the midmalleolus–heel vector from vertical (°).

    Estimated via medial_malleolus, lateral_malleolus and heel.
    Norm 0–5° lateral deviation.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        v_up = np.zeros(3)
        v_up[up] = 1.0
        for sfx in ("_right", "_left"):
            pts = _get(
                gt,
                f"medial_malleolus{sfx}",
                f"lateral_malleolus{sfx}",
                f"heel{sfx}",
            )
            if pts is not None:
                med_malle, lat_malle, heel = pts
                mid_malle = (med_malle + lat_malle) / 2.0
                vec = mid_malle - heel   # heel → midmalleolus (should be upward)
                n = np.linalg.norm(vec)
                if n < 1e-9:
                    return 0.0
                v_norm = vec / n
                cos_a = float(np.dot(v_norm, v_up))
                # Angle from vertical; 0° = heel perfectly centered
                return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return None


# ── Measure 16 — Navicular drop ───────────────────────────────────────────────

@dataclass
class NavicularDrop(AnthroMeasure):
    """Vertical height of the navicular tuberosity below the medial malleolus (mm).

    Proxy for medial arch height.  Norm ≈ 15–20 mm.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        for sfx in ("_right", "_left"):
            pts = _get(gt, f"navicular_tuberosity{sfx}", f"medial_malleolus{sfx}")
            if pts is not None:
                nav, malle = pts
                # Positive = malleolus above navicular (normal arch)
                return abs(float(malle[up] - nav[up]))
        return None


# ── Measure 17 — Tibial torsion ───────────────────────────────────────────────

@dataclass
class TibialTorsion(AnthroMeasure):
    """Angle between the condyle axis (med. knee → lat. knee) and the bimalleolar axis,
    both projected onto the transverse plane (°).

    Norm ≈ 15–30° (external tibial torsion).
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        for sfx in ("_right", "_left"):
            pts = _get(
                gt,
                f"medial_knee{sfx}",
                f"lateral_knee{sfx}",
                f"medial_malleolus{sfx}",
                f"lateral_malleolus{sfx}",
            )
            if pts is not None:
                med_knee, lat_knee, med_malle, lat_malle = pts
                condyle_ax = lat_knee - med_knee
                malle_ax = lat_malle - med_malle
                cond_proj = _project_plane(condyle_ax, up)
                malle_proj = _project_plane(malle_ax, up)
                n1 = np.linalg.norm(cond_proj)
                n2 = np.linalg.norm(malle_proj)
                if n1 < 1e-9 or n2 < 1e-9:
                    return None
                cos_a = float(np.dot(cond_proj, malle_proj) / (n1 * n2))
                return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return None


# ── Measure 18 — Femoral anteversion (clinical proxy) ────────────────────────

@dataclass
class FemoralAnteversionClinical(AnthroMeasure):
    """Clinical proxy: anteroposterior inclination of the GT→ASIS vector (°).

    Measures how far the ASIS lies anterior to the greater trochanter relative
    to the horizontal plane — a rough anteversion estimate from surface landmarks.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        _, ap = _horizontal_axes(up)
        for sfx in ("_right", "_left"):
            pts = _get(gt, f"greater_trochanter{sfx}", f"ASIS{sfx}")
            if pts is not None:
                troch, asis = pts
                vec = asis - troch
                vec_horiz = _project_plane(vec, up)
                n = np.linalg.norm(vec_horiz)
                if n < 1e-9:
                    return 0.0
                ap_comp = abs(float(vec_horiz[ap]))
                return float(np.degrees(np.arcsin(np.clip(ap_comp / n, -1.0, 1.0))))
        return None


# ── Measure 19 — Shoulder protraction ────────────────────────────────────────

@dataclass
class ShoulderProtraction(AnthroMeasure):
    """Mean AP position of both acromions relative to the C7–T8 thoracic midplane (mm).

    Positive = shoulders forward.  Norm ≈ 0.
    """

    def compute(self, gt: dict) -> float | None:
        up = _infer_up_axis(gt)
        _, ap = _horizontal_axes(up)
        pts_acr = _get(gt, "acromion_left", "acromion_right")
        if pts_acr is None:
            return None
        acr_l, acr_r = pts_acr
        c7_raw = gt.get("C7_spinous")
        t8_raw = gt.get("T8_spinous")
        if c7_raw is None:
            return None
        c7 = np.asarray(c7_raw, dtype=np.float64)
        if t8_raw is not None:
            t8 = np.asarray(t8_raw, dtype=np.float64)
            ref_ap = float((c7[ap] + t8[ap]) / 2.0)
        else:
            ref_ap = float(c7[ap])
        left_prot = float(acr_l[ap]) - ref_ap
        right_prot = float(acr_r[ap]) - ref_ap
        return float((left_prot + right_prot) / 2.0)


# ── Measure 20 — Pelvic obliquity ────────────────────────────────────────────

@dataclass
class PelvicObliquity(AnthroMeasure):
    """Vertical height difference between ASIS_left and ASIS_right (mm).

    Norm 0 mm (symmetric pelvis).
    """

    def compute(self, gt: dict) -> float | None:
        pts = _get(gt, "ASIS_left", "ASIS_right")
        if pts is None:
            return None
        up = _infer_up_axis(gt)
        left, right = pts
        return abs(float(left[up] - right[up]))


# ── Public API ─────────────────────────────────────────────────────────────────

_STEP_GUIDE: dict[str, dict[str, list[str]]] = {
    "lower_limb_length": {
        "fr": [
            "Localiser l'EIAS (saillie antérieure de la crête iliaque) du même côté.",
            "Localiser la malléole médiale (saillie interne de la cheville).",
            "La longueur est la distance 3D entre ces deux repères.",
        ],
        "en": [
            "Locate the ASIS (anterior iliac crest prominence) on the same side.",
            "Locate the medial malleolus (inner ankle prominence).",
            "Length is the 3-D distance between these two landmarks.",
        ],
    },
    "upper_limb_length": {
        "fr": [
            "Localiser l'acromion (bord latéral de l'épine de la scapula).",
            "Localiser la styloïde radiale (poignet, côté pouce).",
            "La longueur est la distance 3D entre ces deux repères.",
        ],
        "en": [
            "Locate the acromion (lateral tip of the scapular spine).",
            "Locate the radial styloid (thumb-side wrist prominence).",
            "Length is the 3-D distance between these two landmarks.",
        ],
    },
    "pelvic_tilt": {
        "fr": [
            "Placer un repère sur l'EIAS et un sur l'EIPS du même côté.",
            "Tracer la ligne EIAS–EIPS dans le plan sagittal.",
            "Mesurer l'angle de cette ligne avec l'horizontale (norme 10–12°).",
        ],
        "en": [
            "Place one landmark on the ASIS and one on the PSIS on the same side.",
            "Draw the ASIS–PSIS line in the sagittal plane.",
            "Measure the angle of this line with the horizontal (norm 10–12°).",
        ],
    },
    "Q_angle": {
        "fr": [
            "Identifier l'EIAS, le centre de la patella et la tubérosité tibiale.",
            "Projeter ces trois points sur le plan frontal.",
            "L'angle au sommet patellaire (EIAS–patella–TT) est l'angle Q (norme 10–15° ♀, 8–10° ♂).",
        ],
        "en": [
            "Identify the ASIS, patella center, and tibial tuberosity.",
            "Project these three points onto the frontal plane.",
            "The angle at the patellar vertex (ASIS–patella–TT) is the Q angle (norm 10–15° female, 8–10° male).",
        ],
    },
    "knee_valgus_varus": {
        "fr": [
            "Identifier le grand trochanter, le condyle latéral du genou et la malléole latérale.",
            "Tracer l'axe fémoral (GT→condyle lat.) et l'axe tibial (condyle lat.→malléole lat.) dans le plan frontal.",
            "La déviation par rapport à l'alignement parfait (180°) indique valgus ou varus.",
        ],
        "en": [
            "Identify the greater trochanter, lateral knee condyle, and lateral malleolus.",
            "Draw the femoral axis (GT→lat. condyle) and tibial axis (lat. condyle→lat. malleolus) in the frontal plane.",
            "Deviation from perfect alignment (180°) indicates valgus or varus.",
        ],
    },
    "scapular_height_asymmetry": {
        "fr": [
            "Identifier l'angle inférieur de chaque scapula (pointe inférieure).",
            "Mesurer la hauteur (axe vertical) de chaque point.",
            "La différence entre côté gauche et droit (norme < 15 mm) révèle une asymétrie scapulaire.",
        ],
        "en": [
            "Identify the inferior scapular angle (bottom tip) on each side.",
            "Measure the vertical height of each point.",
            "The left-right difference (norm < 15 mm) reveals scapular asymmetry.",
        ],
    },
    "shoulder_height_asymmetry": {
        "fr": [
            "Identifier l'acromion droit et l'acromion gauche.",
            "Mesurer leur hauteur respective sur l'axe vertical.",
            "La différence (norme < 10 mm) indique une épaule plus haute.",
        ],
        "en": [
            "Identify both acromions (right and left).",
            "Measure their respective heights on the vertical axis.",
            "The difference (norm < 10 mm) indicates one shoulder higher than the other.",
        ],
    },
    "craniovertebral_angle": {
        "fr": [
            "Identifier C7 (vertèbre proéminente à la base du cou) et le tragus de l'oreille.",
            "Tracer la ligne C7→tragus dans le plan sagittal.",
            "L'angle avec l'horizontale est l'angle cranio-vertébral (norme ≈ 50° ; < 46° = tête avancée).",
        ],
        "en": [
            "Identify C7 (prominent vertebra at the base of the neck) and the ear tragus.",
            "Draw the C7→tragus line in the sagittal plane.",
            "The angle with the horizontal is the craniovertebral angle (norm ≈ 50°; < 46° = forward head).",
        ],
    },
    "cervical_lateral_inclination": {
        "fr": [
            "Identifier C7 et le tragus (ou méat acoustique externe) du même côté.",
            "Projeter la ligne C7→tragus sur le plan frontal.",
            "Mesurer l'écart par rapport à la verticale (norme ≈ 0° ; positif = inclinaison latérale).",
        ],
        "en": [
            "Identify C7 and the tragus (or external acoustic meatus) on the same side.",
            "Project the C7→tragus line onto the frontal plane.",
            "Measure the deviation from vertical (norm ≈ 0°; positive = lateral tilt).",
        ],
    },
    "cervical_rotation_asymmetry": {
        "fr": [
            "Identifier les deux méats acoustiques externes et C7.",
            "Mesurer la position antéro-postérieure de chaque méat relativement à C7.",
            "La différence G–D indique une rotation de tête asymétrique (norme ≈ 0 mm).",
        ],
        "en": [
            "Identify both external acoustic meatuses and C7.",
            "Measure the anteroposterior position of each meatus relative to C7.",
            "The left-right difference indicates asymmetric head rotation (norm ≈ 0 mm).",
        ],
    },
    "carrying_angle": {
        "fr": [
            "Identifier l'acromion, l'épicondyle latéral et la styloïde radiale du même côté.",
            "Projeter ces points sur le plan frontal (bras en extension).",
            "L'angle au sommet (épicondyle lat.) est l'angle de transport (norme 5–15°).",
        ],
        "en": [
            "Identify the acromion, lateral epicondyle, and radial styloid on the same side.",
            "Project these points onto the frontal plane (arm in extension).",
            "The angle at the vertex (lat. epicondyle) is the carrying angle (norm 5–15°).",
        ],
    },
    "elbow_hyperextension": {
        "fr": [
            "Identifier l'olécrâne (ou l'acromion), l'épicondyle latéral et la styloïde radiale.",
            "Projeter ces points sur le plan sagittal.",
            "Un angle < 180° en extension indique une hyperextension (norme 0–10°).",
        ],
        "en": [
            "Identify the olecranon (or acromion), lateral epicondyle, and radial styloid.",
            "Project these points onto the sagittal plane.",
            "An angle < 180° in extension indicates hyperextension (norm 0–10°).",
        ],
    },
    "genu_recurvatum": {
        "fr": [
            "Identifier le grand trochanter, le condyle latéral du genou et la malléole latérale.",
            "Projeter ces points sur le plan sagittal.",
            "Un angle au genou < 180° révèle un genu recurvatum (norme ≈ 0°).",
        ],
        "en": [
            "Identify the greater trochanter, lateral knee condyle, and lateral malleolus.",
            "Project these points onto the sagittal plane.",
            "A knee angle < 180° reveals genu recurvatum (norm ≈ 0°).",
        ],
    },
    "hallux_valgus_angle": {
        "fr": [
            "Identifier la tête du 2e métatarse, la tête du 1er métatarse et le hallux.",
            "Projeter ces points sur le plan transverse (vue plantaire).",
            "L'angle au sommet (1er métatarse) mesure le valgus de l'hallux (norme < 15°).",
        ],
        "en": [
            "Identify the 2nd metatarsal head, 1st metatarsal head, and hallux.",
            "Project these points onto the transverse plane (plantar view).",
            "The angle at the vertex (1st metatarsal head) measures hallux valgus (norm < 15°).",
        ],
    },
    "calcaneus_valgus": {
        "fr": [
            "Identifier la malléole médiale, la malléole latérale et le talon (heel).",
            "Calculer le centre bi-malléolaire et tracer le vecteur vers le talon.",
            "L'angle de ce vecteur avec la verticale indique le valgus calcanéen (norme 0–5°).",
        ],
        "en": [
            "Identify the medial malleolus, lateral malleolus, and heel.",
            "Compute the bimalleolar midpoint and draw the vector toward the heel.",
            "The angle of this vector from vertical indicates calcaneal valgus (norm 0–5°).",
        ],
    },
    "navicular_drop": {
        "fr": [
            "Identifier la tubérosité naviculaire (bord médial du pied) et la malléole médiale.",
            "Mesurer la hauteur verticale de chaque point.",
            "La chute naviculaire = hauteur malléole – hauteur naviculaire (norme 15–20 mm).",
        ],
        "en": [
            "Identify the navicular tuberosity (medial foot) and the medial malleolus.",
            "Measure the vertical height of each point.",
            "Navicular drop = malleolus height – navicular height (norm 15–20 mm).",
        ],
    },
    "tibial_torsion": {
        "fr": [
            "Identifier les condyles médiaux/latéraux du genou et les deux malléoles.",
            "Projeter l'axe inter-condylien et l'axe bi-malléolaire sur le plan transverse.",
            "L'angle entre ces deux axes représente la torsion tibiale (norme 15–30°).",
        ],
        "en": [
            "Identify the medial/lateral knee condyles and both malleoli.",
            "Project the intercondylar axis and bimalleolar axis onto the transverse plane.",
            "The angle between these two axes represents tibial torsion (norm 15–30°).",
        ],
    },
    "femoral_anteversion_clinical": {
        "fr": [
            "Identifier le grand trochanter et l'EIAS du même côté.",
            "Calculer la composante antéro-postérieure du vecteur GT→EIAS dans le plan horizontal.",
            "L'inclinaison AP relative (proxy clinique) estime l'antéversion fémorale.",
        ],
        "en": [
            "Identify the greater trochanter and ASIS on the same side.",
            "Compute the anteroposterior component of the GT→ASIS vector in the horizontal plane.",
            "The relative AP inclination (clinical proxy) estimates femoral anteversion.",
        ],
    },
    "shoulder_protraction": {
        "fr": [
            "Identifier les deux acromions et les repères thoraciques C7 (et T8 si disponible).",
            "Calculer la position AP moyenne des acromions relativement au plan thoracique C7–T8.",
            "Une valeur positive indique des épaules en protraction par rapport au thorax (norme ≈ 0 mm).",
        ],
        "en": [
            "Identify both acromions and the thoracic landmarks C7 (and T8 if available).",
            "Compute the mean AP position of both acromions relative to the C7–T8 thoracic midplane.",
            "A positive value indicates shoulders protracted relative to the thorax (norm ≈ 0 mm).",
        ],
    },
    "pelvic_obliquity": {
        "fr": [
            "Identifier l'EIAS gauche et l'EIAS droite.",
            "Mesurer leur hauteur respective sur l'axe vertical.",
            "La différence absolue est l'obliquité pelvienne (norme 0 mm).",
        ],
        "en": [
            "Identify the left and right ASIS landmarks.",
            "Measure their respective heights on the vertical axis.",
            "The absolute difference is the pelvic obliquity (norm 0 mm).",
        ],
    },
}


def get_all_measures() -> list[AnthroMeasure]:
    """Return the full list of 20 AnthroMeasure instances."""
    return [
        # ── Longueurs ──────────────────────────────────────────────────────────
        LowerLimbLength(
            code="lower_limb_length",
            name_fr="Longueur membre inférieur",
            name_en="Lower limb length",
            required_landmarks=["ASIS_right", "medial_malleolus_right"],
            formula_description_fr="Distance 3D EIAS → malléole médiale.",
            formula_description_en="3-D distance ASIS → medial malleolus.",
            unit="mm",
            norm_low=800.0,
            norm_high=1050.0,
            side="bilateral",
            category="longueur",
        ),
        UpperLimbLength(
            code="upper_limb_length",
            name_fr="Longueur membre supérieur",
            name_en="Upper limb length",
            required_landmarks=["acromion_right", "radial_styloid_right"],
            formula_description_fr="Distance 3D acromion → styloïde radiale.",
            formula_description_en="3-D distance acromion → radial styloid.",
            unit="mm",
            norm_low=550.0,
            norm_high=750.0,
            side="bilateral",
            category="longueur",
        ),
        # ── Posture pelvienne ──────────────────────────────────────────────────
        PelvicTilt(
            code="pelvic_tilt",
            name_fr="Bascule pelvienne",
            name_en="Pelvic tilt",
            required_landmarks=["ASIS_right", "PSIS_right"],
            formula_description_fr="Angle EIAS–EIPS par rapport à l'horizontale dans le plan sagittal.",
            formula_description_en="ASIS–PSIS angle relative to horizontal in the sagittal plane.",
            unit="°",
            norm_low=10.0,
            norm_high=12.0,
            side="bilateral",
            category="posture",
        ),
        PelvicObliquity(
            code="pelvic_obliquity",
            name_fr="Obliquité pelvienne",
            name_en="Pelvic obliquity",
            required_landmarks=["ASIS_left", "ASIS_right"],
            formula_description_fr="Différence de hauteur entre EIAS gauche et droite.",
            formula_description_en="Height difference between left and right ASIS.",
            unit="mm",
            norm_low=0.0,
            norm_high=10.0,
            side="bilateral",
            category="posture",
        ),
        # ── Membre inférieur ───────────────────────────────────────────────────
        QAngle(
            code="Q_angle",
            name_fr="Angle Q (patello-fémoral)",
            name_en="Q angle (patellofemoral)",
            required_landmarks=["ASIS_right", "patella_center_right", "tibial_tuberosity_right"],
            formula_description_fr="Angle EIAS–patella–tubérosité tibiale dans le plan frontal.",
            formula_description_en="ASIS–patella–tibial tuberosity angle in the frontal plane.",
            unit="°",
            norm_low=8.0,
            norm_high=15.0,
            side="bilateral",
            category="angle",
        ),
        KneeValgusVarus(
            code="knee_valgus_varus",
            name_fr="Valgus / varus du genou",
            name_en="Knee valgus / varus",
            required_landmarks=[
                "greater_trochanter_right",
                "lateral_knee_right",
                "lateral_malleolus_right",
            ],
            formula_description_fr=(
                "Déviation par rapport à l'alignement parfait des axes fémoral et tibial "
                "dans le plan frontal."
            ),
            formula_description_en=(
                "Deviation from perfect alignment of femoral and tibial axes in the frontal plane."
            ),
            unit="°",
            norm_low=0.0,
            norm_high=6.0,
            side="bilateral",
            category="angle",
        ),
        GenuRecurvatum(
            code="genu_recurvatum",
            name_fr="Genu recurvatum",
            name_en="Genu recurvatum",
            required_landmarks=[
                "greater_trochanter_right",
                "lateral_knee_right",
                "lateral_malleolus_right",
            ],
            formula_description_fr=(
                "Hyperextension du genou : déviation par rapport à 180° "
                "(GT–condyle lat.–malléole lat.) dans le plan sagittal."
            ),
            formula_description_en=(
                "Knee hyperextension: deviation from 180° "
                "(GT–lat. condyle–lat. malleolus) in the sagittal plane."
            ),
            unit="°",
            norm_low=0.0,
            norm_high=5.0,
            side="bilateral",
            category="angle",
        ),
        TibialTorsion(
            code="tibial_torsion",
            name_fr="Torsion tibiale",
            name_en="Tibial torsion",
            required_landmarks=[
                "medial_knee_right",
                "lateral_knee_right",
                "medial_malleolus_right",
                "lateral_malleolus_right",
            ],
            formula_description_fr=(
                "Angle entre l'axe inter-condylien et l'axe bi-malléolaire "
                "projeté sur le plan transverse."
            ),
            formula_description_en=(
                "Angle between condyle axis and bimalleolar axis projected onto the transverse plane."
            ),
            unit="°",
            norm_low=15.0,
            norm_high=30.0,
            side="bilateral",
            category="angle",
        ),
        FemoralAnteversionClinical(
            code="femoral_anteversion_clinical",
            name_fr="Antéversion fémorale (proxy clinique)",
            name_en="Femoral anteversion (clinical proxy)",
            required_landmarks=["greater_trochanter_right", "ASIS_right"],
            formula_description_fr=(
                "Inclinaison AP du vecteur GT→EIAS dans le plan horizontal "
                "(estimation clinique de surface, pas une vraie antéversion)."
            ),
            formula_description_en=(
                "AP inclination of the GT→ASIS vector in the horizontal plane "
                "(surface clinical estimate, not true anteversion)."
            ),
            unit="°",
            norm_low=None,
            norm_high=None,
            side="bilateral",
            category="angle",
        ),
        # ── Pied ──────────────────────────────────────────────────────────────
        HalluxValgusAngle(
            code="hallux_valgus_angle",
            name_fr="Angle valgus de l'hallux",
            name_en="Hallux valgus angle",
            required_landmarks=[
                "second_metatarsal_head_right",
                "first_metatarsal_head_right",
                "hallux_right",
            ],
            formula_description_fr=(
                "Angle 2e métatarse–1er métatarse–hallux dans le plan transverse. "
                "Fallback hallux → first_metatarsal_head."
            ),
            formula_description_en=(
                "2nd metatarsal–1st metatarsal–hallux angle in the transverse plane. "
                "Fallback hallux → first_metatarsal_head."
            ),
            unit="°",
            norm_low=0.0,
            norm_high=15.0,
            side="bilateral",
            category="angle",
        ),
        CalcaneusValgus(
            code="calcaneus_valgus",
            name_fr="Valgus calcanéen",
            name_en="Calcaneal valgus",
            required_landmarks=[
                "medial_malleolus_right",
                "lateral_malleolus_right",
                "heel_right",
            ],
            formula_description_fr=(
                "Angle du vecteur mi-malléolaire→talon avec la verticale. "
                "Estimation de la déviation latérale du calcanéum."
            ),
            formula_description_en=(
                "Angle of the midmalleolus→heel vector from vertical. "
                "Estimate of calcaneal lateral deviation."
            ),
            unit="°",
            norm_low=0.0,
            norm_high=5.0,
            side="bilateral",
            category="angle",
        ),
        NavicularDrop(
            code="navicular_drop",
            name_fr="Chute naviculaire",
            name_en="Navicular drop",
            required_landmarks=["navicular_tuberosity_right", "medial_malleolus_right"],
            formula_description_fr=(
                "Hauteur de la malléole médiale moins hauteur de la tubérosité naviculaire "
                "(proxy hauteur de voûte)."
            ),
            formula_description_en=(
                "Medial malleolus height minus navicular tuberosity height "
                "(arch height proxy)."
            ),
            unit="mm",
            norm_low=15.0,
            norm_high=20.0,
            side="bilateral",
            category="longueur",
        ),
        # ── Membre supérieur ───────────────────────────────────────────────────
        CarryingAngle(
            code="carrying_angle",
            name_fr="Angle de transport du coude",
            name_en="Carrying angle",
            required_landmarks=[
                "acromion_right",
                "lateral_epicondyle_right",
                "radial_styloid_right",
            ],
            formula_description_fr=(
                "Angle acromion–épicondyle latéral–styloïde radiale "
                "dans le plan frontal (cubitus valgus physiologique)."
            ),
            formula_description_en=(
                "Acromion–lateral epicondyle–radial styloid angle in the frontal plane "
                "(physiological cubitus valgus)."
            ),
            unit="°",
            norm_low=5.0,
            norm_high=15.0,
            side="bilateral",
            category="angle",
        ),
        ElbowHyperextension(
            code="elbow_hyperextension",
            name_fr="Hyperextension du coude",
            name_en="Elbow hyperextension",
            required_landmarks=["lateral_epicondyle_right", "radial_styloid_right"],
            formula_description_fr=(
                "Degrés d'hyperextension du coude dans le plan sagittal "
                "(olécrâne ou acromion comme référence proximale)."
            ),
            formula_description_en=(
                "Degrees of elbow hyperextension in the sagittal plane "
                "(olecranon or acromion as proximal reference)."
            ),
            unit="°",
            norm_low=0.0,
            norm_high=10.0,
            side="bilateral",
            category="angle",
        ),
        # ── Scapulo-thoracique ─────────────────────────────────────────────────
        ScapularHeightAsymmetry(
            code="scapular_height_asymmetry",
            name_fr="Asymétrie scapulaire (hauteur)",
            name_en="Scapular height asymmetry",
            required_landmarks=[
                "scapula_inferior_angle_left",
                "scapula_inferior_angle_right",
            ],
            formula_description_fr=(
                "Différence de hauteur entre les angles inférieurs scapulaires G et D."
            ),
            formula_description_en=(
                "Height difference between left and right inferior scapular angles."
            ),
            unit="mm",
            norm_low=0.0,
            norm_high=15.0,
            side="bilateral",
            category="asymetrie",
        ),
        ShoulderHeightAsymmetry(
            code="shoulder_height_asymmetry",
            name_fr="Asymétrie de hauteur des épaules",
            name_en="Shoulder height asymmetry",
            required_landmarks=["acromion_left", "acromion_right"],
            formula_description_fr="Différence de hauteur entre acromion gauche et droit.",
            formula_description_en="Height difference between left and right acromions.",
            unit="mm",
            norm_low=0.0,
            norm_high=10.0,
            side="bilateral",
            category="asymetrie",
        ),
        ShoulderProtraction(
            code="shoulder_protraction",
            name_fr="Protraction des épaules",
            name_en="Shoulder protraction",
            required_landmarks=["acromion_left", "acromion_right", "C7_spinous"],
            formula_description_fr=(
                "Position AP moyenne des deux acromions relativement au plan thoracique C7–T8."
            ),
            formula_description_en=(
                "Mean AP position of both acromions relative to the C7–T8 thoracic midplane."
            ),
            unit="mm",
            norm_low=-20.0,
            norm_high=20.0,
            side="bilateral",
            category="posture",
        ),
        # ── Cervical / crânio ──────────────────────────────────────────────────
        CraniovertebralAngle(
            code="craniovertebral_angle",
            name_fr="Angle cranio-vertébral",
            name_en="Craniovertebral angle",
            required_landmarks=["C7_spinous", "external_acoustic_meatus_right"],
            formula_description_fr=(
                "Angle de la ligne C7→tragus avec l'horizontale dans le plan sagittal. "
                "Fallback tragus → méat acoustique externe."
            ),
            formula_description_en=(
                "Angle of the C7→tragus line with the horizontal in the sagittal plane. "
                "Fallback tragus → external acoustic meatus."
            ),
            unit="°",
            norm_low=46.0,
            norm_high=65.0,
            side="bilateral",
            category="posture",
        ),
        CervicalLateralInclination(
            code="cervical_lateral_inclination",
            name_fr="Inclinaison latérale cervicale",
            name_en="Cervical lateral inclination",
            required_landmarks=["C7_spinous", "external_acoustic_meatus_right"],
            formula_description_fr=(
                "Déviation de la ligne tragus–C7 par rapport à la verticale "
                "dans le plan frontal."
            ),
            formula_description_en=(
                "Deviation of the tragus–C7 line from vertical in the frontal plane."
            ),
            unit="°",
            norm_low=0.0,
            norm_high=5.0,
            side="bilateral",
            category="posture",
        ),
        CervicalRotationAsymmetry(
            code="cervical_rotation_asymmetry",
            name_fr="Asymétrie de rotation cervicale",
            name_en="Cervical rotation asymmetry",
            required_landmarks=[
                "external_acoustic_meatus_left",
                "external_acoustic_meatus_right",
                "C7_spinous",
            ],
            formula_description_fr=(
                "Différence de position AP entre méat acoustique gauche et droit "
                "relativement à C7 (posture de rotation de tête)."
            ),
            formula_description_en=(
                "AP position difference between left and right acoustic meatuses "
                "relative to C7 (head rotation posture)."
            ),
            unit="mm",
            norm_low=0.0,
            norm_high=15.0,
            side="bilateral",
            category="asymetrie",
        ),
    ]


def compute_available(ground_truth: dict[str, np.ndarray]) -> list[AnthroMeasureResult]:
    """Compute all measures against *ground_truth* and return a result for each.

    Measures whose landmarks are absent return a result with ``value=None``.
    """
    results: list[AnthroMeasureResult] = []
    for measure in get_all_measures():
        try:
            value = measure.compute(ground_truth)
        except Exception:
            value = None
        results.append(AnthroMeasureResult(measure=measure, value=value))
    return results


def measure_step_guide(measure_code: str, lang: str = "fr") -> list[str]:
    """Return 2–3 guided steps for performing *measure_code*.

    Parameters
    ----------
    measure_code:
        The ``code`` attribute of any AnthroMeasure (e.g. ``"Q_angle"``).
    lang:
        ``"fr"`` (default) or ``"en"``.

    Returns
    -------
    list[str]
        Steps as plain strings; empty list when code is unknown.
    """
    entry = _STEP_GUIDE.get(measure_code, {})
    return list(entry.get(lang, entry.get("fr", [])))
