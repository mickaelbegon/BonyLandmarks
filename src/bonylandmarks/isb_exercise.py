"""Exercice de construction des repères locaux ISB (Wu et al. 2005).

Ce module est autonome : il ne dépend que de ``numpy`` et ne touche à aucun
autre fichier du projet.  Il fournit :

* :func:`compute_isb_lcs`          — les 7 repères locaux à partir du ground truth
* :func:`get_all_segment_definitions` — métadonnées (FR/EN) de chaque segment
* :func:`segment_step_guide`       — mode guidé, étape par étape
* :func:`compute_deviations`       — comparaison angulaire étudiant / référence

Convention ISB (position anatomique) : X antérieur, Y supérieur (proximal pour
les membres), Z vers la droite du sujet.  Les repères produits sont toujours
orthonormés directs (X × Y = Z).

Références
----------
- Wu G et al. (2005) ISB recommendation on definitions of joint coordinate
  systems — Part II: shoulder, elbow, wrist and hand. J Biomech 38(5):981-992.
- Wu G et al. (2002) ISB recommendation — Part I: ankle, hip, and spine.
  J Biomech 35(4):543-548.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SegmentLCS",
    "ISBExerciseResult",
    "AxisDeviation",
    "SEGMENT_KEYS",
    "compute_isb_lcs",
    "get_all_segment_definitions",
    "segment_step_guide",
    "compute_deviations",
    "angle_between",
]

_EPS = 1e-9

# Ordre canonique d'affichage des segments.
SEGMENT_KEYS: list[str] = [
    "thorax",
    "clavicle_right",
    "scapula_right",
    "humerus_right",
    "pelvis",
    "femur_right",
    "tibia_right",
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SegmentLCS:
    """Repère local ISB d'un segment (axes unitaires, repère direct)."""

    name: str                       # "Thorax", "Clavicule D", …
    name_en: str                    # "Thorax", "Right clavicle", …
    origin: np.ndarray              # shape (3,)
    x: np.ndarray                   # axe X unitaire
    y: np.ndarray                   # axe Y unitaire
    z: np.ndarray                   # axe Z unitaire
    required_landmarks: list[str]   # codes des landmarks nécessaires
    present: bool = True            # False si landmarks manquants / dégénérés
    key: str = ""                   # clé machine ("thorax", "clavicle_right", …)
    missing_landmarks: list[str] = field(default_factory=list)
    note: str = ""                  # raison d'un échec de construction

    @property
    def matrix(self) -> np.ndarray:
        """Matrice de rotation 3×3 dont les colonnes sont X, Y, Z."""
        return np.column_stack((self.x, self.y, self.z))

    @property
    def is_right_handed(self) -> bool:
        """True si le trièdre est direct (déterminant > 0)."""
        return bool(np.linalg.det(self.matrix) > 0.0)

    def axis(self, name: str) -> np.ndarray:
        """Retourne l'axe nommé 'X', 'Y' ou 'Z' (insensible à la casse)."""
        key = name.strip().lower()
        if key not in ("x", "y", "z"):
            raise ValueError(f"axe inconnu: {name!r}")
        return getattr(self, key)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Représentation compacte {'origin', 'x', 'y', 'z'}."""
        return {"origin": self.origin, "x": self.x, "y": self.y, "z": self.z}


@dataclass
class ISBExerciseResult:
    """Ensemble des repères ISB calculés pour un jeu de landmarks."""

    segments: list[SegmentLCS]
    missing_landmarks: list[str] = field(default_factory=list)

    @property
    def available(self) -> list[SegmentLCS]:
        """Sous-liste des segments effectivement construits."""
        return [s for s in self.segments if s.present]

    def get(self, key_or_name: str) -> SegmentLCS | None:
        """Retrouve un segment par sa clé, son nom FR ou son nom EN."""
        needle = key_or_name.strip().lower()
        for seg in self.segments:
            if needle in (seg.key.lower(), seg.name.lower(), seg.name_en.lower()):
                return seg
        return None


@dataclass
class AxisDeviation:
    """Écart angulaire d'un axe entre deux constructions de repère."""

    segment_name: str
    axis: str                # "X", "Y", "Z"
    angle_deg: float         # angle entre l'axe calculé et l'axe de référence


# ── Helpers géométriques ──────────────────────────────────────────────────────

def _get(gt: dict[str, np.ndarray], *codes: str) -> tuple[np.ndarray, ...] | None:
    """Retourne les positions des *codes*, ou None si l'un d'eux est absent."""
    out: list[np.ndarray] = []
    for code in codes:
        if code not in gt or gt[code] is None:
            return None
        out.append(np.asarray(gt[code], dtype=np.float64).reshape(3))
    return tuple(out)


def _unit(v: np.ndarray) -> np.ndarray | None:
    """Normalise *v*, ou None si le vecteur est quasi nul."""
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < _EPS:
        return None
    return vec / norm


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Milieu de deux points."""
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def _perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray | None:
    """Composante de *v* perpendiculaire à *axis* (unitaire), normalisée."""
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    return _unit(vec - float(np.dot(vec, axis)) * axis)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle non signé en degrés entre deux vecteurs (0-180)."""
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(np.dot(ua, ub)), -1.0, 1.0))))


# Paires bilatérales utilisées pour retrouver la direction « vers la droite ».
_RIGHT_PAIRS: tuple[tuple[str, str], ...] = (
    ("ASIS_right", "ASIS_left"),
    ("PSIS_right", "PSIS_left"),
    ("acromial_angle_right", "acromial_angle_left"),
    ("acromion_right", "acromion_left"),
    ("sternoclavicular_joint_right", "sternoclavicular_joint_left"),
    ("external_acoustic_meatus_right", "external_acoustic_meatus_left"),
    ("greater_trochanter_right", "greater_trochanter_left"),
    ("lateral_epicondyle_right", "lateral_epicondyle_left"),
    ("lateral_malleolus_right", "lateral_malleolus_left"),
)

# Paires (postérieur, antérieur) utilisées pour retrouver la direction avant.
_ANTERIOR_PAIRS: tuple[tuple[str, str], ...] = (
    ("C7_spinous", "suprasternal_notch"),
    ("T8_spinous", "xiphoid_process"),
    ("external_occipital_protuberance", "glabella"),
)


def _right_ref(gt: dict[str, np.ndarray]) -> np.ndarray | None:
    """Direction approximative « vers la droite du sujet », si déductible.

    Best-effort : sert uniquement à lever l'ambiguïté de signe, jamais à
    construire un axe.  Retourne None si aucune paire bilatérale n'est connue.
    """
    for right, left in _RIGHT_PAIRS:
        pts = _get(gt, right, left)
        if pts is None:
            continue
        vec = _unit(pts[0] - pts[1])
        if vec is not None:
            return vec
    return None


def _anterior_ref(gt: dict[str, np.ndarray]) -> np.ndarray | None:
    """Direction approximative « vers l'avant du sujet », si déductible."""
    asis = _get(gt, "ASIS_left", "ASIS_right")
    psis = _get(gt, "PSIS_left", "PSIS_right")
    if asis is not None and psis is not None:
        vec = _unit(_mid(*asis) - _mid(*psis))
        if vec is not None:
            return vec
    for back, front in _ANTERIOR_PAIRS:
        pts = _get(gt, back, front)
        if pts is None:
            continue
        vec = _unit(pts[1] - pts[0])
        if vec is not None:
            return vec
    return None


def _orient(axis: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """Retourne *axis*, retourné si besoin pour pointer comme *reference*."""
    if reference is None:
        return axis
    return -axis if float(np.dot(axis, reference)) < 0.0 else axis


# ── Métadonnées des segments ──────────────────────────────────────────────────

_SEGMENT_META: dict[str, dict] = {
    "thorax": {
        "name_fr": "Thorax",
        "name_en": "Thorax",
        "required": [
            "suprasternal_notch", "xiphoid_process", "C7_spinous",
        ],
        "method_fr": (
            "Origine à l'incisure jugulaire (IJ). Y = (milieu C7-IJ − PX) vers le "
            "haut ; Z normal au plan (IJ, C7, PX) orienté vers la droite ; X = Y × Z."
        ),
        "method_en": (
            "Origin at the suprasternal notch (IJ). Y = (midpoint C7-IJ − PX) "
            "upwards; Z normal to the (IJ, C7, PX) plane pointing right; X = Y × Z."
        ),
    },
    "clavicle_right": {
        "name_fr": "Clavicule D",
        "name_en": "Right clavicle",
        "required": [
            "sternoclavicular_joint_right", "acromioclavicular_joint_right",
            "suprasternal_notch", "xiphoid_process", "C7_spinous",
        ],
        "method_fr": (
            "Origine à l'articulation sterno-claviculaire (SC). Z = SC → AC ; "
            "X = Y_thorax × Z (vers l'avant) ; Y = Z × X (vers le haut)."
        ),
        "method_en": (
            "Origin at the sternoclavicular joint (SC). Z = SC → AC; "
            "X = Y_thorax × Z (pointing anteriorly); Y = Z × X (pointing superiorly)."
        ),
    },
    "scapula_right": {
        "name_fr": "Scapula D",
        "name_en": "Right scapula",
        "required": [
            "acromial_angle_right", "scapula_trigonum_right",
            "scapula_inferior_angle_right",
        ],
        "method_fr": (
            "Origine à l'angle acromial (AA). Z = TS → AA ; X normal au plan "
            "(AA, TS, AI) orienté vers l'avant ; Y = Z × X."
        ),
        "method_en": (
            "Origin at the acromial angle (AA). Z = TS → AA; X normal to the "
            "(AA, TS, AI) plane pointing anteriorly; Y = Z × X."
        ),
    },
    "humerus_right": {
        "name_fr": "Humérus D",
        "name_en": "Right humerus",
        "required": [
            "lateral_epicondyle_right", "medial_epicondyle_right",
            "greater_tubercle_right",
        ],
        "method_fr": (
            "Origine au milieu des épicondyles (centre articulaire du coude). "
            "Y = origine → tubercule majeur (proxy du centre gléno-huméral) ; "
            "Z = composante de (EL − EM) perpendiculaire à Y (vers la droite) ; X = Y × Z."
        ),
        "method_en": (
            "Origin at the midpoint of the epicondyles (elbow joint centre). "
            "Y = origin → greater tubercle (GH centre proxy); Z = component of "
            "(EL − EM) perpendicular to Y (pointing laterally); X = Y × Z."
        ),
    },
    "pelvis": {
        "name_fr": "Bassin",
        "name_en": "Pelvis",
        "required": ["ASIS_left", "ASIS_right", "PSIS_left", "PSIS_right"],
        "method_fr": (
            "Origine au milieu des EIAS. Z = milieu EIAS → EIAS droite ; "
            "X_raw = milieu EIPS → milieu EIAS (direction antérieure) ; Y = Z × X_raw (vers le haut) ; X = Y × Z."
        ),
        "method_en": (
            "Origin at the ASIS midpoint. Z = ASIS midpoint → right ASIS; "
            "X_raw = PSIS midpoint → ASIS midpoint (anterior direction); Y = Z × X_raw (pointing superiorly); X = Y × Z."
        ),
    },
    "femur_right": {
        "name_fr": "Fémur D",
        "name_en": "Right femur",
        "required": [
            "lateral_knee_right", "medial_knee_right", "greater_trochanter_right",
        ],
        "method_fr": (
            "Origine au milieu des condyles fémoraux. Y = origine → grand "
            "trochanter ; Z = composante de (condyle médial → condyle latéral) "
            "perpendiculaire à Y ; X = Y × Z."
        ),
        "method_en": (
            "Origin at the femoral condyle midpoint. Y = origin → greater "
            "trochanter; Z = component of (medial → lateral condyle) "
            "perpendicular to Y; X = Y × Z."
        ),
    },
    "tibia_right": {
        "name_fr": "Tibia D",
        "name_en": "Right tibia",
        "required": [
            "lateral_malleolus_right", "medial_malleolus_right",
            "lateral_knee_right", "medial_knee_right",
        ],
        "method_fr": (
            "Origine au milieu des malléoles. Y = origine → milieu des condyles "
            "fémoraux ; Z = composante de (malléole médiale → malléole latérale) "
            "perpendiculaire à Y ; X = Y × Z."
        ),
        "method_en": (
            "Origin at the malleoli midpoint. Y = origin → femoral condyle "
            "midpoint; Z = component of (medial → lateral malleolus) "
            "perpendicular to Y; X = Y × Z."
        ),
    },
}


# ── Constructeurs de repères ──────────────────────────────────────────────────
# Chaque constructeur retourne (origin, x, y, z) ou None si la construction est
# impossible (landmarks colinéaires / confondus).

def _build_thorax(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère thoracique ISB : origine IJ, Y vertical, Z vers la droite."""
    pts = _get(gt, "suprasternal_notch", "C7_spinous", "xiphoid_process")
    if pts is None:
        return None
    ij, c7, px = pts
    y = _unit(_mid(c7, ij) - px)
    if y is None:
        return None
    z = _unit(np.cross(ij - px, c7 - px))
    if z is None:
        return None
    z = _orient(z, _right_ref(gt))
    z = _perp(z, y)          # garantit l'orthogonalité exacte
    if z is None:
        return None
    x = _unit(np.cross(y, z))
    if x is None:
        return None
    return ij, x, y, z


def _build_clavicle_right(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère claviculaire droit : Z le long de SC → AC, X via le Y thoracique."""
    pts = _get(gt, "sternoclavicular_joint_right", "acromioclavicular_joint_right")
    if pts is None:
        return None
    sc, ac = pts
    thorax = _build_thorax(gt)
    if thorax is None:
        return None
    y_thorax = thorax[2]
    z = _unit(ac - sc)
    if z is None:
        return None
    x = _unit(np.cross(y_thorax, z))   # haut × latéral = antérieur ✓ (ISB)
    if x is None:                       # clavicule colinéaire au Y thoracique
        return None
    y = _unit(np.cross(z, x))
    if y is None:
        return None
    return sc, x, y, z


def _build_scapula_right(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère scapulaire droit : origine AA, Z = TS → AA, X antérieur."""
    pts = _get(
        gt, "acromial_angle_right", "scapula_trigonum_right",
        "scapula_inferior_angle_right",
    )
    if pts is None:
        return None
    aa, ts, ai = pts
    z = _unit(aa - ts)
    if z is None:
        return None
    x = _unit(np.cross(ts - aa, ai - aa))
    if x is None:
        return None
    x = _orient(x, _anterior_ref(gt))
    x = _perp(x, z)
    if x is None:
        return None
    y = _unit(np.cross(z, x))
    if y is None:
        return None
    return aa, x, y, z


def _build_humerus_right(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère huméral droit : origine au coude, Y vers le tubercule majeur."""
    pts = _get(
        gt, "lateral_epicondyle_right", "medial_epicondyle_right",
        "greater_tubercle_right",
    )
    if pts is None:
        return None
    el, em, gt_pt = pts
    origin = _mid(el, em)
    y = _unit(gt_pt - origin)
    if y is None:
        return None
    z = _perp(el - em, y)   # EL − EM pointe latéralement (convention ISB)
    if z is None:
        return None
    x = _unit(np.cross(y, z))
    if x is None:
        return None
    return origin, x, y, z


def _build_pelvis(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère pelvien : origine au milieu des EIAS, Z vers l'EIAS droite."""
    pts = _get(gt, "ASIS_left", "ASIS_right", "PSIS_left", "PSIS_right")
    if pts is None:
        return None
    asis_l, asis_r, psis_l, psis_r = pts
    origin = _mid(asis_l, asis_r)
    z = _unit(asis_r - origin)
    if z is None:
        return None
    x_raw = _unit(origin - _mid(psis_l, psis_r))   # EIPS → EIAS = antérieur
    if x_raw is None:
        return None
    y = _unit(np.cross(z, x_raw))   # droite × avant = haut ✓
    if y is None:
        return None
    z = _perp(z, y)
    if z is None:
        return None
    x = _unit(np.cross(y, z))
    if x is None:
        return None
    return origin, x, y, z


def _build_femur_right(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère fémoral droit : origine au genou, Y vers le grand trochanter."""
    pts = _get(
        gt, "lateral_knee_right", "medial_knee_right", "greater_trochanter_right",
    )
    if pts is None:
        return None
    lat_knee, med_knee, troch = pts
    origin = _mid(lat_knee, med_knee)
    y = _unit(troch - origin)
    if y is None:
        return None
    z = _perp(lat_knee - med_knee, y)
    if z is None:
        return None
    x = _unit(np.cross(y, z))
    if x is None:
        return None
    return origin, x, y, z


def _build_tibia_right(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère tibial droit : origine à la cheville, Y vers le genou."""
    pts = _get(
        gt, "lateral_malleolus_right", "medial_malleolus_right",
        "lateral_knee_right", "medial_knee_right",
    )
    if pts is None:
        return None
    lat_mal, med_mal, lat_knee, med_knee = pts
    origin = _mid(lat_mal, med_mal)
    y = _unit(_mid(lat_knee, med_knee) - origin)
    if y is None:
        return None
    z = _perp(lat_mal - med_mal, y)
    if z is None:
        return None
    x = _unit(np.cross(y, z))
    if x is None:
        return None
    return origin, x, y, z


_BUILDERS = {
    "thorax": _build_thorax,
    "clavicle_right": _build_clavicle_right,
    "scapula_right": _build_scapula_right,
    "humerus_right": _build_humerus_right,
    "pelvis": _build_pelvis,
    "femur_right": _build_femur_right,
    "tibia_right": _build_tibia_right,
}


# ── API publique ──────────────────────────────────────────────────────────────

def compute_isb_lcs(ground_truth: dict[str, np.ndarray]) -> ISBExerciseResult:
    """Calcule les 7 repères locaux ISB disponibles à partir de ground_truth."""
    gt = ground_truth or {}
    segments: list[SegmentLCS] = []
    missing_all: list[str] = []

    for key in SEGMENT_KEYS:
        meta = _SEGMENT_META[key]
        required: list[str] = list(meta["required"])
        missing = [code for code in required if code not in gt or gt[code] is None]
        for code in missing:
            if code not in missing_all:
                missing_all.append(code)

        frame = None if missing else _BUILDERS[key](gt)
        if frame is None:
            note = (
                "landmarks manquants" if missing
                else "landmarks confondus ou colinéaires"
            )
            segments.append(SegmentLCS(
                name=meta["name_fr"], name_en=meta["name_en"],
                origin=np.zeros(3), x=np.zeros(3), y=np.zeros(3), z=np.zeros(3),
                required_landmarks=required, present=False, key=key,
                missing_landmarks=missing, note=note,
            ))
            continue

        origin, x, y, z = frame
        segments.append(SegmentLCS(
            name=meta["name_fr"], name_en=meta["name_en"],
            origin=origin, x=x, y=y, z=z,
            required_landmarks=required, present=True, key=key,
        ))

    return ISBExerciseResult(segments=segments, missing_landmarks=missing_all)


def get_all_segment_definitions() -> list[dict]:
    """Retourne, pour chaque segment ISB, ses noms FR/EN, landmarks et méthode."""
    return [
        {
            "key": key,
            "name_fr": _SEGMENT_META[key]["name_fr"],
            "name_en": _SEGMENT_META[key]["name_en"],
            "required_landmarks": list(_SEGMENT_META[key]["required"]),
            "method_fr": _SEGMENT_META[key]["method_fr"],
            "method_en": _SEGMENT_META[key]["method_en"],
        }
        for key in SEGMENT_KEYS
    ]


# ── Mode guidé ────────────────────────────────────────────────────────────────

_STEPS_FR: dict[str, list[str]] = {
    "thorax": [
        "Étape 1 : placer l'origine sur l'incisure jugulaire (IJ), au creux "
        "supérieur du sternum.",
        "Étape 2 : construire l'axe Y en joignant le processus xiphoïde (PX) au "
        "milieu du segment C7–IJ ; il pointe vers le haut.",
        "Étape 3 : construire l'axe Z perpendiculaire au plan formé par IJ, C7 et "
        "PX, orienté vers la droite du sujet.",
        "Étape 4 : obtenir l'axe X par le produit vectoriel Y × Z ; il pointe vers "
        "l'avant, perpendiculaire au plan sagittal.",
    ],
    "clavicle_right": [
        "Étape 1 : placer l'origine sur l'articulation sterno-claviculaire droite "
        "(SC).",
        "Étape 2 : construire l'axe Z de SC vers l'articulation acromio-claviculaire "
        "(AC) : c'est le grand axe de la clavicule, pointant latéralement.",
        "Étape 3 : construire l'axe X par Y_thorax × Z ; comme Y_thorax pointe vers "
        "le haut et Z vers le côté, leur produit vectoriel pointe vers l'avant.",
        "Étape 4 : obtenir l'axe Y par le produit vectoriel Z × X ; il pointe vers "
        "le haut, dans le plan sagittal.",
    ],
    "scapula_right": [
        "Étape 1 : placer l'origine sur l'angle acromial (AA), sommet postéro-latéral "
        "de l'acromion.",
        "Étape 2 : construire l'axe Z du trigone de l'épine (TS) vers AA ; il suit "
        "l'épine scapulaire et pointe latéralement.",
        "Étape 3 : construire l'axe X perpendiculaire au plan de la scapula (AA, TS, "
        "AI), orienté vers l'avant.",
        "Étape 4 : obtenir l'axe Y par le produit vectoriel Z × X ; il pointe vers le "
        "haut, parallèle au bord médial.",
    ],
    "humerus_right": [
        "Étape 1 : placer l'origine au milieu de l'épicondyle latéral (EL) et de "
        "l'épicondyle médial (EM) : centre articulaire estimé du coude.",
        "Étape 2 : construire l'axe Y de cette origine vers le tubercule majeur "
        "(proxy du centre gléno-huméral) ; il pointe proximalement.",
        "Étape 3 : construire l'axe Z à partir de la ligne EL − EM (épicondyle "
        "latéral − médial), en gardant la composante perpendiculaire à Y ; "
        "il pointe latéralement.",
        "Étape 4 : obtenir l'axe X par le produit vectoriel Y × Z ; il pointe "
        "vers l'avant.",
    ],
    "pelvis": [
        "Étape 1 : placer l'origine au milieu des deux épines iliaques antéro-"
        "supérieures (EIAS gauche et droite).",
        "Étape 2 : construire l'axe Z de cette origine vers l'EIAS droite ; il "
        "pointe vers la droite le long de la ligne bi-EIAS.",
        "Étape 3 : construire la direction auxiliaire du milieu des EIPS vers le "
        "milieu des EIAS (direction antérieure), puis obtenir Y = Z × direction_ant "
        "; Y pointe vers le haut.",
        "Étape 4 : obtenir l'axe X par Y × Z ; il pointe vers l'avant.",
    ],
    "femur_right": [
        "Étape 1 : placer l'origine au milieu des condyles fémoraux latéral et "
        "médial : centre articulaire estimé du genou.",
        "Étape 2 : construire l'axe Y de cette origine vers le grand trochanter ; il "
        "pointe proximalement le long de la diaphyse.",
        "Étape 3 : construire l'axe Z à partir de la ligne condyle médial → condyle "
        "latéral, en n'en gardant que la composante perpendiculaire à Y.",
        "Étape 4 : obtenir l'axe X par le produit vectoriel Y × Z ; il pointe vers "
        "l'avant.",
    ],
    "tibia_right": [
        "Étape 1 : placer l'origine au milieu des malléoles latérale et médiale : "
        "centre articulaire estimé de la cheville.",
        "Étape 2 : construire l'axe Y de cette origine vers le milieu des condyles "
        "fémoraux ; il pointe proximalement.",
        "Étape 3 : construire l'axe Z à partir de la ligne malléole médiale → "
        "malléole latérale, en n'en gardant que la composante perpendiculaire à Y.",
        "Étape 4 : obtenir l'axe X par le produit vectoriel Y × Z ; il pointe vers "
        "l'avant.",
    ],
}

_STEPS_EN: dict[str, list[str]] = {
    "thorax": [
        "Step 1: place the origin on the suprasternal notch (IJ), the upper hollow "
        "of the sternum.",
        "Step 2: build the Y axis from the xiphoid process (PX) to the midpoint of "
        "the C7–IJ segment; it points upwards.",
        "Step 3: build the Z axis perpendicular to the plane formed by IJ, C7 and "
        "PX, pointing to the subject's right.",
        "Step 4: obtain the X axis as the cross product Y × Z; it points forward, "
        "normal to the sagittal plane.",
    ],
    "clavicle_right": [
        "Step 1: place the origin on the right sternoclavicular joint (SC).",
        "Step 2: build the Z axis from SC to the acromioclavicular joint (AC): "
        "the long axis of the clavicle, pointing laterally.",
        "Step 3: build the X axis as Y_thorax × Z; since Y_thorax points upward "
        "and Z points laterally, their cross product points anteriorly.",
        "Step 4: obtain the Y axis as Z × X; it points superiorly.",
    ],
    "scapula_right": [
        "Step 1: place the origin on the acromial angle (AA), the posterolateral "
        "tip of the acromion.",
        "Step 2: build the Z axis from the trigonum spinae (TS) to AA; it follows "
        "the scapular spine and points laterally.",
        "Step 3: build the X axis perpendicular to the scapular plane (AA, TS, AI), "
        "pointing anteriorly.",
        "Step 4: obtain the Y axis as the cross product Z × X; it points upwards, "
        "parallel to the medial border.",
    ],
    "humerus_right": [
        "Step 1: place the origin midway between the lateral (EL) and medial (EM) "
        "epicondyles: the estimated elbow joint centre.",
        "Step 2: build the Y axis from that origin to the greater tubercle (GH "
        "centre proxy); it points proximally.",
        "Step 3: build the Z axis from the inter-epicondylar line EL − EM, keeping "
        "only its component perpendicular to Y; it points laterally.",
        "Step 4: obtain the X axis as Y × Z; it points anteriorly.",
    ],
    "pelvis": [
        "Step 1: place the origin midway between the left and right anterior "
        "superior iliac spines (ASIS).",
        "Step 2: build the Z axis from that origin to the right ASIS; it points "
        "rightward along the inter-ASIS line.",
        "Step 3: build the auxiliary direction from the PSIS midpoint to the ASIS "
        "midpoint (anterior direction), then obtain Y = Z × that direction; "
        "Y points superiorly.",
        "Step 4: obtain the X axis as Y × Z; it points anteriorly.",
    ],
    "femur_right": [
        "Step 1: place the origin midway between the lateral and medial femoral "
        "condyles: the estimated knee joint centre.",
        "Step 2: build the Y axis from that origin to the greater trochanter; it "
        "points proximally along the shaft.",
        "Step 3: build the Z axis from the medial → lateral condyle line, keeping "
        "only its component perpendicular to Y.",
        "Step 4: obtain the X axis as the cross product Y × Z; it points forward.",
    ],
    "tibia_right": [
        "Step 1: place the origin midway between the lateral and medial malleoli: "
        "the estimated ankle joint centre.",
        "Step 2: build the Y axis from that origin to the femoral condyle midpoint; "
        "it points proximally.",
        "Step 3: build the Z axis from the medial → lateral malleolus line, keeping "
        "only its component perpendicular to Y.",
        "Step 4: obtain the X axis as the cross product Y × Z; it points forward.",
    ],
}


def _resolve_key(segment_name: str) -> str | None:
    """Retrouve la clé machine d'un segment depuis sa clé ou son nom FR/EN."""
    needle = (segment_name or "").strip().lower()
    if needle in _SEGMENT_META:
        return needle
    for key, meta in _SEGMENT_META.items():
        if needle in (meta["name_fr"].lower(), meta["name_en"].lower()):
            return key
    return None


def segment_step_guide(segment_name: str, lang: str = "fr") -> list[str]:
    """Retourne les étapes guidées pour construire le repère d'un segment.

    *segment_name* accepte la clé ("scapula_right") ou le nom FR/EN
    ("Scapula D", "Right scapula").  Retourne une liste vide si inconnu.
    """
    key = _resolve_key(segment_name)
    if key is None:
        return []
    steps = _STEPS_EN if str(lang).lower().startswith("en") else _STEPS_FR
    return list(steps[key])


# ── Déviations ────────────────────────────────────────────────────────────────

def compute_deviations(
    result1: ISBExerciseResult, result2: ISBExerciseResult
) -> list[AxisDeviation]:
    """Compare deux ISBExerciseResult (ex: student vs ground truth).

    Un angle n'est produit que pour les segments construits des deux côtés.
    """
    deviations: list[AxisDeviation] = []
    by_key = {seg.key or seg.name: seg for seg in result2.segments}

    for seg1 in result1.segments:
        seg2 = by_key.get(seg1.key or seg1.name)
        if seg2 is None:
            seg2 = result2.get(seg1.name)
        if seg2 is None or not seg1.present or not seg2.present:
            continue
        for axis in ("X", "Y", "Z"):
            deviations.append(AxisDeviation(
                segment_name=seg1.name,
                axis=axis,
                angle_deg=angle_between(seg1.axis(axis), seg2.axis(axis)),
            ))
    return deviations
