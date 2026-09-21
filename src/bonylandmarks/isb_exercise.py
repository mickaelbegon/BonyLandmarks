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

Proxies palpables
-----------------
Deux points définis par l'ISB ne sont pas palpables et sont approximés ici par
le landmark palpable le plus proche.  Les axes restent conformes à l'ISB, seule
la position du point de référence est approchée :

* GH (centre gléno-huméral, Wu 2005) → ``greater_tubercle_right``.  L'ISB
  recommande une régression (Meskers et al. 1998) ou le point pivot des axes
  hélicoïdaux instantanés (Stokdijk et al. 2000), aucune palpation.
* Centre articulaire de hanche (Wu 2002) → ``greater_trochanter_right`` pour le
  fémur, milieu des EIAS pour le bassin.  L'ISB recommande une méthode
  fonctionnelle ou une régression.

De même, l'ISB définit le point inter-condylaire IC du tibia à partir des bords
des condyles *tibiaux* (MC/LC) ; le projet ne dispose que des épicondyles
fémoraux (``lateral_knee_right`` / ``medial_knee_right``), utilisés comme proxy.

Références
----------
- Wu G, Cavanagh PR (1995) ISB recommendations for standardization in the
  reporting of kinematic data. J Biomech 28(10):1257-1261.
- Wu G et al. (2002) ISB recommendation — Part I: ankle, hip, and spine.
  J Biomech 35(4):543-548.
- Wu G et al. (2005) ISB recommendation on definitions of joint coordinate
  systems — Part II: shoulder, elbow, wrist and hand. J Biomech 38(5):981-992.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SegmentLCS",
    "ISBExerciseResult",
    "AxisDeviation",
    "JointCenterResult",
    "SEGMENT_KEYS",
    "compute_isb_lcs",
    "get_all_segment_definitions",
    "segment_step_guide",
    "compute_deviations",
    "angle_between",
    "estimate_hjc",
    "estimate_ghc",
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
            "suprasternal_notch", "xiphoid_process", "C7_spinous", "T8_spinous",
        ],
        "method_fr": (
            "Origine à l'incisure jugulaire (IJ). Y = milieu(PX, T8) → milieu(IJ, C7), "
            "vers le haut ; Z normal au plan (IJ, C7, milieu(PX, T8)) orienté vers la "
            "droite ; X = Y × Z (vers l'avant)."
        ),
        "method_en": (
            "Origin at the suprasternal notch (IJ). Y = midpoint(PX, T8) → "
            "midpoint(IJ, C7), pointing upward; Z normal to the (IJ, C7, "
            "midpoint(PX, T8)) plane pointing right; X = Y × Z (forward)."
        ),
    },
    "clavicle_right": {
        "name_fr": "Clavicule D",
        "name_en": "Right clavicle",
        "required": [
            "sternoclavicular_joint_right", "acromioclavicular_joint_right",
            "suprasternal_notch", "xiphoid_process", "C7_spinous", "T8_spinous",
        ],
        "method_fr": (
            "Origine à l'articulation sterno-claviculaire (SC). Z = SC → AC ; "
            "X = Y_thorax × Z (vers l'avant) ; Y = Z × X (vers le haut). X est "
            "défini via le Y du thorax car seuls deux repères osseux (SC, AC) sont "
            "palpables : la rotation axiale de la clavicule reste indéterminée."
        ),
        "method_en": (
            "Origin at the sternoclavicular joint (SC). Z = SC → AC; "
            "X = Y_thorax × Z (pointing anteriorly); Y = Z × X (pointing superiorly). "
            "X is defined from the thorax Y axis because only two bony landmarks "
            "(SC, AC) are palpable: clavicular axial rotation stays indeterminate."
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
            "Origine à l'angle acromial (AA). Z = TS → AA ; X = (AI − AA) × (TS − AA), "
            "normal au plan (AI, AA, TS) orienté vers l'avant ; Y = Z × X (vers le "
            "haut). L'ISB utilise AA plutôt que AC : ce plan n'est donc pas le plan "
            "visuel de la scapula."
        ),
        "method_en": (
            "Origin at the acromial angle (AA). Z = TS → AA; X = (AI − AA) × (TS − AA), "
            "normal to the (AI, AA, TS) plane pointing anteriorly; Y = Z × X (upward). "
            "ISB uses AA rather than AC, so this plane is not the visual plane of the "
            "scapular bone."
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
            "Option 1 de l'ISB. Origine au centre gléno-huméral GH (proxy palpable : "
            "tubercule majeur). Y = milieu(EL, EM) → GH, vers le proximal ; "
            "X = Y × (EL − EM), normal au plan (EL, EM, GH) vers l'avant ; "
            "Z = X × Y (vers la droite)."
        ),
        "method_en": (
            "ISB option 1. Origin at the glenohumeral centre GH (palpable proxy: "
            "greater tubercle). Y = midpoint(EL, EM) → GH, pointing proximally; "
            "X = Y × (EL − EM), normal to the (EL, EM, GH) plane pointing forward; "
            "Z = X × Y (pointing right)."
        ),
    },
    "pelvis": {
        "name_fr": "Bassin",
        "name_en": "Pelvis",
        "required": ["ASIS_left", "ASIS_right", "PSIS_left", "PSIS_right"],
        "method_fr": (
            "Z = EIAS gauche → EIAS droite ; X = composante de (milieu EIPS → milieu "
            "EIAS) orthogonale à Z, vers l'avant ; Y = Z × X (vers le haut). L'ISB "
            "place l'origine au centre articulaire de hanche, non palpable : on "
            "utilise ici le milieu des EIAS."
        ),
        "method_en": (
            "Z = left ASIS → right ASIS; X = component of (PSIS midpoint → ASIS "
            "midpoint) orthogonal to Z, pointing anteriorly; Y = Z × X (upward). ISB "
            "places the origin at the hip joint centre, which is not palpable: the "
            "ASIS midpoint is used here instead."
        ),
    },
    "femur_right": {
        "name_fr": "Fémur D",
        "name_en": "Right femur",
        "required": [
            "lateral_knee_right", "medial_knee_right", "greater_trochanter_right",
        ],
        "method_fr": (
            "Origine au centre articulaire de hanche (proxy palpable : grand "
            "trochanter). Y = milieu des épicondyles fémoraux → origine, vers le "
            "crânial ; Z = composante de (épicondyle médial → latéral) orthogonale "
            "à Y, donc dans le plan (origine, EM, EL), vers la droite ; "
            "X = Y × Z (vers l'avant)."
        ),
        "method_en": (
            "Origin at the hip joint centre (palpable proxy: greater trochanter). "
            "Y = femoral epicondyle midpoint → origin, pointing cranially; "
            "Z = component of (medial → lateral epicondyle) orthogonal to Y, i.e. "
            "lying in the (origin, EM, EL) plane, pointing right; "
            "X = Y × Z (pointing anteriorly)."
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
            "Origine au point inter-malléolaire IM. Z = malléole médiale → malléole "
            "latérale (axe primaire, vers la droite) ; X = (IC − IM) × Z, normal au "
            "plan de torsion (IC, MM, LM), vers l'avant ; Y = Z × X. IC est le point "
            "inter-condylaire (proxy : milieu des condyles du genou). Noter que Y "
            "n'est pas exactement l'axe long IM → IC."
        ),
        "method_en": (
            "Origin at the inter-malleolar point IM. Z = medial → lateral malleolus "
            "(primary axis, pointing right); X = (IC − IM) × Z, normal to the "
            "torsional plane (IC, MM, LM), pointing anteriorly; Y = Z × X. IC is the "
            "inter-condylar point (proxy: knee condyle midpoint). Note that Y is not "
            "exactly the IM → IC long axis."
        ),
    },
}


# ── Constructeurs de repères ──────────────────────────────────────────────────
# Chaque constructeur retourne (origin, x, y, z) ou None si la construction est
# impossible (landmarks colinéaires / confondus).

def _build_thorax(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère thoracique ISB : origine IJ, Y vertical, Z vers la droite."""
    pts = _get(
        gt, "suprasternal_notch", "C7_spinous", "xiphoid_process", "T8_spinous",
    )
    if pts is None:
        return None
    ij, c7, px, t8 = pts
    # ISB : Yt relie milieu(PX, T8) à milieu(IJ, C7).
    caudal = _mid(px, t8)
    y = _unit(_mid(ij, c7) - caudal)
    if y is None:
        return None
    # ISB : Zt normal au plan (IJ, C7, milieu(PX, T8)).  L'ordre du produit
    # vectoriel ci-dessous pointe déjà vers la droite pour une anatomie normale.
    z = _unit(np.cross(ij - caudal, c7 - caudal))
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
    # ISB : Xs normal au plan (AI, AA, TS), vers l'avant.  Cet ordre du produit
    # vectoriel pointe vers l'avant sans dépendre d'un repère externe.
    x = _unit(np.cross(ai - aa, ts - aa))
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
    """Repère huméral droit : origine à GH (proxy tubercule majeur), Y proximal."""
    pts = _get(
        gt, "lateral_epicondyle_right", "medial_epicondyle_right",
        "greater_tubercle_right",
    )
    if pts is None:
        return None
    el, em, gh = pts
    # ISB : Oh1 coïncide avec GH (tubercule majeur utilisé comme proxy palpable),
    # et non avec le milieu des épicondyles, qui ne sert qu'à orienter Yh1.
    origin = gh
    y = _unit(gh - _mid(el, em))          # pointe vers GH (proximal)
    if y is None:
        return None
    x = _unit(np.cross(y, el - em))       # normal au plan (EL, EM, GH), vers l'avant
    if x is None:
        return None
    z = _unit(np.cross(x, y))             # complète le trièdre, vers la droite
    if z is None:
        return None
    return origin, x, y, z


def _build_pelvis(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère pelvien : Z le long de la ligne bi-EIAS, X antérieur, Y crânial."""
    pts = _get(gt, "ASIS_left", "ASIS_right", "PSIS_left", "PSIS_right")
    if pts is None:
        return None
    asis_l, asis_r, psis_l, psis_r = pts
    # ISB : origine au centre articulaire de hanche, non palpable ; le milieu des
    # EIAS est retenu ici comme origine pratique (les axes restent conformes).
    origin = _mid(asis_l, asis_r)
    # ISB : Z est l'axe primaire, parallèle à la ligne joignant les deux EIAS.
    z = _unit(asis_r - asis_l)
    if z is None:
        return None
    # ISB : X dans le plan (EIAS gauche, EIAS droite, milieu EIPS), orthogonal
    # à Z et pointant vers l'avant.
    x = _perp(origin - _mid(psis_l, psis_r), z)
    if x is None:
        return None
    y = _unit(np.cross(z, x))   # droite × avant = crânial ✓
    if y is None:
        return None
    return origin, x, y, z


def _build_femur_right(gt: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    """Repère fémoral droit : origine à la hanche (proxy grand trochanter)."""
    pts = _get(
        gt, "lateral_knee_right", "medial_knee_right", "greater_trochanter_right",
    )
    if pts is None:
        return None
    lat_knee, med_knee, troch = pts
    # ISB : origine au centre articulaire de hanche (grand trochanter = proxy
    # palpable) ; le milieu des épicondyles ne sert qu'à orienter y.
    origin = troch
    y = _unit(troch - _mid(lat_knee, med_knee))   # vers le crânial
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
    """Repère tibial droit : origine IM, Z le long de la ligne malléolaire."""
    pts = _get(
        gt, "lateral_malleolus_right", "medial_malleolus_right",
        "lateral_knee_right", "medial_knee_right",
    )
    if pts is None:
        return None
    lat_mal, med_mal, lat_knee, med_knee = pts
    origin = _mid(lat_mal, med_mal)          # IM, point inter-malléolaire
    ic = _mid(lat_knee, med_knee)            # IC, point inter-condylaire
    # ISB : Z est l'axe primaire (ligne MM–LM), et non une composante
    # orthogonalisée par rapport à l'axe long du tibia.
    z = _unit(lat_mal - med_mal)
    if z is None:
        return None
    # ISB : X normal au plan de torsion (IC, MM, LM), vers l'avant.
    x = _unit(np.cross(ic - origin, z))
    if x is None:
        return None
    # ISB : Y perpendiculaire commune à X et Z (proche de IM → IC, sans l'égaler).
    y = _unit(np.cross(z, x))
    if y is None:
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

# ── Estimation des centres articulaires ──────────────────────────────────────
# Les centres articulaires de la hanche (HJC) et de l'épaule (GHC) ne
# correspondent à aucun repère osseux palpable.  Deux familles de méthodes
# permettent de les estimer :
#
#   * Régressions prédictives : équations linéaires calibrées sur des cohortes
#     CT/IRM, exprimées dans le repère pelvis ou scapula.  Ne nécessitent pas
#     de mouvement, mais dépendent des mesures anthropométriques du bassin ou
#     de l'épaule.
#
#   * Méthodes fonctionnelles : pivotement de l'os autour du centre — axes
#     hélicoïdaux instantanés (Stokdijk 2000) ou sphère ajustée sur les
#     trajectoires de marqueurs (Gamage & Lasenby 2002).  Impossibles à partir
#     de données statiques.
#
# Ce module implémente les régressions ; les méthodes fonctionnelles sont
# mentionnées à titre éducatif.


@dataclass
class JointCenterResult:
    """Centre articulaire estimé et méta-informations associées."""

    joint: str            # "hip_right", "hip_left", "gh_right", "gh_left"
    method: str           # "bell_1990", "harrington_2007", "meskers_1998", …
    position: np.ndarray  # shape (3,) dans le repère monde
    error_mm: float | None = None   # erreur RMS reportée dans la publication (mm)
    note: str = ""

    @property
    def present(self) -> bool:
        return bool(np.any(self.position))


# ─── Centre articulaire de la hanche (HJC) ───────────────────────────────────

def _hjc_pelvis_to_world(
    gt: dict[str, np.ndarray],
    x_p: float, y_p: float, z_p: float,
) -> np.ndarray | None:
    """Convertit un offset (x_p, y_p, z_p) dans le repère pelvis en coordonnées monde.

    Les offsets sont en unités identiques à gt (généralement mètres pour les
    fichiers GLB exportés depuis Blender).  x_p = antérieur, y_p = supérieur,
    z_p = droite (ISB).
    """
    frame = _build_pelvis(gt)
    if frame is None:
        return None
    origin, x_ax, y_ax, z_ax = frame
    return origin + x_p * x_ax + y_p * y_ax + z_p * z_ax


def _hjc_bell(
    gt: dict[str, np.ndarray], side: str
) -> np.ndarray | None:
    """HJC selon Bell et al. (1990) / Davis et al. (1991).

    Offsets normalisés par la largeur inter-EIAS C, dans le repère pelvis ISB.
    Méthode indépendante des unités (pas de constante absolue).

    Références
    ----------
    Bell AL, Pedersen DR, Brand RA (1990) A comparison of the accuracy of several
    hip center location prediction methods. J Biomech 23(6):617-621.
    Davis RB et al. (1991) A gait analysis data collection and reduction technique.
    Hum Mov Sci 10(5):575-587.
    """
    pts = _get(gt, "ASIS_left", "ASIS_right")
    if pts is None:
        return None
    asis_l, asis_r = pts
    c = float(np.linalg.norm(asis_r - asis_l))   # largeur inter-EIAS
    if c < _EPS:
        return None
    sign = 1.0 if side == "right" else -1.0
    return _hjc_pelvis_to_world(gt,
        x_p = -0.24 * c,        # postérieur
        y_p = -0.30 * c,        # inférieur
        z_p =  sign * 0.36 * c, # latéral
    )


def _hjc_harrington(
    gt: dict[str, np.ndarray], side: str
) -> np.ndarray | None:
    """HJC selon Harrington et al. (2007) — régression avec C et D.

    La régression est définie en millimètres ; les coordonnées GT sont supposées
    en mètres (convention GLB/GLTF).  C = largeur inter-EIAS, D = profondeur
    EIAS–EIPS.

    Erreur RMS reportée : 15.4 mm (homme), 18.0 mm (femme).

    Références
    ----------
    Harrington ME et al. (2007) Prediction of the hip joint centre in adults,
    children, and patients with cerebral palsy based on magnetic resonance
    imaging. J Biomech 40(3):595-602.
    """
    pts = _get(gt, "ASIS_left", "ASIS_right", "PSIS_left", "PSIS_right")
    if pts is None:
        return None
    asis_l, asis_r, psis_l, psis_r = pts
    c_m = float(np.linalg.norm(asis_r - asis_l))
    asis_mid = _mid(asis_l, asis_r)
    psis_mid = _mid(psis_l, psis_r)
    d_m = float(np.linalg.norm(asis_mid - psis_mid))
    if c_m < _EPS or d_m < _EPS:
        return None
    # Conversion mètres → mm
    c_mm, d_mm = c_m * 1000.0, d_m * 1000.0
    sign = 1.0 if side == "right" else -1.0
    # Offsets en mm (Harrington 2007, Table 2) → reconversion en mètres
    x_mm = -0.138 * d_mm - 10.4    # postérieur (négatif dans repère ant-post)
    y_mm = -0.283 * d_mm - 22.0    # inférieur
    z_mm =  sign * (0.262 * c_mm + 18.0)  # latéral
    return _hjc_pelvis_to_world(gt,
        x_p = x_mm / 1000.0,
        y_p = y_mm / 1000.0,
        z_p = z_mm / 1000.0,
    )


def estimate_hjc(
    ground_truth: dict[str, np.ndarray],
    side: str = "right",
    method: str = "harrington_2007",
) -> JointCenterResult | None:
    """Estime le centre articulaire de la hanche (HJC) par régression.

    Parameters
    ----------
    ground_truth : dict mapping landmark code → position (3,)
    side : ``"right"`` ou ``"left"``
    method : ``"bell_1990"`` ou ``"harrington_2007"``

    Returns
    -------
    JointCenterResult ou None si les landmarks requis sont absents.
    """
    gt = ground_truth or {}
    joint = f"hip_{side}"
    if method == "bell_1990":
        pos = _hjc_bell(gt, side)
        err = 18.0   # Bell 1990, erreur approximative ~18 mm
        note = (
            "Bell AL et al. (1990). Offsets normalisés par la largeur "
            "inter-EIAS — indépendant des unités."
        )
    elif method == "harrington_2007":
        pos = _hjc_harrington(gt, side)
        err = 17.0   # moyenne reportée
        note = (
            "Harrington ME et al. (2007). Régression sur C (inter-EIAS) et "
            "D (EIAS–EIPS) — suppose les coordonnées GLB en mètres."
        )
    else:
        return None
    if pos is None:
        return None
    return JointCenterResult(
        joint=joint, method=method, position=pos, error_mm=err, note=note,
    )


# ─── Centre articulaire gléno-huméral (GHC) ──────────────────────────────────

def _ghc_meskers_simplified(
    gt: dict[str, np.ndarray], side: str
) -> np.ndarray | None:
    """GHC selon Meskers et al. (1998), version simplifiée sans processus coracoïde.

    L'offset en mm dans le repère scapula (AA origin, X ant, Y sup, Z lat) est
    issu de la régression de Meskers sans PC.  Erreur augmentée par rapport à
    la régression complète (avec PC : ~5 mm ; sans PC : ~10-15 mm).

    Références
    ----------
    Meskers CGM et al. (1998) In vivo estimation of the glenohumeral joint
    rotation center from scapular bony landmarks by linear regression.
    J Biomech 31(1):93-96.
    """
    sfx = f"_{side}"
    pts = _get(
        gt,
        f"acromial_angle{sfx}", f"scapula_trigonum{sfx}",
        f"scapula_inferior_angle{sfx}", f"acromioclavicular_joint{sfx}",
    )
    if pts is None:
        # Tentative sans AC
        pts3 = _get(gt, f"acromial_angle{sfx}", f"scapula_trigonum{sfx}",
                    f"scapula_inferior_angle{sfx}")
        if pts3 is None:
            return None
        aa, ts, ai = pts3
        ac = None
    else:
        aa, ts, ai, ac = pts

    frame = _build_scapula_right(gt) if side == "right" else None
    if frame is None:
        return None
    _, x_s, y_s, z_s = frame   # axes du repère scapula (origin = AA)

    # Offset de GH par rapport à AA dans le repère scapula (en mètres).
    # Valeurs de Meskers 1998 (sans PC) converties mm → m :
    #   -21 mm selon X (postérieur), +28 mm selon Y (supérieur), +22 mm selon Z (latéral)
    # Pour le côté gauche, Z est inversé.
    sign = 1.0 if side == "right" else -1.0
    gh = aa + (-0.021 * x_s + 0.028 * y_s + sign * 0.022 * z_s)
    return gh


def _ghc_functional_note() -> str:
    """Retourne une note pédagogique sur la méthode fonctionnelle (hélicale)."""
    return (
        "Méthode fonctionnelle (Stokdijk 2000 ; Gamage & Lasenby 2002) : "
        "le GH est estimé comme centre de rotation pendant un mouvement actif "
        "de circumduction.  Non disponible sur données statiques."
    )


def estimate_ghc(
    ground_truth: dict[str, np.ndarray],
    side: str = "right",
    method: str = "meskers_1998",
) -> JointCenterResult | None:
    """Estime le centre gléno-huméral (GHC) par régression.

    Parameters
    ----------
    ground_truth : dict mapping landmark code → position (3,)
    side : ``"right"`` ou ``"left"``
    method : ``"meskers_1998"`` (régression scapulaire sans PC) ou
             ``"greater_tubercle"`` (proxy palpable, moins précis)

    Returns
    -------
    JointCenterResult ou None si les landmarks requis sont absents.
    """
    gt = ground_truth or {}
    joint = f"gh_{side}"
    if method == "meskers_1998":
        pos = _ghc_meskers_simplified(gt, side)
        err = 14.0   # estimation sans PC (~14 mm selon Meskers 1998)
        note = (
            "Meskers CGM et al. (1998), version sans PC (processus coracoïde "
            "non palpable dans ce modèle).  "
            + _ghc_functional_note()
        )
    elif method == "greater_tubercle":
        sfx = f"_{side}"
        pts = _get(gt, f"greater_tubercle{sfx}")
        pos = pts[0] if pts else None
        err = 35.0   # le tubercule majeur est ~35 mm latéral au vrai GH
        note = (
            "Proxy palpable : le tubercule majeur est latéral et légèrement "
            "antérieur au vrai centre de rotation gléno-huméral (écart ~30-40 mm)."
        )
    else:
        return None
    if pos is None:
        return None
    return JointCenterResult(
        joint=joint, method=method, position=pos, error_mm=err, note=note,
    )


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
        "Étape 2 : construire l'axe Y en joignant le milieu du segment PX–T8 au "
        "milieu du segment IJ–C7 ; il pointe vers le haut.",
        "Étape 3 : construire l'axe Z perpendiculaire au plan formé par IJ, C7 et "
        "le milieu de PX–T8, orienté vers la droite du sujet.",
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
        "Étape 3 : construire l'axe X perpendiculaire au plan (AI, AA, TS), orienté "
        "vers l'avant. Attention : l'ISB utilisant AA et non AC, ce plan n'est pas "
        "le plan visuel de la scapula.",
        "Étape 4 : obtenir l'axe Y par le produit vectoriel Z × X ; il pointe vers le "
        "haut, parallèle au bord médial.",
    ],
    "humerus_right": [
        "Étape 1 : placer l'origine sur le centre gléno-huméral (GH), approximé "
        "ici par le tubercule majeur. L'ISB place bien l'origine à l'épaule, pas "
        "au coude.",
        "Étape 2 : construire l'axe Y du milieu des épicondyles (EL, EM) vers GH ; "
        "il pointe proximalement.",
        "Étape 3 : construire l'axe X perpendiculaire au plan formé par EL, EM et "
        "GH, orienté vers l'avant (X = Y × (EL − EM)).",
        "Étape 4 : obtenir l'axe Z par le produit vectoriel X × Y ; il pointe "
        "latéralement, vers la droite.",
    ],
    "pelvis": [
        "Étape 1 : placer l'origine au milieu des deux épines iliaques antéro-"
        "supérieures (EIAS). L'ISB spécifie le centre articulaire de hanche, non "
        "palpable : le milieu des EIAS en tient lieu.",
        "Étape 2 : construire l'axe Z de l'EIAS gauche vers l'EIAS droite ; il "
        "pointe vers la droite le long de la ligne bi-EIAS.",
        "Étape 3 : construire l'axe X dans le plan des deux EIAS et du milieu des "
        "EIPS, orthogonal à Z et pointant vers l'avant (composante de « milieu "
        "EIPS → milieu EIAS » perpendiculaire à Z).",
        "Étape 4 : obtenir l'axe Y par le produit vectoriel Z × X ; il pointe vers "
        "le haut.",
    ],
    "femur_right": [
        "Étape 1 : placer l'origine sur le centre articulaire de hanche, approximé "
        "ici par le grand trochanter. L'ISB place l'origine à la hanche, pas au "
        "genou.",
        "Étape 2 : construire l'axe Y du milieu des épicondyles fémoraux vers "
        "l'origine ; il pointe crânialement le long de la diaphyse.",
        "Étape 3 : construire l'axe Z à partir de la ligne épicondyle médial → "
        "épicondyle latéral, en n'en gardant que la composante perpendiculaire à Y ; "
        "il pointe vers la droite.",
        "Étape 4 : obtenir l'axe X par le produit vectoriel Y × Z ; il pointe vers "
        "l'avant.",
    ],
    "tibia_right": [
        "Étape 1 : placer l'origine au point inter-malléolaire IM, milieu des "
        "malléoles latérale et médiale.",
        "Étape 2 : construire l'axe Z de la malléole médiale vers la malléole "
        "latérale ; c'est l'axe primaire du tibia selon l'ISB, il pointe vers la "
        "droite.",
        "Étape 3 : construire l'axe X perpendiculaire au plan de torsion — plan "
        "contenant IM, les deux malléoles et le point inter-condylaire IC (milieu "
        "des condyles du genou) — orienté vers l'avant.",
        "Étape 4 : obtenir l'axe Y par le produit vectoriel Z × X ; il pointe vers "
        "le haut. Attention : il est proche de l'axe long IM → IC sans lui être "
        "exactement confondu.",
    ],
}

_STEPS_EN: dict[str, list[str]] = {
    "thorax": [
        "Step 1: place the origin on the suprasternal notch (IJ), the upper hollow "
        "of the sternum.",
        "Step 2: build the Y axis from the midpoint of the PX–T8 segment to the "
        "midpoint of the IJ–C7 segment; it points upwards.",
        "Step 3: build the Z axis perpendicular to the plane formed by IJ, C7 and "
        "the midpoint of PX–T8, pointing to the subject's right.",
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
        "Step 3: build the X axis perpendicular to the (AI, AA, TS) plane, pointing "
        "anteriorly. Beware: since ISB uses AA rather than AC, this is not the "
        "visual plane of the scapular bone.",
        "Step 4: obtain the Y axis as the cross product Z × X; it points upwards, "
        "parallel to the medial border.",
    ],
    "humerus_right": [
        "Step 1: place the origin on the glenohumeral centre (GH), approximated "
        "here by the greater tubercle. ISB places the origin at the shoulder, not "
        "at the elbow.",
        "Step 2: build the Y axis from the midpoint of the epicondyles (EL, EM) to "
        "GH; it points proximally.",
        "Step 3: build the X axis perpendicular to the plane formed by EL, EM and "
        "GH, pointing anteriorly (X = Y × (EL − EM)).",
        "Step 4: obtain the Z axis as X × Y; it points laterally, to the right.",
    ],
    "pelvis": [
        "Step 1: place the origin midway between the left and right anterior "
        "superior iliac spines (ASIS). ISB specifies the hip joint centre, which "
        "is not palpable: the ASIS midpoint stands in for it.",
        "Step 2: build the Z axis from the left ASIS to the right ASIS; it points "
        "rightward along the inter-ASIS line.",
        "Step 3: build the X axis in the plane of the two ASIS and the PSIS "
        "midpoint, orthogonal to Z and pointing anteriorly (the component of "
        "'PSIS midpoint → ASIS midpoint' perpendicular to Z).",
        "Step 4: obtain the Y axis as the cross product Z × X; it points superiorly.",
    ],
    "femur_right": [
        "Step 1: place the origin on the hip joint centre, approximated here by the "
        "greater trochanter. ISB places the origin at the hip, not at the knee.",
        "Step 2: build the Y axis from the femoral epicondyle midpoint to the "
        "origin; it points cranially along the shaft.",
        "Step 3: build the Z axis from the medial → lateral epicondyle line, keeping "
        "only its component perpendicular to Y; it points to the right.",
        "Step 4: obtain the X axis as the cross product Y × Z; it points forward.",
    ],
    "tibia_right": [
        "Step 1: place the origin on the inter-malleolar point IM, midway between "
        "the lateral and medial malleoli.",
        "Step 2: build the Z axis from the medial to the lateral malleolus; this is "
        "the primary tibial axis in the ISB definition, pointing to the right.",
        "Step 3: build the X axis perpendicular to the torsional plane — the plane "
        "containing IM, both malleoli and the inter-condylar point IC (knee condyle "
        "midpoint) — pointing anteriorly.",
        "Step 4: obtain the Y axis as the cross product Z × X; it points upwards. "
        "Beware: it is close to, but not exactly, the IM → IC long axis.",
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
