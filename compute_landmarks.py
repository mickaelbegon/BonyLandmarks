"""
Compute anatomical landmark positions on BodyParts3D bone meshes.

BodyParts3D coordinate system (empirically confirmed):
  X: LEFT = +X,  RIGHT = -X   (opposite of standard RAS)
  Y: POSTERIOR = +Y, ANTERIOR = -Y   (confirmed: glabella at min Y)
  Z: SUPERIOR = +Z, INFERIOR = -Z  (height from floor, ~0 at feet, ~1750 at head)

For each bony landmark in bone_map.py (excluding EMG / skinfold / anthro),
finds the best-matching vertex using anatomical geometric rules, then:
  - saves  src/bonylandmarks/data/bones/landmark_positions.json   {code: [x, y, z]}
  - saves  src/bonylandmarks/data/bones/validation/<bone>.png      one image per bone

Usage:
  python compute_landmarks.py          # compute + save JSON + render validation PNGs
  python compute_landmarks.py --dry    # compute only, print errors, no save
"""

import json
import numpy as np
import pyvista as pv
from pathlib import Path
from collections import defaultdict

_REPO     = Path(__file__).parent
BONES_DIR = _REPO / "src" / "bonylandmarks" / "data" / "bones"
OUT_JSON  = BONES_DIR / "landmark_positions.json"
OUT_VAL   = BONES_DIR / "validation"

# ── Axis constants  ──────────────────────────────────────────────────────────
# BodyParts3D confirmed: X-laterality reversed, Z=height
AX_X, AX_Y, AX_Z = 0, 1, 2

# Directional signs relative to positive axis
# right = negative X direction (femur_right centroid X < 0)
LAT_RIGHT = -1    # right side = -X direction
LAT_LEFT  = +1    # left side  = +X direction
SUP       = +1    # superior = +Z
ANT       = -1    # anterior = -Y, posterior = +Y  (confirmed by user feedback)

def lat(side: str) -> int:
    return LAT_RIGHT if side == "right" else LAT_LEFT


# ── Geometric helpers  ───────────────────────────────────────────────────────
def pct_mask(pts: np.ndarray, axis: int, lo: float, hi: float) -> np.ndarray:
    """Boolean mask: vertices between lo and hi percentile along axis."""
    v = pts[:, axis]
    return (v >= np.percentile(v, lo)) & (v <= np.percentile(v, hi))


def extreme(pts: np.ndarray, axis: int, sign: int = 1,
            mask: np.ndarray | None = None) -> int:
    """Index of vertex maximized (sign=+1) or minimized (sign=-1) along axis."""
    if mask is not None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0
        best = int(np.argmax(sign * pts[idx, axis]))
        return int(idx[best])
    return int(np.argmax(sign * pts[:, axis]))


def score_extreme(pts: np.ndarray, axes_signs: list,
                  mask: np.ndarray | None = None) -> int:
    """Index of vertex maximizing weighted sum of signed axes."""
    if mask is not None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0
        sub = pts[idx]
        scores = sum(sign * sub[:, ax] for ax, sign in axes_signs)
        return int(idx[int(np.argmax(scores))])
    scores = sum(sign * pts[:, ax] for ax, sign in axes_signs)
    return int(np.argmax(scores))


# ── Landmark rules  ──────────────────────────────────────────────────────────
# Each entry: landmark_code -> (bone_stem, rule_fn(pts: ndarray) -> [x,y,z])
RULES: dict[str, tuple[str, callable]] = {}

# ── Skull  ───────────────────────────────────────────────────────────────────
RULES["vertex"] = ("skull",
    lambda p: p[extreme(p, AX_Z, SUP)].tolist())

RULES["glabella"] = ("skull",
    lambda p: p[extreme(p, AX_Y, ANT,
        mask=pct_mask(p, AX_Z, 46, 74))].tolist())

RULES["external_occipital_protuberance"] = ("skull",
    lambda p: p[extreme(p, AX_Y, -ANT,
        mask=pct_mask(p, AX_Z, 52, 82))].tolist())

def _eam(p, l):
    # EAM: most lateral in mid-low skull Z band
    # Note: zygomatic arch (in skull mesh) is ~15mm more lateral → ~20mm error irreducible
    mask = pct_mask(p, AX_Z, 18, 44)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()
RULES["external_acoustic_meatus_right"] = ("skull", lambda p: _eam(p, LAT_RIGHT))
RULES["external_acoustic_meatus_left"]  = ("skull", lambda p: _eam(p, LAT_LEFT))

def _mastoid(p, l):
    # Mastoid: lateral+inferior in low Z band; inferior weight avoids finding high skull
    mask = pct_mask(p, AX_Z, 5, 19)
    return p[score_extreme(p, [(AX_X, l * 3), (AX_Z, -SUP * 2)], mask=mask)].tolist()
RULES["mastoid_process_right"] = ("skull", lambda p: _mastoid(p, LAT_RIGHT))
RULES["mastoid_process_left"]  = ("skull", lambda p: _mastoid(p, LAT_LEFT))

# ── Sternum  ──────────────────────────────────────────────────────────────────
def _suprasternal_notch(p):
    cx = p[:, AX_X].mean()
    x_range = float(p[:, AX_X].max() - p[:, AX_X].min())
    midline = np.abs(p[:, AX_X] - cx) < x_range * 0.08
    return p[extreme(p, AX_Z, SUP, mask=midline)].tolist()

RULES["suprasternal_notch"] = ("sternum", _suprasternal_notch)

RULES["xiphoid_process"] = ("sternum",
    lambda p: p[extreme(p, AX_Z, -SUP)].tolist())

RULES["sternal_angle_louis"] = ("sternum",
    lambda p: p[extreme(p, AX_Y, ANT,
        mask=pct_mask(p, AX_Z, 78, 94) & (np.abs(p[:, AX_X] - p[:, AX_X].mean()) < 15))].tolist())

# ── Rachis  ───────────────────────────────────────────────────────────────────
def _spinous(p, z_lo, z_hi):
    """Spinous process = most posterior point in given Z-percentile band."""
    mask = pct_mask(p, AX_Z, z_lo, z_hi)
    return p[extreme(p, AX_Y, -ANT, mask=mask)].tolist()

RULES["C7_spinous"]  = ("cervical_vertebrae",  lambda p: _spinous(p,  0, 18))
RULES["T1_spinous"]  = ("thoracic_vertebrae",  lambda p: _spinous(p, 88, 100))
RULES["T4_spinous"]  = ("thoracic_vertebrae",  lambda p: _spinous(p, 62, 75))
RULES["T7_spinous"]  = ("thoracic_vertebrae",  lambda p: _spinous(p, 43, 56))
RULES["T8_spinous"]  = ("thoracic_vertebrae",  lambda p: _spinous(p, 32, 45))
RULES["T10_spinous"] = ("thoracic_vertebrae",  lambda p: _spinous(p, 15, 28))
RULES["T12_spinous"] = ("thoracic_vertebrae",  lambda p: _spinous(p,  0, 12))
RULES["L1_spinous"]  = ("lumbar_vertebrae",    lambda p: _spinous(p, 80, 100))
RULES["L5_spinous"]  = ("lumbar_vertebrae",    lambda p: _spinous(p,  0, 15))
RULES["S1_spinous"]  = ("sacrum",
    lambda p: p[score_extreme(p, [(AX_Y, -ANT * 2), (AX_Z, SUP)],
                              mask=pct_mask(p, AX_Z, 65, 100))].tolist())

# ── Côtes  ────────────────────────────────────────────────────────────────────
def _rib12_tip(p, l):
    mask = pct_mask(p, AX_Z, 0, 18)
    return p[score_extreme(p, [(AX_X, l), (AX_Z, -SUP)], mask=mask)].tolist()
RULES["rib12_tip_right"] = ("rib_right", lambda p: _rib12_tip(p, LAT_RIGHT))
RULES["rib12_tip_left"]  = ("rib_left",  lambda p: _rib12_tip(p, LAT_LEFT))

# ── Scapula  ──────────────────────────────────────────────────────────────────
def _acromion(p, l):
    mask = pct_mask(p, AX_Z, 60, 100)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()

def _acromial_angle(p, l):
    mask = pct_mask(p, AX_Z, 55, 100)
    return p[score_extreme(p, [(AX_X, l), (AX_Y, -ANT)], mask=mask)].tolist()

def _scap_trigonum(p, l):
    # Root of scapular spine at medial border: medial + posterior + mid-height
    mask = pct_mask(p, AX_Z, 35, 62)
    return p[score_extreme(p, [(AX_X, -l), (AX_Y, -ANT * 2)], mask=mask)].tolist()

def _scap_inf_angle(p):
    return p[extreme(p, AX_Z, -SUP)].tolist()

def _scap_sup_angle(p, l):
    mask = pct_mask(p, AX_Z, 80, 100)
    return p[score_extreme(p, [(AX_X, -l), (AX_Z, SUP * 2)], mask=mask)].tolist()

def _scap_spine(p, l):
    # Spine midpoint: corrected X = midpoint of full scapula X range (= geometric center trigonum→acromion)
    # Find the most posterior point near that X in the spine Z/Y region
    mask = (pct_mask(p, AX_Z, 55, 80) &
            (p[:, AX_Y] >= np.percentile(p[:, AX_Y], 70)))
    pts = p[mask]
    x_mid = (p[:, AX_X].min() + p[:, AX_X].max()) / 2  # full scapula X center
    x_range = pts[:, AX_X].max() - pts[:, AX_X].min()
    y_range = pts[:, AX_Y].max() - pts[:, AX_Y].min() + 0.01
    x_dist_norm = np.abs(pts[:, AX_X] - x_mid) / (x_range + 0.01)
    y_norm = (pts[:, AX_Y] - pts[:, AX_Y].min()) / y_range
    return pts[np.argmax(2 * y_norm - x_dist_norm)].tolist()

def _coracoid(p, l):
    # Coracoid tip: anterior and slightly inferior within upper scapula
    mask = pct_mask(p, AX_Z, 65, 90)
    return p[score_extreme(p, [(AX_Y, ANT * 2), (AX_Z, -SUP)], mask=mask)].tolist()

for s in ("right", "left"):
    l = lat(s)
    RULES[f"acromion_{s}"]              = (f"scapula_{s}", lambda p, _l=l: _acromion(p, _l))
    RULES[f"acromial_angle_{s}"]        = (f"scapula_{s}", lambda p, _l=l: _acromial_angle(p, _l))
    RULES[f"scapula_trigonum_{s}"]      = (f"scapula_{s}", lambda p, _l=l: _scap_trigonum(p, _l))
    RULES[f"scapula_inferior_angle_{s}"]= (f"scapula_{s}", lambda p, _l=l: _scap_inf_angle(p))
    RULES[f"scapula_superior_angle_{s}"]= (f"scapula_{s}", lambda p, _l=l: _scap_sup_angle(p, _l))
    RULES[f"scapular_spine_{s}"]        = (f"scapula_{s}", lambda p, _l=l: _scap_spine(p, _l))
    RULES[f"coracoid_process_{s}"]      = (f"scapula_{s}", lambda p, _l=l: _coracoid(p, _l))

# ── Clavicule  ────────────────────────────────────────────────────────────────
def _ac_joint(p, l):
    # ACJ: medial end of acromion at the very top of the scapula (articular face with clavicle)
    # Most MEDIAL (-l direction) and ANTERIOR in the top 4% Z → finds the clavicular facet
    mask = pct_mask(p, AX_Z, 96, 100)
    return p[score_extreme(p, [(AX_X, -l), (AX_Y, ANT)], mask=mask)].tolist()

def _sc_joint(p, l):
    # Medial end of clavicle: most medial, then anterior + superior (articular face)
    medial_val = p[:, AX_X] * (-l)   # high = medial
    mask = medial_val >= np.percentile(medial_val, 65)
    return p[score_extreme(p, [(AX_X, -l * 2), (AX_Y, ANT), (AX_Z, SUP)], mask=mask)].tolist()

for s in ("right", "left"):
    l = lat(s)
    RULES[f"acromioclavicular_joint_{s}"] = (f"scapula_{s}", lambda p, _l=l: _ac_joint(p, _l))
    RULES[f"sternoclavicular_joint_{s}"]  = (f"clavicle_{s}", lambda p, _l=l: _sc_joint(p, _l))

# ── Humerus  ──────────────────────────────────────────────────────────────────
def _greater_tubercle(p, l):
    # Greater tubercle: lateral + slightly superior prominence below the humeral head
    # X weight >> Z weight so lateral is primary; Z breaks ties toward the tubercle apex
    mask = pct_mask(p, AX_Z, 80, 97)
    return p[score_extreme(p, [(AX_X, l * 3), (AX_Z, SUP)], mask=mask)].tolist()

def _deltoid_tub(p, l):
    # Deltoid tuberosity: mid-to-upper shaft, most LATERAL + preferring LOWER Z
    # Mask: above 57% of humeral length (excludes lower shaft and the narrow elbow region)
    # Score lX - Z: prefer lateral and then the lowest possible Z in that band
    # This identifies the tuberosity as the most prominent lateral bump just above mid-shaft
    z = p[:, AX_Z]
    zmin, zmax = z.min(), z.max()
    mask = z >= zmin + 0.57 * (zmax - zmin)
    return p[score_extreme(p, [(AX_X, l), (AX_Z, -SUP)], mask=mask)].tolist()

def _lat_epicondyle(p, l):
    # Lateral epicondyle is on the postero-lateral face → add posterior weight, extend Z range
    mask = pct_mask(p, AX_Z, 0, 22)
    return p[score_extreme(p, [(AX_X, l * 2), (AX_Y, -ANT)], mask=mask)].tolist()

def _med_epicondyle(p, l):
    mask = pct_mask(p, AX_Z, 0, 18)
    return p[extreme(p, AX_X, -l, mask=mask)].tolist()

def _radiale(p, l):
    # Radiale = radial head = proximal end of the radius: highest Z point (lateral tiebreaker)
    mask = pct_mask(p, AX_Z, 88, 100)
    return p[score_extreme(p, [(AX_Z, SUP * 5), (AX_X, l)], mask=mask)].tolist()

for s in ("right", "left"):
    l = lat(s)
    RULES[f"greater_tubercle_{s}"]   = (f"humerus_{s}", lambda p, _l=l: _greater_tubercle(p, _l))
    RULES[f"deltoid_tuberosity_{s}"] = (f"humerus_{s}", lambda p, _l=l: _deltoid_tub(p, _l))
    RULES[f"lateral_epicondyle_{s}"] = (f"humerus_{s}", lambda p, _l=l: _lat_epicondyle(p, _l))
    RULES[f"medial_epicondyle_{s}"]  = (f"humerus_{s}", lambda p, _l=l: _med_epicondyle(p, _l))
    RULES[f"radiale_{s}"]            = (f"radius_{s}",  lambda p, _l=l: _radiale(p, _l))

# ── Radius  ───────────────────────────────────────────────────────────────────
def _radial_styloid(p, l):
    mask = pct_mask(p, AX_Z, 0, 15)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()
RULES["radial_styloid_right"] = ("radius_right", lambda p: _radial_styloid(p, LAT_RIGHT))
RULES["radial_styloid_left"]  = ("radius_left",  lambda p: _radial_styloid(p, LAT_LEFT))

# ── Ulna  ─────────────────────────────────────────────────────────────────────
def _olecranon(p):
    mask = pct_mask(p, AX_Z, 77, 100)
    return p[extreme(p, AX_Y, -ANT, mask=mask)].tolist()

def _ulnar_styloid(p, l):
    mask = pct_mask(p, AX_Z, 0, 15)
    return p[score_extreme(p, [(AX_X, -l), (AX_Z, -SUP)], mask=mask)].tolist()

RULES["olecranon_right"]     = ("ulna_right", lambda p: _olecranon(p))
RULES["olecranon_left"]      = ("ulna_left",  lambda p: _olecranon(p))
RULES["ulnar_styloid_right"] = ("ulna_right", lambda p: _ulnar_styloid(p, LAT_RIGHT))
RULES["ulnar_styloid_left"]  = ("ulna_left",  lambda p: _ulnar_styloid(p, LAT_LEFT))

# ── Pelvis  ───────────────────────────────────────────────────────────────────
def _pelvis_half_mask(p, l):
    cx = p[:, AX_X].mean()
    return p[:, AX_X] < cx if l < 0 else p[:, AX_X] > cx

def _ASIS(p, l):
    half = _pelvis_half_mask(p, l)
    mask = half & pct_mask(p, AX_Z, 60, 100)
    return p[score_extreme(p, [(AX_Y, ANT), (AX_Z, SUP)], mask=mask)].tolist()

def _AIIS(p, l):
    half = _pelvis_half_mask(p, l)
    mask = half & pct_mask(p, AX_Z, 38, 62)
    return p[extreme(p, AX_Y, ANT, mask=mask)].tolist()

def _PSIS(p, l):
    half = _pelvis_half_mask(p, l)
    mask = half & pct_mask(p, AX_Z, 70, 100)
    return p[score_extreme(p, [(AX_Y, -ANT), (AX_Z, SUP)], mask=mask)].tolist()

def _iliac_crest(p, l):
    half = _pelvis_half_mask(p, l)
    return p[extreme(p, AX_Z, SUP, mask=half)].tolist()

def _ischial_tub(p, l):
    half = _pelvis_half_mask(p, l)
    mask = half & pct_mask(p, AX_Z, 0, 22)
    return p[score_extreme(p, [(AX_Z, -SUP), (AX_Y, -ANT)], mask=mask)].tolist()

def _pubic_symphysis(p):
    cx = p[:, AX_X].mean()
    x_range = float(p[:, AX_X].max() - p[:, AX_X].min())
    mask = np.abs(p[:, AX_X] - cx) < x_range * 0.10
    mask &= pct_mask(p, AX_Z, 15, 48)
    return p[extreme(p, AX_Y, ANT, mask=mask)].tolist()

for s in ("right", "left"):
    l = lat(s)
    RULES[f"ASIS_{s}"]               = ("pelvis", lambda p, _l=l: _ASIS(p, _l))
    RULES[f"AIIS_{s}"]               = ("pelvis", lambda p, _l=l: _AIIS(p, _l))
    RULES[f"PSIS_{s}"]               = ("pelvis", lambda p, _l=l: _PSIS(p, _l))
    RULES[f"iliac_crest_{s}"]        = ("pelvis", lambda p, _l=l: _iliac_crest(p, _l))
    RULES[f"ischial_tuberosity_{s}"] = ("pelvis", lambda p, _l=l: _ischial_tub(p, _l))
RULES["pubic_symphysis"] = ("pelvis", _pubic_symphysis)

# ── Sacrum  ───────────────────────────────────────────────────────────────────
RULES["sacrum_S2"] = ("sacrum", lambda p: _spinous(p, 45, 65))

# ── Fémur  ────────────────────────────────────────────────────────────────────
def _greater_troch(p, l):
    mask = pct_mask(p, AX_Z, 72, 94)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()

def _lat_knee_femur(p, l):
    mask = pct_mask(p, AX_Z, 0, 18)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()

def _med_knee_femur(p, l):
    mask = pct_mask(p, AX_Z, 0, 18)
    return p[extreme(p, AX_X, -l, mask=mask)].tolist()

def _adductor_tubercle(p, l):
    mask = pct_mask(p, AX_Z, 15, 28)
    return p[extreme(p, AX_X, -l, mask=mask)].tolist()

for s in ("right", "left"):
    l = lat(s)
    RULES[f"greater_trochanter_{s}"]  = (f"femur_{s}", lambda p, _l=l: _greater_troch(p, _l))
    RULES[f"lateral_knee_{s}"]        = (f"femur_{s}", lambda p, _l=l: _lat_knee_femur(p, _l))
    RULES[f"medial_knee_{s}"]         = (f"femur_{s}", lambda p, _l=l: _med_knee_femur(p, _l))
    RULES[f"adductor_tubercle_{s}"]   = (f"femur_{s}", lambda p, _l=l: _adductor_tubercle(p, _l))

# ── Patella  ──────────────────────────────────────────────────────────────────
for s in ("right", "left"):
    RULES[f"patella_superior_{s}"] = (f"patella_{s}",
        lambda p: p[extreme(p, AX_Z, SUP)].tolist())
    RULES[f"patella_center_{s}"]   = (f"patella_{s}",
        lambda p: p.mean(axis=0).tolist())

# ── Tibia  ────────────────────────────────────────────────────────────────────
def _tib_tub(p):
    mask = pct_mask(p, AX_Z, 76, 92)
    return p[extreme(p, AX_Y, ANT, mask=mask)].tolist()

def _gerdy(p, l):
    mask = pct_mask(p, AX_Z, 73, 90)
    return p[score_extreme(p, [(AX_X, l), (AX_Y, ANT)], mask=mask)].tolist()

def _pes_anserinus(p, l):
    mask = pct_mask(p, AX_Z, 73, 90)
    return p[score_extreme(p, [(AX_X, -l), (AX_Y, ANT)], mask=mask)].tolist()

def _med_malleolus(p, l):
    mask = pct_mask(p, AX_Z, 0, 16)
    return p[extreme(p, AX_X, -l, mask=mask)].tolist()

for s in ("right", "left"):
    l = lat(s)
    RULES[f"tibial_tuberosity_{s}"] = (f"tibia_{s}", lambda p, _l=l: _tib_tub(p))
    RULES[f"gerdy_tubercle_{s}"]    = (f"tibia_{s}", lambda p, _l=l: _gerdy(p, _l))
    RULES[f"pes_anserinus_{s}"]     = (f"tibia_{s}", lambda p, _l=l: _pes_anserinus(p, _l))
    RULES[f"medial_malleolus_{s}"]  = (f"tibia_{s}", lambda p, _l=l: _med_malleolus(p, _l))

# ── Fibula  ───────────────────────────────────────────────────────────────────
def _fib_head(p, l):
    mask = pct_mask(p, AX_Z, 82, 100)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()

def _lat_malleolus(p, l):
    mask = pct_mask(p, AX_Z, 0, 16)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()

RULES["fibular_head_right"]     = ("fibula_right", lambda p: _fib_head(p, LAT_RIGHT))
RULES["fibular_head_left"]      = ("fibula_left",  lambda p: _fib_head(p, LAT_LEFT))
RULES["lateral_malleolus_right"]= ("fibula_right", lambda p: _lat_malleolus(p, LAT_RIGHT))
RULES["lateral_malleolus_left"] = ("fibula_left",  lambda p: _lat_malleolus(p, LAT_LEFT))

# ── Calcaneus  ────────────────────────────────────────────────────────────────
def _heel(p):
    return p[score_extreme(p, [(AX_Y, -ANT), (AX_Z, -SUP)])].tolist()

RULES["heel_right"] = ("calcaneus_right", lambda p: _heel(p))
RULES["heel_left"]  = ("calcaneus_left",  lambda p: _heel(p))

# ── Pied (calcaneus + naviculaire + métatarses)  ─────────────────────────────
def _navicular_tub(p, l):
    mask = pct_mask(p, AX_Y, 35, 65)
    return p[extreme(p, AX_X, -l, mask=mask)].tolist()

def _meta5_base(p, l):
    mask = pct_mask(p, AX_Y, 20, 50)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()

def _meta5_head(p, l):
    mask = pct_mask(p, AX_Y, 0, 28)
    return p[extreme(p, AX_X, l, mask=mask)].tolist()

def _meta1_head(p, l):
    mask = pct_mask(p, AX_Y, 0, 28)
    return p[extreme(p, AX_X, -l, mask=mask)].tolist()

def _meta2_head(p, l):
    mask = pct_mask(p, AX_Y, 0, 28)
    return p[mask].mean(axis=0).tolist()

for s in ("right", "left"):
    l = lat(s)
    RULES[f"navicular_tuberosity_{s}"]  = (f"foot_{s}", lambda p, _l=l: _navicular_tub(p, _l))
    RULES[f"fifth_metatarsal_base_{s}"] = (f"foot_{s}", lambda p, _l=l: _meta5_base(p, _l))
    RULES[f"fifth_metatarsal_head_{s}"] = (f"foot_{s}", lambda p, _l=l: _meta5_head(p, _l))
    RULES[f"first_metatarsal_head_{s}"] = (f"foot_{s}", lambda p, _l=l: _meta1_head(p, _l))
    RULES[f"second_metatarsal_head_{s}"]= (f"foot_{s}", lambda p, _l=l: _meta2_head(p, _l))


# ── Bone mesh loader  ────────────────────────────────────────────────────────
BONE_CACHE: dict[str, np.ndarray] = {}

def get_pts(bone_stem: str) -> np.ndarray:
    if bone_stem not in BONE_CACHE:
        for ext in (".obj", ".stl"):
            path = BONES_DIR / f"{bone_stem}{ext}"
            if path.exists():
                mesh = pv.read(str(path))
                BONE_CACHE[bone_stem] = np.array(mesh.points, dtype=float)
                break
    return BONE_CACHE[bone_stem]


if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv

    OUT_VAL.mkdir(exist_ok=True)

    # ── Compute all positions  ──────────────────────────────────────────────
    positions: dict[str, list[float]] = {}
    errors: list[str] = []

    for code, (bone, rule_fn) in RULES.items():
        try:
            pts = get_pts(bone)
            pos = rule_fn(pts)
            if isinstance(pos, np.ndarray):
                pos = pos.tolist()
            positions[code] = [round(float(v), 4) for v in pos]
        except Exception as exc:
            errors.append(f"{code}: {exc}")
            print(f"  ERROR {code}: {exc}")

    print(f"\nComputed {len(positions)} landmark positions, {len(errors)} errors")

    if dry:
        print("--dry: JSON not saved.")
        sys.exit(0 if not errors else 1)

    # ── Save JSON  ──────────────────────────────────────────────────────────
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUT_JSON}")

    # ── Validation renders  ─────────────────────────────────────────────────
    bone_lm: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for code, pos in positions.items():
        bone = RULES[code][0]
        bone_lm[bone].append((code, pos))

    for bone_stem, lm_list in sorted(bone_lm.items()):
        path = BONES_DIR / f"{bone_stem}.obj"
        if not path.exists():
            path = BONES_DIR / f"{bone_stem}.stl"
        if not path.exists():
            continue
        mesh = pv.read(str(path))
        pl = pv.Plotter(off_screen=True, window_size=[900, 700])
        pl.add_mesh(mesh, color="#d4b896", opacity=0.72, smooth_shading=True)

        for code, pos in lm_list:
            r = mesh.length * 0.013
            sphere = pv.Sphere(radius=r, center=pos)
            pl.add_mesh(sphere, color="red")
            short = code.replace("_right", "_R").replace("_left", "_L")
            pl.add_point_labels(
                [pos], [short[:18]],
                font_size=7, text_color="yellow",
                shape=None, fill_shape=False, show_points=False,
                always_visible=True
            )

        pl.camera_position = "iso"
        pl.reset_camera()
        pl.background_color = "#1a1a2e"
        out_png = OUT_VAL / f"{bone_stem}.png"
        pl.screenshot(str(out_png))
        pl.close()
        print(f"  {bone_stem}.png  ({len(lm_list)} landmarks)")
