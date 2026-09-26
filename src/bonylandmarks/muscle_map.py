"""Mapping os → muscles pertinents (FMA concept IDs) pour l'affichage de référence."""

from __future__ import annotations

import re
from pathlib import Path

# FMA concept IDs des muscles par os (stem → liste de FMA IDs)
# Utiliser les IDs de la base BodyParts3D
BONE_MUSCLES: dict[str, list[int]] = {
    "skull": [],
    "sternum": [13039, 34696, 34699],  # pectoralis major (parts)
    "clavicle_left":  [34681, 33587, 34691],  # deltoid clav L, trapezius desc L, pec maj clav L
    "clavicle_right": [34680, 33586, 34690],
    "scapula_left":   [33583, 33585, 34683, 34685, 32545, 32548, 13039],  # trapeze asc/trans L, deltoid acr/spin L, supra/infraspinatus L
    "scapula_right":  [33581, 33584, 34682, 34684, 32544, 32547, 13039],
    "humerus_left":   [34683, 34685, 34681, 37685, 37687, 37700, 37696],  # deltoid + biceps short/long + triceps long/med L
    "humerus_right":  [34682, 34684, 34680, 37684, 37686, 37699, 37695],
    # --- Avant-bras droit ---
    "radius_right": [
        38486,  # right brachioradialis
        38513,  # right supinator
        38460,  # right flexor carpi radialis
        38560,  # humeral head of right pronator teres
        38495,  # right extensor carpi radialis longus
        38498,  # right extensor carpi radialis brevis
    ],
    # --- Avant-bras gauche ---
    "radius_left": [
        38487,  # left brachioradialis
        38514,  # left supinator
        38461,  # left flexor carpi radialis
        38561,  # humeral head of left pronator teres
        38496,  # left extensor carpi radialis longus
        38499,  # left extensor carpi radialis brevis
    ],
    "ulna_right": [
        38619,  # ulnar head of right flexor carpi ulnaris
        38507,  # right extensor carpi ulnaris
        38479,  # right flexor digitorum profundus
        37705,  # right anconeus
        38454,  # right pronator quadratus
    ],
    "ulna_left": [
        38620,  # ulnar head of left flexor carpi ulnaris
        38508,  # left extensor carpi ulnaris
        38480,  # left flexor digitorum profundus
        37706,  # left anconeus
        38455,  # left pronator quadratus
    ],
    # --- Ceinture pelvienne ---
    "pelvis": [
        22328, 22329,   # gluteus maximus R / L
        22330, 22331,   # gluteus medius R / L
        22425, 22426,   # tensor fasciae latae R / L
        22322, 22323,   # iliacus R / L
        22342, 22343,   # psoas major R / L
        13336, 13337,   # external oblique R / L
    ],
    "sacrum": [
        22328, 22329,   # gluteus maximus R / L (origine sacrée)
        22340, 22341,   # piriformis R / L (origine S2-S4)
        22740, 22741,   # right / left iliocostalis lumborum (érecteurs)
        22751, 22753,   # right / left longissimus thoracis (érecteurs)
        46443, 46444,   # right / left coccygeus
    ],
    # --- Cuisse droite ---
    "femur_right": [
        38928,  # right rectus femoris
        38930,  # right vastus lateralis
        38932,  # right vastus medialis
        38934,  # right vastus intermedius
        45888,  # long head of right biceps femoris
        45891,  # short head of right biceps femoris
        22358,  # right semitendinosus
        22448,  # right semimembranosus
        22456,  # right adductor longus
        22459,  # right adductor magnus
        43883,  # right gracilis
    ],
    # --- Cuisse gauche ---
    "femur_left": [
        38929,  # left rectus femoris
        38931,  # left vastus lateralis
        38933,  # left vastus medialis
        38935,  # left vastus intermedius
        45889,  # long head of left biceps femoris
        45892,  # short head of left biceps femoris
        22359,  # left semitendinosus
        22449,  # left semimembranosus
        22457,  # left adductor longus
        22460,  # left adductor magnus
        43884,  # left gracilis
    ],
    # --- Jambe droite (tibia) ---
    "tibia_right": [
        22544,  # right tibialis anterior
        22558,  # right soleus
        45957,  # medial head of right gastrocnemius
        45960,  # lateral head of right gastrocnemius
        22591,  # right popliteus
        22358,  # right semitendinosus  (tendon du pes anserinus)
        22354,  # right sartorius       (tendon du pes anserinus)
        43883,  # right gracilis        (tendon du pes anserinus)
    ],
    # --- Jambe gauche (tibia) ---
    "tibia_left": [
        22545,  # left tibialis anterior
        22559,  # left soleus
        45958,  # medial head of left gastrocnemius
        45961,  # lateral head of left gastrocnemius
        22592,  # left popliteus
        22359,  # left semitendinosus  (tendon du pes anserinus)
        22355,  # left sartorius       (tendon du pes anserinus)
        43884,  # left gracilis        (tendon du pes anserinus)
    ],
    # --- Fibula droite ---
    "fibula_right": [
        22552,  # right fibularis longus
        22554,  # right fibularis brevis
        22548,  # right extensor digitorum longus
        65014,  # right flexor hallucis longus
        22550,  # right fibularis tertius
    ],
    # --- Fibula gauche ---
    "fibula_left": [
        22553,  # left fibularis longus
        22555,  # left fibularis brevis
        22549,  # left extensor digitorum longus
        65015,  # left flexor hallucis longus
        22551,  # left fibularis tertius
    ],
}

# Regex to extract FMA concept ID from OBJ header comments
# Line format: # Concept ID : FMA34682
_CONCEPT_RE = re.compile(r"#\s*Concept ID\s*:\s*FMA(\d+)", re.IGNORECASE)


def find_bp3d_dir() -> Path | None:
    """Cherche le dossier BP3D dans les emplacements communs."""
    candidates = [
        Path.home() / "Downloads" / "isa_BP3D_4.0_obj_99" / "isa_BP3D_4.0_obj_99",
        Path.home() / "Downloads" / "isa_BP3D_4.0_obj_99",
        Path("C:/Users") / Path.home().name / "Downloads" / "isa_BP3D_4.0_obj_99" / "isa_BP3D_4.0_obj_99",
    ]
    for p in candidates:
        if p.exists() and any(p.glob("FJ*.obj")):
            return p
    return None


def build_muscle_index(bp3d_dir: Path) -> dict[int, Path]:
    """Construit le mapping FMA concept ID → chemin OBJ en scannant les en-têtes.

    Lit uniquement les premières lignes de chaque fichier OBJ pour
    extraire le Concept ID sans charger tout le fichier.
    """
    index: dict[int, Path] = {}
    for obj_path in sorted(bp3d_dir.glob("FJ*.obj")):
        try:
            with obj_path.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= 20:
                        # Headers are in the first ~5 lines; bail out early
                        break
                    m = _CONCEPT_RE.match(line)
                    if m:
                        fma_id = int(m.group(1))
                        # Keep the first occurrence only (non-mirrored preferred)
                        if fma_id not in index:
                            index[fma_id] = obj_path
                        break
        except OSError:
            continue
    return index


def build_muscle_name_index(muscle_index: dict[int, Path]) -> dict[int, str]:
    """Extrait les noms anglais depuis les en-têtes des OBJ déjà indexés.

    Parameters
    ----------
    muscle_index:
        Index pré-construit par ``build_muscle_index`` (FMA ID → chemin OBJ).
    """
    _NAME_RE = re.compile(r"#\s*English name\s*:\s*(.+)", re.IGNORECASE)
    names: dict[int, str] = {}
    for fma_id, obj_path in muscle_index.items():
        try:
            with obj_path.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= 20:
                        break
                    m = _NAME_RE.match(line)
                    if m:
                        names[fma_id] = m.group(1).strip()
                        break
        except OSError:
            continue
    return names


def get_muscle_paths(
    bone_stem: str,
    bp3d_dir: Path,
    muscle_index: dict[int, Path] | None = None,
) -> list[Path]:
    """Retourne les chemins OBJ des muscles associés à un os.

    Parameters
    ----------
    bone_stem:
        Identifiant de l'os (clé de LANDMARK_BONE / BONE_LABEL_FR).
    bp3d_dir:
        Dossier racine de la base BodyParts3D (contient les FJ*.obj).
    muscle_index:
        Index pré-construit par ``build_muscle_index``.  Si None, l'index
        est reconstruit à la volée (lent — à éviter dans les boucles).
    """
    if muscle_index is None:
        muscle_index = build_muscle_index(bp3d_dir)
    fma_ids = BONE_MUSCLES.get(bone_stem, [])
    paths: list[Path] = []
    for fma_id in fma_ids:
        p = muscle_index.get(fma_id)
        if p is not None:
            paths.append(p)
    return paths
