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
    "radius_left":  [],
    "radius_right": [],
    "ulna_left":    [],
    "ulna_right":   [],
    "pelvis":  [],
    "sacrum":  [],
    "femur_left":  [],
    "femur_right": [],
    "tibia_left":  [],
    "tibia_right": [],
    "fibula_left": [],
    "fibula_right": [],
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
