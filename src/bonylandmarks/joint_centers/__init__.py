"""Estimation des centres articulaires non palpables (hanche, gléno-humérale).

Les centres articulaires ne correspondent à aucun repère osseux palpable. Ce
package regroupe les **régressions prédictives** qui les estiment à partir de
repères de surface, chacune appliquée dans son repère anatomique local :

- Hanche (HJC) : Bell/Brand (1990), Harrington (2007) bassin seul, Harrington
  (2007) avec longueur de jambe.
- Épaule (GHJC) : Meskers (1998) avec processus coracoïde, Sobral (2025) sans
  coracoïde mais avec covariables anthropométriques.

Point clé méthodologique
------------------------
Les équations de régression ne sont **pas** valides dans le repère global : il
faut construire le repère anatomique (bassin ou scapula), exprimer les repères
osseux dedans, appliquer les coefficients, puis reconvertir le résultat en
coordonnées globales. C'est ce que font toutes les fonctions de ce package —
les résultats sont donc invariants par translation et par rotation rigide.

Unités
------
Les régressions de Harrington, Meskers et Sobral comportent des termes
constants exprimés en **millimètres**. Les fonctions bas niveau
(``estimate_hjc_*``, ``estimate_ghjc_*``) supposent donc des coordonnées en mm.
Les fonctions haut niveau acceptent ``units="mm"`` (défaut), ``"cm"`` ou
``"m"`` — cette dernière correspond à la convention des fichiers GLB/glTF
utilisés par BonyLandmarks — et rendent le résultat dans l'unité d'entrée.

Examples
--------
>>> import numpy as np
>>> from bonylandmarks.joint_centers import estimate_hip_joint_centers
>>> lm = {"RASI": np.array([95., 985., -40.]),
...       "LASI": np.array([-95., 985., -40.]),
...       "RPSI": np.array([70., 985., 60.]),
...       "LPSI": np.array([-70., 985., 60.])}
>>> hjc = estimate_hip_joint_centers(lm, method="harrington")
>>> sorted(hjc)
['left', 'right']
"""
from __future__ import annotations

from .frames import (
    AnatomicalFrame,
    build_pelvis_frame,
    build_scapula_frame,
    pelvis_dimensions,
    scapula_distances,
)
from .pelvis import (
    HJC_METHODS,
    estimate_hip_joint_centers,
    estimate_hjc_bell,
    estimate_hjc_harrington_leg_length,
    estimate_hjc_harrington_pelvis,
)
from .scapula import (
    GHJC_METHODS,
    estimate_ghjc_meskers,
    estimate_ghjc_sobral,
    estimate_shoulder_joint_center,
)

__all__ = [
    # repères anatomiques
    "AnatomicalFrame",
    "build_pelvis_frame",
    "build_scapula_frame",
    "pelvis_dimensions",
    "scapula_distances",
    # hanche
    "HJC_METHODS",
    "estimate_hjc_bell",
    "estimate_hjc_harrington_pelvis",
    "estimate_hjc_harrington_leg_length",
    "estimate_hip_joint_centers",
    # épaule
    "GHJC_METHODS",
    "estimate_ghjc_meskers",
    "estimate_ghjc_sobral",
    "estimate_shoulder_joint_center",
]
