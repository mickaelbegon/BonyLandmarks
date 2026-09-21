"""Recettes de construction guidée des 7 repères locaux ISB.

Chaque recette est une ``list[Step]`` (voir :mod:`isb_step_engine`) qui conduit
l'étudiant, étape par étape, de la palpation des landmarks au repère orthonormé
direct du segment, conformément à Wu et al. 2002 (membre inférieur, rachis) et
Wu et al. 2005 (ceinture scapulaire, membre supérieur).

Conventions
-----------
* **Repère ISB local** : X antérieur, Y supérieur (proximal), Z vers la droite
  du sujet ; trièdre direct, donc ``X = Y × Z``, ``Y = Z × X``, ``Z = X × Y``.
* **Repère global GLB** (coordonnées brutes des landmarks) : X droite, Y haut,
  −Z antérieur (convention Blender).  Les recettes ne s'appuient jamais sur ces
  axes globaux : tous les axes ISB sont déduits des seuls landmarks du segment,
  ce qui rend la construction indépendante de la pose du scan.
* Les produits vectoriels sont **ordonnés** : inverser A × B retourne l'axe.
  Le moteur détecte cette erreur et l'explique.
* L'orthogonalisation se fait par double produit vectoriel plutôt que par
  projection, afin que chaque étape reste une opération que l'étudiant peut
  exécuter à la main : ``X_raw`` donne d'abord l'axe secondaire, qui redonne
  ensuite l'axe primaire orthogonalisé.

Correspondance des codes
------------------------
Les codes ISB classiques sont mappés sur les codes BonyLandmarks :

==========  ==================================
ISB         BonyLandmarks
==========  ==================================
IJ          ``suprasternal_notch``
C7          ``C7_spinous``
PX          ``xiphoid_process``
T8          ``T8_spinous``
AA          ``acromial_angle_right``
AI          ``scapula_inferior_angle_right``
TS          ``scapula_trigonum_right``
SC          ``sternoclavicular_joint_right``
AC          ``acromioclavicular_joint_right``
GH          ``greater_tubercle_right`` (proxy palpable)
EL / EM     ``lateral_epicondyle_right`` / ``medial_epicondyle_right``
ASIS/PSIS   ``ASIS_right|left`` / ``PSIS_right|left``
HJC         ``greater_trochanter_right`` (proxy palpable)
LC / MC     ``lateral_knee_right`` / ``medial_knee_right``
LM / MM     ``lateral_malleolus_right`` / ``medial_malleolus_right``
==========  ==================================

Proxies assumés (identiques à :mod:`isb_exercise`) : le centre gléno-huméral et
le centre articulaire de hanche ne sont pas palpables ; le point
inter-condylaire tibial IC est approché par le milieu des condyles du genou.
"""

from __future__ import annotations

import numpy as np

from .isb_step_engine import (
    ComputeCrossProduct,
    ComputeMidpoint,
    ComputeVector,
    PickLandmark,
    StepEngine,
    Step,
    ValidateFrame,
)

__all__ = [
    "ISB_RECIPES",
    "ISB_RECIPE_META",
    "RECIPE_KEYS",
    "get_recipe",
    "recipe_label",
    "required_landmarks",
    "make_engine",
]


# ─────────────────────────────────────────────────────────────────────────────
# Bassin — Wu et al. 2002
# ─────────────────────────────────────────────────────────────────────────────

_PELVIS: list[Step] = [
    PickLandmark(
        instruction_fr="Sélectionne l'épine iliaque antéro-supérieure DROITE (EIAS D).",
        instruction_en="Select the RIGHT anterior superior iliac spine (ASIS R).",
        expected_code="ASIS_right",
        result_name="ASIS_R",
        hint_fr="Suis la crête iliaque vers l'avant jusqu'à sa saillie antérieure la plus marquée.",
        hint_en="Follow the iliac crest forward to its most prominent anterior bump.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'épine iliaque antéro-supérieure GAUCHE (EIAS G).",
        instruction_en="Select the LEFT anterior superior iliac spine (ASIS L).",
        expected_code="ASIS_left",
        result_name="ASIS_L",
        hint_fr="Le symétrique exact de l'EIAS droite, de l'autre côté du bassin.",
        hint_en="The exact mirror of the right ASIS, on the other side of the pelvis.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'épine iliaque postéro-supérieure DROITE (EIPS D).",
        instruction_en="Select the RIGHT posterior superior iliac spine (PSIS R).",
        expected_code="PSIS_right",
        result_name="PSIS_R",
        hint_fr="Sous la fossette cutanée de Vénus, à l'extrémité postérieure de la crête iliaque.",
        hint_en="Under the dimple of Venus, at the posterior end of the iliac crest.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'épine iliaque postéro-supérieure GAUCHE (EIPS G).",
        instruction_en="Select the LEFT posterior superior iliac spine (PSIS L).",
        expected_code="PSIS_left",
        result_name="PSIS_L",
    ),
    ComputeMidpoint(
        instruction_fr="Calcule l'origine O = milieu des deux EIAS.",
        instruction_en="Compute the origin O = midpoint of both ASIS.",
        expected_a="ASIS_R",
        expected_b="ASIS_L",
        result_name="O",
        hint_fr=(
            "L'ISB place l'origine au centre articulaire de hanche, non palpable : "
            "le milieu des EIAS en est le substitut pratique."
        ),
        hint_en=(
            "ISB places the origin at the hip joint centre, which is not palpable: "
            "the ASIS midpoint is the practical substitute."
        ),
    ),
    ComputeMidpoint(
        instruction_fr="Calcule le point postérieur PSIS_mid = milieu des deux EIPS.",
        instruction_en="Compute the posterior point PSIS_mid = midpoint of both PSIS.",
        expected_a="PSIS_R",
        expected_b="PSIS_L",
        result_name="PSIS_mid",
    ),
    ComputeVector(
        instruction_fr="Construis l'axe Z (médio-latéral, vers la droite) : EIAS G → EIAS D.",
        instruction_en="Build the Z axis (medio-lateral, pointing right): ASIS L → ASIS R.",
        expected_from="ASIS_L",
        expected_to="ASIS_R",
        result_name="Z",
        hint_fr="Z est l'axe PRIMAIRE du bassin : il est directement porté par la ligne bi-EIAS.",
        hint_en="Z is the pelvis PRIMARY axis: it lies directly along the bi-ASIS line.",
    ),
    ComputeVector(
        instruction_fr="Construis la direction antéro-postérieure brute : PSIS_mid → O.",
        instruction_en="Build the raw antero-posterior direction: PSIS_mid → O.",
        expected_from="PSIS_mid",
        expected_to="O",
        result_name="AP_raw",
        hint_fr="Elle pointe vers l'avant mais n'est pas encore perpendiculaire à Z.",
        hint_en="It points forward but is not yet perpendicular to Z.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe Y (supérieur) = Z × AP_raw.",
        instruction_en="Compute the Y axis (superior) = Z × AP_raw.",
        expected_a="Z",
        expected_b="AP_raw",
        result_name="Y",
        hint_fr="droite × avant = haut. Y est ainsi exactement perpendiculaire à Z.",
        hint_en="right × forward = up. Y is then exactly perpendicular to Z.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe X (antérieur) = Y × Z.",
        instruction_en="Compute the X axis (anterior) = Y × Z.",
        expected_a="Y",
        expected_b="Z",
        result_name="X",
        hint_fr="haut × droite = avant. X est la version orthogonalisée de AP_raw.",
        hint_en="up × right = forward. X is the orthogonalised version of AP_raw.",
    ),
    ValidateFrame(
        instruction_fr="Valide le repère pelvien (origine O, X avant, Y haut, Z droite).",
        instruction_en="Validate the pelvic frame (origin O, X forward, Y up, Z right).",
        origin_name="O",
        x_name="X",
        y_name="Y",
        z_name="Z",
        segment_key="pelvis",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Thorax — Wu et al. 2005
# ─────────────────────────────────────────────────────────────────────────────

_THORAX: list[Step] = [
    PickLandmark(
        instruction_fr="Sélectionne l'incisure jugulaire (IJ, fourchette sternale).",
        instruction_en="Select the suprasternal notch (IJ).",
        expected_code="suprasternal_notch",
        result_name="IJ",
        hint_fr="Le creux en U au sommet du manubrium, entre les deux clavicules.",
        hint_en="The U-shaped hollow at the top of the manubrium, between both clavicles.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne le processus épineux de C7 (vertebra prominens).",
        instruction_en="Select the C7 spinous process (vertebra prominens).",
        expected_code="C7_spinous",
        result_name="C7",
        hint_fr=(
            "L'épineuse la plus saillante à la base du cou : elle s'efface en extension "
            "cervicale, contrairement à T1."
        ),
        hint_en=(
            "The most prominent spinous process at the base of the neck: it recedes on "
            "cervical extension, unlike T1."
        ),
    ),
    PickLandmark(
        instruction_fr="Sélectionne le processus xiphoïde (PX).",
        instruction_en="Select the xiphoid process (PX).",
        expected_code="xiphoid_process",
        result_name="PX",
        hint_fr="La pointe cartilagineuse à l'extrémité inférieure du sternum.",
        hint_en="The cartilaginous tip at the lower end of the sternum.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne le processus épineux de T8.",
        instruction_en="Select the T8 spinous process.",
        expected_code="T8_spinous",
        result_name="T8",
        hint_fr="Compte les épineuses depuis C7 ; T8 est approximativement en regard du xiphoïde.",
        hint_en="Count spinous processes down from C7; T8 roughly faces the xiphoid.",
    ),
    ComputeMidpoint(
        instruction_fr="Calcule le point crânial IJ_C7_mid = milieu de IJ et C7.",
        instruction_en="Compute the cranial point IJ_C7_mid = midpoint of IJ and C7.",
        expected_a="IJ",
        expected_b="C7",
        result_name="IJ_C7_mid",
    ),
    ComputeMidpoint(
        instruction_fr="Calcule le point caudal = milieu de PX et T8.",
        instruction_en="Compute the caudal point = midpoint of PX and T8.",
        expected_a="PX",
        expected_b="T8",
        result_name="caudal",
    ),
    ComputeVector(
        instruction_fr="Construis l'axe Y (supérieur) : caudal → IJ_C7_mid.",
        instruction_en="Build the Y axis (superior): caudal → IJ_C7_mid.",
        expected_from="caudal",
        expected_to="IJ_C7_mid",
        result_name="Y",
        hint_fr="Y est l'axe PRIMAIRE du thorax : il relie les deux milieux vertébro-sternaux.",
        hint_en="Y is the thorax PRIMARY axis: it joins both sterno-vertebral midpoints.",
    ),
    ComputeVector(
        instruction_fr="Construis V1 : caudal → IJ.",
        instruction_en="Build V1: caudal → IJ.",
        expected_from="caudal",
        expected_to="IJ",
        result_name="V1",
    ),
    ComputeVector(
        instruction_fr="Construis V2 : caudal → C7.",
        instruction_en="Build V2: caudal → C7.",
        expected_from="caudal",
        expected_to="C7",
        result_name="V2",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule Z_raw = V1 × V2, normale au plan sagittal (IJ, C7, caudal).",
        instruction_en="Compute Z_raw = V1 × V2, normal to the sagittal plane (IJ, C7, caudal).",
        expected_a="V1",
        expected_b="V2",
        result_name="Z_raw",
        hint_fr=(
            "V1 pointe vers l'avant-haut, V2 vers l'arrière-haut : leur produit vectoriel "
            "sort du plan sagittal vers la droite du sujet."
        ),
        hint_en=(
            "V1 points up-forward, V2 up-backward: their cross product leaves the sagittal "
            "plane towards the subject's right."
        ),
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe X (antérieur) = Y × Z_raw.",
        instruction_en="Compute the X axis (anterior) = Y × Z_raw.",
        expected_a="Y",
        expected_b="Z_raw",
        result_name="X",
        hint_fr="haut × droite = avant. X est ainsi perpendiculaire à Y par construction.",
        hint_en="up × right = forward. X is therefore perpendicular to Y by construction.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe Z (droite) orthogonalisé = X × Y.",
        instruction_en="Compute the orthogonalised Z axis (right) = X × Y.",
        expected_a="X",
        expected_b="Y",
        result_name="Z",
        hint_fr="avant × haut = droite. Z est la version de Z_raw rendue perpendiculaire à Y.",
        hint_en="forward × up = right. Z is Z_raw made perpendicular to Y.",
    ),
    ValidateFrame(
        instruction_fr="Valide le repère thoracique (origine IJ, X avant, Y haut, Z droite).",
        instruction_en="Validate the thoracic frame (origin IJ, X forward, Y up, Z right).",
        origin_name="IJ",
        x_name="X",
        y_name="Y",
        z_name="Z",
        segment_key="thorax",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Scapula droite — Wu et al. 2005
# ─────────────────────────────────────────────────────────────────────────────

_SCAPULA_R: list[Step] = [
    PickLandmark(
        instruction_fr="Sélectionne l'angle acromial droit (AA).",
        instruction_en="Select the right acromial angle (AA).",
        expected_code="acromial_angle_right",
        result_name="AA",
        tolerance_ok=["acromion_right"],
        hint_fr=(
            "Le coin postéro-latéral de l'acromion, là où l'épine de la scapula devient "
            "le bord latéral de l'acromion — pas le sommet de l'acromion."
        ),
        hint_en=(
            "The postero-lateral corner of the acromion, where the scapular spine turns "
            "into the lateral acromial border — not the acromion tip."
        ),
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'angle inférieur de la scapula droite (AI).",
        instruction_en="Select the right inferior angle of the scapula (AI).",
        expected_code="scapula_inferior_angle_right",
        result_name="AI",
        hint_fr="La pointe inférieure de l'omoplate, vers T7-T8 bras le long du corps.",
        hint_en="The lower tip of the shoulder blade, around T7-T8 with the arm at rest.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne le trigone de l'épine scapulaire droite (TS).",
        instruction_en="Select the right trigonum spinae (TS).",
        expected_code="scapula_trigonum_right",
        result_name="TS",
        hint_fr="Le triangle lisse à la jonction entre l'épine de la scapula et son bord médial.",
        hint_en="The smooth triangle where the scapular spine meets the medial border.",
    ),
    ComputeVector(
        instruction_fr="Construis l'axe Z (latéral) : TS → AA.",
        instruction_en="Build the Z axis (lateral): TS → AA.",
        expected_from="TS",
        expected_to="AA",
        result_name="Z",
        hint_fr="Z est l'axe PRIMAIRE de la scapula : il suit l'épine, de médial vers latéral.",
        hint_en="Z is the scapula PRIMARY axis: it follows the spine, medial to lateral.",
    ),
    ComputeVector(
        instruction_fr="Construis V_AI : AA → AI.",
        instruction_en="Build V_AI: AA → AI.",
        expected_from="AA",
        expected_to="AI",
        result_name="V_AI",
    ),
    ComputeVector(
        instruction_fr="Construis V_TS : AA → TS.",
        instruction_en="Build V_TS: AA → TS.",
        expected_from="AA",
        expected_to="TS",
        result_name="V_TS",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule X_raw = V_AI × V_TS, normale au plan scapulaire (AA, AI, TS).",
        instruction_en="Compute X_raw = V_AI × V_TS, normal to the scapular plane (AA, AI, TS).",
        expected_a="V_AI",
        expected_b="V_TS",
        result_name="X_raw",
        hint_fr=(
            "Cet ordre fait sortir la normale vers l'AVANT du thorax (la scapula est "
            "plaquée sur le gril costal) ; l'ordre inverse la ferait pointer en arrière."
        ),
        hint_en=(
            "This order makes the normal point ANTERIORLY (the scapula lies on the rib "
            "cage); the reverse order would send it backwards."
        ),
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe Y (supérieur) = Z × X_raw.",
        instruction_en="Compute the Y axis (superior) = Z × X_raw.",
        expected_a="Z",
        expected_b="X_raw",
        result_name="Y",
        hint_fr="latéral × avant = haut. Y est perpendiculaire à Z par construction.",
        hint_en="lateral × forward = up. Y is perpendicular to Z by construction.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe X (antérieur) orthogonalisé = Y × Z.",
        instruction_en="Compute the orthogonalised X axis (anterior) = Y × Z.",
        expected_a="Y",
        expected_b="Z",
        result_name="X",
        hint_fr="haut × latéral = avant. X est la version de X_raw rendue perpendiculaire à Z.",
        hint_en="up × lateral = forward. X is X_raw made perpendicular to Z.",
    ),
    ValidateFrame(
        instruction_fr="Valide le repère scapulaire droit (origine AA, X avant, Y haut, Z latéral).",
        instruction_en="Validate the right scapular frame (origin AA, X forward, Y up, Z lateral).",
        origin_name="AA",
        x_name="X",
        y_name="Y",
        z_name="Z",
        segment_key="scapula_right",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Clavicule droite — Wu et al. 2005
# ─────────────────────────────────────────────────────────────────────────────
# L'axe X claviculaire n'est pas déductible des seuls SC et AC : la clavicule
# n'a que deux repères palpables, donc un degré de liberté (rotation axiale)
# reste indéterminé.  L'ISB lève l'ambiguïté avec le Y thoracique.  Les étapes
# marquées ``auto=True`` reconstruisent ce Y_thorax sans intervention de
# l'étudiant (il l'a déjà construit dans la recette « thorax »).

_CLAVICLE_R: list[Step] = [
    PickLandmark(expected_code="suprasternal_notch", result_name="IJ", auto=True,
                 instruction_fr="(auto) Incisure jugulaire, reprise du repère thoracique.",
                 instruction_en="(auto) Suprasternal notch, reused from the thoracic frame."),
    PickLandmark(expected_code="C7_spinous", result_name="C7", auto=True,
                 instruction_fr="(auto) C7, reprise du repère thoracique.",
                 instruction_en="(auto) C7, reused from the thoracic frame."),
    PickLandmark(expected_code="xiphoid_process", result_name="PX", auto=True,
                 instruction_fr="(auto) Processus xiphoïde, reprise du repère thoracique.",
                 instruction_en="(auto) Xiphoid process, reused from the thoracic frame."),
    PickLandmark(expected_code="T8_spinous", result_name="T8", auto=True,
                 instruction_fr="(auto) T8, reprise du repère thoracique.",
                 instruction_en="(auto) T8, reused from the thoracic frame."),
    ComputeMidpoint(expected_a="IJ", expected_b="C7", result_name="IJ_C7_mid", auto=True,
                    instruction_fr="(auto) Milieu IJ-C7.",
                    instruction_en="(auto) IJ-C7 midpoint."),
    ComputeMidpoint(expected_a="PX", expected_b="T8", result_name="caudal", auto=True,
                    instruction_fr="(auto) Milieu PX-T8.",
                    instruction_en="(auto) PX-T8 midpoint."),
    ComputeVector(expected_from="caudal", expected_to="IJ_C7_mid", result_name="Y_thorax",
                  auto=True,
                  instruction_fr="(auto) Axe Y thoracique, fourni comme référence verticale.",
                  instruction_en="(auto) Thoracic Y axis, provided as the vertical reference."),
    PickLandmark(
        instruction_fr="Sélectionne l'articulation sterno-claviculaire droite (SC).",
        instruction_en="Select the right sternoclavicular joint (SC).",
        expected_code="sternoclavicular_joint_right",
        result_name="SC",
        hint_fr="L'extrémité médiale renflée de la clavicule, juste latérale à l'incisure jugulaire.",
        hint_en="The bulging medial end of the clavicle, just lateral to the suprasternal notch.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'articulation acromio-claviculaire droite (AC).",
        instruction_en="Select the right acromioclavicular joint (AC).",
        expected_code="acromioclavicular_joint_right",
        result_name="AC",
        hint_fr="Suis la clavicule vers le dehors jusqu'au petit ressaut contre l'acromion.",
        hint_en="Follow the clavicle laterally to the small step against the acromion.",
    ),
    ComputeVector(
        instruction_fr="Construis l'axe Z (latéral) : SC → AC.",
        instruction_en="Build the Z axis (lateral): SC → AC.",
        expected_from="SC",
        expected_to="AC",
        result_name="Z",
        hint_fr="Z est l'axe PRIMAIRE de la clavicule : il suit sa longueur, de médial à latéral.",
        hint_en="Z is the clavicle PRIMARY axis: it runs along its length, medial to lateral.",
    ),
    ComputeCrossProduct(
        instruction_fr=(
            "Calcule l'axe X (antérieur) = Y_thorax × Z. La clavicule n'ayant que deux "
            "repères palpables, l'ISB emprunte la verticale au thorax."
        ),
        instruction_en=(
            "Compute the X axis (anterior) = Y_thorax × Z. With only two palpable "
            "landmarks on the clavicle, ISB borrows the vertical from the thorax."
        ),
        expected_a="Y_thorax",
        expected_b="Z",
        result_name="X",
        hint_fr="haut × latéral = avant.",
        hint_en="up × lateral = forward.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe Y (supérieur) = Z × X.",
        instruction_en="Compute the Y axis (superior) = Z × X.",
        expected_a="Z",
        expected_b="X",
        result_name="Y",
        hint_fr=(
            "latéral × avant = haut. Ce Y claviculaire diffère légèrement du Y thoracique : "
            "il est rendu perpendiculaire à l'axe de la clavicule."
        ),
        hint_en=(
            "lateral × forward = up. This clavicular Y differs slightly from the thoracic "
            "Y: it is made perpendicular to the clavicle axis."
        ),
    ),
    ValidateFrame(
        instruction_fr="Valide le repère claviculaire droit (origine SC, X avant, Y haut, Z latéral).",
        instruction_en="Validate the right clavicular frame (origin SC, X forward, Y up, Z lateral).",
        origin_name="SC",
        x_name="X",
        y_name="Y",
        z_name="Z",
        segment_key="clavicle_right",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Humérus droit — Wu et al. 2005
# ─────────────────────────────────────────────────────────────────────────────

_HUMERUS_R: list[Step] = [
    PickLandmark(
        instruction_fr=(
            "Sélectionne le tubercule majeur droit — proxy palpable du centre "
            "gléno-huméral GH, qui n'est pas accessible à la palpation."
        ),
        instruction_en=(
            "Select the right greater tubercle — palpable proxy for the glenohumeral "
            "centre GH, which cannot be palpated."
        ),
        expected_code="greater_tubercle_right",
        result_name="GH",
        hint_fr=(
            "Juste en dehors et sous l'acromion, bras pendant. L'ISB recommande une "
            "régression (Meskers 1998) ou les axes hélicoïdaux (Stokdijk 2000)."
        ),
        hint_en=(
            "Just lateral and below the acromion, arm hanging. ISB recommends a "
            "regression (Meskers 1998) or helical axes (Stokdijk 2000)."
        ),
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'épicondyle latéral droit (EL).",
        instruction_en="Select the right lateral epicondyle (EL).",
        expected_code="lateral_epicondyle_right",
        result_name="EL",
        hint_fr="La saillie osseuse externe du coude, au-dessus de la tête radiale.",
        hint_en="The outer bony bump of the elbow, above the radial head.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'épicondyle médial droit (EM).",
        instruction_en="Select the right medial epicondyle (EM).",
        expected_code="medial_epicondyle_right",
        result_name="EM",
        hint_fr="La saillie interne du coude, où passe le nerf ulnaire.",
        hint_en="The inner bump of the elbow, where the ulnar nerve runs.",
    ),
    ComputeMidpoint(
        instruction_fr="Calcule elbow_mid = milieu des deux épicondyles.",
        instruction_en="Compute elbow_mid = midpoint of both epicondyles.",
        expected_a="EL",
        expected_b="EM",
        result_name="elbow_mid",
    ),
    ComputeVector(
        instruction_fr="Construis l'axe Y (proximal) : elbow_mid → GH.",
        instruction_en="Build the Y axis (proximal): elbow_mid → GH.",
        expected_from="elbow_mid",
        expected_to="GH",
        result_name="Y",
        hint_fr="Y est l'axe PRIMAIRE de l'humérus : l'axe long, dirigé vers l'épaule.",
        hint_en="Y is the humerus PRIMARY axis: the long axis, pointing to the shoulder.",
    ),
    ComputeVector(
        instruction_fr="Construis EL_vec : EM → EL (médial vers latéral).",
        instruction_en="Build EL_vec: EM → EL (medial to lateral).",
        expected_from="EM",
        expected_to="EL",
        result_name="EL_vec",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe X (antérieur) = Y × EL_vec, normale au plan (GH, EL, EM).",
        instruction_en="Compute the X axis (anterior) = Y × EL_vec, normal to plane (GH, EL, EM).",
        expected_a="Y",
        expected_b="EL_vec",
        result_name="X",
        hint_fr="proximal × latéral = avant, pour le côté DROIT.",
        hint_en="proximal × lateral = forward, for the RIGHT side.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe Z (latéral) = X × Y.",
        instruction_en="Compute the Z axis (lateral) = X × Y.",
        expected_a="X",
        expected_b="Y",
        result_name="Z",
        hint_fr="avant × haut = droite. Z est EL_vec rendu perpendiculaire à l'axe long.",
        hint_en="forward × up = right. Z is EL_vec made perpendicular to the long axis.",
    ),
    ValidateFrame(
        instruction_fr="Valide le repère huméral droit (origine GH, X avant, Y proximal, Z latéral).",
        instruction_en="Validate the right humeral frame (origin GH, X forward, Y proximal, Z lateral).",
        origin_name="GH",
        x_name="X",
        y_name="Y",
        z_name="Z",
        segment_key="humerus_right",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Fémur droit — Wu et al. 2002
# ─────────────────────────────────────────────────────────────────────────────

_FEMUR_R: list[Step] = [
    PickLandmark(
        instruction_fr=(
            "Sélectionne le grand trochanter droit — proxy palpable du centre "
            "articulaire de hanche."
        ),
        instruction_en=(
            "Select the right greater trochanter — palpable proxy for the hip joint centre."
        ),
        expected_code="greater_trochanter_right",
        result_name="GT",
        hint_fr=(
            "La large saillie latérale de la hanche ; elle roule sous les doigts en "
            "rotation interne/externe de la cuisse."
        ),
        hint_en=(
            "The broad lateral bump of the hip; it rolls under the fingers during "
            "internal/external thigh rotation."
        ),
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'épicondyle fémoral latéral droit.",
        instruction_en="Select the right lateral femoral epicondyle.",
        expected_code="lateral_knee_right",
        result_name="lat_knee",
        hint_fr="La saillie externe du genou, au-dessus de l'interligne articulaire.",
        hint_en="The outer bump of the knee, above the joint line.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne l'épicondyle fémoral médial droit.",
        instruction_en="Select the right medial femoral epicondyle.",
        expected_code="medial_knee_right",
        result_name="med_knee",
        hint_fr="La saillie interne du genou, en regard de la latérale.",
        hint_en="The inner bump of the knee, facing the lateral one.",
    ),
    ComputeMidpoint(
        instruction_fr="Calcule knee_mid = milieu des deux épicondyles fémoraux.",
        instruction_en="Compute knee_mid = midpoint of both femoral epicondyles.",
        expected_a="lat_knee",
        expected_b="med_knee",
        result_name="knee_mid",
    ),
    ComputeVector(
        instruction_fr="Construis l'axe Y (proximal) : knee_mid → GT.",
        instruction_en="Build the Y axis (proximal): knee_mid → GT.",
        expected_from="knee_mid",
        expected_to="GT",
        result_name="Y",
        hint_fr="Y est l'axe PRIMAIRE du fémur : l'axe long, du genou vers la hanche.",
        hint_en="Y is the femur PRIMARY axis: the long axis, knee to hip.",
    ),
    ComputeVector(
        instruction_fr="Construis V_lat : épicondyle médial → épicondyle latéral.",
        instruction_en="Build V_lat: medial → lateral epicondyle.",
        expected_from="med_knee",
        expected_to="lat_knee",
        result_name="V_lat",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe X (antérieur) = Y × V_lat, normale au plan (GT, EL, EM).",
        instruction_en="Compute the X axis (anterior) = Y × V_lat, normal to plane (GT, EL, EM).",
        expected_a="Y",
        expected_b="V_lat",
        result_name="X",
        hint_fr="proximal × latéral = avant, pour le côté DROIT.",
        hint_en="proximal × lateral = forward, for the RIGHT side.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe Z (latéral) = X × Y.",
        instruction_en="Compute the Z axis (lateral) = X × Y.",
        expected_a="X",
        expected_b="Y",
        result_name="Z",
        hint_fr="avant × proximal = droite. Z est V_lat rendu perpendiculaire à l'axe long.",
        hint_en="forward × proximal = right. Z is V_lat made perpendicular to the long axis.",
    ),
    ValidateFrame(
        instruction_fr="Valide le repère fémoral droit (origine GT, X avant, Y proximal, Z latéral).",
        instruction_en="Validate the right femoral frame (origin GT, X forward, Y proximal, Z lateral).",
        origin_name="GT",
        x_name="X",
        y_name="Y",
        z_name="Z",
        segment_key="femur_right",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tibia droit — Wu et al. 2002
# ─────────────────────────────────────────────────────────────────────────────
# Wu 2002 définit IC à partir des bords des condyles TIBIAUX ; le projet ne
# dispose que des épicondyles fémoraux, utilisés ici comme proxy de IC.

_TIBIA_R: list[Step] = [
    PickLandmark(
        instruction_fr="Sélectionne la malléole latérale droite (LM).",
        instruction_en="Select the right lateral malleolus (LM).",
        expected_code="lateral_malleolus_right",
        result_name="LM",
        hint_fr="La saillie externe de la cheville ; elle descend plus bas que la médiale.",
        hint_en="The outer ankle bump; it sits lower than the medial one.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne la malléole médiale droite (MM).",
        instruction_en="Select the right medial malleolus (MM).",
        expected_code="medial_malleolus_right",
        result_name="MM",
        hint_fr="La saillie interne de la cheville, plus haute et plus antérieure que la latérale.",
        hint_en="The inner ankle bump, higher and more anterior than the lateral one.",
    ),
    PickLandmark(
        instruction_fr="Sélectionne le condyle latéral du genou droit (proxy du point IC).",
        instruction_en="Select the right lateral knee condyle (proxy for point IC).",
        expected_code="lateral_knee_right",
        result_name="LC",
    ),
    PickLandmark(
        instruction_fr="Sélectionne le condyle médial du genou droit (proxy du point IC).",
        instruction_en="Select the right medial knee condyle (proxy for point IC).",
        expected_code="medial_knee_right",
        result_name="MC",
    ),
    ComputeMidpoint(
        instruction_fr="Calcule l'origine IM = point inter-malléolaire (milieu LM-MM).",
        instruction_en="Compute the origin IM = inter-malleolar point (LM-MM midpoint).",
        expected_a="LM",
        expected_b="MM",
        result_name="IM",
    ),
    ComputeMidpoint(
        instruction_fr="Calcule IC = point inter-condylaire (milieu LC-MC).",
        instruction_en="Compute IC = inter-condylar point (LC-MC midpoint).",
        expected_a="LC",
        expected_b="MC",
        result_name="IC",
    ),
    ComputeVector(
        instruction_fr="Construis l'axe Z (latéral) : MM → LM.",
        instruction_en="Build the Z axis (lateral): MM → LM.",
        expected_from="MM",
        expected_to="LM",
        result_name="Z",
        hint_fr=(
            "Z est l'axe PRIMAIRE du tibia selon Wu 2002 : la ligne malléolaire elle-même, "
            "et non une composante orthogonalisée de l'axe long."
        ),
        hint_en=(
            "Z is the tibia PRIMARY axis per Wu 2002: the malleolar line itself, not an "
            "orthogonalised component of the long axis."
        ),
    ),
    ComputeVector(
        instruction_fr="Construis V_IC : IM → IC (axe long approximatif).",
        instruction_en="Build V_IC: IM → IC (approximate long axis).",
        expected_from="IM",
        expected_to="IC",
        result_name="V_IC",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe X (antérieur) = V_IC × Z, normale au plan de torsion.",
        instruction_en="Compute the X axis (anterior) = V_IC × Z, normal to the torsional plane.",
        expected_a="V_IC",
        expected_b="Z",
        result_name="X",
        hint_fr="proximal × latéral = avant. Le plan (IC, MM, LM) porte la torsion tibiale.",
        hint_en="proximal × lateral = forward. The (IC, MM, LM) plane carries tibial torsion.",
    ),
    ComputeCrossProduct(
        instruction_fr="Calcule l'axe Y (proximal) = Z × X.",
        instruction_en="Compute the Y axis (proximal) = Z × X.",
        expected_a="Z",
        expected_b="X",
        result_name="Y",
        hint_fr=(
            "latéral × avant = haut. Attention : Y n'est PAS exactement IM → IC, car Z est "
            "l'axe primaire et Y lui est rendu perpendiculaire."
        ),
        hint_en=(
            "lateral × forward = up. Note: Y is NOT exactly IM → IC, because Z is the "
            "primary axis and Y is made perpendicular to it."
        ),
    ),
    ValidateFrame(
        instruction_fr="Valide le repère tibial droit (origine IM, X avant, Y proximal, Z latéral).",
        instruction_en="Validate the right tibial frame (origin IM, X forward, Y proximal, Z lateral).",
        origin_name="IM",
        x_name="X",
        y_name="Y",
        z_name="Z",
        segment_key="tibia_right",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Index public
# ─────────────────────────────────────────────────────────────────────────────

#: Recettes indexées par clé de segment (mêmes clés que ``isb_exercise.SEGMENT_KEYS``).
ISB_RECIPES: dict[str, list[Step]] = {
    "thorax": _THORAX,
    "clavicle_right": _CLAVICLE_R,
    "scapula_right": _SCAPULA_R,
    "humerus_right": _HUMERUS_R,
    "pelvis": _PELVIS,
    "femur_right": _FEMUR_R,
    "tibia_right": _TIBIA_R,
}

#: Ordre canonique d'affichage (du proximal/axial vers le distal).
RECIPE_KEYS: list[str] = list(ISB_RECIPES)

#: Métadonnées d'affichage par segment.
ISB_RECIPE_META: dict[str, dict[str, str]] = {
    "thorax": {
        "name_fr": "Thorax", "name_en": "Thorax",
        "reference": "Wu et al. 2005",
    },
    "clavicle_right": {
        "name_fr": "Clavicule D", "name_en": "Right clavicle",
        "reference": "Wu et al. 2005",
    },
    "scapula_right": {
        "name_fr": "Scapula D", "name_en": "Right scapula",
        "reference": "Wu et al. 2005",
    },
    "humerus_right": {
        "name_fr": "Humérus D", "name_en": "Right humerus",
        "reference": "Wu et al. 2005",
    },
    "pelvis": {
        "name_fr": "Bassin", "name_en": "Pelvis",
        "reference": "Wu et al. 2002",
    },
    "femur_right": {
        "name_fr": "Fémur D", "name_en": "Right femur",
        "reference": "Wu et al. 2002",
    },
    "tibia_right": {
        "name_fr": "Tibia D", "name_en": "Right tibia",
        "reference": "Wu et al. 2002",
    },
}


def get_recipe(segment_key: str) -> list[Step]:
    """Recette d'un segment. Lève ``KeyError`` si la clé est inconnue."""
    if segment_key not in ISB_RECIPES:
        raise KeyError(
            f"Segment inconnu : {segment_key!r}. Clés valides : {RECIPE_KEYS}"
        )
    return ISB_RECIPES[segment_key]


def recipe_label(segment_key: str, lang: str = "fr") -> str:
    """Nom lisible d'un segment."""
    meta = ISB_RECIPE_META.get(segment_key, {})
    return meta.get("name_fr" if lang == "fr" else "name_en", segment_key)


def required_landmarks(segment_key: str) -> list[str]:
    """Codes des landmarks nécessaires à une recette, dans l'ordre de la recette."""
    out: list[str] = []
    for step in get_recipe(segment_key):
        if isinstance(step, PickLandmark) and step.expected_code not in out:
            out.append(step.expected_code)
    return out


def make_engine(
    segment_key: str, ground_truth: dict[str, np.ndarray]
) -> StepEngine:
    """Instancie un :class:`~.isb_step_engine.StepEngine` pour un segment."""
    return StepEngine(get_recipe(segment_key), ground_truth, segment_key=segment_key)
