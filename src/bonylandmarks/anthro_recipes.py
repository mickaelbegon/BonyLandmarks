"""Twenty interactive anthropometric recipes for exercice 4.

Each recipe maps a short code to an AnthroRecipe whose steps guide the student
through landmark selection and computation.  Codes mirror the measure codes from
anthro_measures_exercise.py.

Landmark codes are verified against landmarks_extended.py / landmarks.json.
Fallbacks (e.g. tragus → external_acoustic_meatus) are noted in instructions.
"""
from __future__ import annotations

from bonylandmarks.anthro_step_engine import (
    AnthroRecipe,
    PickLandmark,
    ComputeDistance,
    ComputeMidpoint,
    ComputeAngle3Pts,
    ComputeAnglePlane,
    ComputeAsymmetry,
    ComputeAxisAngle,
    ComputeProjection,
)


ANTHRO_RECIPES: dict[str, AnthroRecipe] = {

    # ── 1. Longueur membre inférieur ──────────────────────────────────────────
    "LLL": AnthroRecipe(
        code="LLL",
        measure_code="lower_limb_length",
        name_fr="Longueur membre inférieur",
        unit="mm",
        expected_result_name="LLL",
        interpretation_fr=(
            "La longueur fonctionnelle du membre inférieur est la distance 3D "
            "EIAS → malléole médiale. Norme : 800–1050 mm. "
            "Une asymétrie > 10 mm peut indiquer une inégalité de longueur des membres."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'ÉIAS droite (épine iliaque antéro-supérieure) — "
                    "sailllie antérieure de la crête iliaque."
                ),
                expected_code="ASIS_right",
                result_name="ASIS_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole médiale droite — "
                    "saillie interne de la cheville (extrémité distale du tibia)."
                ),
                expected_code="medial_malleolus_right",
                result_name="MM_R",
            ),
            ComputeDistance(
                instruction_fr=(
                    "Calculez la distance 3D entre l'ÉIAS et la malléole médiale. "
                    "C'est la longueur fonctionnelle du membre inférieur droit."
                ),
                expected_a="ASIS_R",
                expected_b="MM_R",
                result_name="LLL",
            ),
        ],
    ),

    # ── 2. Longueur membre supérieur ──────────────────────────────────────────
    "ULL": AnthroRecipe(
        code="ULL",
        measure_code="upper_limb_length",
        name_fr="Longueur membre supérieur",
        unit="mm",
        expected_result_name="ULL",
        interpretation_fr=(
            "Distance 3D acromion → styloïde radiale. Norme : 550–750 mm. "
            "Mesurée bras le long du corps, elle évalue la longueur fonctionnelle "
            "du membre supérieur."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'acromion droit — "
                    "bord le plus latéral de l'épine de la scapula."
                ),
                expected_code="acromion_right",
                result_name="ACR_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la styloïde radiale droite — "
                    "saillie osseuse du côté pouce au poignet."
                ),
                expected_code="radial_styloid_right",
                result_name="RS_R",
            ),
            ComputeDistance(
                instruction_fr=(
                    "Calculez la distance 3D entre l'acromion et la styloïde radiale. "
                    "C'est la longueur fonctionnelle du membre supérieur droit."
                ),
                expected_a="ACR_R",
                expected_b="RS_R",
                result_name="ULL",
            ),
        ],
    ),

    # ── 3. Bascule pelvienne (Pelvic tilt) ───────────────────────────────────
    "PT": AnthroRecipe(
        code="PT",
        measure_code="pelvic_tilt",
        name_fr="Bascule pelvienne",
        unit="°",
        expected_result_name="pelvic_tilt",
        interpretation_fr=(
            "Angle ÉIAS–ÉIPS par rapport à l'horizontale dans le plan sagittal. "
            "Norme debout : 10–12°. Valeur > 12° = bascule antérieure excessive ; "
            "< 10° = bascule postérieure."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'ÉIAS droite (épine iliaque antéro-supérieure) — "
                    "point de référence antérieur du bassin."
                ),
                expected_code="ASIS_right",
                result_name="ASIS_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'ÉIPS droite (épine iliaque postéro-supérieure) — "
                    "saillie postérieure de la crête iliaque, sous les fossettes de Vénus."
                ),
                expected_code="PSIS_right",
                result_name="PSIS_R",
            ),
            ComputeAnglePlane(
                instruction_fr=(
                    "Calculez l'angle de la ligne ÉIPS→ÉIAS avec l'horizontale "
                    "dans le plan sagittal. Angle positif = bascule antérieure."
                ),
                expected_a="PSIS_R",
                expected_b="ASIS_R",
                plane="horizontal",
                result_name="pelvic_tilt",
            ),
        ],
    ),

    # ── 4. Obliquité pelvienne (Pelvic obliquity) ─────────────────────────────
    "PO": AnthroRecipe(
        code="PO",
        measure_code="pelvic_obliquity",
        name_fr="Obliquité pelvienne",
        unit="mm",
        expected_result_name="pelvic_obliquity",
        interpretation_fr=(
            "Différence de hauteur (axe vertical) entre ÉIAS gauche et droite. "
            "Norme : 0–10 mm. Au-delà = asymétrie pelvienne (inégalité de membres, "
            "scoliose ou déséquilibre musculaire)."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'ÉIAS gauche — "
                    "épine iliaque antéro-supérieure du côté gauche."
                ),
                expected_code="ASIS_left",
                result_name="ASIS_L",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'ÉIAS droite — "
                    "épine iliaque antéro-supérieure du côté droit."
                ),
                expected_code="ASIS_right",
                result_name="ASIS_R",
            ),
            ComputeAsymmetry(
                instruction_fr=(
                    "Calculez la différence de hauteur (composante verticale) "
                    "entre ÉIAS gauche et droite. "
                    "C'est l'obliquité pelvienne."
                ),
                expected_left="ASIS_L",
                expected_right="ASIS_R",
                result_name="pelvic_obliquity",
            ),
        ],
    ),

    # ── 5. Angle Q (patello-fémoral) ──────────────────────────────────────────
    "QA": AnthroRecipe(
        code="QA",
        measure_code="Q_angle",
        name_fr="Angle Q (patello-fémoral)",
        unit="°",
        expected_result_name="Q_angle",
        interpretation_fr=(
            "Angle ÉIAS–patella–tubérosité tibiale dans le plan frontal. "
            "Norme : 10–15° (femme), 8–10° (homme). "
            "Un angle Q élevé augmente le risque de syndrome patello-fémoral."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'ÉIAS droite — "
                    "point proximal de l'axe d'extension du quadriceps."
                ),
                expected_code="ASIS_right",
                result_name="ASIS_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le centre de la patella droite — "
                    "milieu géométrique de la rotule (sommet de l'angle Q)."
                ),
                expected_code="patella_center_right",
                result_name="PAT_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la tubérosité tibiale droite — "
                    "saillie antérieure du tibia proximal, insertion du ligament patellaire."
                ),
                expected_code="tibial_tuberosity_right",
                result_name="TT_R",
            ),
            ComputeAngle3Pts(
                instruction_fr=(
                    "Calculez l'angle ÉIAS–patella–TT projeté sur le plan frontal "
                    "(suppression de la composante antéro-postérieure). "
                    "C'est l'angle Q."
                ),
                expected_a="ASIS_R",
                expected_vertex="PAT_R",
                expected_c="TT_R",
                project_on="frontal",
                result_name="Q_angle",
            ),
        ],
    ),

    # ── 6. Valgus / varus du genou ────────────────────────────────────────────
    "KVV": AnthroRecipe(
        code="KVV",
        measure_code="knee_valgus_varus",
        name_fr="Valgus / varus du genou",
        unit="°",
        expected_result_name="knee_valgus_varus",
        interpretation_fr=(
            "Déviation par rapport à l'alignement parfait (180°) des axes "
            "fémoral (GT→condyle lat.) et tibial (condyle lat.→malléole lat.) "
            "dans le plan frontal. Norme : 0–6°. "
            "Au-delà = genu valgum ; en deçà de 0° = genu varum."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le grand trochanter droit — "
                    "saillie latérale proximale du fémur, référence de l'axe fémoral."
                ),
                expected_code="greater_trochanter_right",
                result_name="GT_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le condyle latéral du genou droit — "
                    "point de jonction des axes fémoral et tibial (sommet de l'angle)."
                ),
                expected_code="lateral_knee_right",
                result_name="LK_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole latérale droite — "
                    "saillie externe de la cheville, référence distale de l'axe tibial."
                ),
                expected_code="lateral_malleolus_right",
                result_name="LM_R",
            ),
            ComputeAngle3Pts(
                instruction_fr=(
                    "Calculez l'angle GT–condyle lat.–malléole lat. dans le plan frontal. "
                    "La déviation par rapport à 180° donne le valgus/varus."
                ),
                expected_a="GT_R",
                expected_vertex="LK_R",
                expected_c="LM_R",
                project_on="frontal",
                deviation_from_180=True,
                result_name="knee_valgus_varus",
            ),
        ],
    ),

    # ── 7. Genu recurvatum ────────────────────────────────────────────────────
    "GR": AnthroRecipe(
        code="GR",
        measure_code="genu_recurvatum",
        name_fr="Genu recurvatum",
        unit="°",
        expected_result_name="genu_recurvatum",
        interpretation_fr=(
            "Hyperextension du genou : déviation par rapport à 180° "
            "du segment GT–condyle lat.–malléole lat. dans le plan sagittal. "
            "Norme : 0–5°. Au-delà = hyperextension pathologique."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le grand trochanter droit — "
                    "référence proximale de l'axe fémoral (vue sagittale)."
                ),
                expected_code="greater_trochanter_right",
                result_name="GT_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le condyle latéral du genou droit — "
                    "centre articulaire du genou (sommet de l'angle en vue sagittale)."
                ),
                expected_code="lateral_knee_right",
                result_name="LK_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole latérale droite — "
                    "référence distale de l'axe tibial."
                ),
                expected_code="lateral_malleolus_right",
                result_name="LM_R",
            ),
            ComputeAngle3Pts(
                instruction_fr=(
                    "Calculez l'angle GT–condyle lat.–malléole lat. dans le plan sagittal. "
                    "La déviation par rapport à 180° indique l'hyperextension."
                ),
                expected_a="GT_R",
                expected_vertex="LK_R",
                expected_c="LM_R",
                project_on="sagittal",
                deviation_from_180=True,
                result_name="genu_recurvatum",
            ),
        ],
    ),

    # ── 8. Torsion tibiale ────────────────────────────────────────────────────
    "TibT": AnthroRecipe(
        code="TibT",
        measure_code="tibial_torsion",
        name_fr="Torsion tibiale",
        unit="°",
        expected_result_name="tibial_torsion",
        interpretation_fr=(
            "Angle entre l'axe inter-condylien du genou et l'axe bi-malléolaire "
            "projeté sur le plan transverse. Norme : 15–30° (torsion externe physiologique). "
            "< 15° = torsion interne anormale."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le condyle médial du genou droit — "
                    "premier point de l'axe inter-condylien."
                ),
                expected_code="medial_knee_right",
                result_name="MK_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le condyle latéral du genou droit — "
                    "deuxième point de l'axe inter-condylien."
                ),
                expected_code="lateral_knee_right",
                result_name="LK_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole médiale droite — "
                    "premier point de l'axe bi-malléolaire."
                ),
                expected_code="medial_malleolus_right",
                result_name="MM_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole latérale droite — "
                    "deuxième point de l'axe bi-malléolaire."
                ),
                expected_code="lateral_malleolus_right",
                result_name="LM_R",
            ),
            ComputeAxisAngle(
                instruction_fr=(
                    "Calculez l'angle entre l'axe condylien (MK→LK) et "
                    "l'axe bi-malléolaire (MM→LM) projeté sur le plan transverse. "
                    "C'est la torsion tibiale."
                ),
                expected_a1="MK_R",
                expected_a2="LK_R",
                expected_b1="MM_R",
                expected_b2="LM_R",
                plane="transverse",
                result_name="tibial_torsion",
            ),
        ],
    ),

    # ── 9. Antéversion fémorale (proxy clinique) ──────────────────────────────
    "FAC": AnthroRecipe(
        code="FAC",
        measure_code="femoral_anteversion_clinical",
        name_fr="Antéversion fémorale (proxy clinique)",
        unit="°",
        expected_result_name="femoral_anteversion",
        interpretation_fr=(
            "Inclinaison AP du vecteur GT→ÉIAS dans le plan horizontal : "
            "estimation de surface de l'antéversion fémorale. "
            "Ce proxy clinique n'est pas une vraie mesure d'antéversion mais "
            "reflète l'orientation antérieure de la hanche."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le grand trochanter droit — "
                    "point de départ du vecteur de référence fémoral."
                ),
                expected_code="greater_trochanter_right",
                result_name="GT_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'ÉIAS droite — "
                    "point d'arrivée du vecteur GT→ÉIAS (direction antérieure du fémur)."
                ),
                expected_code="ASIS_right",
                result_name="ASIS_R",
            ),
            ComputeAnglePlane(
                instruction_fr=(
                    "Calculez la composante AP de la projection horizontale du vecteur GT→ÉIAS. "
                    "Plus cette valeur est grande, plus la hanche est en antéversion."
                ),
                expected_a="GT_R",
                expected_b="ASIS_R",
                plane="horizontal_ap",
                result_name="femoral_anteversion",
            ),
        ],
    ),

    # ── 10. Angle valgus de l'hallux ──────────────────────────────────────────
    "HVA": AnthroRecipe(
        code="HVA",
        measure_code="hallux_valgus_angle",
        name_fr="Angle valgus de l'hallux",
        unit="°",
        expected_result_name="hallux_valgus",
        interpretation_fr=(
            "Angle 2e métatarse–1er métatarse–hallux dans le plan transverse. "
            "Norme < 15°. Au-delà = hallux valgus (oignon). "
            "Note : le repère hallux utilise first_metatarsal_head comme proxy "
            "si le vrai hallux est absent."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la tête du 2e métatarse droit — "
                    "référence de la direction latérale du pied."
                ),
                expected_code="second_metatarsal_head_right",
                result_name="MET2_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la tête du 1er métatarse droit — "
                    "sommet de l'angle de valgus (articulation métatarso-phalangienne)."
                ),
                expected_code="first_metatarsal_head_right",
                result_name="MET1_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la tête du 1er métatarse droit comme proxy du hallux "
                    "(repère hallux absent du dataset : utiliser first_metatarsal_head). "
                    "Dans un dataset complet, ce serait l'extrémité distale du gros orteil."
                ),
                expected_code="first_metatarsal_head_right",
                result_name="HALLUX_R",
            ),
            ComputeAngle3Pts(
                instruction_fr=(
                    "Calculez l'angle 2e métatarse–1er métatarse–hallux "
                    "projeté sur le plan transverse (horizontal). "
                    "C'est l'angle de valgus de l'hallux."
                ),
                expected_a="MET2_R",
                expected_vertex="MET1_R",
                expected_c="HALLUX_R",
                project_on="transverse",
                result_name="hallux_valgus",
            ),
        ],
    ),

    # ── 11. Valgus calcanéen ──────────────────────────────────────────────────
    "CalV": AnthroRecipe(
        code="CalV",
        measure_code="calcaneus_valgus",
        name_fr="Valgus calcanéen",
        unit="°",
        expected_result_name="calcaneus_valgus",
        interpretation_fr=(
            "Angle du vecteur mi-malléolaire→talon avec la verticale. "
            "Norme : 0–5°. Au-delà = pied plat valgus ; négatif = pied creux varus."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole médiale droite — "
                    "repère interne pour le centre bi-malléolaire."
                ),
                expected_code="medial_malleolus_right",
                result_name="MM_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole latérale droite — "
                    "repère externe pour le centre bi-malléolaire."
                ),
                expected_code="lateral_malleolus_right",
                result_name="LM_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le talon droit — "
                    "point le plus postérieur du calcanéum."
                ),
                expected_code="heel_right",
                result_name="HEEL_R",
            ),
            ComputeMidpoint(
                instruction_fr=(
                    "Calculez le point milieu entre malléole médiale et latérale "
                    "(centre bi-malléolaire)."
                ),
                expected_a="MM_R",
                expected_b="LM_R",
                result_name="MID_MALLE_R",
            ),
            ComputeAnglePlane(
                instruction_fr=(
                    "Calculez l'angle du vecteur talon→mi-malléolaire avec la verticale. "
                    "0° = calcanéum parfaitement vertical ; positif = valgus."
                ),
                expected_a="HEEL_R",
                expected_b="MID_MALLE_R",
                plane="vertical",
                result_name="calcaneus_valgus",
            ),
        ],
    ),

    # ── 12. Chute naviculaire ─────────────────────────────────────────────────
    "NavD": AnthroRecipe(
        code="NavD",
        measure_code="navicular_drop",
        name_fr="Chute naviculaire",
        unit="mm",
        expected_result_name="navicular_drop",
        interpretation_fr=(
            "Hauteur de la malléole médiale moins hauteur de la tubérosité naviculaire "
            "(proxy de la hauteur de voûte). Norme : 15–20 mm. "
            "< 15 mm = voûte aplatie (pied plat) ; > 20 mm = voûte très haute."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la malléole médiale droite — "
                    "référence de hauteur haute."
                ),
                expected_code="medial_malleolus_right",
                result_name="MM_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la tubérosité naviculaire droite — "
                    "saillie médio-plantaire du naviculaire (scaphoïde tarsien), "
                    "point bas de la mesure."
                ),
                expected_code="navicular_tuberosity_right",
                result_name="NAV_R",
            ),
            ComputeAsymmetry(
                instruction_fr=(
                    "Calculez la différence de hauteur (axe vertical) "
                    "entre malléole médiale et tubérosité naviculaire. "
                    "C'est la chute naviculaire (proxy de la hauteur de voûte)."
                ),
                expected_left="MM_R",
                expected_right="NAV_R",
                result_name="navicular_drop",
            ),
        ],
    ),

    # ── 13. Angle de transport du coude (Carrying angle) ─────────────────────
    "CarA": AnthroRecipe(
        code="CarA",
        measure_code="carrying_angle",
        name_fr="Angle de transport du coude",
        unit="°",
        expected_result_name="carrying_angle",
        interpretation_fr=(
            "Angle acromion–épicondyle latéral–styloïde radiale dans le plan frontal "
            "(cubitus valgus physiologique). Norme : 5–15°. "
            "Dépassé = cubitus valgus excessif ; inversé = cubitus varus."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'acromion droit — "
                    "référence proximale de l'axe huméral."
                ),
                expected_code="acromion_right",
                result_name="ACR_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'épicondyle latéral droit — "
                    "saillie latérale de l'humérus distal (sommet de l'angle Q du coude)."
                ),
                expected_code="lateral_epicondyle_right",
                result_name="EPI_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la styloïde radiale droite — "
                    "référence distale de l'axe radio-ulnaire."
                ),
                expected_code="radial_styloid_right",
                result_name="RS_R",
            ),
            ComputeAngle3Pts(
                instruction_fr=(
                    "Calculez l'angle acromion–épicondyle lat.–styloïde radiale "
                    "projeté sur le plan frontal. "
                    "C'est l'angle de transport (valgus physiologique du coude)."
                ),
                expected_a="ACR_R",
                expected_vertex="EPI_R",
                expected_c="RS_R",
                project_on="frontal",
                result_name="carrying_angle",
            ),
        ],
    ),

    # ── 14. Hyperextension du coude ───────────────────────────────────────────
    "EH": AnthroRecipe(
        code="EH",
        measure_code="elbow_hyperextension",
        name_fr="Hyperextension du coude",
        unit="°",
        expected_result_name="elbow_hyperextension",
        interpretation_fr=(
            "Degrés d'hyperextension du coude dans le plan sagittal. "
            "Norme : 0–10°. Au-delà = laxité ligamentaire."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'olécrâne droit — "
                    "sailllie postérieure du coude (référence proximale en vue sagittale)."
                ),
                expected_code="olecranon_right",
                result_name="OLE_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'épicondyle latéral droit — "
                    "centre articulaire du coude (sommet de l'angle)."
                ),
                expected_code="lateral_epicondyle_right",
                result_name="EPI_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez la styloïde radiale droite — "
                    "référence distale de l'axe de l'avant-bras."
                ),
                expected_code="radial_styloid_right",
                result_name="RS_R",
            ),
            ComputeAngle3Pts(
                instruction_fr=(
                    "Calculez l'angle olécrâne–épicondyle lat.–styloïde radiale "
                    "dans le plan sagittal. La déviation par rapport à 180° "
                    "indique l'hyperextension."
                ),
                expected_a="OLE_R",
                expected_vertex="EPI_R",
                expected_c="RS_R",
                project_on="sagittal",
                deviation_from_180=True,
                result_name="elbow_hyperextension",
            ),
        ],
    ),

    # ── 15. Asymétrie scapulaire (hauteur) ────────────────────────────────────
    "ScapHA": AnthroRecipe(
        code="ScapHA",
        measure_code="scapular_height_asymmetry",
        name_fr="Asymétrie scapulaire (hauteur)",
        unit="mm",
        expected_result_name="scapular_height_asym",
        interpretation_fr=(
            "Différence de hauteur entre les angles inférieurs scapulaires "
            "gauche et droit. Norme < 15 mm. "
            "Au-delà = asymétrie scapulaire (dyskinésie ou déséquilibre musculaire)."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'angle inférieur scapulaire gauche — "
                    "pointe inférieure de la scapula gauche."
                ),
                expected_code="scapula_inferior_angle_left",
                result_name="SIA_L",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'angle inférieur scapulaire droit — "
                    "pointe inférieure de la scapula droite."
                ),
                expected_code="scapula_inferior_angle_right",
                result_name="SIA_R",
            ),
            ComputeAsymmetry(
                instruction_fr=(
                    "Calculez la différence de hauteur (axe vertical) "
                    "entre angle inférieur gauche et droit. "
                    "C'est l'asymétrie scapulaire en hauteur."
                ),
                expected_left="SIA_L",
                expected_right="SIA_R",
                result_name="scapular_height_asym",
            ),
        ],
    ),

    # ── 16. Asymétrie de hauteur des épaules ──────────────────────────────────
    "ShouHA": AnthroRecipe(
        code="ShouHA",
        measure_code="shoulder_height_asymmetry",
        name_fr="Asymétrie de hauteur des épaules",
        unit="mm",
        expected_result_name="shoulder_height_asym",
        interpretation_fr=(
            "Différence de hauteur entre acromion gauche et droit. "
            "Norme < 10 mm. Au-delà = épaule plus haute d'un côté "
            "(compensation scoliotique ou déséquilibre musculaire)."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'acromion gauche — "
                    "bord latéral de l'épine de la scapula gauche."
                ),
                expected_code="acromion_left",
                result_name="ACR_L",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'acromion droit — "
                    "bord latéral de l'épine de la scapula droite."
                ),
                expected_code="acromion_right",
                result_name="ACR_R",
            ),
            ComputeAsymmetry(
                instruction_fr=(
                    "Calculez la différence de hauteur (axe vertical) "
                    "entre acromion gauche et droit. "
                    "C'est l'asymétrie de hauteur des épaules."
                ),
                expected_left="ACR_L",
                expected_right="ACR_R",
                result_name="shoulder_height_asym",
            ),
        ],
    ),

    # ── 17. Protraction des épaules ───────────────────────────────────────────
    "ShouP": AnthroRecipe(
        code="ShouP",
        measure_code="shoulder_protraction",
        name_fr="Protraction des épaules",
        unit="mm",
        expected_result_name="shoulder_protraction",
        interpretation_fr=(
            "Position antéro-postérieure moyenne des deux acromions "
            "relativement à C7 (plan thoracique). "
            "Valeur positive = épaules en protraction vers l'avant. "
            "Norme ≈ 0 mm (±20 mm acceptable)."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez C7 (7e vertèbre cervicale) — "
                    "vertèbre proéminente à la base du cou, référence thoracique postérieure."
                ),
                expected_code="C7_spinous",
                result_name="C7",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'acromion gauche — "
                    "repère antérieur de l'épaule gauche."
                ),
                expected_code="acromion_left",
                result_name="ACR_L",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez l'acromion droit — "
                    "repère antérieur de l'épaule droite."
                ),
                expected_code="acromion_right",
                result_name="ACR_R",
            ),
            ComputeMidpoint(
                instruction_fr=(
                    "Calculez le point milieu entre les deux acromions — "
                    "centre de la ceinture scapulaire."
                ),
                expected_a="ACR_L",
                expected_b="ACR_R",
                result_name="ACR_MID",
            ),
            ComputeProjection(
                instruction_fr=(
                    "Calculez la composante antéro-postérieure du vecteur C7→milieu acromions. "
                    "Une valeur positive indique des épaules projetées en avant de C7 "
                    "(protraction)."
                ),
                expected_a="C7",
                expected_b="ACR_MID",
                axis="ap",
                result_name="shoulder_protraction",
            ),
        ],
    ),

    # ── 18. Angle cranio-vertébral (CVA) ──────────────────────────────────────
    "CVA": AnthroRecipe(
        code="CVA",
        measure_code="craniovertebral_angle",
        name_fr="Angle cranio-vertébral",
        unit="°",
        expected_result_name="CVA",
        interpretation_fr=(
            "Angle de la ligne C7→méat acoustique externe (= tragus) "
            "avec l'horizontale dans le plan sagittal. "
            "Norme ≈ 50°. CVA < 46° = posture de tête avancée (forward head posture)."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez C7 (7e vertèbre cervicale) — "
                    "vertèbre proéminente à la base du cou, point de départ du vecteur."
                ),
                expected_code="C7_spinous",
                result_name="C7",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le méat acoustique externe droit — "
                    "orifice de l'oreille (proxy du tragus pour l'angle cranio-vertébral)."
                ),
                expected_code="external_acoustic_meatus_right",
                result_name="EAM_R",
            ),
            ComputeAnglePlane(
                instruction_fr=(
                    "Calculez l'angle de la ligne C7→méat acoustique avec l'horizontale. "
                    "Un angle élevé (> 46°) reflète une bonne posture cervicale."
                ),
                expected_a="C7",
                expected_b="EAM_R",
                plane="horizontal",
                result_name="CVA",
            ),
        ],
    ),

    # ── 19. Inclinaison latérale cervicale ────────────────────────────────────
    "CLI": AnthroRecipe(
        code="CLI",
        measure_code="cervical_lateral_inclination",
        name_fr="Inclinaison latérale cervicale",
        unit="°",
        expected_result_name="cervical_lateral_inclination",
        interpretation_fr=(
            "Déviation de la ligne C7→tragus par rapport à la verticale "
            "dans le plan frontal. Norme ≈ 0°. "
            "Valeur positive = tête inclinée latéralement."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez C7 (7e vertèbre cervicale) — "
                    "base du vecteur cervical."
                ),
                expected_code="C7_spinous",
                result_name="C7",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le méat acoustique externe droit — "
                    "proxy du tragus (sommet du vecteur cervical)."
                ),
                expected_code="external_acoustic_meatus_right",
                result_name="EAM_R",
            ),
            ComputeAnglePlane(
                instruction_fr=(
                    "Calculez l'angle de la ligne C7→méat acoustique "
                    "par rapport à la verticale dans le plan frontal. "
                    "0° = tête parfaitement droite."
                ),
                expected_a="C7",
                expected_b="EAM_R",
                plane="frontal",
                result_name="cervical_lateral_inclination",
            ),
        ],
    ),

    # ── 20. Asymétrie de rotation cervicale ───────────────────────────────────
    "CRA": AnthroRecipe(
        code="CRA",
        measure_code="cervical_rotation_asymmetry",
        name_fr="Asymétrie de rotation cervicale",
        unit="mm",
        expected_result_name="cervical_rotation_asymmetry",
        interpretation_fr=(
            "Différence de position antéro-postérieure entre méat acoustique "
            "gauche et droit relativement à C7. Norme : 0–15 mm. "
            "Une valeur élevée indique une rotation asymétrique de la tête "
            "en position de repos."
        ),
        steps=[
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le méat acoustique externe gauche — "
                    "orifice de l'oreille gauche."
                ),
                expected_code="external_acoustic_meatus_left",
                result_name="EAM_L",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez le méat acoustique externe droit — "
                    "orifice de l'oreille droite."
                ),
                expected_code="external_acoustic_meatus_right",
                result_name="EAM_R",
            ),
            PickLandmark(
                instruction_fr=(
                    "Sélectionnez C7 (7e vertèbre cervicale) — "
                    "référence centrale postérieure."
                ),
                expected_code="C7_spinous",
                result_name="C7",
            ),
            ComputeProjection(
                instruction_fr=(
                    "Calculez la composante AP du vecteur C7→méat acoustique gauche "
                    "(position relative du méat gauche par rapport à C7)."
                ),
                expected_a="C7",
                expected_b="EAM_L",
                axis="ap",
                result_name="AP_L",
            ),
            ComputeProjection(
                instruction_fr=(
                    "Calculez la composante AP du vecteur C7→méat acoustique droit "
                    "(position relative du méat droit par rapport à C7)."
                ),
                expected_a="C7",
                expected_b="EAM_R",
                axis="ap",
                result_name="AP_R",
            ),
            ComputeAsymmetry(
                instruction_fr=(
                    "Calculez la différence absolue entre la position AP du méat gauche "
                    "et du méat droit. C'est l'asymétrie de rotation cervicale."
                ),
                expected_left="AP_L",
                expected_right="AP_R",
                result_name="cervical_rotation_asymmetry",
            ),
        ],
    ),
}
