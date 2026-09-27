"""Landmarks osseux de référence pour le placement des électrodes EMG.

Pour chaque code EMG, liste les codes de repères osseux cités dans le hint
de palpation (points de départ/arrivée de la ligne de placement).
"""

from __future__ import annotations

# code EMG → [code_bone_1, code_bone_2]  (1 ou 2 repères)
EMG_REFERENCES: dict[str, list[str]] = {
    # ── Membres inférieurs ────────────────────────────────────────────────────
    "EMG_rectus_femoris_left":          ["ASIS_left",              "patella_superior_left"],
    "EMG_rectus_femoris_right":         ["ASIS_right",             "patella_superior_right"],
    "EMG_vastus_lateralis_left":        ["ASIS_left",              "lateral_knee_left"],
    "EMG_vastus_lateralis_right":       ["ASIS_right",             "lateral_knee_right"],
    "EMG_vastus_medialis_left":         ["ASIS_left",              "medial_knee_left"],
    "EMG_vastus_medialis_right":        ["ASIS_right",             "medial_knee_right"],
    "EMG_biceps_femoris_left":          ["ischial_tuberosity_left",  "lateral_knee_left"],
    "EMG_biceps_femoris_right":         ["ischial_tuberosity_right", "lateral_knee_right"],
    "EMG_semitendinosus_left":          ["ischial_tuberosity_left",  "medial_knee_left"],
    "EMG_semitendinosus_right":         ["ischial_tuberosity_right", "medial_knee_right"],
    "EMG_gastrocnemius_medialis_left":  ["fibular_head_left",      "medial_malleolus_left"],
    "EMG_gastrocnemius_medialis_right": ["fibular_head_right",     "medial_malleolus_right"],
    "EMG_gastrocnemius_lateralis_left": ["fibular_head_left",      "heel_left"],
    "EMG_gastrocnemius_lateralis_right":["fibular_head_right",     "heel_right"],
    "EMG_soleus_left":                  ["medial_knee_left",       "medial_malleolus_left"],
    "EMG_soleus_right":                 ["medial_knee_right",      "medial_malleolus_right"],
    "EMG_tibialis_anterior_left":       ["fibular_head_left",      "medial_malleolus_left"],
    "EMG_tibialis_anterior_right":      ["fibular_head_right",     "medial_malleolus_right"],
    "EMG_peroneus_longus_left":         ["fibular_head_left",      "lateral_malleolus_left"],
    "EMG_peroneus_longus_right":        ["fibular_head_right",     "lateral_malleolus_right"],
    # ── Hanches ───────────────────────────────────────────────────────────────
    "EMG_gluteus_maximus_left":         ["sacrum_S2",              "greater_trochanter_left"],
    "EMG_gluteus_maximus_right":        ["sacrum_S2",              "greater_trochanter_right"],
    "EMG_gluteus_medius_left":          ["iliac_crest_left",       "greater_trochanter_left"],
    "EMG_gluteus_medius_right":         ["iliac_crest_right",      "greater_trochanter_right"],
    "EMG_tensor_fasciae_latae_left":    ["ASIS_left",              "lateral_knee_left"],
    "EMG_tensor_fasciae_latae_right":   ["ASIS_right",             "lateral_knee_right"],
    # ── Épaule ────────────────────────────────────────────────────────────────
    "EMG_deltoid_anterior_left":        ["acromion_left",          "deltoid_tuberosity_left"],
    "EMG_deltoid_anterior_right":       ["acromion_right",         "deltoid_tuberosity_right"],
    "EMG_deltoid_medius_left":          ["acromion_left",          "lateral_epicondyle_left"],
    "EMG_deltoid_medius_right":         ["acromion_right",         "lateral_epicondyle_right"],
    "EMG_deltoid_posterior_left":       ["acromial_angle_left",    "deltoid_tuberosity_left"],
    "EMG_deltoid_posterior_right":      ["acromial_angle_right",   "deltoid_tuberosity_right"],
    # ── Bras ──────────────────────────────────────────────────────────────────
    "EMG_biceps_brachii_left":          ["acromion_left",          "lateral_epicondyle_left"],
    "EMG_biceps_brachii_right":         ["acromion_right",         "lateral_epicondyle_right"],
    "EMG_triceps_brachii_long_left":    ["acromion_left",          "olecranon_left"],
    "EMG_triceps_brachii_long_right":   ["acromion_right",         "olecranon_right"],
    "EMG_triceps_brachii_lateral_left": ["acromion_left",          "olecranon_left"],
    "EMG_triceps_brachii_lateral_right":["acromion_right",         "olecranon_right"],
    # ── Avant-bras ────────────────────────────────────────────────────────────
    "EMG_wrist_extensors_left":         ["lateral_epicondyle_left", "radial_styloid_left"],
    "EMG_wrist_extensors_right":        ["lateral_epicondyle_right","radial_styloid_right"],
    "EMG_wrist_flexors_left":           ["medial_epicondyle_left",  "radial_styloid_left"],
    "EMG_wrist_flexors_right":          ["medial_epicondyle_right", "radial_styloid_right"],
    "EMG_brachioradialis_left":         ["lateral_epicondyle_left", "radial_styloid_left"],
    "EMG_brachioradialis_right":        ["lateral_epicondyle_right","radial_styloid_right"],
    # ── Tronc / Trapèze ───────────────────────────────────────────────────────
    "EMG_trapezius_descendens_left":    ["C7_spinous",             "acromion_left"],
    "EMG_trapezius_descendens_right":   ["C7_spinous",             "acromion_right"],
    "EMG_trapezius_transversalis_left": ["scapula_trigonum_left",  "T8_spinous"],
    "EMG_trapezius_transversalis_right":["scapula_trigonum_right", "T8_spinous"],
    "EMG_trapezius_ascendens_left":     ["scapula_trigonum_left",  "T8_spinous"],
    "EMG_trapezius_ascendens_right":    ["scapula_trigonum_right", "T8_spinous"],
    "EMG_serratus_anterior_left":       ["scapula_inferior_angle_left",  "iliac_crest_left"],
    "EMG_serratus_anterior_right":      ["scapula_inferior_angle_right", "iliac_crest_right"],
    "EMG_pectoralis_major_sternal_left": ["sternal_angle_louis",         "deltoid_tuberosity_left"],
    "EMG_pectoralis_major_sternal_right":["sternal_angle_louis",         "deltoid_tuberosity_right"],
    # ── Rachis ────────────────────────────────────────────────────────────────
    "EMG_erector_spinae_longissimus_left":    ["L1_spinous",  "PSIS_left"],
    "EMG_erector_spinae_longissimus_right":   ["L1_spinous",  "PSIS_right"],
    "EMG_erector_spinae_iliocostalis_left":   ["PSIS_left",   "rib12_tip_left"],
    "EMG_erector_spinae_iliocostalis_right":  ["PSIS_right",  "rib12_tip_right"],
    "EMG_obliquus_externus_left":             ["rib12_tip_left",  "ASIS_left"],
    "EMG_obliquus_externus_right":            ["rib12_tip_right", "ASIS_right"],
    "EMG_rectus_abdominis_left":              ["sternal_angle_louis",  "pubic_symphysis"],
    "EMG_rectus_abdominis_right":             ["sternal_angle_louis",  "pubic_symphysis"],
}

# Couleurs des sphères de référence EMG (par index 0 → point 1, index 1 → point 2)
EMG_REF_COLORS   = ["#00e5ff", "#ff6f00"]   # cyan électrique, orange brûlé
EMG_LINE_COLOR   = "#ffe066"                # jaune doux pour la ligne géodésique
EMG_MARKER_COLOR = "#ff2222"               # rouge vif pour le point de placement EMG

# Pourcentage de distance (depuis le 1er point de référence vers le 2e) auquel
# placer l'électrode EMG selon les recommandations SENIAM / Hermens et al. 2000.
# 0.5 = milieu de la ligne de référence.
EMG_PLACEMENT_PCT: dict[str, float] = {
    # ── Membres inférieurs ────────────────────────────────────────────────────
    "EMG_rectus_femoris_left":           0.50,  # 50% ASIS → patella sup
    "EMG_rectus_femoris_right":          0.50,
    "EMG_vastus_lateralis_left":         0.667, # 2/3 ASIS → lateral knee
    "EMG_vastus_lateralis_right":        0.667,
    "EMG_vastus_medialis_left":          0.80,  # 4/5 ASIS → medial knee
    "EMG_vastus_medialis_right":         0.80,
    "EMG_biceps_femoris_left":           0.50,  # 50% ischial tuberosity → lateral knee
    "EMG_biceps_femoris_right":          0.50,
    "EMG_semitendinosus_left":           0.50,
    "EMG_semitendinosus_right":          0.50,
    "EMG_gastrocnemius_medialis_left":   0.37,  # 37% fibular head → medial malleolus
    "EMG_gastrocnemius_medialis_right":  0.37,
    "EMG_gastrocnemius_lateralis_left":  0.30,  # 30% fibular head → heel
    "EMG_gastrocnemius_lateralis_right": 0.30,
    "EMG_soleus_left":                   0.67,  # 2/3 medial knee → medial malleolus
    "EMG_soleus_right":                  0.67,
    "EMG_tibialis_anterior_left":        0.33,  # 1/3 fibular head → medial malleolus
    "EMG_tibialis_anterior_right":       0.33,
    "EMG_peroneus_longus_left":          0.33,  # 1/3 fibular head → lateral malleolus
    "EMG_peroneus_longus_right":         0.33,
    # ── Hanches ───────────────────────────────────────────────────────────────
    "EMG_gluteus_maximus_left":          0.50,  # 50% sacrum-S2 → greater trochanter
    "EMG_gluteus_maximus_right":         0.50,
    "EMG_gluteus_medius_left":           0.50,  # 50% iliac crest → greater trochanter
    "EMG_gluteus_medius_right":          0.50,
    "EMG_tensor_fasciae_latae_left":     0.33,  # 1/3 ASIS → lateral knee
    "EMG_tensor_fasciae_latae_right":    0.33,
    # ── Épaule / deltoïde ─────────────────────────────────────────────────────
    "EMG_deltoid_anterior_left":         0.33,  # 1/3 acromion → tubérosité deltoïdienne
    "EMG_deltoid_anterior_right":        0.33,
    "EMG_deltoid_medius_left":           0.50,  # 50% acromion → lateral epicondyle
    "EMG_deltoid_medius_right":          0.50,
    "EMG_deltoid_posterior_left":        0.33,  # 1/3 angle acromial → tubérosité deltoïdienne
    "EMG_deltoid_posterior_right":       0.33,
    # ── Bras ──────────────────────────────────────────────────────────────────
    "EMG_biceps_brachii_left":           0.33,  # 1/3 acromion → lateral epicondyle
    "EMG_biceps_brachii_right":          0.33,
    "EMG_triceps_brachii_long_left":     0.50,  # 50% acromion → olecranon
    "EMG_triceps_brachii_long_right":    0.50,
    "EMG_triceps_brachii_lateral_left":  0.50,
    "EMG_triceps_brachii_lateral_right": 0.50,
    # ── Avant-bras ────────────────────────────────────────────────────────────
    "EMG_wrist_extensors_left":          0.67,  # 2/3 lateral epicondyle → radial styloid
    "EMG_wrist_extensors_right":         0.67,
    "EMG_wrist_flexors_left":            0.67,  # 2/3 medial epicondyle → radial styloid
    "EMG_wrist_flexors_right":           0.67,
    "EMG_brachioradialis_left":          0.50,
    "EMG_brachioradialis_right":         0.50,
    # ── Tronc / Trapèze ───────────────────────────────────────────────────────
    "EMG_trapezius_descendens_left":     0.50,  # 50% C7 → acromion
    "EMG_trapezius_descendens_right":    0.50,
    "EMG_trapezius_transversalis_left":  0.50,  # 50% trigonum scapulae → T8
    "EMG_trapezius_transversalis_right": 0.50,
    "EMG_trapezius_ascendens_left":      0.67,  # 2/3 trigonum scapulae → T8
    "EMG_trapezius_ascendens_right":     0.67,
    # ── Rachis ────────────────────────────────────────────────────────────────
    "EMG_erector_spinae_longissimus_left":    0.50,  # 50% L1 → PSIS
    "EMG_erector_spinae_longissimus_right":   0.50,
    "EMG_erector_spinae_iliocostalis_left":   0.50,  # 50% PSIS → rib12 tip
    "EMG_erector_spinae_iliocostalis_right":  0.50,
    "EMG_obliquus_externus_left":             0.50,  # 50% rib12 tip → ASIS
    "EMG_obliquus_externus_right":            0.50,
    # ── Tronc antérieur ───────────────────────────────────────────────────────
    "EMG_pectoralis_major_sternal_left":      0.67,  # 2/3 angle de Louis → tubérosité deltoïdienne
    "EMG_pectoralis_major_sternal_right":     0.67,
    "EMG_serratus_anterior_left":             0.33,  # 1/3 angle inf. scapula → crête iliaque
    "EMG_serratus_anterior_right":            0.33,
    "EMG_rectus_abdominis_left":              0.33,  # 1/3 angle de Louis → symphyse pubienne
    "EMG_rectus_abdominis_right":             0.33,
}

# FMA concept ID du muscle BodyParts3D correspondant à chaque électrode EMG.
# None = muscle absent de BONE_MUSCLES ou FMA inconnu (pas de géodésique tracée).
EMG_MUSCLE_FMA: dict[str, int | None] = {
    # ── Membres inférieurs ────────────────────────────────────────────────────
    "EMG_rectus_femoris_left":           38929,
    "EMG_rectus_femoris_right":          38928,
    "EMG_vastus_lateralis_left":         38931,
    "EMG_vastus_lateralis_right":        38930,
    "EMG_vastus_medialis_left":          38933,
    "EMG_vastus_medialis_right":         38932,
    "EMG_biceps_femoris_left":           45889,   # long head
    "EMG_biceps_femoris_right":          45888,
    "EMG_semitendinosus_left":           22359,
    "EMG_semitendinosus_right":          22358,
    "EMG_gastrocnemius_medialis_left":   45958,
    "EMG_gastrocnemius_medialis_right":  45957,
    "EMG_gastrocnemius_lateralis_left":  45961,
    "EMG_gastrocnemius_lateralis_right": 45960,
    "EMG_soleus_left":                   22559,
    "EMG_soleus_right":                  22558,
    "EMG_tibialis_anterior_left":        22545,
    "EMG_tibialis_anterior_right":       22544,
    "EMG_peroneus_longus_left":          22553,
    "EMG_peroneus_longus_right":         22552,
    # ── Hanches ───────────────────────────────────────────────────────────────
    "EMG_gluteus_maximus_left":          22329,
    "EMG_gluteus_maximus_right":         22328,
    "EMG_gluteus_medius_left":           22331,
    "EMG_gluteus_medius_right":          22330,
    "EMG_tensor_fasciae_latae_left":     22426,
    "EMG_tensor_fasciae_latae_right":    22425,
    # ── Épaule / deltoïde ─────────────────────────────────────────────────────
    "EMG_deltoid_anterior_left":         34681,   # partie claviculaire gauche
    "EMG_deltoid_anterior_right":        34680,
    "EMG_deltoid_medius_left":           34683,   # partie acromiale gauche
    "EMG_deltoid_medius_right":          34682,
    "EMG_deltoid_posterior_left":        34685,   # partie spinale gauche
    "EMG_deltoid_posterior_right":       34684,
    # ── Bras ──────────────────────────────────────────────────────────────────
    "EMG_biceps_brachii_left":           37685,   # long head
    "EMG_biceps_brachii_right":          37684,
    "EMG_triceps_brachii_long_left":     37700,
    "EMG_triceps_brachii_long_right":    37699,
    "EMG_triceps_brachii_lateral_left":  37696,
    "EMG_triceps_brachii_lateral_right": 37695,
    # ── Avant-bras ────────────────────────────────────────────────────────────
    "EMG_wrist_extensors_left":          38499,   # ECRB gauche
    "EMG_wrist_extensors_right":         38498,
    "EMG_wrist_flexors_left":            38461,   # FCR gauche
    "EMG_wrist_flexors_right":           38460,
    "EMG_brachioradialis_left":          38487,
    "EMG_brachioradialis_right":         38486,
    # ── Tronc / Trapèze ───────────────────────────────────────────────────────
    "EMG_trapezius_descendens_left":     33587,
    "EMG_trapezius_descendens_right":    33586,
    "EMG_trapezius_transversalis_left":  33585,
    "EMG_trapezius_transversalis_right": 33584,
    "EMG_trapezius_ascendens_left":      33583,
    "EMG_trapezius_ascendens_right":     33581,
    "EMG_erector_spinae_longissimus_left":    22753,
    "EMG_erector_spinae_longissimus_right":   22751,
    "EMG_erector_spinae_iliocostalis_left":   22741,
    "EMG_erector_spinae_iliocostalis_right":  22740,
    "EMG_obliquus_externus_left":        13337,
    "EMG_obliquus_externus_right":       13336,
    "EMG_pectoralis_major_sternal_left": 13039,
    "EMG_pectoralis_major_sternal_right":13039,
    # ── FMA inconnu / absent ──────────────────────────────────────────────────
    "EMG_serratus_anterior_left":        None,
    "EMG_serratus_anterior_right":       None,
    "EMG_rectus_abdominis_left":         None,
    "EMG_rectus_abdominis_right":        None,
}
