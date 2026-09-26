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
    "EMG_deltoid_anterior_left":        ["acromion_left"],
    "EMG_deltoid_anterior_right":       ["acromion_right"],
    "EMG_deltoid_medius_left":          ["acromion_left",          "lateral_epicondyle_left"],
    "EMG_deltoid_medius_right":         ["acromion_right",         "lateral_epicondyle_right"],
    "EMG_deltoid_posterior_left":       ["acromial_angle_left"],
    "EMG_deltoid_posterior_right":      ["acromial_angle_right"],
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
    "EMG_serratus_anterior_left":       ["scapula_inferior_angle_left"],
    "EMG_serratus_anterior_right":      ["scapula_inferior_angle_right"],
    "EMG_pectoralis_major_sternal_left": ["sternal_angle_louis"],
    "EMG_pectoralis_major_sternal_right":["sternal_angle_louis"],
    # ── Rachis ────────────────────────────────────────────────────────────────
    "EMG_erector_spinae_longissimus_left":    ["L1_spinous",  "PSIS_left"],
    "EMG_erector_spinae_longissimus_right":   ["L1_spinous",  "PSIS_right"],
    "EMG_erector_spinae_iliocostalis_left":   ["PSIS_left",   "rib12_tip_left"],
    "EMG_erector_spinae_iliocostalis_right":  ["PSIS_right",  "rib12_tip_right"],
    "EMG_obliquus_externus_left":             ["rib12_tip_left",  "ASIS_left"],
    "EMG_obliquus_externus_right":            ["rib12_tip_right", "ASIS_right"],
    "EMG_rectus_abdominis_left":              ["sternal_angle_louis"],
    "EMG_rectus_abdominis_right":             ["sternal_angle_louis"],
}

# Couleurs des sphères de référence EMG (par index 0 → point 1, index 1 → point 2)
EMG_REF_COLORS = ["#00e5ff", "#ff6f00"]   # cyan électrique, orange brûlé
EMG_LINE_COLOR  = "#ffe066"               # jaune doux pour la ligne muscle
