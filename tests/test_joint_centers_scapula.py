"""Tests du module joint_centers.scapula (GHJC : Meskers 1998, Sobral 2025).

Comme pour la hanche, le test central est l'invariance par rotation : il
échoue si la régression est appliquée dans le repère global au lieu du repère
scapulaire ISB.
"""
from __future__ import annotations

import numpy as np
import pytest

from bonylandmarks.joint_centers import (
    build_scapula_frame,
    estimate_ghjc_meskers,
    estimate_ghjc_sobral,
    estimate_shoulder_joint_center,
    scapula_distances,
)

# ── Fixtures géométriques ─────────────────────────────────────────────────────

# Scapula droite plausible, en mm, monde ISB (X antérieur, Y supérieur, Z droite).
# Géométrie choisie pour être anatomiquement réaliste dans le repère local :
# AA→TS 150 mm (épine), AA→AI 156 mm, TS→AI 130 mm (bord médial),
# AA→AC 55 mm, AC→PC 29 mm, AA→PC 64 mm.
AA = np.array([-30.0, 1420.0, 170.0])
TS = np.array([-67.6, 1405.0, 25.6])
AI = np.array([-55.1, 1290.6, 86.1])
AC = np.array([-4.9, 1440.4, 125.0])
PC = np.array([2.3, 1414.3, 115.4])

LANDMARKS = {"AA": AA, "TS": TS, "AI": AI, "AC": AC, "PC": PC}

SUBJECT = {"age": 30.0, "sex": 1, "height": 1.75, "mass": 70.0}

# Scapula synthétique dont le repère local coïncide exactement avec le monde
# (origine AA = 0, X = e_x, Y = e_y, Z = e_z) : les coordonnées locales sont
# alors égales aux coordonnées globales, ce qui permet de vérifier les
# coefficients de régression en forme fermée.
ALIGNED = {
    "AA": np.zeros(3),
    "TS": np.array([0.0, 0.0, -100.0]),
    "AI": np.array([0.0, -110.0, -25.0]),
    "AC": np.array([20.0, 15.0, -10.0]),
    "PC": np.array([45.0, -5.0, -35.0]),
}

METHODS = ["meskers", "sobral"]
KWARGS = {"sobral": SUBJECT}

MIRROR_Z = np.diag([1.0, 1.0, -1.0])   # plan sagittal Z = 0


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def _call(method: str, landmarks: dict, **extra):
    return estimate_shoulder_joint_center(
        landmarks, method=method, **KWARGS.get(method, {}), **extra
    )


# ── Repère scapulaire ─────────────────────────────────────────────────────────

def test_scapula_frame_orthonormal_right_handed() -> None:
    frame = build_scapula_frame(AA, AI, TS, side="right")
    np.testing.assert_allclose(frame.rotation.T @ frame.rotation, np.eye(3),
                               atol=1e-12)
    assert frame.handedness == pytest.approx(1.0)
    assert frame.axes == ("X", "Y", "Z")
    np.testing.assert_allclose(frame.origin, AA, atol=1e-12)


def test_scapula_frame_axes_anatomical_directions() -> None:
    """X antérieur (+X monde), Y supérieur (+Y monde), Z latéral (+Z monde)."""
    frame = build_scapula_frame(AA, AI, TS, side="right")
    assert frame.axis("X")[0] > 0.8, "X doit être majoritairement antérieur"
    assert frame.axis("Y")[1] > 0.8, "Y doit être majoritairement supérieur"
    assert frame.axis("Z")[2] > 0.8, "Z doit être majoritairement latéral"


def test_left_frame_is_mirror_of_right() -> None:
    """Le repère gauche est le miroir exact du droit (det = -1)."""
    right = build_scapula_frame(AA, AI, TS, side="right")
    left = build_scapula_frame(MIRROR_Z @ AA, MIRROR_Z @ AI, MIRROR_Z @ TS,
                               side="left")
    np.testing.assert_allclose(left.rotation, MIRROR_Z @ right.rotation, atol=1e-12)
    np.testing.assert_allclose(left.origin, MIRROR_Z @ right.origin, atol=1e-12)
    assert left.handedness == pytest.approx(-1.0)
    # X reste antérieur et Y supérieur pour la scapula gauche.
    assert left.axis("X")[0] > 0.8
    assert left.axis("Y")[1] > 0.8
    assert left.axis("Z")[2] < -0.8   # latéral gauche


def test_left_frame_local_coordinates_match_right() -> None:
    """Coordonnées locales identiques à gauche et à droite pour une géométrie
    symétrique — c'est ce qui autorise l'emploi des mêmes coefficients."""
    right = build_scapula_frame(AA, AI, TS, side="right")
    left = build_scapula_frame(MIRROR_Z @ AA, MIRROR_Z @ AI, MIRROR_Z @ TS,
                               side="left")
    for p in (AC, PC, AI, TS):
        np.testing.assert_allclose(left.to_local(MIRROR_Z @ p),
                                   right.to_local(p), atol=1e-9)


def test_anterior_reference_overrides_side() -> None:
    ant = np.array([1.0, 0.0, 0.0])
    a = build_scapula_frame(AA, AI, TS, side="right")
    b = build_scapula_frame(AA, AI, TS, side="left", anterior_reference=ant)
    np.testing.assert_allclose(a.rotation, b.rotation, atol=1e-12)


def test_ai_x_is_structurally_zero_in_isb_frame() -> None:
    """``AI_x`` est identiquement nul dans le repère ISB (Wu 2005).

    L'axe X est défini comme la normale au plan (AI, AA, TS) : AI appartient
    donc au plan YZ du repère, et sa coordonnée X vaut exactement 0 — quelle
    que soit la morphologie. Conséquence : les deux termes en ``AI_x`` de la
    régression de Meskers (``+0.2341*AI_x`` sur GH_x et ``+0.1205*AI_x`` sur
    GH_y) sont structurellement inertes avec cette définition de repère. Ils
    sont conservés pour rester fidèle à la publication.
    """
    frame = build_scapula_frame(AA, AI, TS, side="right")
    assert frame.to_local(AI)[0] == pytest.approx(0.0, abs=1e-9)
    assert frame.to_local(TS)[0] == pytest.approx(0.0, abs=1e-9)
    # ...et cela reste vrai pour une morphologie quelconque.
    rng = np.random.default_rng(0)
    for _ in range(20):
        aa, ai, ts = (rng.normal(scale=100.0, size=3) for _ in range(3))
        f = build_scapula_frame(aa, ai, ts, side="right")
        assert f.to_local(ai)[0] == pytest.approx(0.0, abs=1e-9)


def test_aligned_fixture_frame_is_identity() -> None:
    frame = build_scapula_frame(ALIGNED["AA"], ALIGNED["AI"], ALIGNED["TS"])
    np.testing.assert_allclose(frame.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame.origin, np.zeros(3), atol=1e-12)


# ── 1. Invariance par translation ─────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_translation_invariance(method: str) -> None:
    t = np.array([100.0, -250.0, 500.0])
    ref = _call(method, LANDMARKS)
    moved = _call(method, {k: v + t for k, v in LANDMARKS.items()})
    np.testing.assert_allclose(moved, ref + t, atol=1e-9)


# ── 2. Invariance par rotation (test principal) ───────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_rotation_invariance(method: str) -> None:
    r = _rotation(np.array([0.3, -0.7, 0.5]), 0.84)
    ref = _call(method, LANDMARKS)
    rotated = _call(method, {k: r @ v for k, v in LANDMARKS.items()})
    np.testing.assert_allclose(rotated, r @ ref, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_rigid_transform_invariance(method: str) -> None:
    r = _rotation(np.array([-1.0, 0.4, 2.0]), 2.7)
    t = np.array([-321.0, 45.0, 6789.0])
    ref = _call(method, LANDMARKS)
    moved = _call(method, {k: r @ v + t for k, v in LANDMARKS.items()})
    np.testing.assert_allclose(moved, r @ ref + t, atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_local_coordinates_are_pose_invariant(method: str) -> None:
    r = _rotation(np.array([0.9, 0.1, -0.4]), -1.3)
    t = np.array([5.0, -7.0, 11.0])
    ref = _call(method, LANDMARKS, return_details=True)
    moved = _call(method, {k: r @ v + t for k, v in LANDMARKS.items()},
                  return_details=True)
    np.testing.assert_allclose(moved["center_local"], ref["center_local"],
                               atol=1e-9)


# ── 3. Symétrie droite / gauche ───────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_mirror_symmetry_left_right(method: str) -> None:
    """Scapula gauche miroir → GHJC miroir du GHJC droit."""
    right = _call(method, LANDMARKS, side="right")
    left = _call(method, {k: MIRROR_Z @ v for k, v in LANDMARKS.items()},
                 side="left")
    np.testing.assert_allclose(left, MIRROR_Z @ right, atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_mirror_symmetry_after_rigid_transform(method: str) -> None:
    r = _rotation(np.array([0.2, 1.0, 0.6]), 0.95)
    t = np.array([40.0, -12.0, 300.0])
    right = _call(method, {k: r @ v + t for k, v in LANDMARKS.items()},
                  side="right", return_details=True)
    left = _call(method, {k: r @ (MIRROR_Z @ v) + t for k, v in LANDMARKS.items()},
                 side="left", return_details=True)
    np.testing.assert_allclose(left["center_local"], right["center_local"],
                               atol=1e-9)


# ── 4. Valeurs numériques en forme fermée ─────────────────────────────────────

def test_meskers_closed_form() -> None:
    """Sur la scapula alignée, local == global : coefficients vérifiables."""
    res = estimate_ghjc_meskers(ALIGNED["AA"], ALIGNED["AC"], ALIGNED["AI"],
                                ALIGNED["TS"], ALIGNED["PC"],
                                return_details=True)
    d = scapula_distances(ALIGNED["AA"], ALIGNED["AI"], ALIGNED["TS"],
                          AC=ALIGNED["AC"], PC=ALIGNED["PC"])
    pc, ai = ALIGNED["PC"], ALIGNED["AI"]
    expected = np.array([
        18.9743 + 0.2434 * pc[0] + 0.2341 * ai[0] + 0.1590 * d["L_AA_AI"]
        + 0.0558 * pc[1],
        -3.8791 - 0.1002 * d["L_AC_PC"] + 0.1732 * pc[1] - 0.3940 * d["L_AA_AC"]
        + 0.1205 * ai[0],
        -9.2629 - 0.2403 * pc[1] + 1.0255 * pc[2] + 0.1720 * d["L_TS_PC"],
    ])
    np.testing.assert_allclose(res["center_local"], expected, atol=1e-12)
    # Origine du repère = AA = 0 et rotation = identité → global == local.
    np.testing.assert_allclose(res["center"], expected, atol=1e-12)


def test_sobral_closed_form() -> None:
    res = estimate_ghjc_sobral(ALIGNED["AA"], ALIGNED["AC"], ALIGNED["AI"],
                               ALIGNED["TS"], age=30.0, sex=1, height=1.75,
                               mass=70.0, return_details=True)
    d = scapula_distances(ALIGNED["AA"], ALIGNED["AI"], ALIGNED["TS"],
                          AC=ALIGNED["AC"])
    ac, ai = ALIGNED["AC"], ALIGNED["AI"]
    a, s, h, w = 30.0, 1.0, 1.75, 70.0
    gh_x = (25.5316 + 0.6334 * ac[0] + 0.7842 * ac[1] - 0.0832 * ai[2]
            - 0.2673 * d["L_AA_TS"] + 0.0365 * d["L_AA_AI"]
            - 0.5353 * d["L_AA_AC"] + 0.0843 * d["L_TS_AI"]
            + 0.2350 * d["L_TS_AC"] - 0.1246 * d["L_AI_AC"]
            - 0.0237 * a + 2.1296 * s - 1.1900 * h + 0.0221 * w)
    gh_y = (-6.7070 - 0.2514 * ac[0] + 0.7558 * ac[1] + 0.0264 * ai[2]
            - 0.0620 * d["L_AA_AI"] - 1.7641 * s - 5.7094 * h)
    gh_z = (-22.5233 + 0.5954 * ac[0] + 0.0600 * ac[1] + 0.1085 * ai[2]
            + 0.3983 * ac[2] - 0.3880 * d["L_AA_AC"] - 0.0197 * d["L_TS_AI"]
            - 0.0934 * d["L_TS_AC"] + 0.0558 * a + 0.2930 * s
            + 26.4715 * h - 0.0361 * w)
    np.testing.assert_allclose(res["center_local"], [gh_x, gh_y, gh_z], atol=1e-12)


def test_sobral_covariates_actually_used() -> None:
    """Changer sexe / taille / masse / âge déplace le centre estimé."""
    base = estimate_ghjc_sobral(AA, AC, AI, TS, **SUBJECT)
    for key, value in (("sex", 0), ("height", 1.60), ("mass", 90.0),
                       ("age", 65.0)):
        other = estimate_ghjc_sobral(AA, AC, AI, TS, **{**SUBJECT, key: value})
        assert not np.allclose(base, other), f"{key} sans effet"


def test_sex_accepts_strings() -> None:
    a = estimate_ghjc_sobral(AA, AC, AI, TS, **{**SUBJECT, "sex": "M"})
    b = estimate_ghjc_sobral(AA, AC, AI, TS, **{**SUBJECT, "sex": 1})
    np.testing.assert_allclose(a, b, atol=1e-12)
    c = estimate_ghjc_sobral(AA, AC, AI, TS, **{**SUBJECT, "sex": "F"})
    d = estimate_ghjc_sobral(AA, AC, AI, TS, **{**SUBJECT, "sex": 0})
    np.testing.assert_allclose(c, d, atol=1e-12)


# ── Cohérence des unités ──────────────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_units_metre_matches_millimetre(method: str) -> None:
    lm_m = {k: v / 1000.0 for k, v in LANDMARKS.items()}
    ref = _call(method, LANDMARKS)
    got = _call(method, lm_m, units="m")
    np.testing.assert_allclose(got, ref / 1000.0, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_units_centimetre_matches_millimetre(method: str) -> None:
    lm_cm = {k: v / 10.0 for k, v in LANDMARKS.items()}
    ref = _call(method, LANDMARKS)
    got = _call(method, lm_cm, units="cm")
    np.testing.assert_allclose(got, ref / 10.0, atol=1e-12)


def test_height_in_centimetres_rejected() -> None:
    with pytest.raises(ValueError, match="mètres"):
        estimate_ghjc_sobral(AA, AC, AI, TS, age=30.0, sex=1, height=175.0,
                             mass=70.0)


# ── Plausibilité anatomique ───────────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_ghjc_close_to_acromial_angle(method: str) -> None:
    center = _call(method, LANDMARKS)
    assert np.linalg.norm(center - AA) < 100.0, "GHJC trop loin de AA"


@pytest.mark.parametrize("method", METHODS)
def test_ghjc_is_inferomedial_to_aa(method: str) -> None:
    res = _call(method, LANDMARKS, return_details=True)
    loc = res["center_local"]
    assert loc[1] < 0.0, "le GHJC doit être sous l'angle acromial"
    assert loc[2] < 0.0, "le GHJC doit être médial à l'angle acromial"


def test_methods_land_in_the_same_neighbourhood() -> None:
    """Smoke test grossier, pas une validation.

    Meskers et Sobral rapportent chacun ~10-15 mm de RMS sur *leur* cohorte ;
    la géométrie utilisée ici est synthétique et n'appartient à aucune des
    deux populations de calibration, si bien qu'un écart de l'ordre de 3 cm
    est attendu. Le seuil ne sert qu'à détecter une erreur grossière de
    câblage des coefficients, pas à vérifier une exactitude clinique.
    """
    a = _call("meskers", LANDMARKS)
    b = _call("sobral", LANDMARKS)
    assert np.linalg.norm(a - b) < 50.0


# ── 5. Cas dégénérés ──────────────────────────────────────────────────────────

def test_aa_equals_ts_raises() -> None:
    bad = dict(LANDMARKS, TS=AA.copy())
    with pytest.raises(ValueError, match="confondus"):
        _call("meskers", bad)


def test_collinear_scapula_raises() -> None:
    """AI aligné avec AA et TS → plan scapulaire indéfini."""
    bad = dict(LANDMARKS, AI=AA + 2.5 * (TS - AA))
    with pytest.raises(ValueError, match="colinéaires"):
        _call("meskers", bad)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_raises(bad_value: float) -> None:
    bad = dict(LANDMARKS, AC=np.array([bad_value, 1435.0, 150.0]))
    with pytest.raises(ValueError, match="non finies"):
        _call("sobral", bad)


def test_wrong_shape_raises() -> None:
    bad = dict(LANDMARKS, TS=np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="shape"):
        _call("sobral", bad)


def test_missing_pc_for_meskers_raises() -> None:
    bad = {k: v for k, v in LANDMARKS.items() if k != "PC"}
    with pytest.raises(ValueError, match="PC"):
        _call("meskers", bad)


def test_sobral_works_without_pc() -> None:
    no_pc = {k: v for k, v in LANDMARKS.items() if k != "PC"}
    center = _call("sobral", no_pc)
    assert center.shape == (3,)


def test_missing_covariates_raise() -> None:
    with pytest.raises(ValueError, match="covariables"):
        estimate_shoulder_joint_center(LANDMARKS, method="sobral", age=30.0)


def test_invalid_sex_raises() -> None:
    with pytest.raises(ValueError, match="sex"):
        estimate_ghjc_sobral(AA, AC, AI, TS, age=30.0, sex=2, height=1.75,
                             mass=70.0)


def test_negative_mass_raises() -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        estimate_ghjc_sobral(AA, AC, AI, TS, age=30.0, sex=1, height=1.75,
                             mass=-70.0)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Méthode GHJC inconnue"):
        estimate_shoulder_joint_center(LANDMARKS, method="nope")


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        estimate_shoulder_joint_center(LANDMARKS, method="meskers", side="up")


# ── API ───────────────────────────────────────────────────────────────────────

def test_project_landmark_aliases_accepted() -> None:
    aliased = {
        "acromial_angle_right": AA,
        "scapula_trigonum_right": TS,
        "scapula_inferior_angle_right": AI,
        "acromioclavicular_joint_right": AC,
        "coracoid_process_right": PC,
    }
    got = estimate_shoulder_joint_center(aliased, method="meskers")
    ref = estimate_shoulder_joint_center(LANDMARKS, method="meskers")
    np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_return_details_payload(method: str) -> None:
    res = _call(method, LANDMARKS, return_details=True)
    for key in ("center", "center_local", "frame_origin", "frame_rotation",
                "frame_axes", "dimensions", "method", "side", "units"):
        assert key in res, key
    assert res["frame_rotation"].shape == (3, 3)
    assert res["dimensions"]["L_AA_TS"] == pytest.approx(
        float(np.linalg.norm(AA - TS))
    )
    np.testing.assert_allclose(res["frame_origin"], AA, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_output_shape_and_dtype(method: str) -> None:
    center = _call(method, LANDMARKS)
    assert isinstance(center, np.ndarray)
    assert center.shape == (3,)
    assert center.dtype == np.float64


def test_inputs_are_not_mutated() -> None:
    before = {k: v.copy() for k, v in LANDMARKS.items()}
    _call("sobral", LANDMARKS)
    _call("meskers", LANDMARKS)
    for k, v in before.items():
        np.testing.assert_array_equal(LANDMARKS[k], v)


def test_list_input_accepted() -> None:
    as_lists = {k: v.tolist() for k, v in LANDMARKS.items()}
    got = estimate_shoulder_joint_center(as_lists, method="meskers")
    ref = estimate_shoulder_joint_center(LANDMARKS, method="meskers")
    np.testing.assert_allclose(got, ref, atol=1e-12)
