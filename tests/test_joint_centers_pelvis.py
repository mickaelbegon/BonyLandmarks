"""Tests du module joint_centers.pelvis (HJC : Bell, Harrington x2).

Le test central est l'**invariance par rotation** : il échoue si les équations
de régression sont appliquées dans le repère global au lieu du repère pelvien.
"""
from __future__ import annotations

import numpy as np
import pytest

from bonylandmarks.joint_centers import (
    build_pelvis_frame,
    estimate_hip_joint_centers,
    estimate_hjc_bell,
    estimate_hjc_harrington_leg_length,
    estimate_hjc_harrington_pelvis,
    pelvis_dimensions,
)

# ── Fixtures géométriques ─────────────────────────────────────────────────────

# Bassin adulte plausible, en mm, repère "monde" ISB (X ant, Y sup, Z droite).
RASI = np.array([95.0, 985.0, -40.0])
LASI = np.array([-95.0, 985.0, -40.0])
RPSI = np.array([70.0, 985.0, 60.0])
LPSI = np.array([-70.0, 985.0, 60.0])

LANDMARKS = {"RASI": RASI, "LASI": LASI, "RPSI": RPSI, "LPSI": LPSI}

METHODS = ["bell", "harrington", "harrington_leg"]
KWARGS = {"harrington_leg": {"leg_length_right": 850.0}}


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Matrice de rotation (Rodrigues) autour de ``axis``."""
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def _call(method: str, landmarks: dict, **extra):
    return estimate_hip_joint_centers(
        landmarks, method=method, **KWARGS.get(method, {}), **extra
    )


# ── Repère pelvien ────────────────────────────────────────────────────────────

def test_pelvis_frame_is_orthonormal_and_right_handed() -> None:
    frame = build_pelvis_frame(RASI, LASI, RPSI, LPSI)
    r = frame.rotation
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)
    assert frame.axes == ("AP", "SI", "ML")


def test_pelvis_frame_axes_point_the_right_way() -> None:
    frame = build_pelvis_frame(RASI, LASI, RPSI, LPSI)
    # Fixtures : X = droite du sujet, Y = haut, Z = arrière (EIPS en Z positif).
    np.testing.assert_allclose(frame.axis("ML"), [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(frame.axis("SI"), [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(frame.axis("AP"), [0.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(frame.origin, [0.0, 985.0, -40.0], atol=1e-9)


def test_frame_roundtrip_local_global() -> None:
    frame = build_pelvis_frame(RASI, LASI, RPSI, LPSI)
    p = np.array([12.3, -45.6, 78.9])
    np.testing.assert_allclose(frame.to_global(frame.to_local(p)), p, atol=1e-9)


# ── 1. Invariance par translation ─────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_translation_invariance(method: str) -> None:
    t = np.array([100.0, -250.0, 500.0])
    ref = _call(method, LANDMARKS)
    moved = _call(method, {k: v + t for k, v in LANDMARKS.items()})
    for side in ("right", "left"):
        np.testing.assert_allclose(moved[side], ref[side] + t, atol=1e-9)


# ── 2. Invariance par rotation (test principal) ───────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_rotation_invariance(method: str) -> None:
    """Une rotation rigide des repères doit faire tourner le HJC à l'identique.

    Ce test échoue si les coefficients sont appliqués sur les coordonnées
    globales plutôt que dans le repère pelvien local.
    """
    r = _rotation(np.array([0.3, -0.7, 0.5]), 0.84)
    ref = _call(method, LANDMARKS)
    rotated = _call(method, {k: r @ v for k, v in LANDMARKS.items()})
    for side in ("right", "left"):
        np.testing.assert_allclose(rotated[side], r @ ref[side], atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_rigid_transform_invariance(method: str) -> None:
    """Rotation + translation combinées."""
    r = _rotation(np.array([1.0, 2.0, -3.0]), -1.9)
    t = np.array([-321.0, 45.0, 6789.0])
    ref = _call(method, LANDMARKS)
    moved = _call(method, {k: r @ v + t for k, v in LANDMARKS.items()})
    for side in ("right", "left"):
        np.testing.assert_allclose(moved[side], r @ ref[side] + t, atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_local_coordinates_are_invariant(method: str) -> None:
    """Les coordonnées locales ne dépendent pas de la pose du bassin."""
    r = _rotation(np.array([0.1, 0.2, 0.9]), 2.4)
    t = np.array([10.0, 20.0, 30.0])
    ref = _call(method, LANDMARKS, return_details=True)
    moved = _call(method, {k: r @ v + t for k, v in LANDMARKS.items()},
                  return_details=True)
    for side in ("right", "left"):
        np.testing.assert_allclose(
            moved["center_local"][side], ref["center_local"][side], atol=1e-9
        )


# ── 3. Symétrie (mirror test) ─────────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_mirror_symmetry(method: str) -> None:
    """Bassin symétrique → HJC droit et gauche symétriques / plan sagittal.

    Les fixtures sont symétriques par rapport au plan ``X = 0`` : HJC_right et
    HJC_left ne doivent différer que par le signe de leur composante ML.
    """
    res = _call(method, LANDMARKS)
    mirror = np.diag([-1.0, 1.0, 1.0])  # miroir du plan sagittal (X = 0)
    np.testing.assert_allclose(res["left"], mirror @ res["right"], atol=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_mirror_symmetry_after_rigid_transform(method: str) -> None:
    """La symétrie tient aussi après transformation rigide du bassin."""
    r = _rotation(np.array([0.5, 0.5, 0.2]), 1.1)
    t = np.array([15.0, -90.0, 250.0])
    res = _call(method, {k: r @ v + t for k, v in LANDMARKS.items()},
                return_details=True)
    loc_r = res["center_local"]["right"]
    loc_l = res["center_local"]["left"]
    np.testing.assert_allclose(loc_l, loc_r * np.array([1.0, 1.0, -1.0]), atol=1e-9)


def test_asymmetric_leg_lengths_break_si_symmetry_only() -> None:
    res = estimate_hip_joint_centers(
        LANDMARKS, method="harrington_leg",
        leg_length_right=880.0, leg_length_left=820.0, return_details=True,
    )
    loc_r = res["center_local"]["right"]
    loc_l = res["center_local"]["left"]
    assert loc_r[0] == pytest.approx(loc_l[0])          # AP identique
    assert loc_r[1] != pytest.approx(loc_l[1])          # SI différent
    assert loc_r[2] == pytest.approx(-loc_l[2])         # ML opposé
    # LL plus longue → HJC plus bas (coefficient -0.04)
    assert loc_r[1] < loc_l[1]


# ── 4. Cohérence des unités / valeurs numériques ──────────────────────────────

def test_pelvis_dimensions_values() -> None:
    dims = pelvis_dimensions(RASI, LASI, RPSI, LPSI)
    assert dims["PW"] == pytest.approx(190.0)   # 95 - (-95)
    assert dims["PD"] == pytest.approx(100.0)   # |(0,985,-40) - (0,985,60)|


def test_bell_closed_form_values() -> None:
    """PW = 190 mm → offsets exacts de Bell dans le repère pelvien."""
    res = estimate_hjc_bell(RASI, LASI, RPSI, LPSI, return_details=True)
    loc = res["center_local"]["right"]
    np.testing.assert_allclose(
        loc, [-0.19 * 190.0, -0.30 * 190.0, 0.36 * 190.0], atol=1e-12
    )
    # Repère aligné sur les axes du monde (ML=+X, SI=+Y, AP=-Z) → position
    # globale directement vérifiable.
    np.testing.assert_allclose(
        res["right"], [0.0 + 68.4, 985.0 - 57.0, -40.0 + 36.1], atol=1e-9
    )


def test_harrington_pelvis_closed_form_values() -> None:
    res = estimate_hjc_harrington_pelvis(RASI, LASI, RPSI, LPSI,
                                         return_details=True)
    loc = res["center_local"]["right"]
    expected = [-0.24 * 100.0 - 9.9, -0.30 * 190.0 - 10.9, 0.33 * 190.0 + 7.3]
    np.testing.assert_allclose(loc, expected, atol=1e-12)
    assert loc[0] == pytest.approx(-33.9)
    assert loc[1] == pytest.approx(-67.9)
    assert loc[2] == pytest.approx(70.0)


def test_harrington_leg_length_closed_form_values() -> None:
    res = estimate_hjc_harrington_leg_length(
        RASI, LASI, RPSI, LPSI, leg_length_right=850.0, return_details=True
    )
    loc = res["center_local"]["right"]
    expected = [
        -0.24 * 100.0 - 9.9,
        -0.16 * 190.0 - 0.04 * 850.0 - 7.1,
        0.28 * 100.0 + 0.16 * 190.0 + 7.9,
    ]
    np.testing.assert_allclose(loc, expected, atol=1e-12)
    assert loc[1] == pytest.approx(-71.5)
    assert loc[2] == pytest.approx(66.3)


def test_units_metre_matches_millimetre() -> None:
    """Les mêmes repères en m doivent donner le même résultat, /1000."""
    lm_m = {k: v / 1000.0 for k, v in LANDMARKS.items()}
    res_mm = estimate_hip_joint_centers(LANDMARKS, method="harrington")
    res_m = estimate_hip_joint_centers(lm_m, method="harrington", units="m")
    for side in ("right", "left"):
        np.testing.assert_allclose(res_m[side], res_mm[side] / 1000.0, atol=1e-12)


def test_units_metre_leg_length_method() -> None:
    lm_m = {k: v / 1000.0 for k, v in LANDMARKS.items()}
    res_mm = estimate_hip_joint_centers(LANDMARKS, method="harrington_leg",
                                        leg_length_right=850.0)
    res_m = estimate_hip_joint_centers(lm_m, method="harrington_leg",
                                       units="m", leg_length_right=0.850)
    for side in ("right", "left"):
        np.testing.assert_allclose(res_m[side], res_mm[side] / 1000.0, atol=1e-12)


def test_scaling_pelvis_scales_bell_but_not_harrington() -> None:
    """Bell est sans échelle absolue ; Harrington a des constantes en mm."""
    big = {k: v * 2.0 for k, v in LANDMARKS.items()}
    bell_ref = estimate_hjc_bell(RASI, LASI, RPSI, LPSI, return_details=True)
    bell_big = estimate_hjc_bell(*[big[k] for k in ("RASI", "LASI", "RPSI", "LPSI")],
                                 return_details=True)
    np.testing.assert_allclose(
        bell_big["center_local"]["right"],
        2.0 * bell_ref["center_local"]["right"], atol=1e-9,
    )
    har_ref = estimate_hjc_harrington_pelvis(RASI, LASI, RPSI, LPSI,
                                             return_details=True)
    har_big = estimate_hjc_harrington_pelvis(
        *[big[k] for k in ("RASI", "LASI", "RPSI", "LPSI")], return_details=True
    )
    assert not np.allclose(har_big["center_local"]["right"],
                           2.0 * har_ref["center_local"]["right"])


# ── Plausibilité anatomique ───────────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_hjc_is_posterior_inferior_and_lateral(method: str) -> None:
    res = _call(method, LANDMARKS, return_details=True)
    loc = res["center_local"]["right"]
    assert loc[0] < 0.0, "le HJC doit être postérieur au milieu des EIAS"
    assert loc[1] < 0.0, "le HJC doit être inférieur au milieu des EIAS"
    assert loc[2] > 0.0, "le HJC droit doit être latéral (à droite)"
    # Ordre de grandeur : dans l'enveloppe du bassin.
    assert np.linalg.norm(loc) < 200.0


@pytest.mark.parametrize("method", METHODS)
def test_methods_agree_within_3_cm(method: str) -> None:
    ref = _call("harrington", LANDMARKS)["right"]
    other = _call(method, LANDMARKS)["right"]
    assert np.linalg.norm(other - ref) < 30.0


# ── 5. Cas dégénérés ──────────────────────────────────────────────────────────

def test_identical_asis_raises() -> None:
    bad = dict(LANDMARKS, LASI=RASI.copy())
    with pytest.raises(ValueError, match="confondus"):
        estimate_hip_joint_centers(bad, method="harrington")


def test_asis_psis_midpoints_coincide_raises() -> None:
    bad = dict(LANDMARKS, RPSI=np.array([70.0, 985.0, -40.0]),
               LPSI=np.array([-70.0, 985.0, -40.0]))
    with pytest.raises(ValueError, match="confondus"):
        estimate_hip_joint_centers(bad, method="harrington")


def test_collinear_landmarks_raise() -> None:
    """Les quatre points alignés (milieux EIAS/EIPS distincts) → AP indéfini."""
    bad = {
        "RASI": np.array([95.0, 0.0, 0.0]),
        "LASI": np.array([-95.0, 0.0, 0.0]),
        "RPSI": np.array([70.0, 0.0, 0.0]),
        "LPSI": np.array([-30.0, 0.0, 0.0]),
    }
    with pytest.raises(ValueError):
        estimate_hip_joint_centers(bad, method="harrington")


def test_parallel_ml_and_ap_raises() -> None:
    """Milieu EIPS décalé purement selon ML → directions ML et AP parallèles."""
    offset = np.array([50.0, 0.0, 0.0])       # décalage purement médio-latéral
    bad = dict(LANDMARKS, RPSI=RASI + offset, LPSI=LASI + offset)
    with pytest.raises(ValueError, match="parallèles"):
        estimate_hip_joint_centers(bad, method="harrington")


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_raises(bad_value: float) -> None:
    bad = dict(LANDMARKS, RASI=np.array([bad_value, 985.0, -40.0]))
    with pytest.raises(ValueError, match="non finies"):
        estimate_hip_joint_centers(bad, method="harrington")


def test_wrong_shape_raises() -> None:
    bad = dict(LANDMARKS, RASI=np.array([95.0, 985.0]))
    with pytest.raises(ValueError, match="shape"):
        estimate_hip_joint_centers(bad, method="harrington")


def test_missing_landmark_raises() -> None:
    bad = {k: v for k, v in LANDMARKS.items() if k != "RPSI"}
    with pytest.raises(ValueError, match="RPSI"):
        estimate_hip_joint_centers(bad, method="harrington")


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Méthode HJC inconnue"):
        estimate_hip_joint_centers(LANDMARKS, method="nope")


def test_leg_length_required() -> None:
    with pytest.raises(ValueError, match="leg_length_right"):
        estimate_hip_joint_centers(LANDMARKS, method="harrington_leg")


def test_negative_leg_length_raises() -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        estimate_hip_joint_centers(LANDMARKS, method="harrington_leg",
                                   leg_length_right=-850.0)


def test_bad_units_raise() -> None:
    with pytest.raises(ValueError, match="units"):
        estimate_hip_joint_centers(LANDMARKS, method="harrington", units="inch")


# ── API ───────────────────────────────────────────────────────────────────────

def test_project_landmark_aliases_accepted() -> None:
    aliased = {"ASIS_right": RASI, "ASIS_left": LASI,
               "PSIS_right": RPSI, "PSIS_left": LPSI}
    res = estimate_hip_joint_centers(aliased, method="harrington")
    ref = estimate_hip_joint_centers(LANDMARKS, method="harrington")
    np.testing.assert_allclose(res["right"], ref["right"], atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_return_details_payload(method: str) -> None:
    res = _call(method, LANDMARKS, return_details=True)
    for key in ("right", "left", "center", "center_local", "frame_origin",
                "frame_rotation", "frame_axes", "dimensions", "method", "units"):
        assert key in res, key
    assert res["frame_rotation"].shape == (3, 3)
    assert res["dimensions"]["PW"] == pytest.approx(190.0)
    assert res["dimensions"]["PD"] == pytest.approx(100.0)


@pytest.mark.parametrize("method", METHODS)
def test_output_shapes_and_dtype(method: str) -> None:
    res = _call(method, LANDMARKS)
    assert set(res) == {"right", "left"}
    for side in ("right", "left"):
        assert isinstance(res[side], np.ndarray)
        assert res[side].shape == (3,)
        assert res[side].dtype == np.float64


def test_inputs_are_not_mutated() -> None:
    before = {k: v.copy() for k, v in LANDMARKS.items()}
    estimate_hip_joint_centers(LANDMARKS, method="harrington_leg",
                               leg_length_right=850.0)
    for k, v in before.items():
        np.testing.assert_array_equal(LANDMARKS[k], v)


def test_list_input_accepted() -> None:
    as_lists = {k: v.tolist() for k, v in LANDMARKS.items()}
    res = estimate_hip_joint_centers(as_lists, method="bell")
    ref = estimate_hip_joint_centers(LANDMARKS, method="bell")
    np.testing.assert_allclose(res["right"], ref["right"], atol=1e-12)
