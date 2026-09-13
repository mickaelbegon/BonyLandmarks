"""Unit tests for the landmarks module."""

import pytest

from bonylandmarks.landmarks import (
    LANDMARK_BY_CODE,
    LANDMARKS,
    get_landmark,
)


def test_all_24_landmarks_present() -> None:
    assert len(LANDMARKS) == 24


def test_landmark_codes_unique() -> None:
    codes = [lm.code for lm in LANDMARKS]
    assert len(codes) == len(set(codes))


def test_bilateral_pairs_exist() -> None:
    codes = {lm.code for lm in LANDMARKS}
    bilateral_roots = [
        "ASIS", "greater_trochanter", "acromion",
        "lateral_epicondyle", "medial_epicondyle",
        "ulnar_styloid", "radial_styloid",
        "lateral_knee", "medial_knee",
        "lateral_malleolus", "medial_malleolus",
        "heel",
    ]
    for root in bilateral_roots:
        assert f"{root}_left" in codes, f"Missing {root}_left"
        assert f"{root}_right" in codes, f"Missing {root}_right"


def test_get_landmark_valid() -> None:
    lm = get_landmark("ASIS_left")
    assert lm.code == "ASIS_left"
    assert "iliaque" in lm.name_fr.lower()
    assert "iliac" in lm.name_en.lower()


def test_get_landmark_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown_code"):
        get_landmark("unknown_code")


def test_name_method_fr() -> None:
    lm = get_landmark("acromion_left")
    assert lm.name("fr") == lm.name_fr


def test_name_method_en() -> None:
    lm = get_landmark("acromion_left")
    assert lm.name("en") == lm.name_en


def test_hint_method() -> None:
    lm = get_landmark("heel_right")
    assert len(lm.hint("fr")) > 0
    assert len(lm.hint("en")) > 0


def test_all_landmarks_have_hints() -> None:
    for lm in LANDMARKS:
        assert lm.hint_fr, f"{lm.code} missing hint_fr"
        assert lm.hint_en, f"{lm.code} missing hint_en"
