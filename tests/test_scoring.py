"""Unit tests for the scoring module."""

import numpy as np
import pytest

from bonylandmarks.scoring import LandmarkResult, SessionScore


def _make_result(gt, pick, code="ASIS_left") -> LandmarkResult:
    return LandmarkResult(
        code=code,
        ground_truth=np.array(gt, dtype=float),
        student_pick=np.array(pick, dtype=float),
    )


def test_error_mm_zero() -> None:
    r = _make_result([0, 0, 0], [0, 0, 0])
    assert r.error_mm == pytest.approx(0.0)


def test_error_mm_known() -> None:
    # 3-4-5 triangle: sqrt(9+16+0) = 5
    r = _make_result([0, 0, 0], [3, 4, 0])
    assert r.error_mm == pytest.approx(5.0)


def test_feedback_color_green() -> None:
    r = _make_result([0, 0, 0], [10, 0, 0])
    assert r.feedback_color() == "green"


def test_feedback_color_orange() -> None:
    r = _make_result([0, 0, 0], [30, 0, 0])
    assert r.feedback_color() == "orange"


def test_feedback_color_red() -> None:
    r = _make_result([0, 0, 0], [50, 0, 0])
    assert r.feedback_color() == "red"


def test_to_dict_keys() -> None:
    r = _make_result([1, 2, 3], [4, 5, 6])
    d = r.to_dict()
    assert {"code", "ground_truth_mm", "student_pick_mm", "error_mm", "feedback_color"} <= d.keys()


def test_session_score_mean() -> None:
    results = [
        _make_result([0, 0, 0], [10, 0, 0], code="a"),
        _make_result([0, 0, 0], [20, 0, 0], code="b"),
    ]
    session = SessionScore(results)
    assert session.mean_error_mm == pytest.approx(15.0)


def test_session_score_max() -> None:
    results = [
        _make_result([0, 0, 0], [10, 0, 0], code="a"),
        _make_result([0, 0, 0], [40, 0, 0], code="b"),
    ]
    session = SessionScore(results)
    assert session.max_error_mm == pytest.approx(40.0)


def test_session_empty() -> None:
    session = SessionScore([])
    assert session.mean_error_mm == 0.0
    assert session.max_error_mm == 0.0


def test_session_to_dict_structure() -> None:
    r = _make_result([0, 0, 0], [5, 0, 0])
    session = SessionScore([r])
    d = session.to_dict()
    assert d["n_landmarks"] == 1
    assert len(d["landmarks"]) == 1
