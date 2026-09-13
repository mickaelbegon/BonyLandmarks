"""Unit tests for mesh_loader (marker parsing only — no GLB fixture yet)."""

import json

import numpy as np
import pytest

from bonylandmarks.mesh_loader import parse_markers


def test_parse_markers_dict_format() -> None:
    data = {
        "ASIS_left": {"x": 38.2, "y": 945.3, "z": -12.1},
        "heel_right": {"x": -18.0, "y": 20.0, "z": -45.0},
    }
    result = parse_markers(data)
    assert "ASIS_left" in result
    np.testing.assert_allclose(result["ASIS_left"], [38.2, 945.3, -12.1])
    np.testing.assert_allclose(result["heel_right"], [-18.0, 20.0, -45.0])


def test_parse_markers_list_format() -> None:
    data = [
        {"label": "acromion_left", "x": 148.0, "y": 1388.0, "z": 4.0},
        {"label": "acromion_right", "x": -148.0, "y": 1388.0, "z": 4.0},
    ]
    result = parse_markers(data)
    assert "acromion_left" in result
    assert "acromion_right" in result


def test_parse_markers_from_json_string() -> None:
    payload = json.dumps({"heel_left": {"x": 18.0, "y": 20.0, "z": -45.0}})
    result = parse_markers(payload)
    assert "heel_left" in result


def test_parse_markers_from_json_bytes() -> None:
    payload = json.dumps({"heel_left": {"x": 1.0, "y": 2.0, "z": 3.0}}).encode()
    result = parse_markers(payload)
    np.testing.assert_allclose(result["heel_left"], [1.0, 2.0, 3.0])


def test_parse_markers_synthetic_fixture() -> None:
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "synthetic_markers.json"
    result = parse_markers(fixture.read_bytes())
    assert len(result) == 24
    for code in ("ASIS_left", "ASIS_right", "heel_left", "heel_right"):
        assert code in result
