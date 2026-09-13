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


def test_load_glb_mesh_returns_tuple_of_two() -> None:
    """load_glb_mesh must return (PolyData, colors_or_None)."""
    import io

    import pyvista as pv
    import trimesh

    from bonylandmarks.mesh_loader import load_glb_mesh

    sphere = trimesh.creation.icosphere(subdivisions=1)
    buf = io.BytesIO()
    sphere.export(buf, file_type="glb")
    glb_bytes = buf.getvalue()

    result = load_glb_mesh(glb_bytes)
    assert isinstance(result, tuple) and len(result) == 2
    poly, colors = result
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points > 0


def test_load_glb_mesh_unit_conversion() -> None:
    """A sphere of radius 0.5 m should be converted to ~500 mm."""
    import io

    import trimesh

    from bonylandmarks.mesh_loader import load_glb_mesh

    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    buf = io.BytesIO()
    sphere.export(buf, file_type="glb")
    poly, _ = load_glb_mesh(buf.getvalue())
    # Max dimension should be ~1000 mm (diameter)
    extent = poly.bounds[1] - poly.bounds[0]
    assert 900 < extent < 1100, f"Expected ~1000 mm, got {extent}"


def test_bodyloop_name_to_code_has_24_entries() -> None:
    from bonylandmarks.mesh_loader import BODYLOOP_NAME_TO_CODE
    assert len(BODYLOOP_NAME_TO_CODE) == 24
    # All values should be known landmark codes
    from bonylandmarks.landmarks import LANDMARK_BY_CODE
    for our_code in BODYLOOP_NAME_TO_CODE.values():
        assert our_code in LANDMARK_BY_CODE, f"Unknown code: {our_code}"
