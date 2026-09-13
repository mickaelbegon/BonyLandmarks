"""Load BodyLoop GLB files into PyVista meshes and extract markers.

Two file types from BodyLoop:
- mesh_3d GLB  : body surface only, PBR texture, no markers
- avatar_3d GLB: body surface + AutoMarkers mesh (83 named reflective markers)
                 + skeleton nodes (joint centres) + Analysis_Spine

All positions returned by this module are in **millimetres**.
glTF stores coordinates in metres; conversion (x1000) is applied automatically.
"""

from __future__ import annotations

import io
import json
import os
import tempfile

import numpy as np
import pyvista as pv
import trimesh

_METRES_THRESHOLD = 10.0   # max bbox dimension below this -> assume metres, convert x1000

# ---------------------------------------------------------------------------
# BodyLoop avatar marker name -> our 24 landmark codes
# ---------------------------------------------------------------------------
BODYLOOP_NAME_TO_CODE: dict[str, str] = {
    "torso.asis.L":                          "ASIS_left",
    "torso.asis.R":                          "ASIS_right",
    "leg.trochanterion.L":                   "greater_trochanter_left",
    "leg.trochanterion.R":                   "greater_trochanter_right",
    "arm.acromion.L":                        "acromion_left",
    "arm.acromion.R":                        "acromion_right",
    "arm.humerus.epicondyle.lateral.L":      "lateral_epicondyle_left",
    "arm.humerus.epicondyle.medial.L":       "medial_epicondyle_left",
    "arm.humerus.epicondyle.lateral.R":      "lateral_epicondyle_right",
    "arm.humerus.epicondyle.medial.R":       "medial_epicondyle_right",
    "arm.ulnar.styloid.L":                   "ulnar_styloid_left",
    "arm.radius.styloid.L":                  "radial_styloid_left",
    "arm.ulnar.styloid.R":                   "ulnar_styloid_right",
    "arm.radius.styloid.R":                  "radial_styloid_right",
    "leg.femur.epicondyle.lateral.L":        "lateral_knee_left",
    "leg.femur.epicondyle.medial.L":         "medial_knee_left",
    "leg.femur.epicondyle.lateral.R":        "lateral_knee_right",
    "leg.femur.epicondyle.medial.R":         "medial_knee_right",
    "leg.lateral.malleolus.L":              "lateral_malleolus_left",
    "leg.medial.malleolus.L":               "medial_malleolus_left",
    "leg.lateral.malleolus.R":              "lateral_malleolus_right",
    "leg.medial.malleolus.R":               "medial_malleolus_right",
    "leg.calcaneous.post.L":                "heel_left",
    "leg.calcaneous.post.R":                "heel_right",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_trimesh_scene(glb_bytes: bytes) -> trimesh.Scene:
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")
    if isinstance(scene, trimesh.Trimesh):
        s = trimesh.Scene()
        s.add_geometry(scene, node_name="Viatar")
        return s
    if isinstance(scene, trimesh.Scene):
        return scene
    raise ValueError(f"Unexpected trimesh type: {type(scene)}")


def _largest_trimesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = {k: v for k, v in scene.geometry.items() if isinstance(v, trimesh.Trimesh)}
    if not meshes:
        raise ValueError("No triangular mesh found in GLB file.")
    if "Viatar" in meshes:
        return meshes["Viatar"]
    return max(meshes.values(), key=lambda m: len(m.vertices))


def _extract_vertex_colors(tm: trimesh.Trimesh) -> "np.ndarray | None":
    try:
        color_mesh = tm.copy()
        color_mesh.visual = tm.visual.to_color()
        vc = color_mesh.visual.vertex_colors
        if vc is not None and len(vc) == len(tm.vertices):
            return np.array(vc, dtype=np.uint8)
    except Exception:
        pass
    return None


def _to_pyvista(tm: trimesh.Trimesh) -> "tuple[pv.PolyData, np.ndarray | None]":
    vertices = np.array(tm.vertices, dtype=float)
    faces = np.array(tm.faces, dtype=int)
    if np.abs(vertices).max() < _METRES_THRESHOLD:
        vertices = vertices * 1000.0
    pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=int), faces]).ravel()
    poly = pv.PolyData(vertices, pv_faces)
    colors = _extract_vertex_colors(tm)
    return poly, colors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_glb_mesh(glb_bytes: bytes) -> "tuple[pv.PolyData, np.ndarray | None]":
    """Return (body PolyData in mm, vertex_colors or None) from a mesh_3d GLB."""
    scene = _to_trimesh_scene(glb_bytes)
    tm = _largest_trimesh(scene)
    return _to_pyvista(tm)


def load_avatar_glb(
    glb_bytes: bytes,
) -> "tuple[pv.PolyData, np.ndarray | None, dict[str, np.ndarray], dict[str, np.ndarray]]":
    """Parse a BodyLoop avatar_3d GLB and return:

    (body_mesh, vertex_colors, landmark_markers, all_markers)

    - body_mesh        : PyVista PolyData in mm
    - vertex_colors    : (N,4) uint8 RGBA or None
    - landmark_markers : {our_code -> xyz_mm}  (24 anatomical landmarks)
    - all_markers      : {bodyloop_name -> xyz_mm}  (all 83 auto-detected markers)
    """
    try:
        import pygltflib  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pygltflib is required: pip install pygltflib") from exc

    scene = _to_trimesh_scene(glb_bytes)
    body_tm = _largest_trimesh(scene)
    body_mesh, vertex_colors = _to_pyvista(body_tm)

    # pygltflib needs a file path, write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp.write(glb_bytes)
        tmp_path = tmp.name
    try:
        gltf = pygltflib.GLTF2().load(tmp_path)
    finally:
        os.unlink(tmp_path)

    all_markers: dict[str, np.ndarray] = {}
    am_node = next((n for n in (gltf.nodes or []) if n.name == "AutoMarkers"), None)

    if am_node is not None and am_node.extras:
        names: list[str] = am_node.extras.get("names", [])
        am_geom = scene.geometry.get("AutoMarkers")
        if am_geom is not None and names:
            pts = np.array(am_geom.vertices, dtype=float)
            if np.abs(pts).max() < _METRES_THRESHOLD:
                pts = pts * 1000.0
            for i, name in enumerate(names):
                if i < len(pts):
                    all_markers[name] = pts[i]

    landmark_markers: dict[str, np.ndarray] = {
        our_code: all_markers[bl_name]
        for bl_name, our_code in BODYLOOP_NAME_TO_CODE.items()
        if bl_name in all_markers
    }

    return body_mesh, vertex_colors, landmark_markers, all_markers


def parse_markers(markers_json: "bytes | str | dict | list") -> "dict[str, np.ndarray]":
    """Parse a markers payload into {code: xyz_array_mm} dict.

    Accepts dict or list formats from the BodyLoop SDK / API.
    Positions assumed to be in mm.
    """
    if isinstance(markers_json, (bytes, str)):
        data = json.loads(markers_json)
    else:
        data = markers_json

    result: dict[str, np.ndarray] = {}

    if isinstance(data, list):
        for item in data:
            label = item.get("label") or item.get("name")
            if label:
                result[label] = np.array([item["x"], item["y"], item["z"]], dtype=float)
    elif isinstance(data, dict):
        for label, coords in data.items():
            if isinstance(coords, dict):
                result[label] = np.array([coords["x"], coords["y"], coords["z"]], dtype=float)
            elif isinstance(coords, (list, tuple)) and len(coords) == 3:
                result[label] = np.array(coords, dtype=float)

    return result
