"""Load BodyLoop GLB files into PyVista meshes and extract markers.

The GLB contains:
- A body surface mesh (first mesh in the scene)
- Marker positions stored in the JSON sidecar or embedded extras

Marker JSON format (markers.json from the BodyLoop normalized export):
    {
        "ASIS_left":  {"x": 12.3, "y": 845.1, "z": -50.2},
        ...
    }
or the list-of-dicts format used by the BodyLoop SDK:
    [{"label": "ASIS_left", "x": 12.3, "y": 845.1, "z": -50.2}, ...]
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


def load_glb_mesh(glb_bytes: bytes) -> pv.PolyData:
    """Parse GLB bytes and return the body surface as a PyVista PolyData."""
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="scene")

    # Merge all geometry into one mesh (body scan is a single surface)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No triangular mesh found in GLB file.")
        combined = trimesh.util.concatenate(meshes)
    elif isinstance(scene, trimesh.Trimesh):
        combined = scene
    else:
        raise ValueError(f"Unexpected trimesh type: {type(scene)}")

    vertices = np.array(combined.vertices, dtype=float)
    faces = np.array(combined.faces, dtype=int)

    # PyVista face format: [3, v0, v1, v2, ...]
    pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=int), faces]).ravel()
    return pv.PolyData(vertices, pv_faces)


def parse_markers(markers_json: bytes | str | dict | list) -> dict[str, np.ndarray]:
    """Parse a markers payload into {code: xyz_array_mm} dict."""
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
