"""Sticker removal (vertex-colour inpainting) for BodyLoop GLB scans.

Green photogrammetry stickers (~20 mm diameter) appear as green patches on the
mesh vertex colours.  This module replaces those patches with skin colours
interpolated from the surrounding annulus using Inverse Distance Weighting (IDW).

Coordinate system: all distances are in **millimetres**.  If the trimesh
vertices are still in metres (max bbox < 10) they are scaled to mm internally
before any distance calculation; the output mesh is returned in the original
coordinate system of the input.
"""

from __future__ import annotations

import numpy as np
import trimesh

_METRES_THRESHOLD = 10.0   # same heuristic as mesh_loader.py


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_vertex_colors(tm: trimesh.Trimesh) -> "np.ndarray | None":
    """Return (N,4) uint8 RGBA vertex colours, or None if unavailable.

    Handles the three common trimesh visual types:
    - ColorVisuals with vertex_colors  → returned as-is
    - ColorVisuals with face_colors    → averaged to vertex colours
    - TextureVisuals (PBR / UV map)    → baked via .to_color()

    For BodyLoop avatar GLBs the body mesh is a PBR texture, so the baking
    path is the normal one.  Baking is lossy (texture resolution → vertex
    density) but perfectly adequate for colour inpainting.
    """
    visual = tm.visual

    # Already per-vertex -------------------------------------------------------
    if isinstance(visual, trimesh.visual.ColorVisuals):
        vc = visual.vertex_colors
        if vc is not None and len(vc) == len(tm.vertices):
            return np.array(vc, dtype=np.uint8)

        # Per-face colours: average the faces touching each vertex
        fc = visual.face_colors
        if fc is not None and len(fc) == len(tm.faces):
            fc = np.array(fc, dtype=np.float64)
            n_verts = len(tm.vertices)
            acc = np.zeros((n_verts, 4), dtype=np.float64)
            cnt = np.zeros(n_verts, dtype=np.float64)
            for col, face in zip(fc, tm.faces):
                acc[face] += col
                cnt[face] += 1.0
            cnt = np.maximum(cnt, 1.0)
            return np.clip(acc / cnt[:, None], 0, 255).astype(np.uint8)

    # Texture / PBR: bake to colour via trimesh --------------------------------
    try:
        color_mesh = tm.copy()
        color_mesh.visual = tm.visual.to_color()
        vc = color_mesh.visual.vertex_colors
        if vc is not None and len(vc) == len(tm.vertices):
            return np.array(vc, dtype=np.uint8)
    except Exception:
        pass

    return None


def _vertices_mm(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return mesh vertices as float64, converted to mm if they look like metres."""
    verts = np.array(mesh.vertices, dtype=np.float64)
    if np.abs(verts).max() < _METRES_THRESHOLD:
        verts = verts * 1000.0
    return verts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_sticker_markers(
    mesh: trimesh.Trimesh,
    marker_positions: "list[np.ndarray] | np.ndarray",
    sticker_radius_mm: float = 16.0,
    blend_margin_mm: float = 6.0,
    idw_neighbors: int = 12,
    idw_power: float = 2.0,
) -> trimesh.Trimesh:
    """Replace green sticker patches on a body-scan mesh with inpainted skin colour.

    Algorithm
    ---------
    For each marker centre *c* (in mm):

    1. **Inner zone** – vertices at distance d ≤ sticker_radius_mm from *c*:
       colours to be replaced.
    2. **Crown zone** – vertices at distance sticker_radius_mm < d ≤
       sticker_radius_mm + blend_margin_mm: reference skin colour.
    3. **IDW reconstruction** – for every inner vertex the replacement colour is
       the Inverse Distance Weighted average of the *k* nearest crown vertices
       (distance measured in 3-D, power = idw_power).  Vertices close to the
       sticker boundary naturally receive colours very similar to the actual
       skin at the boundary → no visible seam.

    If a marker has no crown vertices (e.g. it sits at the very edge of the
    mesh) the crown radius is widened by 50 % before giving up.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (TextureVisuals or ColorVisuals, vertices in m or mm).
    marker_positions : list of (3,) arrays
        Centre of each sticker in **millimetres**.
    sticker_radius_mm : float
        Radius of the sticker disc to erase (default 15 mm → 30 mm diameter).
    blend_margin_mm : float
        Width of the reference skin annulus (default 8 mm).
    idw_neighbors : int
        Number of nearest crown neighbours used per inner vertex (default 12).
    idw_power : float
        IDW exponent; higher = sharper boundary fidelity (default 2).

    Returns
    -------
    trimesh.Trimesh
        A copy of the input mesh with ColorVisuals and the sticker patches
        replaced.  Vertices and faces are unchanged.
    """
    try:
        from scipy.spatial import KDTree  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    # ------------------------------------------------------------------
    # 1. Extract vertex colours
    # ------------------------------------------------------------------
    colors = _extract_vertex_colors(mesh)
    if colors is None:
        # Nothing to do – return mesh as-is
        return mesh

    colors_f = colors.astype(np.float64)   # work in float to avoid rounding

    # ------------------------------------------------------------------
    # 2. Vertices in mm for distance calculations
    # ------------------------------------------------------------------
    verts_mm = _vertices_mm(mesh)
    global_tree = KDTree(verts_mm)

    # ------------------------------------------------------------------
    # 3. Process each marker
    # ------------------------------------------------------------------
    for raw_center in marker_positions:
        center = np.asarray(raw_center, dtype=np.float64).ravel()
        if center.shape[0] != 3:
            continue

        outer_r = sticker_radius_mm + blend_margin_mm

        # All vertices inside the outer sphere
        candidate_idx = np.array(
            global_tree.query_ball_point(center, outer_r), dtype=int
        )
        if len(candidate_idx) == 0:
            continue

        dists = np.linalg.norm(verts_mm[candidate_idx] - center, axis=1)

        inner_mask = dists <= sticker_radius_mm
        crown_mask = ~inner_mask  # everything else inside outer_r is crown

        inner_global = candidate_idx[inner_mask]
        crown_global = candidate_idx[crown_mask]

        if len(inner_global) == 0:
            continue

        # Fallback: widen crown if empty (marker near mesh boundary)
        if len(crown_global) == 0:
            fallback_r = outer_r * 1.5
            extra_idx = np.array(
                global_tree.query_ball_point(center, fallback_r), dtype=int
            )
            extra_dists = np.linalg.norm(verts_mm[extra_idx] - center, axis=1)
            crown_global = extra_idx[extra_dists > sticker_radius_mm]
            if len(crown_global) == 0:
                continue   # truly isolated – skip

        # ------------------------------------------------------------------
        # 4. IDW interpolation: crown colours → inner vertices
        # ------------------------------------------------------------------
        crown_verts = verts_mm[crown_global]          # (M, 3)
        crown_colors = colors_f[crown_global]         # (M, 4)
        crown_tree = KDTree(crown_verts)

        inner_verts = verts_mm[inner_global]          # (n, 3)
        k = min(idw_neighbors, len(crown_global))
        dists_to_crown, nn_idx = crown_tree.query(inner_verts, k=k)

        # Guard against degenerate zero-distance (inner vertex coincides with
        # a crown vertex – can happen at the exact radius boundary)
        dists_to_crown = np.maximum(dists_to_crown, 1e-6)

        weights = 1.0 / (dists_to_crown ** idw_power)           # (n, k)
        weights /= weights.sum(axis=1, keepdims=True)            # normalise

        # Weighted colour sum: einsum over neighbour axis
        # crown_colors[nn_idx] shape: (n, k, 4)
        new_colors = np.einsum("ij,ijk->ik", weights, crown_colors[nn_idx])

        colors_f[inner_global] = new_colors

    # ------------------------------------------------------------------
    # 5. Rebuild mesh with new vertex colours (original coordinates kept)
    # ------------------------------------------------------------------
    new_colors_u8 = np.clip(colors_f, 0, 255).astype(np.uint8)
    result = mesh.copy()
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result,
        vertex_colors=new_colors_u8,
    )
    return result
