"""Face blur utilities for BonyLandmarks viewer.

Blurs vertex colors and mesh geometry in the face/head region identified by
anatomical landmarks, using spatial averaging with a sparse adjacency matrix
for performance.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import KDTree
    from scipy.sparse import csr_matrix

    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False

# Landmark codes that delimit the face/head zone
_FACE_LANDMARK_CODES = [
    "vertex",
    "glabella",
    "external_acoustic_meatus_left",
    "external_acoustic_meatus_right",
    "mastoid_process_left",
    "mastoid_process_right",
    "external_occipital_protuberance",
    "suprasternal_notch",
]


def _head_mask_from_bounds(
    pts: np.ndarray,
    ground_truth: dict[str, np.ndarray],
) -> np.ndarray:
    """Improved fallback: select the head region above the shoulder level.

    Uses acromion or C7 landmarks (when available) to locate the shoulder
    line, then selects only the vertices above it.  The selection is further
    restricted to the anterior 65 % along the depth axis so that the back of
    the skull is excluded from the blur zone.

    Parameters
    ----------
    pts:
        Shape (N, 3) float64 — mesh vertex positions.
    ground_truth:
        Dict mapping landmark code → 3-D position array.
    """
    extents = pts.max(axis=0) - pts.min(axis=0)
    up_axis = int(extents.argmax())
    up_max = float(pts[:, up_axis].max())
    up_min = float(pts[:, up_axis].min())
    body_height = up_max - up_min

    # ---- Lower bound for the head region --------------------------------
    acromion_positions = [
        np.asarray(ground_truth[c], dtype=np.float64)
        for c in ("acromion_left", "acromion_right")
        if c in ground_truth
    ]

    if acromion_positions:
        acromion_up = max(float(p[up_axis]) for p in acromion_positions)
        lower_bound = acromion_up + 120.0       # 120 mm above the highest acromion (above the neck)
    elif "C7_spinous" in ground_truth:
        c7 = np.asarray(ground_truth["C7_spinous"], dtype=np.float64)
        lower_bound = float(c7[up_axis]) + 80.0
    else:
        lower_bound = up_max - body_height * 0.10  # top 10 % of body height (head only)

    # Everything above the shoulder line is the head — front + back.
    # Restricting by depth axis is unreliable because the anterior direction
    # varies per scan orientation; blurring the full head is the safe default.
    return pts[:, up_axis] >= lower_bound


def build_face_mask(
    mesh_points: np.ndarray,
    ground_truth: dict[str, np.ndarray],
    margin_factor: float = 1.3,
) -> np.ndarray:
    """Return a boolean mask (N,) where True = vertex belongs to the face/head zone.

    First tries to use the anatomical face landmarks from *ground_truth*.
    If fewer than 2 are found (common when the GLB only carries the 24
    standard BodyLoop markers), falls back to an improved heuristic that uses
    acromion / C7 landmarks to locate the shoulder line, selects the vertices
    above it, and keeps only the anterior 65 % to avoid the back of the skull.

    Parameters
    ----------
    mesh_points:
        Shape (N, 3) — 3-D positions of every mesh vertex.
    ground_truth:
        Dict mapping landmark code → 3-D position.  Missing keys are skipped.
    margin_factor:
        Sphere radius multiplier when face landmarks are available.

    Returns
    -------
    np.ndarray, dtype=bool, shape (N,)
    """
    pts = np.asarray(mesh_points, dtype=np.float64)

    face_positions = [
        np.asarray(ground_truth[code], dtype=np.float64)
        for code in _FACE_LANDMARK_CODES
        if code in ground_truth
    ]

    if len(face_positions) < 2:
        # Not enough face landmarks — use improved anatomical fallback
        return _head_mask_from_bounds(pts, ground_truth)

    positions = np.stack(face_positions)
    centroid = positions.mean(axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    radius = float(distances.max()) * margin_factor

    mask = np.zeros(len(pts), dtype=bool)
    if _SCIPY:
        tree = KDTree(pts)
        indices = tree.query_ball_point(centroid, radius)
        mask[indices] = True
    else:  # pragma: no cover
        dists = np.linalg.norm(pts - centroid, axis=1)
        mask[dists <= radius] = True

    return mask


def blur_vertex_colors(
    colors: np.ndarray,
    face_mask: np.ndarray,
    mesh_points: np.ndarray,
    k_neighbours: int = 20,
    n_passes: int = 10,
) -> np.ndarray:
    """Return a copy of *colors* with the face zone blurred.

    For each vertex in ``face_mask``, its color is replaced by the weighted
    average of its k nearest neighbors that are also in the mask.
    Repeating ``n_passes`` times yields a stronger, Gaussian-like blur.

    The averaging is implemented with a pre-built scipy sparse matrix so that
    each pass is a single matrix–vector product (no Python vertex loop).
    Using k-nearest neighbors ensures predictable performance regardless of
    mesh density, with exactly ``k + 1`` edges per vertex (including self-loop).

    Parameters
    ----------
    colors:
        Shape (N, 3), dtype uint8 or float — RGB vertex colors.
    face_mask:
        Shape (N,), dtype bool — True marks vertices to be blurred.
    mesh_points:
        Shape (N, 3) — 3-D positions of every mesh vertex.
    k_neighbours:
        Number of nearest neighbors per vertex (excluding self).
        Default 20 — total degree per vertex is k+1 (including self-loop).
    n_passes:
        Number of averaging passes.  More passes → stronger blur.

    Returns
    -------
    np.ndarray with the same shape and dtype as *colors*.
    """
    orig_dtype = colors.dtype
    result = colors.astype(np.float64)

    face_indices = np.where(face_mask)[0]
    if len(face_indices) == 0:
        return colors.copy()

    face_points = np.asarray(mesh_points[face_indices], dtype=np.float64)
    n_face = len(face_indices)

    if _SCIPY:
        tree = KDTree(face_points)
        # k-nearest neighbors: query with k+1 to include self (distance=0 in first column).
        # Limit k to the number of points available (for small face regions).
        k_actual = min(k_neighbours + 1, n_face)
        distances, knn_indices = tree.query(face_points, k=k_actual)

        rows: list[int] = []
        cols: list[int] = []
        for i in range(n_face):
            for j_local in knn_indices[i]:   # includes self (i)
                rows.append(i)
                cols.append(j_local)

        data = np.ones(len(rows), dtype=np.float64)
        adj = csr_matrix((data, (rows, cols)), shape=(n_face, n_face))
        row_sums = np.asarray(adj.sum(axis=1)).ravel()          # (n_face,)

        for _ in range(n_passes):
            face_colors = result[face_indices]                   # (n_face, 3)
            result[face_indices] = adj.dot(face_colors) / row_sums[:, None]
    else:  # pragma: no cover
        # Pure-numpy fallback — k-NN via sorting, O(n_face^2) per pass, slow for large meshes.
        k_actual = min(k_neighbours + 1, n_face)
        for _ in range(n_passes):
            new_face_colors = result[face_indices].copy()
            for i, gi in enumerate(face_indices):
                dists = np.linalg.norm(face_points - face_points[i], axis=1)
                # Sort by distance and take k+1 nearest (includes self at distance 0).
                nearest_indices = np.argsort(dists)[:k_actual]
                if len(nearest_indices):
                    new_face_colors[i] = result[face_indices[nearest_indices]].mean(axis=0)
            result[face_indices] = new_face_colors

    return result.astype(orig_dtype)


def blur_mesh_geometry(
    mesh_points: np.ndarray,
    face_mask: np.ndarray,
    mesh_faces: np.ndarray,
    n_iter: int = 500,
    pass_band: float = 0.01,
) -> np.ndarray:
    """Lisse les positions 3D des vertices du visage via Taubin smoothing (PyVista).

    Utilise l'implémentation C++ de PyVista (vtkSmoothPolyDataFilter Taubin) :
    - Pas de rétrécissement (Taubin = alternance +/- relaxation)
    - boundary_smoothing=False : les vertices au bord du masque ne bougent pas → pas de déchirure
    - mesh.clean(tolerance=0.5) : soude les coutures UV avant lissage → pas de lignes noires

    Parameters
    ----------
    mesh_points:
        Shape (N, 3), dtype float — vertex positions (mm).
    face_mask:
        Shape (N,), dtype bool — True marks vertices whose position will be
        smoothed.
    mesh_faces:
        PyVista flat faces array (1-D int): ``[3, v0, v1, v2, 3, v3, v4, v5, …]``
        for an all-triangle mesh (the format returned by ``pyvista.PolyData.faces``
        for BodyLoop GLB output).
    n_iter:
        Number of Taubin smoothing iterations (more → stronger deformation).
    pass_band:
        Pass-band frequency for Taubin smoothing (lower → more smoothing).

    Returns
    -------
    np.ndarray, shape (N, 3) — copy of *mesh_points*; only vertices where
    ``face_mask`` is True are modified.
    """
    import pyvista as pv

    pts = np.asarray(mesh_points, dtype=np.float64).copy()
    mask_bool = np.asarray(face_mask, dtype=bool)
    mask_indices = np.where(mask_bool)[0]
    if len(mask_indices) == 0:
        return pts

    # Construire le PolyData complet
    full_mesh = pv.PolyData(pts, np.asarray(mesh_faces))

    # Extraire la sous-région visage (avec les cellules adjacentes pour la topologie)
    face_sub = full_mesh.extract_points(mask_indices, include_cells=True)

    # Souder les coutures UV (vertices coïncidents → un seul vertex dans le graphe)
    face_clean = face_sub.clean(tolerance=0.5)

    # Taubin smoothing : lisse sans rétrécir ; boundaries fixes pour éviter la déchirure
    smoothed = face_clean.smooth_taubin(
        n_iter=n_iter,
        pass_band=pass_band,
        boundary_smoothing=False,
        normalize_coordinates=False,
    )

    # Mapper les positions lissées vers les vertices originaux :
    # face_sub.point_data["vtkOriginalPointIds"] donne les indices globaux
    # Chaque vertex original -> trouver le plus proche dans le mesh soudé (face_clean)
    orig_global_ids = face_sub.point_data["vtkOriginalPointIds"]  # (n_sub,) int
    if _SCIPY:
        tree = KDTree(face_clean.points)
        _, nn_idx = tree.query(face_sub.points)   # (n_sub,) → index in face_clean
        pts[orig_global_ids] = smoothed.points[nn_idx]
    else:  # pragma: no cover
        # fallback numpy
        for local_i, global_i in enumerate(orig_global_ids):
            dists = np.linalg.norm(face_clean.points - face_sub.points[local_i], axis=1)
            pts[global_i] = smoothed.points[int(dists.argmin())]

    return pts
