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
    acromion_offset: float = 60.0,
) -> np.ndarray:
    """Select the head/face region above the shoulder line.

    Parameters
    ----------
    pts:
        Shape (N, 3) float64 — mesh vertex positions.
    ground_truth:
        Dict mapping landmark code → 3-D position array.
    acromion_offset:
        Distance in mesh units (mm) above the highest acromion landmark
        where the mask starts.  Smaller values include more of the face/neck.
    """
    extents = pts.max(axis=0) - pts.min(axis=0)
    up_axis = int(extents.argmax())
    up_max = float(pts[:, up_axis].max())
    up_min = float(pts[:, up_axis].min())
    body_height = up_max - up_min

    acromion_positions = [
        np.asarray(ground_truth[c], dtype=np.float64)
        for c in ("acromion_left", "acromion_right")
        if c in ground_truth
    ]

    if acromion_positions:
        acromion_up = max(float(p[up_axis]) for p in acromion_positions)
        lower_bound = acromion_up + float(acromion_offset)
    elif "C7_spinous" in ground_truth:
        c7 = np.asarray(ground_truth["C7_spinous"], dtype=np.float64)
        lower_bound = float(c7[up_axis]) + float(acromion_offset) * 0.6
    else:
        lower_bound = up_max - body_height * 0.13  # top 13 % of body height

    return pts[:, up_axis] >= lower_bound


def build_face_mask(
    mesh_points: np.ndarray,
    ground_truth: dict[str, np.ndarray],
    margin_factor: float = 1.3,
    acromion_offset: float = 60.0,
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
        return _head_mask_from_bounds(pts, ground_truth, acromion_offset=acromion_offset)

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


def gray_face_colors(
    colors: np.ndarray,
    face_mask: np.ndarray,
    gray: int = 160,
) -> np.ndarray:
    """Remplace les couleurs de la zone visage par un gris uniforme (anonymisation totale).

    Parameters
    ----------
    colors:  Shape (N, 3), dtype uint8 ou float.
    face_mask: Shape (N,), dtype bool.
    gray: Valeur de gris 0-255 (défaut 160 = gris neutre).
    """
    result = colors.copy()
    face_indices = np.where(face_mask)[0]
    if len(face_indices) == 0:
        return result
    n_channels = result.shape[1] if result.ndim > 1 else 3
    fill = [gray] * min(n_channels, 3) + [255] * max(0, n_channels - 3)
    result[face_indices] = np.array(fill, dtype=result.dtype)
    return result


def blur_vertex_colors(
    colors: np.ndarray,
    face_mask: np.ndarray,
    mesh_points: np.ndarray,
    k_neighbours: int = 40,
    n_passes: int = 15,
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


def _parse_pyvista_faces(faces_arr: np.ndarray) -> np.ndarray:
    """Parse a non-uniform PyVista faces array into an (M, 3) triangle array."""
    triangles: list[tuple[int, int, int]] = []
    i = 0
    while i < len(faces_arr):
        n = int(faces_arr[i])
        verts = faces_arr[i + 1 : i + 1 + n]
        for k in range(1, n - 1):
            triangles.append((int(verts[0]), int(verts[k]), int(verts[k + 1])))
        i += n + 1
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(triangles, dtype=np.int64)


def blur_mesh_geometry(
    mesh_points: np.ndarray,
    face_mask: np.ndarray,
    mesh_faces: np.ndarray,
    n_iter: int = 750,
    pass_band: float = 0.005,
) -> np.ndarray:
    """Lisse les positions 3D des vertices du visage via Taubin smoothing (PyVista).

    Construit un sous-PolyData directement depuis les triangles masqués (évite
    extract_points qui retourne un UnstructuredGrid sans smooth_taubin).
    Les coutures UV sont soudées avec clean(tolerance=0.5) avant lissage.
    boundary_smoothing=False maintient les vertices de bord en place.

    Returns
    -------
    np.ndarray, shape (N, 3) — copie de mesh_points avec les vertices masqués lissés.
    """
    import pyvista as pv

    pts = np.asarray(mesh_points, dtype=np.float64).copy()
    mask_bool = np.asarray(face_mask, dtype=bool)
    if not mask_bool.any():
        return pts

    # Parse triangle faces
    faces_arr = np.asarray(mesh_faces)
    try:
        tris = faces_arr.reshape(-1, 4)[:, 1:].astype(np.int64)  # (T, 3)
    except ValueError:
        tris = _parse_pyvista_faces(faces_arr)

    if len(tris) == 0:
        return pts

    # Keep triangles that touch at least one masked vertex
    face_in_region = mask_bool[tris].any(axis=1)
    sub_tris = tris[face_in_region]  # (M, 3) global indices
    if len(sub_tris) == 0:
        return pts

    # Remap to contiguous local indices
    unique_global, inv = np.unique(sub_tris.ravel(), return_inverse=True)
    sub_pts = pts[unique_global]          # (n_local, 3)
    local_tris = inv.reshape(-1, 3)       # (M, 3) local indices

    # Build PolyData submesh (triangles only → smooth_taubin available)
    cell_arr = np.c_[
        np.full(len(local_tris), 3, dtype=np.int64), local_tris
    ].ravel()
    sub_poly = pv.PolyData(sub_pts, cell_arr)

    # Weld UV-seam duplicates
    sub_clean = sub_poly.clean(tolerance=0.5)

    # Taubin: smooth without shrinkage; boundary vertices stay fixed (no tear)
    smoothed = sub_clean.smooth_taubin(
        n_iter=n_iter,
        pass_band=pass_band,
        boundary_smoothing=False,
    )

    # Map smoothed positions back: for each local vertex find nearest in welded mesh
    if _SCIPY:
        tree = KDTree(sub_clean.points)
        _, nn_idx = tree.query(sub_pts)                  # (n_local,) → index in clean
        in_mask_local = mask_bool[unique_global]          # only update masked verts
        pts[unique_global[in_mask_local]] = smoothed.points[nn_idx[in_mask_local]]
    else:  # pragma: no cover
        for li, gi in enumerate(unique_global):
            if mask_bool[gi]:
                dists = np.linalg.norm(sub_clean.points - sub_pts[li], axis=1)
                pts[gi] = smoothed.points[int(dists.argmin())]

    return pts
