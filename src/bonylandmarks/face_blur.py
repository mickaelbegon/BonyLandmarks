"""Face blur utilities for BonyLandmarks viewer.

Blurs vertex colors and mesh geometry in the face/head region identified by
anatomical landmarks, using spatial averaging with a sparse adjacency matrix
for performance.
"""
from __future__ import annotations

import collections
import numpy as np

try:
    from scipy.spatial import KDTree
    from scipy.sparse import csr_matrix, diags

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
        lower_bound = acromion_up + 30.0        # 30 mm above the highest acromion
    elif "C7_spinous" in ground_truth:
        c7 = np.asarray(ground_truth["C7_spinous"], dtype=np.float64)
        lower_bound = float(c7[up_axis]) + 50.0
    else:
        lower_bound = up_max - body_height * 0.15  # top 15 % of body height

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
    blur_radius: float = 50.0,
    n_passes: int = 10,
) -> np.ndarray:
    """Return a copy of *colors* with the face zone blurred.

    For each vertex in ``face_mask``, its color is replaced by the weighted
    average of all neighbors within ``blur_radius`` mm that are also in the
    mask.  Repeating ``n_passes`` times yields a stronger, Gaussian-like blur.

    The averaging is implemented with a pre-built scipy sparse matrix so that
    each pass is a single matrix–vector product (no Python vertex loop).

    Parameters
    ----------
    colors:
        Shape (N, 3), dtype uint8 or float — RGB vertex colors.
    face_mask:
        Shape (N,), dtype bool — True marks vertices to be blurred.
    mesh_points:
        Shape (N, 3) — 3-D positions of every mesh vertex.
    blur_radius:
        Neighbourhood radius in the same unit as *mesh_points* (mm).
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
        # Build a symmetric sparse adjacency matrix with self-loops.
        # query_pairs returns all (i, j) pairs with i < j and dist <= blur_radius.
        pairs = tree.query_pairs(r=blur_radius)

        rows: list[int] = list(range(n_face))   # self-loops (diagonal)
        cols: list[int] = list(range(n_face))
        for i, j in pairs:
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

        data = np.ones(len(rows), dtype=np.float64)
        adj = csr_matrix((data, (rows, cols)), shape=(n_face, n_face))
        row_sums = np.asarray(adj.sum(axis=1)).ravel()          # (n_face,)

        for _ in range(n_passes):
            face_colors = result[face_indices]                   # (n_face, 3)
            result[face_indices] = adj.dot(face_colors) / row_sums[:, None]
    else:  # pragma: no cover
        # Pure-numpy fallback — O(n_face^2) per pass, slow for large meshes.
        for _ in range(n_passes):
            new_face_colors = result[face_indices].copy()
            for i, gi in enumerate(face_indices):
                dists = np.linalg.norm(face_points - mesh_points[gi], axis=1)
                nbrs = np.where(dists <= blur_radius)[0]
                if len(nbrs):
                    new_face_colors[i] = result[face_indices[nbrs]].mean(axis=0)
            result[face_indices] = new_face_colors

    return result.astype(orig_dtype)


def _parse_pyvista_faces(faces_arr: np.ndarray) -> np.ndarray:
    """Parse a non-uniform PyVista faces array into an (M, 3) triangle array.

    Polygons with more than 3 vertices are fan-triangulated from the first
    vertex.  This is a fallback for meshes that are not all-triangle.
    """
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
    n_passes: int = 30,
    relax: float = 0.25,
) -> np.ndarray:
    """Return a copy of *mesh_points* with face vertex positions Laplacian-smoothed.

    Applies iterative Laplacian smoothing to the 3-D positions of all vertices
    flagged by ``face_mask``, deforming the geometry to anonymise the scan.
    All topological neighbours (inside and outside the mask) contribute to the
    average so that boundary vertices do not shrink toward the interior.

    The averaging matrix is built once from the mesh topology using a scipy
    sparse matrix and then applied in a tight loop (no Python per-vertex loop).

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
    n_passes:
        Number of Laplacian smoothing passes (more → stronger deformation).
    relax:
        Relaxation factor λ ∈ (0, 1]:
        ``new_pos = (1 - λ) * old_pos + λ * mean(neighbours)``.

    Returns
    -------
    np.ndarray, shape (N, 3) — copy of *mesh_points*; only vertices where
    ``face_mask`` is True are modified.
    """
    pts = np.asarray(mesh_points, dtype=np.float64).copy()
    mask_bool = np.asarray(face_mask, dtype=bool)
    N = len(pts)

    mask_indices = np.where(mask_bool)[0]
    if len(mask_indices) == 0:
        return pts

    faces_arr = np.asarray(mesh_faces)

    # ---- Parse triangles from PyVista faces array -----------------------
    # BodyLoop meshes are all-triangle → reshape to (T, 4) and drop count col.
    try:
        triangles = faces_arr.reshape(-1, 4)[:, 1:]  # (T, 3) int
    except ValueError:
        # Non-uniform face sizes: parse manually with fan-triangulation.
        triangles = _parse_pyvista_faces(faces_arr)

    if _SCIPY:
        # ---- Build row-normalised averaging (Laplacian) matrix ----------
        # Enumerate all directed edges so each undirected edge appears twice.
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        rows = np.concatenate([v0, v1, v1, v2, v0, v2])
        cols = np.concatenate([v1, v0, v2, v1, v2, v0])
        data = np.ones(len(rows), dtype=np.float64)

        # ---- Weld UV-seam vertices (same 3-D position, different UV) ----
        # At UV seams a vertex is duplicated into two indices with identical
        # positions but different UV coordinates.  The topology graph has no
        # edge between them, so Laplacian smoothing moves them independently,
        # creating a small gap that renders as a black grid line.  Detect all
        # such pairs within the face mask and add a unit edge so they are
        # treated as neighbours.
        face_pts = pts[mask_indices]
        weld_tree = KDTree(face_pts)
        weld_pairs = weld_tree.query_pairs(r=0.5)   # 0.5 mm tolerance
        if weld_pairs:
            wp_arr = np.array(list(weld_pairs), dtype=np.int64)
            gi = mask_indices[wp_arr[:, 0]]
            gj = mask_indices[wp_arr[:, 1]]
            rows = np.concatenate([rows, gi, gj])
            cols = np.concatenate([cols, gj, gi])
            data = np.concatenate([data, np.ones(len(gi), dtype=np.float64),
                                         np.ones(len(gj), dtype=np.float64)])

        L_raw = csr_matrix((data, (rows, cols)), shape=(N, N))

        # Row-normalise: each non-zero entry becomes 1 / degree(i).
        row_sums = np.asarray(L_raw.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0          # guard against isolated vertices
        L_norm = diags(1.0 / row_sums) @ L_raw  # row-stochastic, shape (N, N)

        # ---- Soft boundary weights ---------------------------------------
        # Vertices at the mask boundary get a smaller effective relax so the
        # transition is smooth rather than a hard tear.
        # soft_weight[i] = fraction of topological neighbours inside the mask,
        # remapped from [0.3, 0.8] → [0, 1] and clamped.
        mask_float = mask_bool.astype(np.float64)
        nbr_mask_frac = np.asarray(L_norm.dot(mask_float)).ravel()     # (N,)
        soft_weight = np.clip((nbr_mask_frac - 0.3) / 0.5, 0.0, 1.0)  # (N,)
        # Shape (n_mask, 1) for broadcasting against (n_mask, 3) positions.
        w = (relax * soft_weight[mask_bool])[:, np.newaxis]

        # ---- Laplacian smoothing passes ---------------------------------
        for _ in range(n_passes):
            # avg[i] = mean of all topological neighbours of vertex i.
            # Non-masked vertices keep their original positions (fixed anchors)
            # → boundary vertices are pulled toward the surface edge, not inside.
            avg = L_norm.dot(pts)                   # (N, 3)
            pts[mask_bool] = (1.0 - w) * pts[mask_bool] + w * avg[mask_bool]

    else:  # pragma: no cover
        # Pure-numpy fallback — builds adjacency as a dict, slow for large meshes.
        adj: dict[int, set[int]] = collections.defaultdict(set)
        for v0i, v1i, v2i in triangles:
            adj[int(v0i)].update((int(v1i), int(v2i)))
            adj[int(v1i)].update((int(v0i), int(v2i)))
            adj[int(v2i)].update((int(v0i), int(v1i)))

        for _ in range(n_passes):
            new_pts = pts.copy()
            for i in mask_indices:
                nbrs = list(adj[i])
                if nbrs:
                    new_pts[i] = (1.0 - relax) * pts[i] + relax * pts[nbrs].mean(axis=0)
            pts = new_pts

    return pts
