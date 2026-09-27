"""PyVista camera and scene helpers shared across 3D exercises.

All helper functions take the plotter as first argument so they stay stateless
and easy to test. Exercises store a ``CameraAxes`` instance and pass it through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraAxes:
    """Axis metadata derived from a mesh's bounding box."""

    up: int = 2
    front: int = 1
    side: int = 0
    centers: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    distance: float = 500.0


def fit_camera_to_mesh(mesh, distance_factor: float = 2.2) -> CameraAxes:
    """Infer camera axes from a mesh bounding box.

    Parameters
    ----------
    mesh:
        Any PyVista mesh with a ``bounds`` property.
    distance_factor:
        Multiplied by the longest extent to compute the camera stand-off.
        Use 2.2 for individual bones, 1.8 for a full skeleton.
    """
    b = mesh.bounds
    extents = [b[1] - b[0], b[3] - b[2], b[5] - b[4]]
    up = extents.index(max(extents))
    sorted_axes = sorted(range(3), key=lambda i: extents[i])
    front = sorted_axes[0]
    side = ({0, 1, 2} - {up, front}).pop()
    centers = [(b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5]
    distance = max(extents) * distance_factor
    return CameraAxes(up=up, front=front, side=side, centers=centers, distance=distance)


def set_view(plotter, axes: CameraAxes, cam_axis: int, direction: int) -> None:
    """Point the camera along *cam_axis* in *direction* (±1)."""
    pos = list(axes.centers)
    pos[cam_axis] += direction * axes.distance
    up_vec = [0.0, 0.0, 0.0]
    up_vec[axes.up] = 1.0
    cam = plotter.camera
    cam.position = tuple(pos)
    cam.focal_point = tuple(axes.centers)
    cam.up = tuple(up_vec)
    plotter.render()


def view_top(plotter, axes: CameraAxes) -> None:
    """Point the camera from above (along the up axis)."""
    pos = list(axes.centers)
    pos[axes.up] += axes.distance
    up_vec = [0.0, 0.0, 0.0]
    up_vec[axes.front] = 1.0
    cam = plotter.camera
    cam.position = tuple(pos)
    cam.focal_point = tuple(axes.centers)
    cam.up = tuple(up_vec)
    plotter.render()


def reset_view(plotter, axes: CameraAxes) -> None:
    """Reset to front view (−front_axis direction)."""
    set_view(plotter, axes, axes.front, -1)


def pan_camera(plotter, dx: float, dy: float) -> None:
    """Pan camera by (dx, dy) in normalised screen-space units (step = 8% of dist)."""
    cam = plotter.camera
    pos = np.asarray(cam.position, dtype=float)
    fpt = np.asarray(cam.focal_point, dtype=float)
    up = np.asarray(cam.up, dtype=float)
    view_dir = fpt - pos
    dist = float(np.linalg.norm(view_dir))
    if dist < 1e-9:
        return
    view_dir /= dist
    right = np.cross(view_dir, up)
    r_norm = float(np.linalg.norm(right))
    if r_norm < 1e-9:
        return
    right /= r_norm
    up_perp = np.cross(right, view_dir)
    step = dist * 0.08
    delta = right * dx * step + up_perp * dy * step
    cam.position = tuple(pos + delta)
    cam.focal_point = tuple(fpt + delta)
    plotter.render()


def compute_curvature(mesh) -> "pv.PolyData":
    """Smooth a mesh and compute mean-curvature scalars.

    Returns a new PolyData with ``"curvature"`` point data and
    ``field_data["curv_clim"]`` = [lo, hi] (P5–P95 clamped range).
    Callers are responsible for caching the result when needed.
    """
    smoothed = mesh.smooth(n_iter=50, relaxation_factor=0.05)
    curv = smoothed.curvature(curv_type="mean")
    lo = float(np.percentile(curv, 5))
    hi = float(np.percentile(curv, 95))
    if abs(hi - lo) < 1e-9:
        hi = lo + 1e-6
    smoothed["curvature"] = curv
    smoothed.field_data["curv_clim"] = np.array([lo, hi])
    return smoothed


def render_curvature(plotter, curv_mesh) -> None:
    """Render a pre-computed curvature mesh (from :func:`compute_curvature`)."""
    lo, hi = float(curv_mesh.field_data["curv_clim"][0]), float(
        curv_mesh.field_data["curv_clim"][1]
    )
    plotter.clear()
    plotter.add_mesh(
        curv_mesh,
        scalars="curvature",
        clim=[lo, hi],
        cmap="coolwarm",
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Courbure",
            "vertical": True,
            "height": 0.4,
            "position_x": 0.02,
            "position_y": 0.3,
            "title_font_size": 10,
            "label_font_size": 9,
        },
        ambient=0.3,
        diffuse=0.9,
        render=False,
        pickable=False,
    )
    plotter.render()


def apply_curvature_heatmap(plotter, mesh) -> None:
    """Convenience: compute curvature then render immediately (no caching).

    Prefer the :func:`compute_curvature` + :func:`render_curvature` split
    when caching is needed (e.g. repeated toggles on the same bone).
    """
    render_curvature(plotter, compute_curvature(mesh))
