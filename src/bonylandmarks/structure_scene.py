"""Scène 3D fusionnée pour le quiz d'anatomie (os / muscles BodyParts3D).

Module sans Qt.  Il dépend de NumPy et de PyVista (lecture des OBJ + PolyData),
mais jamais d'un contexte OpenGL, donc il est testable hors écran.

Principe de performance
-----------------------
Un squelette complet compte ~200 structures, un système musculaire ~190.  Créer
un acteur VTK par structure est trop lent (une passe de rendu par acteur).  Ici :

* chaque mesh est lu **une seule fois** (cache mémoire par chemin) ;
* toutes les structures d'une scène sont fusionnées dans un unique
  :class:`MergedMesh` (points/normales/triangles concaténés) ;
* on n'affiche qu'**un seul PolyData** par « couche » (squelette, muscles) avec
  un tableau de couleurs par cellule ; changer la structure cible ou colorer un
  retour visuel = réécrire quelques couleurs, sans reconstruire le mesh ;
* le tableau ``sid`` (structure de chaque triangle affiché) sert au picking :
  ``cell_id -> sid -> id de structure``.

Repère : BodyParts3D d'origine (mm, X gauche = +X, Y antérieur = -Y, Z supérieur
= +Z).

Attribution : BodyParts3D, © The Database Center for Life Science, licensed
under CC Attribution-Share Alike 2.1 Japan.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import anatomy_catalog as _cat

# ─── Couleurs ─────────────────────────────────────────────────────────────────

COLOR_BONE = (232, 213, 184)          # #e8d5b8 — os (cohérent avec l'annotateur)
COLOR_BONE_GREY = (172, 175, 184)     # squelette de contexte dans la scène muscles
COLOR_TARGET = (0, 200, 255)          # cible IDENTIFIER : cyan vif
COLOR_OK = (34, 221, 85)              # LOCALISER : bonne structure (vert)
COLOR_WRONG = (255, 31, 31)           # LOCALISER : structure cliquée à tort (rouge)
COLOR_OPPOSITE = (255, 159, 26)       # LOCALISER : côté opposé (orange, demi-point)
MUSCLE_BASE = (150, 52, 48)           # rouge musculaire de base

BACKGROUND = "#0d0d18"


def rgb_hex(rgb: Sequence[int]) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def muscle_color(structure_id: str) -> tuple[int, int, int]:
    """Rouge musculaire légèrement varié, déterministe pour un identifiant."""
    h = zlib.crc32(structure_id.encode("utf-8"))
    lum = 0.82 + 0.36 * ((h & 0xFF) / 255.0)             # 0.82 .. 1.18
    dr = ((h >> 8) & 0xFF) / 255.0 * 40.0 - 12.0         # -12 .. +28
    dg = ((h >> 16) & 0xFF) / 255.0 * 26.0 - 8.0
    r = int(np.clip((MUSCLE_BASE[0] + dr) * lum, 0, 255))
    g = int(np.clip((MUSCLE_BASE[1] + dg) * lum, 0, 255))
    b = int(np.clip((MUSCLE_BASE[2] + dg * 0.6) * lum, 0, 255))
    return r, g, b


# ─── Chargement (cache mémoire) ───────────────────────────────────────────────


@dataclass
class StructureMesh:
    """Géométrie d'une structure : triangles indexés localement."""

    id: str
    points: np.ndarray      # (n, 3) float32
    normals: np.ndarray     # (n, 3) float32
    tris: np.ndarray        # (m, 3) int32


_MESH_CACHE: dict[str, StructureMesh | None] = {}


def clear_cache() -> None:
    """Vide le cache mémoire des meshes (tests / rechargement)."""
    _MESH_CACHE.clear()


def _read_mesh(path: Path, sid: str) -> StructureMesh | None:
    import pyvista as pv

    try:
        m = pv.read(str(path))
        if not isinstance(m, pv.PolyData):
            m = m.extract_surface()
        if not m.is_all_triangles:
            m = m.triangulate()
        if m.n_points == 0 or m.n_cells == 0:
            return None
        if "Normals" not in m.point_data:
            m = m.compute_normals(
                cell_normals=False, point_normals=True, split_vertices=False,
                auto_orient_normals=False,
            )
        tris = np.asarray(m.regular_faces, dtype=np.int32)
        pts = np.ascontiguousarray(m.points, dtype=np.float32)
        nrm = np.ascontiguousarray(m.point_data["Normals"], dtype=np.float32)
        return StructureMesh(sid, pts, nrm, tris)
    except Exception:
        return None


def load_structure_mesh(structure) -> StructureMesh | None:
    """Mesh d'une structure du catalogue (``None`` si absent/illisible)."""
    p = _cat.mesh_path(structure)
    if p is None:
        return None
    key = str(p)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = _read_mesh(p, structure.id)
    cached = _MESH_CACHE[key]
    if cached is not None and cached.id != structure.id:
        # même fichier partagé par deux ids : on ne modifie pas le cache
        return StructureMesh(structure.id, cached.points, cached.normals, cached.tris)
    return cached


# ─── Mesh fusionné ────────────────────────────────────────────────────────────


class MergedMesh:
    """Toutes les structures d'une couche dans un seul jeu de tableaux."""

    def __init__(self, meshes: Sequence[StructureMesh]) -> None:
        self.ids: list[str] = [m.id for m in meshes]
        self.index: dict[str, int] = {sid: i for i, sid in enumerate(self.ids)}
        if meshes:
            offs = np.cumsum([0] + [len(m.points) for m in meshes[:-1]])
            self.points = np.concatenate([m.points for m in meshes])
            self.normals = np.concatenate([m.normals for m in meshes])
            self.tris = np.concatenate(
                [m.tris + np.int32(o) for m, o in zip(meshes, offs)]
            ).astype(np.int32, copy=False)
            self.tri_sid = np.concatenate(
                [np.full(len(m.tris), i, dtype=np.int32) for i, m in enumerate(meshes)]
            )
            self.centroids = np.array(
                [m.points.mean(axis=0) for m in meshes], dtype=np.float64
            )
            self.bounds = np.array(
                [
                    [*m.points.min(axis=0), *m.points.max(axis=0)]
                    for m in meshes
                ],
                dtype=np.float64,
            )  # (K, 6) : xmin ymin zmin xmax ymax zmax
        else:
            self.points = np.zeros((0, 3), np.float32)
            self.normals = np.zeros((0, 3), np.float32)
            self.tris = np.zeros((0, 3), np.int32)
            self.tri_sid = np.zeros((0,), np.int32)
            self.centroids = np.zeros((0, 3))
            self.bounds = np.zeros((0, 6))

    def __len__(self) -> int:
        return len(self.ids)

    def visible_mask(self, hidden_ids: Sequence[str] = ()) -> np.ndarray:
        vis = np.ones(len(self.ids), dtype=bool)
        for sid in hidden_ids:
            i = self.index.get(sid)
            if i is not None:
                vis[i] = False
        return vis

    def polydata(self, visible: np.ndarray | None = None):
        """PolyData des structures visibles + tableau ``sid`` par cellule.

        Retourne ``(poly, cell_sid)`` où ``cell_sid[i]`` est l'index de la
        structure du triangle ``i``.  ``poly`` porte les normales de points
        (rendu lissé sans recalcul) ; les couleurs sont à poser avec
        :func:`set_cell_colors`.
        """
        import pyvista as pv

        if visible is None:
            tris, cell_sid = self.tris, self.tri_sid
        else:
            keep = visible[self.tri_sid]
            tris, cell_sid = self.tris[keep], self.tri_sid[keep]
        poly = pv.PolyData.from_regular_faces(self.points, tris)
        poly.point_data["Normals"] = self.normals
        poly.GetPointData().SetActiveNormals("Normals")
        return poly, cell_sid

    def bounds_of(self, ids: Sequence[str]) -> tuple[float, ...] | None:
        idx = [self.index[i] for i in ids if i in self.index]
        if not idx:
            return None
        b = self.bounds[idx]
        return (
            float(b[:, 0].min()), float(b[:, 3].max()),
            float(b[:, 1].min()), float(b[:, 4].max()),
            float(b[:, 2].min()), float(b[:, 5].max()),
        )


def build_merged(
    structures: Sequence,
    progress: Callable[[int, int], None] | None = None,
) -> MergedMesh:
    """Charge (avec cache) et fusionne les structures dont le mesh existe."""
    meshes: list[StructureMesh] = []
    n = len(structures)
    for i, s in enumerate(structures):
        m = load_structure_mesh(s)
        if m is not None:
            meshes.append(m)
        if progress is not None:
            progress(i + 1, n)
    return MergedMesh(meshes)


def set_cell_colors(poly, cell_sid: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Pose le tableau ``rgb`` (uint8, par cellule) et le renvoie (modifiable)."""
    rgb = np.ascontiguousarray(palette[cell_sid], dtype=np.uint8)
    poly.cell_data["rgb"] = rgb
    return poly.cell_data["rgb"]


def recolor(poly, cell_sid: np.ndarray, palette: np.ndarray,
            overrides: dict[int, tuple[int, int, int]] | None = None) -> None:
    """Réécrit les couleurs cellule par cellule (mise à jour en place)."""
    rgb = poly.cell_data["rgb"]
    rgb[:] = palette[cell_sid]
    for sidx, col in (overrides or {}).items():
        rgb[cell_sid == sidx] = col
    poly.GetCellData().GetArray("rgb").Modified()
    poly.Modified()


# ─── Couche affichée (un acteur) ──────────────────────────────────────────────


class SceneLayer:
    """Une couche de la scène : un MergedMesh, une palette, un seul acteur.

    ``ensure`` (re)construit l'acteur seulement si l'ensemble des structures
    masquées change ; ``recolor`` réécrit les couleurs en place (très rapide).
    """

    def __init__(self, name: str, merged: MergedMesh, palette: np.ndarray,
                 structures: dict) -> None:
        self.name = name
        self.merged = merged
        self.palette = np.ascontiguousarray(palette, dtype=np.uint8)
        self.structures = structures          # id -> Structure (attributs layer, region…)
        self.poly = None
        self.cell_sid: np.ndarray | None = None
        self.tris: np.ndarray | None = None   # triangles affichés (mêmes indices que cell_sid)
        self._starts: np.ndarray | None = None  # début de chaque structure dans tris
        self.actor = None
        self._hidden_key: frozenset | None = None
        self.rebuilds = 0                     # compteur (mesures / tests)

    def reset(self) -> None:
        """À appeler après ``plotter.clear()`` (l'acteur n'existe plus)."""
        self.poly = None
        self.cell_sid = None
        self.tris = None
        self._starts = None
        self.actor = None
        self._hidden_key = None

    def ensure(self, plotter, hidden_ids: Sequence[str] = ()) -> bool:
        """Garantit un acteur à jour ; ``True`` s'il a été reconstruit."""
        key = frozenset(hidden_ids)
        if self.poly is not None and key == self._hidden_key and self.actor is not None:
            return False
        vis = self.merged.visible_mask(key) if key else None
        self.poly, self.cell_sid = self.merged.polydata(vis)
        self.tris = self.merged.tris if vis is None else self.merged.tris[vis[self.merged.tri_sid]]
        self._starts = np.searchsorted(self.cell_sid, np.arange(len(self.merged) + 1))
        set_cell_colors(self.poly, self.cell_sid, self.palette)
        self.actor = plotter.add_mesh(
            self.poly, scalars="rgb", rgb=True, smooth_shading=False,
            ambient=0.3, diffuse=0.9, specular=0.2,
            show_scalar_bar=False, render=False, name=f"layer_{self.name}",
            pickable=True,
        )
        self.actor.prop.interpolation = "Phong"
        self._hidden_key = key
        self.rebuilds += 1
        return True

    def recolor(self, overrides: dict[str, tuple[int, int, int]] | None = None) -> None:
        """Palette de base + couleurs imposées ``{id_structure: rgb}``."""
        if self.poly is None or self.cell_sid is None:
            return
        idx = {
            self.merged.index[sid]: col
            for sid, col in (overrides or {}).items()
            if sid in self.merged.index
        }
        recolor(self.poly, self.cell_sid, self.palette, idx)

    def ids_where(self, pred: Callable[[object], bool]) -> list[str]:
        return [sid for sid in self.merged.ids if pred(self.structures.get(sid))]

    def id_at_cell(self, cell_id: int) -> str | None:
        if self.cell_sid is None or not (0 <= cell_id < len(self.cell_sid)):
            return None
        return self.merged.ids[int(self.cell_sid[cell_id])]


    def ray_pick(self, p0: np.ndarray, p1: np.ndarray) -> tuple[float, int] | None:
        """Premier triangle affiché touché par le segment ``p0 -> p1``.

        Retourne ``(t, indice_de_cellule)`` avec ``t`` dans [0, 1] (0 = ``p0``),
        ou ``None``.  Test des boîtes englobantes des structures (NumPy), puis
        Möller-Trumbore sur les seules structures traversées : quelques ms
        même sur 1,3 million de triangles, sans locator VTK à construire.
        """
        if self.tris is None or self._starts is None or not len(self.tris):
            return None
        m = self.merged
        d = p1 - p0
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        lo = (m.bounds[:, :3] - 0.5 - p0) * inv
        hi = (m.bounds[:, 3:] + 0.5 - p0) * inv
        tmin = np.minimum(lo, hi).max(axis=1)
        tmax = np.maximum(lo, hi).min(axis=1)
        present = self._starts[1:] > self._starts[:-1]
        cand = np.flatnonzero(present & (tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0))
        best_t, best_cell = np.inf, -1
        pts = m.points
        for k in cand[np.argsort(tmin[cand])]:
            if tmin[k] > best_t:
                break                                  # les suivantes sont plus loin
            a, b = int(self._starts[k]), int(self._starts[k + 1])
            tri = self.tris[a:b]
            v0 = pts[tri[:, 0]].astype(np.float64)
            e1 = pts[tri[:, 1]] - v0
            e2 = pts[tri[:, 2]] - v0
            h = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, h)
            ok = np.abs(det) > 1e-12
            f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            sv = p0 - v0
            u = f * np.einsum("ij,ij->i", sv, h)
            q = np.cross(sv, e1)
            v = f * (q @ d)
            t = f * np.einsum("ij,ij->i", e2, q)
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
            if hit.any():
                i = int(np.argmin(np.where(hit, t, np.inf)))
                if t[i] < best_t:
                    best_t, best_cell = float(t[i]), a + i
        return (best_t, best_cell) if best_cell >= 0 else None


def screen_ray(renderer, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    """Segment plan proche -> plan lointain (coordonnées monde) sous le pixel (x, y)."""
    out = []
    for z in (0.0, 1.0):
        renderer.SetDisplayPoint(float(x), float(y), z)
        renderer.DisplayToWorld()
        w = renderer.GetWorldPoint()
        out.append(np.array(w[:3], dtype=np.float64) / (w[3] if abs(w[3]) > 1e-12 else 1.0))
    return out[0], out[1]


def pick_structure(renderer, x: float, y: float,
                   layers: Sequence[SceneLayer]) -> tuple[str | None, str | None]:
    """``(nom_de_couche, id_structure)`` sous le pixel (x, y), sinon ``(None, None)``.

    La surface la plus proche parmi toutes les couches l'emporte : un os situé
    devant un muscle le masque, comme à l'écran.  (Un ``vtkCellPicker`` teste
    toutes les cellules de tous les acteurs : ~300 ms sur 1,3 M de triangles.)
    """
    p0, p1 = screen_ray(renderer, x, y)
    best_t, best = np.inf, (None, None)
    for layer in layers:
        r = layer.ray_pick(p0, p1)
        if r is not None and r[0] < best_t:
            best_t, best = r[0], (layer.name, layer.id_at_cell(r[1]))
    return best


# ─── Caméra ───────────────────────────────────────────────────────────────────

VIEW_FRONT, VIEW_BACK, VIEW_LEFT, VIEW_RIGHT = "front", "back", "left", "right"

#: (axe de la caméra, sens) pour scene3d.set_view — axes BP3D : 0=X, 1=Y, 2=Z.
VIEW_AXIS: dict[str, tuple[int, int]] = {
    VIEW_FRONT: (1, -1),   # antérieur = -Y : la caméra est en -Y
    VIEW_BACK: (1, +1),
    VIEW_LEFT: (0, +1),    # gauche du sujet = +X
    VIEW_RIGHT: (0, -1),
}

_ANT_LIMIT = 0.40
_POST_LIMIT = 0.60
_LATERAL_FRACTION = 0.10      # écart à la ligne médiane, en fraction de la largeur
_NEIGHBOUR_X = 50.0           # mm : voisinage (X) pour estimer l'épaisseur locale
_NEIGHBOUR_Z = 120.0          # mm : voisinage (Z)


def local_depth_span(
    centroid: Sequence[float], ref_bounds: np.ndarray
) -> tuple[float, float] | None:
    """Étendue antéro-postérieure (Y) du squelette autour de ``centroid``.

    On ne peut pas comparer à la profondeur globale du corps : le thorax est
    beaucoup plus épais que la jambe.  On prend donc les structures de
    référence dont la boîte englobante recoupe un voisinage en X et en Z du
    centroïde (± 50 / 120 mm).
    """
    rb = np.asarray(ref_bounds, dtype=float).reshape(-1, 6)
    if not len(rb):
        return None
    cx, _cy, cz = centroid
    near = (
        (rb[:, 0] <= cx + _NEIGHBOUR_X) & (rb[:, 3] >= cx - _NEIGHBOUR_X)
        & (rb[:, 2] <= cz + _NEIGHBOUR_Z) & (rb[:, 5] >= cz - _NEIGHBOUR_Z)
    )
    if not near.any():
        return None
    return float(rb[near, 1].min()), float(rb[near, 4].max())


def choose_view(
    centroid: Sequence[float],
    ref_bounds: np.ndarray,
    midline_x: float = 0.0,
    width: float = 650.0,
) -> str:
    """Vue (avant / arrière / gauche / droite) qui montre le mieux la cible.

    Position du centroïde de la cible dans la profondeur locale (Y) du corps
    (voir :func:`local_depth_span`) :

    * tiers antérieur (< 40 %)  -> vue de face ;
    * tiers postérieur (> 60 %) -> vue de dos ;
    * au milieu : structure latérale (écart à la ligne médiane > 10 % de la
      largeur) -> vue du côté correspondant (gauche = +X), sinon face/dos selon
      le côté le plus proche.
    """
    span = local_depth_span(centroid, ref_bounds)
    dx = (centroid[0] - midline_x) / max(width, 1e-6)
    if span is None or span[1] - span[0] < 1e-6:
        return (VIEW_LEFT if dx > 0 else VIEW_RIGHT) if abs(dx) > _LATERAL_FRACTION else VIEW_FRONT
    ny = (centroid[1] - span[0]) / (span[1] - span[0])
    if ny < _ANT_LIMIT:
        return VIEW_FRONT
    if ny > _POST_LIMIT:
        return VIEW_BACK
    if abs(dx) > _LATERAL_FRACTION:
        return VIEW_LEFT if dx > 0 else VIEW_RIGHT
    return VIEW_FRONT if ny <= 0.5 else VIEW_BACK


def focus_distance(extent_mm: float, lo: float = 350.0, hi: float = 1500.0) -> float:
    """Distance caméra pour cadrer une structure de taille ``extent_mm``."""
    return float(min(max(extent_mm * 4.0, lo), hi))


def bounds_center(b: Sequence[float]) -> list[float]:
    return [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]


def bounds_extent(b: Sequence[float]) -> float:
    return float(max(b[1] - b[0], b[3] - b[2], b[5] - b[4]))
