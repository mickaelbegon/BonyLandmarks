"""Lecture OBJ / écriture PLY binaire — utilitaires communs aux scripts.

Format PLY produit : ``binary_little_endian 1.0``, sommets ``float32`` (x, y, z),
faces = triangles (``list uchar int``), sans normales ni couleurs.  Les
coordonnées sont celles de la source (aucun recentrage).

Lisible par pyvista/VTK, trimesh, MeshLab, Blender.

BodyParts3D, (c) The Database Center for Life Science, licensed under
CC Attribution-Share Alike 2.1 Japan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_FACE_DTYPE = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Lit un OBJ -> (sommets float64 (N,3), faces int64 (M,3), lignes d'en-tête ``#``).

    Sommets non fusionnés/non réordonnés ; polygones triangulés en éventail ;
    indices négatifs (relatifs) gérés ; ``vt``/``vn`` ignorés.
    """
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    header: list[str] = []
    in_header = True
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if in_header:
                if line.startswith("#"):
                    header.append(line[1:].strip())
                    continue
                if line.strip():
                    in_header = False
                else:
                    continue
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return (np.asarray(verts, dtype=np.float64).reshape(-1, 3),
            np.asarray(faces, dtype=np.int64).reshape(-1, 3), header)


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray,
              comments: list[str] | None = None) -> None:
    """Écrit un PLY binaire little-endian (sommets float32, faces triangles)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.ascontiguousarray(verts, dtype="<f4").reshape(-1, 3)
    f = np.empty(len(faces), dtype=_FACE_DTYPE)
    f["n"] = 3
    f["idx"] = np.asarray(faces, dtype="<i4").reshape(-1, 3)
    lines = ["ply", "format binary_little_endian 1.0"]
    for c in comments or []:
        c = " ".join(c.split()).encode("ascii", "replace").decode("ascii")
        if c:
            lines.append(f"comment {c}")
    lines += [f"element vertex {len(v)}", "property float x", "property float y",
              "property float z", f"element face {len(f)}",
              "property list uchar int vertex_indices", "end_header"]
    with open(path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode("ascii"))
        fh.write(v.tobytes())
        fh.write(f.tobytes())


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Relit un PLY produit par :func:`write_ply` -> (sommets float32, faces int32)."""
    with open(path, "rb") as fh:
        nv = nf = 0
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith("element face"):
                nf = int(line.split()[-1])
            elif line == "end_header":
                break
            elif not line:
                raise ValueError(f"En-tête PLY invalide : {path}")
        v = np.frombuffer(fh.read(nv * 12), dtype="<f4").reshape(nv, 3)
        f = np.frombuffer(fh.read(nf * _FACE_DTYPE.itemsize), dtype=_FACE_DTYPE)
    return v.copy(), f["idx"].copy()
