#!/usr/bin/env python
"""Convertit les meshes OBJ existants (bones/, bones_full/, muscles/) en PLY binaire.

Aucune re-décimation, aucun recentrage : sommets et faces sont conservés tels
quels (sommets en float32).  Les OBJ sources ne sont PAS supprimés.  Chaque
conversion est vérifiée par relecture (nombre de sommets/faces identique,
boîte englobante identique à la précision float32).

Exemples ::

    python scripts/convert_meshes.py --dry-run
    python scripts/convert_meshes.py
    python scripts/convert_meshes.py --force --dirs bones

BodyParts3D, (c) The Database Center for Life Science, licensed under
CC Attribution-Share Alike 2.1 Japan.
https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _meshio import read_obj, read_ply, write_ply  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "src" / "bonylandmarks" / "data"
SUBDIRS = ("bones", "bones_full", "muscles")
ATTRIBUTION = ("BodyParts3D, (c) The Database Center for Life Science, "
               "licensed under CC Attribution-Share Alike 2.1 Japan")


def convert_one(obj: Path, ply: Path) -> tuple[int, int]:
    """Convertit et vérifie ; renvoie (nb sommets, nb faces)."""
    verts, faces, header = read_obj(obj)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("mesh vide")
    comments = [h for h in header if h][:20]
    if not any("BodyParts3D" in c for c in comments):
        comments.insert(0, ATTRIBUTION)
    comments.append("https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html")
    comments.append("Converted from OBJ to binary PLY; coordinates unchanged (float32)")
    write_ply(ply, verts, faces, comments)

    v2, f2 = read_ply(ply)
    if v2.shape != verts.shape or f2.shape != faces.shape:
        raise ValueError(f"nombre de sommets/faces différent ({v2.shape} / {f2.shape})")
    if not np.array_equal(f2, faces):
        raise ValueError("indices de faces différents")
    if not np.allclose(v2.min(0), verts.min(0), atol=1e-3, rtol=1e-6) or \
            not np.allclose(v2.max(0), verts.max(0), atol=1e-3, rtol=1e-6):
        raise ValueError("boîte englobante différente")
    return len(verts), len(faces)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DATA_DIR,
                    help="dossier data/ [défaut: %(default)s]")
    ap.add_argument("--dirs", nargs="+", default=list(SUBDIRS), choices=SUBDIRS,
                    help="sous-dossiers à convertir [tous]")
    ap.add_argument("--dry-run", action="store_true", help="liste sans écrire")
    ap.add_argument("--force", action="store_true", help="réécrit les PLY déjà présents")
    args = ap.parse_args(argv)

    n_conv = n_skip = 0
    size_obj = size_ply = 0
    errors: list[str] = []
    for sub in args.dirs:
        d = args.data / sub
        if not d.is_dir():
            print(f"[{sub}] dossier absent, ignoré")
            continue
        objs = sorted(d.glob("*.obj"))
        sub_obj = sub_ply = 0
        for obj in objs:
            ply = obj.with_suffix(".ply")
            if ply.is_file() and not args.force:
                n_skip += 1
                continue
            if args.dry_run:
                print(f"  {obj.relative_to(args.data)} -> {ply.name}")
                continue
            try:
                convert_one(obj, ply)
                n_conv += 1
                sub_obj += obj.stat().st_size
                sub_ply += ply.stat().st_size
            except Exception as exc:
                errors.append(f"{obj.name}: {exc}")
                ply.unlink(missing_ok=True)
        size_obj += sub_obj
        size_ply += sub_ply
        if not args.dry_run:
            print(f"[{sub}] {len(objs)} OBJ  |  converti: {sub_obj / 1e6:.1f} Mo OBJ -> "
                  f"{sub_ply / 1e6:.1f} Mo PLY")

    print()
    if args.dry_run:
        print("Dry-run : aucun fichier écrit.")
    else:
        print(f"Convertis : {n_conv}, déjà présents (ignorés) : {n_skip}, erreurs : {len(errors)}")
        if size_obj:
            print(f"Total converti : {size_obj / 1e6:.1f} Mo OBJ -> {size_ply / 1e6:.1f} Mo PLY")
    for e in errors:
        print(f"  ERREUR {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
