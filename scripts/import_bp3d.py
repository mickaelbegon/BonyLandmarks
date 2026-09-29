#!/usr/bin/env python
"""Importe des meshes BodyParts3D 4.0 (OBJ) vers ``src/bonylandmarks/data/``.

Lit ``anatomy_bones.json`` et/ou ``anatomy_muscles.json``, retrouve pour chaque
entrée le(s) fichier(s) ``FJ*.obj`` correspondant(s), décime au besoin et écrit
``data/<mesh_file>`` (``bones_full/`` pour les os, ``muscles/`` pour les muscles).

ATTENTION : le numéro dans ``FJxxxx.obj`` n'est PAS le numéro FMA.  Le concept
FMA est dans l'en-tête de chaque fichier (``# Concept ID : FMA13039``), c'est
donc l'en-tête qui est indexé.  Plusieurs fichiers peuvent porter le même FMA :

* ``src_files`` renseigné dans le JSON  -> ces fichiers-là uniquement ;
* sinon, tous les fichiers du FMA sont fusionnés en un seul mesh, après
  suppression des doublons exacts (même boîte englobante et même taille).

Les coordonnées d'origine BodyParts3D sont conservées telles quelles (aucun
recentrage) pour que les structures restent à leur place relative dans le corps.

Exemples ::

    python scripts/import_bp3d.py --kind bone
    python scripts/import_bp3d.py --kind all --target-faces 10000
    python scripts/import_bp3d.py --kind muscle --dry-run

BodyParts3D, © The Database Center for Life Science, licensed under
CC Attribution-Share Alike 2.1 Japan.
https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "src" / "bonylandmarks" / "data"
DEFAULT_SRC = Path.home() / "Downloads" / "isa_BP3D_4.0_obj_99"
ATTRIBUTION = (
    "BodyParts3D, (c) The Database Center for Life Science, licensed under "
    "CC Attribution-Share Alike 2.1 Japan"
)
JSON_BY_KIND = {"bone": "anatomy_bones.json", "muscle": "anatomy_muscles.json"}


# --------------------------------------------------------------------------
# Index des fichiers sources
# --------------------------------------------------------------------------
def find_obj_dir(src: Path) -> Path:
    """Dossier contenant les ``FJ*.obj`` (src lui-même ou un sous-dossier)."""
    for cand in (src, src / src.name, src / "isa_BP3D_4.0_obj_99"):
        if cand.is_dir() and any(cand.glob("FJ*.obj")):
            return cand
    for sub in sorted(p for p in src.glob("*") if p.is_dir()):
        if any(sub.glob("FJ*.obj")):
            return sub
    raise SystemExit(f"Aucun fichier FJ*.obj trouvé sous {src}")


def read_header(path: Path) -> tuple[str | None, str | None, str | None]:
    """(concept id, nom anglais, bounds) lus dans les premières lignes."""
    fma = name = bounds = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 30 or not line.startswith("#"):
                break
            if line.startswith("# Concept ID"):
                fma = line.split(":", 1)[1].strip()
            elif line.startswith("# English name"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("# Bounds"):
                bounds = line.split(":", 1)[1].strip()
    return fma, name, bounds


def build_index(obj_dir: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """``fma -> [fichiers]`` et ``stem -> fichier`` (dicts {stem,path,name,key})."""
    by_fma: dict[str, list[dict]] = {}
    by_stem: dict[str, dict] = {}
    for p in sorted(obj_dir.glob("FJ*.obj")):
        fma, name, bounds = read_header(p)
        rec = {"stem": p.stem, "path": p, "name": name or "",
               "key": (bounds, p.stat().st_size)}
        by_stem[p.stem] = rec
        if fma:
            by_fma.setdefault(fma, []).append(rec)
    return by_fma, by_stem


def resolve_sources(entry: dict, by_fma, by_stem) -> list[dict]:
    """Fichiers sources d'une entrée du catalogue (liste vide si introuvable)."""
    if entry.get("src_files"):
        return [by_stem[s] for s in entry["src_files"] if s in by_stem]
    recs = by_fma.get(entry.get("fma_id", ""), [])
    uniq, seen = [], set()
    for r in recs:  # doublons exacts (ex. os hyoïde présent 2 fois)
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        uniq.append(r)
    return uniq


# --------------------------------------------------------------------------
# Géométrie
# --------------------------------------------------------------------------
def load_merged(paths: list[Path]):
    """Charge et fusionne plusieurs OBJ -> (vertices (N,3), faces (M,3))."""
    import trimesh

    meshes = []
    for p in paths:
        m = trimesh.load(str(p), force="mesh", process=True)
        if len(m.faces):
            meshes.append(m)
    if not meshes:
        raise ValueError("mesh vide")
    m = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def decimate(verts: np.ndarray, faces: np.ndarray, target_faces: int):
    """Décime à ~target_faces triangles (coordonnées conservées)."""
    if target_faces <= 0 or len(faces) <= target_faces:
        return verts, faces
    import pyvista as pv

    cells = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    pd = pv.PolyData(verts, cells)
    reduction = 1.0 - target_faces / len(faces)
    try:
        out = pd.decimate(reduction)
    except Exception:
        out = pd.decimate_pro(reduction, preserve_topology=True)
    out = out.triangulate().clean()
    if out.n_cells == 0:  # échec silencieux : on garde l'original
        return verts, faces
    f = out.faces.reshape(-1, 4)[:, 1:]
    return np.asarray(out.points, dtype=np.float64), f.astype(np.int64)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normales sommet pondérées par l'aire."""
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vn / norm


def write_obj(path: Path, verts, faces, header_lines: list[str], normals: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for h in header_lines:
            fh.write(f"# {h}\n")
        for v in verts:
            fh.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        if normals:
            for n in vertex_normals(verts, faces):
                fh.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            for a, b, c in faces + 1:
                fh.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
        else:
            for a, b, c in faces + 1:
                fh.write(f"f {a} {b} {c}\n")


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------
def load_catalog(kind: str) -> list[dict]:
    p = DATA_DIR / JSON_BY_KIND[kind]
    if not p.is_file():
        print(f"[{kind}] {p.name} absent : rien à importer.")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="dossier BodyParts3D (contenant les FJ*.obj) [défaut: %(default)s]")
    ap.add_argument("--kind", choices=("bone", "muscle", "all"), default="all")
    ap.add_argument("--target-faces", type=int, default=15000,
                    help="nombre max de triangles par structure (0 = pas de décimation) [15000]")
    ap.add_argument("--dry-run", action="store_true", help="résout les fichiers sans rien écrire")
    ap.add_argument("--force", action="store_true", help="réécrit les fichiers déjà présents")
    ap.add_argument("--no-normals", action="store_true", help="n'écrit pas les normales (fichiers plus légers)")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not args.src.is_dir():
        print(f"Dossier source introuvable : {args.src}", file=sys.stderr)
        return 2
    obj_dir = find_obj_dir(args.src)
    print(f"Source : {obj_dir}")
    by_fma, by_stem = build_index(obj_dir)
    print(f"{len(by_stem)} fichiers OBJ indexés ({len(by_fma)} concepts FMA)")

    kinds = ("bone", "muscle") if args.kind == "all" else (args.kind,)
    missing: list[str] = []
    errors: list[str] = []
    written = skipped = 0
    total_bytes = 0

    for kind in kinds:
        entries = load_catalog(kind)
        for e in entries:
            srcs = resolve_sources(e, by_fma, by_stem)
            label = f"{e['id']} ({e.get('fma_id', '?')})"
            if not srcs:
                missing.append(label)
                continue
            dest = DATA_DIR / e["mesh_file"]
            if dest.is_file() and not args.force and not args.dry_run:
                skipped += 1
                total_bytes += dest.stat().st_size
                continue
            if args.dry_run:
                print(f"  {label} <- {'+'.join(s['stem'] for s in srcs)} -> {e['mesh_file']}")
                continue
            try:
                verts, faces = load_merged([s["path"] for s in srcs])
                n0 = len(faces)
                verts, faces = decimate(verts, faces, args.target_faces)
                header = [
                    ATTRIBUTION,
                    "https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html",
                    f"Concept ID : {e.get('fma_id', '')}",
                    f"English name : {e.get('name_en', '')}",
                    "Source files : " + ", ".join(s["stem"] for s in srcs),
                    f"Faces : {len(faces)} (source: {n0}); coordinates: original BodyParts3D (mm)",
                ]
                write_obj(dest, verts, faces, header, normals=not args.no_normals)
                written += 1
                total_bytes += dest.stat().st_size
            except Exception as exc:  # un mesh défectueux ne doit pas arrêter le lot
                errors.append(f"{label}: {exc}")

    print()
    if args.dry_run:
        print("Dry-run : aucun fichier écrit.")
    else:
        print(f"Écrits : {written}, déjà présents (ignorés) : {skipped}, "
              f"taille totale : {total_bytes / 1e6:.1f} Mo")
    if missing:
        print(f"FMA introuvables dans la source ({len(missing)}) :")
        for m in missing:
            print(f"  - {m}")
    if errors:
        print(f"Erreurs ({len(errors)}) :")
        for m in errors:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
