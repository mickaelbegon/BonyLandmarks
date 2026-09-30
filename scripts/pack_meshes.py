#!/usr/bin/env python
"""Empaquette les meshes PLY en un zip publiable comme asset de GitHub Release.

Contenu de ``dist/bonylandmarks-meshes-v1.zip`` (chemins relatifs à ``data/``,
donc extractible directement dans ``src/bonylandmarks/data/``) ::

    bones/*.ply  bones_full/*.ply  muscles/*.ply
    MESHES_LICENSE.txt   (attribution + licence CC BY-SA 2.1 JP, FR/EN)
    MANIFEST.json        (fichiers, tailles, SHA-256)

Le zip est reproductible (ordre trié, dates fixes).  Publication ::

    gh release create meshes-v1 dist/bonylandmarks-meshes-v1.zip --title ... --notes ...

BodyParts3D, (c) The Database Center for Life Science, licensed under
CC Attribution-Share Alike 2.1 Japan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "src" / "bonylandmarks" / "data"
SUBDIRS = ("bones", "bones_full", "muscles")
DEFAULT_TAG = "meshes-v1"
DEFAULT_OUT = REPO / "dist" / "bonylandmarks-meshes-v1.zip"
LICENSE_NAME = "MESHES_LICENSE.txt"
MANIFEST_NAME = "MANIFEST.json"
_FIXED_DATE = (2024, 1, 1, 0, 0, 0)

LICENSE_TEXT = """\
BonyLandmarks - meshes anatomiques 3D / 3D anatomical meshes
=============================================================

FRANCAIS
--------
Source
  BodyParts3D, (c) The Database Center for Life Science,
  licence Creative Commons Attribution-Share Alike 2.1 Japan (CC BY-SA 2.1 JP).
  Base de donnees : https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html
  Licence         : https://creativecommons.org/licenses/by-sa/2.1/jp/

Modifications apportees a l'oeuvre d'origine
  - selection des structures anatomiques (os et muscles) utilisees par BonyLandmarks ;
  - fusion de plusieurs fichiers sources en un seul mesh lorsqu'une structure
    en comprend plusieurs (regroupements osseux, doublons exacts supprimes) ;
  - decimation a 15 000 faces au plus par structure ;
  - conversion du format OBJ (texte) vers PLY binaire little-endian
    (sommets en float32, sans normales ni couleurs) ;
  - aucune modification des coordonnees : le referentiel BodyParts3D d'origine
    (millimetres) est conserve tel quel, sans recentrage.

Conditions
  Ces fichiers sont des oeuvres derivees de BodyParts3D. Vous pouvez les copier,
  les redistribuer et les modifier, y compris a usage commercial, a condition de :
  (1) crediter BodyParts3D et The Database Center for Life Science comme
  ci-dessus, (2) indiquer les modifications apportees, et (3) distribuer toute
  oeuvre derivee sous la meme licence (CC BY-SA 2.1 JP) ou une licence
  compatible (partage dans les memes conditions).
  Cette licence s'applique a ces meshes et non au code source de BonyLandmarks,
  qui reste regi par sa propre licence.

ENGLISH
-------
Source
  BodyParts3D, (c) The Database Center for Life Science,
  licensed under Creative Commons Attribution-Share Alike 2.1 Japan (CC BY-SA 2.1 JP).
  Database : https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html
  License  : https://creativecommons.org/licenses/by-sa/2.1/jp/

Modifications made to the original work
  - selection of the anatomical structures (bones and muscles) used by BonyLandmarks;
  - merging of several source files into a single mesh when a structure is made
    of several parts (bone groups; exact duplicates removed);
  - decimation to at most 15,000 faces per structure;
  - conversion from text OBJ to little-endian binary PLY (float32 vertices,
    no normals, no colors);
  - no change to the coordinates: the original BodyParts3D frame (millimetres)
    is kept as is, without recentering.

Terms
  These files are derivative works of BodyParts3D. You may copy, redistribute
  and modify them, including for commercial use, provided that you:
  (1) credit BodyParts3D and The Database Center for Life Science as above,
  (2) indicate the changes made, and (3) distribute any derivative work under
  the same license (CC BY-SA 2.1 JP) or a compatible one (ShareAlike).
  This license applies to these meshes, not to the BonyLandmarks source code,
  which remains under its own license.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(data: Path) -> list[Path]:
    files: list[Path] = []
    for sub in SUBDIRS:
        files += sorted((data / sub).glob("*.ply"))
    return files


def build_zip(data: Path, out: Path, tag: str = DEFAULT_TAG) -> dict:
    """Construit le zip ; renvoie le manifest écrit."""
    files = collect(data)
    if not files:
        raise SystemExit(f"Aucun .ply sous {data} (lancer scripts/convert_meshes.py d'abord).")
    manifest = {
        "tag": tag,
        "format": "PLY binary little-endian",
        "license": "CC BY-SA 2.1 JP - BodyParts3D, (c) The Database Center for Life Science",
        "files": [
            {"path": p.relative_to(data).as_posix(), "size": p.stat().st_size,
             "sha256": sha256_file(p)}
            for p in files
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)

    def add(zf: zipfile.ZipFile, name: str, payload: bytes) -> None:
        zi = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        zf.writestr(zi, payload, compresslevel=9)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        add(zf, LICENSE_NAME, LICENSE_TEXT.encode("utf-8"))
        add(zf, MANIFEST_NAME, (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
        for p in files:
            add(zf, p.relative_to(data).as_posix(), p.read_bytes())
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="dossier data/ [défaut: %(default)s]")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="zip de sortie [défaut: %(default)s]")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="tag de release inscrit dans le manifest")
    args = ap.parse_args(argv)

    manifest = build_zip(args.data, args.out, args.tag)
    raw = sum(f["size"] for f in manifest["files"])
    print(f"{len(manifest['files'])} fichiers PLY : {raw / 1e6:.1f} Mo bruts")
    print(f"Zip : {args.out} ({args.out.stat().st_size / 1e6:.1f} Mo)")
    print(f"SHA-256 du zip : {sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
