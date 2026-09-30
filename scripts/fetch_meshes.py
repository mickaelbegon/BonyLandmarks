#!/usr/bin/env python
"""Télécharge et installe les meshes BodyParts3D (PLY) publiés en release GitHub.

Les meshes ne sont pas dans git (licence CC BY-SA 2.1 JP, volume).  Ce script
récupère l'asset ``bonylandmarks-meshes-v1.zip`` de la release ``meshes-v1`` et
l'extrait dans ``src/bonylandmarks/data/`` (``bones/``, ``bones_full/``,
``muscles/``, ``MESHES_LICENSE.txt``).  Les SHA-256 du ``MANIFEST.json`` du zip
sont vérifiés ; l'opération est idempotente (rien n'est retéléchargé si les
fichiers installés sont déjà conformes).

Stratégie de téléchargement : ``gh release download`` si ``gh`` est installé et
authentifié, sinon URL publique
``https://github.com/<repo>/releases/download/<tag>/<asset>`` via httpx.

Exemples ::

    python scripts/fetch_meshes.py
    python scripts/fetch_meshes.py --tag meshes-v1 --dest /chemin/data
    python scripts/fetch_meshes.py --zip dist/bonylandmarks-meshes-v1.zip   # hors ligne

BodyParts3D, (c) The Database Center for Life Science, licensed under
CC Attribution-Share Alike 2.1 Japan.
https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "src" / "bonylandmarks" / "data"
DEFAULT_REPO = "mickaelbegon/BonyLandmarks"
DEFAULT_TAG = "meshes-v1"
LICENSE_NAME = "MESHES_LICENSE.txt"
MANIFEST_NAME = "MANIFEST.json"
INSTALLED_MANIFEST = "meshes_manifest.json"  # copie locale (dans --dest) pour l'idempotence


def default_asset(tag: str) -> str:
    return f"bonylandmarks-meshes-{tag.removeprefix('meshes-')}.zip"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Téléchargement
# --------------------------------------------------------------------------
def gh_available() -> bool:
    if shutil.which("gh") is None:
        return False
    try:
        return subprocess.run(["gh", "auth", "status"], capture_output=True,
                              timeout=30).returncode == 0
    except Exception:
        return False


def download_with_gh(repo: str, tag: str, asset: str, out_dir: Path) -> Path | None:
    try:
        r = subprocess.run(
            ["gh", "release", "download", tag, "--repo", repo, "--pattern", asset,
             "--dir", str(out_dir), "--clobber"],
            capture_output=True, text=True, timeout=600)
    except Exception as exc:
        print(f"  gh indisponible ({exc})")
        return None
    if r.returncode != 0:
        lines = (r.stderr or r.stdout).strip().splitlines()
        print(f"  gh a échoué : {lines[-1] if lines else 'erreur inconnue'}")
        return None
    p = out_dir / asset
    return p if p.is_file() else None


def download_with_httpx(url: str, dest: Path) -> Path:
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        if r.status_code == 404:
            raise SystemExit(
                f"Asset introuvable (404) : {url}\n"
                "La release n'existe pas (encore) ou le nom de l'asset est incorrect.")
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    return dest


def obtain_zip(args, tmp: Path) -> Path:
    if args.zip:
        if not args.zip.is_file():
            raise SystemExit(f"Zip introuvable : {args.zip}")
        return args.zip
    asset = args.asset or default_asset(args.tag)
    if gh_available():
        print(f"Téléchargement via gh : {args.repo}@{args.tag} / {asset}")
        p = download_with_gh(args.repo, args.tag, asset, tmp)
        if p is not None:
            return p
        print("Repli sur l'URL publique.")
    url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{asset}"
    print(f"Téléchargement : {url}")
    return download_with_httpx(url, tmp / asset)


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------
def _safe_target(dest: Path, rel: str) -> Path:
    target = (dest / rel).resolve()
    if dest.resolve() not in target.parents:
        raise SystemExit(f"Chemin suspect dans le zip : {rel}")
    return target


def is_up_to_date(dest: Path, tag: str) -> bool:
    """Vrai si la copie locale du manifest correspond au tag et que tous les fichiers sont conformes."""
    mp = dest / INSTALLED_MANIFEST
    if not mp.is_file():
        return False
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if manifest.get("tag") != tag:
            return False
        for f in manifest["files"]:
            p = dest / f["path"]
            if not p.is_file() or p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]:
                return False
    except Exception:
        return False
    return (dest / LICENSE_NAME).is_file()


def install_zip(zip_path: Path, dest: Path) -> tuple[int, int]:
    """Extrait en vérifiant les SHA-256 ; renvoie (installés, déjà conformes)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        try:
            manifest = json.loads(zf.read(MANIFEST_NAME))
        except KeyError:
            raise SystemExit(f"{MANIFEST_NAME} absent du zip : archive invalide.")
        wrote = same = 0
        for f in manifest["files"]:
            rel, size, digest = f["path"], f["size"], f["sha256"]
            target = _safe_target(dest, rel)
            if target.is_file() and target.stat().st_size == size and sha256_file(target) == digest:
                same += 1
                continue
            data = zf.read(rel)
            if len(data) != size or sha256_bytes(data) != digest:
                raise SystemExit(f"SHA-256 invalide pour {rel} : archive corrompue.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            wrote += 1
        if LICENSE_NAME in zf.namelist():
            (dest / LICENSE_NAME).write_bytes(zf.read(LICENSE_NAME))
        (dest / INSTALLED_MANIFEST).write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return wrote, same


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default=DEFAULT_TAG, help="tag de la release [défaut: %(default)s]")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="dépôt GitHub [défaut: %(default)s]")
    ap.add_argument("--asset", default=None, help="nom de l'asset [défaut: dérivé du tag]")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help="dossier data/ cible [défaut: %(default)s]")
    ap.add_argument("--zip", type=Path, default=None,
                    help="installe depuis ce zip local (aucun accès réseau)")
    ap.add_argument("--force", action="store_true",
                    help="ne saute pas le téléchargement même si les fichiers sont à jour")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not args.zip and not args.force and is_up_to_date(args.dest, args.tag):
        print(f"Meshes déjà installés et conformes ({args.tag}) dans {args.dest}")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="bony_meshes_"))
    try:
        zip_path = obtain_zip(args, tmp)
        wrote, same = install_zip(zip_path, args.dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"Meshes installés dans {args.dest} : {wrote} écrits, {same} déjà conformes.")
    print("BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 JP "
          f"(voir {LICENSE_NAME}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
