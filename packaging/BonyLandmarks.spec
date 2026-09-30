# -*- mode: python ; coding: utf-8 -*-
"""Spec PyInstaller commun Windows (onefile .exe) / macOS (onedir .app).

Usage (depuis la racine du dépôt) :
    pyinstaller --noconfirm packaging/BonyLandmarks.spec
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
PKG = SRC / "bonylandmarks"
IS_MAC = sys.platform == "darwin"
VERSION = "0.1.0"

# --- Données : JSON/icônes + meshes PLY (BodyParts3D, CC BY-SA 2.1 JP) ---
# Les meshes ne sont pas dans git : la CI les installe avant PyInstaller avec
# `python scripts/fetch_meshes.py` (release GitHub `meshes-v1`).  Seuls les .ply
# sont embarqués (pas les éventuels .obj/.stl locaux, plus volumineux).
datas = []
for pattern, dest in [
    ("data/*.json", "bonylandmarks/data"),
    ("data/MESHES_LICENSE.txt", "bonylandmarks/data"),
    ("data/bones/*.json", "bonylandmarks/data/bones"),
    ("data/bones/*.md", "bonylandmarks/data/bones"),
    ("data/bones/*.ply", "bonylandmarks/data/bones"),
    ("data/bones_full/*.ply", "bonylandmarks/data/bones_full"),
    ("data/muscles/*.ply", "bonylandmarks/data/muscles"),
    ("icons/*.png", "bonylandmarks/icons"),
    ("manifest.dev.json", "bonylandmarks"),
]:
    for f in PKG.glob(pattern):
        datas.append((str(f), dest))

n_meshes = sum(1 for src, _dest in datas if src.endswith(".ply"))
print(f"[spec] {n_meshes} mesh(es) .ply embarqué(s)")
if n_meshes == 0:
    msg = ("[spec] AUCUN mesh .ply à embarquer : lancer `python scripts/fetch_meshes.py` "
           "avant PyInstaller (exercices Anatomie 3D / quiz osseux inutilisables sinon).")
    if os.environ.get("BONY_REQUIRE_MESHES") == "1":
        raise SystemExit(msg)
    print("WARNING " + msg)

for mod in ("certifi", "pyvista", "trimesh", "pyvistaqt"):
    try:
        datas += collect_data_files(mod)
    except Exception:
        pass

# --- Imports dynamiques que l'analyse statique rate ---
hiddenimports = []
for mod in ("bonylandmarks", "vtkmodules", "pyvista", "pyvistaqt",
            "trimesh", "pygltflib", "cryptography", "httpx", "certifi"):
    try:
        hiddenimports += collect_submodules(mod)
    except Exception:
        pass
hiddenimports += ["scipy.spatial.transform", "scipy.spatial", "scipy.sparse.csgraph",
                  "PySide6.QtOpenGLWidgets", "PySide6.QtSvg"]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if IS_MAC:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="BonyLandmarks",
        console=False,
        codesign_identity=None,
    )
    coll = COLLECT(exe, a.binaries, a.datas, name="BonyLandmarks")
    app = BUNDLE(
        coll,
        name="BonyLandmarks.app",
        icon=None,
        bundle_identifier="ca.umontreal.bonylandmarks",
        version=VERSION,
        info_plist={
            "CFBundleName": "BonyLandmarks",
            "CFBundleDisplayName": "BonyLandmarks",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "LSMinimumSystemVersion": "12.0",
            "NSHumanReadableCopyright": "Mickael Begon, Université de Montréal",
        },
    )
else:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="BonyLandmarks",
        console=False,
        upx=False,
    )
