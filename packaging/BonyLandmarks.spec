# -*- mode: python ; coding: utf-8 -*-
"""Spec PyInstaller commun Windows (onefile .exe) / macOS (onedir .app).

Usage (depuis la racine du dépôt) :
    pyinstaller --noconfirm packaging/BonyLandmarks.spec
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
PKG = SRC / "bonylandmarks"
IS_MAC = sys.platform == "darwin"
VERSION = "0.1.0"

# --- Données : uniquement ce qui est nécessaire (pas les meshes STL/OBJ) ---
datas = []
for pattern, dest in [
    ("data/*.json", "bonylandmarks/data"),
    ("data/bones/*.json", "bonylandmarks/data/bones"),
    ("data/bones/*.md", "bonylandmarks/data/bones"),
    ("icons/*.png", "bonylandmarks/icons"),
    ("manifest.dev.json", "bonylandmarks"),
]:
    for f in PKG.glob(pattern):
        datas.append((str(f), dest))

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
