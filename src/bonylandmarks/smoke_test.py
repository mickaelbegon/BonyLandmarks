"""Smoke test du binaire figé (PyInstaller) — utilisé par la CI.

Déclenché par ``--smoke-test`` ou ``BONY_SMOKE_TEST=1``. Vérifie, sans
interaction : imports de tous les modules, données embarquées, certificats SSL,
QApplication, fenêtre principale, rendu VTK (QtInteractor). Sort en code 0 si
tout va bien. Les traces sont écrites sur stdout ET dans le fichier
``BONY_SMOKE_LOG`` (indispensable sous Windows où l'exe fenêtré n'a pas de stdout).
"""

from __future__ import annotations

import faulthandler
import importlib
import os
import pkgutil
import platform
import sys
import traceback
from pathlib import Path

_LOG_FH = None


def _log(msg: str) -> None:
    line = f"[smoke] {msg}"
    try:
        if sys.stdout is not None:
            print(line, flush=True)
    except Exception:
        pass
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def is_requested() -> bool:
    return "--smoke-test" in sys.argv or os.environ.get("BONY_SMOKE_TEST") == "1"


def _import_all_modules() -> list[str]:
    import bonylandmarks

    names = sorted(
        m.name for m in pkgutil.walk_packages(bonylandmarks.__path__, "bonylandmarks.")
    )
    if len(names) < 20:
        raise RuntimeError(f"Seulement {len(names)} modules trouvés : {names}")
    failures = []
    for name in names:
        if name.endswith("__main__"):
            continue
        try:
            importlib.import_module(name)
        except Exception:
            failures.append(name)
            _log(f"IMPORT ECHEC {name}\n{traceback.format_exc()}")
    _log(f"{len(names)} modules, {len(failures)} échec(s) d'import")
    return failures


def _check_data() -> list[str]:
    import bonylandmarks

    root = Path(bonylandmarks.__file__).parent
    _log(f"package root: {root}")
    missing = []
    for rel in (
        "data/landmarks.json",
        "data/evaluation_sets.json",
        "data/bones/landmark_positions.json",
        "icons/view_front.png",
    ):
        ok = (root / rel).is_file()
        _log(f"data {rel}: {'OK' if ok else 'MANQUANT'}")
        if not ok:
            missing.append(rel)
    from bonylandmarks.landmarks_extended import LANDMARKS

    _log(f"LANDMARKS: {len(LANDMARKS)}")
    if len(LANDMARKS) < 100:
        missing.append("LANDMARKS")
    return missing


def _check_ssl() -> None:
    import certifi
    import httpx

    path = certifi.where()
    if not Path(path).is_file():
        raise RuntimeError(f"bundle CA introuvable : {path}")
    httpx.Client().close()
    _log(f"certifi OK: {path}")


def run() -> int:
    global _LOG_FH
    log_path = os.environ.get("BONY_SMOKE_LOG")
    if log_path:
        _LOG_FH = open(log_path, "a", encoding="utf-8")
    faulthandler.enable(_LOG_FH if _LOG_FH is not None else sys.stderr)

    _log(f"python {sys.version.split()[0]} frozen={getattr(sys, 'frozen', False)}")
    _log(f"machine={platform.machine()} platform={platform.platform()}")
    _log(f"executable={sys.executable}")

    errors: list[str] = []
    try:
        failures = _import_all_modules()
        errors += [f"import {n}" for n in failures]
        errors += [f"data {n}" for n in _check_data()]
        _check_ssl()

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)
        app.setApplicationName("BonyLandmarks")
        _log(f"QApplication OK (platform={app.platformName()})")

        from .main import MainWindow

        win = MainWindow()
        _log("MainWindow construite")

        # Rendu VTK : c'est là que Metal/OpenGL peut faire planter le binaire.
        import pyvista as pv
        from pyvistaqt import QtInteractor

        plotter = QtInteractor(win)
        plotter.add_mesh(pv.Sphere(), color="white")
        plotter.render()
        _log("QtInteractor + rendu VTK OK")
        plotter.close()

        QTimer.singleShot(3000, app.quit)
        app.exec()
        _log("boucle d'événements OK")
    except Exception:
        _log("ECHEC:\n" + traceback.format_exc())
        errors.append("exception")

    if errors:
        _log(f"RESULTAT: ECHEC ({errors})")
        return 1
    _log("RESULTAT: OK")
    return 0
