"""BonyLandmarks student application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from .client import TeacherServerClient
from .export import export_session
from .i18n import tr
from .login_dialog import LoginDialog
from .mesh_loader import load_glb_mesh, parse_markers
from .viewer import LandmarkViewer
from .scoring import SessionScore


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._lang = "fr"
        self.setWindowTitle(tr("app_title", self._lang))
        self.resize(1280, 800)

    def start(self) -> None:
        dialog = LoginDialog(self._lang, self)
        while True:
            if dialog.exec() != LoginDialog.Accepted:
                sys.exit(0)

            matricule = dialog.matricule
            birthdate = dialog.birthdate
            server_url = dialog.server_url

            try:
                client = TeacherServerClient(server_url)
                glb_bytes, markers_raw = client.fetch_scan(matricule, birthdate)
                break
            except ValueError:
                dialog.set_error(tr("error_credentials", self._lang))
            except (httpx.HTTPError, ConnectionError) as exc:
                dialog.set_error(tr("error_network", self._lang, detail=str(exc)))

        markers = parse_markers(markers_raw)
        mesh = load_glb_mesh(glb_bytes)
        landmark_codes = list(markers.keys())

        viewer = LandmarkViewer(mesh, markers, landmark_codes, lang=self._lang)
        viewer.session_complete.connect(lambda score: self._on_session_done(score, matricule))
        self.setCentralWidget(viewer)
        self.show()

    def _on_session_done(self, score: SessionScore, matricule: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("save_button", self._lang),
            f"{matricule}_landmarks.json",
            "JSON (*.json)",
        )
        if path:
            export_session(score, matricule, Path(path))
            QMessageBox.information(self, "OK", f"Saved: {path}")


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("BonyLandmarks")
    window = MainWindow()
    window.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
