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
from .mesh_loader import load_avatar_glb
from .scoring import SessionScore
from .viewer import LandmarkViewer


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
                avatar_bytes = client.fetch_avatar(matricule, birthdate)
                break
            except ValueError:
                dialog.set_error(tr("error_credentials", self._lang))
            except (httpx.HTTPError, ConnectionError) as exc:
                dialog.set_error(tr("error_network", self._lang, detail=str(exc)))

        body_mesh, vertex_colors, landmark_markers, all_markers = load_avatar_glb(avatar_bytes)
        landmark_codes = list(landmark_markers.keys())

        viewer = LandmarkViewer(
            mesh=body_mesh,
            ground_truth=landmark_markers,
            landmark_codes=landmark_codes,
            lang=self._lang,
            vertex_colors=vertex_colors,
            all_markers=all_markers,
        )
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
