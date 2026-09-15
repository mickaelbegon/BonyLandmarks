"""BonyLandmarks student application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from .client import TeacherServerClient
from .export import export_session
from .i18n import tr
from .login_dialog import LoginDialog
from .manifest import get_server_url, load_manifest
from .landmarks_extended import LANDMARKS
from .mesh_loader import load_avatar_glb
from .scoring import SessionScore
from .tutorial import TutorialViewer
from .viewer import LandmarkViewer
from .welcome import WelcomeDialog


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._lang = "fr"
        self.setWindowTitle(tr("app_title", self._lang))
        self.resize(1280, 800)

    def start(self) -> None:
        import os
        dev_scan = os.environ.get("BONY_DEV_SCAN")
        if dev_scan:
            avatar_bytes = Path(dev_scan).read_bytes()
            matricule = "dev"
        else:
            students = load_manifest()
            server_url = get_server_url()

            dialog = LoginDialog(students=students, server_url=server_url, lang=self._lang, parent=self)
            while True:
                if dialog.exec() != LoginDialog.Accepted:
                    sys.exit(0)

                matricule = dialog.matricule

                if dialog.local_glb_bytes is not None:
                    avatar_bytes = dialog.local_glb_bytes
                    break

                birthdate = dialog.birthdate

                # Defensive: dialog disables Connect when server_url is None,
                # but guard here to avoid passing None to the client.
                if server_url is None:
                    dialog.set_error(tr("error_server_config", self._lang))
                    continue

                try:
                    client = TeacherServerClient(server_url)
                    avatar_bytes = client.fetch_avatar(matricule, birthdate)
                    break
                except ValueError:
                    dialog.set_error(tr("error_credentials", self._lang))
                except (httpx.HTTPError, ConnectionError) as exc:
                    dialog.set_error(tr("error_network", self._lang, detail=str(exc)))

        body_mesh, vertex_colors, landmark_markers, all_markers = load_avatar_glb(
            avatar_bytes, remove_stickers=True
        )
        _, vertex_colors_raw, _, _ = load_avatar_glb(
            avatar_bytes, remove_stickers=False
        )

        # Welcome dialog — skip if the user has checked "don't show again"
        from PySide6.QtWidgets import QDialog
        tutorial_mode = False
        if not WelcomeDialog.dont_show():
            welcome = WelcomeDialog(lang=self._lang, parent=self)
            result = welcome.exec()
            if result == QDialog.Rejected:
                return  # user closed the window
            tutorial_mode = (result == WelcomeDialog.TUTORIAL_RESULT)

        # All codes from the extended set; ground_truth available only for the
        # 24 BodyLoop-mapped BONE landmarks — EMG/SKINFOLD/ANTHRO are shown
        # but marked "non évalué" when no ground truth exists.
        landmark_codes = [lm.code for lm in LANDMARKS]

        if tutorial_mode:
            # Guided tour first; the exercise starts when the student asks for it.
            tutorial = TutorialViewer(
                mesh=body_mesh,
                ground_truth=landmark_markers,
                landmark_codes=landmark_codes,
                lang=self._lang,
                vertex_colors=vertex_colors,
            )

            def _on_tutorial_done() -> None:
                # Release the tutorial's VTK render window, then swap the
                # central widget on the next event-loop turn: replacing it
                # from inside the button's own signal would delete the widget
                # that is still emitting.
                tutorial.shutdown()
                QTimer.singleShot(
                    0,
                    lambda: self._launch_session(
                        body_mesh,
                        vertex_colors,
                        vertex_colors_raw,
                        landmark_markers,
                        all_markers,
                        landmark_codes,
                        matricule,
                    ),
                )

            tutorial.start_session.connect(_on_tutorial_done)
            self.setCentralWidget(tutorial)
            self.show()
        else:
            self._launch_session(
                body_mesh,
                vertex_colors,
                vertex_colors_raw,
                landmark_markers,
                all_markers,
                landmark_codes,
                matricule,
            )

    def _launch_session(
        self,
        body_mesh,
        vertex_colors,
        vertex_colors_raw,
        landmark_markers,
        all_markers,
        landmark_codes,
        matricule: str,
    ) -> None:
        """Create the graded LandmarkViewer and make it the central widget.

        Called directly when the student skips the tutorial, or from the
        tutorial's ``start_session`` signal (which replaces the central widget).
        """
        viewer = LandmarkViewer(
            mesh=body_mesh,
            ground_truth=landmark_markers,
            landmark_codes=landmark_codes,
            lang=self._lang,
            vertex_colors=vertex_colors,
            vertex_colors_raw=vertex_colors_raw,
            all_markers=all_markers,
        )
        viewer.session_complete.connect(
            lambda score: self._on_session_done(score, matricule)
        )
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
