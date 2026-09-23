"""BonyLandmarks student application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from .client import TeacherServerClient
from .export import export_session
from .i18n import tr
from .manifest import get_server_url, load_manifest
from .landmarks_extended import LANDMARKS
from .mesh_loader import load_avatar_glb
from .scoring import SessionScore
from .splash import SplashDialog
from .tutorial import TutorialViewer
from .validation import validate_markers, report_validation
from .viewer import LandmarkViewer


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._lang = "fr"
        self.setWindowTitle(tr("app_title", self._lang))
        self.resize(1280, 800)

    def start(self) -> None:
        import os
        dev_scan = os.environ.get("BONY_DEV_SCAN")
        dev_mode = bool(dev_scan)
        if dev_scan:
            avatar_bytes = Path(dev_scan).read_bytes()
            matricule = "dev"
            tutorial_mode = False
        else:
            students = load_manifest()
            server_url = get_server_url()

            splash = SplashDialog(
                students=students,
                server_url=server_url,
                lang=self._lang,
                parent=None,  # no hidden parent — avoids dialog falling behind on Windows
            )
            splash.raise_()
            splash.activateWindow()
            while True:
                result = splash.exec()
                if result == QDialog.Rejected:
                    sys.exit(0)

                if result == SplashDialog.ANNOTATOR_RESULT:
                    # Teacher tool: independent window, no server session.
                    self._open_annotator()
                    return

                tutorial_mode = (result == SplashDialog.TUTORIAL_RESULT)
                matricule = splash.matricule

                # Local file bypass — no credentials needed
                if splash.local_glb_bytes is not None:
                    avatar_bytes = splash.local_glb_bytes
                    break

                birthdate = splash.birthdate

                # Defensive: the splash disables "Commencer" when server_url is
                # None, but guard here to avoid passing None to the client.
                if server_url is None:
                    splash.set_error(tr("error_server_config", self._lang))
                    continue

                try:
                    client = TeacherServerClient(server_url)
                    avatar_bytes = client.fetch_avatar(matricule, birthdate)
                    break
                except ValueError:
                    splash.set_error(tr("error_credentials", self._lang))
                except (httpx.HTTPError, ConnectionError) as exc:
                    splash.set_error(tr("error_network", self._lang, detail=str(exc)))

        body_mesh, vertex_colors, landmark_markers, all_markers = load_avatar_glb(
            avatar_bytes, remove_stickers=True
        )
        _, vertex_colors_raw, _, _ = load_avatar_glb(
            avatar_bytes, remove_stickers=False
        )

        # Validate that BONE markers required for evaluation are present in the GLB.
        bone_codes = [lm.code for lm in LANDMARKS if lm.category == "BONE"]
        _val = validate_markers(landmark_markers, bone_codes)
        if _val["missing"]:
            print(
                f"[WARNING] {len(_val['missing'])} repère(s) BONE absent(s) du GLB : "
                + ", ".join(_val["missing"])
            )
            report_validation(_val)

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
                        dev_mode=dev_mode,
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
                dev_mode=dev_mode,
            )

    def _open_annotator(self) -> None:
        """Open the stand-alone manual annotation tool (teacher / researcher).

        The window is kept on ``self`` so it is not garbage-collected; the
        MainWindow itself stays hidden, so closing the annotator quits the app.
        """
        from .annotator import AnnotatorWindow

        self._annotator = AnnotatorWindow()
        self._annotator.show()
        self._annotator.raise_()
        self._annotator.activateWindow()

    def _launch_session(
        self,
        body_mesh,
        vertex_colors,
        vertex_colors_raw,
        landmark_markers,
        all_markers,
        landmark_codes,
        matricule: str,
        dev_mode: bool = False,
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
            dev_mode=dev_mode,
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
