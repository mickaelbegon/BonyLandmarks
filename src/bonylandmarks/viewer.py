"""Main 3D viewer widget: PySide6 window embedding a PyVista plotter.

Workflow per landmark:
  1. The instruction panel shows the landmark name + hint.
  2. The student clicks anywhere on the body mesh.
  3. PyVista ray-casts the click to find the surface point.
  4. A yellow sphere marks the candidate position.
  5. The student confirms or redoes the pick.
  6. After confirmation the error vs the BodyLoop ground truth is shown.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .i18n import Language, tr
from .landmarks import LANDMARKS, Landmark
from .scoring import LandmarkResult, SessionScore


# ─── colour constants ────────────────────────────────────────────────────────

_MESH_COLOR = "#c8b8a8"
_MARKER_COLOR = "#00cc44"       # BodyLoop reference markers (green)
_CANDIDATE_COLOR = "#ffdd00"    # Student candidate pick (yellow)
_CONFIRMED_COLOR = "#3399ff"    # Confirmed student pick (blue)
_ACTOR_PREFIX_REF = "ref_"
_ACTOR_PREFIX_CONFIRMED = "confirmed_"


class LandmarkViewer(QWidget):
    """Main widget containing the PyVista 3D view and control panels."""

    session_complete = Signal(SessionScore)

    def __init__(
        self,
        mesh: pv.PolyData,
        ground_truth: dict[str, np.ndarray],
        landmark_codes: list[str],
        lang: Language = "fr",
        vertex_colors: np.ndarray | None = None,
        all_markers: dict[str, np.ndarray] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mesh = mesh
        self._vertex_colors = vertex_colors
        self._all_markers = all_markers or {}
        self._ground_truth = ground_truth
        self._lang = lang

        # Filter to landmarks that have ground-truth positions
        self._landmarks: list[Landmark] = [
            lm for lm in LANDMARKS
            if lm.code in landmark_codes and lm.code in ground_truth
        ]
        self._index = 0
        self._results: list[LandmarkResult] = []
        self._candidate_point: np.ndarray | None = None

        self._build_ui()
        self._setup_scene()
        self._update_instruction_panel()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Left: 3D plotter
        self._plotter = QtInteractor(self)
        self._plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._plotter.interactor, stretch=3)

        # Right: control panel
        panel = QVBoxLayout()
        panel.setSpacing(12)

        # Language toggle
        lang_row = QHBoxLayout()
        self._lang_btn = QPushButton(tr("lang_toggle", self._lang))
        self._lang_btn.setFixedWidth(90)
        self._lang_btn.clicked.connect(self._toggle_lang)
        lang_row.addStretch()
        lang_row.addWidget(self._lang_btn)
        panel.addLayout(lang_row)

        # Progress label
        self._progress_label = QLabel()
        self._progress_label.setAlignment(Qt.AlignCenter)
        self._progress_label.setStyleSheet("font-size: 13px; color: #555;")
        panel.addWidget(self._progress_label)

        # Landmark name
        self._name_label = QLabel()
        self._name_label.setWordWrap(True)
        self._name_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        panel.addWidget(self._name_label)

        # Hint text
        self._hint_text = QTextEdit()
        self._hint_text.setReadOnly(True)
        self._hint_text.setFixedHeight(80)
        self._hint_text.setStyleSheet("font-size: 12px; color: #444;")
        panel.addWidget(self._hint_text)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        panel.addWidget(sep)

        # Instructions
        self._instr_label = QLabel(tr("instructions", self._lang))
        self._instr_label.setWordWrap(True)
        self._instr_label.setStyleSheet("font-size: 12px; font-style: italic;")
        panel.addWidget(self._instr_label)

        # Error display
        self._error_label = QLabel("")
        self._error_label.setAlignment(Qt.AlignCenter)
        self._error_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        panel.addWidget(self._error_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._confirm_btn = QPushButton(tr("confirm_button", self._lang))
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._on_confirm)
        self._redo_btn = QPushButton(tr("redo_button", self._lang))
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self._on_redo)
        btn_row.addWidget(self._confirm_btn)
        btn_row.addWidget(self._redo_btn)
        panel.addLayout(btn_row)

        panel.addStretch()

        # Legend
        legend_label = QLabel(
            f"<span style='color:{_MARKER_COLOR}'>●</span> {tr('green_marker_tooltip', self._lang)}<br>"
            "<span style='color:#ffdd00'>●</span> Votre sélection / Your pick<br>"
            "<span style='color:#3399ff'>●</span> Confirmé / Confirmed"
        )
        legend_label.setStyleSheet("font-size: 11px;")
        panel.addWidget(legend_label)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(280)
        root.addWidget(panel_widget)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        pl = self._plotter
        pl.background_color = "#1a1a2e"

        # Use photographic vertex colours when available, otherwise flat skin tone
        if self._vertex_colors is not None:
            rgba = self._vertex_colors[:, :3]   # RGB only for PyVista scalars
            self._mesh.point_data["RGB"] = rgba
            pl.add_mesh(
                self._mesh,
                scalars="RGB",
                rgb=True,
                smooth_shading=True,
                show_scalar_bar=False,
            )
        else:
            pl.add_mesh(self._mesh, color=_MESH_COLOR, opacity=0.85, smooth_shading=True)

        # All BodyLoop auto-markers as small grey spheres
        gt_codes = {lm.code for lm in self._landmarks}
        for bl_name, xyz in self._all_markers.items():
            # Skip the 24 that will be shown as green below
            from .mesh_loader import BODYLOOP_NAME_TO_CODE
            if BODYLOOP_NAME_TO_CODE.get(bl_name) in gt_codes:
                continue
            sphere = pv.Sphere(radius=5, center=xyz)
            pl.add_mesh(sphere, color="#888888", opacity=0.6, name=f"extra_{bl_name}")

        # BodyLoop reference markers as green spheres (the 24 to place)
        for code, xyz in self._ground_truth.items():
            if code in gt_codes:
                sphere = pv.Sphere(radius=10, center=xyz)
                pl.add_mesh(sphere, color=_MARKER_COLOR, name=f"{_ACTOR_PREFIX_REF}{code}")

        pl.enable_surface_point_picking(
            callback=self._on_surface_pick,
            show_message=False,
            pickable_window=False,
            use_picker=True,
            font_size=10,
        )
        pl.reset_camera()
        pl.show()

    # ── Interaction callbacks ─────────────────────────────────────────────────

    def _on_surface_pick(self, point: np.ndarray) -> None:
        """Called by PyVista when the user clicks the mesh surface."""
        if self._index >= len(self._landmarks):
            return
        self._candidate_point = np.array(point, dtype=float)

        # Remove previous candidate sphere
        self._plotter.remove_actor("candidate_sphere", render=False)
        sphere = pv.Sphere(radius=10, center=self._candidate_point)
        self._plotter.add_mesh(sphere, color=_CANDIDATE_COLOR, name="candidate_sphere")
        self._plotter.render()

        self._confirm_btn.setEnabled(True)
        self._redo_btn.setEnabled(True)
        self._error_label.setText("")

    def _on_confirm(self) -> None:
        if self._candidate_point is None:
            return
        lm = self._landmarks[self._index]
        gt = self._ground_truth[lm.code]
        result = LandmarkResult(
            code=lm.code,
            ground_truth=gt,
            student_pick=self._candidate_point.copy(),
        )
        self._results.append(result)

        # Show confirmed sphere
        sphere = pv.Sphere(radius=8, center=self._candidate_point)
        self._plotter.add_mesh(
            sphere, color=_CONFIRMED_COLOR, name=f"{_ACTOR_PREFIX_CONFIRMED}{lm.code}"
        )
        self._plotter.remove_actor("candidate_sphere", render=False)
        self._plotter.render()

        # Show error
        color = result.feedback_color()
        self._error_label.setText(tr("error_mm_label", self._lang, value=result.error_mm))
        self._error_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {color};"
        )

        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._candidate_point = None
        self._index += 1

        if self._index >= len(self._landmarks):
            self._finish_session()
        else:
            self._update_instruction_panel()

    def _on_redo(self) -> None:
        self._plotter.remove_actor("candidate_sphere", render=True)
        self._candidate_point = None
        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._error_label.setText("")

    def _finish_session(self) -> None:
        score = SessionScore(self._results)
        self._name_label.setText(
            tr("session_complete", self._lang, mean=score.mean_error_mm)
        )
        self._instr_label.setText("")
        self._hint_text.setText("")
        self._progress_label.setText("")
        self.session_complete.emit(score)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_instruction_panel(self) -> None:
        if self._index >= len(self._landmarks):
            return
        lm = self._landmarks[self._index]
        self._progress_label.setText(
            tr("landmark_label", self._lang, index=self._index + 1, total=len(self._landmarks))
        )
        self._name_label.setText(lm.name(self._lang))
        self._hint_text.setText(lm.hint(self._lang))
        self._instr_label.setText(tr("instructions", self._lang))
        self._error_label.setText("")

    def _toggle_lang(self) -> None:
        self._lang = "en" if self._lang == "fr" else "fr"
        self._lang_btn.setText(tr("lang_toggle", self._lang))
        self._instr_label.setText(tr("instructions", self._lang))
        if self._index < len(self._landmarks):
            self._update_instruction_panel()
