"""Main 3D viewer widget: PySide6 window embedding a PyVista plotter.

Workflow per landmark:
  1. The instruction panel shows the landmark name + hint.
  2. The student clicks anywhere on the body mesh.
  3. PyVista ray-casts the click to find the surface point.
  4. A yellow sphere marks the candidate position.
  5. The student confirms or redoes the pick.
  6. After confirmation the error vs the BodyLoop ground truth is shown.
  7. A debrief overlay displays the hint + grade before advancing (Task A).
  8. A category transition screen appears when switching domains (Task B).
  9. Grade-D landmarks are retried until the student reaches at least C (Task C).
"""

from __future__ import annotations

import random

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .i18n import Language, tr
from .landmarks_extended import CATEGORY_LABELS, LANDMARKS, Landmark
from .scoring import LandmarkResult, SessionScore


# ─── colour constants ────────────────────────────────────────────────────────

_MESH_COLOR = "#c8b8a8"
_MARKER_COLOR = "#00cc44"       # BodyLoop reference markers (green)
_CANDIDATE_COLOR = "#ffdd00"    # Student candidate pick (yellow)
_CONFIRMED_COLOR = "#3399ff"    # Confirmed student pick (blue)
_ACTOR_PREFIX_REF = "ref_"
_ACTOR_PREFIX_CONFIRMED = "confirmed_"

_GRADE_COLORS: dict[str, str] = {
    "A": "#2e7d32",   # dark green
    "B": "#1565c0",   # dark blue
    "C": "#e65100",   # dark orange
    "D": "#c62828",   # dark red
}

# Category descriptions shown on the transition screen (FR, EN)
_CATEGORY_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "BONE": (
        "Repères osseux palpables — base de toute mesure en kinésiologie",
        "Palpable bony landmarks — the foundation of every kinesiology measurement",
    ),
    "EMG": (
        "Sites d'électrodes de surface selon les recommandations SENIAM — "
        "chaque site se construit à partir de deux repères osseux",
        "Surface electrode sites per SENIAM recommendations — "
        "each site is defined relative to two bony landmarks",
    ),
    "SKINFOLD": (
        "Sites de plis cutanés — protocole ISAK. La pastille marque le centre de la pince.",
        "Skinfold sites — ISAK protocol. The marker indicates the centre of the caliper.",
    ),
    "ANTHRO": (
        "Sites de circonférences et diamètres osseux — protocole ISAK côté droit",
        "Girths and bone diameters — ISAK protocol, right side only",
    ),
}


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
        vertex_colors_raw: np.ndarray | None = None,
        all_markers: dict[str, np.ndarray] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mesh = mesh
        self._vertex_colors_clean = vertex_colors
        self._vertex_colors_raw = vertex_colors_raw
        self._vertex_colors = vertex_colors          # currently active
        self._all_markers = all_markers or {}
        self._ground_truth = ground_truth
        self._lang = lang

        # Pick a random side for this session then filter bilateral landmarks
        self._chosen_side = random.choice(("left", "right"))
        self._landmarks: list[Landmark] = [
            lm for lm in LANDMARKS
            if lm.code in landmark_codes
            and not (
                (lm.code.endswith("_left") and self._chosen_side == "right")
                or (lm.code.endswith("_right") and self._chosen_side == "left")
            )
        ]
        self._index = 0
        self._results: list[LandmarkResult] = []
        # Task C — best result per code across all passes (normal + retry)
        self._best_results: dict[str, LandmarkResult] = {}
        # Task C — queue of landmarks to retry (grade D)
        self._retry_queue: list[Landmark] = []
        self._retry_mode: bool = False

        # Inverse identification mode
        self._inverse_mode: bool = False
        self._inverse_queue: list[Landmark] = []   # landmarks to identify, shuffled
        self._inverse_index: int = 0
        self._inverse_shown_actor: str | None = None  # PyVista actor name of the shown sphere

        self._candidate_point: np.ndarray | None = None
        self._redo_count: int = 0

        self._build_ui()
        self._setup_scene()
        self._update_instruction_panel()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _current_landmark(self) -> Landmark:
        """Return the landmark currently being placed."""
        if self._retry_mode:
            return self._retry_queue[0]
        return self._landmarks[self._index]

    def _center_dialog(self, dlg: QDialog) -> None:
        """Move *dlg* so it is centred over the 3-D plotter area."""
        interactor = self._plotter.interactor
        center_global = interactor.mapToGlobal(interactor.rect().center())
        dlg.adjustSize()
        dlg.move(
            center_global.x() - dlg.width() // 2,
            center_global.y() - dlg.height() // 2,
        )

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

        # Theme badge
        self._theme_badge = QLabel()
        self._theme_badge.setAlignment(Qt.AlignCenter)
        self._theme_badge.setFixedHeight(24)
        self._theme_badge.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: white; "
            "border-radius: 4px; padding: 2px 8px;"
        )
        panel.addWidget(self._theme_badge)

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

        # Application context
        self._application_text = QTextEdit()
        self._application_text.setReadOnly(True)
        self._application_text.setFixedHeight(60)
        self._application_text.setStyleSheet(
            "font-size: 11px; color: #336; font-style: italic; "
            "background: #f0f4ff; border: 1px solid #c0c8e8; border-radius: 4px;"
        )
        panel.addWidget(self._application_text)

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

        # Results table (grows as landmarks are confirmed)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        panel.addWidget(sep2)

        self._results_table = QTableWidget(0, 3)
        self._results_table.setHorizontalHeaderLabels(["Repère / Landmark", "mm", "Score"])
        self._results_table.horizontalHeader().setStretchLastSection(False)
        self._results_table.setColumnWidth(0, 150)
        self._results_table.setColumnWidth(1, 50)
        self._results_table.setColumnWidth(2, 55)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._results_table.setSelectionMode(QTableWidget.NoSelection)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setStyleSheet("font-size: 11px;")
        self._results_table.setMinimumHeight(120)
        panel.addWidget(self._results_table, stretch=1)

        # Opacity slider
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Transparence :"))
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(10, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setTickInterval(10)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        panel.addLayout(opacity_row)

        # Texture toggle (only visible when vertex colors are available)
        self._texture_btn = QPushButton("Mode : Texturé")
        self._texture_btn.setCheckable(True)
        self._texture_btn.setChecked(True)
        self._texture_btn.setVisible(self._vertex_colors is not None)
        self._texture_btn.clicked.connect(self._on_texture_toggled)
        panel.addWidget(self._texture_btn)

        # Sticker toggle (only visible when both clean and raw colors are available)
        self._sticker_btn = QPushButton("Stickers : masqués")
        self._sticker_btn.setCheckable(True)
        self._sticker_btn.setChecked(True)
        self._sticker_btn.setVisible(
            self._vertex_colors_clean is not None and self._vertex_colors_raw is not None
        )
        self._sticker_btn.clicked.connect(self._on_sticker_toggled)
        panel.addWidget(self._sticker_btn)

        # Mode toggle: Placement ↔ Identification inverse
        self._mode_btn = QPushButton("Mode : Placement")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(False)
        self._mode_btn.clicked.connect(self._on_mode_toggled)
        panel.addWidget(self._mode_btn)

        # Inverse identification panel (hidden by default)
        self._inverse_panel = QWidget()
        inv_layout = QVBoxLayout(self._inverse_panel)
        inv_layout.setContentsMargins(0, 0, 0, 0)
        inv_layout.setSpacing(6)

        inv_title = QLabel("Quel est ce repère ?")
        inv_title.setStyleSheet("font-size: 13px; font-weight: bold;")
        inv_layout.addWidget(inv_title)

        self._inverse_filter = QLineEdit()
        self._inverse_filter.setPlaceholderText("Filtrer...")
        self._inverse_filter.textChanged.connect(self._on_inverse_filter_changed)
        inv_layout.addWidget(self._inverse_filter)

        self._inverse_list = QListWidget()
        self._inverse_list.setMinimumHeight(120)
        self._inverse_item_codes: dict[int, str] = {}

        lm_sorted = sorted(self._landmarks, key=lambda lm: lm.name(self._lang))
        for i, lm_item in enumerate(lm_sorted):
            cat = lm_item.category if hasattr(lm_item, "category") else ""
            item_text = f"{cat} • {lm_item.name(self._lang)}"
            self._inverse_list.addItem(item_text)
            self._inverse_item_codes[i] = lm_item.code

        self._inverse_validate_btn = QPushButton("Valider ma réponse")
        self._inverse_validate_btn.setEnabled(False)
        self._inverse_validate_btn.clicked.connect(self._on_inverse_validate)

        self._inverse_list.itemSelectionChanged.connect(
            lambda: self._inverse_validate_btn.setEnabled(
                bool(self._inverse_list.selectedItems())
            )
        )
        inv_layout.addWidget(self._inverse_list)
        inv_layout.addWidget(self._inverse_validate_btn)

        self._inverse_panel.setVisible(False)
        panel.addWidget(self._inverse_panel)

        # Enter key confirms (works even when the 3-D view has focus)
        for key in (Qt.Key_Return, Qt.Key_Enter):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(self._on_confirm_if_enabled)

        # Legend
        legend_label = QLabel(
            f"<span style='color:{_MARKER_COLOR}'>●</span> Repère (après validation)<br>"
            "<span style='color:#ffdd00'>●</span> Votre sélection / Your pick<br>"
            "<span style='color:#3399ff'>●</span> Confirmé / Confirmed"
        )
        legend_label.setStyleSheet("font-size: 11px;")
        panel.addWidget(legend_label)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(320)
        root.addWidget(panel_widget)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _add_body_mesh(self, textured: bool = True) -> None:
        """Add (or replace) the body mesh actor. textured=True uses UV-sampled colours."""
        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor, render=False)
        opacity = self._opacity_slider.value() / 100.0
        if textured and self._vertex_colors is not None:
            rgba = self._vertex_colors[:, :3]
            self._mesh.point_data["RGB"] = rgba
            self._mesh_actor = self._plotter.add_mesh(
                self._mesh,
                scalars="RGB",
                rgb=True,
                smooth_shading=True,
                show_scalar_bar=False,
                opacity=opacity,
                ambient=0.4,
                diffuse=0.8,
            )
        else:
            self._mesh_actor = self._plotter.add_mesh(
                self._mesh, color=_MESH_COLOR, smooth_shading=True, opacity=opacity,
                ambient=0.3, diffuse=0.9,
            )

    def _setup_scene(self) -> None:
        pl = self._plotter
        pl.background_color = "#1a1a2e"
        # Three-point lighting for better depth on skin tones
        pl.enable_3_lights()
        self._mesh_actor = None
        self._add_body_mesh(textured=True)

        # BodyLoop markers are hidden during the exercise to avoid guiding students.
        # Ground-truth spheres are revealed one by one in _on_confirm().

        pl.enable_surface_point_picking(
            callback=self._on_surface_pick,
            show_message=False,
            left_clicking=True,
            pickable_window=False,
        )
        pl.reset_camera()
        pl.show()

    # ── Interaction callbacks ─────────────────────────────────────────────────

    def _on_surface_pick(self, point: np.ndarray) -> None:
        """Called by PyVista when the user clicks the mesh surface."""
        if self._retry_mode:
            if not self._retry_queue:
                return
        elif self._index >= len(self._landmarks):
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
        lm = self._current_landmark()
        gt = self._ground_truth.get(lm.code)   # None for EMG/SKINFOLD/ANTHRO
        self._redo_count_saved = self._redo_count
        self._redo_count = 0

        # Remove candidate sphere; add confirmed (blue)
        self._plotter.remove_actor("candidate_sphere", render=False)
        sphere = pv.Sphere(radius=8, center=self._candidate_point)
        self._plotter.add_mesh(
            sphere, color=_CONFIRMED_COLOR, name=f"{_ACTOR_PREFIX_CONFIRMED}{lm.code}"
        )

        row = self._results_table.rowCount()
        self._results_table.insertRow(row)
        name_item = QTableWidgetItem(lm.name(self._lang))
        name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._results_table.setItem(row, 0, name_item)

        result: LandmarkResult | None = None
        if gt is not None:
            result = LandmarkResult(
                code=lm.code,
                ground_truth=gt,
                student_pick=self._candidate_point.copy(),
                redo_count=self._redo_count_saved,
            )
            # In normal mode, append to the main results list
            if not self._retry_mode:
                self._results.append(result)
            # Always keep the best score across all passes
            if (
                lm.code not in self._best_results
                or result.composite_score > self._best_results[lm.code].composite_score
            ):
                self._best_results[lm.code] = result

            # Reveal correct position
            gt_sphere = pv.Sphere(radius=10, center=gt)
            self._plotter.add_mesh(gt_sphere, color=_MARKER_COLOR, name=f"guide_{lm.code}")

            # Feedback in the side panel
            color = result.feedback_color()
            self._error_label.setText(tr("error_mm_label", self._lang, value=result.error_mm))
            self._error_label.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {color};"
            )
            err_item = QTableWidgetItem(f"{result.error_mm:.1f}")
            err_item.setTextAlignment(Qt.AlignCenter)
            err_item.setBackground(QColor(color))
            err_item.setForeground(QColor("#ffffff"))
            score_item = QTableWidgetItem(
                f"{result.composite_score:.0f} ({result.grade_letter()})"
            )
            score_item.setTextAlignment(Qt.AlignCenter)
            self._results_table.setItem(row, 1, err_item)
            self._results_table.setItem(row, 2, score_item)
        else:
            # No ground truth — placed for practice, not scored
            self._error_label.setText(
                "Non évalué" if self._lang == "fr" else "Not scored"
            )
            self._error_label.setStyleSheet("font-size: 13px; color: #888;")
            self._results_table.setItem(row, 1, QTableWidgetItem("—"))
            self._results_table.setItem(row, 2, QTableWidgetItem("—"))

        self._plotter.render()
        self._results_table.scrollToBottom()

        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._candidate_point = None

        # Show debrief overlay — advancement happens inside the debrief close callback
        self._show_landmark_debrief(lm, result)

    def _on_opacity_changed(self, value: int) -> None:
        if self._mesh_actor is not None:
            self._mesh_actor.prop.opacity = value / 100.0
            self._plotter.render()

    def _on_texture_toggled(self, checked: bool) -> None:
        self._texture_btn.setText("Mode : Texturé" if checked else "Mode : Gris")
        self._add_body_mesh(textured=checked)
        self._plotter.render()

    def _on_sticker_toggled(self, checked: bool) -> None:
        self._sticker_btn.setText("Stickers : masqués" if checked else "Stickers : visibles")
        self._vertex_colors = self._vertex_colors_clean if checked else self._vertex_colors_raw
        self._add_body_mesh(textured=self._texture_btn.isChecked())
        self._plotter.render()

    def _on_confirm_if_enabled(self) -> None:
        if self._confirm_btn.isEnabled():
            self._on_confirm()

    def _on_redo(self) -> None:
        self._redo_count += 1
        self._plotter.remove_actor("candidate_sphere", render=True)
        self._candidate_point = None
        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._error_label.setText("")

    # ── Task A — Debrief overlay ──────────────────────────────────────────────

    def _show_landmark_debrief(
        self, lm: Landmark, result: LandmarkResult | None
    ) -> None:
        """Show a non-blocking modal overlay with hint + grade after validation.

        Advancement to the next landmark happens when the dialog closes.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle(
            "Résumé du repère" if self._lang == "fr" else "Landmark summary"
        )
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(440)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 20)

        # Landmark name (large, bold)
        name_lbl = QLabel(lm.name(self._lang))
        name_lbl.setWordWrap(True)
        name_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5f5f5;")
        layout.addWidget(name_lbl)

        # Grade badge (scored landmarks only)
        if result is not None:
            grade = result.grade_letter()
            grade_color = _GRADE_COLORS.get(grade, "#607D8B")
            grade_lbl = QLabel(
                f"Note : <b style='color:{grade_color};font-size:16px'>{grade}</b>"
                f"&nbsp;&nbsp;{result.composite_score:.0f}/100"
                f"&nbsp;({result.error_mm:.1f} mm)"
            )
            grade_lbl.setStyleSheet("font-size: 14px; color: #e0e0e0;")
            layout.addWidget(grade_lbl)
        else:
            not_scored_lbl = QLabel(
                "Non évalué" if self._lang == "fr" else "Not scored"
            )
            not_scored_lbl.setStyleSheet("font-size: 13px; color: #aaa;")
            layout.addWidget(not_scored_lbl)

        # Palpation hint
        hint_title = QLabel(
            "<b>Palpation</b>" if self._lang == "fr" else "<b>Palpation guide</b>"
        )
        layout.addWidget(hint_title)
        hint_box = QTextEdit()
        hint_box.setReadOnly(True)
        hint_box.setText(lm.hint(self._lang))
        hint_box.setFixedHeight(110)
        hint_box.setStyleSheet(
            "font-size: 12px; color: #111; background: #ffffff; border: 1px solid #bbb; border-radius: 4px;"
        )
        layout.addWidget(hint_box)

        # Clinical application (only when non-empty)
        app_text = lm.application(self._lang) if hasattr(lm, "application") else ""
        if app_text:
            app_title = QLabel(
                "<b>Application clinique</b>"
                if self._lang == "fr"
                else "<b>Clinical application</b>"
            )
            layout.addWidget(app_title)
            app_box = QTextEdit()
            app_box.setReadOnly(True)
            app_box.setText(app_text)
            app_box.setFixedHeight(70)
            app_box.setStyleSheet(
                "font-size: 11px; color: #1a237e; font-style: italic; "
                "background: #e8edf8; border: 1px solid #90a4d4; border-radius: 4px;"
            )
            layout.addWidget(app_box)

        # Continue button
        continue_btn = QPushButton(
            "Continuer →" if self._lang == "fr" else "Continue →"
        )
        continue_btn.setDefault(True)
        continue_btn.setStyleSheet(
            "font-size: 14px; padding: 8px 24px; font-weight: bold;"
        )
        layout.addWidget(continue_btn, alignment=Qt.AlignCenter)

        # Enter / Return shortcut inside the dialog
        for key in (Qt.Key_Return, Qt.Key_Enter):
            sc = QShortcut(QKeySequence(key), dlg)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(dlg.accept)

        continue_btn.clicked.connect(dlg.accept)

        # Advancement happens exactly once, when the dialog closes
        dlg.finished.connect(lambda _result: self._advance_after_debrief(lm, result))

        dlg.show()
        dlg.raise_()
        self._center_dialog(dlg)

    def _advance_after_debrief(
        self, prev_lm: Landmark, result: LandmarkResult | None
    ) -> None:
        """Called when the debrief dialog closes. Advances to the next landmark."""
        if self._retry_mode:
            # Pop the front of the queue
            done_lm = self._retry_queue.pop(0)
            # If still D, push back for another attempt
            if result is not None and result.grade_letter() == "D":
                self._retry_queue.append(done_lm)
            if not self._retry_queue:
                self._finish_session()
            else:
                self._update_instruction_panel()
        else:
            self._index += 1
            if self._index >= len(self._landmarks):
                self._finish_session()
                return
            next_lm = self._landmarks[self._index]
            shown = self._maybe_show_category_transition(prev_lm, next_lm)
            if not shown:
                self._update_instruction_panel()

    # ── Task B — Category transition screen ───────────────────────────────────

    def _maybe_show_category_transition(
        self, prev_lm: Landmark | None, next_lm: Landmark
    ) -> bool:
        """Show a transition screen when switching landmark categories.

        Returns True if a transition dialog was displayed (caller must NOT call
        _update_instruction_panel directly — the dialog's close callback does it).
        """
        if prev_lm is not None and prev_lm.category == next_lm.category:
            return False

        cat = next_lm.category
        cat_fr, cat_en = CATEGORY_LABELS.get(cat, (cat, cat))
        desc_fr, desc_en = _CATEGORY_DESCRIPTIONS.get(cat, ("", ""))
        cat_label = cat_fr if self._lang == "fr" else cat_en
        desc = desc_fr if self._lang == "fr" else desc_en
        theme_color = (
            next_lm.theme_color() if hasattr(next_lm, "theme_color") else "#607D8B"
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(cat_label)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(400)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 32, 28, 24)

        # Category title in theme colour
        title_lbl = QLabel(cat_label)
        title_lbl.setWordWrap(True)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {theme_color};"
        )
        layout.addWidget(title_lbl)

        # Description sentence
        if desc:
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setAlignment(Qt.AlignCenter)
            desc_lbl.setStyleSheet("font-size: 13px; color: #444; padding: 0 8px;")
            layout.addWidget(desc_lbl)

        layout.addSpacing(8)

        # Start button
        start_btn = QPushButton(
            "Commencer →" if self._lang == "fr" else "Start →"
        )
        start_btn.setDefault(True)
        start_btn.setStyleSheet(
            f"font-size: 14px; padding: 10px 28px; font-weight: bold; "
            f"background-color: {theme_color}; color: white; border-radius: 6px;"
        )
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        for key in (Qt.Key_Return, Qt.Key_Enter):
            sc = QShortcut(QKeySequence(key), dlg)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(dlg.accept)

        start_btn.clicked.connect(dlg.accept)
        dlg.finished.connect(lambda _: self._update_instruction_panel())

        dlg.show()
        dlg.raise_()
        self._center_dialog(dlg)

        return True

    # ── Task C — D-grade retry ────────────────────────────────────────────────

    def _show_retry_intro(self, d_landmarks: list[Landmark]) -> None:
        """Show the retry introduction and let the student start or skip."""
        n = len(d_landmarks)
        if self._lang == "fr":
            body = (
                f"{n} repère{'s ont' if n > 1 else ' a'} une note D.\n"
                "Vous allez les reprendre jusqu'à obtenir au moins C."
            )
        else:
            body = (
                f"{n} landmark{'s have' if n > 1 else ' has'} a grade D.\n"
                "You will redo them until you reach at least a C."
            )

        dlg = QDialog(self)
        dlg.setWindowTitle(
            "Reprise des repères insuffisants"
            if self._lang == "fr"
            else "Retry: insufficient landmarks"
        )
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(400)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(14)
        layout.setContentsMargins(28, 28, 28, 24)

        title_lbl = QLabel(
            "Reprise des repères D"
            if self._lang == "fr"
            else "Retrying grade-D landmarks"
        )
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #c62828;"
        )
        layout.addWidget(title_lbl)

        body_lbl = QLabel(body)
        body_lbl.setWordWrap(True)
        body_lbl.setAlignment(Qt.AlignCenter)
        body_lbl.setStyleSheet("font-size: 13px; color: #444;")
        layout.addWidget(body_lbl)

        layout.addSpacing(4)

        btn_row = QHBoxLayout()

        start_btn = QPushButton(
            "Commencer la reprise →"
            if self._lang == "fr"
            else "Start retry →"
        )
        start_btn.setDefault(True)
        start_btn.setStyleSheet(
            "font-size: 13px; padding: 8px 16px; font-weight: bold;"
        )

        skip_btn = QPushButton(
            "Terminer quand même"
            if self._lang == "fr"
            else "Finish anyway"
        )
        skip_btn.setStyleSheet("font-size: 12px; padding: 8px 12px; color: #888;")

        btn_row.addWidget(start_btn)
        btn_row.addWidget(skip_btn)
        layout.addLayout(btn_row)

        for key in (Qt.Key_Return, Qt.Key_Enter):
            sc = QShortcut(QKeySequence(key), dlg)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(dlg.accept)

        # Default finished handler (e.g. window X closed) → start retry
        dlg.finished.connect(lambda _: self._begin_retry_session(d_landmarks))

        def _start() -> None:
            dlg.finished.disconnect()
            dlg.accept()
            self._begin_retry_session(d_landmarks)

        def _skip() -> None:
            dlg.finished.disconnect()
            dlg.accept()
            self._emit_final_session()

        start_btn.clicked.connect(_start)
        skip_btn.clicked.connect(_skip)

        dlg.show()
        dlg.raise_()
        self._center_dialog(dlg)

    def _begin_retry_session(self, d_landmarks: list[Landmark]) -> None:
        """Initialise the retry queue and reset UI state for the retry pass."""
        self._retry_mode = True
        self._retry_queue = list(d_landmarks)
        random.shuffle(self._retry_queue)
        self._candidate_point = None
        self._redo_count = 0
        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._error_label.setText("")
        self._update_instruction_panel()

    def _emit_final_session(self) -> None:
        """Emit session_complete using the best score accumulated for every landmark."""
        final_results = list(self._best_results.values())
        # Include results for unscored landmarks (no ground truth) from normal pass
        scored_codes = {r.code for r in final_results}
        for r in self._results:
            if r.code not in scored_codes:
                final_results.append(r)

        score = SessionScore(final_results)
        self._name_label.setText(
            tr("session_complete", self._lang, mean=score.mean_error_mm)
        )
        self._error_label.setText(
            f"Score global : {score.global_score:.0f}/100 — {score.global_grade}"
        )
        self._error_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #3399ff;"
        )
        self._instr_label.setText("")
        self._hint_text.setText("")
        self._progress_label.setText("")
        self._theme_badge.setVisible(False)
        self._application_text.setVisible(False)
        self.session_complete.emit(score)

    def _finish_session(self) -> None:
        """Called when the current pass (normal or retry) is exhausted."""
        if self._retry_mode:
            # Retry pass done — emit best results
            self._emit_final_session()
            return

        # Normal pass — check for grade-D landmarks to retry
        d_codes = {r.code for r in self._results if r.grade_letter() == "D"}
        d_landmarks = [lm for lm in self._landmarks if lm.code in d_codes]

        if d_landmarks:
            self._show_retry_intro(d_landmarks)
            return

        # No D grades — emit the normal session results directly
        score = SessionScore(self._results)
        self._name_label.setText(
            tr("session_complete", self._lang, mean=score.mean_error_mm)
        )
        self._error_label.setText(
            f"Score global : {score.global_score:.0f}/100 — {score.global_grade}"
        )
        self._error_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #3399ff;"
        )
        self._instr_label.setText("")
        self._hint_text.setText("")
        self._progress_label.setText("")
        self._theme_badge.setVisible(False)
        self._application_text.setVisible(False)
        self.session_complete.emit(score)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_instruction_panel(self) -> None:
        if self._retry_mode:
            if not self._retry_queue:
                return
            lm = self._retry_queue[0]
            n = len(self._retry_queue)
            s = "s" if n > 1 else ""
            if self._lang == "fr":
                self._progress_label.setText(
                    f"Reprise — {n} repère{s} restant{s}"
                )
            else:
                self._progress_label.setText(
                    f"Retry — {n} landmark{s} remaining"
                )
        else:
            if self._index >= len(self._landmarks):
                return
            lm = self._landmarks[self._index]
            side_label = "Côté gauche" if self._chosen_side == "left" else "Côté droit"
            if self._lang == "en":
                side_label = "Right side" if self._chosen_side == "right" else "Left side"
            self._progress_label.setText(
                f"{side_label} — "
                + tr(
                    "landmark_label",
                    self._lang,
                    index=self._index + 1,
                    total=len(self._landmarks),
                )
            )

        # Theme badge
        color = lm.theme_color() if hasattr(lm, "theme_color") else "#607D8B"
        label = lm.theme_label(self._lang) if hasattr(lm, "theme_label") else ""
        self._theme_badge.setText(label)
        self._theme_badge.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: white; "
            f"background-color: {color}; border-radius: 4px; padding: 2px 8px;"
        )
        self._theme_badge.setVisible(bool(label))

        self._name_label.setText(lm.name(self._lang))
        self._hint_text.setText(lm.hint(self._lang))

        # Application context
        app_text = lm.application(self._lang) if hasattr(lm, "application") else ""
        self._application_text.setText(app_text)
        self._application_text.setVisible(bool(app_text))

        self._instr_label.setText(tr("instructions", self._lang))
        self._error_label.setText("")

        self._plotter.render()

    def _toggle_lang(self) -> None:
        self._lang = "en" if self._lang == "fr" else "fr"
        self._lang_btn.setText(tr("lang_toggle", self._lang))
        self._instr_label.setText(tr("instructions", self._lang))
        if self._retry_mode or self._index < len(self._landmarks):
            self._update_instruction_panel()

    # ── Mode inverse — identification anatomique ──────────────────────────────

    def _on_mode_toggled(self, checked: bool) -> None:
        # Prevent entering inverse mode during a retry session
        if self._retry_mode and checked:
            self._mode_btn.setChecked(False)
            return
        self._inverse_mode = checked
        label = "Mode : Identification" if checked else "Mode : Placement"
        self._mode_btn.setText(label)
        if checked:
            self._start_inverse_session()
        else:
            self._stop_inverse_session()

    def _start_inverse_session(self) -> None:
        """Build the identification queue and enter inverse mode."""
        self._inverse_queue = [
            lm for lm in self._landmarks if lm.code in self._ground_truth
        ]
        random.shuffle(self._inverse_queue)
        self._inverse_index = 0

        self._plotter.disable_picking()

        # Hide placement widgets; show identification panel
        self._confirm_btn.setVisible(False)
        self._redo_btn.setVisible(False)
        self._instr_label.setVisible(False)
        self._inverse_panel.setVisible(True)

        self._show_inverse_landmark()

    def _stop_inverse_session(self) -> None:
        """Leave inverse mode and restore normal surface picking."""
        if self._inverse_shown_actor is not None:
            self._plotter.remove_actor(self._inverse_shown_actor, render=False)
            self._inverse_shown_actor = None

        # Re-enable picking with the same parameters used in _setup_scene
        self._plotter.enable_surface_point_picking(
            callback=self._on_surface_pick,
            show_message=False,
            left_clicking=True,
            pickable_window=False,
        )
        self._plotter.render()

        # Restore placement widgets; hide identification panel
        self._confirm_btn.setVisible(True)
        self._redo_btn.setVisible(True)
        self._instr_label.setVisible(True)
        self._inverse_panel.setVisible(False)

        if self._mode_btn.isChecked():
            self._mode_btn.setChecked(False)
        self._inverse_mode = False

    def _show_inverse_landmark(self) -> None:
        """Display the ground-truth sphere for the current inverse landmark."""
        if self._inverse_index >= len(self._inverse_queue):
            self._finish_inverse_session()
            return

        lm = self._inverse_queue[self._inverse_index]
        gt = self._ground_truth[lm.code]

        # Remove previous sphere
        if self._inverse_shown_actor is not None:
            self._plotter.remove_actor(self._inverse_shown_actor, render=False)

        actor_name = f"inverse_shown_{lm.code}"
        sphere = pv.Sphere(radius=22, center=gt)
        self._plotter.add_mesh(sphere, color=_MARKER_COLOR, name=actor_name)
        self._inverse_shown_actor = actor_name

        # Orient camera toward the shown sphere.
        # Compute a side-view position: project gt onto the horizontal plane,
        # find the outward direction from the body axis, then place the camera
        # at gt + direction * distance so the landmark is always facing the viewer.
        import numpy as _np
        body_cx, body_cz = 0.0, 0.0  # body is centred at origin horizontally
        horiz = _np.array([gt[0] - body_cx, 0.0, gt[2] - body_cz], dtype=float)
        horiz_norm = _np.linalg.norm(horiz)
        if horiz_norm < 1.0:
            direction = _np.array([0.0, 0.0, 1.0])  # fallback: front view
        else:
            direction = horiz / horiz_norm
        cam_distance = 650.0  # mm — close enough to see the landmark clearly
        cam_pos = gt + direction * cam_distance
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(gt)
        cam.up = (0.0, 1.0, 0.0)
        self._plotter.render()

        # Update side panel (name hidden so as not to reveal the answer)
        n = len(self._inverse_queue)
        self._progress_label.setText(
            f"Identification — {self._inverse_index + 1} / {n}"
        )
        self._name_label.setText("?")
        self._hint_text.setText("")
        self._application_text.setText("")
        self._error_label.setText("")

        # Reset identification list
        self._inverse_filter.clear()
        self._inverse_list.clearSelection()
        self._inverse_validate_btn.setEnabled(False)
        for i in range(self._inverse_list.count()):
            item = self._inverse_list.item(i)
            item.setHidden(False)
            item.setData(Qt.BackgroundRole, None)

    def _on_inverse_filter_changed(self, text: str) -> None:
        """Show/hide list items according to the filter text (case-insensitive)."""
        lower = text.lower()
        for i in range(self._inverse_list.count()):
            item = self._inverse_list.item(i)
            item.setHidden(lower not in item.text().lower())

    def _on_inverse_validate(self) -> None:
        """Check whether the selected landmark matches the displayed sphere."""
        selected = self._inverse_list.selectedItems()
        if not selected:
            return
        item = selected[0]
        row = self._inverse_list.row(item)
        selected_code = self._inverse_item_codes.get(row)
        expected_code = self._inverse_queue[self._inverse_index].code

        if selected_code == expected_code:
            # Correct — swap green sphere for blue, then advance after 1 s
            if self._inverse_shown_actor is not None:
                self._plotter.remove_actor(self._inverse_shown_actor, render=False)
            lm = self._inverse_queue[self._inverse_index]
            gt = self._ground_truth[lm.code]
            blue_name = f"inverse_correct_{lm.code}"
            self._plotter.add_mesh(
                pv.Sphere(radius=12, center=gt), color=_CONFIRMED_COLOR, name=blue_name
            )
            self._inverse_shown_actor = blue_name
            self._plotter.render()

            self._error_label.setText("✓ Correct !")
            self._error_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #2e7d32;"
            )
            self._inverse_validate_btn.setEnabled(False)

            def _advance() -> None:
                self._inverse_index += 1
                self._show_inverse_landmark()

            QTimer.singleShot(1000, _advance)
        else:
            # Incorrect — let the student try again
            self._error_label.setText("✗ Essayez encore")
            self._error_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #c62828;"
            )
            item.setBackground(QColor("#ffcccc"))

    def _finish_inverse_session(self) -> None:
        """Display a summary dialog and return to placement mode."""
        if self._inverse_shown_actor is not None:
            self._plotter.remove_actor(self._inverse_shown_actor, render=False)
            self._inverse_shown_actor = None
        self._plotter.render()

        n = len(self._inverse_queue)
        dlg = QDialog(self)
        dlg.setWindowTitle("Session terminée")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(360)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 28, 28, 24)

        msg = QLabel(
            f"Session d'identification terminée !\n"
            f"Score : {n} / {n}"
        )
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet("font-size: 15px;")
        layout.addWidget(msg)

        close_btn = QPushButton("Fermer")
        close_btn.setDefault(True)
        close_btn.setStyleSheet("font-size: 13px; padding: 8px 20px;")
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

        close_btn.clicked.connect(dlg.accept)
        dlg.finished.connect(lambda _: self._stop_inverse_session())

        dlg.show()
        dlg.raise_()
        self._center_dialog(dlg)
