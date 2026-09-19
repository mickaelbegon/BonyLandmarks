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

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .dialogs import (
    build_category_transition_dialog,
    build_debrief_dialog,
    build_phase2_dialog,
    build_retry_dialog,
)
from .i18n import Language, tr
from .landmarks_extended import LANDMARKS, Landmark
from .scoring import LandmarkResult, SessionScore
from .session import SessionController


# ─── colour constants ────────────────────────────────────────────────────────

_MESH_COLOR = "#c8b8a8"
_MARKER_COLOR = "#00cc44"       # BodyLoop reference markers (green)
_CANDIDATE_COLOR = "#ffdd00"    # Student candidate pick (yellow)
_CONFIRMED_COLOR = "#3399ff"    # Confirmed student pick (blue)
_ACTOR_PREFIX_REF = "ref_"
_ACTOR_PREFIX_CONFIRMED = "confirmed_"

# Inverse identification — category labels (fr, en) and item text colours
_INVERSE_CAT_LABELS: dict[str, tuple[str, str]] = {
    "BONE":     ("Repères osseux",               "Bone landmarks"),
    "EMG":      ("Sites électrodes",              "EMG electrode sites"),
    "SKINFOLD": ("Plis cutanés",                  "Skinfold sites"),
    "ANTHRO":   ("Mesures anthropométriques",     "Anthropometric measures"),
}
_INVERSE_CAT_COLORS: dict[str, str] = {
    "BONE":     "#b0d0ff",
    "EMG":      "#ffd0a0",
    "SKINFOLD": "#b0f0c0",
    "ANTHRO":   "#e0b0ff",
}

# ─── Navigation overlay — PNG body silhouette icons ──────────────────────────

from pathlib import Path as _Path

_ICONS_DIR = _Path(__file__).parent / "icons"
_body_icon_cache: dict[tuple[str, int, str], QIcon] = {}


def _body_icon(filename: str, size: int = 44) -> QIcon:
    """Load a white-bg body silhouette PNG and make the white background transparent."""
    cache_key = (filename, size)
    if cache_key in _body_icon_cache:
        return _body_icon_cache[cache_key]

    path = _ICONS_DIR / filename
    pix = QPixmap(str(path))
    if pix.isNull():
        return QIcon()
    pix = pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    img = pix.toImage().convertToFormat(QImage.Format_ARGB32)

    w, h = img.width(), img.height()
    for y in range(h):
        for x in range(w):
            pixel = QColor(img.pixel(x, y))
            r, g, b, a = pixel.red(), pixel.green(), pixel.blue(), pixel.alpha()
            brightness = (r + g + b) / 3
            if brightness > 230:
                # Pure white background → fully transparent
                img.setPixel(x, y, QColor(0, 0, 0, 0).rgba())
            elif brightness > 180:
                # Anti-aliased edge → partial transparency proportional to whiteness
                alpha = int((230 - brightness) / 50 * 255)
                img.setPixel(x, y, QColor(r, g, b, alpha).rgba())
            # else: silhouette pixels keep their original color (light gray visible on dark bg)

    icon = QIcon(QPixmap.fromImage(img))
    _body_icon_cache[cache_key] = icon
    return icon


class _BlurWorker(QThread):
    """Calcule le blur du visage dans un thread de fond."""

    result_ready = Signal(object, object, object)  # (mask, blurred_colors, blurred_points)
    error = Signal(str)

    def __init__(self, mesh_points, mesh_faces, ground_truth, base_colors, acromion_offset: int = 60):
        super().__init__()
        self._mesh_points = mesh_points
        self._mesh_faces = mesh_faces
        self._ground_truth = ground_truth
        self._base_colors = base_colors
        self._acromion_offset = acromion_offset

    def run(self):
        try:
            from .face_blur import build_face_mask, blur_vertex_colors, blur_mesh_geometry
            mask = build_face_mask(self._mesh_points, self._ground_truth,
                                   acromion_offset=float(self._acromion_offset))
            blurred_colors = blur_vertex_colors(self._base_colors.copy(), mask, self._mesh_points)
            blurred_points = blur_mesh_geometry(self._mesh_points.copy(), mask, self._mesh_faces)
            self.result_ready.emit(mask, blurred_colors, blurred_points)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error.emit(str(exc))


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
        tutorial_mode: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tutorial_mode = tutorial_mode
        self._mesh = mesh
        self._vertex_colors_clean = vertex_colors
        self._vertex_colors_raw = vertex_colors_raw
        self._vertex_colors = vertex_colors          # currently active
        self._all_markers = all_markers or {}
        self._ground_truth = ground_truth
        self._lang = lang

        # All session state and logic lives in a Qt-free controller.
        self._session = SessionController(
            landmarks=[lm for lm in LANDMARKS if lm.code in landmark_codes],
            ground_truth=ground_truth,
            lang=lang,
        )

        # Inverse identification mode (UI state)
        self._inverse_mode: bool = False
        self._inverse_shown_actor: str | None = None  # PyVista actor name of the shown sphere

        self._candidate_point: np.ndarray | None = None
        self._redo_count: int = 0

        # Face-blur state
        self._face_blurred: bool = False
        self._colors_blurred: np.ndarray | None = None  # cached blur result
        self._points_blurred: np.ndarray | None = None   # lissage géométrique mis en cache
        self._points_original: np.ndarray | None = None  # positions originales (sauvegardées)
        self._blur_worker: _BlurWorker | None = None      # thread actif
        self._blur_zone_offset: int = 60                  # mm above acromions (slider)

        self._build_ui()
        self._setup_scene()
        if self._session.mixed_session:
            self._start_inverse_session(queue=self._session.inverse_landmarks)
        else:
            self._update_instruction_panel()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _current_landmark(self) -> Landmark:
        """Return the landmark currently being placed."""
        if self._session.retry_mode:
            return self._session.current_retry_landmark()
        return self._session.current_landmark()

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

        # Floating navigation overlay (positioned top-right of the 3D viewport)
        self._build_nav_overlay()

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

        # Mode toggle: Placement ↔ Identification inverse
        # Hidden in mixed sessions where phases auto-sequence.
        self._mode_btn = QPushButton("Mode : Placement")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(False)
        self._mode_btn.clicked.connect(self._on_mode_toggled)
        self._mode_btn.setVisible(not self._session.mixed_session)
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
        self._inverse_filter.setPlaceholderText(
            "Rechercher un repère... (ex: acromion)"
        )
        self._inverse_filter.textChanged.connect(self._on_inverse_filter_changed)
        inv_layout.addWidget(self._inverse_filter)

        self._inverse_list = QListWidget()
        self._inverse_list.setMinimumHeight(120)

        # All inverse-mode landmarks sorted alphabetically by name (flat list, no headers).
        # Landmark code is stored in Qt.UserRole; category key in Qt.UserRole+1.
        all_inverse = sorted(
            self._session.all_session_landmarks,
            key=lambda lm: lm.name(self._lang).lower()
        )
        for lm_item in all_inverse:
            item_text = lm_item.name(self._lang)
            list_item = QListWidgetItem(item_text)
            color = _INVERSE_CAT_COLORS.get(lm_item.category, "#cccccc")
            list_item.setForeground(QColor(color))
            list_item.setData(Qt.UserRole, lm_item.code)
            list_item.setData(Qt.UserRole + 1, lm_item.category)
            self._inverse_list.addItem(list_item)

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

        # ── Display settings (always at the bottom) ───────────────────────────
        sep_settings = QFrame()
        sep_settings.setFrameShape(QFrame.HLine)
        sep_settings.setFrameShadow(QFrame.Sunken)
        panel.addWidget(sep_settings)

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

        # Face blur toggle (only visible when vertex colors are available)
        self._blur_face_btn = QPushButton("Visage : affiché")
        self._blur_face_btn.setCheckable(True)
        self._blur_face_btn.setChecked(False)
        self._blur_face_btn.setVisible(self._vertex_colors is not None)
        self._blur_face_btn.clicked.connect(self._on_blur_face_toggled)
        panel.addWidget(self._blur_face_btn)

        # Zone slider (visible with blur button)
        _zone_row = QHBoxLayout()
        _zone_lbl = QLabel("Zone :")
        _zone_lbl.setFixedWidth(46)
        self._blur_zone_slider = QSlider(Qt.Horizontal)
        self._blur_zone_slider.setRange(0, 200)
        self._blur_zone_slider.setValue(self._blur_zone_offset)
        self._blur_zone_val_lbl = QLabel(f"{self._blur_zone_offset} mm")
        self._blur_zone_val_lbl.setFixedWidth(46)
        self._blur_zone_slider.valueChanged.connect(self._on_blur_zone_label)
        self._blur_zone_slider.sliderReleased.connect(self._on_blur_zone_released)
        _zone_row.addWidget(_zone_lbl)
        _zone_row.addWidget(self._blur_zone_slider)
        _zone_row.addWidget(self._blur_zone_val_lbl)
        _zone_widget = QWidget()
        _zone_widget.setLayout(_zone_row)
        _zone_widget.setVisible(self._vertex_colors is not None)
        panel.addWidget(_zone_widget)
        self._blur_zone_widget = _zone_widget

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
        pl.show()

        # Set initial camera: avatar upright, seen from front, full body in view.
        # Detect the vertical axis from bounds (longest extent) to work regardless
        # of whether the GLB uses Y-up or Z-up conventions.
        bounds = self._mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        extents = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
        up_axis = extents.index(max(extents))   # body's vertical axis (0=X, 1=Y, 2=Z)
        sorted_axes = sorted(range(3), key=lambda i: extents[i])
        front_axis = sorted_axes[0]             # smallest extent = depth = "front"
        side_axis = ({0, 1, 2} - {up_axis, front_axis}).pop()
        centers = [
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        ]
        height = extents[up_axis]
        cam_distance = height * 2.5             # far enough to see the whole body
        cam_pos = list(centers)
        cam_pos[front_axis] += cam_distance
        up_vec = [0.0, 0.0, 0.0]
        up_vec[up_axis] = 1.0
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(centers)
        cam.up = tuple(up_vec)
        self._plotter.render()

        # Save camera geometry for navigation buttons / shortcuts
        self._up_axis = up_axis
        self._front_axis = front_axis
        self._side_axis = side_axis
        self._body_centers = centers
        self._body_extents = extents
        self._cam_distance = cam_distance

        # Wire keyboard shortcuts now that axes are known
        self._setup_nav_shortcuts()

        # Ensure overlay stays on top after the plotter renders
        if hasattr(self, "_nav_overlay"):
            self._nav_overlay.raise_()

    # ── Navigation toolbar ────────────────────────────────────────────────────

    def _set_view(self, cam_axis: int, direction: int) -> None:
        """Set camera to look along *cam_axis* (0=X,1=Y,2=Z), direction +1 or -1."""
        cam_pos = list(self._body_centers)
        cam_pos[cam_axis] += direction * self._cam_distance
        up_vec = [0.0, 0.0, 0.0]
        up_vec[self._up_axis] = 1.0
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(self._body_centers)
        cam.up = tuple(up_vec)
        self._plotter.render()

    def _view_top(self) -> None:
        """Set camera directly above, with the front axis pointing 'up' in screen space."""
        cam_pos = list(self._body_centers)
        cam_pos[self._up_axis] += self._cam_distance
        up_vec = [0.0, 0.0, 0.0]
        up_vec[self._front_axis] = 1.0  # front axis becomes screen-up when looking down
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(self._body_centers)
        cam.up = tuple(up_vec)
        self._plotter.render()

    def _reset_view(self) -> None:
        """Return to the initial full-body front view."""
        self._set_view(self._front_axis, -1)

    def _build_nav_overlay(self) -> None:
        """Create a semi-transparent floating toolbar anchored to the 3D viewport."""
        container = self._plotter.interactor

        overlay = QWidget(container)
        overlay.setObjectName("nav_overlay")
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setStyleSheet("""
            QWidget#nav_overlay {
                background: transparent;
            }
            QPushButton {
                background-color: rgba(26, 26, 46, 180);
                color: #e0e0e0;
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 6px;
                font-size: 11px;
                padding: 0px;
                min-width: 84px;
                min-height: 84px;
                max-width: 84px;
                max-height: 84px;
            }
            QPushButton:hover {
                background-color: rgba(60, 80, 140, 210);
                border-color: rgba(100,160,255,0.6);
            }
            QPushButton:pressed {
                background-color: rgba(30, 60, 120, 230);
            }
        """)
        layout = QVBoxLayout(overlay)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # (png_filename_or_None, tooltip_fr, callback)
        buttons = [
            ("view_front.png",      "Vue avant [1]",     lambda: self._set_view(self._front_axis, -1)),
            ("view_back.png",       "Vue arrière [2]",   lambda: self._set_view(self._front_axis, +1)),
            ("view_side_right.png", "Vue droite [3]",    lambda: self._set_view(self._side_axis, -1)),
            ("view_side_left.png",  "Vue gauche [4]",    lambda: self._set_view(self._side_axis, +1)),
            ("view_top.png",        "Vue dessus [5]",    self._view_top),
            (None,                  "Réinitialiser [R]", self._reset_view),
        ]
        for filename, tooltip, cb in buttons:
            btn = QPushButton()
            btn.setToolTip(tooltip)
            if filename is not None:
                icon = _body_icon(filename, size=66)
                btn.setIcon(icon)
                btn.setIconSize(QSize(66, 66))
            else:
                btn.setText("↺")
                btn.setStyleSheet(btn.styleSheet() + "font-size: 20px;")
            btn.clicked.connect(cb)
            layout.addWidget(btn)

        overlay.adjustSize()
        self._nav_overlay = overlay
        self._position_nav_overlay()

    def _position_nav_overlay(self) -> None:
        """Move the overlay to the top-right corner of the 3D viewport."""
        if not hasattr(self, "_nav_overlay"):
            return
        container = self._plotter.interactor
        margin = 10
        w = self._nav_overlay.width() or self._nav_overlay.sizeHint().width()
        h = self._nav_overlay.height() or self._nav_overlay.sizeHint().height()
        x = container.width() - w - margin
        y = margin
        self._nav_overlay.move(x, y)
        self._nav_overlay.raise_()

    def _setup_nav_shortcuts(self) -> None:
        """Register keyboard shortcuts for the 6 navigation views (called after axes are set)."""
        shortcuts = [
            ("1", lambda: self._set_view(self._front_axis, +1)),
            ("2", lambda: self._set_view(self._front_axis, -1)),
            ("3", lambda: self._set_view(self._side_axis, -1)),
            ("4", lambda: self._set_view(self._side_axis, +1)),
            ("5", self._view_top),
            ("r", self._reset_view),
            ("R", self._reset_view),
        ]
        for key, cb in shortcuts:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(cb)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_nav_overlay()

    # ── Interaction callbacks ─────────────────────────────────────────────────

    def _on_surface_pick(self, point: np.ndarray) -> None:
        """Called by PyVista when the user clicks the mesh surface."""
        if self._session.retry_mode:
            if self._session.retry_finished():
                return
        elif self._session.placement_finished():
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
            self._session.record_result(result)

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
        self._colors_blurred = None  # invalidate blur cache (base colors changed)
        self._points_blurred = None  # invalider aussi le cache géométrique
        if self._face_blurred:
            self._blur_face_btn.setChecked(False)
            self._face_blurred = False
            self._blur_face_btn.setText("Visage : affiché")
        self._texture_btn.setText("Mode : Texturé" if checked else "Mode : Gris")
        self._add_body_mesh(textured=checked)
        self._plotter.render()

    def _on_sticker_toggled(self, checked: bool) -> None:
        self._colors_blurred = None  # invalidate blur cache (base colors changed)
        self._points_blurred = None  # invalider aussi le cache géométrique
        if self._face_blurred:
            self._blur_face_btn.setChecked(False)
            self._face_blurred = False
            self._blur_face_btn.setText("Visage : affiché")
        self._sticker_btn.setText("Stickers : masqués" if checked else "Stickers : visibles")
        self._vertex_colors = self._vertex_colors_clean if checked else self._vertex_colors_raw
        self._add_body_mesh(textured=self._texture_btn.isChecked())
        self._plotter.render()

    def _refresh_mesh_colors(self) -> None:
        """Re-add the body mesh actor with the current vertex colors.

        VTK does not pick up in-place point_data changes without removing and
        re-adding the actor, so we delegate to ``_add_body_mesh`` — the same
        path used by the sticker and texture toggles.
        """
        if self._vertex_colors is not None:
            self._add_body_mesh(textured=self._texture_btn.isChecked())
            self._plotter.render()

    def _on_blur_face_toggled(self, checked: bool) -> None:
        """Apply or remove face blurring."""
        if checked:
            if self._colors_blurred is not None and self._points_blurred is not None:
                # Cache disponible — appliquer immédiatement
                self._apply_blur()
            else:
                # Lancer le calcul en arrière-plan
                self._start_blur_computation()
        else:
            self._remove_blur()

    def _start_blur_computation(self) -> None:
        """Lance le calcul du blur dans un QThread de fond."""
        if self._blur_worker is not None and self._blur_worker.isRunning():
            return  # déjà en cours

        base = self._vertex_colors_clean if self._vertex_colors_clean is not None \
               else self._vertex_colors
        if base is None:
            self._blur_face_btn.setChecked(False)
            return

        # Sauvegarder les points originaux si pas encore fait
        if self._points_original is None:
            self._points_original = np.asarray(self._mesh.points, dtype=np.float64).copy()

        self._blur_face_btn.setText("Visage : calcul...")
        self._blur_face_btn.setEnabled(False)

        self._blur_worker = _BlurWorker(
            mesh_points=self._points_original.copy(),
            mesh_faces=np.asarray(self._mesh.faces).copy(),
            ground_truth=self._ground_truth,
            base_colors=base.copy(),
            acromion_offset=self._blur_zone_offset,
        )
        self._blur_worker.result_ready.connect(self._on_blur_computed)
        self._blur_worker.error.connect(self._on_blur_error)
        self._blur_worker.start()

    def _on_blur_computed(self, mask, blurred_colors, blurred_points) -> None:
        """Appelé dans le thread principal quand le calcul est terminé."""
        self._colors_blurred = blurred_colors
        self._points_blurred = blurred_points
        self._blur_face_btn.setEnabled(True)
        # Le bouton est déjà coché (l'utilisateur a cliqué) — appliquer
        if self._blur_face_btn.isChecked():
            self._apply_blur()
        else:
            # L'utilisateur a décoché pendant le calcul
            self._blur_face_btn.setText("Visage : affiché")

    def _on_blur_zone_label(self, value: int) -> None:
        """Met à jour le label pendant le drag sans relancer le calcul."""
        self._blur_zone_offset = value
        self._blur_zone_val_lbl.setText(f"{value} mm")

    def _on_blur_zone_released(self) -> None:
        """Relance le calcul au relâchement du slider."""
        self._colors_blurred = None
        self._points_blurred = None
        if self._face_blurred:
            # Restaurer l'original puis relancer avec la nouvelle zone
            self._remove_blur()
            self._blur_face_btn.setChecked(True)
            self._start_blur_computation()

    def _on_blur_error(self, msg: str) -> None:
        """Appelé si le calcul échoue."""
        print(f"[face_blur] erreur : {msg}")
        self._blur_face_btn.setChecked(False)
        self._blur_face_btn.setText("Visage : affiché")
        self._blur_face_btn.setEnabled(True)

    def _apply_blur(self) -> None:
        """Applique les couleurs et la géométrie floutées (cache disponible)."""
        self._vertex_colors = self._colors_blurred
        if self._points_blurred is not None:
            self._mesh.points = self._points_blurred
            self._mesh.compute_normals(inplace=True)
        self._face_blurred = True
        self._blur_face_btn.setText("Visage : flouté")
        self._blur_face_btn.setEnabled(True)
        self._refresh_mesh_colors()

    def _remove_blur(self) -> None:
        """Restaure les couleurs et la géométrie originales."""
        # Restaurer les points originaux
        if self._points_original is not None:
            self._mesh.points = self._points_original
            self._mesh.compute_normals(inplace=True)
        # Restaurer les couleurs selon l'état actuel des toggles
        if self._sticker_btn.isChecked():
            self._vertex_colors = self._vertex_colors_clean
        else:
            self._vertex_colors = self._vertex_colors_raw
        self._face_blurred = False
        self._blur_face_btn.setText("Visage : affiché")
        self._refresh_mesh_colors()

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
        dlg = build_debrief_dialog(lm, result, self._lang, self, self._center_dialog)
        dlg.finished.connect(lambda _result: self._advance_after_debrief(lm, result))

    def _advance_after_debrief(
        self, prev_lm: Landmark, result: LandmarkResult | None
    ) -> None:
        """Called when the debrief dialog closes. Advances to the next landmark."""
        if self._session.retry_mode:
            # Pop the front of the queue; a still-D landmark is re-queued.
            if self._session.advance_retry(result):
                self._finish_session()
            else:
                self._update_instruction_panel()
        else:
            if self._session.advance():
                self._finish_session()
                return
            next_lm = self._session.current_landmark()
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
        prev_cat = prev_lm.category if prev_lm is not None else None
        dlg = build_category_transition_dialog(
            prev_cat, next_lm, self._lang, self, self._center_dialog
        )
        if dlg is None:
            return False
        dlg.finished.connect(lambda _: self._update_instruction_panel())
        return True

    # ── Task C — D-grade retry ────────────────────────────────────────────────

    def _show_retry_intro(self, d_landmarks: list[Landmark]) -> None:
        """Show the retry introduction and let the student start or skip."""
        build_retry_dialog(
            d_landmarks,
            self._lang,
            self,
            self._center_dialog,
            on_start=lambda: self._begin_retry_session(d_landmarks),
            on_skip=self._emit_final_session,
        )

    def _begin_retry_session(self, d_landmarks: list[Landmark]) -> None:
        """Initialise the retry queue and reset UI state for the retry pass."""
        self._session.begin_retry(d_landmarks)
        self._candidate_point = None
        self._redo_count = 0
        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._error_label.setText("")
        self._update_instruction_panel()

    def _display_session_complete(self, score: SessionScore) -> None:
        """Update all result widgets and emit session_complete."""
        self._name_label.setText(
            tr("session_complete", self._lang, mean=score.mean_error_mm)
        )
        n_correct, n_inv = self._session.inverse_score()
        if self._session.mixed_session and n_inv:
            self._error_label.setText(
                f"Identification : {n_correct}/{n_inv} — "
                f"Placement : {score.global_score:.0f}/100 — {score.global_grade}"
            )
        else:
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

    def _emit_final_session(self) -> None:
        """Emit session_complete using the best score accumulated for every landmark."""
        self._display_session_complete(self._session.session_score())

    def _finish_session(self) -> None:
        """Called when the current pass (normal or retry) is exhausted."""
        if self._session.retry_mode:
            # Retry pass done — emit best results
            self._emit_final_session()
            return

        # Normal pass — check for grade-D landmarks to retry
        d_landmarks = self._session.build_retry_queue()

        if d_landmarks:
            self._show_retry_intro(d_landmarks)
            return

        # No D grades — emit the normal session results directly
        self._display_session_complete(self._session.normal_score())

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_instruction_panel(self) -> None:
        if self._session.retry_mode:
            if self._session.retry_finished():
                return
            lm = self._session.current_retry_landmark()
            n = len(self._session.retry_queue)
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
            if self._session.placement_finished():
                return
            lm = self._session.current_landmark()
            side = self._session.chosen_side
            side_label = "Côté gauche" if side == "left" else "Côté droit"
            if self._lang == "en":
                side_label = "Right side" if side == "right" else "Left side"
            self._progress_label.setText(
                f"{side_label} — "
                + tr(
                    "landmark_label",
                    self._lang,
                    index=self._session.current_placement_index + 1,
                    total=self._session.n_placement,
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
        # Hint and application are revealed only in the post-confirmation debrief
        self._hint_text.setText("")
        self._application_text.setVisible(False)

        self._instr_label.setText(tr("instructions", self._lang))
        self._error_label.setText("")

        self._plotter.render()

    def _toggle_lang(self) -> None:
        self._lang = "en" if self._lang == "fr" else "fr"
        self._session.lang = self._lang
        self._lang_btn.setText(tr("lang_toggle", self._lang))
        self._instr_label.setText(tr("instructions", self._lang))
        if self._session.retry_mode or not self._session.placement_finished():
            self._update_instruction_panel()

    # ── Mode inverse — identification anatomique ──────────────────────────────

    def _on_mode_toggled(self, checked: bool) -> None:
        # Prevent entering inverse mode during a retry session
        if self._session.retry_mode and checked:
            self._mode_btn.setChecked(False)
            return
        self._inverse_mode = checked
        label = "Mode : Identification" if checked else "Mode : Placement"
        self._mode_btn.setText(label)
        if checked:
            self._start_inverse_session()
        else:
            self._stop_inverse_session()

    def _start_inverse_session(self, queue: list[Landmark] | None = None) -> None:
        """Build the identification queue and enter inverse mode.

        If *queue* is provided (mixed-session auto-start) it is used directly.
        Otherwise the queue is built from all placement landmarks that have a
        ground truth (manual-toggle path).
        """
        self._inverse_mode = True
        self._session.start_inverse(queue)

        self._plotter.disable_picking()

        # Hide placement widgets; show identification panel
        self._confirm_btn.setVisible(False)
        self._redo_btn.setVisible(False)
        self._instr_label.setVisible(False)
        self._hint_text.setVisible(False)
        self._application_text.setVisible(False)
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
        self._hint_text.setVisible(True)
        self._inverse_panel.setVisible(False)

        if self._mode_btn.isChecked():
            self._mode_btn.setChecked(False)
        self._inverse_mode = False

    def _orbit_camera_to(self, point: np.ndarray) -> None:
        """Orbit the camera around the body axis to face *point*."""
        bounds = self._mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        extents = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
        up_axis = extents.index(max(extents))  # body's vertical axis (0=X, 1=Y, 2=Z)
        centers = [
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        ]
        # Focus is at body center horizontally, at landmark height along the vertical axis
        focus = np.array(centers, dtype=float)
        focus[up_axis] = point[up_axis]
        # Horizontal direction from body axis to landmark (zeroing the vertical component)
        horiz = np.array(point, dtype=float) - np.array(centers, dtype=float)
        horiz[up_axis] = 0.0
        horiz_norm = np.linalg.norm(horiz)
        if horiz_norm < 1.0:
            sorted_axes = sorted(range(3), key=lambda i: extents[i])
            direction = np.zeros(3)
            direction[sorted_axes[0]] = 1.0     # fallback: front along smallest (depth) axis
        else:
            direction = horiz / horiz_norm
        cam_distance = 1950.0  # mm from body axis
        cam_pos = focus + direction * cam_distance
        up_vec = [0.0, 0.0, 0.0]
        up_vec[up_axis] = 1.0
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(focus)
        cam.up = tuple(up_vec)
        self._plotter.render()

    def _show_inverse_landmark(self) -> None:
        """Display the ground-truth sphere for the current inverse landmark."""
        if self._session.inverse_finished():
            self._finish_inverse_session()
            return

        lm = self._session.current_inverse_landmark()
        gt = self._ground_truth[lm.code]

        # Remove previous sphere
        if self._inverse_shown_actor is not None:
            self._plotter.remove_actor(self._inverse_shown_actor, render=False)

        actor_name = f"inverse_shown_{lm.code}"
        sphere = pv.Sphere(radius=22, center=gt)
        self._plotter.add_mesh(sphere, color=_MARKER_COLOR, name=actor_name)
        self._inverse_shown_actor = actor_name

        # Orbit camera to face the landmark, keeping the avatar upright.
        self._orbit_camera_to(gt)

        # Update side panel (name hidden so as not to reveal the answer)
        n = len(self._session.inverse_queue)
        self._progress_label.setText(
            f"Identification — {self._session.inverse_index + 1} / {n}"
        )
        self._name_label.setText("?")
        self._hint_text.setText("")
        self._application_text.setText("")
        self._error_label.setText("")

        # Reset identification list — restore original category colours
        self._inverse_filter.clear()
        self._inverse_list.clearSelection()
        self._inverse_validate_btn.setEnabled(False)
        for i in range(self._inverse_list.count()):
            item = self._inverse_list.item(i)
            item.setHidden(False)
            item.setData(Qt.BackgroundRole, None)
            if item.flags() & Qt.ItemIsEnabled:
                cat = item.data(Qt.UserRole + 1) or ""
                item.setForeground(QColor(_INVERSE_CAT_COLORS.get(cat, "#cccccc")))

    def _on_inverse_filter_changed(self, text: str) -> None:
        """Show/hide list items according to the filter text (case-insensitive)."""
        lower = text.lower()
        for i in range(self._inverse_list.count()):
            item = self._inverse_list.item(i)
            item.setHidden(bool(lower) and lower not in item.text().lower())

    def _on_inverse_validate(self) -> None:
        """Check whether the selected landmark matches the displayed sphere."""
        selected = self._inverse_list.selectedItems()
        if not selected:
            return
        item = selected[0]
        selected_code = item.data(Qt.UserRole)
        lm = self._session.current_inverse_landmark()
        expected_code = lm.code

        if selected_code == expected_code:
            # Correct — record result, swap green sphere for blue, advance after 1 s
            self._session.record_inverse_correct(expected_code)
            if self._inverse_shown_actor is not None:
                self._plotter.remove_actor(self._inverse_shown_actor, render=False)
            gt = self._ground_truth[lm.code]
            blue_name = f"inverse_correct_{lm.code}"
            self._plotter.add_mesh(
                pv.Sphere(radius=12, center=gt), color=_CONFIRMED_COLOR, name=blue_name
            )
            self._inverse_shown_actor = blue_name
            self._plotter.render()

            # Visual feedback: green item, reveal landmark name
            item.setBackground(QColor("#1b5e20"))
            item.setForeground(QColor("#a5d6a7"))
            lm_name = lm.name(self._lang)
            if self._lang == "fr":
                self._error_label.setText(f"✓ Correct ! — {lm_name}")
            else:
                self._error_label.setText(f"✓ Correct! — {lm_name}")
            self._error_label.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #2e7d32;"
            )
            self._inverse_validate_btn.setEnabled(False)

            def _advance() -> None:
                self._session.inverse_advance()
                self._show_inverse_landmark()

            QTimer.singleShot(1000, _advance)
        else:
            # Incorrect — highlight selection red, correct item orange, advance after 2 s
            item.setBackground(QColor("#b71c1c"))
            item.setForeground(QColor("#ffffff"))

            # Highlight the correct item in orange (without selecting it)
            for i in range(self._inverse_list.count()):
                candidate = self._inverse_list.item(i)
                if candidate.data(Qt.UserRole) == expected_code:
                    candidate.setBackground(QColor("#e65100"))
                    candidate.setForeground(QColor("#ffffff"))
                    self._inverse_list.scrollToItem(candidate)
                    break

            lm_name = lm.name(self._lang)
            if self._lang == "fr":
                self._error_label.setText(
                    f"✗ Incorrect — La bonne réponse était : {lm_name}"
                )
            else:
                self._error_label.setText(
                    f"✗ Incorrect — The correct answer was: {lm_name}"
                )
            self._error_label.setStyleSheet(
                "font-size: 13px; font-weight: bold; color: #c62828;"
            )
            self._inverse_validate_btn.setEnabled(False)

            def _advance_after_wrong() -> None:
                self._session.inverse_advance()
                self._show_inverse_landmark()

            QTimer.singleShot(2000, _advance_after_wrong)

    def _finish_inverse_session(self) -> None:
        """Called when all inverse landmarks have been identified.

        In a mixed session: transitions directly to the placement phase.
        In standalone mode: shows a summary dialog then returns to placement.
        """
        if self._inverse_shown_actor is not None:
            self._plotter.remove_actor(self._inverse_shown_actor, render=False)
            self._inverse_shown_actor = None
        self._plotter.render()

        if self._session.mixed_session:
            self._stop_inverse_session()
            self._show_phase2_transition()
            return

        # Standalone mode — show summary dialog
        n_correct, n = self._session.inverse_score()
        dlg = QDialog(self)
        dlg.setWindowTitle("Session terminée")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(360)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 28, 28, 24)

        msg = QLabel(
            f"Session d'identification terminée !\n"
            f"Score : {n_correct} / {n}"
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

    def _show_phase2_transition(self) -> None:
        """Show a transition dialog from inverse (Phase 1) to placement (Phase 2)."""
        n_correct, n_inv = self._session.inverse_score()
        dlg = build_phase2_dialog(
            n_correct=n_correct,
            n_inv=n_inv,
            n_placement=self._session.n_placement,
            lang=self._lang,
            parent=self,
            center_fn=self._center_dialog,
        )
        dlg.finished.connect(lambda _: self._update_instruction_panel())
