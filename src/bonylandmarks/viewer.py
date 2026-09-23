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

import html

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
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

# ─── Guided-workshop widget styles (ISB / anthropometry panels) ──────────────

_FEEDBACK_OK_STYLE = (
    "font-size: 11px; color: #7fe08a; background: rgba(46,125,50,0.15); "
    "border-left: 3px solid #4caf50; padding: 5px 6px;"
)
_FEEDBACK_ERR_STYLE = (
    "font-size: 11px; color: #ff9a9a; background: rgba(139,26,26,0.18); "
    "border-left: 3px solid #e05555; padding: 5px 6px;"
)
_FEEDBACK_NEUTRAL_STYLE = (
    "font-size: 11px; color: #bbb; background: rgba(255,255,255,0.04); "
    "border-left: 3px solid rgba(255,255,255,0.18); padding: 5px 6px;"
)
_SECTION_LABEL_STYLE = "font-size: 10px; color: #9aa; font-weight: bold;"
_COMBO_STYLE = (
    "QComboBox { font-size: 12px; padding: 4px 6px; border-radius: 4px; "
    "background: rgba(255,255,255,0.07); color: #ddd; "
    "border: 1px solid rgba(255,255,255,0.12); }"
)
_PRIMARY_BTN_STYLE = (
    "QPushButton { background: #3399ff; color: white; border: none; "
    "border-radius: 6px; padding: 6px 10px; font-size: 12px; font-weight: bold; }"
    "QPushButton:hover { background: #4aa5ff; }"
    "QPushButton:disabled { background: rgba(255,255,255,0.08); color: #777; }"
)
_SECONDARY_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,0.07); color: #ccc; "
    "border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; "
    "padding: 4px 10px; font-size: 11px; }"
    "QPushButton:hover { background: rgba(255,255,255,0.13); }"
)

#: Human-readable name of each :mod:`isb_step_engine` step kind.
_ISB_OP_LABELS: dict[str, str] = {
    "pick_landmark": "Sélectionner",
    "midpoint": "Midpoint",
    "vector": "Vecteur",
    "cross_product": "Produit ×",
    "validate_frame": "Valider le repère",
}

# ─── Floating workshop overlay (bottom of the 3-D viewport) ──────────────────

_WS_OVERLAY_STYLE = """
    QWidget#workshop_overlay {
        background: rgba(14, 14, 30, 210);
        border-top: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
    }
    QLabel { background: transparent; }
    QScrollArea { background: transparent; border: none; }
    QScrollArea > QWidget > QWidget { background: transparent; }
    QScrollBar:horizontal {
        height: 6px; background: transparent; margin: 0px;
    }
    QScrollBar::handle:horizontal {
        background: rgba(255,255,255,0.22); border-radius: 3px; min-width: 24px;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px; background: transparent;
    }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: transparent;
    }
"""

#: Chip for a ground-truth landmark already available to the student.
_WS_CHIP_STYLE = (
    "QPushButton { background: rgba(255,255,255,0.08); color: #dce7f5; "
    "border: 1px solid rgba(255,255,255,0.16); border-radius: 13px; "
    "padding: 3px 12px; font-size: 11px; }"
    "QPushButton:hover { background: rgba(120,170,255,0.30); }"
    "QPushButton:checked { background: #3399ff; color: white; border-color: #3399ff; }"
)
#: Chip for an object the student built (midpoint, vector, cross product…).
_WS_CHIP_DERIVED_STYLE = (
    "QPushButton { background: rgba(80,200,120,0.16); color: #9ff0b5; "
    "border: 1px solid rgba(120,230,160,0.45); border-radius: 13px; "
    "padding: 3px 12px; font-size: 11px; }"
    "QPushButton:hover { background: rgba(120,230,160,0.34); }"
    "QPushButton:checked { background: #34c26a; color: #06230f; border-color: #34c26a; }"
)
#: Chip shown in the "Sél. :" row — clicking it removes the object.
_WS_SEL_CHIP_STYLE = (
    "QPushButton { background: rgba(51,153,255,0.28); color: #eaf3ff; "
    "border: 1px solid rgba(120,180,255,0.55); border-radius: 12px; "
    "padding: 3px 10px; font-size: 11px; font-weight: bold; }"
    "QPushButton:hover { background: rgba(224,85,85,0.55); border-color: #e05555; }"
)
_WS_EXEC_BTN_STYLE = (
    "QPushButton { background: #3399ff; color: white; border: none; "
    "border-radius: 6px; padding: 6px 16px; font-size: 12px; font-weight: bold; }"
    "QPushButton:hover { background: #4aa5ff; }"
    "QPushButton:disabled { background: rgba(255,255,255,0.08); color: #777; }"
)
_WS_STEP_LABEL_STYLE = (
    "font-size: 11px; font-weight: bold; color: #7ab8ff; background: transparent;"
)
_WS_INSTR_LABEL_STYLE = (
    "font-size: 13px; font-weight: bold; color: #f0f0f0; background: transparent;"
)
_WS_SECTION_STYLE = "font-size: 10px; color: #8a97a8; background: transparent;"
_WS_FEEDBACK_OK_STYLE = "font-size: 11px; color: #4caf50; background: transparent;"
_WS_FEEDBACK_ERR_STYLE = "font-size: 11px; color: #f44336; background: transparent;"
_WS_FEEDBACK_NEUTRAL_STYLE = "font-size: 11px; color: #9aa; background: transparent;"

#: Label of every operation button in the floating workshop overlay.
_WS_OP_LABELS: dict[str, str] = {
    "pick_landmark": "Sélectionner",
    "midpoint": "Milieu",
    "vector": "Vecteur A→B",
    "cross_product": "A × B",
    "distance": "Distance",
    "angle": "Angle",
}

#: Operation buttons offered by each workshop mode, in display order.
_WS_MODE_OPS: dict[str, tuple[str, ...]] = {
    "isb": ("pick_landmark", "midpoint", "vector", "cross_product"),
    "anthro": ("pick_landmark", "midpoint", "distance", "angle"),
}

#: :mod:`anthro_step_engine` step kind → operation button key.
#: Kinds absent from this map (asymmetry, projection) skip the coherence check.
_ANTHRO_STEP_OPS: dict[str, str] = {
    "pick_landmark": "pick_landmark",
    "midpoint": "midpoint",
    "distance": "distance",
    "angle_3pts": "angle",
    "angle_plane": "angle",
    "axis_angle": "angle",
}

_WS_OP_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,0.06); color: #ccc; "
    "border: 1px solid rgba(255,255,255,0.12); border-radius: 5px; "
    "padding: 5px 10px; font-size: 11px; }"
    "QPushButton:checked { background: #3399ff; color: white; border-color: #3399ff; }"
    "QPushButton:hover { background: rgba(255,255,255,0.14); }"
)

#: Tooltip of each operation button, by key.
_WS_OP_TIPS: dict[str, str] = {
    "pick_landmark": "Choisir un landmark du ground truth",
    "midpoint": "Milieu de 2 points",
    "vector": "Vecteur orienté entre 2 points (origine puis extrémité)",
    "cross_product": "Produit vectoriel de 2 vecteurs (A × B)",
    "distance": "Distance euclidienne entre 2 points",
    "angle": "Angle (3 points, plan de référence ou entre axes)",
}


def _ws_op_label(key: str) -> str:
    """Readable name of an operation key, for mismatch feedback."""
    return _WS_OP_LABELS.get(key, _ISB_OP_LABELS.get(key, key))

#: 3-D colour of an ISB axis vector, by workspace name.
_ISB_AXIS_COLORS: dict[str, str] = {"X": "red", "Y": "green", "Z": "blue"}
_ISB_POINT_COLOR = "white"
_ISB_POINT_RADIUS = 8.0
#: Arrow length as a fraction of the body's largest extent (unit-agnostic).
_ISB_ARROW_SCALE = 0.08
_ANTHRO_LINE_COLOR = "#ffb347"

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

    result_ready = Signal(object, object, object, object)  # (mask, blurred_colors, gray_colors, blurred_points)
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
            from .face_blur import build_face_mask, blur_vertex_colors, blur_mesh_geometry, gray_face_colors
            mask = build_face_mask(self._mesh_points, self._ground_truth,
                                   acromion_offset=float(self._acromion_offset))
            blurred_colors = blur_vertex_colors(self._base_colors.copy(), mask, self._mesh_points)
            gray_colors = gray_face_colors(self._base_colors.copy(), mask)
            blurred_points = blur_mesh_geometry(self._mesh_points.copy(), mask, self._mesh_faces)
            self.result_ready.emit(mask, blurred_colors, gray_colors, blurred_points)
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
        dev_mode: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tutorial_mode = tutorial_mode
        self._dev_mode = dev_mode
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

        # ISB and Anthropo exercise modes
        self._isb_mode: bool = False
        self._anthro_mode: bool = False
        self._placement_done: bool = False
        self._isb_actors: list = []
        self._anthro_actors: list = []
        self._isb_result = None
        self._isb_defs: dict = {}
        self._anthro_results = None

        # Guided workshops (interactive step engines)
        self._isb_engine = None              # isb_step_engine.StepEngine
        self._isb_segment_key: str = ""
        self._anthro_engine = None           # anthro_step_engine.AnthroStepEngine
        self._anthro_recipe = None           # anthro_recipes.AnthroRecipe

        # Floating workshop overlay state ("" | "isb" | "anthro")
        self._ws_mode: str = ""
        self._ws_current_selection: list[str] = []
        self._ws_chip_buttons: dict[str, QPushButton] = {}

        self._candidate_point: np.ndarray | None = None
        self._redo_count: int = 0

        # Face-blur state
        self._face_blurred: bool = False
        self._face_gray: bool = False
        self._colors_blurred: np.ndarray | None = None  # cached blur result
        self._colors_gray: np.ndarray | None = None      # cached gray result
        self._points_blurred: np.ndarray | None = None   # lissage géométrique mis en cache
        self._points_original: np.ndarray | None = None  # positions originales (sauvegardées)
        self._blur_worker: _BlurWorker | None = None      # thread actif
        self._blur_zone_offset: int = 35                  # mm above acromions (slider)

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
        self._progress_label.setStyleSheet("font-size: 11px; color: #666;")
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
        self._name_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #999;")
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

        # Separator (between hint/application widgets and the instruction label;
        # assigned to self so it can be hidden during ISB / anthro modes)
        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.HLine)
        self._sep.setFrameShadow(QFrame.Sunken)
        panel.addWidget(self._sep)

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
        self._sep2 = QFrame()
        self._sep2.setFrameShape(QFrame.HLine)
        self._sep2.setFrameShadow(QFrame.Sunken)
        panel.addWidget(self._sep2)

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

        # ISB exercise panel (hidden by default)
        self._isb_panel = QWidget()
        isb_layout = QVBoxLayout(self._isb_panel)
        isb_layout.setContentsMargins(0, 0, 0, 0)
        isb_layout.setSpacing(6)

        isb_title = QLabel("Repères locaux ISB")
        isb_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #3399ff;")
        isb_layout.addWidget(isb_title)

        isb_title_sep = QFrame()
        isb_title_sep.setFrameShape(QFrame.HLine)
        isb_title_sep.setStyleSheet(
            "border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 2px 0px;"
        )
        isb_layout.addWidget(isb_title_sep)

        # Segment chooser + start button (always visible in ISB mode)
        self._isb_segment_combo = QComboBox()
        self._isb_segment_combo.setStyleSheet(_COMBO_STYLE)
        self._isb_segment_combo.currentIndexChanged.connect(
            lambda _: self._on_isb_segment_changed()
        )
        isb_layout.addWidget(self._isb_segment_combo)

        self._isb_meta_label = QLabel()
        self._isb_meta_label.setWordWrap(True)
        self._isb_meta_label.setStyleSheet(
            "font-size: 10px; color: #9aa; background: transparent;"
        )
        isb_layout.addWidget(self._isb_meta_label)

        self._isb_start_btn = QPushButton("Démarrer")
        self._isb_start_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        self._isb_start_btn.clicked.connect(self._on_isb_start)
        isb_layout.addWidget(self._isb_start_btn)

        # ── Result area — visible only while a segment is being built ─────────
        # The step-by-step mechanics live in the floating workshop overlay
        # (:meth:`_build_workshop_overlay`); this side panel only reports the
        # final FrameReport and offers a restart.
        self._isb_work = QWidget()
        isb_work_layout = QVBoxLayout(self._isb_work)
        isb_work_layout.setContentsMargins(0, 4, 0, 0)
        isb_work_layout.setSpacing(4)

        _isb_res_label = QLabel("Résultats :")
        _isb_res_label.setStyleSheet(_SECTION_LABEL_STYLE)
        isb_work_layout.addWidget(_isb_res_label)

        self._isb_result_label = QLabel()
        self._isb_result_label.setWordWrap(True)
        self._isb_result_label.setStyleSheet(_FEEDBACK_NEUTRAL_STYLE)
        self._isb_result_label.setMinimumHeight(60)
        self._isb_result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        isb_work_layout.addWidget(self._isb_result_label)

        self._isb_reset_btn = QPushButton("Recommencer")
        self._isb_reset_btn.setStyleSheet(_SECONDARY_BTN_STYLE)
        self._isb_reset_btn.clicked.connect(self._on_isb_reset)
        isb_work_layout.addWidget(self._isb_reset_btn)

        self._isb_work.setVisible(False)
        isb_layout.addWidget(self._isb_work)

        self._isb_panel.setVisible(False)
        panel.addWidget(self._isb_panel)

        # Anthropo exercise panel (hidden by default)
        self._anthro_panel = QWidget()
        anthro_layout = QVBoxLayout(self._anthro_panel)
        anthro_layout.setContentsMargins(0, 0, 0, 0)
        anthro_layout.setSpacing(6)

        anthro_title = QLabel("Mesures anthropométriques")
        anthro_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #cc6600;")
        anthro_layout.addWidget(anthro_title)

        anthro_title_sep = QFrame()
        anthro_title_sep.setFrameShape(QFrame.HLine)
        anthro_title_sep.setStyleSheet(
            "border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 2px 0px;"
        )
        anthro_layout.addWidget(anthro_title_sep)

        self._anthro_measure_combo = QComboBox()
        self._anthro_measure_combo.setStyleSheet(_COMBO_STYLE)
        self._anthro_measure_combo.currentIndexChanged.connect(
            lambda _: self._on_anthro_measure_changed()
        )
        anthro_layout.addWidget(self._anthro_measure_combo)

        self._anthro_meta_label = QLabel()
        self._anthro_meta_label.setWordWrap(True)
        self._anthro_meta_label.setStyleSheet(
            "font-size: 10px; color: #9aa; background: transparent;"
        )
        anthro_layout.addWidget(self._anthro_meta_label)

        self._anthro_start_btn = QPushButton("Démarrer")
        self._anthro_start_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        self._anthro_start_btn.clicked.connect(self._on_anthro_start)
        anthro_layout.addWidget(self._anthro_start_btn)

        # ── Result area — visible only while a measure is in progress ────────
        # Step mechanics live in the floating workshop overlay; the side panel
        # only shows the final value and its comparison to the norm.
        self._anthro_work = QWidget()
        anthro_work_layout = QVBoxLayout(self._anthro_work)
        anthro_work_layout.setContentsMargins(0, 4, 0, 0)
        anthro_work_layout.setSpacing(4)

        _anthro_res_label = QLabel("Résultat :")
        _anthro_res_label.setStyleSheet(_SECTION_LABEL_STYLE)
        anthro_work_layout.addWidget(_anthro_res_label)

        self._anthro_result_label = QLabel()
        self._anthro_result_label.setWordWrap(True)
        self._anthro_result_label.setAlignment(Qt.AlignCenter)
        self._anthro_result_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #ffc078;"
        )
        anthro_work_layout.addWidget(self._anthro_result_label)

        self._anthro_norm_label = QLabel()
        self._anthro_norm_label.setWordWrap(True)
        self._anthro_norm_label.setStyleSheet(_FEEDBACK_NEUTRAL_STYLE)
        self._anthro_norm_label.setMinimumHeight(60)
        self._anthro_norm_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        anthro_work_layout.addWidget(self._anthro_norm_label)

        self._anthro_reset_btn = QPushButton("Recommencer")
        self._anthro_reset_btn.setStyleSheet(_SECONDARY_BTN_STYLE)
        self._anthro_reset_btn.clicked.connect(self._on_anthro_reset)
        anthro_work_layout.addWidget(self._anthro_reset_btn)

        self._anthro_work.setVisible(False)
        anthro_layout.addWidget(self._anthro_work)

        self._anthro_panel.setVisible(False)
        panel.addWidget(self._anthro_panel)

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

        # Texture / sticker / blur controls — created here, displayed in _build_blur_overlay()
        self._texture_btn = QPushButton("Mode : Texturé")
        self._texture_btn.setCheckable(True)
        self._texture_btn.setChecked(True)
        self._texture_btn.clicked.connect(self._on_texture_toggled)

        self._sticker_btn = QPushButton("Stickers : masqués")
        self._sticker_btn.setCheckable(True)
        self._sticker_btn.setChecked(True)
        self._sticker_btn.clicked.connect(self._on_sticker_toggled)

        self._blur_face_btn = QPushButton("Lissage visage : non")
        self._blur_face_btn.setCheckable(True)
        self._blur_face_btn.setChecked(False)
        self._blur_face_btn.clicked.connect(self._on_blur_face_toggled)

        self._gray_face_btn = QPushButton("Mode gris visage : non")
        self._gray_face_btn.setCheckable(True)
        self._gray_face_btn.setChecked(False)
        self._gray_face_btn.clicked.connect(self._on_gray_face_toggled)

        _zone_row = QHBoxLayout()
        _zone_lbl = QLabel("Zone :")
        _zone_lbl.setFixedWidth(40)
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
        self._blur_zone_widget = _zone_widget

        # Biomechanics measurements section (collapsible)
        sep_biom = QFrame()
        sep_biom.setFrameShape(QFrame.HLine)
        sep_biom.setFrameShadow(QFrame.Sunken)
        panel.addWidget(sep_biom)

        self._biom_toggle_btn = QPushButton("Mesures biomécaniques ▸")
        self._biom_toggle_btn.setCheckable(True)
        self._biom_toggle_btn.setChecked(False)
        self._biom_toggle_btn.setStyleSheet("text-align: left; padding: 4px 8px;")
        self._biom_toggle_btn.clicked.connect(self._on_biom_toggled)
        panel.addWidget(self._biom_toggle_btn)

        self._biom_scroll = QScrollArea()
        self._biom_scroll.setWidgetResizable(True)
        self._biom_scroll.setMaximumHeight(340)
        self._biom_scroll.setVisible(False)
        self._biom_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel.addWidget(self._biom_scroll)
        self._biom_initialized: bool = False

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(320)
        root.addWidget(panel_widget)

        # Floating overlay for landmark navigation (left edge of viewport)
        self._build_landmark_nav_overlay()

        # Floating overlay for texture/blur controls (bottom-left of viewport)
        self._build_blur_overlay()

        # HUD — titre exercice + nom du repère, centré en haut du viewport
        self._build_hud_overlay()

        # Atelier guidé (ISB / anthropo) — overlay flottant ancré en bas
        self._build_workshop_overlay()

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

        # Ensure overlays stay on top after the plotter renders
        if hasattr(self, "_nav_overlay"):
            self._nav_overlay.raise_()
        if hasattr(self, "_blur_overlay"):
            self._blur_overlay.raise_()
        if hasattr(self, "_lm_nav_overlay"):
            self._lm_nav_overlay.raise_()
        if hasattr(self, "_hud_overlay"):
            self._hud_overlay.raise_()
        if hasattr(self, "_workshop_overlay"):
            self._workshop_overlay.raise_()

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

    def _build_blur_overlay(self) -> None:
        """Create a floating panel for texture/blur controls, anchored bottom-left."""
        container = self._plotter.interactor
        overlay = QWidget(container)
        overlay.setObjectName("blur_overlay")
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setStyleSheet("""
            QWidget#blur_overlay {
                background: rgba(18, 18, 28, 170);
                border-radius: 6px;
            }
            QLabel {
                color: #aaa;
                font-size: 10px;
            }
            QPushButton {
                background-color: rgba(45, 45, 65, 210);
                color: #ccc;
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 4px;
                font-size: 10px;
                padding: 3px 7px;
                min-height: 22px;
            }
            QPushButton:checked {
                background-color: rgba(25, 90, 170, 220);
                color: #fff;
                border-color: rgba(100,170,255,0.55);
            }
            QPushButton:hover {
                background-color: rgba(60, 60, 100, 230);
            }
            QPushButton:disabled {
                color: #666;
            }
        """)
        layout = QVBoxLayout(overlay)
        layout.setSpacing(3)
        layout.setContentsMargins(8, 6, 8, 6)

        has_colors = self._vertex_colors is not None
        has_both = (
            self._vertex_colors_clean is not None
            and self._vertex_colors_raw is not None
        )

        self._texture_btn.setVisible(has_colors)
        self._sticker_btn.setVisible(has_both)
        self._blur_face_btn.setVisible(has_colors)
        self._gray_face_btn.setVisible(has_colors)
        self._blur_zone_widget.setVisible(has_colors)

        layout.addWidget(self._texture_btn)
        layout.addWidget(self._sticker_btn)
        layout.addWidget(self._blur_face_btn)
        layout.addWidget(self._gray_face_btn)
        layout.addWidget(self._blur_zone_widget)

        overlay.adjustSize()
        self._blur_overlay = overlay
        self._position_blur_overlay()

    def _position_blur_overlay(self) -> None:
        """Move the blur overlay to the bottom-left corner of the 3D viewport."""
        if not hasattr(self, "_blur_overlay"):
            return
        container = self._plotter.interactor
        margin = 10
        w = self._blur_overlay.width() or self._blur_overlay.sizeHint().width()
        h = self._blur_overlay.height() or self._blur_overlay.sizeHint().height()
        x = margin
        y = container.height() - h - margin
        # Keep the texture/blur controls reachable: the guided-workshop bar is
        # anchored to the same bottom edge and spans the full viewport width.
        workshop = getattr(self, "_workshop_overlay", None)
        if workshop is not None and workshop.isVisible():
            y -= workshop.height() + 6
        self._blur_overlay.move(x, max(margin, y))
        self._blur_overlay.raise_()

    def _build_hud_overlay(self) -> None:
        """HUD semi-transparent centré en haut du viewport : titre exercice + nom repère."""
        container = self._plotter.interactor
        overlay = QWidget(container)
        overlay.setObjectName("hud_overlay")
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setStyleSheet("""
            QWidget#hud_overlay {
                background: rgba(18, 18, 38, 175);
                border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.10);
            }
            QLabel#hud_title {
                color: rgba(160, 190, 255, 200);
                font-size: 11px;
            }
            QLabel#hud_lm {
                color: #f0f0f0;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout = QVBoxLayout(overlay)
        layout.setSpacing(2)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setAlignment(Qt.AlignCenter)

        self._hud_title_label = QLabel("")
        self._hud_title_label.setObjectName("hud_title")
        self._hud_title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._hud_title_label)

        self._hud_lm_label = QLabel("")
        self._hud_lm_label.setObjectName("hud_lm")
        self._hud_lm_label.setAlignment(Qt.AlignCenter)
        self._hud_lm_label.setWordWrap(True)
        layout.addWidget(self._hud_lm_label)

        overlay.adjustSize()
        self._hud_overlay = overlay
        self._hud_overlay.setVisible(False)
        self._position_hud_overlay()

    def _position_hud_overlay(self) -> None:
        """Centre le HUD en haut du viewport, 14 px sous le bord."""
        if not hasattr(self, "_hud_overlay"):
            return
        container = self._plotter.interactor
        overlay = self._hud_overlay
        overlay.adjustSize()
        w = overlay.sizeHint().width()
        x = max(0, (container.width() - w) // 2)
        overlay.move(x, 14)
        overlay.raise_()

    def _update_hud(self, title: str = "", lm_name: str = "") -> None:
        """Met à jour le HUD et l'affiche/cache selon le contenu."""
        if not hasattr(self, "_hud_overlay"):
            return
        self._hud_title_label.setText(title.upper())
        self._hud_lm_label.setText(lm_name)
        self._hud_lm_label.setVisible(bool(lm_name))
        visible = bool(title or lm_name)
        self._hud_overlay.setVisible(visible)
        if visible:
            self._position_hud_overlay()

    # ── Floating workshop overlay (guided ISB / anthropometry ateliers) ───────

    def _build_workshop_overlay(self) -> None:
        """Interactive atelier bar, anchored to the bottom of the 3-D viewport.

        Four rows: progress + instruction, workspace chips, selection +
        operations + execute, feedback.  Hidden until an engine is started.
        """
        container = self._plotter.interactor
        overlay = QWidget(container)
        overlay.setObjectName("workshop_overlay")
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setStyleSheet(_WS_OVERLAY_STYLE)

        root = QVBoxLayout(overlay)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(6)

        # ── Row 1 — progress + instruction ───────────────────────────────────
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(10)

        self._ws_step_label = QLabel("")
        self._ws_step_label.setStyleSheet(_WS_STEP_LABEL_STYLE)
        self._ws_step_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        row1.addWidget(self._ws_step_label, 0)

        self._ws_row1_sep = QLabel("·")
        self._ws_row1_sep.setStyleSheet(_WS_SECTION_STYLE)
        row1.addWidget(self._ws_row1_sep, 0)

        self._ws_instr_label = QLabel("")
        self._ws_instr_label.setWordWrap(True)
        self._ws_instr_label.setStyleSheet(_WS_INSTR_LABEL_STYLE)
        self._ws_instr_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        row1.addWidget(self._ws_instr_label, 1)
        root.addLayout(row1)

        # ── Row 2 — workspace chips (horizontal scroll) ──────────────────────
        self._ws_chips_scroll = QScrollArea()
        self._ws_chips_scroll.setWidgetResizable(True)
        self._ws_chips_scroll.setFixedHeight(44)
        self._ws_chips_scroll.setFrameShape(QFrame.NoFrame)
        self._ws_chips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._ws_chips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._ws_chips_widget = QWidget()
        self._ws_chips_layout = QHBoxLayout(self._ws_chips_widget)
        self._ws_chips_layout.setContentsMargins(0, 2, 0, 2)
        self._ws_chips_layout.setSpacing(6)
        self._ws_chips_layout.addStretch()      # keeps chips left-aligned
        self._ws_chips_scroll.setWidget(self._ws_chips_widget)
        root.addWidget(self._ws_chips_scroll)

        # ── Row 3 — selection + operations + execute ─────────────────────────
        row3 = QHBoxLayout()
        row3.setContentsMargins(0, 0, 0, 0)
        row3.setSpacing(6)

        self._ws_sel_label = QLabel("Sél. :")
        self._ws_sel_label.setStyleSheet(_WS_SECTION_STYLE)
        row3.addWidget(self._ws_sel_label, 0)

        self._ws_sel_btn_a = QPushButton("")
        self._ws_sel_btn_a.setStyleSheet(_WS_SEL_CHIP_STYLE)
        self._ws_sel_btn_a.setCursor(Qt.PointingHandCursor)
        self._ws_sel_btn_a.clicked.connect(lambda: self._ws_remove_selection(0))
        self._ws_sel_btn_a.setVisible(False)
        row3.addWidget(self._ws_sel_btn_a, 0)

        self._ws_sel_btn_b = QPushButton("")
        self._ws_sel_btn_b.setStyleSheet(_WS_SEL_CHIP_STYLE)
        self._ws_sel_btn_b.setCursor(Qt.PointingHandCursor)
        self._ws_sel_btn_b.clicked.connect(lambda: self._ws_remove_selection(1))
        self._ws_sel_btn_b.setVisible(False)
        row3.addWidget(self._ws_sel_btn_b, 0)

        row3.addStretch(1)

        self._ws_op_group = QButtonGroup(self)
        self._ws_op_group.setExclusive(True)
        self._ws_op_buttons: dict[str, QPushButton] = {}
        for op_key in ("pick_landmark", "midpoint", "vector",
                       "cross_product", "distance", "angle"):
            btn = QPushButton(_WS_OP_LABELS[op_key])
            btn.setCheckable(True)
            btn.setToolTip(_WS_OP_TIPS.get(op_key, ""))
            btn.setStyleSheet(_WS_OP_BTN_STYLE)
            btn.setCursor(Qt.PointingHandCursor)
            self._ws_op_group.addButton(btn)
            self._ws_op_buttons[op_key] = btn
            row3.addWidget(btn, 0)

        # Named handles required by the spec
        self._ws_op_select_btn = self._ws_op_buttons["pick_landmark"]
        self._ws_op_midpoint_btn = self._ws_op_buttons["midpoint"]
        self._ws_op_vector_btn = self._ws_op_buttons["vector"]
        self._ws_op_cross_btn = self._ws_op_buttons["cross_product"]
        self._ws_op_distance_btn = self._ws_op_buttons["distance"]
        self._ws_op_angle_btn = self._ws_op_buttons["angle"]

        self._ws_exec_btn = QPushButton("▶ Exécuter")
        self._ws_exec_btn.setStyleSheet(_WS_EXEC_BTN_STYLE)
        self._ws_exec_btn.setCursor(Qt.PointingHandCursor)
        self._ws_exec_btn.clicked.connect(self._on_ws_execute)
        row3.addWidget(self._ws_exec_btn, 0)
        root.addLayout(row3)

        # ── Row 4 — feedback ─────────────────────────────────────────────────
        self._ws_feedback_label = QLabel("")
        self._ws_feedback_label.setWordWrap(True)
        self._ws_feedback_label.setTextFormat(Qt.RichText)
        self._ws_feedback_label.setStyleSheet(_WS_FEEDBACK_NEUTRAL_STYLE)
        self._ws_feedback_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        root.addWidget(self._ws_feedback_label)

        overlay.adjustSize()
        self._workshop_overlay = overlay
        self._workshop_overlay.setVisible(False)
        self._position_workshop_overlay()

    def _position_workshop_overlay(self) -> None:
        """Anchor the atelier bar to the bottom edge, full viewport width."""
        if not hasattr(self, "_workshop_overlay"):
            return
        container = self._plotter.interactor
        overlay = self._workshop_overlay
        overlay.setFixedWidth(max(320, container.width() - 20))
        overlay.adjustSize()
        h = overlay.height() or overlay.sizeHint().height()
        x = 10
        y = max(0, container.height() - h - 10)
        overlay.move(x, y)
        overlay.raise_()
        # The blur controls share this bottom edge — push them above the bar.
        self._position_blur_overlay()

    def _show_workshop_overlay(self, mode: str) -> None:
        """Reveal the atelier bar for *mode* (``"isb"`` or ``"anthro"``)."""
        if not hasattr(self, "_workshop_overlay"):
            return
        self._ws_mode = mode
        self._workshop_overlay.setVisible(True)
        self._position_workshop_overlay()

    def _hide_workshop_overlay(self) -> None:
        """Hide the atelier bar and drop every transient selection."""
        if not hasattr(self, "_workshop_overlay"):
            return
        self._ws_mode = ""
        self._ws_clear_selection()
        self._ws_reset_op_buttons()
        self._ws_set_feedback(None, "")
        self._workshop_overlay.setVisible(False)
        self._position_blur_overlay()

    # ── Workshop overlay — operations ─────────────────────────────────────────

    def _ws_reset_op_buttons(self) -> None:
        """Uncheck every operation button (the group is exclusive)."""
        if not hasattr(self, "_ws_op_group"):
            return
        self._ws_op_group.setExclusive(False)
        for btn in self._ws_op_buttons.values():
            btn.setChecked(False)
        self._ws_op_group.setExclusive(True)

    def _ws_current_op(self) -> str:
        """Key of the operation the student declared, or ``""``."""
        for op_key, btn in self._ws_op_buttons.items():
            if btn.isVisible() and btn.isChecked():
                return op_key
        return ""

    def _ws_apply_mode_ops(self, mode: str) -> None:
        """Show only the operation buttons that make sense in *mode*."""
        allowed = _WS_MODE_OPS.get(mode, ())
        for op_key, btn in self._ws_op_buttons.items():
            btn.setVisible(op_key in allowed)

    # ── Workshop overlay — selection ──────────────────────────────────────────

    def _ws_on_chip_clicked(self, name: str) -> None:
        """Add the object *name* to the current selection (2 items maximum)."""
        sel = self._ws_current_selection
        if name in sel:
            sel.remove(name)
        elif len(sel) < 2:
            sel.append(name)
        else:
            self._ws_set_feedback(
                False,
                "Deux objets au maximum : retire une sélection (clic sur le chip ✕) "
                "avant d'en ajouter une autre.",
            )
        self._refresh_ws_selection_display()

    def _ws_remove_selection(self, index: int) -> None:
        """Drop the selection slot *index* (0 = A, 1 = B)."""
        if 0 <= index < len(self._ws_current_selection):
            del self._ws_current_selection[index]
        self._refresh_ws_selection_display()

    def _ws_clear_selection(self) -> None:
        self._ws_current_selection.clear()
        self._refresh_ws_selection_display()

    def _refresh_ws_selection_display(self) -> None:
        """Sync the two selection chips and the checked state of the workspace chips."""
        if not hasattr(self, "_ws_sel_btn_a"):
            return
        sel = self._ws_current_selection
        for index, btn in ((0, self._ws_sel_btn_a), (1, self._ws_sel_btn_b)):
            if index < len(sel):
                btn.setText(f"{sel[index]}  ✕")
                btn.setToolTip(f"Retirer « {sel[index]} » de la sélection")
                btn.setVisible(True)
            else:
                btn.setText("")
                btn.setToolTip("")
                btn.setVisible(False)
        for name, chip in self._ws_chip_buttons.items():
            chip.setChecked(name in sel)
        self._ws_sel_label.setText("Sél. :" if sel else "Sél. :  —")

    # ── Workshop overlay — workspace chips ────────────────────────────────────

    def _ws_workspace_entries(self) -> list[tuple[str, str, str, bool]]:
        """``(name, chip_text, tooltip, is_derived)`` for the current mode."""
        if self._ws_mode == "isb":
            return self._ws_isb_entries()
        if self._ws_mode == "anthro":
            return self._ws_anthro_entries()
        return []

    def _ws_isb_entries(self) -> list[tuple[str, str, str, bool]]:
        """Landmarks required by the recipe, then every object already built."""
        from .isb_recipes import required_landmarks

        engine = self._isb_engine
        if engine is None:
            return []
        try:
            codes = required_landmarks(self._isb_segment_key)
        except Exception:                                # pragma: no cover
            codes = []

        ws = engine.workspace
        gt_codes = set(self._ground_truth)
        out: list[tuple[str, str, str, bool]] = []
        shown: set[str] = set()

        for code in codes:
            if code in ws.points and code not in shown:
                shown.add(code)
                out.append((code, code, self._lm_label(code), False))

        for name in ws.points:
            if name in shown or name in gt_codes:
                continue
            shown.add(name)
            out.append((name, f"{name} ✓", f"Point construit : {name}", True))

        for name in ws.vectors:
            if name in shown:
                continue
            shown.add(name)
            out.append((name, f"→ {name}", f"Vecteur construit : {name}", True))

        return out

    def _ws_anthro_entries(self) -> list[tuple[str, str, str, bool]]:
        """Every ground-truth landmark, then the points already placed."""
        engine = self._anthro_engine
        if engine is None:
            return []

        out: list[tuple[str, str, str, bool]] = []
        shown: set[str] = set()
        for code in sorted(
            (c for c, v in self._ground_truth.items() if v is not None),
            key=lambda c: c.lower(),
        ):
            shown.add(code)
            out.append((code, code, self._lm_label(code), False))

        for name, value in engine.workspace.items():
            if name in shown or not isinstance(value, np.ndarray):
                continue
            shown.add(name)
            out.append((name, f"{name} ✓", f"Point construit : {name}", True))

        return out

    def _ws_rebuild_chips(self) -> None:
        """Rebuild the workspace chip row from scratch."""
        if not hasattr(self, "_ws_chips_layout"):
            return
        layout = self._ws_chips_layout
        while layout.count() > 1:                 # keep the trailing stretch
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._ws_chip_buttons = {}

        for name, text, tooltip, derived in self._ws_workspace_entries():
            chip = QPushButton(text)
            chip.setCheckable(True)
            chip.setToolTip(tooltip)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(
                _WS_CHIP_DERIVED_STYLE if derived else _WS_CHIP_STYLE
            )
            chip.clicked.connect(
                lambda _checked=False, n=name: self._ws_on_chip_clicked(n)
            )
            layout.insertWidget(layout.count() - 1, chip)
            self._ws_chip_buttons[name] = chip

        self._refresh_ws_selection_display()

    # ── Workshop overlay — step refresh ───────────────────────────────────────

    def _update_workshop_overlay(self, engine, mode: str) -> None:
        """Refresh every row of the atelier bar for the current step of *engine*."""
        if not hasattr(self, "_workshop_overlay") or engine is None:
            return

        self._ws_mode = mode
        self._ws_apply_mode_ops(mode)
        self._ws_clear_selection()
        self._ws_reset_op_buttons()
        self._ws_set_feedback(None, "")

        if mode == "isb":
            from .isb_recipes import recipe_label
            from .isb_step_engine import ValidateFrame

            index, total = engine.progress()
            hud_title = f"Exercice ISB — {recipe_label(self._isb_segment_key, self._lang)}"
        else:
            recipe = self._anthro_recipe
            total = len(recipe.steps) if recipe is not None else 0
            index = engine.step_index
            ValidateFrame = ()                   # no validate step in anthro mode
            hud_title = "Mesures anthropométriques"

        if engine.finished:
            self._ws_step_label.setText(f"Étape {total} / {total}")
            self._ws_instr_label.setText(
                "Construction terminée — consulte les résultats dans le panneau latéral."
                if mode == "isb"
                else "Mesure terminée — consulte le résultat dans le panneau latéral."
            )
            self._ws_exec_btn.setEnabled(False)
            self._update_hud(hud_title, "Terminé")
        else:
            step = engine.current_step
            self._ws_step_label.setText(f"Étape {index + 1} / {total}")
            self._ws_instr_label.setText(getattr(step, "instruction_fr", ""))
            self._ws_exec_btn.setEnabled(True)
            is_validate = bool(ValidateFrame) and isinstance(step, ValidateFrame)
            self._ws_exec_btn.setText(
                "▶ Valider le repère" if is_validate else "▶ Exécuter"
            )
            self._update_hud(hud_title, f"Étape {index + 1}/{total}")

        self._ws_rebuild_chips()
        self._position_workshop_overlay()

    def _ws_set_feedback(self, ok, msg: str) -> None:
        """Colour-code row 4 (``True`` green / ``False`` red / ``None`` neutral)."""
        if not hasattr(self, "_ws_feedback_label"):
            return
        if ok is True:
            prefix, style = "✓ ", _WS_FEEDBACK_OK_STYLE
        elif ok is False:
            prefix, style = "✗ ", _WS_FEEDBACK_ERR_STYLE
        else:
            prefix, style = "", _WS_FEEDBACK_NEUTRAL_STYLE
        self._ws_feedback_label.setStyleSheet(style)
        self._ws_feedback_label.setText((prefix + self._rich(msg)) if msg else "")

    # ── Workshop overlay — execution ──────────────────────────────────────────

    def _on_ws_execute(self) -> None:
        """Run the current step of whichever engine owns the atelier bar."""
        if self._ws_mode == "isb":
            self._ws_execute_isb()
        elif self._ws_mode == "anthro":
            self._ws_execute_anthro()

    def _ws_execute_isb(self) -> None:
        """Run the current ISB step with the student's selection."""
        from .isb_step_engine import ValidateFrame

        engine = self._isb_engine
        if engine is None or engine.finished:
            return

        step = engine.current_step
        step_kind = getattr(step, "kind", "")
        op = self._ws_current_op()
        if op and op != step_kind:
            self._ws_set_feedback(
                False,
                f"Cette étape attend l'opération « {_ws_op_label(step_kind)} », "
                f"pas « {_ws_op_label(op)} ». Relis la consigne.",
            )
            return

        ok, msg = engine.try_execute(list(self._ws_current_selection))

        if isinstance(step, ValidateFrame):
            # Always draw the final triad — even a wrong frame is worth seeing.
            self._isb_draw_frame(step)
            self._isb_set_result(ok, msg)
            self._update_workshop_overlay(engine, "isb")
        elif ok:
            self._isb_draw_step_result(step)
            self._update_workshop_overlay(engine, "isb")

        self._ws_set_feedback(ok, msg)

    def _ws_execute_anthro(self) -> None:
        """Run the current anthropometry step with the student's selection."""
        engine = self._anthro_engine
        if engine is None or engine.finished:
            return

        step = engine.current_step
        step_kind = getattr(step, "kind", "")
        op = self._ws_current_op()
        expected_op = _ANTHRO_STEP_OPS.get(step_kind, "")
        if op and expected_op and op != expected_op:
            self._ws_set_feedback(
                False,
                f"Cette étape attend l'opération « {_ws_op_label(expected_op)} », "
                f"pas « {_ws_op_label(op)} ». Relis la consigne.",
            )
            return

        ok, msg = engine.try_execute(list(self._ws_current_selection))

        if ok:
            self._anthro_draw_step(step)
            value = engine.workspace.get(getattr(step, "result_name", ""))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._anthro_result_label.setText(
                    f"{step.result_name} = {float(value):.1f}"
                )
            self._update_workshop_overlay(engine, "anthro")

        self._ws_set_feedback(ok, msg)

        if engine.finished:
            self._anthro_show_final()

    def _build_landmark_nav_overlay(self) -> None:
        """Floating ◀ ▶ navigator — left edge of the 3D viewport, always visible."""
        container = self._plotter.interactor
        overlay = QWidget(container)
        overlay.setObjectName("lm_nav_overlay")
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setStyleSheet("""
            QWidget#lm_nav_overlay { background: transparent; }
            QPushButton {
                background-color: rgba(26, 26, 46, 180);
                color: #e0e0e0;
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 6px;
                font-size: 20px;
                font-weight: bold;
                min-width: 44px;
                max-width: 44px;
                min-height: 44px;
                max-height: 44px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(60, 80, 140, 210);
                border-color: rgba(100,160,255,0.6);
            }
            QPushButton:pressed {
                background-color: rgba(30, 60, 120, 230);
            }
            QPushButton:disabled {
                color: #555;
                background-color: rgba(26, 26, 46, 80);
            }
        """)
        layout = QVBoxLayout(overlay)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        self._lm_prev_btn = QPushButton("◀")
        self._lm_prev_btn.setToolTip("Exercice précédent")
        self._lm_prev_btn.clicked.connect(self._on_prev_exercise)
        self._lm_next_btn = QPushButton("▶")
        self._lm_next_btn.setToolTip("Exercice suivant")
        self._lm_next_btn.clicked.connect(self._on_next_exercise)

        layout.addWidget(self._lm_prev_btn)
        layout.addWidget(self._lm_next_btn)

        overlay.adjustSize()
        self._lm_nav_overlay = overlay
        self._position_landmark_nav_overlay()

    def _position_landmark_nav_overlay(self) -> None:
        """Centre the nav overlay on the left edge of the 3D viewport."""
        if not hasattr(self, "_lm_nav_overlay"):
            return
        container = self._plotter.interactor
        margin = 10
        w = self._lm_nav_overlay.width() or self._lm_nav_overlay.sizeHint().width()
        h = self._lm_nav_overlay.height() or self._lm_nav_overlay.sizeHint().height()
        x = margin
        y = (container.height() - h) // 2
        self._lm_nav_overlay.move(x, y)
        self._lm_nav_overlay.raise_()

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
        self._position_blur_overlay()
        self._position_landmark_nav_overlay()
        self._position_hud_overlay()
        self._position_workshop_overlay()

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
        self._colors_blurred = None
        self._colors_gray = None
        self._points_blurred = None
        if self._face_blurred:
            self._blur_face_btn.setChecked(False)
            self._face_blurred = False
            self._blur_face_btn.setText("Lissage visage : non")
        if self._face_gray:
            self._gray_face_btn.setChecked(False)
            self._face_gray = False
            self._gray_face_btn.setText("Mode gris visage : non")
        self._texture_btn.setText("Mode : Texturé" if checked else "Mode : Gris")
        self._add_body_mesh(textured=checked)
        self._plotter.render()

    def _on_sticker_toggled(self, checked: bool) -> None:
        self._colors_blurred = None
        self._colors_gray = None
        self._points_blurred = None
        if self._face_blurred:
            self._blur_face_btn.setChecked(False)
            self._face_blurred = False
            self._blur_face_btn.setText("Lissage visage : non")
        if self._face_gray:
            self._gray_face_btn.setChecked(False)
            self._face_gray = False
            self._gray_face_btn.setText("Mode gris visage : non")
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
        """Apply or remove face blurring (mutual exclusion with gray mode)."""
        if checked:
            if self._face_gray:
                self._remove_gray()
                self._gray_face_btn.setChecked(False)
            if self._colors_blurred is not None and self._points_blurred is not None:
                self._apply_blur()
            else:
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

        self._blur_face_btn.setText("Lissage : calcul...")
        self._blur_face_btn.setEnabled(False)
        self._gray_face_btn.setEnabled(False)

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

    def _on_blur_computed(self, mask, blurred_colors, gray_colors, blurred_points) -> None:
        """Appelé dans le thread principal quand le calcul est terminé."""
        self._colors_blurred = blurred_colors
        self._colors_gray = gray_colors
        self._points_blurred = blurred_points
        self._blur_face_btn.setEnabled(True)
        self._gray_face_btn.setEnabled(True)
        if self._blur_face_btn.isChecked():
            self._apply_blur()
        elif self._gray_face_btn.isChecked():
            self._apply_gray()
        else:
            self._blur_face_btn.setText("Lissage visage : non")

    def _on_blur_zone_label(self, value: int) -> None:
        """Met à jour le label pendant le drag sans relancer le calcul."""
        self._blur_zone_offset = value
        self._blur_zone_val_lbl.setText(f"{value} mm")

    def _on_blur_zone_released(self) -> None:
        """Relance le calcul au relâchement du slider."""
        self._colors_blurred = None
        self._colors_gray = None
        self._points_blurred = None
        was_blur = self._face_blurred
        was_gray = self._face_gray
        if was_blur:
            self._remove_blur()
            self._blur_face_btn.setChecked(True)
            self._start_blur_computation()
        elif was_gray:
            self._remove_gray()
            self._gray_face_btn.setChecked(True)
            self._start_gray_computation()

    def _on_blur_error(self, msg: str) -> None:
        """Appelé si le calcul échoue."""
        print(f"[face_blur] erreur : {msg}")
        self._blur_face_btn.setChecked(False)
        self._blur_face_btn.setText("Lissage visage : non")
        self._blur_face_btn.setEnabled(True)
        self._gray_face_btn.setChecked(False)
        self._gray_face_btn.setText("Mode gris visage : non")
        self._gray_face_btn.setEnabled(True)

    def _apply_blur(self) -> None:
        """Applique les couleurs floutées + géométrie lissée (cache disponible)."""
        self._vertex_colors = self._colors_blurred
        if self._points_blurred is not None:
            self._mesh.points = self._points_blurred
            self._mesh.compute_normals(inplace=True)
        self._face_blurred = True
        self._blur_face_btn.setText("Lissage visage : oui")
        self._blur_face_btn.setEnabled(True)
        self._refresh_mesh_colors()

    def _remove_blur(self) -> None:
        """Restaure les couleurs et la géométrie originales."""
        if self._points_original is not None:
            self._mesh.points = self._points_original
            self._mesh.compute_normals(inplace=True)
        if self._sticker_btn.isChecked():
            self._vertex_colors = self._vertex_colors_clean
        else:
            self._vertex_colors = self._vertex_colors_raw
        self._face_blurred = False
        self._blur_face_btn.setText("Lissage visage : non")
        self._refresh_mesh_colors()

    def _on_gray_face_toggled(self, checked: bool) -> None:
        """Apply or remove gray face mode (mutual exclusion with blur mode)."""
        if checked:
            if self._face_blurred:
                self._remove_blur()
                self._blur_face_btn.setChecked(False)
            if self._colors_gray is not None and self._points_blurred is not None:
                self._apply_gray()
            else:
                self._start_gray_computation()
        else:
            self._remove_gray()

    def _start_gray_computation(self) -> None:
        """Lance le calcul (blur + gray) dans un QThread de fond pour le mode gris."""
        if self._blur_worker is not None and self._blur_worker.isRunning():
            return

        base = self._vertex_colors_clean if self._vertex_colors_clean is not None \
               else self._vertex_colors
        if base is None:
            self._gray_face_btn.setChecked(False)
            return

        if self._points_original is None:
            self._points_original = np.asarray(self._mesh.points, dtype=np.float64).copy()

        self._gray_face_btn.setText("Mode gris : calcul...")
        self._gray_face_btn.setEnabled(False)
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

    def _apply_gray(self) -> None:
        """Applique la texture grise uniforme + géométrie lissée."""
        self._vertex_colors = self._colors_gray
        if self._points_blurred is not None:
            self._mesh.points = self._points_blurred
            self._mesh.compute_normals(inplace=True)
        self._face_gray = True
        self._gray_face_btn.setText("Mode gris visage : oui")
        self._gray_face_btn.setEnabled(True)
        self._refresh_mesh_colors()

    def _remove_gray(self) -> None:
        """Restaure les couleurs et la géométrie originales (mode gris → normal)."""
        if self._points_original is not None:
            self._mesh.points = self._points_original
            self._mesh.compute_normals(inplace=True)
        if self._sticker_btn.isChecked():
            self._vertex_colors = self._vertex_colors_clean
        else:
            self._vertex_colors = self._vertex_colors_raw
        self._face_gray = False
        self._gray_face_btn.setText("Mode gris visage : non")
        self._refresh_mesh_colors()

    # ── Biomechanics panel ────────────────────────────────────────────────────

    def _on_biom_toggled(self, checked: bool) -> None:
        self._biom_toggle_btn.setText(
            "Mesures biomécaniques ▾" if checked else "Mesures biomécaniques ▸"
        )
        if checked and not self._biom_initialized:
            self._init_biom_widget()
        self._biom_scroll.setVisible(checked)

    def _init_biom_widget(self) -> None:
        """Build and populate the biomechanics measurements table (lazy, once)."""
        from .biomechanics import compute_all, Measurement

        measures = compute_all(self._ground_truth)
        self._biom_initialized = True

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        _STATUS_COLORS = {
            "normal":    ("#1a7a40", "#d4edda"),  # text, bg
            "attention": ("#7a5c00", "#fff3cd"),
            "alerte":    ("#8b1a1a", "#f8d7da"),
            "info":      ("#1a4a7a", "#d1ecf1"),
        }

        current_category = None
        for m in measures:
            if m.category != current_category:
                current_category = m.category
                cat_lbl = QLabel(m.category.upper())
                cat_lbl.setStyleSheet(
                    "font-size: 10px; font-weight: bold; color: #666;"
                    "padding: 6px 4px 2px 4px; letter-spacing: 1px;"
                )
                layout.addWidget(cat_lbl)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(6)

            # Label + side
            name_text = m.label
            if m.side:
                name_text += f" ({m.side})"
            name_lbl = QLabel(name_text)
            name_lbl.setWordWrap(True)
            name_lbl.setStyleSheet("font-size: 11px;")
            name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            # Value chip
            txt_color, bg_color = _STATUS_COLORS.get(m.status, _STATUS_COLORS["info"])
            val_lbl = QLabel(m.value_str)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet(
                f"font-size: 11px; font-weight: bold;"
                f"color: {txt_color}; background: {bg_color};"
                f"border-radius: 3px; padding: 1px 5px;"
            )
            val_lbl.setToolTip(m.note)
            val_lbl.setFixedWidth(72)

            row_layout.addWidget(name_lbl)
            row_layout.addWidget(val_lbl)
            layout.addWidget(row)

        if not measures:
            layout.addWidget(QLabel("Repères insuffisants pour calculer les mesures."))

        layout.addStretch()
        self._biom_scroll.setWidget(container)

    # ── Confirm / Redo ────────────────────────────────────────────────────────

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

    def _on_prev_exercise(self) -> None:
        """Jump to the previous exercise phase."""
        self._plotter.remove_actor("candidate_sphere", render=False)
        self._candidate_point = None
        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._error_label.setText("")

        if self._anthro_mode:
            # Anthropo → back to ISB
            self._stop_anthro_session()
            self._start_isb_session()
            return

        if self._isb_mode:
            # ISB → back to session-complete view
            self._stop_isb_session()
            return

        if self._inverse_mode:
            return  # already at the first exercise

        if self._session.retry_mode:
            # Exit retry → restart placement from the beginning
            self._session.exit_retry_to_placement()
            self._update_instruction_panel()
        elif self._session.mixed_session:
            # Placement phase → go back to inverse identification
            self._session.restart_inverse()
            self._start_inverse_session(queue=self._session.inverse_landmarks)
        # else: standalone placement only — nothing before it

    def _on_next_exercise(self) -> None:
        """Jump to the next exercise phase, skipping all remaining items."""
        self._plotter.remove_actor("candidate_sphere", render=False)
        self._candidate_point = None
        self._confirm_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        self._error_label.setText("")

        if self._anthro_mode:
            return  # already at the last exercise

        elif self._isb_mode:
            # ISB → Anthropo
            self._stop_isb_session()
            self._start_anthro_session()

        elif self._placement_done:
            # Session complete → ISB exercise
            self._start_isb_session()

        elif self._inverse_mode:
            # Skip remaining inverse → go to placement phase
            if self._inverse_shown_actor is not None:
                self._plotter.remove_actor(self._inverse_shown_actor, render=False)
                self._inverse_shown_actor = None
            self._plotter.render()
            if self._session.mixed_session:
                self._stop_inverse_session()
                self._show_phase2_transition()
            else:
                self._stop_inverse_session()
                self._update_instruction_panel()

        elif self._session.retry_mode:
            # Skip remaining retry → finish
            self._emit_final_session()

        else:
            # Skip remaining placement → retry or finish
            self._finish_session()

    # ── Task A — Debrief overlay ──────────────────────────────────────────────

    def _show_landmark_debrief(
        self, lm: Landmark, result: LandmarkResult | None
    ) -> None:
        """Show a non-blocking modal overlay with hint + grade after validation.

        Advancement to the next landmark happens when the dialog closes.
        """
        from .biomechanics import measures_for_landmark
        biom = measures_for_landmark(self._ground_truth, lm.code)
        dlg = build_debrief_dialog(lm, result, self._lang, self, self._center_dialog,
                                   biom_measures=biom or None)
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
        self._placement_done = True
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
        self._update_hud("Session terminée")
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
        # Establish the placement/retry layout regardless of how we got here.
        # This makes the method safe to call after leaving inverse mode or
        # any other context that may have hidden some of these widgets.
        self._confirm_btn.setVisible(True)
        self._redo_btn.setVisible(True)
        self._instr_label.setVisible(True)
        self._hint_text.setVisible(False)   # hint is revealed only inside the debrief dialog
        self._inverse_panel.setVisible(False)
        self._isb_panel.setVisible(False)
        self._anthro_panel.setVisible(False)
        self._hide_workshop_overlay()
        self._sep.setVisible(True)
        self._error_label.setVisible(True)
        self._mode_btn.setVisible(not self._session.mixed_session)
        self._sep2.setVisible(True)
        self._results_table.setVisible(True)

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

        # HUD 3-D
        hud_title = self._progress_label.text()
        self._update_hud(hud_title, lm.name(self._lang))

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
        # Hide results table and its separator — irrelevant during identification
        self._sep2.setVisible(False)
        self._results_table.setVisible(False)
        # Hide theme badge — it is meaningless while the landmark name is concealed
        self._theme_badge.setVisible(False)
        self._inverse_panel.setVisible(True)
        self._hide_workshop_overlay()

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
        # _hint_text visibility is left to _update_instruction_panel
        self._sep2.setVisible(True)
        self._results_table.setVisible(True)
        self._inverse_panel.setVisible(False)

        if self._mode_btn.isChecked():
            self._mode_btn.setChecked(False)
        self._inverse_mode = False

        # Refresh the instruction panel so all placement widgets (including
        # theme_badge) are restored to their correct state regardless of the
        # path that led here (toggle off, standalone finish, or mixed finish).
        self._update_instruction_panel()

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
        inv_progress = f"Identification — {self._session.inverse_index + 1} / {n}"
        self._progress_label.setText(inv_progress)
        self._name_label.setText("?")
        self._update_hud(inv_progress, "?")
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
            self._show_inverse_debrief(lm)
        else:
            # Incorrect — highlight selection red, correct item orange, then debrief
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
            self._show_inverse_debrief(lm)

    def _show_inverse_debrief(self, lm: "Landmark") -> None:
        """Show the hint/biom debrief after an inverse identification answer.

        Advancing the inverse queue happens when the dialog closes.
        """
        from .biomechanics import measures_for_landmark
        biom = measures_for_landmark(self._ground_truth, lm.code)
        dlg = build_debrief_dialog(
            lm, None, self._lang, self, self._center_dialog,
            biom_measures=biom or None,
        )

        def _advance() -> None:
            self._session.inverse_advance()
            self._show_inverse_landmark()

        dlg.finished.connect(lambda _: _advance())

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

    # ── Shared helpers for the guided workshops ───────────────────────────────

    def _lm_label(self, code: str) -> str:
        """Readable landmark label ``Nom (code)``, or the raw code."""
        from .landmarks_extended import LANDMARK_BY_CODE
        lm = LANDMARK_BY_CODE.get(code)
        return f"{lm.name(self._lang)}  ({code})" if lm is not None else code

    def _model_scale(self) -> float:
        """Largest extent of the body mesh, used to size 3-D annotations."""
        try:
            b = self._mesh.bounds
            return float(max(b[1] - b[0], b[3] - b[2], b[5] - b[4]))
        except Exception:                                # pragma: no cover
            return 1000.0

    def _add_label_actor(self, pos, text: str, color: str = "white"):
        """Add a floating 3-D text label; returns the actor or ``None``."""
        try:
            return self._plotter.add_point_labels(
                np.asarray([pos], dtype=float),
                [text],
                font_size=12,
                text_color=color,
                shape=None,
                always_visible=True,
                render=False,
            )
        except Exception:                                # pragma: no cover
            return None

    @staticmethod
    def _rich(msg: str) -> str:
        """Escape *msg* and turn newlines into ``<br>`` for a QLabel."""
        return html.escape(msg).replace("\n", "<br>")

    # ── ISB guided workshop ───────────────────────────────────────────────────

    def _start_isb_session(self) -> None:
        """Enter ISB workshop mode: pick a segment, then build its frame step by step."""
        from .isb_exercise import get_all_segment_definitions
        from .isb_recipes import ISB_RECIPE_META, RECIPE_KEYS, required_landmarks

        self._isb_mode = True

        # Cache segment definitions (key → dict with the method description)
        try:
            self._isb_defs = {d["key"]: d for d in get_all_segment_definitions()}
        except Exception:                                # pragma: no cover
            self._isb_defs = {}

        # Hide placement / inverse widgets
        self._confirm_btn.setVisible(False)
        self._redo_btn.setVisible(False)
        self._instr_label.setVisible(False)
        self._hint_text.setVisible(False)
        self._application_text.setVisible(False)
        self._sep.setVisible(False)
        self._error_label.setVisible(False)
        self._mode_btn.setVisible(False)
        self._sep2.setVisible(False)
        self._results_table.setVisible(False)
        self._theme_badge.setVisible(False)
        self._inverse_panel.setVisible(False)
        self._anthro_panel.setVisible(False)
        self._isb_panel.setVisible(True)

        self._progress_label.setText("Exercice ISB — Repères locaux")
        self._name_label.setText("Sélectionnez un segment")
        self._update_hud("Exercice ISB — Repères locaux", "Sélectionnez un segment")

        # Populate the segment chooser: constructible segments first
        combo = self._isb_segment_combo
        combo.blockSignals(True)
        combo.clear()
        first_available = -1
        for key in RECIPE_KEYS:
            missing = [
                code for code in required_landmarks(key)
                if self._ground_truth.get(code) is None
            ]
            label = ISB_RECIPE_META.get(key, {}).get("name_fr", key)
            combo.addItem(f"{'✓' if not missing else '✗'}  {label}", key)
            if not missing and first_available < 0:
                first_available = combo.count() - 1
        combo.blockSignals(False)

        if combo.count():
            combo.setCurrentIndex(max(0, first_available))
        self._on_isb_segment_changed()

    def _stop_isb_session(self) -> None:
        """Leave ISB workshop mode: clean up actors and hide the panel."""
        self._isb_teardown_workshop()
        self._isb_mode = False
        self._isb_panel.setVisible(False)
        self._isb_result = None
        self._hide_workshop_overlay()
        # Restore widgets that were hidden when entering ISB mode
        self._sep.setVisible(True)
        self._error_label.setVisible(True)
        self._mode_btn.setVisible(not self._session.mixed_session)

    def _isb_teardown_workshop(self) -> None:
        """Drop the running engine, clear the 3-D overlay and the workshop widgets."""
        self._isb_engine = None
        self._isb_segment_key = ""
        self._isb_work.setVisible(False)
        self._isb_set_result(None, "")
        self._hide_workshop_overlay()
        self._clear_isb_actors()

    def _on_isb_segment_changed(self) -> None:
        """Describe the selected segment and enable/disable the start button."""
        if not self._isb_mode:
            return
        self._isb_teardown_workshop()

        from .isb_recipes import ISB_RECIPE_META, get_recipe, required_landmarks

        key = self._isb_segment_combo.currentData()
        if not key:
            self._isb_meta_label.setText("")
            self._isb_start_btn.setEnabled(False)
            return

        meta = ISB_RECIPE_META.get(key, {})
        label = meta.get("name_fr", key)
        needed = required_landmarks(key)
        missing = [c for c in needed if self._ground_truth.get(c) is None]

        if missing:
            self._isb_meta_label.setText(
                f"<b>{label}</b> — <span style='color:#e05555;'>non constructible</span><br>"
                f"Landmarks manquants : {', '.join(missing)}"
            )
            self._isb_start_btn.setEnabled(False)
        else:
            method = self._isb_defs.get(key, {}).get("method_fr", "")
            text = (
                f"<b>{label}</b> — {meta.get('reference', '')} · "
                f"{len(get_recipe(key))} étapes<br>"
                f"Landmarks : {', '.join(needed)}"
            )
            if method:
                text += f"<br><i>{method}</i>"
            self._isb_meta_label.setText(text)
            self._isb_start_btn.setEnabled(True)

        self._update_hud("Exercice ISB — Repères locaux", label)

    def _on_isb_start(self) -> None:
        """Create a :class:`StepEngine` for the selected segment and open the workshop."""
        from .isb_recipes import make_engine, recipe_label

        key = self._isb_segment_combo.currentData()
        if not key:
            return

        self._clear_isb_actors()
        try:
            engine = make_engine(key, self._ground_truth)
        except Exception as exc:                          # pragma: no cover
            self._isb_work.setVisible(True)
            self._isb_set_result(False, f"Impossible de démarrer ce segment : {exc}")
            return

        self._isb_engine = engine
        self._isb_segment_key = key
        self._isb_work.setVisible(True)
        self._isb_set_result(None, "")
        self._isb_draw_existing()
        self._show_workshop_overlay("isb")
        self._update_workshop_overlay(engine, "isb")
        self._ws_set_feedback(
            None,
            "Clique les objets dans la barre ci-dessus, choisis l'opération, "
            "puis « ▶ Exécuter ».",
        )
        self._name_label.setText(recipe_label(key, self._lang))

    def _on_isb_reset(self) -> None:
        """Restart the current segment from step 1."""
        engine = self._isb_engine
        if engine is None:
            self._hide_workshop_overlay()
            return
        engine.reset()
        self._clear_isb_actors()
        self._isb_set_result(None, "")
        self._isb_draw_existing()
        self._show_workshop_overlay("isb")
        self._update_workshop_overlay(engine, "isb")
        self._ws_set_feedback(None, "Construction réinitialisée — retour à l'étape 1.")

    # ── ISB — 3-D visualisation ───────────────────────────────────────────────

    def _isb_arrow_length(self) -> float:
        """Arrow length: a fraction of the body's largest extent (unit-agnostic)."""
        return _ISB_ARROW_SCALE * self._model_scale()

    def _isb_frame_origin(self):
        """Best anchor point for vector arrows (the recipe's frame origin)."""
        engine = self._isb_engine
        if engine is None:
            return None
        from .isb_step_engine import ValidateFrame
        ws = engine.workspace
        for st in engine.steps:
            if isinstance(st, ValidateFrame) and st.origin_name:
                pt = ws.points.get(st.origin_name)
                if pt is not None:
                    return pt
        gt_codes = set(self._ground_truth)
        for name, pt in ws.points.items():
            if name not in gt_codes:
                return pt
        return None

    def _isb_draw_point(self, name: str, pos) -> None:
        """White sphere + label for a landmark or a constructed midpoint."""
        center = tuple(float(v) for v in np.asarray(pos, dtype=float).reshape(3))
        sphere = pv.Sphere(radius=_ISB_POINT_RADIUS, center=center)
        self._isb_actors.append(
            self._plotter.add_mesh(sphere, color=_ISB_POINT_COLOR, render=False)
        )
        label = self._add_label_actor(center, name)
        if label is not None:
            self._isb_actors.append(label)

    def _isb_draw_vector(self, name: str, direction, start) -> None:
        """Arrow for a normalised vector (red=X, green=Y, blue=Z, white otherwise)."""
        vec = np.asarray(direction, dtype=float).reshape(3)
        if float(np.linalg.norm(vec)) < 1e-9:
            return
        color = _ISB_AXIS_COLORS.get(name, "white")
        arrow = pv.Arrow(
            start=tuple(float(v) for v in np.asarray(start, dtype=float).reshape(3)),
            direction=tuple(float(v) for v in vec),
            scale=self._isb_arrow_length(),
        )
        self._isb_actors.append(
            self._plotter.add_mesh(arrow, color=color, render=False)
        )

    def _isb_draw_step_result(self, step) -> None:
        """Draw the object produced by *step* (point or vector)."""
        from .isb_step_engine import (
            ComputeCrossProduct,
            ComputeMidpoint,
            ComputeVector,
            PickLandmark,
        )
        engine = self._isb_engine
        if engine is None:
            return
        ws = engine.workspace
        name = getattr(step, "result_name", "")

        if isinstance(step, (PickLandmark, ComputeMidpoint)):
            pos = ws.points.get(name)
            if pos is not None:
                self._isb_draw_point(name, pos)
        elif isinstance(step, ComputeVector):
            vec = ws.vectors.get(name)
            start = ws.points.get(step.expected_from)
            if vec is not None and start is not None:
                self._isb_draw_vector(name, vec, start)
        elif isinstance(step, ComputeCrossProduct):
            vec = ws.vectors.get(name)
            start = self._isb_frame_origin()
            if vec is not None and start is not None:
                self._isb_draw_vector(name, vec, start)

        self._plotter.render()

    def _isb_draw_existing(self) -> None:
        """Draw every derived object already in the workspace (auto steps, reset)."""
        engine = self._isb_engine
        if engine is None:
            return
        ws = engine.workspace
        gt_codes = set(self._ground_truth)
        for name, pos in ws.points.items():
            if name not in gt_codes:
                self._isb_draw_point(name, pos)
        origin = self._isb_frame_origin()
        if origin is not None:
            for name, vec in ws.vectors.items():
                self._isb_draw_vector(name, vec, origin)
        self._plotter.render()

    def _isb_draw_frame(self, step) -> None:
        """Show only the final triad X/Y/Z at the frame origin."""
        engine = self._isb_engine
        if engine is None:
            return
        self._clear_isb_actors()
        ws = engine.workspace
        origin = ws.points.get(step.origin_name) if step.origin_name else None
        if origin is None:
            origin = self._isb_frame_origin()
        if origin is None:
            return
        self._isb_draw_point(step.origin_name or "O", origin)
        for axis, ws_name in (("X", step.x_name), ("Y", step.y_name), ("Z", step.z_name)):
            vec = ws.vectors.get(ws_name) if ws_name else None
            if vec is not None:
                self._isb_draw_vector(axis, vec, origin)
        self._plotter.render()

    def _isb_set_result(self, ok, msg: str) -> None:
        """Write the final :class:`FrameReport` into the side-panel result area."""
        self._isb_result_label.setTextFormat(Qt.RichText)
        if ok is True:
            prefix, style = "✓ ", _FEEDBACK_OK_STYLE
        elif ok is False:
            prefix, style = "✗ ", _FEEDBACK_ERR_STYLE
        else:
            prefix, style = "", _FEEDBACK_NEUTRAL_STYLE
        self._isb_result_label.setStyleSheet(style)
        self._isb_result_label.setText((prefix + self._rich(msg)) if msg else "")

    def _clear_isb_actors(self) -> None:
        """Remove every ISB overlay actor from the 3-D view."""
        for actor in self._isb_actors:
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._isb_actors = []
        self._plotter.render()

    # ── Anthropometry guided workshop ─────────────────────────────────────────

    def _start_anthro_session(self) -> None:
        """Enter anthropometry workshop mode: pick a measure, then build it step by step."""
        from .anthro_measures_exercise import compute_available
        from .anthro_recipes import ANTHRO_RECIPES

        self._anthro_mode = True

        # Hide placement / ISB widgets
        self._confirm_btn.setVisible(False)
        self._redo_btn.setVisible(False)
        self._instr_label.setVisible(False)
        self._hint_text.setVisible(False)
        self._application_text.setVisible(False)
        self._sep.setVisible(False)
        self._error_label.setVisible(False)
        self._mode_btn.setVisible(False)
        self._sep2.setVisible(False)
        self._results_table.setVisible(False)
        self._theme_badge.setVisible(False)
        self._inverse_panel.setVisible(False)
        self._isb_panel.setVisible(False)
        self._anthro_panel.setVisible(True)

        self._progress_label.setText("Exercice — Mesures anthropométriques")
        self._name_label.setText("Sélectionnez une mesure")
        self._update_hud("Mesures anthropométriques", "Sélectionnez une mesure")
        self._error_label.setText("")

        # Reference values pre-computed by the non-interactive module
        try:
            self._anthro_results = compute_available(self._ground_truth)
        except Exception:                                # pragma: no cover
            self._anthro_results = None

        combo = self._anthro_measure_combo
        combo.blockSignals(True)
        combo.clear()
        for code, recipe in ANTHRO_RECIPES.items():
            combo.addItem(f"{recipe.name_fr}  ({recipe.unit})", code)
        combo.blockSignals(False)
        if combo.count():
            combo.setCurrentIndex(0)
        self._on_anthro_measure_changed()

    def _stop_anthro_session(self) -> None:
        """Leave anthropometry workshop mode."""
        self._anthro_teardown_workshop()
        self._anthro_mode = False
        self._anthro_panel.setVisible(False)
        self._anthro_results = None
        self._hide_workshop_overlay()
        # Restore widgets that were hidden when entering anthro mode
        self._sep.setVisible(True)
        self._error_label.setVisible(True)
        self._mode_btn.setVisible(not self._session.mixed_session)

    def _anthro_teardown_workshop(self) -> None:
        """Drop the running engine, clear the 3-D overlay and the workshop widgets."""
        self._anthro_engine = None
        self._anthro_recipe = None
        self._anthro_work.setVisible(False)
        self._anthro_result_label.setText("")
        self._anthro_set_norm(None, "")
        self._hide_workshop_overlay()
        self._clear_anthro_actors()

    @staticmethod
    def _anthro_required_codes(recipe) -> list[str]:
        """Landmark codes a recipe asks the student to pick, without duplicates."""
        from .anthro_step_engine import PickLandmark as APickLandmark
        out: list[str] = []
        for st in recipe.steps:
            if isinstance(st, APickLandmark) and st.expected_code not in out:
                out.append(st.expected_code)
        return out

    def _anthro_measure_result(self, recipe):
        """The :class:`AnthroMeasureResult` matching *recipe*, or ``None``."""
        if not self._anthro_results or not recipe.measure_code:
            return None
        for res in self._anthro_results:
            if res.measure.code == recipe.measure_code:
                return res
        return None

    def _on_anthro_measure_changed(self) -> None:
        """Describe the selected measure and enable/disable the start button."""
        if not self._anthro_mode:
            return
        self._anthro_teardown_workshop()

        from .anthro_recipes import ANTHRO_RECIPES

        code = self._anthro_measure_combo.currentData()
        recipe = ANTHRO_RECIPES.get(code) if code else None
        if recipe is None:
            self._anthro_meta_label.setText("")
            self._anthro_start_btn.setEnabled(False)
            return

        needed = self._anthro_required_codes(recipe)
        missing = [c for c in needed if self._ground_truth.get(c) is None]

        if missing:
            self._anthro_meta_label.setText(
                f"<b>{recipe.name_fr}</b> — "
                f"<span style='color:#e05555;'>non calculable</span><br>"
                f"Landmarks manquants : {', '.join(missing)}"
            )
            self._anthro_start_btn.setEnabled(False)
        else:
            res = self._anthro_measure_result(recipe)
            norm = ""
            if res is not None:
                m = res.measure
                if m.norm_low is not None and m.norm_high is not None:
                    norm = (
                        f" · norme {m.value_str(m.norm_low)} – "
                        f"{m.value_str(m.norm_high)}"
                    )
            self._anthro_meta_label.setText(
                f"<b>{recipe.name_fr}</b> — {len(recipe.steps)} étapes{norm}<br>"
                f"Landmarks : {', '.join(needed)}"
            )
            self._anthro_start_btn.setEnabled(True)

        self._update_hud("Mesures anthropométriques", recipe.name_fr)

    def _on_anthro_start(self) -> None:
        """Create an :class:`AnthroStepEngine` for the selected measure."""
        from .anthro_recipes import ANTHRO_RECIPES
        from .anthro_step_engine import AnthroStepEngine

        code = self._anthro_measure_combo.currentData()
        recipe = ANTHRO_RECIPES.get(code) if code else None
        if recipe is None:
            return

        self._clear_anthro_actors()
        self._anthro_recipe = recipe
        self._anthro_engine = AnthroStepEngine(recipe, self._ground_truth)
        self._anthro_work.setVisible(True)
        self._anthro_result_label.setText("")
        self._anthro_set_norm(None, "")
        self._show_workshop_overlay("anthro")
        self._update_workshop_overlay(self._anthro_engine, "anthro")
        self._ws_set_feedback(
            None,
            "Clique un landmark dans la barre ci-dessus, choisis l'opération, "
            "puis « ▶ Exécuter ».",
        )
        self._name_label.setText(recipe.name_fr)

    def _on_anthro_reset(self) -> None:
        """Restart the current measure from step 1 (the engine has no in-place reset)."""
        if self._anthro_recipe is None:
            self._hide_workshop_overlay()
            return
        self._on_anthro_start()
        self._ws_set_feedback(None, "Mesure réinitialisée — retour à l'étape 1.")

    def _anthro_show_final(self) -> None:
        """Display the final value, the norm verdict and the reference comparison."""
        engine = self._anthro_engine
        recipe = self._anthro_recipe
        if engine is None or recipe is None:
            return

        value = engine.final_result()
        if value is None:
            self._anthro_result_label.setText("Résultat indisponible")
            self._anthro_set_norm(
                False, "Le résultat final n'a pas pu être extrait du workspace."
            )
            self._ws_set_feedback(
                False, "Le résultat final n'a pas pu être extrait du workspace."
            )
            return

        res = self._anthro_measure_result(recipe)
        measure = res.measure if res is not None else None
        shown = measure.value_str(value) if measure is not None else (
            f"{value:.1f} {recipe.unit}"
        )
        self._anthro_result_label.setText(f"{recipe.name_fr} = {shown}")

        verdict = None
        lines: list[str] = []
        if measure is not None:
            status = measure.status_for(value)
            if measure.norm_low is not None and measure.norm_high is not None:
                norm = (
                    f"{measure.value_str(measure.norm_low)} – "
                    f"{measure.value_str(measure.norm_high)}"
                )
                verdict = status == "normal"
                lines.append(
                    f"Dans la norme ({norm})" if verdict
                    else f"Hors norme ({norm}) — statut : {status}"
                )
            else:
                lines.append("Mesure informative : pas de norme de référence.")
            if res is not None and res.value is not None:
                lines.append(
                    f"Valeur calculée par le module de référence : "
                    f"{measure.value_str(res.value)} "
                    f"(écart {abs(res.value - value):.1f})."
                )
        lines.append(recipe.interpretation_fr)
        self._anthro_set_norm(verdict, "\n".join(lines))
        self._ws_set_feedback(verdict, f"{recipe.name_fr} = {shown}")

    # ── Anthropometry — 3-D visualisation ─────────────────────────────────────

    def _anthro_draw_point(self, pos, label: str = "") -> None:
        center = tuple(float(v) for v in np.asarray(pos, dtype=float).reshape(3))
        sphere = pv.Sphere(radius=_ISB_POINT_RADIUS, center=center)
        self._anthro_actors.append(
            self._plotter.add_mesh(sphere, color=_ISB_POINT_COLOR, render=False)
        )
        if label:
            actor = self._add_label_actor(center, label)
            if actor is not None:
                self._anthro_actors.append(actor)

    def _anthro_draw_line(self, a, b, label: str = "") -> None:
        pa = np.asarray(a, dtype=float).reshape(3)
        pb = np.asarray(b, dtype=float).reshape(3)
        line = pv.Line(tuple(float(v) for v in pa), tuple(float(v) for v in pb))
        self._anthro_actors.append(
            self._plotter.add_mesh(
                line, color=_ANTHRO_LINE_COLOR, line_width=4, render=False
            )
        )
        if label:
            actor = self._add_label_actor((pa + pb) / 2.0, label, color="#ffd9a0")
            if actor is not None:
                self._anthro_actors.append(actor)

    def _anthro_draw_step(self, step) -> None:
        """Draw the geometry of the step that was just executed."""
        from .anthro_step_engine import (
            ComputeAngle3Pts,
            ComputeAnglePlane,
            ComputeAsymmetry,
            ComputeAxisAngle,
            ComputeDistance,
            ComputeMidpoint as AComputeMidpoint,
            ComputeProjection,
            PickLandmark as APickLandmark,
        )
        engine = self._anthro_engine
        if engine is None:
            return
        ws = engine.workspace
        name = getattr(step, "result_name", "")
        value = ws.get(name)
        unit = self._anthro_recipe.unit if self._anthro_recipe is not None else ""

        def _num_label() -> str:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{float(value):.1f} {unit}".strip()
            return ""

        if isinstance(step, APickLandmark):
            pt = ws.get(name)
            if pt is not None:
                self._anthro_draw_point(pt, name)

        elif isinstance(step, AComputeMidpoint):
            a, b = ws.get(step.expected_a), ws.get(step.expected_b)
            mid = ws.get(name)
            if a is not None and b is not None:
                self._anthro_draw_line(a, b)
            if mid is not None:
                self._anthro_draw_point(mid, name)

        elif isinstance(step, (ComputeDistance, ComputeAnglePlane, ComputeProjection)):
            a, b = ws.get(step.expected_a), ws.get(step.expected_b)
            if a is not None and b is not None:
                self._anthro_draw_line(a, b, _num_label())

        elif isinstance(step, ComputeAngle3Pts):
            a = ws.get(step.expected_a)
            vertex = ws.get(step.expected_vertex)
            c = ws.get(step.expected_c)
            if vertex is not None and a is not None:
                self._anthro_draw_line(vertex, a)
            if vertex is not None and c is not None:
                self._anthro_draw_line(vertex, c)
            if vertex is not None and _num_label():
                actor = self._add_label_actor(vertex, _num_label(), color="#ffd9a0")
                if actor is not None:
                    self._anthro_actors.append(actor)

        elif isinstance(step, ComputeAxisAngle):
            a1, a2 = ws.get(step.expected_a1), ws.get(step.expected_a2)
            b1, b2 = ws.get(step.expected_b1), ws.get(step.expected_b2)
            if a1 is not None and a2 is not None:
                self._anthro_draw_line(a1, a2, _num_label())
            if b1 is not None and b2 is not None:
                self._anthro_draw_line(b1, b2)

        elif isinstance(step, ComputeAsymmetry):
            left, right = ws.get(step.expected_left), ws.get(step.expected_right)
            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                self._anthro_draw_line(left, right, _num_label())

        self._plotter.render()

    def _anthro_set_norm(self, ok, msg: str) -> None:
        """Write the norm verdict into the side-panel result area."""
        self._anthro_norm_label.setTextFormat(Qt.RichText)
        if ok is True:
            prefix, style = "✓ ", _FEEDBACK_OK_STYLE
        elif ok is False:
            prefix, style = "✗ ", _FEEDBACK_ERR_STYLE
        else:
            prefix, style = "", _FEEDBACK_NEUTRAL_STYLE
        self._anthro_norm_label.setStyleSheet(style)
        self._anthro_norm_label.setText((prefix + self._rich(msg)) if msg else "")

    def _clear_anthro_actors(self) -> None:
        """Remove every anthropometry overlay actor from the 3-D view."""
        for actor in self._anthro_actors:
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._anthro_actors = []
        self._plotter.render()
