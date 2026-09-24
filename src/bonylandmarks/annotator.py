"""Manual landmark annotation tool (teacher / researcher side).

This module is **independent from the student exercise**: it opens a "blank"
3-D scan (a GLB with no BodyLoop AutoMarkers) and lets an expert click each of
the 182 anatomical landmarks of ``data/landmarks.json`` on the body surface, in
order to produce reusable ground-truth data.

Workflow
--------
1. *Ouvrir GLB…* loads the scan (mesh + baked texture).
2. The left panel lists the landmarks, filtered by theme and by body side.
3. Selecting a landmark makes it *active*; its palpation hint is shown on the
   right and its name in the HUD at the top of the viewport.
4. Clicking the surface places the active landmark (orange sphere); already
   annotated landmarks stay green with a floating 3-D label.
5. *Enregistrer* / *Exporter JSON…* writes a ``bonylandmarks_annotation``
   document::

       {
         "format": "bonylandmarks_annotation",
         "version": "1.0",
         "scan_file": "scan.glb",
         "annotated_at": "2026-09-23T14:30:00+00:00",
         "annotator": "",
         "n_landmarks": 3,
         "landmarks": {"ASIS_left": [95.2, 985.3, -40.1], ...}
       }

Coordinates are in **millimetres**, in the scan's global frame — the same
convention as ``mesh_loader.load_avatar_glb`` → ``landmark_markers``, so the
viewer can consume the ``landmarks`` dict directly as ``dict[str, ndarray]``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .landmarks_extended import LANDMARKS, THEME_LABELS, Landmark
from .mesh_loader import load_avatar_glb, load_glb_mesh

# ─── Annotation file format ──────────────────────────────────────────────────

ANNOTATION_FORMAT = "bonylandmarks_annotation"
ANNOTATION_VERSION = "1.0"

# ─── Colours (same dark theme as viewer.py / splash.py) ──────────────────────

_BG_COLOR = "#1a1a2e"
_MESH_COLOR = "#c8b8a8"
_DONE_COLOR = "#00cc44"      # landmark already annotated
_ACTIVE_COLOR = "#ff8800"    # sphere of the landmark currently being edited
_TODO_TEXT = "#ffdd00"       # list text of a landmark still to annotate
_DONE_TEXT = "#7fe08a"

_ACTOR_PREFIX = "lm_"
_LABEL_PREFIX = "lbl_"

# ─── Widget styles ───────────────────────────────────────────────────────────

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
    "QPushButton:checked { background: #2e7d32; }"
)
_SECONDARY_BTN_STYLE = (
    "QPushButton { background: rgba(255,255,255,0.07); color: #ccc; "
    "border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; "
    "padding: 4px 10px; font-size: 11px; }"
    "QPushButton:hover { background: rgba(255,255,255,0.13); }"
    "QPushButton:disabled { color: #666; }"
)
_DANGER_BTN_STYLE = (
    "QPushButton { background: rgba(224,85,85,0.18); color: #ff9a9a; "
    "border: 1px solid rgba(224,85,85,0.45); border-radius: 6px; "
    "padding: 4px 10px; font-size: 11px; }"
    "QPushButton:hover { background: rgba(224,85,85,0.38); color: #fff; }"
    "QPushButton:disabled { color: #666; border-color: rgba(255,255,255,0.10); "
    "background: rgba(255,255,255,0.05); }"
)
_LIST_STYLE = (
    "QListWidget { background: rgba(255,255,255,0.04); color: #ddd; "
    "border: 1px solid rgba(255,255,255,0.10); border-radius: 4px; "
    "font-size: 12px; }"
    "QListWidget::item { padding: 3px 4px; }"
    "QListWidget::item:selected { background: #3399ff; color: white; }"
)
_LINE_EDIT_STYLE = (
    "QLineEdit { background: rgba(255,255,255,0.07); color: #ddd; "
    "border: 1px solid rgba(255,255,255,0.12); border-radius: 4px; "
    "padding: 4px 6px; font-size: 12px; }"
)
_WINDOW_STYLE = f"""
    QMainWindow {{ background-color: {_BG_COLOR}; }}
    QWidget#side_panel {{ background-color: #202038; }}
    QLabel {{ color: #e8e8f0; background: transparent; }}
    QToolBar {{ background-color: #252540; border: none; spacing: 4px;
                padding: 4px; }}
    QToolBar QToolButton {{ color: #e8e8f0; font-size: 12px; padding: 5px 10px;
                            border-radius: 4px; }}
    QToolBar QToolButton:hover {{ background: rgba(255,255,255,0.12); }}
    QStatusBar {{ color: #a0a0c0; font-size: 11px; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
"""
_HUD_STYLE = """
    QWidget#annot_hud {
        background: rgba(18, 18, 38, 190);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    QLabel#hud_code { color: rgba(160, 190, 255, 200); font-size: 11px; }
    QLabel#hud_name { color: #f0f0f0; font-size: 18px; font-weight: bold; }
"""

#: Human-readable name of every ``body_side`` value found in landmarks.json.
_SIDE_LABELS: dict[str, str] = {
    "left": "Gauche",
    "right": "Droite",
    "midline": "Médian",
    "bilateral": "Bilatéral",
}


# ─── Annotation document helpers ─────────────────────────────────────────────

def build_annotation_document(
    landmarks: dict[str, np.ndarray],
    scan_file: str = "",
    annotator: str = "",
) -> dict:
    """Return the JSON-serialisable ``bonylandmarks_annotation`` document."""
    return {
        "format": ANNOTATION_FORMAT,
        "version": ANNOTATION_VERSION,
        "scan_file": scan_file,
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotator": annotator,
        "n_landmarks": len(landmarks),
        "landmarks": {
            code: [round(float(v), 2) for v in np.asarray(pos, dtype=float).ravel()[:3]]
            for code, pos in sorted(landmarks.items())
        },
    }


def parse_annotation_document(data: dict) -> "tuple[dict[str, np.ndarray], str, str]":
    """Return ``(landmarks, scan_file, annotator)`` from a loaded annotation dict.

    Accepts ``[x, y, z]`` lists as well as ``{"x": …, "y": …, "z": …}`` mappings
    so that files hand-edited or produced by the BodyLoop SDK still load.
    """
    if not isinstance(data, dict):
        raise ValueError("Le fichier d'annotation doit contenir un objet JSON.")
    fmt = data.get("format")
    if fmt not in (None, ANNOTATION_FORMAT):
        raise ValueError(f"Format d'annotation inconnu : {fmt!r}")

    raw = data.get("landmarks", data if fmt is None else {})
    if not isinstance(raw, dict):
        raise ValueError("La clé 'landmarks' doit être un objet {code: [x, y, z]}.")

    parsed: dict[str, np.ndarray] = {}
    for code, coords in raw.items():
        if isinstance(coords, dict):
            try:
                xyz = [coords["x"], coords["y"], coords["z"]]
            except KeyError:
                continue
        elif isinstance(coords, (list, tuple)) and len(coords) >= 3:
            xyz = list(coords[:3])
        else:
            continue
        try:
            parsed[str(code)] = np.array(xyz, dtype=float)
        except (TypeError, ValueError):
            continue

    return parsed, str(data.get("scan_file", "")), str(data.get("annotator", ""))


# ─── Main window ─────────────────────────────────────────────────────────────

class AnnotatorWindow(QMainWindow):
    """Stand-alone window to place the reference landmarks on a blank scan."""

    def __init__(self, glb_path: "str | Path | None" = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("BonyLandmarks — Outil d'annotation")
        self.resize(1440, 880)
        self.setStyleSheet(_WINDOW_STYLE)

        # ── State ────────────────────────────────────────────────────────────
        self._landmarks: list[Landmark] = list(LANDMARKS)
        self._filtered: list[Landmark] = list(LANDMARKS)
        self._placed: dict[str, np.ndarray] = {}
        self._active_code: str | None = None
        self._mesh: pv.PolyData | None = None
        self._vertex_colors: np.ndarray | None = None
        self._mesh_actor = None
        self._glb_path: Path | None = None
        self._json_path: Path | None = None
        self._dirty: bool = False
        self._updating_list: bool = False
        self._sphere_radius: float = 8.0

        # Camera geometry, filled by _setup_camera()
        self._up_axis: int = 2
        self._front_axis: int = 1
        self._side_axis: int = 0
        self._body_centers: list[float] = [0.0, 0.0, 0.0]
        self._cam_distance: float = 2000.0

        self._build_ui()
        self._build_toolbar()
        self._build_hud_overlay()
        self._setup_shortcuts()

        self._refresh_filters()
        self._refresh_list()
        self._update_info_panel()
        self._update_progress()
        self._update_title()

        if glb_path is not None:
            self.load_glb(Path(glb_path))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_left_panel())

        self._plotter = QtInteractor(central)
        self._plotter.set_background(_BG_COLOR)
        self._plotter.enable_3_lights()
        self._plotter.interactor.installEventFilter(self)
        root.addWidget(self._plotter.interactor, stretch=1)

        root.addWidget(self._build_right_panel())

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ouvrez un scan GLB pour commencer.")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("side_panel")
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("REPÈRES")
        title.setStyleSheet("font-size: 12px; font-weight: bold; color: #7cb9ff;")
        layout.addWidget(title)

        theme_lbl = QLabel("Thème")
        theme_lbl.setStyleSheet(_SECTION_LABEL_STYLE)
        layout.addWidget(theme_lbl)
        self._theme_combo = QComboBox()
        self._theme_combo.setStyleSheet(_COMBO_STYLE)
        self._theme_combo.currentIndexChanged.connect(self._refresh_list)
        layout.addWidget(self._theme_combo)

        side_lbl = QLabel("Côté")
        side_lbl.setStyleSheet(_SECTION_LABEL_STYLE)
        layout.addWidget(side_lbl)
        self._side_combo = QComboBox()
        self._side_combo.setStyleSheet(_COMBO_STYLE)
        self._side_combo.currentIndexChanged.connect(self._refresh_list)
        layout.addWidget(self._side_combo)

        self._list = QListWidget()
        self._list.setStyleSheet(_LIST_STYLE)
        self._list.currentItemChanged.connect(self._on_list_selection)
        layout.addWidget(self._list, stretch=1)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        self._prev_btn = QPushButton("◀  Précédent")
        self._prev_btn.setStyleSheet(_SECONDARY_BTN_STYLE)
        self._prev_btn.clicked.connect(lambda: self._step_landmark(-1))
        nav_row.addWidget(self._prev_btn)
        self._next_btn = QPushButton("Suivant  ▶")
        self._next_btn.setStyleSheet(_SECONDARY_BTN_STYLE)
        self._next_btn.clicked.connect(lambda: self._step_landmark(+1))
        nav_row.addWidget(self._next_btn)
        layout.addLayout(nav_row)

        self._progress_label = QLabel("0/0 annotés")
        self._progress_label.setAlignment(Qt.AlignCenter)
        self._progress_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #7fe08a;"
        )
        layout.addWidget(self._progress_label)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("side_panel")
        panel.setFixedWidth(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._name_label = QLabel("—")
        self._name_label.setWordWrap(True)
        self._name_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #f0f0f0;"
        )
        layout.addWidget(self._name_label)

        self._name_en_label = QLabel("")
        self._name_en_label.setWordWrap(True)
        self._name_en_label.setStyleSheet("font-size: 12px; color: #a0a0c0;")
        layout.addWidget(self._name_en_label)

        self._meta_label = QLabel("")
        self._meta_label.setWordWrap(True)
        self._meta_label.setStyleSheet("font-size: 11px; color: #8a97a8;")
        layout.addWidget(self._meta_label)

        layout.addWidget(self._make_sep())

        hint_hdr = QLabel("PALPATION")
        hint_hdr.setStyleSheet(_SECTION_LABEL_STYLE)
        layout.addWidget(hint_hdr)

        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setAlignment(Qt.AlignTop)
        self._hint_label.setStyleSheet("font-size: 12px; color: #dcdce8;")
        hint_scroll = QScrollArea()
        hint_scroll.setWidgetResizable(True)
        hint_scroll.setWidget(self._hint_label)
        layout.addWidget(hint_scroll, stretch=2)

        self._app_hdr = QLabel("APPLICATION CLINIQUE")
        self._app_hdr.setStyleSheet(_SECTION_LABEL_STYLE)
        layout.addWidget(self._app_hdr)

        self._app_label = QLabel("")
        self._app_label.setWordWrap(True)
        self._app_label.setAlignment(Qt.AlignTop)
        self._app_label.setStyleSheet("font-size: 12px; color: #b8c4d8;")
        self._app_scroll = QScrollArea()
        self._app_scroll.setWidgetResizable(True)
        self._app_scroll.setWidget(self._app_label)
        layout.addWidget(self._app_scroll, stretch=1)

        layout.addWidget(self._make_sep())

        self._coord_label = QLabel("Non placé")
        self._coord_label.setWordWrap(True)
        self._coord_label.setStyleSheet("font-size: 11px; color: #9aa;")
        layout.addWidget(self._coord_label)

        self._place_btn = QPushButton("Placer  (clic sur le corps)")
        self._place_btn.setCheckable(True)
        self._place_btn.setChecked(True)
        self._place_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        self._place_btn.toggled.connect(self._on_place_mode_toggled)
        layout.addWidget(self._place_btn)

        self._clear_btn = QPushButton("Effacer ce repère  (Suppr)")
        self._clear_btn.setStyleSheet(_DANGER_BTN_STYLE)
        self._clear_btn.clicked.connect(self._clear_active)
        layout.addWidget(self._clear_btn)

        self._next_todo_btn = QPushButton("Aller au suivant non annoté")
        self._next_todo_btn.setStyleSheet(_SECONDARY_BTN_STYLE)
        self._next_todo_btn.clicked.connect(self._goto_next_unannotated)
        layout.addWidget(self._next_todo_btn)

        layout.addWidget(self._make_sep())

        annot_lbl = QLabel("Annotateur")
        annot_lbl.setStyleSheet(_SECTION_LABEL_STYLE)
        layout.addWidget(annot_lbl)
        self._annotator_edit = QLineEdit()
        self._annotator_edit.setStyleSheet(_LINE_EDIT_STYLE)
        self._annotator_edit.setPlaceholderText("nom ou initiales (optionnel)")
        self._annotator_edit.textEdited.connect(lambda _t: self._mark_dirty())
        layout.addWidget(self._annotator_edit)

        return panel

    @staticmethod
    def _make_sep() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("color: #333355; background-color: #333355;")
        return sep

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Annotation")
        bar.setMovable(False)

        def _act(text: str, slot, shortcut: str = "", tip: str = "") -> QAction:
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            if tip:
                action.setToolTip(tip)
            action.triggered.connect(slot)
            bar.addAction(action)
            return action

        _act("📂  Ouvrir GLB…", self._on_open_glb, "Ctrl+O")
        _act("📑  Ouvrir annotation JSON…", self._on_open_annotation)
        bar.addSeparator()
        self._save_action = _act("💾  Enregistrer", self._on_save, "Ctrl+S")
        _act("⬇  Exporter JSON…", self._on_export, "Ctrl+Shift+S")
        bar.addSeparator()

        _mouse_tip = (
            "🖱 Rotation : clic-gauche glisser\n"
            "↕ Zoom : molette  ·  Shift+clic : translation\n"
            "Clic droit glisser : zoom alternatif"
        )
        for label, tip, slot in (
            ("Face",  "Vue avant [1]",   lambda: self._set_view(self._front_axis, +1)),
            ("Dos",   "Vue arrière [2]", lambda: self._set_view(self._front_axis, -1)),
            ("G",     "Vue gauche [3]",  lambda: self._set_view(self._side_axis, -1)),
            ("D",     "Vue droite [4]",  lambda: self._set_view(self._side_axis, +1)),
            ("Haut",  "Vue dessus [5]",  self._view_top),
            ("Reset", "Réinitialiser [R]", self._reset_view),
        ):
            btn = QPushButton(label)
            btn.setStyleSheet(_SECONDARY_BTN_STYLE)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            bar.addWidget(btn)

        bar.addSeparator()

        _pan_tip = (
            "Translation (déplacement latéral)\n"
            "Aussi : Shift + clic-gauche glisser\n"
            "Zoom : molette  ·  Rotation : clic-gauche glisser"
        )
        for sym, dx, dy in (("◀", -1, 0), ("▶", +1, 0), ("▲", 0, +1), ("▼", 0, -1)):
            btn = QPushButton(sym)
            btn.setFixedWidth(32)
            btn.setStyleSheet(_SECONDARY_BTN_STYLE)
            btn.setToolTip(_pan_tip)
            btn.clicked.connect(
                lambda _=None, _dx=dx, _dy=dy: self._pan_camera(_dx, _dy)
            )
            bar.addWidget(btn)

        bar.addSeparator()
        bar.addWidget(QLabel(" "))   # espace visuel
        _tip_lbl = QLabel("🖱 clic=rotation · molette=zoom · Shift+clic=déplacement")
        _tip_lbl.setStyleSheet("font-size: 10px; color: #8888aa;")
        bar.addWidget(_tip_lbl)
        bar.addSeparator()

        self._file_label = QLabel("  Aucun scan chargé")
        self._file_label.setStyleSheet("font-size: 11px; color: #a0a0c0;")
        bar.addWidget(self._file_label)

    def _build_hud_overlay(self) -> None:
        """Floating HUD centred at the top of the viewport: active landmark."""
        container = self._plotter.interactor
        overlay = QWidget(container)
        overlay.setObjectName("annot_hud")
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setStyleSheet(_HUD_STYLE)

        layout = QVBoxLayout(overlay)
        layout.setSpacing(2)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setAlignment(Qt.AlignCenter)

        self._hud_code_label = QLabel("")
        self._hud_code_label.setObjectName("hud_code")
        self._hud_code_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._hud_code_label)

        self._hud_name_label = QLabel("")
        self._hud_name_label.setObjectName("hud_name")
        self._hud_name_label.setAlignment(Qt.AlignCenter)
        self._hud_name_label.setWordWrap(True)
        layout.addWidget(self._hud_name_label)

        overlay.adjustSize()
        overlay.setVisible(False)
        self._hud_overlay = overlay

    def _position_hud_overlay(self) -> None:
        if not hasattr(self, "_hud_overlay"):
            return
        container = self._plotter.interactor
        overlay = self._hud_overlay
        overlay.adjustSize()
        x = max(0, (container.width() - overlay.sizeHint().width()) // 2)
        overlay.move(x, 12)
        overlay.raise_()

    def _setup_shortcuts(self) -> None:
        shortcuts = [
            (QKeySequence(Qt.Key_Delete), self._clear_active),
            (QKeySequence(Qt.Key_Backspace), self._clear_active),
            (QKeySequence("N"), self._goto_next_unannotated),
            (QKeySequence("1"), lambda: self._set_view(self._front_axis, +1)),
            (QKeySequence("2"), lambda: self._set_view(self._front_axis, -1)),
            (QKeySequence("3"), lambda: self._set_view(self._side_axis, -1)),
            (QKeySequence("4"), lambda: self._set_view(self._side_axis, +1)),
            (QKeySequence("5"), self._view_top),
            (QKeySequence("R"), self._reset_view),
        ]
        for seq, slot in shortcuts:
            sc = QShortcut(seq, self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(slot)

    # ── Qt events ─────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt naming)
        if obj is self._plotter.interactor and event.type() == QEvent.Resize:
            self._position_hud_overlay()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._position_hud_overlay()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._dirty and self._placed:
            answer = QMessageBox.question(
                self,
                "Annotations non enregistrées",
                f"{len(self._placed)} repère(s) placé(s) n'ont pas été enregistrés.\n"
                "Enregistrer avant de fermer ?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save and not self._on_save():
                event.ignore()
                return
        try:
            self._plotter.close()
        except Exception:                                # pragma: no cover
            pass
        super().closeEvent(event)

    # ── Scan loading ──────────────────────────────────────────────────────────

    def load_glb(self, path: Path) -> None:
        """Load *path* as the working scan, replacing any mesh already shown."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Chargement de {path.name}…")
        try:
            glb_bytes = path.read_bytes()
            # Stickers are kept: on a scan prepared for annotation the physical
            # markers are precisely what the expert wants to see and click.
            try:
                mesh, colors, _landmark_markers, _all_markers = load_avatar_glb(
                    glb_bytes, remove_stickers=False
                )
            except Exception:
                # Plain mesh_3d GLB (no AutoMarkers node / no pygltflib).
                mesh, colors = load_glb_mesh(glb_bytes)
        except Exception as exc:
            QMessageBox.critical(
                self, "Erreur de chargement",
                f"Impossible de charger le scan :\n{exc}",
            )
            self.statusBar().showMessage("Échec du chargement.")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._mesh = mesh
        self._vertex_colors = colors
        self._glb_path = path
        self._sphere_radius = max(4.0, self._model_scale() * 0.006)

        self._plotter.clear()
        # Plotter.clear() also removes every light — restore the 3-point setup.
        self._plotter.enable_3_lights()
        self._add_body_mesh()
        self._setup_camera()
        try:
            # Loading a second scan must not stack a second picking observer.
            self._plotter.disable_picking()
        except Exception:                                # pragma: no cover
            pass
        self._plotter.enable_surface_point_picking(
            callback=self._on_surface_pick,
            show_message=False,
            left_clicking=True,
            pickable_window=False,
        )
        self._redraw_all_markers()
        self._plotter.render()

        self._file_label.setText(f"  {path.name}")
        self._update_title()
        self.statusBar().showMessage(
            f"{path.name} — {mesh.n_points} sommets. "
            "Sélectionnez un repère puis cliquez sur le corps."
        )
        if self._active_code is None:
            self._goto_next_unannotated()
        self._position_hud_overlay()

    def _add_body_mesh(self) -> None:
        if self._mesh is None:
            return
        if self._vertex_colors is not None:
            self._mesh.point_data["RGB"] = self._vertex_colors[:, :3]
            self._mesh_actor = self._plotter.add_mesh(
                self._mesh, scalars="RGB", rgb=True, smooth_shading=True,
                show_scalar_bar=False, ambient=0.4, diffuse=0.8, name="body",
            )
        else:
            self._mesh_actor = self._plotter.add_mesh(
                self._mesh, color=_MESH_COLOR, smooth_shading=True,
                ambient=0.3, diffuse=0.9, name="body",
            )

    def _model_scale(self) -> float:
        if self._mesh is None:
            return 1000.0
        b = self._mesh.bounds
        return float(max(b[1] - b[0], b[3] - b[2], b[5] - b[4]))

    # ── Camera ────────────────────────────────────────────────────────────────

    def _setup_camera(self) -> None:
        """Detect the body axes from the bounding box and frame the whole body."""
        if self._mesh is None:
            return
        b = self._mesh.bounds
        extents = [b[1] - b[0], b[3] - b[2], b[5] - b[4]]
        up_axis = extents.index(max(extents))
        sorted_axes = sorted(range(3), key=lambda i: extents[i])
        front_axis = sorted_axes[0]
        self._up_axis = up_axis
        self._front_axis = front_axis
        self._side_axis = ({0, 1, 2} - {up_axis, front_axis}).pop()
        self._body_centers = [
            (b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5
        ]
        self._cam_distance = extents[up_axis] * 2.5
        self._reset_view()

    def _set_view(self, cam_axis: int, direction: int) -> None:
        if self._mesh is None:
            return
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
        if self._mesh is None:
            return
        cam_pos = list(self._body_centers)
        cam_pos[self._up_axis] += self._cam_distance
        up_vec = [0.0, 0.0, 0.0]
        up_vec[self._front_axis] = 1.0
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(self._body_centers)
        cam.up = tuple(up_vec)
        self._plotter.render()

    def _reset_view(self) -> None:
        self._set_view(self._front_axis, -1)

    def _pan_camera(self, dx: float, dy: float) -> None:
        """Translate the camera laterally; dx > 0 = right, dy > 0 = up."""
        if self._mesh is None:
            return
        cam = self._plotter.camera
        pos = np.asarray(cam.position, dtype=float)
        fpt = np.asarray(cam.focal_point, dtype=float)
        up  = np.asarray(cam.up, dtype=float)
        view_dir = fpt - pos
        dist = float(np.linalg.norm(view_dir))
        if dist < 1e-9:
            return
        view_dir /= dist
        right = np.cross(view_dir, up)
        r_norm = float(np.linalg.norm(right))
        if r_norm < 1e-9:
            return
        right /= r_norm
        up_perp = np.cross(right, view_dir)
        step = dist * 0.05
        delta = right * dx * step + up_perp * dy * step
        cam.position    = tuple(pos + delta)
        cam.focal_point = tuple(fpt + delta)
        self._plotter.render()

    # ── Landmark list & filters ───────────────────────────────────────────────

    def _refresh_filters(self) -> None:
        self._theme_combo.blockSignals(True)
        self._theme_combo.clear()
        self._theme_combo.addItem("Tous les thèmes", None)
        themes = sorted({lm.theme for lm in self._landmarks})
        for theme in themes:
            label = THEME_LABELS.get(theme, (theme, theme))[0]
            n = sum(1 for lm in self._landmarks if lm.theme == theme)
            self._theme_combo.addItem(f"{label}  ({n})", theme)
        self._theme_combo.blockSignals(False)

        self._side_combo.blockSignals(True)
        self._side_combo.clear()
        self._side_combo.addItem("Tous les côtés", None)
        for side in sorted({lm.body_side for lm in self._landmarks}):
            n = sum(1 for lm in self._landmarks if lm.body_side == side)
            self._side_combo.addItem(f"{_SIDE_LABELS.get(side, side)}  ({n})", side)
        self._side_combo.blockSignals(False)

    def _refresh_list(self) -> None:
        """Rebuild the landmark list from the current filters, keeping selection."""
        theme = self._theme_combo.currentData()
        side = self._side_combo.currentData()
        self._filtered = [
            lm for lm in self._landmarks
            if (theme is None or lm.theme == theme)
            and (side is None or lm.body_side == side)
        ]

        self._updating_list = True
        self._list.clear()
        for lm in self._filtered:
            done = lm.code in self._placed
            item = QListWidgetItem(
                f"{'✓ ' if done else '   '}{lm.name_fr} — {lm.code}"
            )
            item.setData(Qt.UserRole, lm.code)
            item.setForeground(QColor(_DONE_TEXT if done else _TODO_TEXT))
            item.setToolTip(f"{lm.name_en}\n{lm.category} · {lm.theme}")
            self._list.addItem(item)
        self._updating_list = False

        if self._active_code is not None:
            self._select_in_list(self._active_code)
        self._update_progress()

    def _select_in_list(self, code: str) -> bool:
        """Select *code* in the list widget; return False when it is filtered out."""
        for row in range(self._list.count()):
            if self._list.item(row).data(Qt.UserRole) == code:
                self._updating_list = True
                self._list.setCurrentRow(row)
                self._updating_list = False
                self._list.scrollToItem(self._list.item(row))
                return True
        return False

    def _on_list_selection(self, current: "QListWidgetItem | None", _previous=None) -> None:
        if self._updating_list or current is None:
            return
        self.set_active(current.data(Qt.UserRole))

    def _step_landmark(self, delta: int) -> None:
        """Move the active landmark *delta* places inside the filtered list."""
        if not self._filtered:
            return
        row = self._list.currentRow()
        if row < 0:
            row = 0 if delta > 0 else len(self._filtered) - 1
        else:
            row = (row + delta) % len(self._filtered)
        self.set_active(self._filtered[row].code)

    def _goto_next_unannotated(self) -> None:
        """Activate the first landmark after the current one that has no position."""
        if not self._filtered:
            self.statusBar().showMessage("Aucun repère dans le filtre courant.")
            return
        start = self._list.currentRow()
        n = len(self._filtered)
        order = range(1, n + 1) if start >= 0 else range(0, n)
        base = start if start >= 0 else 0
        for step in order:
            lm = self._filtered[(base + step) % n]
            if lm.code not in self._placed:
                self.set_active(lm.code)
                return
        self.statusBar().showMessage(
            "Tous les repères du filtre courant sont annotés."
        )

    def set_active(self, code: "str | None") -> None:
        """Make *code* the landmark being annotated and refresh every panel."""
        previous = self._active_code
        self._active_code = code
        if previous is not None and previous != code:
            self._draw_marker(previous)
        if code is not None:
            self._draw_marker(code)
            self._select_in_list(code)
        self._plotter.render()
        self._update_info_panel()

    def _current_landmark(self) -> "Landmark | None":
        if self._active_code is None:
            return None
        for lm in self._landmarks:
            if lm.code == self._active_code:
                return lm
        return None

    # ── Info panel / HUD ──────────────────────────────────────────────────────

    def _update_info_panel(self) -> None:
        lm = self._current_landmark()
        if lm is None:
            self._name_label.setText("—")
            self._name_en_label.setText("")
            self._meta_label.setText("Sélectionnez un repère dans la liste.")
            self._hint_label.setText("")
            self._app_label.setText("")
            self._app_hdr.setVisible(False)
            self._app_scroll.setVisible(False)
            self._coord_label.setText("Non placé")
            self._clear_btn.setEnabled(False)
            self._hud_overlay.setVisible(False)
            return

        self._name_label.setText(lm.name_fr)
        self._name_en_label.setText(lm.name_en)
        theme_label = THEME_LABELS.get(lm.theme, (lm.theme, lm.theme))[0]
        self._meta_label.setText(
            f"{lm.code}  ·  {lm.category}  ·  {theme_label}  ·  "
            f"{_SIDE_LABELS.get(lm.body_side, lm.body_side)}"
        )
        self._hint_label.setText(lm.hint_fr)

        has_app = bool(lm.application_fr)
        self._app_hdr.setVisible(has_app)
        self._app_scroll.setVisible(has_app)
        self._app_label.setText(lm.application_fr)

        pos = self._placed.get(lm.code)
        if pos is None:
            self._coord_label.setText("Non placé")
            self._clear_btn.setEnabled(False)
        else:
            self._coord_label.setText(
                f"x {pos[0]:.1f}   y {pos[1]:.1f}   z {pos[2]:.1f}   (mm)"
            )
            self._clear_btn.setEnabled(True)

        self._hud_code_label.setText(lm.code)
        self._hud_name_label.setText(lm.name_fr)
        self._hud_overlay.setVisible(True)
        self._position_hud_overlay()

    def _update_progress(self) -> None:
        n_filtered_done = sum(1 for lm in self._filtered if lm.code in self._placed)
        self._progress_label.setText(
            f"{n_filtered_done}/{len(self._filtered)} annotés "
            f"(total {len(self._placed)}/{len(self._landmarks)})"
        )

    def _update_title(self) -> None:
        scan = self._glb_path.name if self._glb_path else "aucun scan"
        target = self._json_path.name if self._json_path else "non enregistré"
        star = " *" if self._dirty else ""
        self.setWindowTitle(
            f"BonyLandmarks — Outil d'annotation — {scan} → {target}{star}"
        )

    def _mark_dirty(self, dirty: bool = True) -> None:
        self._dirty = dirty
        self._update_title()

    # ── 3-D markers ───────────────────────────────────────────────────────────

    def _draw_marker(self, code: str) -> None:
        """Draw (or remove) the sphere + label of *code* according to its state."""
        actor_name = f"{_ACTOR_PREFIX}{code}"
        label_name = f"{_LABEL_PREFIX}{code}"
        self._plotter.remove_actor(actor_name, render=False)
        self._plotter.remove_actor(label_name, render=False)

        pos = self._placed.get(code)
        if pos is None:
            return

        is_active = code == self._active_code
        color = _ACTIVE_COLOR if is_active else _DONE_COLOR
        radius = self._sphere_radius * (1.25 if is_active else 1.0)
        self._plotter.add_mesh(
            pv.Sphere(radius=radius, center=pos),
            color=color, name=actor_name, pickable=False, render=False,
        )
        try:
            self._plotter.add_point_labels(
                np.asarray([pos], dtype=float), [code],
                name=label_name, font_size=11,
                text_color="#ffcc66" if is_active else "#d0f0d8",
                shape=None, always_visible=True, render=False,
            )
        except Exception:                                # pragma: no cover
            pass

    def _redraw_all_markers(self) -> None:
        for code in list(self._placed):
            self._draw_marker(code)

    # ── Placement callbacks ───────────────────────────────────────────────────

    def _on_place_mode_toggled(self, checked: bool) -> None:
        self._place_btn.setText(
            "Placer  (clic sur le corps)" if checked
            else "Placement désactivé  (rotation libre)"
        )
        self.statusBar().showMessage(
            "Mode placement actif : cliquez sur le corps."
            if checked else
            "Mode placement désactivé : le clic ne fait que tourner la vue."
        )

    def _on_surface_pick(self, point) -> None:
        """PyVista callback: the user clicked a point on the body surface."""
        if point is None or self._mesh is None:
            return
        if not self._place_btn.isChecked():
            return
        lm = self._current_landmark()
        if lm is None:
            self.statusBar().showMessage(
                "Sélectionnez d'abord un repère dans la liste de gauche."
            )
            return

        self._placed[lm.code] = np.array(point, dtype=float).ravel()[:3]
        self._draw_marker(lm.code)
        self._plotter.render()
        self._mark_dirty()
        self._refresh_list()
        self._update_info_panel()
        pos = self._placed[lm.code]
        self.statusBar().showMessage(
            f"{lm.code} placé à ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm"
        )

    def _clear_active(self) -> None:
        # Del / Backspace must keep editing the annotator name when it has focus.
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return
        code = self._active_code
        if code is None or code not in self._placed:
            return
        del self._placed[code]
        self._draw_marker(code)
        self._plotter.render()
        self._mark_dirty()
        self._refresh_list()
        self._update_info_panel()
        self.statusBar().showMessage(f"{code} effacé.")

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _on_open_glb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un scan GLB", str(self._glb_path.parent) if self._glb_path else "",
            "GLB (*.glb);;Tous les fichiers (*)",
        )
        if path:
            self.load_glb(Path(path))

    def _on_open_annotation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une annotation", "", "JSON (*.json);;Tous les fichiers (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            landmarks, scan_file, annotator = parse_annotation_document(data)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.critical(
                self, "Erreur de lecture", f"Fichier d'annotation invalide :\n{exc}"
            )
            return

        known = {lm.code for lm in self._landmarks}
        unknown = sorted(set(landmarks) - known)
        for code in unknown:
            landmarks.pop(code, None)

        for code in list(self._placed):
            self._plotter.remove_actor(f"{_ACTOR_PREFIX}{code}", render=False)
            self._plotter.remove_actor(f"{_LABEL_PREFIX}{code}", render=False)
        self._placed = landmarks
        self._json_path = Path(path)
        self._annotator_edit.setText(annotator)
        self._redraw_all_markers()
        self._plotter.render()
        self._mark_dirty(False)
        self._refresh_list()
        self._update_info_panel()

        msg = f"{len(landmarks)} repère(s) chargé(s) depuis {Path(path).name}."
        if scan_file and self._glb_path and scan_file != self._glb_path.name:
            msg += f"  ⚠ annotation faite sur « {scan_file} »."
        if unknown:
            msg += f"  {len(unknown)} code(s) inconnu(s) ignoré(s)."
        self.statusBar().showMessage(msg)

    def _write_annotation(self, path: Path) -> bool:
        doc = build_annotation_document(
            self._placed,
            scan_file=self._glb_path.name if self._glb_path else "",
            annotator=self._annotator_edit.text().strip(),
        )
        try:
            path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.critical(
                self, "Erreur d'écriture", f"Impossible d'enregistrer :\n{exc}"
            )
            return False
        self._json_path = path
        self._mark_dirty(False)
        self.statusBar().showMessage(
            f"{len(self._placed)} repère(s) enregistré(s) dans {path.name}."
        )
        return True

    def _default_export_name(self) -> str:
        stem = self._glb_path.stem if self._glb_path else "scan"
        return f"{stem}_annotation.json"

    def _on_save(self) -> bool:
        """Save to the current file; fall back to *Exporter* when there is none."""
        if self._json_path is None:
            return self._on_export()
        return self._write_annotation(self._json_path)

    def _on_export(self) -> bool:
        start_dir = str(
            self._json_path
            if self._json_path
            else (self._glb_path.parent / self._default_export_name()
                  if self._glb_path else self._default_export_name())
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter les annotations", start_dir,
            "JSON (*.json);;Tous les fichiers (*)",
        )
        if not path:
            return False
        return self._write_annotation(Path(path))


def main() -> None:
    """Run the annotation tool stand-alone (``python -m bonylandmarks.annotator``)."""
    app = QApplication(sys.argv)
    app.setApplicationName("BonyLandmarks Annotator")
    glb = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    window = AnnotatorWindow(glb_path=glb)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
