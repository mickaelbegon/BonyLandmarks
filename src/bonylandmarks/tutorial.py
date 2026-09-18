"""Guided tour ("mode tutoriel") of the anatomical landmarks.

This is a **free exploration** mode shown *before* the graded session: the
student walks through the landmarks one by one, the camera orbits to face each
one, and the side panel reveals the full palpation hint plus its clinical
application.

Two sub-modes are available, toggled by a button in the panel header:

* **Reconnaissance** (default) — green sphere shows the ground-truth position,
  full hint + clinical application displayed, no clicking on the mesh.
* **Placement** — hint is hidden (student must find the landmark independently),
  the mesh is clickable, a yellow candidate sphere appears on click, and after
  confirmation the distance to ground truth + the full hint are revealed as a
  debrief.  No score is recorded — purely for practice.

Nothing is scored here — deliberately.  ``session.py`` is therefore **not**
imported: the tutorial owns a tiny cursor over a filtered landmark list and
knows nothing about grades, retries or ground-truth errors.

Pedagogical intent
------------------
* the theme badge contextualises the landmark (posture, EMG, ISAK...);
* the hint is shown *in full* in Reconnaissance (in the exercise it is only
  revealed afterwards);
* the clinical application answers "why do I need to find this?";
* protocol reminders (SENIAM inter-electrode distance, ISAK right-side rule)
  are surfaced for the non-bony categories;
* the tone stays encouraging and non-evaluative.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .landmarks_extended import (
    CATEGORY_LABELS,
    LANDMARKS,
    THEME_LABELS,
    Landmark,
)

# ─── colour constants (kept in sync with viewer.py) ──────────────────────────

_MESH_COLOR = "#c8b8a8"
_TUTORIAL_MARKER_COLOR = "#00cc44"   # green sphere on the ground truth (recon mode)
_CANDIDATE_COLOR = "#ffdd00"         # yellow candidate sphere (placement mode)
_BACKGROUND = "#1a1a2e"
_SPHERE_ACTOR = "tutorial_sphere"
_CANDIDATE_ACTOR = "candidate_sphere_tut"
_SPHERE_RADIUS = 22.0
_CAM_DISTANCE = 1950.0               # mm from the body axis (same as viewer)

_ALL = "__all__"


# ─── local bilingual strings (tutorial-only, keeps i18n.py untouched) ────────

_T: dict[str, tuple[str, str]] = {
    "title": ("Mode Tutoriel", "Tutorial Mode"),
    "subtitle": (
        "Visite guidée — explorez librement, rien n'est noté.",
        "Guided tour — explore freely, nothing is graded.",
    ),
    "category_filter": ("Catégorie", "Category"),
    "theme_filter": ("Thème", "Theme"),
    "all_categories": ("Toutes les catégories", "All categories"),
    "all_themes": ("Tous les thèmes", "All themes"),
    "counter": ("Repère {index} / {total}", "Landmark {index} / {total}"),
    "empty": (
        "Aucun repère ne correspond à ce filtre.",
        "No landmark matches this filter.",
    ),
    "prev": ("← Précédent", "← Previous"),
    "next": ("Suivant →", "Next →"),
    "start": ("Commencer l'exercice", "Start the exercise"),
    "not_scored": (
        "Position approximative — non évaluée",
        "Approximate position — not scored",
    ),
    "palpation": ("Palpation", "Palpation"),
    "application": ("Application clinique", "Clinical application"),
    "opacity": ("Transparence", "Opacity"),
    "lang_toggle": ("English", "Français"),
    "hint_navigation": (
        "Utilisez ← / → pour naviguer entre les repères.",
        "Use ← / → to move between landmarks.",
    ),
    # ── Mode switch ───────────────────────────────────────────────────────────
    "mode_switch_to_place": (
        "Mode : Reconnaissance  →  Placement",
        "Mode: Recognition  →  Placement",
    ),
    "mode_switch_to_recon": (
        "Mode : Placement  →  Reconnaissance",
        "Mode: Placement  →  Recognition",
    ),
    # ── Placement mode widgets ────────────────────────────────────────────────
    "placement_instr": (
        "Cliquez sur le mesh pour positionner le repère.",
        "Click on the mesh to place the landmark.",
    ),
    "placement_confirm": ("Confirmer", "Confirm"),
    "placement_restart": ("Recommencer", "Restart"),
    "debrief_distance": (
        "Distance : {value:.1f} mm",
        "Distance: {value:.1f} mm",
    ),
    "debrief_no_gt": (
        "Aucune vérité terrain disponible pour ce repère.",
        "No ground truth available for this landmark.",
    ),
    "debrief_hint_label": ("Indice de palpation :", "Palpation hint:"),
    # ── Scan selector ─────────────────────────────────────────────────────────
    "scan_label": ("Scan actif", "Active scan"),
    "scan_male": ("Homme (défaut)", "Male (default)"),
}

# Protocol reminders shown for the non-bony categories.
_PROTOCOL_NOTES: dict[str, tuple[str, str]] = {
    "EMG": (
        "Protocole SENIAM — la pastille marque le centre de la paire "
        "d'électrodes (inter-électrode 20 mm), orientée selon l'axe des fibres.",
        "SENIAM protocol — the marker is the centre of the electrode pair "
        "(20 mm inter-electrode distance), aligned with the muscle fibres.",
    ),
    "SKINFOLD": (
        "Protocole ISAK — site mesuré du côté droit ; la pastille marque "
        "le centre de la pince.",
        "ISAK protocol — measured on the right side; the marker is the "
        "centre of the caliper.",
    ),
    "ANTHRO": (
        "Protocole ISAK — circonférences et diamètres mesurés du côté droit.",
        "ISAK protocol — girths and diameters measured on the right side.",
    ),
    "BONE": (
        "Repère osseux palpable — base de toute mesure en kinésiologie.",
        "Palpable bony landmark — the foundation of every kinesiology measurement.",
    ),
}


def _t(key: str, lang: str = "fr", **kwargs: object) -> str:
    """Translate a tutorial-local *key*, formatting with **kwargs."""
    fr, en = _T.get(key, (key, key))
    text = fr if lang == "fr" else en
    return text.format(**kwargs) if kwargs else text


class TutorialViewer(QWidget):
    """Free-exploration guided tour of the landmark set (no scoring).

    Two sub-modes are toggled by the user via a button at the top of the panel:

    * ``"recon"`` — Reconnaissance: green sphere + full hint (default).
    * ``"place"`` — Placement: mesh clickable, yellow candidate sphere, debrief
      after confirmation (no score recorded).
    """

    start_session = Signal()

    def __init__(
        self,
        mesh: pv.PolyData,
        ground_truth: dict[str, np.ndarray],
        landmark_codes: list[str],
        lang: str = "fr",
        vertex_colors: np.ndarray | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mesh = mesh
        self._ground_truth = ground_truth
        self._vertex_colors = vertex_colors
        self._lang = lang

        codes = set(landmark_codes)
        # Declaration order of LANDMARKS keeps BONE → EMG → SKINFOLD → ANTHRO.
        self._all_landmarks: list[Landmark] = [
            lm for lm in LANDMARKS if lm.code in codes
        ]
        self._landmarks: list[Landmark] = list(self._all_landmarks)
        self._index: int = 0

        self._mesh_actor = None
        self._sphere_shown: bool = False

        # ── Sub-mode state ────────────────────────────────────────────────────
        # "recon": Reconnaissance (default) — green sphere, full hint.
        # "place": Placement — mesh picking, yellow candidate sphere, debrief.
        self._mode: str = "recon"
        self._candidate_point: np.ndarray | None = None

        self._build_ui()
        self._setup_scene()
        self._show_current()

    # ── landmark list helpers ────────────────────────────────────────────────

    def current_landmark(self) -> Landmark | None:
        """Landmark currently presented, or None when the filter is empty."""
        if not self._landmarks:
            return None
        return self._landmarks[self._index]

    def _selected_category(self) -> str:
        return self._category_combo.currentData()

    def _selected_theme(self) -> str:
        return self._theme_combo.currentData()

    def _apply_filters(self) -> None:
        """Rebuild the visible landmark list from both combo boxes."""
        category = self._selected_category()
        theme = self._selected_theme()
        keep_code = None
        current = self.current_landmark()
        if current is not None:
            keep_code = current.code

        self._landmarks = [
            lm
            for lm in self._all_landmarks
            if (category == _ALL or lm.category == category)
            and (theme == _ALL or lm.theme == theme)
        ]
        # Stay on the same landmark when it survives the new filter.
        self._index = 0
        if keep_code is not None:
            for i, lm in enumerate(self._landmarks):
                if lm.code == keep_code:
                    self._index = i
                    break
        self._show_current()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Left: 3-D plotter
        self._plotter = QtInteractor(self)
        self._plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._plotter.interactor, stretch=3)

        # Right: control panel
        panel = QVBoxLayout()
        panel.setSpacing(10)

        # Title + language toggle
        header = QHBoxLayout()
        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 17px; font-weight: bold;")
        header.addWidget(self._title_label, stretch=1)
        self._lang_btn = QPushButton()
        self._lang_btn.setFixedWidth(90)
        self._lang_btn.clicked.connect(self._toggle_lang)
        header.addWidget(self._lang_btn)
        panel.addLayout(header)

        self._subtitle_label = QLabel()
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setStyleSheet("font-size: 12px; color: #555;")
        panel.addWidget(self._subtitle_label)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.HLine)
        sep0.setFrameShadow(QFrame.Sunken)
        panel.addWidget(sep0)

        # ── Mode switch button ────────────────────────────────────────────────
        # Toggles between Reconnaissance (green sphere + full hint) and
        # Placement (mesh clickable, yellow sphere, debrief after confirm).
        self._mode_switch_btn = QPushButton()
        self._mode_switch_btn.setCheckable(True)
        self._mode_switch_btn.setChecked(False)  # False = Reconnaissance
        self._mode_switch_btn.setStyleSheet(
            "font-size: 12px; padding: 6px; "
            "background-color: #37474f; color: white; border-radius: 5px;"
        )
        self._mode_switch_btn.clicked.connect(self._on_mode_toggled)
        panel.addWidget(self._mode_switch_btn)

        # ── Scan selector (disabled — second scan not yet available) ──────────
        # TODO: Enable when a second scan (female) GLB/URL is loaded.
        #       Add the item with its data key and wire up a signal that calls
        #       a future _load_scan(key) method.  Until then the combo is
        #       disabled so the affordance is visible but non-functional.
        scan_row = QHBoxLayout()
        self._scan_caption = QLabel()
        self._scan_caption.setStyleSheet("font-size: 11px;")
        self._scan_caption.setFixedWidth(80)
        scan_row.addWidget(self._scan_caption)
        self._scan_combo = QComboBox()
        self._scan_combo.addItem(_t("scan_male", self._lang), "male")
        self._scan_combo.setEnabled(False)  # disabled — only one scan available
        scan_row.addWidget(self._scan_combo, stretch=1)
        panel.addLayout(scan_row)

        sep0b = QFrame()
        sep0b.setFrameShape(QFrame.HLine)
        sep0b.setFrameShadow(QFrame.Sunken)
        panel.addWidget(sep0b)

        # Category filter
        cat_row = QHBoxLayout()
        self._category_caption = QLabel()
        self._category_caption.setFixedWidth(80)
        self._category_caption.setStyleSheet("font-size: 12px;")
        cat_row.addWidget(self._category_caption)
        self._category_combo = QComboBox()
        self._category_combo.addItem(_t("all_categories", self._lang), _ALL)
        for cat in ("BONE", "EMG", "SKINFOLD", "ANTHRO"):
            if any(lm.category == cat for lm in self._all_landmarks):
                self._category_combo.addItem(cat, cat)
        self._category_combo.currentIndexChanged.connect(
            lambda _i: self._apply_filters()
        )
        cat_row.addWidget(self._category_combo, stretch=1)
        panel.addLayout(cat_row)

        # Theme filter
        theme_row = QHBoxLayout()
        self._theme_caption = QLabel()
        self._theme_caption.setFixedWidth(80)
        self._theme_caption.setStyleSheet("font-size: 12px;")
        theme_row.addWidget(self._theme_caption)
        self._theme_combo = QComboBox()
        self._theme_combo.addItem(_t("all_themes", self._lang), _ALL)
        seen_themes: list[str] = []
        for lm in self._all_landmarks:
            if lm.theme not in seen_themes:
                seen_themes.append(lm.theme)
        for theme in seen_themes:
            labels = THEME_LABELS.get(theme, (theme, theme))
            self._theme_combo.addItem(
                labels[0] if self._lang == "fr" else labels[1], theme
            )
        self._theme_combo.currentIndexChanged.connect(lambda _i: self._apply_filters())
        theme_row.addWidget(self._theme_combo, stretch=1)
        panel.addLayout(theme_row)

        # Counter
        self._counter_label = QLabel()
        self._counter_label.setAlignment(Qt.AlignCenter)
        self._counter_label.setStyleSheet("font-size: 13px; color: #555;")
        panel.addWidget(self._counter_label)

        # Landmark name
        self._name_label = QLabel()
        self._name_label.setWordWrap(True)
        self._name_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        panel.addWidget(self._name_label)

        # Theme badge
        self._theme_badge = QLabel()
        self._theme_badge.setAlignment(Qt.AlignCenter)
        self._theme_badge.setFixedHeight(24)
        panel.addWidget(self._theme_badge)

        # "Not scored" notice for landmarks without a ground truth
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            "font-size: 11px; color: #8a6d00; background: #fff7d6; "
            "border: 1px solid #e6cf7a; border-radius: 4px; padding: 4px;"
        )
        self._status_label.setVisible(False)
        panel.addWidget(self._status_label)

        # ── Reconnaissance-mode content ───────────────────────────────────────
        # Palpation hint
        self._hint_caption = QLabel()
        self._hint_caption.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #444;"
        )
        panel.addWidget(self._hint_caption)

        self._hint_text = QTextEdit()
        self._hint_text.setReadOnly(True)
        self._hint_text.setMinimumHeight(110)
        self._hint_text.setStyleSheet("font-size: 12px; color: #333;")
        panel.addWidget(self._hint_text, stretch=1)

        # Clinical application
        self._application_caption = QLabel()
        self._application_caption.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #336;"
        )
        panel.addWidget(self._application_caption)

        self._application_text = QTextEdit()
        self._application_text.setReadOnly(True)
        self._application_text.setMinimumHeight(70)
        self._application_text.setStyleSheet(
            "font-size: 11px; color: #336; font-style: italic; "
            "background: #f0f4ff; border: 1px solid #c0c8e8; border-radius: 4px;"
        )
        panel.addWidget(self._application_text)

        # Protocol note (SENIAM / ISAK / bony landmark)
        self._protocol_label = QLabel()
        self._protocol_label.setWordWrap(True)
        self._protocol_label.setStyleSheet(
            "font-size: 11px; color: #2e5d34; background: #eef7ef; "
            "border: 1px solid #b9d9bd; border-radius: 4px; padding: 5px;"
        )
        panel.addWidget(self._protocol_label)

        # ── Placement-mode content ────────────────────────────────────────────
        # Grouped in a QWidget so it can be shown/hidden as a unit.
        self._placement_area = QWidget()
        pa_layout = QVBoxLayout(self._placement_area)
        pa_layout.setContentsMargins(0, 0, 0, 0)
        pa_layout.setSpacing(6)

        self._placement_instr = QLabel()
        self._placement_instr.setWordWrap(True)
        self._placement_instr.setStyleSheet(
            "font-size: 12px; font-style: italic; color: #555;"
        )
        pa_layout.addWidget(self._placement_instr)

        placement_btn_row = QHBoxLayout()
        self._placement_confirm_btn = QPushButton()
        self._placement_confirm_btn.setEnabled(False)
        self._placement_confirm_btn.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 8px; "
            "background-color: #1565c0; color: white; border-radius: 5px;"
        )
        self._placement_confirm_btn.clicked.connect(self._on_placement_confirm)
        placement_btn_row.addWidget(self._placement_confirm_btn)

        self._placement_restart_btn = QPushButton()
        self._placement_restart_btn.setEnabled(False)
        self._placement_restart_btn.setStyleSheet(
            "font-size: 12px; padding: 8px; "
            "background-color: #555; color: white; border-radius: 5px;"
        )
        self._placement_restart_btn.clicked.connect(self._on_placement_restart)
        placement_btn_row.addWidget(self._placement_restart_btn)
        pa_layout.addLayout(placement_btn_row)

        # Debrief: shown after the student confirms a pick
        self._placement_debrief_widget = QWidget()
        pd_layout = QVBoxLayout(self._placement_debrief_widget)
        pd_layout.setContentsMargins(0, 4, 0, 0)
        pd_layout.setSpacing(4)

        self._placement_debrief_dist = QLabel()
        self._placement_debrief_dist.setAlignment(Qt.AlignCenter)
        self._placement_debrief_dist.setWordWrap(True)
        self._placement_debrief_dist.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1a5276; "
            "background: #d6eaf8; border-radius: 4px; padding: 4px;"
        )
        pd_layout.addWidget(self._placement_debrief_dist)

        self._placement_debrief_hint_caption = QLabel()
        self._placement_debrief_hint_caption.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #444;"
        )
        pd_layout.addWidget(self._placement_debrief_hint_caption)

        self._placement_debrief_hint_text = QTextEdit()
        self._placement_debrief_hint_text.setReadOnly(True)
        self._placement_debrief_hint_text.setMinimumHeight(90)
        self._placement_debrief_hint_text.setStyleSheet(
            "font-size: 12px; color: #333;"
        )
        pd_layout.addWidget(self._placement_debrief_hint_text)

        self._placement_debrief_widget.setVisible(False)
        pa_layout.addWidget(self._placement_debrief_widget)

        # Placement area hidden by default (Reconnaissance is the default mode)
        self._placement_area.setVisible(False)
        panel.addWidget(self._placement_area, stretch=1)

        # ── Common bottom controls (both modes) ───────────────────────────────

        # Opacity slider (helps see deep landmarks through the skin)
        opacity_row = QHBoxLayout()
        self._opacity_caption = QLabel()
        self._opacity_caption.setStyleSheet("font-size: 11px;")
        opacity_row.addWidget(self._opacity_caption)
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(20, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider, stretch=1)
        panel.addLayout(opacity_row)

        # Navigation
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton()
        self._prev_btn.clicked.connect(self._on_prev)
        self._next_btn = QPushButton()
        self._next_btn.clicked.connect(self._on_next)
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        panel.addLayout(nav_row)

        self._nav_hint_label = QLabel()
        self._nav_hint_label.setAlignment(Qt.AlignCenter)
        self._nav_hint_label.setStyleSheet("font-size: 10px; color: #888;")
        panel.addWidget(self._nav_hint_label)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        panel.addWidget(sep1)

        # Main call to action
        self._start_btn = QPushButton()
        self._start_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 10px; "
            "background-color: #2e7d32; color: white; border-radius: 6px;"
        )
        self._start_btn.clicked.connect(self.start_session.emit)
        panel.addWidget(self._start_btn)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(340)
        root.addWidget(panel_widget)

        # Keyboard navigation
        for key, slot in (
            (Qt.Key_Left, self._on_prev),
            (Qt.Key_Right, self._on_next),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(slot)

        self._retranslate()

    # ── Scene setup ──────────────────────────────────────────────────────────

    def _add_body_mesh(self) -> None:
        """Add (or replace) the body mesh actor, reusing the caller's colours."""
        if self._mesh_actor is not None:
            self._plotter.remove_actor(self._mesh_actor, render=False)
        opacity = self._opacity_slider.value() / 100.0
        if self._vertex_colors is not None:
            self._mesh.point_data["RGB"] = self._vertex_colors[:, :3]
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
                self._mesh,
                color=_MESH_COLOR,
                smooth_shading=True,
                opacity=opacity,
                ambient=0.3,
                diffuse=0.9,
            )

    def _setup_scene(self) -> None:
        pl = self._plotter
        pl.background_color = _BACKGROUND
        pl.enable_3_lights()
        self._add_body_mesh()
        pl.show()
        self._reset_camera_front()

    def _reset_camera_front(self) -> None:
        """Full-body front view, avatar upright (same framing as the viewer)."""
        bounds = self._mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        extents = [bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]
        up_axis = extents.index(max(extents))   # body's vertical axis (0=X, 1=Y, 2=Z)
        sorted_axes = sorted(range(3), key=lambda i: extents[i])
        front_axis = sorted_axes[-2]            # second-largest extent = "front"
        centers = [
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        ]
        height = extents[up_axis]
        cam_pos = list(centers)
        cam_pos[front_axis] += height * 1.8
        up_vec = [0.0, 0.0, 0.0]
        up_vec[up_axis] = 1.0
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(centers)
        cam.up = tuple(up_vec)
        self._plotter.render()

    def _orbit_camera_to(self, point: np.ndarray) -> None:
        """Orbit the camera around the body axis to face *point*.

        Same logic as ``LandmarkViewer._orbit_camera_to``: the focus sits on the
        body axis at the landmark's height, and the camera is pushed out along
        the horizontal direction of the landmark (front view as fallback).
        Vertical axis is detected from bounds to work for both Y-up and Z-up GLBs.
        """
        bounds = self._mesh.bounds
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
        horiz_norm = float(np.linalg.norm(horiz))
        if horiz_norm < 1.0:
            sorted_axes = sorted(range(3), key=lambda i: extents[i])
            direction = np.zeros(3)
            direction[sorted_axes[-2]] = 1.0    # fallback: front along second-largest axis
        else:
            direction = horiz / horiz_norm
        cam_pos = focus + direction * _CAM_DISTANCE
        up_vec = [0.0, 0.0, 0.0]
        up_vec[up_axis] = 1.0
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(focus)
        cam.up = tuple(up_vec)
        self._plotter.render()

    # ── Presentation of the current landmark ─────────────────────────────────

    def _clear_sphere(self) -> None:
        """Remove the ground-truth (green) sphere actor if present."""
        if self._sphere_shown:
            self._plotter.remove_actor(_SPHERE_ACTOR, render=False)
            self._sphere_shown = False

    def _clear_candidate(self) -> None:
        """Remove the candidate (yellow) sphere actor if present."""
        self._plotter.remove_actor(_CANDIDATE_ACTOR, render=False)
        self._candidate_point = None

    def _show_current(self) -> None:
        """Refresh the 3-D scene and the whole right panel for the cursor."""
        lm = self.current_landmark()
        self._clear_sphere()

        # In placement mode, reset the candidate state on every navigation
        if self._mode == "place":
            self._clear_candidate()
            self._placement_confirm_btn.setEnabled(False)
            self._placement_restart_btn.setEnabled(False)
            self._placement_debrief_widget.setVisible(False)

        if lm is None:
            self._counter_label.setText(_t("empty", self._lang))
            self._name_label.setText("—")
            self._theme_badge.setVisible(False)
            self._status_label.setVisible(False)
            if self._mode == "recon":
                self._hint_text.setPlainText("")
                self._application_text.setVisible(False)
                self._application_caption.setVisible(False)
                self._protocol_label.setVisible(False)
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            self._plotter.render()
            return

        total = len(self._landmarks)
        self._counter_label.setText(
            _t("counter", self._lang, index=self._index + 1, total=total)
        )
        self._name_label.setText(lm.name(self._lang))

        # Theme badge (common to both modes)
        theme_label = lm.theme_label(self._lang)
        self._theme_badge.setText(theme_label)
        self._theme_badge.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: white; "
            f"background-color: {lm.theme_color()}; "
            "border-radius: 4px; padding: 2px 8px;"
        )
        self._theme_badge.setVisible(bool(theme_label))

        gt = self._ground_truth.get(lm.code)

        if self._mode == "recon":
            # ── Reconnaissance: full hint + green sphere ───────────────────────
            # Palpation hint (shown in full — this is the teaching mode)
            self._hint_text.setPlainText(lm.hint(self._lang))

            # Clinical application (hidden when the landmark has none)
            application = lm.application(self._lang)
            self._application_text.setPlainText(application)
            self._application_text.setVisible(bool(application))
            self._application_caption.setVisible(bool(application))

            # Protocol reminder for the category
            note = _PROTOCOL_NOTES.get(lm.category)
            if note is not None:
                self._protocol_label.setText(note[0] if self._lang == "fr" else note[1])
                self._protocol_label.setVisible(True)
            else:
                self._protocol_label.setVisible(False)

            # 3-D: green sphere + camera orbit when a ground truth exists
            if gt is not None:
                gt_arr = np.asarray(gt, dtype=float)
                self._plotter.add_mesh(
                    pv.Sphere(radius=_SPHERE_RADIUS, center=gt_arr),
                    color=_TUTORIAL_MARKER_COLOR,
                    name=_SPHERE_ACTOR,
                )
                self._sphere_shown = True
                self._status_label.setVisible(False)
                self._orbit_camera_to(gt_arr)
            else:
                # EMG / SKINFOLD / ANTHRO without a BodyLoop marker
                self._status_label.setText(_t("not_scored", self._lang))
                self._status_label.setVisible(True)
                self._plotter.render()

        else:
            # ── Placement: orbit camera but do NOT reveal the sphere ───────────
            if gt is not None:
                gt_arr = np.asarray(gt, dtype=float)
                self._status_label.setVisible(False)
                self._orbit_camera_to(gt_arr)
            else:
                # No ground truth — student can still click, but distance will
                # not be computed.
                self._status_label.setText(_t("not_scored", self._lang))
                self._status_label.setVisible(True)
                self._plotter.render()

        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < total - 1)

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_mode_toggled(self) -> None:
        """Handle the mode-switch button click."""
        new_mode = "place" if self._mode_switch_btn.isChecked() else "recon"
        self._set_mode(new_mode)

    def _set_mode(self, mode: str) -> None:
        """Switch between ``'recon'`` (Reconnaissance) and ``'place'`` (Placement).

        * Reconnaissance: disable mesh picking, restore green sphere, show full
          hint/application/protocol widgets.
        * Placement: enable surface picking, clear green sphere, show only the
          landmark name + placement instructions + confirm/restart/debrief.
        """
        self._mode = mode
        is_recon = (mode == "recon")

        # Toggle visibility of mode-specific content areas
        self._hint_caption.setVisible(is_recon)
        self._hint_text.setVisible(is_recon)
        self._application_caption.setVisible(is_recon)
        self._application_text.setVisible(is_recon)
        self._protocol_label.setVisible(is_recon)
        self._placement_area.setVisible(not is_recon)

        # Update the switch button label
        key = "mode_switch_to_place" if is_recon else "mode_switch_to_recon"
        self._mode_switch_btn.setText(_t(key, self._lang))

        if is_recon:
            # Disable surface picking (ignore errors — may not have been enabled)
            try:
                self._plotter.disable_picking()
            except Exception:
                pass
            # Clear any candidate sphere left over from placement mode
            self._clear_candidate()

        else:
            # Enable surface picking — same API as LandmarkViewer._setup_scene
            self._plotter.enable_surface_point_picking(
                callback=self._on_surface_pick_tutorial,
                show_message=False,
                left_clicking=True,
                pickable_window=False,
            )
            # Clear the recognition sphere so the student can't cheat
            self._clear_sphere()
            self._placement_confirm_btn.setEnabled(False)
            self._placement_restart_btn.setEnabled(False)
            self._placement_debrief_widget.setVisible(False)

        # Refresh the panel and 3-D scene for the current landmark
        self._show_current()

    # ── Placement mode callbacks ──────────────────────────────────────────────

    def _on_surface_pick_tutorial(self, point: np.ndarray) -> None:
        """Called by PyVista when the user clicks the mesh in Placement mode.

        Mirrors the same pattern used in ``LandmarkViewer._on_surface_pick``.
        """
        if self._mode != "place":
            return
        lm = self.current_landmark()
        if lm is None:
            return

        self._candidate_point = np.array(point, dtype=float)

        # Replace previous candidate sphere
        sphere = pv.Sphere(radius=_SPHERE_RADIUS, center=self._candidate_point)
        self._plotter.add_mesh(sphere, color=_CANDIDATE_COLOR, name=_CANDIDATE_ACTOR)
        self._plotter.render()

        # Enable action buttons; hide any previous debrief
        self._placement_confirm_btn.setEnabled(True)
        self._placement_restart_btn.setEnabled(True)
        self._placement_debrief_widget.setVisible(False)

    def _on_placement_confirm(self) -> None:
        """Confirm the candidate pick: compute distance, reveal ground truth, show hint."""
        if self._candidate_point is None:
            return
        lm = self.current_landmark()
        if lm is None:
            return

        # Disable buttons during debrief (student navigates manually)
        self._placement_confirm_btn.setEnabled(False)
        self._placement_restart_btn.setEnabled(False)

        gt = self._ground_truth.get(lm.code)
        if gt is not None:
            gt_arr = np.asarray(gt, dtype=float)
            dist = float(np.linalg.norm(self._candidate_point - gt_arr))
            self._placement_debrief_dist.setText(
                _t("debrief_distance", self._lang, value=dist)
            )
            # Reveal the ground-truth sphere (green) next to the candidate
            self._plotter.add_mesh(
                pv.Sphere(radius=_SPHERE_RADIUS, center=gt_arr),
                color=_TUTORIAL_MARKER_COLOR,
                name=_SPHERE_ACTOR,
            )
            self._sphere_shown = True
        else:
            self._placement_debrief_dist.setText(_t("debrief_no_gt", self._lang))

        self._plotter.render()

        # Reveal the full palpation hint as a post-placement debrief
        self._placement_debrief_hint_text.setPlainText(lm.hint(self._lang))
        self._placement_debrief_widget.setVisible(True)

    def _on_placement_restart(self) -> None:
        """Clear the candidate sphere so the student can pick again."""
        self._clear_candidate()
        self._clear_sphere()
        self._plotter.render()
        self._placement_confirm_btn.setEnabled(False)
        self._placement_restart_btn.setEnabled(False)
        self._placement_debrief_widget.setVisible(False)

    # ── Teardown ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release the VTK render window before the widget is replaced.

        ``QMainWindow.setCentralWidget`` deletes the previous widget; closing
        the plotter first avoids leaking the interactor/render window.
        """
        self._clear_sphere()
        self._clear_candidate()
        try:
            self._plotter.disable_picking()
        except Exception:
            pass
        try:
            self._plotter.close()
        except Exception:  # pragma: no cover - VTK teardown is best-effort
            pass

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_prev(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def _on_next(self) -> None:
        if self._index < len(self._landmarks) - 1:
            self._index += 1
            self._show_current()

    def _on_opacity_changed(self, value: int) -> None:
        if self._mesh_actor is not None:
            self._mesh_actor.prop.opacity = value / 100.0
            self._plotter.render()

    def _toggle_lang(self) -> None:
        self._lang = "en" if self._lang == "fr" else "fr"
        self._retranslate()
        self._show_current()

    def _retranslate(self) -> None:
        """Apply the current language to every static widget label."""
        lang = self._lang
        self._title_label.setText(f"{_t('title', lang)}")
        self._subtitle_label.setText(_t("subtitle", lang))
        self._lang_btn.setText(_t("lang_toggle", lang))
        self._category_caption.setText(_t("category_filter", lang))
        self._theme_caption.setText(_t("theme_filter", lang))
        self._hint_caption.setText(_t("palpation", lang))
        self._application_caption.setText(_t("application", lang))
        self._opacity_caption.setText(_t("opacity", lang))
        self._prev_btn.setText(_t("prev", lang))
        self._next_btn.setText(_t("next", lang))
        self._nav_hint_label.setText(_t("hint_navigation", lang))
        self._start_btn.setText(_t("start", lang))

        # Mode-switch button: text depends on current mode
        mode_key = (
            "mode_switch_to_place" if self._mode == "recon" else "mode_switch_to_recon"
        )
        self._mode_switch_btn.setText(_t(mode_key, lang))

        # Scan selector label
        self._scan_caption.setText(_t("scan_label", lang))
        self._scan_combo.setItemText(0, _t("scan_male", lang))

        # Placement-mode labels
        self._placement_instr.setText(_t("placement_instr", lang))
        self._placement_confirm_btn.setText(_t("placement_confirm", lang))
        self._placement_restart_btn.setText(_t("placement_restart", lang))
        self._placement_debrief_hint_caption.setText(_t("debrief_hint_label", lang))

        # Combo boxes: refresh the localised item texts without losing the
        # current selection (signals blocked to avoid a filter round-trip).
        self._category_combo.blockSignals(True)
        self._category_combo.setItemText(0, _t("all_categories", lang))
        for i in range(1, self._category_combo.count()):
            code = self._category_combo.itemData(i)
            labels = CATEGORY_LABELS.get(code)
            if labels is not None:
                self._category_combo.setItemText(
                    i, f"{code} — {labels[0] if lang == 'fr' else labels[1]}"
                )
        self._category_combo.blockSignals(False)

        self._theme_combo.blockSignals(True)
        self._theme_combo.setItemText(0, _t("all_themes", lang))
        for i in range(1, self._theme_combo.count()):
            theme = self._theme_combo.itemData(i)
            labels = THEME_LABELS.get(theme, (theme, theme))
            self._theme_combo.setItemText(i, labels[0] if lang == "fr" else labels[1])
        self._theme_combo.blockSignals(False)
