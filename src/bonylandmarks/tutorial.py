"""Guided tour ("mode tutoriel") of the anatomical landmarks.

This is a **free exploration** mode shown *before* the graded session: the
student walks through the landmarks one by one, the camera orbits to face each
one, a green sphere shows where it sits, and the side panel reveals the full
palpation hint plus its clinical application.

Nothing is scored here — deliberately.  ``session.py`` is therefore **not**
imported: the tutorial owns a tiny cursor over a filtered landmark list and
knows nothing about grades, retries or ground-truth errors.

Pedagogical intent
------------------
* the theme badge contextualises the landmark (posture, EMG, ISAK...);
* the hint is shown *in full* (in the exercise it is only revealed afterwards);
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
_TUTORIAL_MARKER_COLOR = "#00cc44"   # green sphere on the ground truth
_BACKGROUND = "#1a1a2e"
_SPHERE_ACTOR = "tutorial_sphere"
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
    """Free-exploration guided tour of the landmark set (no scoring)."""

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
        cx = (bounds[0] + bounds[1]) * 0.5
        cy = (bounds[2] + bounds[3]) * 0.5
        cz = (bounds[4] + bounds[5]) * 0.5
        height = bounds[3] - bounds[2]
        cam = self._plotter.camera
        cam.position = (cx, cy, cz + height * 1.8)
        cam.focal_point = (cx, cy, cz)
        cam.up = (0.0, 1.0, 0.0)
        self._plotter.render()

    def _orbit_camera_to(self, point: np.ndarray) -> None:
        """Orbit the camera around the body axis to face *point*.

        Same logic as ``LandmarkViewer._orbit_camera_to``: the focus sits on the
        body axis at the landmark's height, and the camera is pushed out along
        the horizontal direction of the landmark (Z+ front view as fallback).
        """
        bounds = self._mesh.bounds
        body_cx = (bounds[0] + bounds[1]) * 0.5
        body_cz = (bounds[4] + bounds[5]) * 0.5
        focus = np.array([body_cx, point[1], body_cz], dtype=float)
        horiz = np.array([point[0] - body_cx, 0.0, point[2] - body_cz], dtype=float)
        horiz_norm = float(np.linalg.norm(horiz))
        if horiz_norm < 1.0:
            direction = np.array([0.0, 0.0, 1.0])  # fallback: front view
        else:
            direction = horiz / horiz_norm
        cam_pos = focus + direction * _CAM_DISTANCE
        cam = self._plotter.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(focus)
        cam.up = (0.0, 1.0, 0.0)
        self._plotter.render()

    # ── Presentation of the current landmark ─────────────────────────────────

    def _clear_sphere(self) -> None:
        if self._sphere_shown:
            self._plotter.remove_actor(_SPHERE_ACTOR, render=False)
            self._sphere_shown = False

    def _show_current(self) -> None:
        """Refresh the 3-D scene and the whole right panel for the cursor."""
        lm = self.current_landmark()
        self._clear_sphere()

        if lm is None:
            self._counter_label.setText(_t("empty", self._lang))
            self._name_label.setText("—")
            self._theme_badge.setVisible(False)
            self._status_label.setVisible(False)
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

        # Theme badge
        theme_label = lm.theme_label(self._lang)
        self._theme_badge.setText(theme_label)
        self._theme_badge.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: white; "
            f"background-color: {lm.theme_color()}; "
            "border-radius: 4px; padding: 2px 8px;"
        )
        self._theme_badge.setVisible(bool(theme_label))

        # Palpation hint (always shown in full — this is the teaching mode)
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
        gt = self._ground_truth.get(lm.code)
        if gt is not None:
            gt = np.asarray(gt, dtype=float)
            self._plotter.add_mesh(
                pv.Sphere(radius=_SPHERE_RADIUS, center=gt),
                color=_TUTORIAL_MARKER_COLOR,
                name=_SPHERE_ACTOR,
            )
            self._sphere_shown = True
            self._status_label.setVisible(False)
            self._orbit_camera_to(gt)
        else:
            # EMG / SKINFOLD / ANTHRO without a BodyLoop marker
            self._status_label.setText(_t("not_scored", self._lang))
            self._status_label.setVisible(True)
            self._plotter.render()

        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < total - 1)

    # ── Teardown ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release the VTK render window before the widget is replaced.

        ``QMainWindow.setCentralWidget`` deletes the previous widget; closing
        the plotter first avoids leaking the interactor/render window.
        """
        self._clear_sphere()
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
