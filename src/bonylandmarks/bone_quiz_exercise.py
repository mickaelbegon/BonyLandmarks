"""Bone quiz exercise — quiz students on anatomical landmarks using BodyParts3D meshes.

Two phases:
  Phase 1 — Individual bones: 8 random bones, show each bone with its landmarks,
             alternate between Placement and Identification modes.
  Phase 2 — Full skeleton: all bones shown simultaneously, 20 random landmarks
             across all bones, same two modes.

Architecture
------------
BoneQuizExercise(QWidget)
    exercise_complete = Signal()   # emitted after all phases and summary screen
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .bone_map import BONE_JOINTS, BONE_LABEL_FR, LANDMARK_BONE
from .landmarks_extended import LANDMARK_BY_CODE, LANDMARKS

# ─── Paths ────────────────────────────────────────────────────────────────────

_BONES_DIR = Path(__file__).parent / "data" / "bones"
_LM_POS_FILE = _BONES_DIR / "landmark_positions.json"

# ─── Colors ───────────────────────────────────────────────────────────────────

_BG = "#1a1a2e"
_CARD = "#252540"
_TEXT = "#e8e8f0"
_MUTED = "#a0a0c0"
_BLUE = "#7cb9ff"

_COLOR_BONE_PRIMARY = "#e8d5b0"
_COLOR_BONE_REF     = "#555577"
_COLOR_BONE_HILIGHT = "#ff8800"
_COLOR_SPHERE_CAND  = "#ffdd00"
_COLOR_SPHERE_REF   = "#ff2222"
_COLOR_SPHERE_OK    = "#00cc44"

_N_BONES_PHASE1 = 8
_N_LANDMARKS_PHASE2 = 20

_OPACITY_REF = 0.4

# ─── Style helpers ─────────────────────────────────────────────────────────────

_BASE_STYLE = f"""
QWidget {{
    background-color: {_BG};
    color: {_TEXT};
    font-family: "Segoe UI", Arial, sans-serif;
}}
"""

_BTN_PRIMARY = (
    f"QPushButton {{ background: #3a5faa; color: {_TEXT}; border: none; "
    f"border-radius: 6px; padding: 8px 14px; font-size: 13px; }}"
    f"QPushButton:hover {{ background: #4a70cc; }}"
    f"QPushButton:disabled {{ background: #2a2a44; color: #606080; }}"
)

_BTN_MCQ = (
    f"QPushButton {{ background: {_CARD}; color: {_TEXT}; border: 1px solid #3a3a6a; "
    f"border-radius: 6px; padding: 8px 10px; font-size: 12px; text-align: left; }}"
    f"QPushButton:hover {{ background: #303060; }}"
)

_BTN_MCQ_CORRECT = (
    "QPushButton { background: #1a4a1a; color: #60ff60; border: 1px solid #60ff60; "
    "border-radius: 6px; padding: 8px 10px; font-size: 12px; text-align: left; }"
)

_BTN_MCQ_WRONG = (
    "QPushButton { background: #4a1a1a; color: #ff6060; border: 1px solid #ff6060; "
    "border-radius: 6px; padding: 8px 10px; font-size: 12px; text-align: left; }"
)

_BTN_MCQ_ANSWER = (
    "QPushButton { background: #1a4a1a; color: #60ff60; border: 2px solid #60ff60; "
    "border-radius: 6px; padding: 8px 10px; font-size: 12px; font-weight: bold; text-align: left; }"
)

_BTN_QUIT = (
    "QPushButton { background: #3a1a1a; color: #ff8080; border: 1px solid #6a3a3a; "
    "border-radius: 5px; padding: 5px 10px; font-size: 11px; }"
    "QPushButton:hover { background: #5a2a2a; }"
)

_PROGRESS_BAR_STYLE = (
    "QFrame { background: #2a2a44; border: none; border-radius: 4px; }"
)

_SEP_STYLE = (
    "QFrame { color: #333355; background: #333355; }"
)


# ─── Data loading helpers ─────────────────────────────────────────────────────

_lm_pos_cache: dict[str, list[float]] | None = None


def _lm_positions() -> dict[str, list[float]]:
    global _lm_pos_cache
    if _lm_pos_cache is None:
        if _LM_POS_FILE.exists():
            _lm_pos_cache = json.loads(_LM_POS_FILE.read_text(encoding="utf-8"))
        else:
            _lm_pos_cache = {}
    return _lm_pos_cache


def _load_bone(stem: str) -> Optional[pv.PolyData]:
    """Load one bone mesh from data/bones/ (STL or OBJ). Returns None if missing."""
    for ext in (".stl", ".obj", ".STL", ".OBJ"):
        p = _BONES_DIR / f"{stem}{ext}"
        if p.exists():
            try:
                return pv.read(str(p))
            except Exception:
                return None
    return None


def _bone_has_mesh(stem: str) -> bool:
    for ext in (".stl", ".obj", ".STL", ".OBJ"):
        if (_BONES_DIR / f"{stem}{ext}").exists():
            return True
    return False


def _build_eligible_landmarks() -> list[str]:
    """Return codes of BONE-category landmarks that have position + mesh."""
    positions = _lm_positions()
    eligible: list[str] = []
    for lm in LANDMARKS:
        if lm.category != "BONE":
            continue
        if lm.code not in positions:
            continue
        stem = LANDMARK_BONE.get(lm.code)
        if stem is None:
            continue
        if not _bone_has_mesh(stem):
            continue
        eligible.append(lm.code)
    return eligible


def _group_by_bone(codes: list[str]) -> dict[str, list[str]]:
    """Map bone_stem → [landmark codes] for the given code list."""
    groups: dict[str, list[str]] = {}
    for code in codes:
        stem = LANDMARK_BONE.get(code)
        if stem:
            groups.setdefault(stem, []).append(code)
    return groups


def _distractors(correct_code: str, pool: list[str], n: int = 3) -> list[str]:
    """Pick *n* distractor landmark codes for a multiple-choice question.

    Priority:
    1. Other landmarks on the same bone stem.
    2. Landmarks from adjacent bones (BONE_JOINTS).
    3. Any other landmark in pool.
    """
    correct_stem = LANDMARK_BONE.get(correct_code)
    same_bone = [c for c in pool if c != correct_code and LANDMARK_BONE.get(c) == correct_stem]
    adjacent_stems: set[str] = set(BONE_JOINTS.get(correct_stem or "", []))
    adjacent = [c for c in pool if c != correct_code and LANDMARK_BONE.get(c) in adjacent_stems and c not in same_bone]
    rest = [c for c in pool if c != correct_code and c not in same_bone and c not in adjacent]

    chosen: list[str] = []
    for bucket in (same_bone, adjacent, rest):
        if len(chosen) >= n:
            break
        need = n - len(chosen)
        pick = random.sample(bucket, min(need, len(bucket)))
        chosen.extend(pick)

    return chosen[:n]


# ─── Score helpers ─────────────────────────────────────────────────────────────

def _grade(dist_mm: float) -> str:
    if dist_mm < 10:
        return "A"
    if dist_mm < 20:
        return "B"
    if dist_mm < 40:
        return "C"
    return "D"


def _grade_color(grade: str) -> str:
    return {"A": "#00cc44", "B": "#7cb9ff", "C": "#ffaa00", "D": "#ff4444"}.get(grade, _TEXT)


# ─── Main widget ──────────────────────────────────────────────────────────────

class BoneQuizExercise(QWidget):
    """Full quiz widget for two-phase bone landmark quizzing.

    Signals
    -------
    exercise_complete : emitted when the student dismisses the summary screen.
    """

    exercise_complete = Signal()

    def __init__(self, lang: str = "fr", parent=None) -> None:
        super().__init__(parent)
        self._lang = lang
        self.setStyleSheet(_BASE_STYLE)

        # ── Data ──────────────────────────────────────────────────────────────
        self._positions = _lm_positions()
        self._eligible: list[str] = _build_eligible_landmarks()
        self._bone_cache: dict[str, pv.PolyData] = {}

        # All unique stems that have mesh + at least one eligible landmark
        by_bone = _group_by_bone(self._eligible)
        self._all_stems: list[str] = list(by_bone.keys())

        # ── Session plan ──────────────────────────────────────────────────────
        # Phase 1: N_BONES_PHASE1 random bones, each with their landmarks
        p1_stems = random.sample(self._all_stems, min(_N_BONES_PHASE1, len(self._all_stems)))
        # For each phase-1 bone, collect its landmarks
        self._phase1_items: list[tuple[str, str]] = []  # (bone_stem, lm_code)
        for stem in p1_stems:
            for code in by_bone.get(stem, []):
                self._phase1_items.append((stem, code))

        # Phase 2: all bones, 20 random landmarks
        p2_codes = random.sample(self._eligible, min(_N_LANDMARKS_PHASE2, len(self._eligible)))
        self._phase2_items: list[tuple[str, str]] = [
            (stem2, c)
            for c in p2_codes
            if (stem2 := LANDMARK_BONE.get(c)) is not None
        ]

        # ── Quiz state ────────────────────────────────────────────────────────
        self._current_phase: int = 1       # 1 or 2
        self._current_idx: int = 0
        self._mode_seq: list[str] = []    # "placement" | "identification"
        self._current_mode: str = "placement"
        self._awaiting_pick: bool = False
        self._candidate_pos: Optional[np.ndarray] = None

        # Score tracking
        self._results: list[dict] = []  # {mode, code, grade/correct, dist_mm}

        # Picking state (must stay alive during observers)
        self._picker: vtk.vtkCellPicker | None = None
        self._press_xy: tuple[int, int] | None = None
        self._press_cb = None
        self._release_cb = None
        self._press_obs: int | None = None
        self._release_obs: int | None = None

        # Actors
        self._candidate_actor = None
        self._ref_sphere_actor = None

        self._build_ui()
        self._start_phase(1)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar
        root.addWidget(self._build_topbar())

        # Main split
        main = QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # PyVista plotter (70%)
        self._plotter = QtInteractor(self)
        self._plotter.set_background(_BG)
        self._plotter.enable_3_lights()
        main.addWidget(self._plotter.interactor, stretch=7)

        # Right panel (30%)
        self._right_panel = self._build_right_panel()
        main.addWidget(self._right_panel, stretch=3)

        root.addLayout(main, stretch=1)

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(f"background: {_CARD}; border-bottom: 1px solid #333355;")
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(12, 4, 12, 4)
        hl.setSpacing(12)

        self._phase_label = QLabel("Phase 1 / 2")
        self._phase_label.setStyleSheet(f"font-size: 13px; color: {_BLUE}; font-weight: bold;")
        hl.addWidget(self._phase_label)

        # Progress bar area
        prog_container = QWidget()
        prog_container.setFixedHeight(10)
        prog_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        prog_container.setStyleSheet("background: #2a2a44; border-radius: 5px;")
        self._prog_container = prog_container

        self._prog_fill = QFrame(prog_container)
        self._prog_fill.setStyleSheet(f"background: {_BLUE}; border-radius: 5px;")
        self._prog_fill.setGeometry(0, 0, 0, 10)
        hl.addWidget(prog_container)

        self._progress_text = QLabel("0/0")
        self._progress_text.setStyleSheet(f"font-size: 12px; color: {_MUTED};")
        hl.addWidget(self._progress_text)

        quit_btn = QPushButton("Quitter")
        quit_btn.setStyleSheet(_BTN_QUIT)
        quit_btn.clicked.connect(self.exercise_complete)
        hl.addWidget(quit_btn)

        return bar

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setStyleSheet(f"background: {_CARD}; border-left: 1px solid #333355;")

        vl = QVBoxLayout(panel)
        vl.setContentsMargins(12, 12, 12, 12)
        vl.setSpacing(8)

        # Landmark counter
        self._lm_counter = QLabel("Repère 1/1")
        self._lm_counter.setStyleSheet(f"font-size: 11px; color: {_MUTED};")
        vl.addWidget(self._lm_counter)

        # Mode badge
        self._mode_badge = QLabel("PLACEMENT")
        self._mode_badge.setAlignment(Qt.AlignCenter)
        self._mode_badge.setStyleSheet(
            "font-size: 10px; font-weight: bold; color: #1a1a2e; "
            "background: #ff8800; border-radius: 4px; padding: 2px 8px;"
        )
        vl.addWidget(self._mode_badge)

        # Bone name
        self._bone_name_lbl = QLabel("")
        self._bone_name_lbl.setStyleSheet(f"font-size: 11px; color: {_MUTED};")
        self._bone_name_lbl.setWordWrap(True)
        vl.addWidget(self._bone_name_lbl)

        # Landmark name
        self._lm_name_lbl = QLabel("")
        self._lm_name_lbl.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {_TEXT};")
        self._lm_name_lbl.setWordWrap(True)
        vl.addWidget(self._lm_name_lbl)

        # Hint (collapsible via toggle label)
        self._hint_toggle = QLabel("▶ Indice")
        self._hint_toggle.setStyleSheet(f"font-size: 11px; color: {_BLUE}; cursor: pointer;")
        self._hint_toggle.mousePressEvent = self._toggle_hint
        vl.addWidget(self._hint_toggle)

        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setVisible(False)
        self._hint_label.setStyleSheet(f"font-size: 11px; color: {_MUTED}; padding: 4px;")
        vl.addWidget(self._hint_label)

        vl.addWidget(self._make_sep())

        # Instruction label
        self._instruction_lbl = QLabel("")
        self._instruction_lbl.setWordWrap(True)
        self._instruction_lbl.setStyleSheet(f"font-size: 12px; color: {_TEXT};")
        vl.addWidget(self._instruction_lbl)

        # Confirm button (placement mode)
        self._confirm_btn = QPushButton("Confirmer")
        self._confirm_btn.setStyleSheet(_BTN_PRIMARY)
        self._confirm_btn.setVisible(False)
        self._confirm_btn.clicked.connect(self._on_confirm)
        vl.addWidget(self._confirm_btn)

        # MCQ buttons (identification mode)
        self._mcq_btns: list[QPushButton] = []
        self._mcq_codes: list[str] = []
        for i in range(4):
            btn = QPushButton("")
            btn.setStyleSheet(_BTN_MCQ)
            btn.setVisible(False)
            btn.setMinimumHeight(48)

            def _make_handler(j: int):
                return lambda: self._on_mcq_choice(j)

            btn.clicked.connect(_make_handler(i))
            self._mcq_btns.append(btn)
            vl.addWidget(btn)

        vl.addWidget(self._make_sep())

        # Result label
        self._result_lbl = QLabel("")
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setAlignment(Qt.AlignCenter)
        self._result_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_TEXT};")
        vl.addWidget(self._result_lbl)

        self._result_detail = QLabel("")
        self._result_detail.setWordWrap(True)
        self._result_detail.setAlignment(Qt.AlignCenter)
        self._result_detail.setStyleSheet(f"font-size: 11px; color: {_MUTED};")
        vl.addWidget(self._result_detail)

        vl.addStretch()

        return panel

    @staticmethod
    def _make_sep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet(_SEP_STYLE)
        return f

    # ── Phase management ──────────────────────────────────────────────────────

    def _start_phase(self, phase: int) -> None:
        self._current_phase = phase
        self._current_idx = 0
        self._candidate_pos = None
        self._awaiting_pick = False
        self._remove_pick_observers()

        if phase == 1:
            self._items = self._phase1_items
            self._phase_label.setText("Phase 1 / 2 — Os individuels")
        else:
            self._items = self._phase2_items
            self._phase_label.setText("Phase 2 / 2 — Squelette complet")

        if not self._items:
            # Skip to next phase or summary
            if phase == 1:
                self._start_phase(2)
            else:
                self._show_summary()
            return

        # Build mode sequence: alternate placement / identification
        modes: list[str] = []
        for i in range(len(self._items)):
            modes.append("placement" if i % 2 == 0 else "identification")
        self._mode_seq = modes

        self._load_landmark(0)

    def _load_landmark(self, idx: int) -> None:
        """Set up the 3D view and right panel for item at position idx."""
        self._current_idx = idx
        self._candidate_pos = None
        self._awaiting_pick = False
        self._result_lbl.setText("")
        self._result_detail.setText("")
        self._hint_label.setVisible(False)
        self._hint_toggle.setText("▶ Indice")
        self._remove_pick_observers()

        total = len(self._items)
        stem, code = self._items[idx]
        mode = self._mode_seq[idx]
        self._current_mode = mode

        # Update counters
        lm_total_for_bone = sum(1 for s, _ in self._items if s == stem)
        lm_idx_in_bone = sum(1 for i, (s, _) in enumerate(self._items) if s == stem and i <= idx)
        self._lm_counter.setText(f"Repère {idx + 1}/{total}")
        self._progress_text.setText(f"{idx + 1}/{total}")
        # Update progress bar
        QTimer.singleShot(0, lambda: self._update_progress_bar(idx + 1, total))

        # Mode badge
        if mode == "placement":
            self._mode_badge.setText("PLACEMENT")
            self._mode_badge.setStyleSheet(
                "font-size: 10px; font-weight: bold; color: #1a1a2e; "
                "background: #ff8800; border-radius: 4px; padding: 2px 8px;"
            )
        else:
            self._mode_badge.setText("IDENTIFICATION")
            self._mode_badge.setStyleSheet(
                "font-size: 10px; font-weight: bold; color: #1a1a2e; "
                "background: #7cb9ff; border-radius: 4px; padding: 2px 8px;"
            )

        # Bone and landmark names
        bone_label = BONE_LABEL_FR.get(stem, stem)
        self._bone_name_lbl.setText(bone_label)

        lm_obj = LANDMARK_BY_CODE.get(code)
        lm_name = lm_obj.name(self._lang) if lm_obj else code
        lm_hint = lm_obj.hint(self._lang) if lm_obj else ""

        self._lm_name_lbl.setText(lm_name)
        self._hint_label.setText(lm_hint)

        # ── 3D view ──────────────────────────────────────────────────────────
        self._plotter.clear()
        self._candidate_actor = None
        self._ref_sphere_actor = None

        if self._current_phase == 1:
            self._render_single_bone(stem, code, mode)
        else:
            self._render_full_skeleton(stem, code, mode)

        self._plotter.reset_camera()
        self._plotter.render()

        # ── Right panel controls ──────────────────────────────────────────────
        self._confirm_btn.setVisible(False)
        self._confirm_btn.setEnabled(False)
        for btn in self._mcq_btns:
            btn.setVisible(False)
            btn.setEnabled(True)
            btn.setStyleSheet(_BTN_MCQ)

        if mode == "placement":
            if self._lang == "fr":
                self._instruction_lbl.setText(f"Cliquez sur le repère : {lm_name}")
            else:
                self._instruction_lbl.setText(f"Click on the landmark: {lm_name}")
            self._confirm_btn.setVisible(True)
            self._confirm_btn.setEnabled(False)
            self._arm_pick_observers()
            self._awaiting_pick = True
        else:
            ref_pos = self._positions.get(code)
            if ref_pos is not None:
                r = self._sphere_radius(stem)
                sph = pv.Sphere(radius=r, center=ref_pos)
                self._ref_sphere_actor = self._plotter.add_mesh(
                    sph, color=_COLOR_SPHERE_REF,
                    ambient=1.0, diffuse=0.0, specular=0.0,
                    show_scalar_bar=False, render=False, pickable=False,
                )
                self._plotter.render()

            if self._lang == "fr":
                self._instruction_lbl.setText("Quel est ce repère ? (sphère rouge)")
            else:
                self._instruction_lbl.setText("Which landmark is this? (red sphere)")

            # Build MCQ options
            pool = list(self._positions.keys())
            distractors = _distractors(code, pool, n=3)
            options = [code] + distractors
            random.shuffle(options)
            self._mcq_codes = options

            for i, opt_code in enumerate(options):
                lm_opt = LANDMARK_BY_CODE.get(opt_code)
                label = lm_opt.name(self._lang) if lm_opt else opt_code
                self._mcq_btns[i].setText(label)
                self._mcq_btns[i].setStyleSheet(_BTN_MCQ)
                self._mcq_btns[i].setVisible(True)
                self._mcq_btns[i].setEnabled(True)

    def _update_progress_bar(self, done: int, total: int) -> None:
        if total <= 0:
            return
        w = self._prog_container.width()
        fill_w = int(w * done / total)
        self._prog_fill.setGeometry(0, 0, fill_w, 10)

    # ── 3D rendering helpers ──────────────────────────────────────────────────

    def _get_bone(self, stem: str) -> Optional[pv.PolyData]:
        if stem not in self._bone_cache:
            mesh = _load_bone(stem)
            if mesh is None:
                return None
            self._bone_cache[stem] = mesh
        return self._bone_cache[stem]

    def _sphere_radius(self, stem: str) -> float:
        mesh = self._get_bone(stem)
        if mesh is None:
            return 8.0
        return float(mesh.length) * 0.015

    def _render_single_bone(self, primary_stem: str, code: str, mode: str) -> None:
        """Phase 1: show one bone in natural bone color."""
        mesh = self._get_bone(primary_stem)
        if mesh is None:
            return

        self._plotter.add_mesh(
            mesh, color=_COLOR_BONE_PRIMARY, smooth_shading=True,
            ambient=0.3, diffuse=0.9, specular=0.2,
            show_scalar_bar=False, render=False,
            pickable=(mode == "placement"),
        )

        # Show all other landmarks on this bone as small gray spheres
        if mode == "identification":
            r = self._sphere_radius(primary_stem)
            for lm_code, pos in self._positions.items():
                stem_lm = LANDMARK_BONE.get(lm_code)
                if stem_lm != primary_stem or lm_code == code:
                    continue
                sph = pv.Sphere(radius=r * 0.5, center=pos)
                self._plotter.add_mesh(
                    sph, color="#444466", ambient=0.8, diffuse=0.2,
                    show_scalar_bar=False, render=False, pickable=False,
                )

    def _render_full_skeleton(self, primary_stem: str, code: str, mode: str) -> None:
        """Phase 2: show all bones dim, highlight primary bone."""
        for stem in self._all_stems:
            mesh = self._get_bone(stem)
            if mesh is None:
                continue
            is_primary = stem == primary_stem
            if is_primary:
                color = _COLOR_BONE_PRIMARY
                opacity = 1.0
            else:
                color = _COLOR_BONE_REF
                opacity = _OPACITY_REF
            self._plotter.add_mesh(
                mesh, color=color, smooth_shading=True,
                ambient=0.25, diffuse=0.8, specular=0.1,
                opacity=opacity, show_scalar_bar=False, render=False,
                pickable=(mode == "placement"),
            )

    # ── VTK picking (placement mode) ──────────────────────────────────────────

    def _arm_pick_observers(self) -> None:
        """Attach LeftButtonPress/Release VTK observers for clean-click picking."""
        iren = self._plotter.iren
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.005)
        self._press_xy = None

        def _on_press(caller, event):
            self._press_xy = iren.get_event_position()

        def _on_release(caller, event):
            if self._press_xy is None or not self._awaiting_pick:
                return
            rx, ry = iren.get_event_position()
            px, py = self._press_xy
            self._press_xy = None
            if (rx - px) ** 2 + (ry - py) ** 2 > 25:
                return  # drag, not click
            self._picker.Pick(rx, ry, 0, self._plotter.renderer)
            if self._picker.GetCellId() >= 0:
                pos = np.array(self._picker.GetPickPosition(), dtype=float)
                self._on_placement_pick(pos)

        self._press_cb = _on_press
        self._release_cb = _on_release
        self._press_obs = iren.add_observer("LeftButtonPressEvent", self._press_cb)
        self._release_obs = iren.add_observer("LeftButtonReleaseEvent", self._release_cb)

    def _remove_pick_observers(self) -> None:
        """Detach VTK observers (if any) and reset picking state."""
        if not hasattr(self, "_plotter") or self._plotter is None:
            return
        try:
            iren = self._plotter.iren
            for attr in ("_press_obs", "_release_obs"):
                obs_id = getattr(self, attr, None)
                if obs_id is not None:
                    try:
                        iren.remove_observer(obs_id)
                    except Exception:
                        pass
                    setattr(self, attr, None)
        except Exception:
            pass
        self._press_cb = None
        self._release_cb = None
        self._picker = None
        self._press_xy = None

    def _on_placement_pick(self, world_pos: np.ndarray) -> None:
        """Called when student picks a point on the mesh in placement mode."""
        self._candidate_pos = world_pos

        # Remove previous candidate sphere
        if self._candidate_actor is not None:
            try:
                self._plotter.remove_actor(self._candidate_actor)
            except Exception:
                pass
            self._candidate_actor = None

        stem, code = self._items[self._current_idx]
        r = self._sphere_radius(stem)
        sph = pv.Sphere(radius=r, center=world_pos)
        self._candidate_actor = self._plotter.add_mesh(
            sph, color=_COLOR_SPHERE_CAND,
            ambient=1.0, diffuse=0.0, specular=0.0,
            show_scalar_bar=False, render=False, pickable=False,
        )
        self._plotter.render()
        self._confirm_btn.setEnabled(True)

    # ── Confirmation and scoring ──────────────────────────────────────────────

    def _on_confirm(self) -> None:
        """Student pressed Confirm in placement mode."""
        if self._candidate_pos is None:
            return
        self._awaiting_pick = False
        self._remove_pick_observers()

        stem, code = self._items[self._current_idx]
        ref_pos = self._positions.get(code)
        if ref_pos is None:
            self._advance()
            return

        ref = np.asarray(ref_pos, dtype=float)
        dist = float(np.linalg.norm(self._candidate_pos - ref))
        grade = _grade(dist)
        color = _grade_color(grade)

        # Show green sphere at reference position
        r = self._sphere_radius(stem)
        if self._candidate_actor is not None:
            try:
                self._plotter.remove_actor(self._candidate_actor)
            except Exception:
                pass
            self._candidate_actor = None

        sph_cand = pv.Sphere(radius=r, center=self._candidate_pos)
        self._plotter.add_mesh(
            sph_cand, color=_COLOR_SPHERE_CAND,
            ambient=1.0, diffuse=0.0, specular=0.0,
            show_scalar_bar=False, render=False, pickable=False,
        )
        sph_ref = pv.Sphere(radius=r * 0.8, center=ref_pos)
        self._plotter.add_mesh(
            sph_ref, color=_COLOR_SPHERE_OK,
            ambient=1.0, diffuse=0.0, specular=0.0,
            show_scalar_bar=False, render=False, pickable=False,
        )
        self._plotter.render()

        self._result_lbl.setText(f"Note : {grade}")
        self._result_lbl.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color};"
        )
        dist_str = f"{dist:.1f} mm"
        self._result_detail.setText(dist_str)
        self._confirm_btn.setEnabled(False)

        self._results.append({"mode": "placement", "code": code, "grade": grade, "dist_mm": dist})

        QTimer.singleShot(2000, self._advance)

    def _on_mcq_choice(self, btn_idx: int) -> None:
        """Student selected a MCQ option in identification mode."""
        if btn_idx >= len(self._mcq_codes):
            return

        chosen_code = self._mcq_codes[btn_idx]
        _, correct_code = self._items[self._current_idx]
        is_correct = chosen_code == correct_code

        # Disable all buttons
        for btn in self._mcq_btns:
            btn.setEnabled(False)

        # Style chosen button and reveal correct
        for i, btn in enumerate(self._mcq_btns):
            if not btn.isVisible():
                continue
            if self._mcq_codes[i] == correct_code:
                if is_correct and i == btn_idx:
                    btn.setStyleSheet(_BTN_MCQ_CORRECT)
                else:
                    btn.setStyleSheet(_BTN_MCQ_ANSWER)
            elif i == btn_idx:
                btn.setStyleSheet(_BTN_MCQ_WRONG)

        if is_correct:
            self._result_lbl.setText("Correct !" if self._lang == "fr" else "Correct!")
            self._result_lbl.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {_COLOR_SPHERE_OK};")
        else:
            lm_obj = LANDMARK_BY_CODE.get(correct_code)
            correct_name = lm_obj.name(self._lang) if lm_obj else correct_code
            self._result_lbl.setText("Incorrect" if self._lang == "fr" else "Wrong")
            self._result_lbl.setStyleSheet(f"font-size: 16px; font-weight: bold; color: #ff4444;")
            self._result_detail.setText(
                f"Réponse : {correct_name}" if self._lang == "fr" else f"Answer: {correct_name}"
            )

        self._results.append({
            "mode": "identification",
            "code": correct_code,
            "correct": is_correct,
        })

        QTimer.singleShot(1500, self._advance)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _advance(self) -> None:
        """Move to next landmark or next phase."""
        next_idx = self._current_idx + 1
        if next_idx < len(self._items):
            self._load_landmark(next_idx)
        elif self._current_phase == 1:
            self._start_phase(2)
        else:
            self._show_summary()

    # ── Hint toggle ───────────────────────────────────────────────────────────

    def _toggle_hint(self, event=None) -> None:
        visible = not self._hint_label.isVisible()
        self._hint_label.setVisible(visible)
        self._hint_toggle.setText(("▼" if visible else "▶") + " Indice")

    # ── Summary screen ────────────────────────────────────────────────────────

    def _show_summary(self) -> None:
        """Replace content with a summary and Terminé button."""
        self._remove_pick_observers()
        self._plotter.clear()
        self._plotter.render()

        # Count results
        total = len(self._results)
        placement_grades = {"A": 0, "B": 0, "C": 0, "D": 0}
        id_correct = 0
        id_total = 0
        for r in self._results:
            if r["mode"] == "placement":
                g = r.get("grade", "D")
                placement_grades[g] = placement_grades.get(g, 0) + 1
            else:
                id_total += 1
                if r.get("correct"):
                    id_correct += 1

        pl_total = sum(placement_grades.values())

        lines = []
        if self._lang == "fr":
            lines.append(f"<b>Exercice terminé !</b><br>")
            lines.append(f"Total de repères : <b>{total}</b><br><br>")
            if pl_total:
                lines.append("<b>Placement :</b><br>")
                for g in ("A", "B", "C", "D"):
                    if placement_grades[g]:
                        col = _grade_color(g)
                        lines.append(f"&nbsp;&nbsp;<span style='color:{col}'>{g}</span> : {placement_grades[g]}<br>")
            if id_total:
                pct = int(100 * id_correct / id_total)
                lines.append(f"<br><b>Identification :</b> {id_correct}/{id_total} ({pct}%)<br>")
        else:
            lines.append(f"<b>Exercise complete!</b><br>")
            lines.append(f"Total landmarks: <b>{total}</b><br><br>")
            if pl_total:
                lines.append("<b>Placement:</b><br>")
                for g in ("A", "B", "C", "D"):
                    if placement_grades[g]:
                        col = _grade_color(g)
                        lines.append(f"&nbsp;&nbsp;<span style='color:{col}'>{g}</span>: {placement_grades[g]}<br>")
            if id_total:
                pct = int(100 * id_correct / id_total)
                lines.append(f"<br><b>Identification:</b> {id_correct}/{id_total} ({pct}%)<br>")

        # Replace right panel content
        for i in reversed(range(self._right_panel.layout().count())):
            item = self._right_panel.layout().itemAt(i)
            if item.widget():
                item.widget().setParent(None)

        vl = self._right_panel.layout()

        summary_lbl = QLabel("".join(lines))
        summary_lbl.setTextFormat(Qt.RichText)
        summary_lbl.setWordWrap(True)
        summary_lbl.setAlignment(Qt.AlignTop)
        summary_lbl.setStyleSheet(f"font-size: 13px; color: {_TEXT}; padding: 8px;")
        vl.addWidget(summary_lbl)

        vl.addStretch()

        done_btn = QPushButton("Terminé" if self._lang == "fr" else "Done")
        done_btn.setStyleSheet(_BTN_PRIMARY)
        done_btn.clicked.connect(self.exercise_complete)
        vl.addWidget(done_btn)

        # Update topbar
        self._phase_label.setText("Terminé !" if self._lang == "fr" else "Done!")
        total_all = len(self._phase1_items) + len(self._phase2_items)
        self._update_progress_bar(total_all, total_all)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release VTK resources. Call before destroying the widget."""
        self._remove_pick_observers()
        try:
            self._plotter.close()
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
