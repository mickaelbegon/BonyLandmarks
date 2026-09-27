"""Shared-scan free-placement exercise widget.

All students work on the **same** GLB scan (no individual scans required).
Landmarks have no reference position (SKINFOLD, ANTHRO, EMG and any BONE code
absent from ``landmark_positions.json``).

After submitting, placements are saved as JSON and the widget loads peer
submissions from a shared folder to show a non-evaluative distribution overlay.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .landmarks_extended import LANDMARKS, CATEGORY_LABELS, Landmark

if TYPE_CHECKING:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────

_LM_POS_FILE = Path(__file__).parent / "data" / "bones" / "landmark_positions.json"

# ── Style constants ───────────────────────────────────────────────────────────

_BG = "#1a1a2e"
_CARD = "#252540"
_TEXT = "#e8e8f0"
_MUTED = "#a0a0c0"
_BLUE = "#7cb9ff"

_CAT_COLORS: dict[str, str] = {
    "SKINFOLD": "#b0f0c0",   # mint green
    "ANTHRO":   "#e0b0ff",   # lavender
    "EMG":      "#ffd0a0",   # peach
    "BONE":     "#b0d0ff",   # light blue
}

# Peer sphere colors (by std-dev agreement)
_PEER_GREEN  = "#00cc44"   # std < 15 mm
_PEER_ORANGE = "#ffaa00"   # std < 40 mm
_PEER_RED    = "#cc2222"   # std >= 40 mm

_BTN_PRIMARY = (
    "QPushButton { background: #3399ff; color: white; border: none; "
    "border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: bold; }"
    "QPushButton:hover { background: #4aa5ff; }"
    "QPushButton:disabled { background: rgba(255,255,255,0.08); color: #777; }"
)
_BTN_SECONDARY = (
    "QPushButton { background: rgba(255,255,255,0.07); color: #ccc; "
    "border: 1px solid rgba(255,255,255,0.15); border-radius: 6px; "
    "padding: 5px 12px; font-size: 11px; }"
    "QPushButton:hover { background: rgba(255,255,255,0.14); }"
    "QPushButton:disabled { background: rgba(255,255,255,0.04); color: #555; }"
)
_LABEL_CARD = (
    f"background: {_CARD}; color: {_TEXT}; border-radius: 6px; padding: 6px 8px; "
    "font-size: 12px;"
)

# ── Helper: get open (no-reference) landmarks ─────────────────────────────────

def _get_open_landmarks() -> list[Landmark]:
    """Return landmarks that have no entry in landmark_positions.json."""
    try:
        lm_pos: dict = json.loads(_LM_POS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        lm_pos = {}
    return [lm for lm in LANDMARKS if lm.code not in lm_pos]


def _sort_key(lm: Landmark) -> tuple[str, str]:
    cat_order = {"BONE": "0", "EMG": "1", "SKINFOLD": "2", "ANTHRO": "3"}
    return (cat_order.get(lm.category, "9"), lm.code)


# ── Main widget ───────────────────────────────────────────────────────────────

class SharedScanExercise(QWidget):
    """Free-placement widget on a shared GLB scan with peer comparison.

    Parameters
    ----------
    glb_bytes:
        Raw bytes of the GLB file (mesh_3d or avatar_3d format).
    scan_name:
        Logical name for the scan; used as a prefix for saved JSON files.
    lang:
        UI language: ``"fr"`` or ``"en"``.
    submissions_dir:
        Folder where placement JSON files are read/written.  When ``None``
        peer comparison is unavailable and saving is skipped.
    parent:
        Optional Qt parent widget.
    """

    exercise_complete = Signal()

    # Candidate sphere name (single, replaced each pick)
    _CAND_ACTOR = "ss_candidate"

    def __init__(
        self,
        glb_bytes: bytes | None,
        scan_name: str = "scan_partage",
        lang: str = "fr",
        submissions_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._lang = lang
        self._scan_name = scan_name
        self._submissions_dir = Path(submissions_dir) if submissions_dir else None

        # Placement state
        self._open_landmarks: list[Landmark] = sorted(_get_open_landmarks(), key=_sort_key)
        self._placements: dict[str, np.ndarray] = {}   # code -> xyz_mm
        self._candidate: np.ndarray | None = None
        self._active_lm: Landmark | None = None
        self._submitted: bool = False

        # Peer-overlay actor names (to remove on re-render)
        self._peer_actors: list[str] = []
        self._own_actors: list[str] = []
        self._placed_actors: dict[str, str] = {}   # code -> actor name

        self._mesh: pv.PolyData | None = None
        self._vertex_colors: np.ndarray | None = None

        self._build_ui()
        self._load_mesh(glb_bytes)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(
            f"QWidget {{ background: {_BG}; color: {_TEXT}; "
            "font-family: 'Segoe UI', sans-serif; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setContentsMargins(10, 6, 10, 6)
        top.setSpacing(8)

        title_text = (
            "Scan partagé — Placement libre"
            if self._lang == "fr"
            else "Shared scan — Free placement"
        )
        title = QLabel(title_text)
        title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_TEXT};")
        top.addWidget(title, stretch=1)

        self._lang_btn = QPushButton("EN" if self._lang == "fr" else "FR")
        self._lang_btn.setFixedWidth(44)
        self._lang_btn.setStyleSheet(_BTN_SECONDARY)
        self._lang_btn.clicked.connect(self._toggle_lang)
        top.addWidget(self._lang_btn)

        quit_btn = QPushButton("Quitter" if self._lang == "fr" else "Quit")
        quit_btn.setStyleSheet(_BTN_SECONDARY)
        quit_btn.clicked.connect(self.exercise_complete)
        top.addWidget(quit_btn)

        top_widget = QWidget()
        top_widget.setLayout(top)
        top_widget.setStyleSheet(
            f"background: {_CARD}; "
            "border-bottom: 1px solid rgba(255,255,255,0.08);"
        )
        root.addWidget(top_widget)

        # ── Main area: 3D + right panel ───────────────────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # 3D plotter
        self._plotter = QtInteractor(self)
        self._plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self._plotter.interactor, stretch=7)

        # Right panel
        right_widget = QWidget()
        right_widget.setFixedWidth(280)
        right_widget.setStyleSheet(
            f"background: {_CARD}; "
            "border-left: 1px solid rgba(255,255,255,0.08);"
        )
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(10, 10, 10, 10)
        right.setSpacing(8)

        # Category filter
        cat_lbl = QLabel("Catégorie" if self._lang == "fr" else "Category")
        cat_lbl.setStyleSheet(
            f"font-size: 10px; color: {_MUTED}; font-weight: bold;"
        )
        right.addWidget(cat_lbl)

        self._cat_combo = QComboBox()
        self._cat_combo.setStyleSheet(
            "QComboBox { background: rgba(255,255,255,0.07); color: #ddd; "
            "border: 1px solid rgba(255,255,255,0.12); border-radius: 4px; "
            "padding: 3px 6px; font-size: 11px; }"
        )
        all_label = "Toutes" if self._lang == "fr" else "All"
        self._cat_combo.addItem(all_label, None)
        cats_seen: list[str] = []
        for lm in self._open_landmarks:
            if lm.category not in cats_seen:
                cats_seen.append(lm.category)
        for cat in cats_seen:
            labels = CATEGORY_LABELS.get(cat, (cat, cat))
            label = labels[0] if self._lang == "fr" else labels[1]
            self._cat_combo.addItem(label, cat)
        self._cat_combo.currentIndexChanged.connect(self._refresh_list)
        right.addWidget(self._cat_combo)

        # Landmark list
        self._lm_list = QListWidget()
        self._lm_list.setStyleSheet(
            "QListWidget { background: rgba(255,255,255,0.04); "
            "border: 1px solid rgba(255,255,255,0.09); "
            "border-radius: 4px; font-size: 11px; color: #ddd; }"
            "QListWidget::item:selected { background: rgba(51,153,255,0.35); }"
            "QListWidget::item:hover { background: rgba(255,255,255,0.07); }"
        )
        self._lm_list.currentItemChanged.connect(self._on_list_selection)
        right.addWidget(self._lm_list, stretch=1)

        self._refresh_list()

        # Active landmark info
        sep_lbl = QLabel("—" * 20)
        sep_lbl.setStyleSheet(f"color: rgba(255,255,255,0.12); font-size: 9px;")
        right.addWidget(sep_lbl)

        active_lbl_caption = QLabel(
            "Repère actif" if self._lang == "fr" else "Active landmark"
        )
        active_lbl_caption.setStyleSheet(
            f"font-size: 10px; color: {_MUTED}; font-weight: bold;"
        )
        right.addWidget(active_lbl_caption)

        self._active_lm_label = QLabel("—")
        self._active_lm_label.setWordWrap(True)
        self._active_lm_label.setStyleSheet(_LABEL_CARD)
        right.addWidget(self._active_lm_label)

        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet(
            f"font-size: 11px; color: {_MUTED}; background: transparent;"
        )
        right.addWidget(self._hint_label)

        # Placement buttons
        btn_row = QHBoxLayout()
        self._place_btn = QPushButton(
            "Placer" if self._lang == "fr" else "Place"
        )
        self._place_btn.setStyleSheet(_BTN_PRIMARY)
        self._place_btn.setEnabled(False)
        self._place_btn.clicked.connect(self._on_place)
        btn_row.addWidget(self._place_btn)

        self._erase_btn = QPushButton(
            "Effacer" if self._lang == "fr" else "Erase"
        )
        self._erase_btn.setStyleSheet(_BTN_SECONDARY)
        self._erase_btn.setEnabled(False)
        self._erase_btn.clicked.connect(self._on_erase)
        btn_row.addWidget(self._erase_btn)
        right.addLayout(btn_row)

        # Progress counter
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"font-size: 11px; color: {_MUTED};"
        )
        self._progress_label.setAlignment(Qt.AlignCenter)
        right.addWidget(self._progress_label)
        self._update_progress()

        # Matricule entry
        mat_lbl = QLabel(
            "Matricule (optionnel)"
            if self._lang == "fr"
            else "Student ID (optional)"
        )
        mat_lbl.setStyleSheet(
            f"font-size: 10px; color: {_MUTED}; font-weight: bold;"
        )
        right.addWidget(mat_lbl)

        self._matricule_edit = QLineEdit()
        self._matricule_edit.setPlaceholderText("anonymous")
        self._matricule_edit.setStyleSheet(
            "QLineEdit { background: rgba(255,255,255,0.07); color: #ddd; "
            "border: 1px solid rgba(255,255,255,0.12); border-radius: 4px; "
            "padding: 4px 6px; font-size: 11px; }"
        )
        right.addWidget(self._matricule_edit)

        # Submit button
        self._submit_btn = QPushButton(
            "Soumettre et comparer"
            if self._lang == "fr"
            else "Submit and compare"
        )
        self._submit_btn.setStyleSheet(_BTN_PRIMARY)
        self._submit_btn.clicked.connect(self._on_submit)
        right.addWidget(self._submit_btn)

        # Peer comparison status
        self._peer_status_label = QLabel("")
        self._peer_status_label.setWordWrap(True)
        self._peer_status_label.setStyleSheet(
            f"font-size: 10px; color: {_MUTED};"
        )
        right.addWidget(self._peer_status_label)

        body_widget = QWidget()
        body_widget.setLayout(body)
        root.addWidget(body_widget, stretch=1)

    def _refresh_list(self) -> None:
        """Populate the landmark list according to the active category filter."""
        from PySide6.QtGui import QColor  # local import to avoid top-level Qt dep at module load

        cat_filter: str | None = self._cat_combo.currentData()
        self._lm_list.clear()
        for lm in self._open_landmarks:
            if cat_filter is not None and lm.category != cat_filter:
                continue
            name = lm.name(self._lang)
            placed = lm.code in self._placements
            prefix = "[x] " if placed else "[ ] "
            item = QListWidgetItem(prefix + name)
            item.setData(Qt.UserRole, lm.code)
            color_hex = _CAT_COLORS.get(lm.category, "#ccc")
            item.setForeground(
                QColor(color_hex if placed else "#aaaaaa")
            )
            self._lm_list.addItem(item)

    def _update_progress(self) -> None:
        n_placed = len(self._placements)
        n_total = len(self._open_landmarks)
        if self._lang == "fr":
            self._progress_label.setText(f"Places : {n_placed}/{n_total}")
        else:
            self._progress_label.setText(f"Placed: {n_placed}/{n_total}")

    # ── Mesh loading ──────────────────────────────────────────────────────────

    def _load_mesh(self, glb_bytes: bytes | None) -> None:
        """Load the GLB mesh and initialise the 3D scene."""
        if glb_bytes is None:
            self._show_error(
                "Aucun scan fourni."
                if self._lang == "fr"
                else "No scan provided."
            )
            return

        try:
            from .mesh_loader import load_avatar_glb
            body_mesh, vertex_colors, _, _ = load_avatar_glb(
                glb_bytes, remove_stickers=False
            )
            self._mesh = body_mesh
            self._vertex_colors = vertex_colors
        except Exception:
            # Fallback: try plain GLB mesh load (mesh_3d type)
            try:
                from .mesh_loader import load_glb_mesh
                body_mesh, vertex_colors = load_glb_mesh(glb_bytes)
                self._mesh = body_mesh
                self._vertex_colors = vertex_colors
            except Exception as exc:
                self._show_error(
                    f"Impossible de charger le scan : {exc}"
                    if self._lang == "fr"
                    else f"Cannot load scan: {exc}"
                )
                return

        self._setup_scene()

    def _show_error(self, msg: str) -> None:
        lbl = QLabel(msg)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 14px; color: #ff6060;")
        self.layout().addWidget(lbl)

    # ── 3D scene ───────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        if self._mesh is None:
            return

        pl = self._plotter
        pl.background_color = _BG
        pl.enable_3_lights()

        # Add body mesh with texture if available
        if self._vertex_colors is not None:
            rgba = self._vertex_colors
            # Use RGB channels (drop alpha if present)
            rgb = rgba[:, :3] if rgba.shape[1] == 4 else rgba
            self._mesh.point_data["colors"] = rgb
            pl.add_mesh(
                self._mesh,
                scalars="colors",
                rgb=True,
                smooth_shading=True,
                name="body_mesh",
            )
        else:
            pl.add_mesh(
                self._mesh,
                color="#c8b8a8",
                smooth_shading=True,
                ambient=0.3,
                diffuse=0.9,
                name="body_mesh",
            )

        # Enable surface picking
        pl.enable_surface_point_picking(
            callback=self._on_surface_pick,
            show_message=False,
            left_clicking=True,
            pickable_window=False,
        )
        pl.show()

        # Set initial camera to see full body
        bounds = self._mesh.bounds
        extents = [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ]
        up_axis = extents.index(max(extents))
        sorted_axes = sorted(range(3), key=lambda i: extents[i])
        front_axis = sorted_axes[0]
        centers = [
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        ]
        height = extents[up_axis]
        cam_distance = height * 2.5
        cam_pos = list(centers)
        cam_pos[front_axis] += cam_distance
        up_vec = [0.0, 0.0, 0.0]
        up_vec[up_axis] = 1.0

        cam = pl.camera
        cam.position = tuple(cam_pos)
        cam.focal_point = tuple(centers)
        cam.up = tuple(up_vec)
        pl.render()

    # ── Interaction ───────────────────────────────────────────────────────────

    def _on_surface_pick(self, point: np.ndarray) -> None:
        """Called by PyVista on mesh left-click."""
        if self._submitted:
            return  # no new picks after submission
        self._candidate = np.array(point, dtype=float)

        # Show candidate sphere (yellow)
        self._plotter.remove_actor(self._CAND_ACTOR, render=False)
        sphere = pv.Sphere(radius=10, center=self._candidate)
        self._plotter.add_mesh(sphere, color="#ffdd00", name=self._CAND_ACTOR)
        self._plotter.render()

        # Enable confirm button if a landmark is selected
        if self._active_lm is not None:
            self._place_btn.setEnabled(True)

    def _on_list_selection(
        self, current: QListWidgetItem | None, _prev: QListWidgetItem | None
    ) -> None:
        if current is None:
            self._active_lm = None
            self._active_lm_label.setText("—")
            self._hint_label.setText("")
            self._place_btn.setEnabled(False)
            self._erase_btn.setEnabled(False)
            return

        code: str = current.data(Qt.UserRole)
        lm = next((x for x in self._open_landmarks if x.code == code), None)
        if lm is None:
            return
        self._active_lm = lm
        self._active_lm_label.setText(lm.name(self._lang))
        self._hint_label.setText(lm.hint(self._lang))

        # Enable/disable erase based on whether it's placed
        self._erase_btn.setEnabled(code in self._placements)

        # Enable place if candidate exists and not yet submitted
        self._place_btn.setEnabled(
            self._candidate is not None and not self._submitted
        )

        # Orbit camera to placed position if already placed
        if code in self._placements and self._mesh is not None:
            self._orbit_to(self._placements[code])

    def _on_place(self) -> None:
        """Confirm the current candidate for the active landmark."""
        if self._candidate is None or self._active_lm is None:
            return

        lm = self._active_lm
        code = lm.code

        # Remove previous placed actor for this code (if any)
        prev_actor = self._placed_actors.get(code)
        if prev_actor:
            self._plotter.remove_actor(prev_actor, render=False)

        # Remove candidate sphere
        self._plotter.remove_actor(self._CAND_ACTOR, render=False)

        # Determine color by category
        color = _CAT_COLORS.get(lm.category, "#ffffff")

        # Add confirmed sphere
        sphere = pv.Sphere(radius=8, center=self._candidate)
        actor_name = f"placed_{code}"
        self._plotter.add_mesh(sphere, color=color, name=actor_name)
        self._plotter.render()

        self._placed_actors[code] = actor_name
        self._placements[code] = self._candidate.copy()
        self._candidate = None
        self._place_btn.setEnabled(False)
        self._erase_btn.setEnabled(True)

        self._update_progress()
        self._refresh_list()

    def _on_erase(self) -> None:
        """Erase the placement for the active landmark."""
        if self._active_lm is None:
            return
        code = self._active_lm.code
        if code not in self._placements:
            return

        actor = self._placed_actors.pop(code, None)
        if actor:
            self._plotter.remove_actor(actor, render=False)
            self._plotter.render()

        del self._placements[code]
        self._erase_btn.setEnabled(False)
        self._update_progress()
        self._refresh_list()

    def _orbit_to(self, point: np.ndarray) -> None:
        """Move camera focal point to *point* while keeping camera distance."""
        cam = self._plotter.camera
        old_pos = np.array(cam.position)
        old_focal = np.array(cam.focal_point)
        direction = old_pos - old_focal
        new_pos = point + direction
        cam.position = tuple(new_pos)
        cam.focal_point = tuple(point)
        self._plotter.render()

    # ── Language toggle ───────────────────────────────────────────────────────

    def _toggle_lang(self) -> None:
        self._lang = "en" if self._lang == "fr" else "fr"
        self._lang_btn.setText("EN" if self._lang == "fr" else "FR")
        self._refresh_list()
        self._update_progress()
        if self._active_lm:
            self._active_lm_label.setText(self._active_lm.name(self._lang))
            self._hint_label.setText(self._active_lm.hint(self._lang))

    # ── Submission & peer comparison ──────────────────────────────────────────

    def _on_submit(self) -> None:
        """Save placements and load peer comparison overlay."""
        if not self._placements:
            QMessageBox.information(
                self,
                "Aucun placement" if self._lang == "fr" else "No placements",
                "Veuillez placer au moins un repere avant de soumettre."
                if self._lang == "fr"
                else "Please place at least one landmark before submitting.",
            )
            return

        # Build JSON payload
        matricule = self._matricule_edit.text().strip() or "anonymous"
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        payload = {
            "scan_name": self._scan_name,
            "timestamp": timestamp,
            "matricule": matricule,
            "placements": {
                code: pos.tolist() for code, pos in self._placements.items()
            },
        }

        # Save to submissions dir
        if self._submissions_dir is not None:
            try:
                self._submissions_dir.mkdir(parents=True, exist_ok=True)
                ts_safe = timestamp.replace(":", "").replace("-", "")
                fname = f"{self._scan_name}_{matricule}_{ts_safe}.json"
                fpath = self._submissions_dir / fname
                fpath.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Erreur" if self._lang == "fr" else "Error",
                    (
                        f"Impossible de sauvegarder : {exc}"
                        if self._lang == "fr"
                        else f"Cannot save: {exc}"
                    ),
                )

        self._submitted = True
        self._submit_btn.setEnabled(False)
        self._place_btn.setEnabled(False)

        # Load peer data and show overlay
        peers = self._load_peer_submissions()
        if len(peers) <= 1:
            msg = (
                "Aucune soumission comparative disponible. "
                "Votre placement a ete sauvegarde."
                if self._lang == "fr"
                else "No comparative submission available. "
                "Your placement has been saved."
            )
            self._peer_status_label.setText(msg)
        else:
            self._show_peer_overlay(peers)
            n = len(peers)
            plural = "s" if n > 1 else ""
            if self._lang == "fr":
                msg = (
                    f"{n} etudiant{plural} ont soumis. "
                    "Votre placement vs. le groupe."
                )
            else:
                msg = (
                    f"{n} student{plural} submitted. "
                    "Your placement vs. the group."
                )
            self._peer_status_label.setText(msg)

    def _load_peer_submissions(self) -> list[dict]:
        """Load all JSON submission files for the current scan."""
        if self._submissions_dir is None:
            return []
        results: list[dict] = []
        for p in self._submissions_dir.glob(f"{self._scan_name}_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data.get("placements"), dict):
                    results.append(data)
            except Exception:
                continue
        return results

    def _show_peer_overlay(self, peers: list[dict]) -> None:
        """Compute per-landmark statistics across peer submissions and add spheres."""
        # Collect positions per landmark code from all peers
        per_code: dict[str, list[np.ndarray]] = {}
        for sub in peers:
            for code, pos in sub.get("placements", {}).items():
                if isinstance(pos, (list, tuple)) and len(pos) == 3:
                    per_code.setdefault(code, []).append(
                        np.array(pos, dtype=float)
                    )

        # Remove old peer actors
        for name in self._peer_actors:
            self._plotter.remove_actor(name, render=False)
        self._peer_actors.clear()

        for code, positions in per_code.items():
            if len(positions) < 2:
                continue
            pts = np.array(positions)
            centroid = pts.mean(axis=0)
            # Scalar std dev: mean of per-axis std deviations
            std_dev = float(pts.std(axis=0).mean())

            if std_dev < 15.0:
                color = _PEER_GREEN
            elif std_dev < 40.0:
                color = _PEER_ORANGE
            else:
                color = _PEER_RED

            # Semi-transparent peer centroid sphere
            peer_sphere = pv.Sphere(radius=12, center=centroid)
            peer_name = f"peer_{code}"
            self._plotter.add_mesh(
                peer_sphere, color=color, opacity=0.35, name=peer_name
            )
            self._peer_actors.append(peer_name)

        # Highlight this student's placements with a bright white sphere
        for code, pos in self._placements.items():
            own_sphere = pv.Sphere(radius=9, center=pos)
            own_name = f"own_{code}"
            self._plotter.add_mesh(
                own_sphere, color="white", opacity=1.0, name=own_name
            )
            self._own_actors.append(own_name)

        self._plotter.render()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Close the PyVista plotter. Call this before destroying the widget."""
        try:
            self._plotter.close()
        except Exception:
            pass
