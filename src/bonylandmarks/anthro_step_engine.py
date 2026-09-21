"""Step engine for interactive anthropometric exercises (exercice 4).

Defines step dataclasses, AnthroRecipe, and AnthroStepEngine.
Does NOT modify viewer.py or anthro_measures_exercise.py.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Union


# ── Helpers (mirrors of anthro_measures_exercise helpers) ─────────────────────

def _infer_up_axis(gt: dict) -> int:
    if not gt:
        return 1
    pts = np.array([np.asarray(v, dtype=np.float64) for v in gt.values()
                    if isinstance(v, (list, tuple, np.ndarray))])
    if len(pts) == 0:
        return 1
    extents = pts.max(axis=0) - pts.min(axis=0)
    return int(extents.argmax())


def _horizontal_axes(up: int) -> tuple[int, int]:
    axes = [i for i in range(3) if i != up]
    return axes[0], axes[1]


def _project_plane(v: np.ndarray, normal_axis: int) -> np.ndarray:
    v2 = v.copy()
    v2[normal_axis] = 0.0
    return v2


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (°) at vertex B."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle3_proj(a: np.ndarray, b: np.ndarray, c: np.ndarray, normal_axis: int) -> float:
    """Angle (°) at vertex B, projected onto plane ⊥ normal_axis."""
    a2 = _project_plane(a - b, normal_axis)
    c2 = _project_plane(c - b, normal_axis)
    n1 = np.linalg.norm(a2)
    n2 = np.linalg.norm(c2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.dot(a2, c2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_horiz(vec: np.ndarray, up: int) -> float:
    """Elevation angle (°) above horizontal."""
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return 0.0
    return float(np.degrees(np.arcsin(np.clip(float(vec[up] / n), -1.0, 1.0))))


def _angle_from_vertical(vec: np.ndarray, up: int) -> float:
    """Angle (°) from the vertical axis (0° = perfectly vertical)."""
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return 0.0
    v_up = np.zeros(3)
    v_up[up] = 1.0
    cos_a = float(np.dot(vec / n, v_up))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_frontal_from_vertical(vec: np.ndarray, up: int) -> float:
    """Angle (°) from vertical after projecting vec onto the frontal plane.

    Frontal plane = plane containing up axis and mediolateral axis
    (i.e., we remove the anteroposterior component).
    """
    ml, ap = _horizontal_axes(up)
    v_front = _project_plane(vec, ap)
    n = np.linalg.norm(v_front)
    if n < 1e-9:
        return 0.0
    v_up = np.zeros(3)
    v_up[up] = 1.0
    cos_a = float(np.dot(v_front / n, v_up))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_horiz_ap(vec: np.ndarray, up: int) -> float:
    """AP-inclination in horizontal plane (°): how much the horizontal
    projection tilts toward the AP axis vs the ML axis.

    Used for femoral anteversion clinical proxy.
    """
    ml, ap = _horizontal_axes(up)
    vec_horiz = _project_plane(vec, up)
    n = np.linalg.norm(vec_horiz)
    if n < 1e-9:
        return 0.0
    ap_comp = abs(float(vec_horiz[ap]))
    return float(np.degrees(np.arcsin(np.clip(ap_comp / n, -1.0, 1.0))))


# ── Step dataclasses ──────────────────────────────────────────────────────────

@dataclass
class PickLandmark:
    """Student picks a landmark from the ground-truth list."""
    kind: Literal["pick_landmark"] = "pick_landmark"
    instruction_fr: str = ""
    expected_code: str = ""    # landmark code in ground_truth
    result_name: str = ""      # key in workspace


@dataclass
class ComputeDistance:
    """3-D Euclidean distance between two workspace points."""
    kind: Literal["distance"] = "distance"
    instruction_fr: str = ""
    expected_a: str = ""       # workspace names
    expected_b: str = ""
    result_name: str = ""      # scalar (mm)


@dataclass
class ComputeMidpoint:
    """Midpoint between two workspace points."""
    kind: Literal["midpoint"] = "midpoint"
    instruction_fr: str = ""
    expected_a: str = ""
    expected_b: str = ""
    result_name: str = ""      # 3-D point


@dataclass
class ComputeAngle3Pts:
    """Angle at vertex B in triangle A-B-C (degrees).

    project_on: '3d' (full 3-D), 'frontal' (remove AP component),
                'sagittal' (remove ML component), 'transverse' (remove UP component).
    deviation_from_180: if True, returns max(0, 180 - angle) instead.
    """
    kind: Literal["angle_3pts"] = "angle_3pts"
    instruction_fr: str = ""
    expected_a: str = ""
    expected_vertex: str = ""
    expected_c: str = ""
    result_name: str = ""
    project_on: Literal["3d", "frontal", "sagittal", "transverse"] = "3d"
    deviation_from_180: bool = False


@dataclass
class ComputeAnglePlane:
    """Angle between segment A→B and a reference direction.

    plane options:
    - 'horizontal'   : elevation above horizontal (arcsin of up-component / norm)
    - 'vertical'     : angle from vertical axis (0° = perfectly vertical)
    - 'frontal'      : angle from vertical after projecting on frontal plane
    - 'horizontal_ap': AP inclination in horizontal plane (femoral anteversion)
    """
    kind: Literal["angle_plane"] = "angle_plane"
    instruction_fr: str = ""
    expected_a: str = ""       # start point
    expected_b: str = ""       # end point (direction = B - A)
    plane: Literal["horizontal", "vertical", "frontal", "horizontal_ap"] = "horizontal"
    result_name: str = ""


@dataclass
class ComputeAsymmetry:
    """Absolute height difference between two workspace items.

    If items are 3-D points: abs(left[up] - right[up]).
    If items are scalars: abs(left - right).
    """
    kind: Literal["asymmetry"] = "asymmetry"
    instruction_fr: str = ""
    expected_left: str = ""
    expected_right: str = ""
    result_name: str = ""


@dataclass
class ComputeAxisAngle:
    """Angle between two axes (A1→A2) and (B1→B2) projected in a plane.

    Used for tibial torsion (angle between condyle axis and bimalleolar axis
    in the transverse plane).
    """
    kind: Literal["axis_angle"] = "axis_angle"
    instruction_fr: str = ""
    expected_a1: str = ""
    expected_a2: str = ""
    expected_b1: str = ""
    expected_b2: str = ""
    plane: Literal["transverse", "frontal", "sagittal", "3d"] = "transverse"
    result_name: str = ""


@dataclass
class ComputeProjection:
    """Signed component of vector A→B along a world axis.

    axis: 'ap' (anteroposterior), 'ml' (mediolateral), 'vertical'.
    Returns a scalar in mm.
    """
    kind: Literal["projection"] = "projection"
    instruction_fr: str = ""
    expected_a: str = ""
    expected_b: str = ""
    axis: Literal["ap", "ml", "vertical"] = "ap"
    result_name: str = ""


AnthroStep = Union[
    PickLandmark,
    ComputeDistance,
    ComputeMidpoint,
    ComputeAngle3Pts,
    ComputeAnglePlane,
    ComputeAsymmetry,
    ComputeAxisAngle,
    ComputeProjection,
]


# ── Recipe dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnthroRecipe:
    code: str                       # short code, e.g. "LLL"
    name_fr: str
    steps: list[AnthroStep]
    expected_result_name: str       # final result key in workspace
    unit: str                       # "mm", "°", etc.
    interpretation_fr: str          # explanatory text shown at end
    measure_code: str = ""          # original code from anthro_measures_exercise


# ── Step engine ───────────────────────────────────────────────────────────────

class AnthroStepEngine:
    """Interactive step engine for a single anthropometric recipe.

    Usage::

        engine = AnthroStepEngine(recipe, ground_truth)
        while not engine.finished:
            step = engine.current_step
            show_ui(step.instruction_fr)
            selection = get_user_input()
            ok, msg = engine.try_execute(selection)
            show_feedback(ok, msg)
        result = engine.final_result()
    """

    def __init__(self, recipe: AnthroRecipe, ground_truth: dict) -> None:
        self._recipe = recipe
        self._gt = {k: np.asarray(v, dtype=np.float64)
                    for k, v in ground_truth.items()
                    if isinstance(v, (list, tuple, np.ndarray))}
        self._index: int = 0
        self._workspace: dict = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_step(self) -> AnthroStep:
        if self.finished:
            raise IndexError("Recipe finished — no current step.")
        return self._recipe.steps[self._index]

    @property
    def step_index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        return self._index >= len(self._recipe.steps)

    @property
    def workspace(self) -> dict:
        return dict(self._workspace)

    # ── Execution ─────────────────────────────────────────────────────────────

    def try_execute(self, selection: list[str]) -> tuple[bool, str]:
        """Attempt to execute the current step.

        For PickLandmark: *selection* = [landmark_code].
        For all Compute* steps: *selection* is ignored — the engine executes
        automatically using workspace items named in the step's fields.

        Returns (success, feedback_fr).
        """
        if self.finished:
            return False, "Exercice terminé."

        step = self.current_step
        up = _infer_up_axis(self._gt)

        # ── PickLandmark ──────────────────────────────────────────────────────
        if isinstance(step, PickLandmark):
            if not selection:
                return False, "Sélectionnez un repère dans la liste."
            chosen = selection[0]
            if chosen != step.expected_code:
                return False, (
                    f"Repère incorrect.\n"
                    f"Attendu : {step.expected_code}\n"
                    f"Sélectionné : {chosen}"
                )
            pt = self._gt.get(step.expected_code)
            if pt is None:
                return False, f"Repère « {step.expected_code} » absent des données du sujet."
            self._workspace[step.result_name] = pt
            self._index += 1
            return True, f"Correct ! Repère {step.expected_code} ajouté au workspace."

        # ── ComputeDistance ───────────────────────────────────────────────────
        if isinstance(step, ComputeDistance):
            a = self._workspace.get(step.expected_a)
            b = self._workspace.get(step.expected_b)
            if a is None or b is None:
                missing = [n for n, v in [(step.expected_a, a), (step.expected_b, b)] if v is None]
                return False, f"Élément(s) manquant(s) dans le workspace : {missing}"
            dist = float(np.linalg.norm(b - a))
            self._workspace[step.result_name] = dist
            self._index += 1
            return True, f"Distance calculée : {dist:.1f} mm"

        # ── ComputeMidpoint ───────────────────────────────────────────────────
        if isinstance(step, ComputeMidpoint):
            a = self._workspace.get(step.expected_a)
            b = self._workspace.get(step.expected_b)
            if a is None or b is None:
                missing = [n for n, v in [(step.expected_a, a), (step.expected_b, b)] if v is None]
                return False, f"Élément(s) manquant(s) dans le workspace : {missing}"
            mid = (a + b) / 2.0
            self._workspace[step.result_name] = mid
            self._index += 1
            return True, "Point milieu calculé."

        # ── ComputeAngle3Pts ──────────────────────────────────────────────────
        if isinstance(step, ComputeAngle3Pts):
            a = self._workspace.get(step.expected_a)
            v = self._workspace.get(step.expected_vertex)
            c = self._workspace.get(step.expected_c)
            missing = [n for n, val in [
                (step.expected_a, a), (step.expected_vertex, v), (step.expected_c, c)
            ] if val is None]
            if missing:
                return False, f"Élément(s) manquant(s) dans le workspace : {missing}"
            proj = step.project_on
            if proj == "3d":
                angle = _angle3(a, v, c)
            elif proj == "frontal":
                ml, ap = _horizontal_axes(up)
                angle = _angle3_proj(a, v, c, ap)
            elif proj == "sagittal":
                ml, ap = _horizontal_axes(up)
                angle = _angle3_proj(a, v, c, ml)
            else:  # transverse
                angle = _angle3_proj(a, v, c, up)
            if step.deviation_from_180:
                angle = max(0.0, 180.0 - angle)
            self._workspace[step.result_name] = angle
            self._index += 1
            return True, f"Angle calculé : {angle:.1f}°"

        # ── ComputeAnglePlane ─────────────────────────────────────────────────
        if isinstance(step, ComputeAnglePlane):
            a = self._workspace.get(step.expected_a)
            b = self._workspace.get(step.expected_b)
            if a is None or b is None:
                missing = [n for n, val in [(step.expected_a, a), (step.expected_b, b)] if val is None]
                return False, f"Élément(s) manquant(s) dans le workspace : {missing}"
            vec = b - a
            pl = step.plane
            if pl == "horizontal":
                angle = abs(_angle_horiz(vec, up))
            elif pl == "vertical":
                angle = _angle_from_vertical(vec, up)
            elif pl == "frontal":
                angle = _angle_frontal_from_vertical(vec, up)
            else:  # horizontal_ap
                angle = _angle_horiz_ap(vec, up)
            self._workspace[step.result_name] = angle
            self._index += 1
            return True, f"Angle calculé : {angle:.1f}°"

        # ── ComputeAsymmetry ──────────────────────────────────────────────────
        if isinstance(step, ComputeAsymmetry):
            left = self._workspace.get(step.expected_left)
            right = self._workspace.get(step.expected_right)
            if left is None or right is None:
                missing = [n for n, val in [(step.expected_left, left), (step.expected_right, right)] if val is None]
                return False, f"Élément(s) manquant(s) dans le workspace : {missing}"
            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                asym = abs(float(left[up] - right[up]))
            else:
                asym = abs(float(left) - float(right))
            self._workspace[step.result_name] = asym
            self._index += 1
            return True, f"Asymétrie : {asym:.1f}"

        # ── ComputeAxisAngle ──────────────────────────────────────────────────
        if isinstance(step, ComputeAxisAngle):
            a1 = self._workspace.get(step.expected_a1)
            a2 = self._workspace.get(step.expected_a2)
            b1 = self._workspace.get(step.expected_b1)
            b2 = self._workspace.get(step.expected_b2)
            missing = [n for n, val in [
                (step.expected_a1, a1), (step.expected_a2, a2),
                (step.expected_b1, b1), (step.expected_b2, b2),
            ] if val is None]
            if missing:
                return False, f"Élément(s) manquant(s) dans le workspace : {missing}"
            ax1 = a2 - a1
            ax2 = b2 - b1
            pl = step.plane
            if pl == "transverse":
                ax1 = _project_plane(ax1, up)
                ax2 = _project_plane(ax2, up)
            elif pl == "frontal":
                ml, ap = _horizontal_axes(up)
                ax1 = _project_plane(ax1, ap)
                ax2 = _project_plane(ax2, ap)
            elif pl == "sagittal":
                ml, ap = _horizontal_axes(up)
                ax1 = _project_plane(ax1, ml)
                ax2 = _project_plane(ax2, ml)
            n1 = np.linalg.norm(ax1)
            n2 = np.linalg.norm(ax2)
            if n1 < 1e-9 or n2 < 1e-9:
                return False, "Axes colinéaires ou nuls — calcul impossible."
            cos_a = float(np.dot(ax1, ax2) / (n1 * n2))
            angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            self._workspace[step.result_name] = angle
            self._index += 1
            return True, f"Angle entre axes : {angle:.1f}°"

        # ── ComputeProjection ─────────────────────────────────────────────────
        if isinstance(step, ComputeProjection):
            a = self._workspace.get(step.expected_a)
            b = self._workspace.get(step.expected_b)
            if a is None or b is None:
                missing = [n for n, val in [(step.expected_a, a), (step.expected_b, b)] if val is None]
                return False, f"Élément(s) manquant(s) dans le workspace : {missing}"
            vec = b - a
            ml, ap = _horizontal_axes(up)
            if step.axis == "ap":
                proj = float(vec[ap])
            elif step.axis == "ml":
                proj = float(vec[ml])
            else:  # vertical
                proj = float(vec[up])
            self._workspace[step.result_name] = proj
            self._index += 1
            return True, f"Projection : {proj:.1f} mm"

        return False, "Type d'étape inconnu."

    # ── Final result ──────────────────────────────────────────────────────────

    def final_result(self) -> float | None:
        """Return the final computed scalar, or None if not yet available."""
        val = self._workspace.get(self._recipe.expected_result_name)
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return None  # should be scalar
        return float(val)
