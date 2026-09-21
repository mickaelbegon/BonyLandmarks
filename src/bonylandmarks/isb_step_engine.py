"""Moteur de construction guidée des repères locaux ISB (mode mixte).

Ce module transforme l'exercice ISB d'un **affichage passif** (résultats
pré-calculés par :mod:`isb_exercise`) en une **construction active guidée** :

* le wizard impose la *séquence* des étapes (l'étudiant ne se perd pas) ;
* l'étudiant choisit lui-même les *objets* (landmarks, points, vecteurs) ;
* chaque choix erroné déclenche un feedback pédagogique explicite.

Vocabulaire
-----------
``Workspace``
    L'état courant de l'étudiant : points nommés (landmarks du ground truth et
    milieux construits), vecteurs normalisés, scalaires.
``Step``
    Une opération élémentaire : choisir un landmark, calculer un milieu, un
    vecteur, un produit vectoriel, ou valider le repère final.
``StepEngine``
    L'automate qui avance dans la recette, valide les sélections et produit le
    feedback.

Conventions géométriques
------------------------
Repère ISB (position anatomique) : **X antérieur, Y supérieur** (proximal pour
les membres), **Z vers la droite** du sujet.  Le trièdre est direct, donc :

    X = Y × Z        Y = Z × X        Z = X × Y

L'ordre des produits vectoriels des recettes de :mod:`isb_recipes` respecte
strictement ces identités — un ordre inversé donne l'axe opposé et est rejeté
par le moteur avec un feedback dédié.

Les coordonnées brutes des landmarks sont exprimées dans le repère global GLB
(X droite, Y haut, −Z antérieur, convention Blender) ; les axes ISB ci-dessus
sont construits *dans le repère local du segment* et ne dépendent donc pas de
l'orientation globale du scan.

Le module ne dépend que de ``numpy`` et de :mod:`isb_exercise` (pour l'angle
entre deux vecteurs et le repère ISB de référence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .isb_exercise import angle_between, compute_isb_lcs

__all__ = [
    "PickLandmark",
    "ComputeMidpoint",
    "ComputeVector",
    "ComputeCrossProduct",
    "ValidateFrame",
    "Step",
    "Workspace",
    "AxisCheck",
    "FrameReport",
    "StepEngine",
    "step_instruction",
    "expected_selection",
    "TOL_EXCELLENT_DEG",
    "TOL_ACCEPTABLE_DEG",
    "TOL_VECTOR_DEG",
]

_EPS = 1e-9

#: Écart angulaire (degrés) en dessous duquel un axe est « excellent ».
TOL_EXCELLENT_DEG = 5.0
#: Écart angulaire (degrés) en dessous duquel un axe reste « acceptable ».
TOL_ACCEPTABLE_DEG = 15.0
#: Tolérance sur un vecteur intermédiaire avant de signaler une dérive.
TOL_VECTOR_DEG = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Opérations (étapes)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PickLandmark:
    """Étape : l'étudiant sélectionne un landmark du ground truth."""

    kind: Literal["pick_landmark"] = "pick_landmark"
    instruction_fr: str = ""
    instruction_en: str = ""
    expected_code: str = ""       # code du landmark attendu dans ground_truth
    result_name: str = ""         # clé dans le workspace (ex. "O", "ASIS_R")
    tolerance_ok: list[str] = field(default_factory=list)  # codes équivalents
    hint_fr: str = ""             # rappel de palpation affiché en cas d'erreur
    hint_en: str = ""
    auto: bool = False            # exécutée par le moteur, sans choix étudiant


@dataclass
class ComputeMidpoint:
    """Étape : l'étudiant sélectionne 2 points du workspace → milieu."""

    kind: Literal["midpoint"] = "midpoint"
    instruction_fr: str = ""
    instruction_en: str = ""
    expected_a: str = ""          # nom dans le workspace
    expected_b: str = ""
    result_name: str = ""
    hint_fr: str = ""
    hint_en: str = ""
    auto: bool = False


@dataclass
class ComputeVector:
    """Étape : l'étudiant sélectionne 2 points → vecteur A→B."""

    kind: Literal["vector"] = "vector"
    instruction_fr: str = ""
    instruction_en: str = ""
    expected_from: str = ""
    expected_to: str = ""
    result_name: str = ""
    normalize: bool = True
    hint_fr: str = ""
    hint_en: str = ""
    auto: bool = False


@dataclass
class ComputeCrossProduct:
    """Étape : l'étudiant sélectionne 2 vecteurs du workspace → A × B."""

    kind: Literal["cross_product"] = "cross_product"
    instruction_fr: str = ""
    instruction_en: str = ""
    expected_a: str = ""          # premier vecteur (A × B)
    expected_b: str = ""
    result_name: str = ""
    normalize: bool = True
    hint_fr: str = ""
    hint_en: str = ""
    auto: bool = False


@dataclass
class ValidateFrame:
    """Étape finale : valider le repère construit par l'étudiant."""

    kind: Literal["validate_frame"] = "validate_frame"
    instruction_fr: str = "Valide ton repère."
    instruction_en: str = "Validate your frame."
    origin_name: str = ""
    x_name: str = ""
    y_name: str = ""
    z_name: str = ""
    segment_key: str = ""         # clé isb_exercise pour le repère de référence
    hint_fr: str = ""
    hint_en: str = ""
    auto: bool = False


Step = PickLandmark | ComputeMidpoint | ComputeVector | ComputeCrossProduct | ValidateFrame


def step_instruction(step: Step, lang: str = "fr") -> str:
    """Consigne localisée d'une étape."""
    return step.instruction_fr if lang == "fr" else (step.instruction_en or step.instruction_fr)


def step_hint(step: Step, lang: str = "fr") -> str:
    """Indice localisé d'une étape (peut être vide)."""
    return step.hint_fr if lang == "fr" else (step.hint_en or step.hint_fr)


def expected_selection(step: Step) -> list[str]:
    """Sélection attendue pour *step* (dans l'ordre), ``[]`` si aucune."""
    if isinstance(step, PickLandmark):
        return [step.expected_code]
    if isinstance(step, ComputeMidpoint):
        return [step.expected_a, step.expected_b]
    if isinstance(step, ComputeVector):
        return [step.expected_from, step.expected_to]
    if isinstance(step, ComputeCrossProduct):
        return [step.expected_a, step.expected_b]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Workspace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Workspace:
    """Stocke les objets nommés construits par l'étudiant."""

    points: dict[str, np.ndarray] = field(default_factory=dict)    # landmarks + milieux
    vectors: dict[str, np.ndarray] = field(default_factory=dict)   # vecteurs (normalisés)
    scalars: dict[str, float] = field(default_factory=dict)        # angles, distances

    def load_ground_truth(self, ground_truth: dict[str, np.ndarray]) -> None:
        """Charge les landmarks GT dans le workspace (nom = code landmark)."""
        for code, pos in ground_truth.items():
            if pos is None:
                continue
            self.points[code] = np.asarray(pos, dtype=np.float64).reshape(3)

    def clear_derived(self, ground_truth: dict[str, np.ndarray]) -> None:
        """Vide le workspace et le recharge avec le seul ground truth."""
        self.points.clear()
        self.vectors.clear()
        self.scalars.clear()
        self.load_ground_truth(ground_truth)

    def has(self, name: str) -> bool:
        return name in self.points or name in self.vectors

    def kind_of(self, name: str) -> str:
        """``'point'``, ``'vector'``, ``'scalar'`` ou ``''`` si inconnu."""
        if name in self.points:
            return "point"
        if name in self.vectors:
            return "vector"
        if name in self.scalars:
            return "scalar"
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de validation du repère
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AxisCheck:
    """Écart d'un axe étudiant par rapport à la référence ISB."""

    axis: str                    # "X", "Y", "Z"
    angle_deg: float             # NaN si l'axe est manquant
    verdict: str                 # "excellent", "acceptable", "a_revoir", "manquant"

    @property
    def ok(self) -> bool:
        return self.verdict in ("excellent", "acceptable")


@dataclass
class FrameReport:
    """Résultat détaillé d'une étape :class:`ValidateFrame`."""

    axes: list[AxisCheck] = field(default_factory=list)
    origin_error_mm: float = float("nan")
    orthogonality_deg: float = float("nan")   # pire écart à 90° entre axes
    right_handed: bool = True
    has_reference: bool = True
    message_fr: str = ""

    @property
    def worst_angle_deg(self) -> float:
        vals = [a.angle_deg for a in self.axes if np.isfinite(a.angle_deg)]
        return max(vals) if vals else float("nan")

    @property
    def success(self) -> bool:
        return bool(self.axes) and all(a.ok for a in self.axes)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers géométriques
# ─────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray | None:
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < _EPS:
        return None
    return vec / norm


def _verdict(angle_deg: float) -> str:
    if not np.isfinite(angle_deg):
        return "manquant"
    if angle_deg < TOL_EXCELLENT_DEG:
        return "excellent"
    if angle_deg <= TOL_ACCEPTABLE_DEG:
        return "acceptable"
    return "a_revoir"


def _landmark_label(code: str) -> str:
    """Nom FR lisible d'un landmark, ou son code brut."""
    try:
        from .landmarks_extended import LANDMARK_BY_CODE
    except Exception:       # pragma: no cover - données absentes
        return code
    lm = LANDMARK_BY_CODE.get(code)
    return f"{lm.name_fr} ({code})" if lm is not None else code


def _landmark_hint(code: str) -> str:
    try:
        from .landmarks_extended import LANDMARK_BY_CODE
    except Exception:       # pragma: no cover
        return ""
    lm = LANDMARK_BY_CODE.get(code)
    return lm.hint_fr if lm is not None else ""


# ─────────────────────────────────────────────────────────────────────────────
# Moteur
# ─────────────────────────────────────────────────────────────────────────────

class StepEngine:
    """Automate de construction guidée d'un repère ISB.

    Parameters
    ----------
    steps
        La recette du segment (voir :mod:`isb_recipes`).
    ground_truth
        ``{code_landmark: np.ndarray(3,)}`` — positions de référence.
    segment_key
        Clé :mod:`isb_exercise` (``"pelvis"``, ``"thorax"``, …) utilisée pour
        récupérer le repère ISB de référence lors de la validation finale.  Si
        elle est absente ou non constructible, le moteur se rabat sur le
        « workspace de référence » obtenu en rejouant la recette parfaitement.
    """

    def __init__(
        self,
        steps: list[Step],
        ground_truth: dict[str, np.ndarray],
        segment_key: str | None = None,
    ) -> None:
        self._steps: list[Step] = list(steps)
        self._gt: dict[str, np.ndarray] = {
            code: np.asarray(pos, dtype=np.float64).reshape(3)
            for code, pos in ground_truth.items()
            if pos is not None
        }
        self._segment_key = segment_key or self._infer_segment_key()
        self._index = 0
        self._workspace = Workspace()
        self._workspace.load_ground_truth(self._gt)
        self._attempts: list[int] = [0] * len(self._steps)
        self._last_report: FrameReport | None = None

        # Workspace « parfait » : la recette rejouée avec les bonnes sélections.
        self._reference_ws = self._build_reference_workspace()
        self._reference_frame = self._lookup_reference_frame()

        self._run_auto_steps()

    # ── Introspection ────────────────────────────────────────────────────────

    @property
    def steps(self) -> list[Step]:
        return list(self._steps)

    @property
    def current_step(self) -> Step:
        if self.finished:
            raise IndexError("La recette est terminée : plus d'étape courante.")
        return self._steps[self._index]

    @property
    def step_index(self) -> int:
        return self._index

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    @property
    def finished(self) -> bool:
        return self._index >= len(self._steps)

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def segment_key(self) -> str:
        return self._segment_key

    @property
    def attempts(self) -> list[int]:
        """Nombre de tentatives (réussies ou non) par étape."""
        return list(self._attempts)

    @property
    def errors(self) -> int:
        """Nombre total de sélections erronées depuis le début."""
        done = self._index
        return max(0, sum(self._attempts) - done)

    @property
    def last_report(self) -> FrameReport | None:
        """Rapport de la dernière étape :class:`ValidateFrame` exécutée."""
        return self._last_report

    @property
    def reference_frame(self) -> dict[str, np.ndarray] | None:
        """Repère ISB de référence ``{'origin','x','y','z'}``, si disponible."""
        return self._reference_frame

    def progress(self) -> tuple[int, int]:
        """``(étapes_validées, total)``."""
        return self._index, len(self._steps)

    def instruction(self, lang: str = "fr") -> str:
        """Consigne de l'étape courante (chaîne vide si terminé)."""
        if self.finished:
            return "" if lang != "fr" else ""
        return step_instruction(self.current_step, lang)

    def reset(self) -> None:
        """Remet l'exercice à zéro (workspace, index, compteurs)."""
        self._index = 0
        self._workspace.clear_derived(self._gt)
        self._attempts = [0] * len(self._steps)
        self._last_report = None
        self._run_auto_steps()

    # ── Exécution ────────────────────────────────────────────────────────────

    def try_execute(self, selection: list[str]) -> tuple[bool, str]:
        """Tente d'exécuter l'étape courante avec la sélection de l'étudiant.

        ``selection`` : liste de 0 à 2 noms d'objets du workspace (un
        :class:`ValidateFrame` n'en attend aucun).

        Retourne ``(success, feedback_fr)``.  En cas de succès le résultat est
        écrit dans le workspace et le moteur avance à l'étape suivante (en
        exécutant au passage les étapes marquées ``auto=True``).
        """
        if self.finished:
            return False, "La construction est terminée : il n'y a plus d'étape à exécuter."

        step = self.current_step
        names = [str(n) for n in (selection or [])]
        self._attempts[self._index] += 1

        if isinstance(step, PickLandmark):
            ok, msg = self._exec_pick(step, names)
        elif isinstance(step, ComputeMidpoint):
            ok, msg = self._exec_midpoint(step, names)
        elif isinstance(step, ComputeVector):
            ok, msg = self._exec_vector(step, names)
        elif isinstance(step, ComputeCrossProduct):
            ok, msg = self._exec_cross(step, names)
        elif isinstance(step, ValidateFrame):
            ok, msg = self._exec_validate(step)
        else:                                   # pragma: no cover - garde-fou
            return False, f"Type d'étape inconnu : {type(step).__name__}."

        if ok:
            self._index += 1
            self._run_auto_steps()
        return ok, msg

    # ── Implémentations par type d'étape ─────────────────────────────────────

    def _exec_pick(self, step: PickLandmark, names: list[str]) -> tuple[bool, str]:
        if len(names) != 1:
            return False, (
                f"Sélectionne exactement 1 landmark ({len(names)} sélectionné(s)). "
                f"{step.instruction_fr}"
            )
        chosen = names[0]
        accepted = [step.expected_code, *step.tolerance_ok]

        if chosen not in accepted:
            hint = step.hint_fr or _landmark_hint(step.expected_code)
            msg = (
                f"Ce landmark est « {_landmark_label(chosen)} », "
                f"pas « {_landmark_label(step.expected_code)} » attendu."
            )
            if hint:
                msg += f" Pense à : {hint}"
            return False, msg

        if chosen not in self._gt:
            return False, (
                f"« {_landmark_label(chosen)} » est absent du ground truth : "
                "impossible de poursuivre cette étape."
            )

        self._workspace.points[step.result_name] = self._gt[chosen].copy()
        if chosen != step.expected_code:
            return True, (
                f"Accepté : « {_landmark_label(chosen)} » est un équivalent tolérable de "
                f"« {_landmark_label(step.expected_code)} » → stocké sous « {step.result_name} ». "
                "Attention : le repère final sera légèrement dégradé."
            )
        return True, (
            f"Correct : « {_landmark_label(chosen)} » → « {step.result_name} »."
        )

    def _exec_midpoint(self, step: ComputeMidpoint, names: list[str]) -> tuple[bool, str]:
        if len(names) != 2:
            return False, (
                f"Un milieu se calcule à partir de 2 points ({len(names)} sélectionné(s)). "
                f"{step.instruction_fr}"
            )
        expected = {step.expected_a, step.expected_b}
        bad_kind = self._check_kinds(names, "point")
        if bad_kind:
            return False, bad_kind
        if set(names) != expected:
            return False, (
                f"Le milieu attendu est celui de « {step.expected_a} » et « {step.expected_b} », "
                f"pas de « {names[0]} » et « {names[1]} »."
            )
        a = self._workspace.points[names[0]]
        b = self._workspace.points[names[1]]
        self._workspace.points[step.result_name] = (a + b) / 2.0
        # Le milieu est commutatif : pas de piège d'ordre à signaler.
        return True, (
            f"Correct : milieu de « {step.expected_a} » et « {step.expected_b} » "
            f"→ « {step.result_name} »."
        )

    def _exec_vector(self, step: ComputeVector, names: list[str]) -> tuple[bool, str]:
        if len(names) != 2:
            return False, (
                f"Un vecteur se construit à partir de 2 points ({len(names)} sélectionné(s)). "
                f"{step.instruction_fr}"
            )
        bad_kind = self._check_kinds(names, "point")
        if bad_kind:
            return False, bad_kind

        if names[0] == step.expected_to and names[1] == step.expected_from:
            return False, (
                f"Ordre inversé : le vecteur va de « {step.expected_from} » vers "
                f"« {step.expected_to} ». Sélectionné dans l'autre sens, il pointerait à 180° "
                "de la direction anatomique attendue. Sélectionne d'abord l'origine, "
                "puis l'extrémité."
            )
        if names[0] != step.expected_from or names[1] != step.expected_to:
            return False, (
                f"Le vecteur attendu va de « {step.expected_from} » vers "
                f"« {step.expected_to} », pas de « {names[0]} » vers « {names[1]} »."
            )

        raw = self._workspace.points[names[1]] - self._workspace.points[names[0]]
        vec = _unit(raw) if step.normalize else raw
        if vec is None:
            return False, (
                f"« {step.expected_from} » et « {step.expected_to} » sont confondus : "
                "le vecteur est nul, impossible de le normaliser."
            )
        self._workspace.vectors[step.result_name] = np.asarray(vec, dtype=np.float64)
        self._workspace.scalars[f"{step.result_name}_norm_mm"] = float(np.linalg.norm(raw))
        return True, self._with_drift_note(
            step.result_name,
            f"Correct : vecteur « {step.expected_from} » → « {step.expected_to} » "
            f"→ « {step.result_name} »"
            + (" (normalisé)." if step.normalize else "."),
        )

    def _exec_cross(self, step: ComputeCrossProduct, names: list[str]) -> tuple[bool, str]:
        if len(names) != 2:
            return False, (
                f"Un produit vectoriel se calcule à partir de 2 vecteurs "
                f"({len(names)} sélectionné(s)). {step.instruction_fr}"
            )
        bad_kind = self._check_kinds(names, "vector")
        if bad_kind:
            return False, bad_kind

        if names[0] == step.expected_b and names[1] == step.expected_a:
            return False, (
                f"Ordre inversé : le produit vectoriel n'est pas commutatif "
                f"(A × B = −B × A). Il faut « {step.expected_a} » × « {step.expected_b} » ; "
                f"dans ton ordre, « {step.result_name} » pointerait exactement à l'opposé "
                "de la direction anatomique attendue."
            )
        if names[0] != step.expected_a or names[1] != step.expected_b:
            return False, (
                f"Le produit vectoriel attendu est « {step.expected_a} » × "
                f"« {step.expected_b} », pas « {names[0]} » × « {names[1]} »."
            )

        a = self._workspace.vectors[names[0]]
        b = self._workspace.vectors[names[1]]
        raw = np.cross(a, b)
        vec = _unit(raw) if step.normalize else raw
        if vec is None:
            return False, (
                f"« {step.expected_a} » et « {step.expected_b} » sont colinéaires : "
                "leur produit vectoriel est nul. Vérifie les points qui les définissent."
            )
        self._workspace.vectors[step.result_name] = np.asarray(vec, dtype=np.float64)
        self._workspace.scalars[f"{step.result_name}_angle_deg"] = angle_between(a, b)
        return True, self._with_drift_note(
            step.result_name,
            f"Correct : « {step.expected_a} » × « {step.expected_b} » "
            f"→ « {step.result_name} ».",
        )

    def _exec_validate(self, step: ValidateFrame) -> tuple[bool, str]:
        report = self.build_frame_report(step)
        self._last_report = report
        return report.success, report.message_fr

    # ── Validation du repère ─────────────────────────────────────────────────

    def build_frame_report(self, step: ValidateFrame | None = None) -> FrameReport:
        """Compare le repère de l'étudiant au repère ISB de référence."""
        if step is None:
            step = next(
                (s for s in self._steps if isinstance(s, ValidateFrame)), None
            )
        if step is None:
            return FrameReport(has_reference=False, message_fr="Aucune étape de validation.")

        report = FrameReport()
        student: dict[str, np.ndarray | None] = {}
        missing: list[str] = []
        for axis, name in (("X", step.x_name), ("Y", step.y_name), ("Z", step.z_name)):
            vec = self._workspace.vectors.get(name) if name else None
            if vec is None:
                missing.append(f"{axis} (« {name or '?'} »)")
            student[axis] = None if vec is None else _unit(vec)

        reference = self._reference_frame or self._reference_from_workspace(step)

        if reference is None:
            report.has_reference = False
            report.message_fr = (
                "Repère de référence indisponible (landmarks manquants) : "
                "impossible de noter ta construction."
            )
            return report

        lines: list[str] = []
        for axis in ("X", "Y", "Z"):
            vec = student[axis]
            ref = reference.get(axis.lower())
            if vec is None or ref is None:
                report.axes.append(AxisCheck(axis, float("nan"), "manquant"))
                lines.append(f"  {axis} : axe manquant dans ton workspace.")
                continue
            ang = angle_between(vec, ref)
            verdict = _verdict(ang)
            report.axes.append(AxisCheck(axis, ang, verdict))
            label = {
                "excellent": "excellent",
                "acceptable": "acceptable",
                "a_revoir": "à revoir",
                "manquant": "manquant",
            }[verdict]
            extra = ""
            if ang > 150.0:
                extra = "  ← axe quasiment inversé : vérifie l'ordre d'un produit vectoriel."
            elif verdict == "a_revoir":
                extra = "  ← vérifie les landmarks qui définissent cet axe."
            lines.append(f"  {axis} : {ang:5.1f}°  ({label}){extra}")

        # Origine
        origin = self._workspace.points.get(step.origin_name) if step.origin_name else None
        ref_origin = reference.get("origin")
        if origin is not None and ref_origin is not None:
            report.origin_error_mm = float(np.linalg.norm(origin - ref_origin))

        # Orthogonalité et sens du trièdre
        axes = [student["X"], student["Y"], student["Z"]]
        if all(a is not None for a in axes):
            devs = [
                abs(90.0 - angle_between(axes[0], axes[1])),
                abs(90.0 - angle_between(axes[1], axes[2])),
                abs(90.0 - angle_between(axes[0], axes[2])),
            ]
            report.orthogonality_deg = float(max(devs))
            det = float(np.linalg.det(np.column_stack(axes)))
            report.right_handed = det > 0.0

        header = (
            "Repère validé — excellent travail !"
            if report.success
            else "Repère à corriger."
        )
        tail: list[str] = []
        if np.isfinite(report.origin_error_mm):
            tail.append(f"Origine : écart de {report.origin_error_mm:.1f} mm vs référence.")
        if np.isfinite(report.orthogonality_deg):
            tail.append(f"Orthogonalité : écart max à 90° = {report.orthogonality_deg:.1f}°.")
            if not report.right_handed:
                tail.append(
                    "Trièdre INDIRECT (gaucher) : un de tes axes est inversé. "
                    "Le repère ISB doit vérifier X × Y = Z."
                )
        if missing:
            tail.append("Axes absents : " + ", ".join(missing) + ".")

        report.message_fr = "\n".join([header, *lines, *tail])
        return report

    # ── Aides internes ───────────────────────────────────────────────────────

    def _check_kinds(self, names: list[str], expected_kind: str) -> str | None:
        """Vérifie que chaque nom existe et est du bon type. Retourne un message ou None."""
        human = {"point": "un point", "vector": "un vecteur"}[expected_kind]
        other = {"point": "un vecteur", "vector": "un point"}[expected_kind]
        for name in names:
            kind = self._workspace.kind_of(name)
            if not kind:
                return (
                    f"« {name} » n'existe pas encore dans ton espace de travail. "
                    "Tu dois d'abord le construire (ou sélectionner un objet déjà créé)."
                )
            if kind != expected_kind:
                return (
                    f"Cette opération attend {human}, mais « {name} » est {other}. "
                    "Un produit vectoriel combine deux vecteurs ; un milieu ou un "
                    "vecteur se construit à partir de deux points."
                )
        if len(set(names)) != len(names):
            return "Sélectionne deux objets distincts."
        return None

    def _with_drift_note(self, result_name: str, base_msg: str) -> str:
        """Ajoute un avertissement si le résultat s'écarte de la référence."""
        ref = self._reference_ws.vectors.get(result_name) if self._reference_ws else None
        got = self._workspace.vectors.get(result_name)
        if ref is None or got is None:
            return base_msg
        ang = angle_between(got, ref)
        if np.isfinite(ang) and ang > TOL_VECTOR_DEG:
            return (
                f"{base_msg} Attention : « {result_name} » s'écarte de {ang:.1f}° de la "
                "direction de référence (conséquence d'un landmark équivalent accepté "
                "plus tôt). Le repère final en portera la trace."
            )
        return base_msg

    def _run_auto_steps(self) -> None:
        """Exécute sans intervention les étapes marquées ``auto=True``."""
        guard = 0
        while not self.finished and getattr(self.current_step, "auto", False):
            guard += 1
            if guard > len(self._steps) + 1:    # pragma: no cover - garde-fou
                break
            step = self.current_step
            if isinstance(step, ValidateFrame):
                break
            ok, _ = self._apply(self._workspace, step, expected_selection(step))
            self._index += 1
            if not ok:
                break

    def _apply(
        self, ws: Workspace, step: Step, names: list[str]
    ) -> tuple[bool, str]:
        """Applique *step* à *ws* avec des noms supposés corrects (sans feedback)."""
        try:
            if isinstance(step, PickLandmark):
                code = names[0]
                if code not in self._gt:
                    return False, "landmark absent"
                ws.points[step.result_name] = self._gt[code].copy()
                return True, ""
            if isinstance(step, ComputeMidpoint):
                a, b = ws.points.get(names[0]), ws.points.get(names[1])
                if a is None or b is None:
                    return False, "point absent"
                ws.points[step.result_name] = (a + b) / 2.0
                return True, ""
            if isinstance(step, ComputeVector):
                a, b = ws.points.get(names[0]), ws.points.get(names[1])
                if a is None or b is None:
                    return False, "point absent"
                raw = b - a
                vec = _unit(raw) if step.normalize else raw
                if vec is None:
                    return False, "vecteur nul"
                ws.vectors[step.result_name] = np.asarray(vec, dtype=np.float64)
                ws.scalars[f"{step.result_name}_norm_mm"] = float(np.linalg.norm(raw))
                return True, ""
            if isinstance(step, ComputeCrossProduct):
                a, b = ws.vectors.get(names[0]), ws.vectors.get(names[1])
                if a is None or b is None:
                    return False, "vecteur absent"
                raw = np.cross(a, b)
                vec = _unit(raw) if step.normalize else raw
                if vec is None:
                    return False, "vecteurs colinéaires"
                ws.vectors[step.result_name] = np.asarray(vec, dtype=np.float64)
                return True, ""
        except Exception:                        # pragma: no cover - garde-fou
            return False, "erreur"
        return True, ""

    def _build_reference_workspace(self) -> Workspace | None:
        """Rejoue la recette avec les sélections parfaites (workspace témoin)."""
        ws = Workspace()
        ws.load_ground_truth(self._gt)
        for step in self._steps:
            if isinstance(step, ValidateFrame):
                continue
            ok, _ = self._apply(ws, step, expected_selection(step))
            if not ok:
                return ws
        return ws

    def _reference_from_workspace(
        self, step: ValidateFrame
    ) -> dict[str, np.ndarray] | None:
        """Repère de secours, extrait du workspace témoin."""
        ws = self._reference_ws
        if ws is None:
            return None
        out: dict[str, np.ndarray] = {}
        for key, name in (("x", step.x_name), ("y", step.y_name), ("z", step.z_name)):
            vec = ws.vectors.get(name) if name else None
            unit = _unit(vec) if vec is not None else None
            if unit is None:
                return None
            out[key] = unit
        origin = ws.points.get(step.origin_name) if step.origin_name else None
        if origin is not None:
            out["origin"] = origin
        return out

    def _lookup_reference_frame(self) -> dict[str, np.ndarray] | None:
        """Repère ISB canonique du segment, via :func:`isb_exercise.compute_isb_lcs`."""
        if not self._segment_key:
            return None
        try:
            result = compute_isb_lcs(self._gt)
        except Exception:                        # pragma: no cover - garde-fou
            return None
        seg = result.get(self._segment_key)
        if seg is None or not seg.present:
            return None
        return {"origin": seg.origin, "x": seg.x, "y": seg.y, "z": seg.z}

    def _infer_segment_key(self) -> str:
        for step in self._steps:
            if isinstance(step, ValidateFrame) and step.segment_key:
                return step.segment_key
        return ""
