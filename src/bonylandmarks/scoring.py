"""Scoring: compare student-placed landmarks against BodyLoop ground truth.

All coordinates are in millimetres (BodyLoop convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LandmarkResult:
    code: str
    ground_truth: np.ndarray   # shape (3,) in mm
    student_pick: np.ndarray   # shape (3,) in mm
    redo_count: int = 0        # number of redo clicks before confirming

    @property
    def error_mm(self) -> float:
        """Euclidean distance between student pick and ground truth (mm)."""
        return float(np.linalg.norm(self.student_pick - self.ground_truth))

    def feedback_color(
        self,
        threshold_a: float = 30.0,
        threshold_b: float = 80.0,
        threshold_c: float = 150.0,
    ) -> str:
        """Return a color string based on error thresholds.

        A (≤ 30 mm)  → 'green'
        B (≤ 80 mm)  → 'teal'
        C (≤ 150 mm) → 'orange'
        D (> 150 mm) → 'red'
        """
        err = self.error_mm
        if err <= threshold_a:
            return "green"
        if err <= threshold_b:
            return "teal"
        if err <= threshold_c:
            return "orange"
        return "red"

    @property
    def composite_score(self) -> float:
        """0–100 score combining accuracy and number of attempts.

        Accuracy component: 100 − error_mm / 1.5  (0 at ≥ 150 mm)
        Attempt penalty   : 5 points per redo beyond the first click
        """
        accuracy = max(0.0, 100.0 - self.error_mm / 1.5)
        penalty = max(0, self.redo_count - 1) * 5.0
        return max(0.0, accuracy - penalty)

    def grade_letter(self) -> str:
        """Return A/B/C/D based on Euclidean error in mm.

        A : error ≤  30 mm
        B : error ≤  80 mm
        C : error ≤ 150 mm
        D : error >  150 mm
        """
        err = self.error_mm
        if err <= 30.0:
            return "A"
        if err <= 80.0:
            return "B"
        if err <= 150.0:
            return "C"
        return "D"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "ground_truth_mm": self.ground_truth.tolist(),
            "student_pick_mm": self.student_pick.tolist(),
            "error_mm": round(self.error_mm, 2),
            "redo_count": self.redo_count,
            "composite_score": round(self.composite_score, 1),
            "grade": self.grade_letter(),
            "feedback_color": self.feedback_color(),
        }


@dataclass
class SessionScore:
    results: list[LandmarkResult]

    @property
    def mean_error_mm(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.error_mm for r in self.results]))

    @property
    def max_error_mm(self) -> float:
        if not self.results:
            return 0.0
        return float(np.max([r.error_mm for r in self.results]))

    @property
    def global_score(self) -> float:
        """Mean composite score across all landmarks (0–100)."""
        if not self.results:
            return 0.0
        return float(np.mean([r.composite_score for r in self.results]))

    @property
    def global_grade(self) -> str:
        """Return A/B/C/D based on mean error across all landmarks.

        A : mean error ≤  30 mm
        B : mean error ≤  80 mm
        C : mean error ≤ 150 mm
        D : mean error >  150 mm
        """
        err = self.mean_error_mm
        if err <= 30.0:
            return "A"
        if err <= 80.0:
            return "B"
        if err <= 150.0:
            return "C"
        return "D"

    def to_dict(self) -> dict:
        return {
            "mean_error_mm": round(self.mean_error_mm, 2),
            "max_error_mm": round(self.max_error_mm, 2),
            "global_score": round(self.global_score, 1),
            "global_grade": self.global_grade,
            "n_landmarks": len(self.results),
            "landmarks": [r.to_dict() for r in self.results],
        }
