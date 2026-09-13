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

    @property
    def error_mm(self) -> float:
        """Euclidean distance between student pick and ground truth (mm)."""
        return float(np.linalg.norm(self.student_pick - self.ground_truth))

    def feedback_color(
        self,
        threshold_ok: float = 20.0,
        threshold_warn: float = 40.0,
    ) -> str:
        """Return 'green', 'orange', or 'red' based on error thresholds."""
        err = self.error_mm
        if err <= threshold_ok:
            return "green"
        if err <= threshold_warn:
            return "orange"
        return "red"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "ground_truth_mm": self.ground_truth.tolist(),
            "student_pick_mm": self.student_pick.tolist(),
            "error_mm": round(self.error_mm, 2),
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

    def to_dict(self) -> dict:
        return {
            "mean_error_mm": round(self.mean_error_mm, 2),
            "max_error_mm": round(self.max_error_mm, 2),
            "n_landmarks": len(self.results),
            "landmarks": [r.to_dict() for r in self.results],
        }
