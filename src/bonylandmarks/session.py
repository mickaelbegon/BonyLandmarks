"""Pure-Python session state machine for a BonyLandmarks exercise.

This module deliberately contains **no Qt import**: it owns the whole session
logic (side choice, mixed inverse/placement split, progression, retry queue and
scoring) so it can be unit-tested without a GUI.  ``LandmarkViewer`` keeps a
reference to a :class:`SessionController` and only handles rendering, widgets
and user events.

Session flow
------------
1. A random body side is drawn; bilateral landmarks of the other side are
   dropped.
2. Landmarks that own a ground truth are shuffled and split 50/50 between an
   *inverse identification* phase and a *placement* phase (mixed session).
3. The placement phase walks ``landmarks`` in order, recording one
   :class:`~bonylandmarks.scoring.LandmarkResult` per landmark.
4. Landmarks graded ``D`` are queued for a retry pass; the best result per code
   across all passes is what the final score uses.
"""

from __future__ import annotations

import random

import numpy as np

from .i18n import Language
from .landmarks_extended import Landmark
from .scoring import LandmarkResult, SessionScore


class SessionController:
    """Owns every piece of session state and the rules that move it forward."""

    def __init__(
        self,
        landmarks: list[Landmark],
        ground_truth: dict[str, np.ndarray],
        lang: Language = "fr",
    ) -> None:
        self._ground_truth = ground_truth
        self._lang: Language = lang

        # Pick a random side for this session then filter bilateral landmarks
        self._chosen_side: str = random.choice(("left", "right"))
        session_landmarks: list[Landmark] = [
            lm for lm in landmarks
            if not (
                (lm.code.endswith("_left") and self._chosen_side == "right")
                or (lm.code.endswith("_right") and self._chosen_side == "left")
            )
        ]

        # ── Mixed session: split landmarks into inverse (50 %) and placement (50 %) ──
        # Save the full list so the inverse identification widget shows all choices.
        self._all_session_landmarks: list[Landmark] = list(session_landmarks)
        # Only landmarks with a ground truth can be displayed in inverse mode.
        _gt_lms = [lm for lm in session_landmarks if lm.code in ground_truth]
        random.shuffle(_gt_lms)
        n_inv = len(_gt_lms) // 2          # 50 % → inverse phase
        self._inverse_landmarks: list[Landmark] = _gt_lms[:n_inv]
        _inv_codes = {lm.code for lm in self._inverse_landmarks}
        # Placement list: remaining landmarks in their original session order.
        self._landmarks: list[Landmark] = [
            lm for lm in session_landmarks if lm.code not in _inv_codes
        ]
        # Mixed session = True when there is at least one inverse landmark.
        self._mixed_session: bool = bool(self._inverse_landmarks)
        # Inverse phase score tracking
        self._inverse_correct_codes: set[str] = set()

        self._index: int = 0
        self._results: list[LandmarkResult] = []
        # Task C — best result per code across all passes (normal + retry)
        self._best_results: dict[str, LandmarkResult] = {}
        # Task C — queue of landmarks to retry (grade D)
        self._retry_queue: list[Landmark] = []
        self._retry_mode: bool = False

        # Inverse identification queue
        self._inverse_queue: list[Landmark] = []   # landmarks to identify, shuffled
        self._inverse_index: int = 0

    # ── Read-only views on the session state ─────────────────────────────────

    @property
    def lang(self) -> Language:
        return self._lang

    @lang.setter
    def lang(self, value: Language) -> None:
        self._lang = value

    @property
    def ground_truth(self) -> dict[str, np.ndarray]:
        return self._ground_truth

    @property
    def chosen_side(self) -> str:
        """``"left"`` or ``"right"`` — the side drawn for this session."""
        return self._chosen_side

    @property
    def landmarks(self) -> list[Landmark]:
        """Landmarks of the placement phase, in session order."""
        return self._landmarks

    @property
    def all_session_landmarks(self) -> list[Landmark]:
        """Every landmark of the session (placement + inverse)."""
        return self._all_session_landmarks

    @property
    def inverse_landmarks(self) -> list[Landmark]:
        """Landmarks reserved for the inverse identification phase."""
        return self._inverse_landmarks

    @property
    def mixed_session(self) -> bool:
        return self._mixed_session

    @property
    def results(self) -> list[LandmarkResult]:
        """Results of the normal (non-retry) placement pass."""
        return self._results

    @property
    def best_results(self) -> dict[str, LandmarkResult]:
        return self._best_results

    @property
    def current_placement_index(self) -> int:
        return self._index

    @property
    def n_placement(self) -> int:
        return len(self._landmarks)

    @property
    def retry_mode(self) -> bool:
        return self._retry_mode

    @property
    def retry_queue(self) -> list[Landmark]:
        return self._retry_queue

    @property
    def inverse_queue(self) -> list[Landmark]:
        return self._inverse_queue

    @property
    def inverse_index(self) -> int:
        return self._inverse_index

    # ── Placement phase ──────────────────────────────────────────────────────

    def placement_finished(self) -> bool:
        """True when every placement landmark has been visited."""
        return self._index >= len(self._landmarks)

    def current_landmark(self) -> Landmark:
        """Return the landmark currently being placed (normal pass)."""
        return self._landmarks[self._index]

    def record_result(self, result: LandmarkResult) -> None:
        """Store *result*, keeping the best composite score per landmark code."""
        # In normal mode, append to the main results list
        if not self._retry_mode:
            self._results.append(result)
        # Always keep the best score across all passes
        if (
            result.code not in self._best_results
            or result.composite_score > self._best_results[result.code].composite_score
        ):
            self._best_results[result.code] = result

    def advance(self) -> bool:
        """Move to the next placement landmark.

        Returns True when the placement pass is exhausted.
        """
        self._index += 1
        return self.placement_finished()

    # ── Task C — D-grade retry ───────────────────────────────────────────────

    def needs_retry(self) -> bool:
        """True when at least one landmark of the normal pass was graded D."""
        return bool(self.build_retry_queue())

    def build_retry_queue(self) -> list[Landmark]:
        """Return the landmarks graded D during the normal pass (session order)."""
        d_codes = {r.code for r in self._results if r.grade_letter() == "D"}
        return [lm for lm in self._landmarks if lm.code in d_codes]

    def begin_retry(self, d_landmarks: list[Landmark] | None = None) -> None:
        """Enter retry mode with a shuffled queue of the grade-D landmarks."""
        if d_landmarks is None:
            d_landmarks = self.build_retry_queue()
        self._retry_mode = True
        self._retry_queue = list(d_landmarks)
        random.shuffle(self._retry_queue)

    def retry_finished(self) -> bool:
        return not self._retry_queue

    def current_retry_landmark(self) -> Landmark:
        """Return the landmark at the front of the retry queue."""
        return self._retry_queue[0]

    def advance_retry(self, result: LandmarkResult | None = None) -> bool:
        """Pop the front of the retry queue, re-queueing it when still graded D.

        Returns True when the retry pass is exhausted.
        """
        done_lm = self._retry_queue.pop(0)
        if result is not None and result.grade_letter() == "D":
            self._retry_queue.append(done_lm)
        return self.retry_finished()

    # ── Scoring ──────────────────────────────────────────────────────────────

    def session_score(self) -> SessionScore:
        """Final score built from the best result of every landmark."""
        final_results = list(self._best_results.values())
        # Include results for unscored landmarks (no ground truth) from normal pass
        scored_codes = {r.code for r in final_results}
        for r in self._results:
            if r.code not in scored_codes:
                final_results.append(r)
        return SessionScore(final_results)

    def normal_score(self) -> SessionScore:
        """Score of the normal pass only (used when no retry is needed)."""
        return SessionScore(self._results)

    # ── Inverse identification phase ─────────────────────────────────────────

    def start_inverse(self, queue: list[Landmark] | None = None) -> None:
        """Build (and shuffle) the identification queue, then reset its cursor.

        If *queue* is provided (mixed-session auto-start) it is used directly.
        Otherwise the queue is built from all placement landmarks that have a
        ground truth (manual-toggle path).
        """
        if queue is not None:
            self._inverse_queue = list(queue)
        else:
            self._inverse_queue = [
                lm for lm in self._landmarks if lm.code in self._ground_truth
            ]
        random.shuffle(self._inverse_queue)
        self._inverse_index = 0

    def inverse_finished(self) -> bool:
        return self._inverse_index >= len(self._inverse_queue)

    def current_inverse_landmark(self) -> Landmark:
        """Return the landmark whose ground-truth sphere is being shown."""
        return self._inverse_queue[self._inverse_index]

    def inverse_advance(self) -> bool:
        """Move to the next inverse landmark.

        Returns True when the inverse phase is exhausted.
        """
        self._inverse_index += 1
        return self.inverse_finished()

    def record_inverse_correct(self, code: str) -> None:
        """Mark *code* as correctly identified during the inverse phase."""
        self._inverse_correct_codes.add(code)

    def inverse_score(self) -> tuple[int, int]:
        """Return ``(n_correct, n_total)`` for the inverse identification phase."""
        return len(self._inverse_correct_codes), len(self._inverse_queue)
