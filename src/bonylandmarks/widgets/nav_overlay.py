"""Floating 3D navigation overlay for PyVista exercises.

Drop it on any ``QtInteractor.interactor`` and it self-positions top-right.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..ui.styles import NAV_BTN, NAV_BTN_CHECK


class NavOverlay(QWidget):
    """Floating nav panel (top-right of a PyVista interactor).

    Parameters
    ----------
    container:
        The QWidget to overlay on — usually ``plotter.interactor``.
        It becomes the Qt parent so the overlay stays clipped to the viewport.
    face_fn / back_fn / left_fn / right_fn / top_fn / reset_fn:
        Callables invoked when the corresponding view button is clicked.
    pan_fn:
        ``callable(dx: float, dy: float)`` — called with ±1 normalised units.
    toggle_curvature_fn:
        ``callable(checked: bool)`` to show/hide a curvature heatmap.
        Omit (or pass ``None``) to hide the curvature section entirely.
    """

    def __init__(
        self,
        container: QWidget,
        *,
        face_fn: Callable,
        back_fn: Callable,
        left_fn: Callable,
        right_fn: Callable,
        top_fn: Callable,
        reset_fn: Callable,
        pan_fn: Callable,
        toggle_curvature_fn: Optional[Callable] = None,
        parent=None,
    ) -> None:
        super().__init__(container if parent is None else parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._toggle_curvature_fn = toggle_curvature_fn
        self._curv_btn: QPushButton | None = None
        self._curv_legend: QLabel | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── View presets ─────────────────────────────────────────────────────
        layout.addWidget(self._label("Vues"))

        views_row = QHBoxLayout()
        views_row.setSpacing(3)
        for text, tip, slot in (
            ("Face", "Vue avant",     face_fn),
            ("Dos",  "Vue arrière",   back_fn),
            ("G",    "Vue gauche",    left_fn),
            ("D",    "Vue droite",    right_fn),
            ("↑",    "Vue dessus",    top_fn),
            ("⟳",    "Réinitialiser", reset_fn),
        ):
            b = QPushButton(text)
            b.setFixedHeight(24)
            b.setStyleSheet(NAV_BTN)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            views_row.addWidget(b)
        layout.addLayout(views_row)

        # ── Pan arrows ───────────────────────────────────────────────────────
        layout.addWidget(self._label("Translation"))

        _pan_tip = (
            "Translation · aussi : Shift + clic-gauche glisser\n"
            "Zoom : molette · Rotation : clic-gauche glisser"
        )
        pan_w = QWidget()
        pan_w.setAttribute(Qt.WA_TranslucentBackground)
        grid = QGridLayout(pan_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        for row, col, sym, dx, dy in (
            (0, 1, "▲",  0, +1),
            (1, 0, "◀", -1,  0),
            (1, 2, "▶", +1,  0),
            (2, 1, "▼",  0, -1),
        ):
            b = QPushButton(sym)
            b.setFixedSize(26, 26)
            b.setStyleSheet(NAV_BTN)
            b.setToolTip(_pan_tip)
            b.clicked.connect(lambda _=None, _dx=dx, _dy=dy: pan_fn(_dx, _dy))
            grid.addWidget(b, row, col)
        layout.addWidget(pan_w)

        # ── Curvature toggle (optional) ───────────────────────────────────────
        if toggle_curvature_fn is not None:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(
                "background: rgba(255,255,255,0.12); border: none; max-height: 1px;"
            )
            layout.addWidget(sep)

            self._curv_btn = QPushButton("🌡 Courbure")
            self._curv_btn.setCheckable(True)
            self._curv_btn.setStyleSheet(NAV_BTN_CHECK)
            self._curv_btn.setToolTip(
                "Heatmap de courbure moyenne\n"
                "Rouge = saillant (éminence osseuse)\n"
                "Bleu = creux (sillon, fossette)"
            )
            self._curv_btn.toggled.connect(self._on_curvature_toggled)
            layout.addWidget(self._curv_btn)

            self._curv_legend = QLabel("Rouge = saillant · Bleu = creux")
            self._curv_legend.setStyleSheet(
                "font-size: 9px; color: rgba(255,180,80,0.85); "
                "background: transparent; padding: 1px 2px;"
            )
            self._curv_legend.setAlignment(Qt.AlignCenter)
            self._curv_legend.setVisible(False)
            layout.addWidget(self._curv_legend)

        # ── Mouse legend ─────────────────────────────────────────────────────
        mouse_info = QLabel("🖱 clic · ↕ molette · Shift+clic")
        mouse_info.setStyleSheet(
            "font-size: 9px; color: rgba(160,160,200,0.7); "
            "background: transparent; padding: 1px 0;"
        )
        mouse_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(mouse_info)

        self.adjustSize()
        self._position()
        container.installEventFilter(self)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size: 9px; color: rgba(160,160,200,0.8); "
            "background: transparent; padding: 0;"
        )
        return lbl

    def _on_curvature_toggled(self, checked: bool) -> None:
        if self._curv_legend is not None:
            self._curv_legend.setVisible(checked)
        if self._toggle_curvature_fn is not None:
            self._toggle_curvature_fn(checked)
        self.adjustSize()
        self._position()

    def _position(self) -> None:
        container = self.parent()
        if container is None:
            return
        margin = 8
        w = self.width() or self.sizeHint().width()
        self.move(container.width() - w - margin, margin)
        self.raise_()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset_curvature(self) -> None:
        """Uncheck the curvature button silently (without firing the callback)."""
        if self._curv_btn is not None and self._curv_btn.isChecked():
            self._curv_btn.blockSignals(True)
            self._curv_btn.setChecked(False)
            self._curv_btn.blockSignals(False)
        if self._curv_legend is not None:
            self._curv_legend.setVisible(False)

    # ── Qt overrides ─────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if obj is self.parent() and event.type() == QEvent.Resize:
            self._position()
        return super().eventFilter(obj, event)
