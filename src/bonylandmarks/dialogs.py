"""Standalone QDialog builders for BonyLandmarks sessions.

Each function creates, wires internal mechanics, shows, centers, and returns
a QDialog.  The caller (LandmarkViewer) connects ``dlg.finished`` to any
session-level callbacks.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .i18n import Language
from .landmarks_extended import CATEGORY_LABELS, Landmark
from .scoring import LandmarkResult


# ─── colour / text constants (shared with viewer via this module) ─────────────

_GRADE_COLORS: dict[str, str] = {
    "A": "#2e7d32",   # dark green
    "B": "#1565c0",   # dark blue
    "C": "#e65100",   # dark orange
    "D": "#c62828",   # dark red
}

# Category descriptions shown on the transition screen (FR, EN)
_CATEGORY_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "BONE": (
        "Repères osseux palpables — base de toute mesure en kinésiologie",
        "Palpable bony landmarks — the foundation of every kinesiology measurement",
    ),
    "EMG": (
        "Sites d'électrodes de surface selon les recommandations SENIAM — "
        "chaque site se construit à partir de deux repères osseux",
        "Surface electrode sites per SENIAM recommendations — "
        "each site is defined relative to two bony landmarks",
    ),
    "SKINFOLD": (
        "Sites de plis cutanés — protocole ISAK. La pastille marque le centre de la pince.",
        "Skinfold sites — ISAK protocol. The marker indicates the centre of the caliper.",
    ),
    "ANTHRO": (
        "Sites de circonférences et diamètres osseux — protocole ISAK côté droit",
        "Girths and bone diameters — ISAK protocol, right side only",
    ),
}


# ─── internal helper ──────────────────────────────────────────────────────────

def _wire_enter(dlg: QDialog) -> None:
    """Connect Return/Enter keys to dlg.accept (window-scoped shortcut)."""
    for key in (Qt.Key_Return, Qt.Key_Enter):
        sc = QShortcut(QKeySequence(key), dlg)
        sc.setContext(Qt.WindowShortcut)
        sc.activated.connect(dlg.accept)


# ─── public builders ──────────────────────────────────────────────────────────

def build_debrief_dialog(
    lm: Landmark,
    result: LandmarkResult | None,
    lang: Language,
    parent: QWidget,
    center_fn: Callable[[QDialog], None],
    biom_measures: list | None = None,
) -> QDialog:
    """Build, show, and return the post-confirmation debrief overlay.

    The caller must connect ``dlg.finished`` to advance the session.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(
        "Résumé du repère" if lang == "fr" else "Landmark summary"
    )
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumWidth(440)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(12)
    layout.setContentsMargins(24, 24, 24, 20)

    # Landmark name (large, bold)
    name_lbl = QLabel(lm.name(lang))
    name_lbl.setWordWrap(True)
    name_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #f5f5f5;")
    layout.addWidget(name_lbl)

    # Grade badge (scored landmarks only)
    if result is not None:
        grade = result.grade_letter()
        grade_color = _GRADE_COLORS.get(grade, "#607D8B")
        grade_lbl = QLabel(
            f"Note : <b style='color:{grade_color};font-size:16px'>{grade}</b>"
            f"&nbsp;&nbsp;{result.composite_score:.0f}/100"
            f"&nbsp;({result.error_mm:.1f} mm)"
        )
        grade_lbl.setStyleSheet("font-size: 14px; color: #e0e0e0;")
        layout.addWidget(grade_lbl)
    else:
        not_scored_lbl = QLabel(
            "Non évalué" if lang == "fr" else "Not scored"
        )
        not_scored_lbl.setStyleSheet("font-size: 13px; color: #aaa;")
        layout.addWidget(not_scored_lbl)

    # Palpation hint
    hint_title = QLabel(
        "<b>Palpation</b>" if lang == "fr" else "<b>Palpation guide</b>"
    )
    layout.addWidget(hint_title)
    hint_box = QTextEdit()
    hint_box.setReadOnly(True)
    hint_box.setText(lm.hint(lang))
    hint_box.setFixedHeight(110)
    hint_box.setStyleSheet(
        "font-size: 12px; color: #111; background: #ffffff; "
        "border: 1px solid #bbb; border-radius: 4px;"
    )
    layout.addWidget(hint_box)

    # Clinical application (only when non-empty)
    app_text = lm.application(lang) if hasattr(lm, "application") else ""
    if app_text:
        app_title = QLabel(
            "<b>Application clinique</b>"
            if lang == "fr"
            else "<b>Clinical application</b>"
        )
        layout.addWidget(app_title)
        app_box = QTextEdit()
        app_box.setReadOnly(True)
        app_box.setText(app_text)
        app_box.setFixedHeight(70)
        app_box.setStyleSheet(
            "font-size: 11px; color: #1a237e; font-style: italic; "
            "background: #e8edf8; border: 1px solid #90a4d4; border-radius: 4px;"
        )
        layout.addWidget(app_box)

    # Biomechanical measurements involving this landmark
    if biom_measures:
        _STATUS_COLORS = {
            "normal":    ("#1a7a40", "#d4edda"),
            "attention": ("#7a5c00", "#fff3cd"),
            "alerte":    ("#8b1a1a", "#f8d7da"),
            "info":      ("#1a4a7a", "#d1ecf1"),
        }
        biom_title = QLabel(
            "<b>Mesures associées</b>" if lang == "fr" else "<b>Related measurements</b>"
        )
        biom_title.setStyleSheet("font-size: 12px; color: #ccc; margin-top: 4px;")
        layout.addWidget(biom_title)

        for m in biom_measures:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 1, 0, 1)
            row_layout.setSpacing(8)
            name_text = m.label + (f" ({m.side})" if m.side else "")
            name_lbl = QLabel(name_text)
            name_lbl.setStyleSheet("font-size: 11px; color: #ddd;")
            name_lbl.setToolTip(m.note)
            txt_color, bg_color = _STATUS_COLORS.get(m.status, _STATUS_COLORS["info"])
            val_lbl = QLabel(m.value_str)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: {txt_color};"
                f"background: {bg_color}; border-radius: 3px; padding: 1px 6px;"
            )
            val_lbl.setFixedWidth(78)
            row_layout.addWidget(name_lbl, stretch=1)
            row_layout.addWidget(val_lbl)
            layout.addWidget(row)

    # Continue button
    continue_btn = QPushButton(
        "Continuer →" if lang == "fr" else "Continue →"
    )
    continue_btn.setDefault(True)
    continue_btn.setStyleSheet(
        "font-size: 14px; padding: 8px 24px; font-weight: bold;"
    )
    layout.addWidget(continue_btn, alignment=Qt.AlignCenter)

    _wire_enter(dlg)
    continue_btn.clicked.connect(dlg.accept)

    dlg.show()
    dlg.raise_()
    center_fn(dlg)
    return dlg


def build_retry_dialog(
    d_landmarks: list[Landmark],
    lang: Language,
    parent: QWidget,
    center_fn: Callable[[QDialog], None],
    on_start: Callable[[], None],
    on_skip: Callable[[], None],
) -> QDialog:
    """Build, show, and return the grade-D retry introduction dialog.

    *on_start* is called when the student chooses to retry (or closes with X).
    *on_skip* is called when the student chooses to finish immediately.
    """
    n = len(d_landmarks)
    if lang == "fr":
        body = (
            f"{n} repère{'s ont' if n > 1 else ' a'} une note D.\n"
            "Vous allez les reprendre jusqu'à obtenir au moins C."
        )
    else:
        body = (
            f"{n} landmark{'s have' if n > 1 else ' has'} a grade D.\n"
            "You will redo them until you reach at least a C."
        )

    dlg = QDialog(parent)
    dlg.setWindowTitle(
        "Reprise des repères insuffisants"
        if lang == "fr"
        else "Retry: insufficient landmarks"
    )
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumWidth(400)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(14)
    layout.setContentsMargins(28, 28, 28, 24)

    title_lbl = QLabel(
        "Reprise des repères D"
        if lang == "fr"
        else "Retrying grade-D landmarks"
    )
    title_lbl.setAlignment(Qt.AlignCenter)
    title_lbl.setStyleSheet(
        "font-size: 18px; font-weight: bold; color: #c62828;"
    )
    layout.addWidget(title_lbl)

    body_lbl = QLabel(body)
    body_lbl.setWordWrap(True)
    body_lbl.setAlignment(Qt.AlignCenter)
    body_lbl.setStyleSheet("font-size: 13px; color: #444;")
    layout.addWidget(body_lbl)

    layout.addSpacing(4)

    btn_row = QHBoxLayout()

    start_btn = QPushButton(
        "Commencer la reprise →"
        if lang == "fr"
        else "Start retry →"
    )
    start_btn.setDefault(True)
    start_btn.setStyleSheet(
        "font-size: 13px; padding: 8px 16px; font-weight: bold;"
    )

    skip_btn = QPushButton(
        "Terminer quand même"
        if lang == "fr"
        else "Finish anyway"
    )
    skip_btn.setStyleSheet("font-size: 12px; padding: 8px 12px; color: #888;")

    btn_row.addWidget(start_btn)
    btn_row.addWidget(skip_btn)
    layout.addLayout(btn_row)

    _wire_enter(dlg)

    # Default finished handler (e.g. window X closed) → start retry
    dlg.finished.connect(lambda _: on_start())

    def _start() -> None:
        dlg.finished.disconnect()
        dlg.accept()
        on_start()

    def _skip() -> None:
        dlg.finished.disconnect()
        dlg.accept()
        on_skip()

    start_btn.clicked.connect(_start)
    skip_btn.clicked.connect(_skip)

    dlg.show()
    dlg.raise_()
    center_fn(dlg)
    return dlg


def build_category_transition_dialog(
    prev_cat: str | None,
    next_lm: Landmark,
    lang: Language,
    parent: QWidget,
    center_fn: Callable[[QDialog], None],
) -> QDialog | None:
    """Build and show a category transition dialog, or return None if no transition needed.

    Returns None when *prev_cat* equals *next_lm.category* (same category, skip).
    The caller must connect ``dlg.finished`` to resume the session.
    """
    if prev_cat is not None and prev_cat == next_lm.category:
        return None

    cat = next_lm.category
    cat_fr, cat_en = CATEGORY_LABELS.get(cat, (cat, cat))
    desc_fr, desc_en = _CATEGORY_DESCRIPTIONS.get(cat, ("", ""))
    cat_label = cat_fr if lang == "fr" else cat_en
    desc = desc_fr if lang == "fr" else desc_en
    theme_color = (
        next_lm.theme_color() if hasattr(next_lm, "theme_color") else "#607D8B"
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle(cat_label)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumWidth(400)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(16)
    layout.setContentsMargins(28, 32, 28, 24)

    # Category title in theme colour
    title_lbl = QLabel(cat_label)
    title_lbl.setWordWrap(True)
    title_lbl.setAlignment(Qt.AlignCenter)
    title_lbl.setStyleSheet(
        f"font-size: 22px; font-weight: bold; color: {theme_color};"
    )
    layout.addWidget(title_lbl)

    # Description sentence
    if desc:
        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        desc_lbl.setAlignment(Qt.AlignCenter)
        desc_lbl.setStyleSheet("font-size: 13px; color: #444; padding: 0 8px;")
        layout.addWidget(desc_lbl)

    layout.addSpacing(8)

    # Start button
    start_btn = QPushButton(
        "Commencer →" if lang == "fr" else "Start →"
    )
    start_btn.setDefault(True)
    start_btn.setStyleSheet(
        f"font-size: 14px; padding: 10px 28px; font-weight: bold; "
        f"background-color: {theme_color}; color: white; border-radius: 6px;"
    )
    layout.addWidget(start_btn, alignment=Qt.AlignCenter)

    _wire_enter(dlg)
    start_btn.clicked.connect(dlg.accept)

    dlg.show()
    dlg.raise_()
    center_fn(dlg)
    return dlg


def build_phase2_dialog(
    n_correct: int,
    n_inv: int,
    n_placement: int,
    lang: Language,
    parent: QWidget,
    center_fn: Callable[[QDialog], None],
) -> QDialog:
    """Build and show the Phase 1 → Phase 2 transition dialog.

    The caller must connect ``dlg.finished`` to start the placement phase.
    """
    if lang == "fr":
        title_text = "Phase 2 — Placement"
        body_text = (
            f"Identification terminée : {n_correct} / {n_inv} "
            f"correct{'s' if n_correct != 1 else ''}.\n\n"
            f"Maintenant placez {n_placement} "
            f"repère{'s' if n_placement != 1 else ''} sur le modèle 3D."
        )
        btn_text = "Commencer →"
    else:
        title_text = "Phase 2 — Placement"
        body_text = (
            f"Identification done: {n_correct} / {n_inv} correct.\n\n"
            f"Now place {n_placement} "
            f"landmark{'s' if n_placement != 1 else ''} on the 3D model."
        )
        btn_text = "Start →"

    dlg = QDialog(parent)
    dlg.setWindowTitle(title_text)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumWidth(380)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(16)
    layout.setContentsMargins(28, 28, 28, 24)

    title_lbl = QLabel(title_text)
    title_lbl.setAlignment(Qt.AlignCenter)
    title_lbl.setStyleSheet(
        "font-size: 18px; font-weight: bold; color: #1565c0;"
    )
    layout.addWidget(title_lbl)

    body_lbl = QLabel(body_text)
    body_lbl.setWordWrap(True)
    body_lbl.setAlignment(Qt.AlignCenter)
    body_lbl.setStyleSheet("font-size: 13px; color: #333;")
    layout.addWidget(body_lbl)

    layout.addSpacing(8)

    btn = QPushButton(btn_text)
    btn.setDefault(True)
    btn.setStyleSheet(
        "font-size: 14px; padding: 10px 28px; font-weight: bold; "
        "background-color: #1565c0; color: white; border-radius: 6px;"
    )
    layout.addWidget(btn, alignment=Qt.AlignCenter)

    _wire_enter(dlg)
    btn.clicked.connect(dlg.accept)

    dlg.show()
    dlg.raise_()
    center_fn(dlg)
    return dlg
