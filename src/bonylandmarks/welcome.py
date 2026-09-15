"""Welcome dialog displayed at application startup, before the landmark session."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .i18n import Language


_DARK_BG = "#1a1a2e"
_CARD_BG = "#252540"
_TEXT_MAIN = "#e8e8f0"
_TEXT_MUTED = "#a0a0c0"
_ACCENT_BLUE = "#1565C0"
_SETTINGS_ORG = "BonyLandmarks"
_SETTINGS_APP = "BonyLandmarks"
_SETTINGS_KEY = "welcome/dont_show"


def _make_sep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("color: #333355; background-color: #333355;")
    sep.setFixedHeight(1)
    return sep


class WelcomeDialog(QDialog):
    """Splash/tutorial dialog shown before each session begins.

    Returns:
        QDialog.Accepted (1) — normal start
        2                    — tutorial mode
        QDialog.Rejected (0) — user closed the window
    """

    TUTORIAL_RESULT: int = 2

    def __init__(self, lang: Language = "fr", parent=None) -> None:
        super().__init__(parent)
        self._lang = lang
        self.setWindowTitle(
            "BonyLandmarks — Bienvenue" if lang == "fr" else "BonyLandmarks — Welcome"
        )
        self.setFixedSize(700, 560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {_DARK_BG};
            }}
            QLabel {{
                color: {_TEXT_MAIN};
                background: transparent;
            }}
            QCheckBox {{
                color: {_TEXT_MUTED};
                font-size: 11px;
            }}
            """
        )
        self._build_ui()

        # Auto-accept when the user previously checked "don't show again"
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        if settings.value(_SETTINGS_KEY, False, type=bool):
            QTimer.singleShot(0, self.accept)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        fr = self._lang == "fr"

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(32, 22, 32, 18)

        # ── Section 1 — Title / Subtitle / Tagline ────────────────────────────
        app_title = QLabel("BonyLandmarks")
        app_title.setAlignment(Qt.AlignCenter)
        app_title.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: #7cb9ff;"
        )
        root.addWidget(app_title)

        subtitle = QLabel(
            "Entraînement aux repères anatomiques"
            if fr else
            "Anatomical Landmark Training"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"font-size: 13px; color: {_TEXT_MUTED};")
        root.addWidget(subtitle)

        tagline = QLabel(
            "Développez votre précision dans la palpation des structures osseuses."
            if fr else
            "Build your precision in palpating bony structures."
        )
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet(
            f"font-size: 11px; font-style: italic; color: {_TEXT_MUTED};"
        )
        root.addWidget(tagline)

        root.addWidget(_make_sep())

        # ── Section 2 — Connection ────────────────────────────────────────────
        connect_lbl = QLabel(
            "🔐  Entrez votre matricule et date de naissance "
            "pour accéder à votre avatar 3D personnalisé."
            if fr else
            "🔐  Enter your student ID and date of birth "
            "to access your personalised 3D avatar."
        )
        connect_lbl.setWordWrap(True)
        connect_lbl.setStyleSheet(f"font-size: 12px; color: {_TEXT_MAIN}; padding: 2px 0;")
        root.addWidget(connect_lbl)

        # ── Section 3 — Steps (3 cards) ───────────────────────────────────────
        if fr:
            steps = [
                (
                    "①",
                    "Identification",
                    "On vous montre un point sur le corps 3D. "
                    "Identifiez-le dans la liste.",
                ),
                (
                    "②",
                    "Placement",
                    "Le nom d'un repère est affiché. "
                    "Cliquez à l'endroit exact sur le corps.",
                ),
                (
                    "③",
                    "Résultats",
                    "Après chaque repère, vous voyez votre précision et des "
                    "informations cliniques. Les notes D sont reprises jusqu'à obtenir C.",
                ),
            ]
        else:
            steps = [
                (
                    "①",
                    "Identification",
                    "A point is shown on the 3D body. "
                    "Identify it from the list.",
                ),
                (
                    "②",
                    "Placement",
                    "A landmark name is shown. "
                    "Click the exact location on the body.",
                ),
                (
                    "③",
                    "Results",
                    "After each landmark, see your accuracy and clinical info. "
                    "Grade D landmarks are retried until you reach C.",
                ),
            ]

        steps_row = QHBoxLayout()
        steps_row.setSpacing(10)
        for icon, card_title, desc in steps:
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{ background-color: {_CARD_BG}; "
                f"border: 1px solid #333366; border-radius: 8px; }}"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 14, 12, 14)
            card_layout.setSpacing(6)

            icon_lbl = QLabel(icon)
            icon_lbl.setAlignment(Qt.AlignCenter)
            icon_lbl.setStyleSheet(
                "font-size: 22px; background: transparent; border: none;"
            )
            card_layout.addWidget(icon_lbl)

            title_lbl = QLabel(card_title)
            title_lbl.setAlignment(Qt.AlignCenter)
            title_lbl.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #7cb9ff; "
                "background: transparent; border: none;"
            )
            card_layout.addWidget(title_lbl)

            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setAlignment(Qt.AlignCenter)
            desc_lbl.setStyleSheet(
                f"font-size: 11px; color: {_TEXT_MUTED}; "
                f"background: transparent; border: none;"
            )
            card_layout.addWidget(desc_lbl)
            card_layout.addStretch()

            steps_row.addWidget(card, stretch=1)

        root.addLayout(steps_row)

        # ── Section 4 — Grading ───────────────────────────────────────────────
        grade_row = QHBoxLayout()
        grade_row.setSpacing(8)

        grade_hdr = QLabel("Notation :" if fr else "Grading:")
        grade_hdr.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {_TEXT_MUTED};"
        )
        grade_row.addWidget(grade_hdr)

        grades = [
            ("A", "< 30 mm", "#2e7d32"),
            ("B", "< 80 mm", "#00695c"),
            ("C", "< 150 mm", "#e65100"),
            ("D", "≥ 150 mm", "#c62828"),
        ]
        for letter, dist, color in grades:
            badge = QLabel(f"  {letter}  {dist}  ")
            badge.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: white; "
                f"background-color: {color}; border-radius: 4px; padding: 2px 6px;"
            )
            grade_row.addWidget(badge)

        grade_row.addStretch()
        root.addLayout(grade_row)

        root.addWidget(_make_sep())

        # ── Section 5 — Buttons + checkbox ───────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._no_show_cb = QCheckBox(
            "Ne plus afficher ce message" if fr else "Don't show this again"
        )
        self._no_show_cb.setChecked(
            settings.value(_SETTINGS_KEY, False, type=bool)
        )
        btn_row.addWidget(self._no_show_cb)
        btn_row.addStretch()

        tutorial_btn = QPushButton(
            "Mode tutoriel" if fr else "Tutorial mode"
        )
        tutorial_btn.setStyleSheet(
            f"font-size: 12px; padding: 8px 16px; "
            f"background-color: #37474f; color: {_TEXT_MAIN}; "
            f"border-radius: 4px; border: 1px solid #546e7a;"
        )
        tutorial_btn.clicked.connect(self._on_tutorial)
        btn_row.addWidget(tutorial_btn)

        start_btn = QPushButton("Commencer" if fr else "Start")
        start_btn.setDefault(True)
        start_btn.setStyleSheet(
            f"font-size: 14px; font-weight: bold; padding: 10px 32px; "
            f"background-color: {_ACCENT_BLUE}; color: white; "
            f"border-radius: 6px; border: none;"
        )
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(start_btn)

        root.addLayout(btn_row)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _save_settings(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.setValue(_SETTINGS_KEY, self._no_show_cb.isChecked())

    def _on_start(self) -> None:
        self._save_settings()
        self.accept()  # returns QDialog.Accepted == 1

    def _on_tutorial(self) -> None:
        self._save_settings()
        self.done(WelcomeDialog.TUTORIAL_RESULT)  # returns 2

    # ── Static helper ─────────────────────────────────────────────────────────

    @staticmethod
    def dont_show() -> bool:
        """Return True if the user has permanently dismissed the welcome dialog."""
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        return bool(settings.value(_SETTINGS_KEY, False, type=bool))
