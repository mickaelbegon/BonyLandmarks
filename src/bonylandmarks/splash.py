"""Unified splash screen: app onboarding + login in a single dialog.

Replaces the former two-step flow (``LoginDialog`` then ``WelcomeDialog``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from .i18n import Language, tr


_DARK_BG = "#1a1a2e"
_CARD_BG = "#252540"
_TEXT_MAIN = "#e8e8f0"
_TEXT_MUTED = "#a0a0c0"
_ACCENT_BLUE = "#1565C0"

_GRADES = [
    ("A", "< 30 mm", "#2e7d32"),
    ("B", "< 80 mm", "#00695c"),
    ("C", "< 150 mm", "#e65100"),
    ("D", "≥ 150 mm", "#c62828"),
]


def _make_sep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("color: #333355; background-color: #333355;")
    sep.setFixedHeight(1)
    return sep


class SplashDialog(QDialog):
    """Single startup dialog: branding, steps, grading scale and login form.

    Returns:
        QDialog.Accepted (1) — normal start (credentials validated, or a
                               local ``.glb`` file was chosen)
        2                    — tutorial mode
        QDialog.Rejected (0) — user closed the window
    """

    TUTORIAL_RESULT: int = 2

    def __init__(
        self,
        students: list[dict],
        server_url: str | None,
        lang: Language = "fr",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._lang: Language = lang
        self._students = students
        self._server_url = server_url
        self._matricule: str = ""
        self._birthdate: date | None = None
        self._local_glb_bytes: bytes | None = None

        self.setFixedSize(700, 620)
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
            QLineEdit {{
                background-color: {_CARD_BG};
                color: {_TEXT_MAIN};
                border: 1px solid #333366;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 1px solid {_ACCENT_BLUE};
            }}
            """
        )
        self._build_ui()
        self._refresh_labels()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(32, 18, 32, 16)

        # ── Header — title + language toggle ─────────────────────────────────
        header = QHBoxLayout()
        header.addStretch()

        self._app_title = QLabel("BonyLandmarks")
        self._app_title.setAlignment(Qt.AlignCenter)
        self._app_title.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #7cb9ff;"
        )
        header.addWidget(self._app_title)
        header.addStretch()

        self._lang_btn = QPushButton()
        self._lang_btn.setFixedWidth(80)
        self._lang_btn.setAutoDefault(False)
        self._lang_btn.setStyleSheet(
            f"font-size: 11px; padding: 4px 8px; background-color: {_CARD_BG}; "
            f"color: {_TEXT_MUTED}; border: 1px solid #333366; border-radius: 4px;"
        )
        self._lang_btn.clicked.connect(self._toggle_lang)
        header.addWidget(self._lang_btn)
        root.addLayout(header)

        self._subtitle = QLabel()
        self._subtitle.setAlignment(Qt.AlignCenter)
        self._subtitle.setStyleSheet(f"font-size: 13px; color: {_TEXT_MUTED};")
        root.addWidget(self._subtitle)

        self._tagline = QLabel()
        self._tagline.setWordWrap(True)
        self._tagline.setAlignment(Qt.AlignCenter)
        self._tagline.setStyleSheet(
            f"font-size: 11px; font-style: italic; color: {_TEXT_MUTED};"
        )
        root.addWidget(self._tagline)

        root.addWidget(_make_sep())

        # ── Steps — three compact cards ──────────────────────────────────────
        steps_row = QHBoxLayout()
        steps_row.setSpacing(10)
        self._step_titles: list[QLabel] = []
        self._step_descs: list[QLabel] = []

        for icon in ("①", "②", "③"):
            card = QFrame()
            card.setMaximumHeight(120)
            card.setStyleSheet(
                f"QFrame {{ background-color: {_CARD_BG}; "
                f"border: 1px solid #333366; border-radius: 8px; }}"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(4)

            icon_lbl = QLabel(icon)
            icon_lbl.setAlignment(Qt.AlignCenter)
            icon_lbl.setStyleSheet(
                "font-size: 18px; background: transparent; border: none;"
            )
            card_layout.addWidget(icon_lbl)

            title_lbl = QLabel()
            title_lbl.setAlignment(Qt.AlignCenter)
            title_lbl.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #7cb9ff; "
                "background: transparent; border: none;"
            )
            card_layout.addWidget(title_lbl)
            self._step_titles.append(title_lbl)

            desc_lbl = QLabel()
            desc_lbl.setWordWrap(True)
            desc_lbl.setAlignment(Qt.AlignCenter)
            desc_lbl.setStyleSheet(
                f"font-size: 10px; color: {_TEXT_MUTED}; "
                f"background: transparent; border: none;"
            )
            card_layout.addWidget(desc_lbl)
            self._step_descs.append(desc_lbl)
            card_layout.addStretch()

            steps_row.addWidget(card, stretch=1)

        root.addLayout(steps_row)

        # ── Grading scale ────────────────────────────────────────────────────
        grade_row = QHBoxLayout()
        grade_row.setSpacing(8)

        self._grade_hdr = QLabel()
        self._grade_hdr.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {_TEXT_MUTED};"
        )
        grade_row.addWidget(self._grade_hdr)

        for letter, dist, color in _GRADES:
            badge = QLabel(f"  {letter}  {dist}  ")
            badge.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: white; "
                f"background-color: {color}; border-radius: 4px; padding: 2px 6px;"
            )
            grade_row.addWidget(badge)

        grade_row.addStretch()
        root.addLayout(grade_row)

        root.addWidget(_make_sep())

        # ── Login form ───────────────────────────────────────────────────────
        self._login_hdr = QLabel()
        self._login_hdr.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #7cb9ff;"
        )
        root.addWidget(self._login_hdr)

        form_row = QHBoxLayout()
        form_row.setSpacing(16)

        matricule_col = QVBoxLayout()
        matricule_col.setSpacing(4)
        self._matricule_label = QLabel()
        self._matricule_label.setStyleSheet(f"font-size: 11px; color: {_TEXT_MUTED};")
        matricule_col.addWidget(self._matricule_label)

        self._matricule_edit = QLineEdit()
        self._matricule_edit.setMaxLength(12)
        self._matricule_edit.setPlaceholderText("Ex: A3745")
        self._matricule_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"[A-Za-z0-9]{0,12}"))
        )
        self._matricule_edit.returnPressed.connect(self._on_start)
        matricule_col.addWidget(self._matricule_edit)
        form_row.addLayout(matricule_col, stretch=1)

        dob_col = QVBoxLayout()
        dob_col.setSpacing(4)
        self._dob_label = QLabel()
        self._dob_label.setStyleSheet(f"font-size: 11px; color: {_TEXT_MUTED};")
        dob_col.addWidget(self._dob_label)

        self._dob_edit = QLineEdit()
        self._dob_edit.setInputMask("99/99/9999;_")
        self._dob_edit.setEchoMode(QLineEdit.Password)
        self._dob_edit.returnPressed.connect(self._on_start)
        dob_col.addWidget(self._dob_edit)
        form_row.addLayout(dob_col, stretch=1)

        root.addLayout(form_row)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet(
            "color: #ff6b6b; font-size: 12px; font-weight: bold;"
        )
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        root.addWidget(self._error_label)

        self._offline_label = QLabel()
        self._offline_label.setWordWrap(True)
        self._offline_label.setStyleSheet(f"font-size: 11px; color: #ffa726;")
        if self._server_url is not None:
            self._offline_label.hide()
        root.addWidget(self._offline_label)

        root.addStretch()
        root.addWidget(_make_sep())

        # ── Action buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._local_btn = QPushButton()
        self._local_btn.setAutoDefault(False)
        self._local_btn.setStyleSheet(
            f"font-size: 11px; padding: 8px 12px; background-color: {_CARD_BG}; "
            f"color: {_TEXT_MUTED}; border: 1px solid #333366; border-radius: 4px;"
        )
        self._local_btn.clicked.connect(self._on_load_local)
        btn_row.addWidget(self._local_btn)

        btn_row.addStretch()

        self._tutorial_btn = QPushButton()
        self._tutorial_btn.setAutoDefault(False)
        self._tutorial_btn.setStyleSheet(
            f"font-size: 12px; padding: 8px 16px; "
            f"background-color: #37474f; color: {_TEXT_MAIN}; "
            f"border-radius: 4px; border: 1px solid #546e7a;"
        )
        self._tutorial_btn.clicked.connect(self._on_tutorial)
        btn_row.addWidget(self._tutorial_btn)

        self._start_btn = QPushButton()
        self._start_btn.setDefault(True)
        self._start_btn.setAutoDefault(True)
        self._start_btn.setStyleSheet(
            f"font-size: 14px; font-weight: bold; padding: 10px 28px; "
            f"background-color: {_ACCENT_BLUE}; color: white; "
            f"border-radius: 6px; border: none;"
        )
        self._start_btn.clicked.connect(self._on_start)
        if self._server_url is None:
            self._start_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)

        root.addLayout(btn_row)

        self._matricule_edit.setFocus()

    # ── Localised text ────────────────────────────────────────────────────────

    def _refresh_labels(self) -> None:
        fr = self._lang == "fr"

        self.setWindowTitle(
            "BonyLandmarks — Bienvenue" if fr else "BonyLandmarks — Welcome"
        )
        self._lang_btn.setText(tr("lang_toggle", self._lang))

        self._subtitle.setText(
            "Entraînement aux repères anatomiques"
            if fr else
            "Anatomical Landmark Training"
        )
        self._tagline.setText(
            "Développez votre précision dans la palpation des structures osseuses."
            if fr else
            "Build your precision in palpating bony structures."
        )

        if fr:
            steps = [
                ("Identification",
                 "Un point est montré sur le corps 3D. Identifiez-le dans la liste."),
                ("Placement",
                 "Le nom d'un repère est affiché. Cliquez à l'endroit exact."),
                ("Résultats",
                 "Votre précision et des infos cliniques. Les notes D sont reprises."),
            ]
        else:
            steps = [
                ("Identification",
                 "A point is shown on the 3D body. Identify it from the list."),
                ("Placement",
                 "A landmark name is shown. Click the exact location."),
                ("Results",
                 "Your accuracy plus clinical info. Grade D landmarks are retried."),
            ]
        for lbl, (title, _desc) in zip(self._step_titles, steps):
            lbl.setText(title)
        for lbl, (_title, desc) in zip(self._step_descs, steps):
            lbl.setText(desc)

        self._grade_hdr.setText("Notation :" if fr else "Grading:")
        self._login_hdr.setText("🔑  Connexion" if fr else "🔑  Login")
        self._matricule_label.setText(tr("matricule_label", self._lang))
        self._dob_label.setText(tr("dob_label", self._lang))

        self._local_btn.setText(
            "📂  Charger un fichier local..."
            if fr else
            "📂  Load a local file..."
        )
        self._tutorial_btn.setText("Mode tutoriel" if fr else "Tutorial mode")
        self._start_btn.setText("Commencer →" if fr else "Start →")

        self._offline_label.setText(
            "Mode hors-ligne — seul le chargement local est disponible."
            if fr else
            "Offline mode — only local file loading is available."
        )

    def _toggle_lang(self) -> None:
        self._lang = "en" if self._lang == "fr" else "fr"
        self._refresh_labels()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_login(self) -> bool:
        """Parse and store matricule + date of birth; show an error if invalid."""
        matricule_text = self._matricule_edit.text().strip().upper()
        if len(matricule_text) < 3 or not matricule_text.isalnum():
            self.set_error(tr("error_invalid_matricule", self._lang))
            return False

        # inputMask "99/99/9999;_" yields "DD/MM/YYYY" — drop separators/placeholders
        digits = (
            self._dob_edit.text()
            .replace("/", "").replace("_", "").replace(" ", "")
        )
        if len(digits) != 8 or not digits.isdigit():
            self.set_error(tr("error_invalid_dob", self._lang))
            return False

        try:
            parsed_date = date(int(digits[4:8]), int(digits[2:4]), int(digits[0:2]))
        except ValueError:
            self.set_error(tr("error_invalid_dob", self._lang))
            return False

        self._matricule = matricule_text
        self._birthdate = parsed_date
        self._local_glb_bytes = None
        self._error_label.hide()
        return True

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if not self._start_btn.isEnabled():
            return
        if self._validate_login():
            self.accept()

    def _on_tutorial(self) -> None:
        if self._validate_login():
            self.done(SplashDialog.TUTORIAL_RESULT)

    def _on_load_local(self) -> None:
        caption = (
            "Charger un scan BodyLoop"
            if self._lang == "fr" else
            "Load a BodyLoop scan"
        )
        path, _ = QFileDialog.getOpenFileName(self, caption, "", "GLB (*.glb)")
        if not path:
            return
        self._local_glb_bytes = Path(path).read_bytes()
        self._matricule = self._matricule_edit.text().strip().upper() or "LOCAL"
        self._birthdate = None
        self._error_label.hide()
        self.accept()

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def matricule(self) -> str:
        return self._matricule

    @property
    def birthdate(self) -> date | None:
        """Parsed date of birth, or ``None`` when a local file was chosen."""
        return self._birthdate

    @property
    def local_glb_bytes(self) -> bytes | None:
        return self._local_glb_bytes

    # ── Public methods ────────────────────────────────────────────────────────

    def set_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()

    # ── Static helper ─────────────────────────────────────────────────────────

    @staticmethod
    def dont_show() -> bool:
        """Always False — the splash carries the login form, so it never hides."""
        return False
