"""Login dialog: student name dropdown, student ID, date of birth."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from .i18n import Language, tr


class LoginDialog(QDialog):
    def __init__(
        self,
        students: list[dict],
        server_url: str | None,
        lang: Language = "fr",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._lang = lang
        self._students = students
        self._server_url = server_url
        self._matricule: str = ""
        self._birthdate: date | None = None
        self.setWindowTitle(tr("login_title", lang))
        self.setMinimumWidth(400)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title row with language toggle on the right
        title_row = QHBoxLayout()
        self._title_label = QLabel(tr("app_title", self._lang))
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_row.addWidget(self._title_label, stretch=1)

        self._lang_btn = QPushButton(tr("lang_toggle", self._lang))
        self._lang_btn.setFixedWidth(80)
        self._lang_btn.clicked.connect(self._toggle_lang)
        title_row.addWidget(self._lang_btn)
        layout.addLayout(title_row)

        # Server config error banner (shown only when server_url is None)
        if self._server_url is None:
            self._server_error_label: QLabel | None = QLabel(
                tr("error_server_config", self._lang)
            )
            self._server_error_label.setStyleSheet(
                "color: red; font-weight: bold; padding: 4px;"
            )
            self._server_error_label.setWordWrap(True)
            layout.addWidget(self._server_error_label)
        else:
            self._server_error_label = None

        # Student name selector
        self._select_name_label = QLabel(tr("select_name_label", self._lang))
        layout.addWidget(self._select_name_label)

        self._name_combo = QComboBox()
        self._name_combo.setEditable(True)
        self._name_combo.setInsertPolicy(QComboBox.NoInsert)
        items = [f"{s['nom']} {s['prenom']}" for s in self._students]
        self._name_combo.addItems(items)

        completer = QCompleter(items)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._name_combo.setCompleter(completer)
        layout.addWidget(self._name_combo)

        # Matricule (student ID)
        self._matricule_label = QLabel(tr("matricule_label", self._lang))
        layout.addWidget(self._matricule_label)

        self._matricule_edit = QLineEdit()
        self._matricule_edit.setEchoMode(QLineEdit.Password)
        self._matricule_edit.setMaxLength(8)
        self._matricule_edit.setPlaceholderText("••••••••")
        self._matricule_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"\d{0,8}"))
        )
        layout.addWidget(self._matricule_edit)

        # Date of birth
        self._dob_label = QLabel(tr("dob_label", self._lang))
        layout.addWidget(self._dob_label)

        self._dob_edit = QLineEdit()
        self._dob_edit.setInputMask("99/99/9999;_")
        self._dob_edit.setEchoMode(QLineEdit.Password)
        layout.addWidget(self._dob_edit)

        # Error label (hidden until an error occurs)
        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: red; font-size: 12px;")
        self._error_label.setWordWrap(True)
        self._error_label.hide()
        layout.addWidget(self._error_label)

        # Connect button
        self._connect_btn = QPushButton(tr("connect_button", self._lang))
        self._connect_btn.clicked.connect(self._on_connect)
        if self._server_url is None:
            self._connect_btn.setEnabled(False)
        layout.addWidget(self._connect_btn)

    def _on_connect(self) -> None:
        matricule_text = self._matricule_edit.text().strip()
        if len(matricule_text) != 8 or not matricule_text.isdigit():
            self.set_error(tr("error_invalid_matricule", self._lang))
            return

        # inputMask "99/99/9999;_" yields "DD/MM/YYYY" — strip separators and placeholders
        dob_raw = self._dob_edit.text()
        digits = dob_raw.replace("/", "").replace("_", "").replace(" ", "")
        if len(digits) != 8 or not digits.isdigit():
            self.set_error(tr("error_invalid_dob", self._lang))
            return

        dd = int(digits[0:2])
        mm = int(digits[2:4])
        yyyy = int(digits[4:8])

        try:
            parsed_date = date(yyyy, mm, dd)
        except ValueError:
            self.set_error(tr("error_invalid_dob", self._lang))
            return

        self._matricule = matricule_text
        self._birthdate = parsed_date
        self._error_label.hide()
        self.accept()

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def matricule(self) -> str:
        return self._matricule

    @property
    def birthdate(self) -> date:
        if self._birthdate is None:
            raise RuntimeError("birthdate not set — dialog has not been accepted yet")
        return self._birthdate

    @property
    def selected_student_name(self) -> str:
        return self._name_combo.currentText()

    # ── Public methods ────────────────────────────────────────────────────────

    def set_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _toggle_lang(self) -> None:
        self._lang = "en" if self._lang == "fr" else "fr"
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        self.setWindowTitle(tr("login_title", self._lang))
        self._title_label.setText(tr("app_title", self._lang))
        self._lang_btn.setText(tr("lang_toggle", self._lang))
        self._select_name_label.setText(tr("select_name_label", self._lang))
        self._matricule_label.setText(tr("matricule_label", self._lang))
        self._dob_label.setText(tr("dob_label", self._lang))
        self._connect_btn.setText(tr("connect_button", self._lang))
        if self._server_error_label is not None:
            self._server_error_label.setText(tr("error_server_config", self._lang))
