"""Login dialog: student ID, date of birth, teacher server URL."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from .i18n import Language, tr


class LoginDialog(QDialog):
    def __init__(self, lang: Language = "fr", parent=None) -> None:
        super().__init__(parent)
        self._lang = lang
        self.setWindowTitle(tr("login_title", lang))
        self.setMinimumWidth(360)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel(tr("app_title", self._lang))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)

        self._matricule_edit = QLineEdit()
        self._matricule_edit.setPlaceholderText("ex: 20123456")
        form.addRow(tr("matricule_label", self._lang), self._matricule_edit)

        self._date_edit = QDateEdit()
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDisplayFormat("yyyy-MM-dd")
        form.addRow(tr("birthdate_label", self._lang), self._date_edit)

        self._server_edit = QLineEdit()
        self._server_edit.setPlaceholderText("http://192.168.1.10:8765")
        form.addRow(tr("server_url_label", self._lang), self._server_edit)

        layout.addLayout(form)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: red; font-size: 12px;")
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("connect_button", self._lang))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def matricule(self) -> str:
        return self._matricule_edit.text().strip()

    @property
    def birthdate(self) -> date:
        qdate = self._date_edit.date()
        return date(qdate.year(), qdate.month(), qdate.day())

    @property
    def server_url(self) -> str:
        return self._server_edit.text().strip().rstrip("/")

    def set_error(self, message: str) -> None:
        self._error_label.setText(message)
