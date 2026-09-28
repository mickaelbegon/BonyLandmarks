"""Dialog de chargement d'un scan — fichier local ou BodyLoop direct.

Utilisé par main.py pour l'exercice "Scan partagé" et potentiellement
pour d'autres exercices nécessitant un GLB BodyLoop.

Retourne (glb_bytes, scan_name, model_name) ou lève Rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .ui.styles import BG, BTN_PRIMARY, CARD, MUTED, TEXT

_STYLE = f"""
QDialog, QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
}}
QTabWidget::pane {{ border: 1px solid #333355; }}
QTabBar::tab {{
    background: {CARD}; color: {MUTED};
    padding: 6px 16px; border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{ color: {TEXT}; background: #303060; }}
QGroupBox {{
    border: 1px solid #333355; border-radius: 6px;
    margin-top: 8px; padding: 8px;
    color: {MUTED}; font-size: 11px;
}}
QGroupBox::title {{ subcontrol-position: top left; left: 8px; }}
QLineEdit {{
    background: {CARD}; color: {TEXT};
    border: 1px solid #444466; border-radius: 4px;
    padding: 4px 8px;
}}
QComboBox {{
    background: {CARD}; color: {TEXT};
    border: 1px solid #444466; border-radius: 4px;
    padding: 4px 8px;
}}
QComboBox QAbstractItemView {{ background: {CARD}; color: {TEXT}; }}
QPushButton {{
    background: #2a2a44; color: {TEXT};
    border: 1px solid #444466; border-radius: 5px;
    padding: 5px 12px;
}}
QPushButton:hover {{ background: #3a3a64; }}
"""


class _FetchThread(QThread):
    """Thread de téléchargement GLB depuis BodyLoop (ne bloque pas l'UI)."""

    fetched = Signal(bytes, str)    # (glb_bytes, model_name)
    error = Signal(str)

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        viatar_id: int,
        model_name: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._username = username
        self._password = password
        self._viatar_id = viatar_id
        self._model_name = model_name

    def run(self) -> None:
        try:
            from .bodyloop_client import BodyLoopClient
            with BodyLoopClient(self._base_url) as client:
                client.login(self._username, self._password)
                glb = client.get_glb(self._viatar_id, self._model_name)
            self.fetched.emit(glb, self._model_name)
        except Exception as exc:
            self.error.emit(str(exc))


class ScanSourceDialog(QDialog):
    """Choix de la source GLB : fichier local ou BodyLoop.

    Attributs publics après acceptation :
    - ``glb_bytes`` : bytes bruts du GLB
    - ``scan_name`` : nom logique du scan (pour nommer les JSON de soumission)
    - ``model_name`` : ``"avatar_3d"`` ou ``"mesh_3d"``
    """

    def __init__(self, lang: str = "fr", parent=None) -> None:
        super().__init__(parent)
        self._lang = lang
        self.glb_bytes: Optional[bytes] = None
        self.scan_name: str = "scan"
        self.model_name: str = "avatar_3d"

        self.setWindowTitle(
            "Charger un scan GLB"
            if lang == "fr" else
            "Load a GLB scan"
        )
        self.setStyleSheet(_STYLE)
        self.setMinimumWidth(460)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel(
            "Choisissez la source du scan 3D"
            if self._lang == "fr" else
            "Choose the 3D scan source"
        )
        title.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {TEXT};")
        root.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_file_tab(), "📁 Fichier local")
        self._tabs.addTab(self._build_bodyloop_tab(), "🔗 BodyLoop")
        root.addWidget(self._tabs)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(f"font-size: 11px; color: #ff8080;")
        root.addWidget(self._status_lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText(
            "Charger" if self._lang == "fr" else "Load"
        )
        btns.button(QDialogButtonBox.Ok).setStyleSheet(BTN_PRIMARY)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        self._ok_btn = btns.button(QDialogButtonBox.Ok)
        root.addWidget(btns)

    # ── Onglet fichier local ──────────────────────────────────────────────────

    def _build_file_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        desc = QLabel(
            "Ouvrez un fichier .glb exporté depuis BodyLoop ou tout autre scan."
            if self._lang == "fr" else
            "Open a .glb file exported from BodyLoop or any other scan."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        vl.addWidget(desc)

        row = QWidget()
        hl_row = __import__("PySide6.QtWidgets", fromlist=["QHBoxLayout"]).QHBoxLayout(row)
        hl_row.setContentsMargins(0, 0, 0, 0)
        self._file_path_edit = QLineEdit()
        self._file_path_edit.setPlaceholderText(
            "Chemin du fichier .glb…"
            if self._lang == "fr" else
            "Path to .glb file…"
        )
        self._file_path_edit.setReadOnly(True)
        hl_row.addWidget(self._file_path_edit)

        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(self._browse_file)
        hl_row.addWidget(browse_btn)
        vl.addWidget(row)

        vl.addStretch()
        return w

    def _browse_file(self) -> None:
        caption = (
            "Ouvrir un scan GLB"
            if self._lang == "fr" else
            "Open a GLB scan"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, caption, "", "GLB (*.glb);;Tous (*)"
        )
        if path:
            self._file_path_edit.setText(path)

    # ── Onglet BodyLoop ───────────────────────────────────────────────────────

    def _build_bodyloop_tab(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(8)

        grp_conn = QGroupBox(
            "Connexion BodyLoop" if self._lang == "fr" else "BodyLoop connection"
        )
        form = QFormLayout(grp_conn)
        form.setSpacing(6)

        self._url_edit = QLineEdit("https://192.168.38.12")
        self._user_edit = QLineEdit()
        self._user_edit.setPlaceholderText("admin")
        self._pass_edit = QLineEdit()
        self._pass_edit.setEchoMode(QLineEdit.Password)

        form.addRow("URL serveur :", self._url_edit)
        form.addRow("Utilisateur :", self._user_edit)
        form.addRow("Mot de passe :", self._pass_edit)

        self._connect_btn = QPushButton(
            "Connexion & liste des scans"
            if self._lang == "fr" else
            "Connect & list scans"
        )
        self._connect_btn.clicked.connect(self._on_connect)
        form.addRow("", self._connect_btn)
        vl.addWidget(grp_conn)

        grp_scan = QGroupBox(
            "Sélection du scan" if self._lang == "fr" else "Scan selection"
        )
        form2 = QFormLayout(grp_scan)
        form2.setSpacing(6)

        self._viatar_combo = QComboBox()
        self._viatar_combo.setEnabled(False)
        self._viatar_combo.setPlaceholderText(
            "— connectez-vous d'abord —"
            if self._lang == "fr" else
            "— connect first —"
        )

        self._model_combo = QComboBox()
        self._model_combo.addItems(["avatar_3d", "mesh_3d"])
        self._model_combo.setToolTip(
            "avatar_3d : avec AutoMarkers (scan personal)\n"
            "mesh_3d   : sans markers (annotateur / scan partagé)"
        )

        form2.addRow("Scan :", self._viatar_combo)
        form2.addRow("Type GLB :", self._model_combo)
        vl.addWidget(grp_scan)

        # thread state
        self._fetch_thread: Optional[_FetchThread] = None
        self._viatars: list = []

        vl.addStretch()
        return w

    def _on_connect(self) -> None:
        """Lance la connexion et charge la liste des viatars."""
        from .bodyloop_client import BodyLoopClient

        url = self._url_edit.text().strip()
        user = self._user_edit.text().strip()
        pwd = self._pass_edit.text()

        if not url or not user or not pwd:
            self._set_error(
                "Remplissez URL, utilisateur et mot de passe."
                if self._lang == "fr" else
                "Please fill in URL, username and password."
            )
            return

        self._connect_btn.setEnabled(False)
        self._connect_btn.setText(
            "Connexion…" if self._lang == "fr" else "Connecting…"
        )
        self._status_lbl.setText("")

        try:
            with BodyLoopClient(url) as client:
                client.login(user, pwd)
                viatars = client.list_viatars()
        except Exception as exc:
            self._set_error(str(exc))
            self._connect_btn.setEnabled(True)
            self._connect_btn.setText(
                "Connexion & liste des scans"
                if self._lang == "fr" else
                "Connect & list scans"
            )
            return

        self._viatars = viatars
        self._viatar_combo.clear()
        for v in viatars:
            self._viatar_combo.addItem(str(v), userData=v.id)
        self._viatar_combo.setEnabled(bool(viatars))

        msg = (
            f"{len(viatars)} scan(s) disponible(s)."
            if self._lang == "fr" else
            f"{len(viatars)} scan(s) available."
        )
        self._status_lbl.setStyleSheet("font-size: 11px; color: #80ff80;")
        self._status_lbl.setText(msg)
        self._connect_btn.setEnabled(True)
        self._connect_btn.setText(
            "Actualiser" if self._lang == "fr" else "Refresh"
        )

    # ── Acceptation ──────────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        tab_idx = self._tabs.currentIndex()

        if tab_idx == 0:
            # Fichier local
            path = self._file_path_edit.text().strip()
            if not path:
                self._set_error(
                    "Choisissez un fichier GLB."
                    if self._lang == "fr" else
                    "Please select a GLB file."
                )
                return
            try:
                self.glb_bytes = Path(path).read_bytes()
                self.scan_name = Path(path).stem
                self.model_name = "avatar_3d"
                self.accept()
            except OSError as exc:
                self._set_error(str(exc))

        else:
            # BodyLoop
            if self._viatar_combo.currentIndex() < 0 or not self._viatars:
                self._set_error(
                    "Connectez-vous et sélectionnez un scan."
                    if self._lang == "fr" else
                    "Connect and select a scan."
                )
                return

            url = self._url_edit.text().strip()
            user = self._user_edit.text().strip()
            pwd = self._pass_edit.text()
            viatar_id = self._viatar_combo.currentData()
            model_name = self._model_combo.currentText()
            viatar_label = self._viatar_combo.currentText()

            self._ok_btn.setEnabled(False)
            self._status_lbl.setStyleSheet("font-size: 11px; color: #80c0ff;")
            self._status_lbl.setText(
                "Téléchargement en cours…"
                if self._lang == "fr" else
                "Downloading…"
            )

            self._fetch_thread = _FetchThread(
                url, user, pwd, viatar_id, model_name, parent=self
            )
            self._fetch_thread.fetched.connect(
                lambda glb, mn, label=viatar_label: self._on_fetched(glb, mn, label)
            )
            self._fetch_thread.error.connect(self._on_fetch_error)
            self._fetch_thread.start()

    def _on_fetched(self, glb_bytes: bytes, model_name: str, label: str) -> None:
        self.glb_bytes = glb_bytes
        self.scan_name = label.replace(" ", "_")[:40]
        self.model_name = model_name
        self.accept()

    def _on_fetch_error(self, msg: str) -> None:
        self._set_error(msg)
        self._ok_btn.setEnabled(True)

    def _set_error(self, msg: str) -> None:
        self._status_lbl.setStyleSheet("font-size: 11px; color: #ff8080;")
        self._status_lbl.setText(msg)
