"""Exercice « Anatomie 3D — os et muscles » : identifier / localiser une structure.

Deux modes sur le squelette complet ou le système musculaire BodyParts3D
(sans connexion serveur) :

* **Identifier** — une structure est colorée en cyan vif, tout le reste est
  opaque et neutre ; l'étudiant la nomme (niveau 1 : choix multiple, niveau 2 :
  liste complète filtrable, niveau 3 : saisie libre tolérante aux fautes).
* **Localiser** — un nom est donné, l'étudiant clique la structure sur la scène
  (cible verte, clic erroné rouge, côté opposé orange = demi-point).

Muscles : la scène montre le squelette (gris clair) + les muscles des régions
choisies.  Quand la cible est un muscle profond, la couche superficielle est
retirée (bandeau d'information dans l'interface).

Architecture
------------
* logique de quiz : :mod:`structure_quiz_engine` (pur Python) ;
* catalogue / meshes : :mod:`anatomy_catalog` ;
* géométrie fusionnée, couleurs, picking, caméra : :mod:`structure_scene` ;
* ce module : interface Qt + rendu PyVista.

Attribution : BodyParts3D, © The Database Center for Life Science, licensed
under CC Attribution-Share Alike 2.1 Japan.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import anatomy_catalog as _cat
from . import structure_scene as _sc
from .scene3d import CameraAxes, pan_camera, reset_view, set_view, view_top
from .scoring import grade_color
from .structure_quiz_engine import (
    Level,
    Mode,
    QuizConfig,
    QuizSession,
    Verdict,
    display_name,
    grade_letter,
    normalize,
)
from .ui.styles import (
    BASE_STYLE as _BASE_STYLE,
    BLUE as _BLUE,
    BTN_PRIMARY as _BTN_PRIMARY,
    CARD as _CARD,
    MUTED as _MUTED,
    TEXT as _TEXT,
)
from .widgets.nav_overlay import NavOverlay

ATTRIBUTION = (
    "BodyParts3D, © The Database Center for Life Science licensed under "
    "CC Attribution-Share Alike 2.1 Japan"
)
ATTRIBUTION_URL = "https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html"

# ─── Chaînes (FR, EN) ─────────────────────────────────────────────────────────

_S: dict[str, tuple[str, str]] = {
    "title": ("Anatomie 3D — os et muscles", "3D Anatomy — bones and muscles"),
    "subtitle": (
        "Identifiez ou localisez des structures sur le squelette et le système musculaire.",
        "Identify or locate structures on the skeleton and the muscular system.",
    ),
    "kind": ("Type de structures", "Structure type"),
    "kind_bone": ("Os", "Bones"),
    "kind_muscle": ("Muscles", "Muscles"),
    "mode": ("Exercice", "Exercise"),
    "mode_identify": (
        "Identifier — une structure est colorée, donnez son nom",
        "Identify — a structure is highlighted, name it",
    ),
    "mode_locate": (
        "Localiser — un nom est donné, cliquez la structure",
        "Locate — a name is given, click the structure",
    ),
    "level": ("Niveau (réponse)", "Level (answer format)"),
    "level1": ("1 · Choix multiple (4 réponses)", "1 · Multiple choice (4 answers)"),
    "level2": ("2 · Liste complète (avec filtre)", "2 · Full list (with filter)"),
    "level3": ("3 · Saisie libre", "3 · Free typing"),
    "regions": ("Régions", "Regions"),
    "tier": ("Détail", "Detail"),
    "tier1": ("Structures majeures", "Major structures"),
    "tier2": ("Majeures + détaillées", "Major + detailed"),
    "n_questions": ("Nombre de questions", "Number of questions"),
    "superficial_only": (
        "Cibles superficielles uniquement (localiser)",
        "Superficial targets only (locate)",
    ),
    "superficial_only_tip": (
        "Les muscles profonds sont cachés sous les superficiels : on ne demande "
        "que les muscles que l'on peut cliquer.",
        "Deep muscles are hidden under superficial ones: only clickable muscles are asked.",
    ),
    "available": (
        "{n} structure(s) disponible(s) dans la sélection",
        "{n} structure(s) available in the selection",
    ),
    "no_mesh": (
        "Aucun mesh {kind} n'est installé. Lancez « python scripts/fetch_meshes.py » "
        "pour les télécharger (ou « python scripts/import_bp3d.py --kind {arg} » pour "
        "les régénérer depuis BodyParts3D).",
        "No {kind} mesh is installed. Run “python scripts/fetch_meshes.py” to download "
        "them (or “python scripts/import_bp3d.py --kind {arg}” to regenerate them "
        "from BodyParts3D).",
    ),
    "no_question": (
        "Aucune question possible avec cette sélection.",
        "No question is possible with this selection.",
    ),
    "kind_word_bone": ("d'os", "bone"),
    "kind_word_muscle": ("de muscle", "muscle"),
    "start": ("Commencer", "Start"),
    "back_menu": ("Retour au menu", "Back to menu"),
    "loading": ("Chargement des meshes… {i}/{n}", "Loading meshes… {i}/{n}"),
    "question_n": ("Question {i} / {n}", "Question {i} / {n}"),
    "score_now": ("Score : {s} / {m}", "Score: {s} / {m}"),
    "q_identify": (
        "Quelle est la structure colorée en cyan ?",
        "Which structure is highlighted in cyan?",
    ),
    "q_identify_side": (
        "Précisez le côté (gauche / droite) quand il existe.",
        "Give the side (left / right) when there is one.",
    ),
    "q_locate": ("Cliquez sur :", "Click on:"),
    "q_locate_tip": (
        "Cliquez sur la structure dans la vue 3D (glisser = tourner). "
        "Cible : vert · erreur : rouge · côté opposé : orange.",
        "Click the structure in the 3D view (drag = rotate). "
        "Target: green · error: red · opposite side: orange.",
    ),
    "filter_ph": ("Filtrer la liste…", "Filter the list…"),
    "text_ph": ("Tapez le nom de la structure…", "Type the structure name…"),
    "validate": ("Valider", "Submit"),
    "skip": ("Passer", "Skip"),
    "abandon": ("Abandonner", "Give up"),
    "abandon_confirm": (
        "Abandonner le quiz ? Les questions restantes compteront 0.",
        "Give up the quiz? Remaining questions will score 0.",
    ),
    "next": ("Suivant ▶", "Next ▶"),
    "see_results": ("Voir le résultat ▶", "See results ▶"),
    "correct": ("Correct !", "Correct!"),
    "half": (
        "Bon nom, mauvais côté (0,5 point)",
        "Right name, wrong side (0.5 point)",
    ),
    "wrong": ("Incorrect", "Incorrect"),
    "skipped": ("Question passée", "Question skipped"),
    "typo": ("(faute de frappe acceptée)", "(typo accepted)"),
    "answer_is": ("Réponse : {fr}", "Answer: {fr}"),
    "you_clicked": ("Vous avez cliqué : {name}", "You clicked: {name}"),
    "you_answered": ("Votre réponse : {name}", "Your answer: {name}"),
    "clicked_bone": (
        "C'est un os : cliquez sur un muscle.",
        "That is a bone: click a muscle.",
    ),
    "clicked_nothing": (
        "Aucune structure ici : cliquez sur une structure.",
        "No structure here: click a structure.",
    ),
    "layer_superficial": ("couche superficielle", "superficial layer"),
    "layer_deep": ("couche profonde", "deep layer"),
    "layer_removed": (
        "Couche superficielle retirée ({region}) : le muscle cible est profond.",
        "Superficial layer removed ({region}): the target muscle is deep.",
    ),
    "layer_removed_all": (
        "Couche superficielle retirée : le muscle cible est profond.",
        "Superficial layer removed: the target muscle is deep.",
    ),
    "legend_identify": (
        "Cible en cyan · squelette {bones}",
        "Target in cyan · skeleton {bones}",
    ),
    "results": ("Résultat", "Results"),
    "score_line": (
        "{s} / {m} points ({p} %)",
        "{s} / {m} points ({p} %)",
    ),
    "grade": ("Note", "Grade"),
    "col_num": ("#", "#"),
    "col_target": ("Structure", "Structure"),
    "col_given": ("Votre réponse", "Your answer"),
    "col_points": ("Points", "Points"),
    "col_status": ("Résultat", "Result"),
    "none_given": ("—", "—"),
    "errors": ("Types d'erreurs", "Error types"),
    "err_name": ("Mauvais nom : {n}", "Wrong name: {n}"),
    "err_side": ("Mauvais côté : {n}", "Wrong side: {n}"),
    "err_skip": ("Passées : {n}", "Skipped: {n}"),
    "frequent": ("Confusions fréquentes", "Frequent confusions"),
    "replay": ("Rejouer les ratés ({n})", "Replay the misses ({n})"),
    "new_config": ("Nouvelle configuration", "New configuration"),
    "finish": ("Terminer", "Finish"),
    "perfect": ("Sans faute !", "Perfect!"),
    "not_enough": (
        "Pas assez de structures pour ce quiz.", "Not enough structures for this quiz.",
    ),
    "region_all": ("Toutes", "All"),
}

_REGION_LABELS: dict[str, tuple[str, str]] = {
    "skull": ("Crâne", "Skull"),
    "thorax": ("Thorax", "Thorax"),
    "spine": ("Colonne vertébrale", "Spine"),
    "upper_limb": ("Membre supérieur", "Upper limb"),
    "pelvis": ("Bassin", "Pelvis"),
    "lower_limb": ("Membre inférieur", "Lower limb"),
    "trunk": ("Tronc", "Trunk"),
    "head_neck": ("Tête et cou", "Head and neck"),
}


def _tt(lang: str, key: str, **kw) -> str:
    """Traduction d'une chaîne du module (repli sur la clé)."""
    row = _S.get(key)
    if row is None:
        return key
    text = row[1] if lang == "en" else row[0]
    return text.format(**kw) if kw else text


def region_label(region: str, lang: str = "fr") -> str:
    row = _REGION_LABELS.get(region)
    if row is None:
        return region
    return row[1] if lang == "en" else row[0]


# ─── Styles locaux ────────────────────────────────────────────────────────────

_BTN_ANSWER = (
    f"QPushButton {{ background: {_CARD}; color: {_TEXT}; border: 1px solid #3a3a6a; "
    f"border-radius: 6px; padding: 8px 10px; font-size: 12px; text-align: left; }}"
    f"QPushButton:hover {{ background: #303060; }}"
    f"QPushButton:disabled {{ color: #8888a8; }}"
)
_BTN_ANSWER_OK = (
    "QPushButton { background: #1a4a1a; color: #60ff60; border: 2px solid #60ff60; "
    "border-radius: 6px; padding: 8px 10px; font-size: 12px; text-align: left; }"
)
_BTN_ANSWER_KO = (
    "QPushButton { background: #4a1a1a; color: #ff6060; border: 1px solid #ff6060; "
    "border-radius: 6px; padding: 8px 10px; font-size: 12px; text-align: left; }"
)
_BTN_SECONDARY = (
    "QPushButton { background: #2a2a44; color: #c8c8e0; border: 1px solid #3a3a6a; "
    "border-radius: 6px; padding: 7px 12px; font-size: 12px; }"
    "QPushButton:hover { background: #34345a; }"
    "QPushButton:disabled { color: #606080; }"
)
_BTN_QUIT = (
    "QPushButton { background: #3a1a1a; color: #ff8080; border: 1px solid #6a3a3a; "
    "border-radius: 5px; padding: 5px 10px; font-size: 11px; }"
    "QPushButton:hover { background: #5a2a2a; }"
)
_INPUT_STYLE = (
    f"QLineEdit, QSpinBox, QComboBox {{ background: {_CARD}; color: {_TEXT}; "
    f"border: 1px solid #3a3a6a; border-radius: 4px; padding: 5px 8px; font-size: 13px; }}"
    f"QLineEdit:focus {{ border: 1px solid {_BLUE}; }}"
    f"QListWidget {{ background: {_CARD}; color: {_TEXT}; border: 1px solid #3a3a6a; "
    f"border-radius: 4px; font-size: 12px; }}"
    f"QListWidget::item:selected {{ background: #3a5faa; color: white; }}"
    f"QComboBox QAbstractItemView {{ background: {_CARD}; color: {_TEXT}; }}"
)

_MAX_CHOICES = 6
_CLICK_TOLERANCE_PX2 = 25   # distance² max entre appui et relâchement pour un « clic »


def _hex(rgb) -> str:
    return _sc.rgb_hex(rgb)


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet("QFrame { color: #333355; background: #333355; }")
    return f


# ─── Widget principal ─────────────────────────────────────────────────────────

_PAGE_CONFIG, _PAGE_QUIZ, _PAGE_RESULTS = 0, 1, 2


class StructureQuizExercise(QWidget):
    """Exercice complet : configuration → quiz 3D → résultats.

    Parameters
    ----------
    lang:
        ``"fr"`` ou ``"en"``.
    kind, mode:
        Présélections de l'écran de configuration (``"bone"`` / ``"muscle"``,
        ``Mode`` ou ``"identify"`` / ``"locate"``).
    structures:
        Catalogue alternatif (tests) : liste de ``Structure`` dont ``mesh_file``
        peut être un chemin absolu.  Par défaut, le catalogue BodyParts3D.

    Signals
    -------
    exercise_complete : l'étudiant quitte l'exercice.
    """

    exercise_complete = Signal()

    def __init__(
        self,
        lang: str = "fr",
        kind: str | None = None,
        mode: Mode | str | None = None,
        parent=None,
        *,
        structures: Sequence | None = None,
    ) -> None:
        super().__init__(parent)
        self._lang = "en" if lang == "en" else "fr"
        self.setStyleSheet(_BASE_STYLE + _INPUT_STYLE)
        self._override = list(structures) if structures is not None else None
        self._closed = False

        # Session / scène
        self._session: QuizSession | None = None
        self._config: QuizConfig | None = None
        self._answered = False
        self._awaiting_pick = False
        self._layers: dict[str, _sc.SceneLayer] = {}
        self._scene_key: tuple | None = None
        self._body_ref: np.ndarray = np.zeros((0, 6))
        self._merged_cache: dict[tuple, _sc.MergedMesh] = {}
        self._by_id: dict[str, object] = {}
        self._pool_by_id: dict[str, object] = {}
        self._cam_axes = CameraAxes(up=2, front=1, side=0, centers=[0.0, 0.0, 900.0],
                                    distance=3500.0)
        self._nav_overlay: NavOverlay | None = None
        self._plotter = None
        self._press_xy: tuple[int, int] | None = None
        self._press_obs = self._release_obs = None
        self._press_cb = self._release_cb = None
        self._first_locate_camera = True
        self.timings: dict[str, float] = {}   # dernières mesures (ms)

        self._build_ui()
        self._preset(kind, mode)
        self._refresh_config()

    # ── API test / lecture ────────────────────────────────────────────────────

    @property
    def session(self) -> QuizSession | None:
        return self._session

    @property
    def page(self) -> int:
        return self._stack.currentIndex()

    # ── Catalogue ─────────────────────────────────────────────────────────────

    def _available(self, kind: str) -> list:
        if self._override is not None:
            return [s for s in self._override
                    if s.kind == kind and _cat.mesh_path(s) is not None]
        return _cat.available(kind)

    def _selected_regions_for(self, kind: str) -> list[str]:
        seen: list[str] = []
        for s in self._available(kind):
            if s.region not in seen:
                seen.append(s.region)
        return seen

    # ── Construction de l'interface ───────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._stack = QStackedWidget()
        root.addWidget(self._stack)
        self._stack.addWidget(self._build_config_page())
        self._stack.addWidget(self._build_quiz_page())
        self._results_page = QWidget()
        QVBoxLayout(self._results_page).setContentsMargins(24, 16, 24, 16)
        self._stack.addWidget(self._results_page)

    # -- page de configuration ------------------------------------------------

    def _build_config_page(self) -> QWidget:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        page.setWidget(inner)
        outer = QVBoxLayout(inner)
        outer.setContentsMargins(0, 16, 0, 16)

        card = QWidget()
        card.setMaximumWidth(720)
        card.setStyleSheet(f"QWidget#cfgcard {{ background: {_CARD}; border-radius: 10px; }}")
        card.setObjectName("cfgcard")
        outer.addWidget(card, alignment=Qt.AlignHCenter)
        vl = QVBoxLayout(card)
        vl.setContentsMargins(28, 22, 28, 20)
        vl.setSpacing(10)

        self._cfg_title = QLabel()
        self._cfg_title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {_BLUE};")
        vl.addWidget(self._cfg_title)
        self._cfg_sub = QLabel()
        self._cfg_sub.setWordWrap(True)
        self._cfg_sub.setStyleSheet(f"font-size: 12px; color: {_MUTED};")
        vl.addWidget(self._cfg_sub)
        vl.addWidget(_sep())

        # Type
        self._kind_hdr = self._hdr()
        vl.addWidget(self._kind_hdr)
        row = QHBoxLayout()
        self._kind_group = QButtonGroup(self)
        self._rb_kind: dict[str, QRadioButton] = {}
        for k in ("bone", "muscle"):
            rb = QRadioButton()
            self._kind_group.addButton(rb)
            self._rb_kind[k] = rb
            row.addWidget(rb)
            rb.toggled.connect(self._on_config_changed)
        row.addStretch()
        vl.addLayout(row)

        # Mode
        self._mode_hdr = self._hdr()
        vl.addWidget(self._mode_hdr)
        self._mode_group = QButtonGroup(self)
        self._rb_mode: dict[str, QRadioButton] = {}
        for m in ("identify", "locate"):
            rb = QRadioButton()
            self._mode_group.addButton(rb)
            self._rb_mode[m] = rb
            vl.addWidget(rb)
            rb.toggled.connect(self._on_config_changed)

        # Niveau
        self._level_hdr = self._hdr()
        vl.addWidget(self._level_hdr)
        self._level_box = QWidget()
        lv = QVBoxLayout(self._level_box)
        lv.setContentsMargins(16, 0, 0, 0)
        lv.setSpacing(2)
        self._level_group = QButtonGroup(self)
        self._rb_level: dict[int, QRadioButton] = {}
        for lvl in (1, 2, 3):
            rb = QRadioButton()
            self._level_group.addButton(rb)
            self._rb_level[lvl] = rb
            lv.addWidget(rb)
        vl.addWidget(self._level_box)

        # Régions
        self._regions_hdr = self._hdr()
        vl.addWidget(self._regions_hdr)
        self._regions_box = QWidget()
        self._regions_layout = QHBoxLayout(self._regions_box)
        self._regions_layout.setContentsMargins(16, 0, 0, 0)
        self._region_checks: dict[str, QCheckBox] = {}
        vl.addWidget(self._regions_box)

        # Détail + nombre de questions
        grid = QHBoxLayout()
        self._tier_lbl = QLabel()
        self._tier_combo = QComboBox()
        self._tier_combo.addItems(["", ""])
        self._tier_combo.currentIndexChanged.connect(self._on_config_changed)
        self._n_lbl = QLabel()
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 60)
        self._n_spin.setValue(10)
        grid.addWidget(self._tier_lbl)
        grid.addWidget(self._tier_combo)
        grid.addSpacing(20)
        grid.addWidget(self._n_lbl)
        grid.addWidget(self._n_spin)
        grid.addStretch()
        vl.addLayout(grid)

        self._sup_check = QCheckBox()
        self._sup_check.setChecked(True)
        self._sup_check.toggled.connect(self._on_config_changed)
        vl.addWidget(self._sup_check)

        vl.addWidget(_sep())
        self._avail_lbl = QLabel()
        self._avail_lbl.setWordWrap(True)
        self._avail_lbl.setStyleSheet(f"font-size: 12px; color: {_MUTED};")
        vl.addWidget(self._avail_lbl)

        self._load_bar = QProgressBar()
        self._load_bar.setTextVisible(True)
        self._load_bar.setVisible(False)
        self._load_bar.setStyleSheet(
            "QProgressBar { background: #2a2a44; border: none; border-radius: 4px; "
            f"color: {_TEXT}; height: 16px; text-align: center; }}"
            f"QProgressBar::chunk {{ background: {_BLUE}; border-radius: 4px; }}"
        )
        vl.addWidget(self._load_bar)

        btns = QHBoxLayout()
        self._menu_btn = QPushButton()
        self._menu_btn.setStyleSheet(_BTN_SECONDARY)
        self._menu_btn.clicked.connect(self.exercise_complete)
        btns.addWidget(self._menu_btn)
        btns.addStretch()
        self._start_btn = QPushButton()
        self._start_btn.setStyleSheet(_BTN_PRIMARY)
        self._start_btn.setMinimumWidth(160)
        self._start_btn.clicked.connect(self._on_start_clicked)
        btns.addWidget(self._start_btn)
        vl.addLayout(btns)

        self._attr_lbl = QLabel()
        self._attr_lbl.setWordWrap(True)
        self._attr_lbl.setOpenExternalLinks(True)
        self._attr_lbl.setStyleSheet("font-size: 10px; color: #7878a0;")
        vl.addWidget(self._attr_lbl)
        outer.addStretch()
        return page

    @staticmethod
    def _hdr() -> QLabel:
        lbl = QLabel()
        lbl.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {_BLUE};")
        return lbl

    # -- page de quiz ---------------------------------------------------------

    def _build_quiz_page(self) -> QWidget:
        page = QWidget()
        hl = QHBoxLayout(page)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        left = QWidget()
        self._left_layout = QVBoxLayout(left)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_layout.setSpacing(0)
        self._banner = QLabel()
        self._banner.setWordWrap(True)
        self._banner.setVisible(False)
        self._banner.setStyleSheet(
            "background: #4a3a10; color: #ffd070; font-size: 12px; padding: 5px 10px; "
            "border-bottom: 1px solid #6a5a20;"
        )
        self._left_layout.addWidget(self._banner)
        self._plot_holder = QVBoxLayout()   # le QtInteractor y est inséré au 1er quiz
        self._plot_holder.setContentsMargins(0, 0, 0, 0)
        self._left_layout.addLayout(self._plot_holder, stretch=1)
        hl.addWidget(left, stretch=1)

        panel = QWidget()
        panel.setFixedWidth(340)
        panel.setStyleSheet(f"background: {_CARD}; border-left: 1px solid #333355;")
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(14, 12, 14, 12)
        vl.setSpacing(8)

        top = QHBoxLayout()
        self._q_counter = QLabel()
        self._q_counter.setStyleSheet(f"font-size: 12px; color: {_BLUE}; font-weight: bold;")
        top.addWidget(self._q_counter)
        top.addStretch()
        self._score_lbl = QLabel()
        self._score_lbl.setStyleSheet(f"font-size: 12px; color: {_MUTED};")
        top.addWidget(self._score_lbl)
        vl.addLayout(top)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet(
            "QProgressBar { background: #2a2a44; border: none; border-radius: 4px; }"
            f"QProgressBar::chunk {{ background: {_BLUE}; border-radius: 4px; }}"
        )
        vl.addWidget(self._progress)
        vl.addWidget(_sep())

        self._prompt = QLabel()
        self._prompt.setWordWrap(True)
        self._prompt.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {_TEXT};")
        vl.addWidget(self._prompt)
        self._prompt_sub = QLabel()
        self._prompt_sub.setWordWrap(True)
        self._prompt_sub.setStyleSheet(f"font-size: 11px; color: {_MUTED};")
        vl.addWidget(self._prompt_sub)

        # Zone de réponse
        self._answer_stack = QStackedWidget()
        vl.addWidget(self._answer_stack, stretch=1)

        # 0 : choix multiple
        w = QWidget()
        cl = QVBoxLayout(w)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        self._choice_btns: list[QPushButton] = []
        for i in range(_MAX_CHOICES):
            b = QPushButton()
            b.setStyleSheet(_BTN_ANSWER)
            b.setMinimumHeight(44)
            b.setVisible(False)
            b.clicked.connect(lambda _=False, j=i: self._on_choice(j))
            self._choice_btns.append(b)
            cl.addWidget(b)
        cl.addStretch()
        self._answer_stack.addWidget(w)

        # 1 : liste
        w = QWidget()
        ll = QVBoxLayout(w)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        self._filter_edit = QLineEdit()
        self._filter_edit.textChanged.connect(self._filter_list)
        self._filter_edit.returnPressed.connect(self._on_list_submit)
        ll.addWidget(self._filter_edit)
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _it: self._on_list_submit())
        ll.addWidget(self._list, stretch=1)
        self._list_btn = QPushButton()
        self._list_btn.setStyleSheet(_BTN_PRIMARY)
        self._list_btn.clicked.connect(self._on_list_submit)
        ll.addWidget(self._list_btn)
        self._answer_stack.addWidget(w)

        # 2 : saisie libre
        w = QWidget()
        tl = QVBoxLayout(w)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)
        self._text_edit = QLineEdit()
        self._text_edit.returnPressed.connect(self._on_text_submit)
        tl.addWidget(self._text_edit)
        self._text_btn = QPushButton()
        self._text_btn.setStyleSheet(_BTN_PRIMARY)
        self._text_btn.clicked.connect(self._on_text_submit)
        tl.addWidget(self._text_btn)
        tl.addStretch()
        self._answer_stack.addWidget(w)

        # 3 : localiser (aucun champ, on clique dans la vue)
        w = QWidget()
        pl = QVBoxLayout(w)
        pl.setContentsMargins(0, 0, 0, 0)
        self._locate_tip = QLabel()
        self._locate_tip.setWordWrap(True)
        self._locate_tip.setStyleSheet(f"font-size: 11px; color: {_MUTED};")
        pl.addWidget(self._locate_tip)
        pl.addStretch()
        self._answer_stack.addWidget(w)

        vl.addWidget(_sep())
        # Retour
        self._feedback = QLabel()
        self._feedback.setWordWrap(True)
        self._feedback.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {_TEXT};")
        vl.addWidget(self._feedback)
        self._feedback_detail = QLabel()
        self._feedback_detail.setWordWrap(True)
        self._feedback_detail.setTextFormat(Qt.RichText)
        self._feedback_detail.setStyleSheet(f"font-size: 12px; color: {_MUTED};")
        vl.addWidget(self._feedback_detail)

        row = QHBoxLayout()
        self._skip_btn = QPushButton()
        self._skip_btn.setStyleSheet(_BTN_SECONDARY)
        self._skip_btn.clicked.connect(self._on_skip)
        row.addWidget(self._skip_btn)
        self._next_btn = QPushButton()
        self._next_btn.setStyleSheet(_BTN_PRIMARY)
        self._next_btn.clicked.connect(self._on_next)
        row.addWidget(self._next_btn, stretch=1)
        vl.addLayout(row)

        self._abandon_btn = QPushButton()
        self._abandon_btn.setStyleSheet(_BTN_QUIT)
        self._abandon_btn.clicked.connect(self._on_abandon)
        vl.addWidget(self._abandon_btn)

        hl.addWidget(panel)
        self._retranslate()
        return page

    # ── Textes ────────────────────────────────────────────────────────────────

    def _retranslate(self) -> None:
        t = lambda k, **kw: _tt(self._lang, k, **kw)  # noqa: E731
        self._cfg_title.setText(t("title"))
        self._cfg_sub.setText(t("subtitle"))
        self._kind_hdr.setText(t("kind"))
        self._rb_kind["bone"].setText("🦴 " + t("kind_bone"))
        self._rb_kind["muscle"].setText("💪 " + t("kind_muscle"))
        self._mode_hdr.setText(t("mode"))
        self._rb_mode["identify"].setText(t("mode_identify"))
        self._rb_mode["locate"].setText(t("mode_locate"))
        self._level_hdr.setText(t("level"))
        for lvl in (1, 2, 3):
            self._rb_level[lvl].setText(t(f"level{lvl}"))
        self._regions_hdr.setText(t("regions"))
        self._tier_lbl.setText(t("tier"))
        self._tier_combo.setItemText(0, t("tier1"))
        self._tier_combo.setItemText(1, t("tier2"))
        self._n_lbl.setText(t("n_questions"))
        self._sup_check.setText(t("superficial_only"))
        self._sup_check.setToolTip(t("superficial_only_tip"))
        self._start_btn.setText(t("start"))
        self._menu_btn.setText(t("back_menu"))
        self._attr_lbl.setText(f'{ATTRIBUTION} — <a href="{ATTRIBUTION_URL}">{ATTRIBUTION_URL}</a>')
        self._filter_edit.setPlaceholderText(t("filter_ph"))
        self._text_edit.setPlaceholderText(t("text_ph"))
        self._list_btn.setText(t("validate"))
        self._text_btn.setText(t("validate"))
        self._skip_btn.setText(t("skip"))
        self._abandon_btn.setText(t("abandon"))
        self._locate_tip.setText(t("q_locate_tip"))

    # ── Écran de configuration : logique ──────────────────────────────────────

    def _preset(self, kind: str | None, mode) -> None:
        kind = kind if kind in ("bone", "muscle") else "bone"
        if isinstance(mode, Mode):
            mode = mode.value
        mode = mode if mode in ("identify", "locate") else "identify"
        self._rb_kind[kind].setChecked(True)
        self._rb_mode[mode].setChecked(True)
        self._rb_level[1].setChecked(True)
        self._tier_combo.setCurrentIndex(0)

    @property
    def _kind(self) -> str:
        return "muscle" if self._rb_kind["muscle"].isChecked() else "bone"

    @property
    def _mode(self) -> Mode:
        return Mode.LOCATE if self._rb_mode["locate"].isChecked() else Mode.IDENTIFY

    @property
    def _level(self) -> Level:
        for lvl in (3, 2, 1):
            if self._rb_level[lvl].isChecked():
                return Level(lvl)
        return Level.CHOICE

    def _refresh_config(self) -> None:
        """Reconstruit les cases de régions et l'état des contrôles."""
        self._retranslate()
        kind = self._kind
        prev = {r: c.isChecked() for r, c in self._region_checks.items()}
        while self._regions_layout.count():
            item = self._regions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._region_checks = {}
        for region in self._selected_regions_for(kind):
            n = sum(1 for s in self._available(kind) if s.region == region)
            cb = QCheckBox(f"{region_label(region, self._lang)} ({n})")
            cb.setChecked(prev.get(region, True))
            cb.toggled.connect(self._on_config_changed)
            self._region_checks[region] = cb
            self._regions_layout.addWidget(cb)
        self._regions_layout.addStretch()
        self._on_config_changed()

    def _on_config_changed(self, *_a) -> None:
        if not hasattr(self, "_avail_lbl"):
            return
        kind_changed = set(self._region_checks) != set(self._selected_regions_for(self._kind))
        if kind_changed:
            self._refresh_config()
            return
        identify = self._mode == Mode.IDENTIFY
        self._level_hdr.setVisible(identify)
        self._level_box.setVisible(identify)
        self._sup_check.setVisible(self._kind == "muscle" and not identify)
        cfg = self._config_from_ui()
        avail = self._available(cfg.kind)
        if not avail:
            word = _tt(self._lang, f"kind_word_{cfg.kind}")
            self._avail_lbl.setStyleSheet("font-size: 12px; color: #ff8080;")
            self._avail_lbl.setText(
                _tt(self._lang, "no_mesh", kind=word, arg=cfg.kind)
            )
            self._start_btn.setEnabled(False)
            return
        n = self._eligible_count(cfg)
        ok = n > 0
        self._avail_lbl.setStyleSheet(
            f"font-size: 12px; color: {_MUTED if ok else '#ff8080'};"
        )
        self._avail_lbl.setText(
            _tt(self._lang, "available", n=n) if ok else _tt(self._lang, "no_question")
        )
        self._start_btn.setEnabled(ok)

    def _eligible_count(self, cfg: QuizConfig) -> int:
        regions = set(cfg.regions) if cfg.regions else None
        n = 0
        for s in self._available(cfg.kind):
            if s.tier > cfg.tier_max or (regions is not None and s.region not in regions):
                continue
            if (cfg.mode == Mode.LOCATE and cfg.kind == "muscle"
                    and cfg.superficial_targets_only and s.layer != "superficial"):
                continue
            n += 1
        return n

    def _config_from_ui(self) -> QuizConfig:
        kind = self._kind
        checked = tuple(r for r, c in self._region_checks.items() if c.isChecked())
        regions = None if len(checked) == len(self._region_checks) else checked
        return QuizConfig(
            kind=kind,
            mode=self._mode,
            level=self._level if self._mode == Mode.IDENTIFY else Level.CHOICE,
            n_questions=self._n_spin.value(),
            regions=regions,
            tier_max=2 if self._tier_combo.currentIndex() == 1 else 1,
            superficial_targets_only=(
                kind == "muscle" and self._mode == Mode.LOCATE and self._sup_check.isChecked()
            ),
            lang=self._lang,
            seed=None,
        )

    # ── Démarrage d'un quiz ───────────────────────────────────────────────────

    def _on_start_clicked(self) -> None:
        self.start_quiz(self._config_from_ui())

    def start_quiz(self, config: QuizConfig) -> bool:
        """Prépare la scène, crée la session et affiche la 1re question."""
        pool = self._available(config.kind)
        if not pool:
            return False
        session = QuizSession(config, pool)
        if not session.questions:
            QMessageBox.information(self, _tt(self._lang, "title"), _tt(self._lang, "not_enough"))
            return False
        self._config = config
        if not self._prepare_scene(config):
            return False
        self._begin_session(session)
        return True

    def _begin_session(self, session: QuizSession) -> None:
        self._session = session
        cfg = session.config
        self._first_locate_camera = True
        self._stack.setCurrentIndex(_PAGE_QUIZ)
        self._ensure_plotter()
        self._pool_by_id = {s.id: s for s in session.answer_pool}
        # Liste complète (niveau 2) : une seule fois par session.
        if cfg.mode == Mode.IDENTIFY and cfg.level == Level.LIST:
            self._populate_list(session)
        self._show_question()

    # ── Scène ─────────────────────────────────────────────────────────────────

    def _prepare_scene(self, cfg: QuizConfig) -> bool:
        """Charge/fusionne les meshes nécessaires (avec cache)."""
        regions = tuple(sorted(cfg.regions)) if cfg.regions else None
        key = (cfg.kind, regions)
        if key == self._scene_key and self._layers:
            return True

        self._load_bar.setVisible(True)
        self._load_bar.setValue(0)
        self._start_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        t0 = time.perf_counter()
        try:
            bones = self._available("bone")
            if cfg.kind == "bone":
                groups = [("bone", bones)]
            else:
                muscles = [s for s in self._available("muscle")
                           if regions is None or s.region in regions]
                groups = [("bone", bones), ("muscle", muscles)]
            total = sum(len(g[1]) for g in groups)
            done = [0]

            def progress(_i: int, _n: int) -> None:
                done[0] += 1
                self._load_bar.setMaximum(max(total, 1))
                self._load_bar.setValue(done[0])
                self._load_bar.setFormat(_tt(self._lang, "loading", i=done[0], n=total))
                if done[0] % 15 == 0:
                    QApplication.processEvents()

            layers: dict[str, _sc.SceneLayer] = {}
            for name, structs in groups:
                merged = _sc.build_merged(structs, progress)
                by_id = {s.id: s for s in structs}
                if name == "bone":
                    colour = _sc.COLOR_BONE if cfg.kind == "bone" else _sc.COLOR_BONE_GREY
                    pal = np.tile(np.array(colour, np.uint8), (len(merged), 1))
                else:
                    pal = np.array([_sc.muscle_color(i) for i in merged.ids], np.uint8) \
                        if len(merged) else np.zeros((0, 3), np.uint8)
                layers[name] = _sc.SceneLayer(name, merged, pal, by_id)
        finally:
            QApplication.restoreOverrideCursor()
            self._load_bar.setVisible(False)
            self._start_btn.setEnabled(True)
        self.timings["load_ms"] = (time.perf_counter() - t0) * 1000.0

        main = layers["muscle"] if cfg.kind == "muscle" else layers["bone"]
        if len(main.merged) == 0:
            return False
        self._layers = layers
        self._scene_key = key
        ref_layer = layers["bone"] if len(layers["bone"].merged) else main
        self._body_ref = ref_layer.merged.bounds
        self._by_id = {}
        for lay in layers.values():
            self._by_id.update(lay.structures)
        for lay in layers.values():
            lay.reset()
        if self._plotter is not None:
            self._plotter.clear()
            self._plotter.enable_3_lights()
        return True

    def _ensure_plotter(self) -> None:
        if self._plotter is not None:
            return
        from pyvistaqt import QtInteractor

        self._plotter = QtInteractor(self)
        self._plotter.set_background(_sc.BACKGROUND)
        self._plotter.enable_3_lights()
        self._plot_holder.addWidget(self._plotter.interactor)
        self._arm_pick_observers()
        QTimer.singleShot(0, self._create_nav_overlay)
        for lay in self._layers.values():
            lay.reset()

    def _create_nav_overlay(self) -> None:
        if self._closed or self._plotter is None or self._nav_overlay is not None:
            return
        p = self._plotter
        self._nav_overlay = NavOverlay(
            p.interactor,
            face_fn=lambda: set_view(p, self._cam_axes, self._cam_axes.front, -1),
            back_fn=lambda: set_view(p, self._cam_axes, self._cam_axes.front, +1),
            left_fn=lambda: set_view(p, self._cam_axes, self._cam_axes.side, +1),
            right_fn=lambda: set_view(p, self._cam_axes, self._cam_axes.side, -1),
            top_fn=lambda: view_top(p, self._cam_axes),
            reset_fn=self._reset_camera,
            pan_fn=lambda dx, dy: pan_camera(p, dx, dy),
            toggle_curvature_fn=None,
        )
        self._nav_overlay.show()

    # ── Picking (observers VTK) ───────────────────────────────────────────────

    def _arm_pick_observers(self) -> None:
        iren = self._plotter.iren

        def _on_press(_caller, _event) -> None:
            self._press_xy = iren.get_event_position()

        def _on_release(_caller, _event) -> None:
            if self._press_xy is None:
                return
            rx, ry = iren.get_event_position()
            px, py = self._press_xy
            self._press_xy = None
            if (rx - px) ** 2 + (ry - py) ** 2 > _CLICK_TOLERANCE_PX2:
                return  # glissé : rotation de caméra, pas un clic
            self._on_view_click(rx, ry)

        self._press_cb, self._release_cb = _on_press, _on_release
        self._press_obs = iren.add_observer("LeftButtonPressEvent", _on_press)
        self._release_obs = iren.add_observer("LeftButtonReleaseEvent", _on_release)

    def _remove_pick_observers(self) -> None:
        if self._plotter is None:
            return
        try:
            iren = self._plotter.iren
            for attr in ("_press_obs", "_release_obs"):
                oid = getattr(self, attr)
                if oid is not None:
                    try:
                        iren.remove_observer(oid)
                    except Exception:
                        pass
                    setattr(self, attr, None)
        except Exception:
            pass
        self._press_cb = self._release_cb = None

    def _on_view_click(self, x: int, y: int) -> None:
        """Clic simple dans la vue 3D (mode LOCALISER)."""
        if not self._awaiting_pick or self._session is None or self._plotter is None:
            return
        layers = [lay for lay in self._layers.values() if lay.actor is not None]
        layer_name, sid = _sc.pick_structure(self._plotter.renderer, x, y, layers)
        self._handle_pick(layer_name, sid)

    def _handle_pick(self, layer_name: str | None, sid: str | None) -> None:
        """Interprète un clic : (couche, id) — testable sans OpenGL."""
        if not self._awaiting_pick or self._session is None:
            return
        cfg = self._session.config
        if sid is None:
            self._prompt_sub.setText(_tt(self._lang, "clicked_nothing"))
            return
        struct = self._by_id.get(sid)
        if struct is None or struct.kind != cfg.kind:
            self._prompt_sub.setText(_tt(self._lang, "clicked_bone"))
            return
        self._awaiting_pick = False
        verdict = self._session.submit_pick(sid)
        self._after_answer(verdict, given_id=sid)

    # ── Affichage d'une question ──────────────────────────────────────────────

    def _target_layer(self) -> _sc.SceneLayer:
        return self._layers["muscle" if self._config.kind == "muscle" else "bone"]

    def _show_question(self) -> None:
        s = self._session
        q = s.current
        if q is None:
            self._show_results()
            return
        cfg = s.config
        t0 = time.perf_counter()
        self._answered = False
        n = len(s.questions)
        self._q_counter.setText(_tt(self._lang, "question_n", i=q.index + 1, n=n))
        self._progress.setMaximum(n)
        self._progress.setValue(q.index)
        self._update_score()
        self._feedback.setText("")
        self._feedback.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {_TEXT};")
        self._feedback_detail.setText("")
        self._skip_btn.setEnabled(True)
        self._skip_btn.setVisible(True)
        self._next_btn.setEnabled(False)
        self._next_btn.setText(
            _tt(self._lang, "see_results" if q.index == n - 1 else "next")
        )

        target = q.target
        self._render_question(target)

        if cfg.mode == Mode.IDENTIFY:
            self._prompt.setText(_tt(self._lang, "q_identify"))
            side_tip = target.side in ("left", "right") and cfg.level != Level.CHOICE
            self._prompt_sub.setText(_tt(self._lang, "q_identify_side") if side_tip else "")
            self._awaiting_pick = False
            if cfg.level == Level.CHOICE:
                self._answer_stack.setCurrentIndex(0)
                for i, b in enumerate(self._choice_btns):
                    if i < len(q.choices):
                        b.setText(display_name(q.choices[i], self._lang))
                        b.setStyleSheet(_BTN_ANSWER)
                        b.setEnabled(True)
                        b.setVisible(True)
                    else:
                        b.setVisible(False)
            elif cfg.level == Level.LIST:
                self._answer_stack.setCurrentIndex(1)
                self._filter_edit.clear()
                self._filter_edit.setEnabled(True)
                self._list.setEnabled(True)
                self._list.clearSelection()
                self._list.setCurrentRow(-1)
                self._list_btn.setEnabled(True)
                self._filter_edit.setFocus()
            else:
                self._answer_stack.setCurrentIndex(2)
                self._text_edit.clear()
                self._text_edit.setEnabled(True)
                self._text_btn.setEnabled(True)
                self._text_edit.setFocus()
        else:
            self._prompt.setText(
                f"{_tt(self._lang, 'q_locate')} {display_name(target, self._lang)}"
            )
            self._prompt_sub.setText("")
            self._answer_stack.setCurrentIndex(3)
            self._awaiting_pick = True
        self.timings["question_ms"] = (time.perf_counter() - t0) * 1000.0

    def _hidden_for(self, target) -> tuple[list[str], str]:
        """Muscles superficiels à masquer pour rendre la cible profonde visible."""
        cfg = self._session.config
        if cfg.kind != "muscle" or getattr(target, "layer", "") != "deep":
            return [], ""
        ml = self._layers["muscle"]
        if cfg.mode == Mode.IDENTIFY:
            region = target.region
            hidden = ml.ids_where(
                lambda s: s is not None and s.layer == "superficial" and s.region == region
            )
            note = _tt(self._lang, "layer_removed", region=region_label(region, self._lang))
        else:
            # LOCALISER : on ne révèle pas la région de la cible -> tout est retiré.
            hidden = ml.ids_where(lambda s: s is not None and s.layer == "superficial")
            note = _tt(self._lang, "layer_removed_all")
        return hidden, note

    def _render_question(self, target) -> None:
        """Met à jour la scène 3D pour la question courante (sans tout reconstruire)."""
        cfg = self._session.config
        p = self._plotter
        identify = cfg.mode == Mode.IDENTIFY
        overrides = {target.id: _sc.COLOR_TARGET} if identify else {}
        note = ""
        rebuilt = False
        if cfg.kind == "muscle":
            hidden, note = self._hidden_for(target)
            rebuilt |= self._layers["bone"].ensure(p)   # squelette gris : couleurs de base
            ml = self._layers["muscle"]
            rebuilt |= ml.ensure(p, hidden)
            ml.recolor(overrides)
        else:
            bl = self._layers["bone"]
            rebuilt |= bl.ensure(p)
            bl.recolor(overrides)
        self._banner.setText("ⓘ " + note if note else "")
        self._banner.setVisible(bool(note))
        self._frame_camera(target, identify)
        p.render()
        self.timings["scene_rebuilt"] = float(rebuilt)

    # ── Caméra ────────────────────────────────────────────────────────────────

    def _pool_bounds(self) -> tuple[float, ...] | None:
        lay = self._target_layer()
        ids = [s.id for s in self._session.answer_pool if s.id in lay.merged.index]
        return lay.merged.bounds_of(ids)

    def _frame_camera(self, target, identify: bool) -> None:
        """IDENTIFIER : face à la cible.  LOCALISER : vue d'ensemble (sans indice)."""
        lay = self._target_layer()
        idx = lay.merged.index.get(target.id)
        if identify and idx is not None:
            centroid = lay.merged.centroids[idx]
            b = lay.merged.bounds_of([target.id])
            view = _sc.choose_view(centroid, self._body_ref)
            self._cam_axes = CameraAxes(
                up=2, front=1, side=0, centers=[float(c) for c in centroid],
                distance=_sc.focus_distance(_sc.bounds_extent(b)),
            )
            axis, sign = _sc.VIEW_AXIS[view]
            set_view(self._plotter, self._cam_axes, axis, sign)
        else:
            # LOCALISER : vue d'ensemble de la sélection, orientation conservée
            # (aucun indice sur la position de la cible).
            b = self._pool_bounds() or lay.merged.bounds_of(lay.merged.ids)
            self._cam_axes = CameraAxes(
                up=2, front=1, side=0, centers=_sc.bounds_center(b),
                distance=max(_sc.bounds_extent(b) * 2.6, 600.0),
            )
            self._aim_camera(self._cam_axes.centers, self._cam_axes.distance,
                             keep_direction=not self._first_locate_camera)
            self._first_locate_camera = False
        self._plotter.reset_camera_clipping_range()

    def _aim_camera(self, center, distance: float, keep_direction: bool = True) -> None:
        """Vise ``center`` à ``distance`` ; garde (ou non) la direction de vue actuelle."""
        p = self._plotter
        cam = p.camera
        if keep_direction:
            d = np.asarray(cam.position, float) - np.asarray(cam.focal_point, float)
            n = float(np.linalg.norm(d))
            keep_direction = n > 1e-6
        if not keep_direction:
            reset_view(p, CameraAxes(up=2, front=1, side=0, centers=list(center),
                                     distance=distance))
            return
        d = d / n
        cam.focal_point = tuple(float(c) for c in center)
        cam.position = tuple(float(c) + float(v) * distance for c, v in zip(center, d))

    def _focus_feedback(self, target_id: str, given_id: str | None) -> None:
        """Après une réponse LOCALISER : recadre sur la cible (et le clic erroné)."""
        lay = self._target_layer()
        m = lay.merged
        if target_id not in m.index:
            return
        pts = [m.centroids[m.index[target_id]]]
        if given_id and given_id != target_id and given_id in m.index:
            pts.append(m.centroids[m.index[given_id]])
        center = np.mean(pts, axis=0)
        sep = float(np.linalg.norm(pts[0] - pts[-1])) if len(pts) > 1 else 0.0
        ext = _sc.bounds_extent(m.bounds_of([target_id]))
        dist = min(max(_sc.focus_distance(ext), 2.2 * sep + 300.0), self._cam_axes.distance)
        self._aim_camera(center, dist, keep_direction=True)
        self._plotter.reset_camera_clipping_range()

    def _reset_camera(self) -> None:
        if self._plotter is not None:
            reset_view(self._plotter, self._cam_axes)

    # ── Réponses ──────────────────────────────────────────────────────────────

    def _on_choice(self, i: int) -> None:
        q = self._session.current if self._session else None
        if self._answered or q is None or i >= len(q.choices):
            return
        sid = q.choices[i].id
        self._after_answer(self._session.submit_choice(sid), given_id=sid)

    def _populate_list(self, session: QuizSession) -> None:
        self._list.clear()
        for s in session.options:
            item = QListWidgetItem(display_name(s, self._lang))
            item.setData(Qt.UserRole, s.id)
            item.setData(
                Qt.UserRole + 1,
                normalize(f"{s.name_fr} {s.name_en}"),
            )
            self._list.addItem(item)

    def _filter_list(self, text: str) -> None:
        needle = normalize(text) if text.strip() else ""
        first = None
        for i in range(self._list.count()):
            it = self._list.item(i)
            hide = bool(needle) and needle not in it.data(Qt.UserRole + 1)
            it.setHidden(hide)
            if first is None and not hide:
                first = it
        if needle and first is not None:
            self._list.setCurrentItem(first)

    def _on_list_submit(self) -> None:
        if self._answered or self._session is None:
            return
        it = self._list.currentItem()
        if it is None or it.isHidden():
            return
        sid = it.data(Qt.UserRole)
        self._after_answer(self._session.submit_list(sid), given_id=sid)

    def _on_text_submit(self) -> None:
        if self._answered or self._session is None:
            return
        text = self._text_edit.text().strip()
        if not text:
            return
        self._after_answer(self._session.submit_text(text), given_text=text)

    def _on_skip(self) -> None:
        if self._answered or self._session is None or self._session.current is None:
            return
        self._awaiting_pick = False
        self._after_answer(self._session.skip(), skipped=True)

    def _after_answer(self, v: Verdict, given_id: str | None = None,
                      given_text: str | None = None, skipped: bool = False) -> None:
        """Retour visuel + texte après une réponse (commun aux quatre formats)."""
        self._answered = True
        s = self._session
        cfg = s.config
        q = s.current
        target = q.target
        L = self._lang

        # Verrouillage des champs
        for b in self._choice_btns:
            b.setEnabled(False)
        self._filter_edit.setEnabled(False)
        self._list.setEnabled(False)
        self._list_btn.setEnabled(False)
        self._text_edit.setEnabled(False)
        self._text_btn.setEnabled(False)
        self._skip_btn.setEnabled(False)

        # Choix multiple : bonne réponse en vert, erreur en rouge
        if cfg.mode == Mode.IDENTIFY and cfg.level == Level.CHOICE:
            for i, c in enumerate(q.choices):
                if c.id == target.id:
                    self._choice_btns[i].setStyleSheet(_BTN_ANSWER_OK)
                elif c.id == given_id:
                    self._choice_btns[i].setStyleSheet(_BTN_ANSWER_KO)

        # Texte principal
        if skipped:
            head, colour = _tt(L, "skipped"), _MUTED
        elif v.correct:
            head, colour = _tt(L, "correct"), _hex(_sc.COLOR_OK)
            if v.close:
                head += " " + _tt(L, "typo")
        elif v.name_ok:
            head, colour = _tt(L, "half"), _hex(_sc.COLOR_OPPOSITE)
        else:
            head, colour = _tt(L, "wrong"), _hex(_sc.COLOR_WRONG)
        self._feedback.setText(head)
        self._feedback.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {colour};")

        # Nom correct FR + EN (+ couche pour les muscles)
        lines = [
            f"<b style='color:{_TEXT}'>{_tt(L, 'answer_is', fr=target.name_fr)}</b>",
            f"<i>{target.name_en}</i>",
        ]
        if cfg.kind == "muscle" and getattr(target, "layer", ""):
            lines.append(_tt(L, f"layer_{target.layer}"))
        if not v.correct and not skipped:
            if given_text:
                lines.append(_tt(L, "you_answered", name=given_text))
            elif given_id and given_id != target.id:
                other = self._by_id.get(given_id) or self._pool_by_id.get(given_id)
                if other is not None:
                    key = "you_clicked" if cfg.mode == Mode.LOCATE else "you_answered"
                    lines.append(_tt(L, key, name=display_name(other, L)))
        self._feedback_detail.setText("<br>".join(lines))

        # LOCALISER : couleurs de retour sur la scène
        if cfg.mode == Mode.LOCATE:
            col: dict[str, tuple[int, int, int]] = {target.id: _sc.COLOR_OK}
            if given_id and given_id != target.id:
                col[given_id] = _sc.COLOR_OPPOSITE if v.name_ok else _sc.COLOR_WRONG
            lay = self._target_layer()
            # La cible profonde doit rester visible : déjà garanti à l'affichage.
            lay.recolor(col)
            if self._plotter is not None:
                self._focus_feedback(target.id, given_id)
                self._plotter.render()

        self._update_score()
        self._next_btn.setEnabled(True)
        self._next_btn.setFocus()

    def _update_score(self) -> None:
        s = self._session
        self._score_lbl.setText(
            _tt(self._lang, "score_now", s=f"{s.score:g}", m=f"{s.max_score:g}")
        )

    def _on_next(self) -> None:
        if self._session is None or not self._answered:
            return
        self._session.next()
        if self._session.finished:
            self._show_results()
        else:
            self._show_question()

    def _on_abandon(self) -> None:
        if self._session is None:
            return
        ans = QMessageBox.question(
            self, _tt(self._lang, "abandon"), _tt(self._lang, "abandon_confirm"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            self.abandon()

    def abandon(self) -> None:
        """Termine le quiz : les questions restantes comptent 0."""
        if self._session is None:
            return
        self._awaiting_pick = False
        while not self._session.finished:
            self._session.next()
        self._show_results()

    # ── Résultats ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                StructureQuizExercise._clear_layout(item.layout())

    def _given_label(self, row: dict) -> str:
        name = row.get("given_name")
        if name:
            other = self._by_id.get(name) or self._pool_by_id.get(name)
            if other is not None:
                return display_name(other, self._lang)
            return str(name)
        return _tt(self._lang, "none_given")

    def _show_results(self) -> None:
        self._awaiting_pick = False
        s = self._session
        L = self._lang
        info = s.summary()
        pct = info["percent"]
        grade = grade_letter(pct)
        self._progress.setValue(self._progress.maximum())

        lay = self._results_page.layout()
        self._clear_layout(lay)

        title = QLabel(_tt(L, "results"))
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {_BLUE};")
        lay.addWidget(title)

        head = QHBoxLayout()
        g = QLabel(grade)
        g.setStyleSheet(f"font-size: 48px; font-weight: bold; color: {grade_color(grade)};")
        head.addWidget(g)
        col = QVBoxLayout()
        sc_lbl = QLabel(
            _tt(L, "score_line", s=f"{info['score']:g}", m=f"{info['max_score']:g}",
                p=f"{pct:.0f}")
        )
        sc_lbl.setStyleSheet(f"font-size: 16px; color: {_TEXT};")
        col.addWidget(sc_lbl)
        et = info["error_types"]
        if info["score"] >= info["max_score"]:
            extra = _tt(L, "perfect")
        else:
            parts = []
            if et["wrong_name"]:
                parts.append(_tt(L, "err_name", n=et["wrong_name"]))
            if et["wrong_side"]:
                parts.append(_tt(L, "err_side", n=et["wrong_side"]))
            if et["skipped"]:
                parts.append(_tt(L, "err_skip", n=et["skipped"]))
            extra = " · ".join(parts)
        det = QLabel(extra)
        det.setStyleSheet(f"font-size: 12px; color: {_MUTED};")
        col.addWidget(det)
        head.addLayout(col)
        head.addStretch()
        lay.addLayout(head)

        table = QTableWidget(len(info["questions"]), 4)
        table.setHorizontalHeaderLabels(
            [_tt(L, "col_num"), _tt(L, "col_target"), _tt(L, "col_given"), _tt(L, "col_points")]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setStyleSheet(
            f"QTableWidget {{ background: {_CARD}; color: {_TEXT}; gridline-color: #333355; "
            "border: 1px solid #333355; }} QHeaderView::section { background: #2a2a44; "
            f"color: {_TEXT}; border: none; padding: 4px; }}"
        )
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        for r, row in enumerate(info["questions"]):
            pts = row["points"]
            colour = _hex(_sc.COLOR_OK) if pts >= 1 else (
                _hex(_sc.COLOR_OPPOSITE) if pts > 0 else _hex(_sc.COLOR_WRONG)
            )
            st = None
            try:
                st = _cat.get(row["target_id"])
            except KeyError:
                st = self._by_id.get(row["target_id"])
            tname = display_name(st, L) if st is not None else row["target"]
            cells = [str(row["index"] + 1), tname, self._given_label(row), f"{pts:g}"]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if c == 3:
                    it.setForeground(Qt.white)
                    it.setBackground(_qcolor(colour))
                table.setItem(r, c, it)
        lay.addWidget(table, stretch=1)

        if info["frequent_errors"]:
            fe = QLabel(
                "<b>" + _tt(L, "frequent") + "</b><br>" + "<br>".join(
                    f"• {e['target']} ← {e['given']} (×{e['count']})"
                    for e in info["frequent_errors"]
                )
            )
            fe.setTextFormat(Qt.RichText)
            fe.setStyleSheet(f"font-size: 12px; color: {_MUTED};")
            lay.addWidget(fe)

        btns = QHBoxLayout()
        missed = len(s.retry_queue())
        self._replay_btn = QPushButton(_tt(L, "replay", n=missed))
        self._replay_btn.setStyleSheet(_BTN_PRIMARY)
        self._replay_btn.setVisible(missed > 0)
        self._replay_btn.clicked.connect(self._on_replay)
        btns.addWidget(self._replay_btn)
        newc = QPushButton(_tt(L, "new_config"))
        newc.setStyleSheet(_BTN_SECONDARY)
        newc.clicked.connect(self._on_new_config)
        btns.addWidget(newc)
        btns.addStretch()
        fin = QPushButton(_tt(L, "finish"))
        fin.setStyleSheet(_BTN_PRIMARY)
        fin.clicked.connect(self.exercise_complete)
        btns.addWidget(fin)
        lay.addLayout(btns)

        self._stack.setCurrentIndex(_PAGE_RESULTS)

    def _on_replay(self) -> None:
        """« Rejouer les ratés » : nouvelle session sur les cibles non réussies."""
        if self._session is None or not self._session.retry_queue():
            return
        self._begin_session(self._session.retry_session())

    def _on_new_config(self) -> None:
        self._stack.setCurrentIndex(_PAGE_CONFIG)
        self._refresh_config()

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Libère les ressources VTK.  À appeler avant de détruire le widget."""
        self._closed = True
        self._remove_pick_observers()
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


def _qcolor(hexstr: str):
    from PySide6.QtGui import QColor

    return QColor(hexstr)
