"""Extended landmark set for BonyLandmarks (FR/EN).

Ce module remplace ``landmarks.py``. Il conserve **tous** les codes des 24
repères osseux d'origine (compatibilité des manifestes BodyLoop déjà exportés)
et ajoute trois nouveaux domaines pédagogiques.

Catégories
----------
- ``BONE``     : repères osseux palpables (anatomie appliquée)
- ``EMG``      : sites d'électrodes de surface (recommandations SENIAM)
- ``SKINFOLD`` : sites de plis cutanés (protocole ISAK)
- ``ANTHRO``   : sites de circonférences et de diamètres osseux (ISAK)

Conventions
-----------
- Les pastilles vertes sont collées sur le sujet **avant** le scan BodyLoop ;
  chaque hint décrit donc le geste de palpation *et* le point exact à marquer.
- Les repères bilatéraux portent le suffixe ``_left`` / ``_right``.
- Par convention ISAK, les sites ``SKINFOLD`` et ``ANTHRO`` sont mesurés
  **du côté droit** uniquement, quelle que soit la dominance du sujet.
- Les sites EMG sont décrits en référence aux codes ``BONE`` de ce module
  (p. ex. ``ASIS_right``, ``fibular_head_left``) afin que l'étudiant place
  d'abord les repères osseux, puis en déduise les sites musculaires.
- Sauf mention contraire, l'inter-électrode SENIAM est de 20 mm
  (centre à centre) ; la pastille marque le **centre** de la paire et
  l'orientation indiquée est celle de l'axe reliant les deux électrodes.

Notes pédagogiques
------------------
- Le **vaste intermédiaire** (4e chef du quadriceps) et le **petit fessier**
  sont profonds : ils n'ont **pas** de site EMG de surface valide. Ils sont
  volontairement absents de la liste (à discuter en cours : cross-talk et
  limites de l'EMG de surface).
- L'oblique interne partage la zone de l'oblique externe / supra-iliaque ;
  un seul site abdominal latéral est proposé (cross-talk important).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Language = Literal["fr", "en"]
Category = Literal["BONE", "EMG", "SKINFOLD", "ANTHRO"]
BodySide = Literal["left", "right", "bilateral", "midline"]
Theme = Literal[
    "gait",            # analyse de la marche
    "shoulder",        # épaule / ceinture scapulaire
    "core",            # stabilité du tronc / core stability
    "knee_rehab",      # réhabilitation genou / LCA
    "body_comp",       # composition corporelle (plis + circonférences)
    "cpr",             # RCP / gestes d'urgence
    "posture",         # analyse posturale globale
    "upper_limb",      # membre supérieur (coude, poignet)
    "lower_limb",      # membre inférieur (cheville, pied)
    "anatomy",         # anatomie de surface générale (pas d'application spécifique)
]

THEME_LABELS: dict[str, tuple[str, str]] = {
    "gait":       ("Analyse de la marche",          "Gait analysis"),
    "shoulder":   ("Épaule & ceinture scapulaire",  "Shoulder & scapular girdle"),
    "core":       ("Stabilité du tronc",             "Core stability"),
    "knee_rehab": ("Réhabilitation genou / LCA",     "Knee / ACL rehabilitation"),
    "body_comp":  ("Composition corporelle",         "Body composition"),
    "cpr":        ("RCP & gestes d'urgence",         "CPR & emergency skills"),
    "posture":    ("Analyse posturale",              "Postural analysis"),
    "upper_limb": ("Membre supérieur",               "Upper limb"),
    "lower_limb": ("Membre inférieur",               "Lower limb"),
    "anatomy":    ("Anatomie de surface",            "Surface anatomy"),
}

THEME_COLORS: dict[str, str] = {
    "gait":       "#2196F3",   # bleu
    "shoulder":   "#9C27B0",   # violet
    "core":       "#FF9800",   # orange
    "knee_rehab": "#F44336",   # rouge
    "body_comp":  "#4CAF50",   # vert
    "cpr":        "#E91E63",   # rose / urgence
    "posture":    "#00BCD4",   # cyan
    "upper_limb": "#673AB7",   # indigo
    "lower_limb": "#009688",   # teal
    "anatomy":    "#607D8B",   # gris-bleu
}


@dataclass(frozen=True)
class Landmark:
    code: str
    category: Category
    name_fr: str
    name_en: str
    hint_fr: str
    hint_en: str
    body_side: BodySide = "bilateral"
    theme: str = "anatomy"
    application_fr: str = ""
    application_en: str = ""

    def name(self, lang: Language = "fr") -> str:
        return self.name_fr if lang == "fr" else self.name_en

    def hint(self, lang: Language = "fr") -> str:
        return self.hint_fr if lang == "fr" else self.hint_en

    def application(self, lang: Language = "fr") -> str:
        return self.application_fr if lang == "fr" else self.application_en

    def theme_label(self, lang: Language = "fr") -> str:
        labels = THEME_LABELS.get(self.theme, ("", ""))
        return labels[0] if lang == "fr" else labels[1]

    def theme_color(self) -> str:
        return THEME_COLORS.get(self.theme, "#607D8B")


CATEGORY_LABELS: dict[Category, tuple[str, str]] = {
    "BONE": ("Repères osseux", "Bony landmarks"),
    "EMG": ("Sites EMG (SENIAM)", "EMG sites (SENIAM)"),
    "SKINFOLD": ("Plis cutanés (ISAK)", "Skinfolds (ISAK)"),
    "ANTHRO": ("Anthropométrie (ISAK)", "Anthropometry (ISAK)"),
}


_DATA_FILE = Path(__file__).parent / "data" / "landmarks.json"


def _load_landmarks() -> list[Landmark]:
    with open(_DATA_FILE, encoding="utf-8") as f:
        return [Landmark(**entry) for entry in json.load(f)]


LANDMARKS: list[Landmark] = _load_landmarks()


# ─────────────────────────────────────────────────────────────────────────
# Index et accesseurs
# ─────────────────────────────────────────────────────────────────────────

LANDMARK_BY_CODE: dict[str, Landmark] = {lm.code: lm for lm in LANDMARKS}

if len(LANDMARK_BY_CODE) != len(LANDMARKS):
    _seen: set[str] = set()
    _dupes = sorted({lm.code for lm in LANDMARKS if lm.code in _seen or _seen.add(lm.code)})
    raise ValueError(f"Duplicate landmark codes: {_dupes}")


def get_landmark(code: str) -> Landmark:
    """Return the Landmark for the given BodyLoop marker code."""
    if code not in LANDMARK_BY_CODE:
        raise KeyError(f"Unknown landmark code: {code!r}")
    return LANDMARK_BY_CODE[code]


def landmarks_by_category(category: Category) -> list[Landmark]:
    """Return every landmark of a given category, in declaration order."""
    return [lm for lm in LANDMARKS if lm.category == category]


def landmarks_by_side(side: BodySide, include_midline: bool = True) -> list[Landmark]:
    """Return landmarks for one body side.

    ``include_midline`` also returns the non-lateralised markers
    (``'bilateral'`` or ``'midline'``) such as ``C7_spinous`` or ``sacrum_S2``.
    """
    return [
        lm
        for lm in LANDMARKS
        if lm.body_side == side
        or (include_midline and lm.body_side in ("bilateral", "midline"))
    ]


def category_label(category: Category, lang: Language = "fr") -> str:
    """Human-readable, localised name of a category (for GUI grouping)."""
    fr, en = CATEGORY_LABELS[category]
    return fr if lang == "fr" else en


# Sous-ensembles pratiques pour composer un exercice
BONE_LANDMARKS: list[Landmark] = landmarks_by_category("BONE")
EMG_LANDMARKS: list[Landmark] = landmarks_by_category("EMG")
SKINFOLD_LANDMARKS: list[Landmark] = landmarks_by_category("SKINFOLD")
ANTHRO_LANDMARKS: list[Landmark] = landmarks_by_category("ANTHRO")


if __name__ == "__main__":  # pragma: no cover - petit auto-test
    print(f"Total: {len(LANDMARKS)} repères")
    for _cat in ("BONE", "EMG", "SKINFOLD", "ANTHRO"):
        print(f"  {_cat:<9} {len(landmarks_by_category(_cat)):>3}")
