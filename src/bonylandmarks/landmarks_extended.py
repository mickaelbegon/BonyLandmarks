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

from dataclasses import dataclass
from typing import Literal

Language = Literal["fr", "en"]
Category = Literal["BONE", "EMG", "SKINFOLD", "ANTHRO"]
BodySide = Literal["left", "right", "bilateral"]
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


LANDMARKS: list[Landmark] = [
    # ══════════════════════════════════════════════════════════════════════
    # BONE — Tête et tronc
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="vertex",
        category="BONE",
        name_fr="Vertex",
        name_en="Vertex",
        hint_fr=(
            "Point le plus haut du crâne lorsque la tête est orientée selon le "
            "plan de Francfort (bord inférieur de l'orbite et conduit auditif "
            "externe sur une même horizontale). Aplatir les cheveux avec la "
            "règle avant de marquer ; c'est le repère de mesure de la stature."
        ),
        hint_en=(
            "Highest point of the skull with the head in the Frankfort plane "
            "(lower orbital margin and ear canal horizontally aligned). Flatten "
            "the hair with the rule before marking; this is the stature "
            "measurement point."
        ),
    ),
    Landmark(
        code="suprasternal_notch",
        category="BONE",
        name_fr="Fourchette sternale (incisure jugulaire)",
        name_en="Suprasternal notch",
        hint_fr=(
            "Creux médian au sommet du manubrium sternal, entre les deux "
            "extrémités médiales des clavicules. Faites glisser le doigt le long "
            "de la clavicule vers la ligne médiane jusqu'à tomber dans la "
            "dépression : marquer le bord supérieur du sternum, pas la peau du creux."
        ),
        hint_en=(
            "Midline depression at the top of the manubrium, between the medial "
            "ends of both clavicles. Slide a finger along the clavicle toward the "
            "midline until it drops into the notch; mark the superior border of "
            "the sternum, not the soft tissue of the hollow."
        ),
    ),
    Landmark(
        code="xiphoid_process",
        category="BONE",
        name_fr="Processus xiphoïde",
        name_en="Xiphoid process",
        hint_fr=(
            "Extrémité inférieure du sternum, à l'angle infrasternal formé par "
            "les deux rebords costaux. Remonter les rebords costaux avec les deux "
            "pouces jusqu'à leur jonction médiane. Palper doucement : la pointe "
            "est cartilagineuse et sensible."
        ),
        hint_en=(
            "Lower tip of the sternum, at the infrasternal angle formed by the "
            "two costal margins. Follow both costal margins upward with the thumbs "
            "to their midline junction. Palpate gently: the tip is cartilaginous "
            "and tender."
        ),
    ),
    Landmark(
        code="sternal_angle_louis",
        category="BONE",
        name_fr="Angle sternal (angle de Louis)",
        name_en="Sternal angle (angle of Louis)",
        hint_fr=(
            "Jonction manubrio-sternale palpée comme une crête transversale à environ "
            "5 cm sous la fourchette sternale. Glisser le doigt vers le bas depuis la "
            "fourchette jusqu'à sentir la crête horizontale. C'est le niveau de la 2e "
            "côte et des disques T4-T5. Repère proximal pour le comptage des espaces "
            "intercostaux et la limite supérieure de la zone de compression en RCP."
        ),
        hint_en=(
            "Manubriosternal junction felt as a transverse ridge ~5 cm below the "
            "suprasternal notch. Slide a finger downward from the notch until you feel "
            "the horizontal ridge. This is the level of the 2nd rib and T4-T5 disc. "
            "Proximal reference for counting intercostal spaces and the upper limit of "
            "the CPR compression zone."
        ),
        theme="cpr",
        application_fr=(
            "Utilisé pour délimiter la zone de compression thoracique en RCP (moitié "
            "inférieure du sternum) et pour le comptage des côtes en auscultation "
            "et défibrillation."
        ),
        application_en=(
            "Used to delimit the chest compression zone in CPR (lower half of sternum) "
            "and for rib counting in auscultation and defibrillation."
        ),
    ),
    Landmark(
        code="C7_spinous",
        category="BONE",
        name_fr="Processus épineux de C7",
        name_en="C7 spinous process",
        hint_fr=(
            "Épineuse la plus saillante à la base du cou (vertebra prominens). "
            "Pour la distinguer de C6 : demander au sujet de fléchir puis "
            "d'étendre le cou — C6 s'efface sous le doigt en extension, C7 reste "
            "saillante. Sert de référence au trapèze descendant (EMG)."
        ),
        hint_en=(
            "Most prominent spinous process at the base of the neck (vertebra "
            "prominens). To tell it from C6, ask the subject to flex then extend "
            "the neck: C6 slides away under the finger in extension while C7 stays "
            "prominent. Reference point for the upper trapezius EMG site."
        ),
    ),
    Landmark(
        code="T8_spinous",
        category="BONE",
        name_fr="Processus épineux de T8",
        name_en="T8 spinous process",
        hint_fr=(
            "Compter les épineuses vers le bas à partir de C7, bras le long du "
            "corps. T8 se situe approximativement au niveau du bord inférieur de "
            "la scapula (angle inférieur ≈ T7-T8). Référence distale du trapèze "
            "ascendant (EMG)."
        ),
        hint_en=(
            "Count spinous processes downward from C7 with the arms at the side. "
            "T8 lies roughly level with the inferior angle of the scapula "
            "(≈ T7-T8). Distal reference for the lower trapezius EMG site."
        ),
    ),
    Landmark(
        code="L1_spinous",
        category="BONE",
        name_fr="Processus épineux de L1",
        name_en="L1 spinous process",
        hint_fr=(
            "Repérer d'abord L4 (sur la ligne horizontale joignant les points les "
            "plus hauts des crêtes iliaques), puis remonter trois épineuses. "
            "Référence des sites EMG du longissimus et de l'iliocostal (2 travers "
            "de doigt latéralement)."
        ),
        hint_en=(
            "First locate L4 (on the horizontal line joining the highest points of "
            "both iliac crests), then count three spinous processes upward. "
            "Reference for the longissimus and iliocostalis EMG sites (two finger "
            "widths laterally)."
        ),
    ),
    Landmark(
        code="L5_spinous",
        category="BONE",
        name_fr="Processus épineux de L5",
        name_en="L5 spinous process",
        hint_fr=(
            "Dernière épineuse lombaire mobile, juste au-dessus du sacrum, sur la "
            "ligne médiane entre les deux EIPS. Repère clé de la charnière "
            "lombo-sacrée (L5-S1) en ergonomie et en analyse de la manutention."
        ),
        hint_en=(
            "Last mobile lumbar spinous process, just above the sacrum, on the "
            "midline between both PSIS. Key landmark for the lumbosacral junction "
            "(L5-S1) in ergonomics and lifting analysis."
        ),
    ),
    # ══════════════════════════════════════════════════════════════════════
    # BONE — Ceinture scapulaire
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="acromion_left",
        category="BONE",
        name_fr="Acromion gauche",
        name_en="Left acromion",
        hint_fr=(
            "Processus plat et large de la scapula formant le sommet de l'épaule "
            "gauche. Suivre l'épine de la scapula vers le dehors jusqu'à sa "
            "terminaison plate. Marquer le point le plus latéral et supérieur du "
            "bord acromial (acromiale en anthropométrie ISAK)."
        ),
        hint_en=(
            "Flat, broad process of the left scapula forming the tip of the "
            "shoulder. Follow the scapular spine laterally to its flat end. Mark "
            "the most lateral and superior point of the acromial border "
            "(acromiale in ISAK anthropometry)."
        ),
        body_side="left",
    ),
    Landmark(
        code="acromion_right",
        category="BONE",
        name_fr="Acromion droit",
        name_en="Right acromion",
        hint_fr=(
            "Processus plat et large de la scapula formant le sommet de l'épaule "
            "droite. Suivre l'épine de la scapula vers le dehors jusqu'à sa "
            "terminaison plate. Marquer le point le plus latéral et supérieur du "
            "bord acromial (acromiale ISAK, origine de la mesure acromiale-radiale)."
        ),
        hint_en=(
            "Flat, broad process of the right scapula forming the tip of the "
            "shoulder. Follow the scapular spine laterally to its flat end. Mark "
            "the most lateral and superior point of the acromial border "
            "(ISAK acromiale, origin of the acromiale-radiale measurement)."
        ),
        body_side="right",
    ),
    Landmark(
        code="acromial_angle_left",
        category="BONE",
        name_fr="Angle acromial gauche",
        name_en="Left acromial angle",
        hint_fr=(
            "Coin postéro-latéral de l'acromion, là où le bord latéral rejoint le "
            "bord postérieur de l'épine scapulaire. Suivre l'épine de la scapula "
            "vers le dehors : l'angle est le changement de direction net que l'on "
            "sent sous le doigt. Référence du deltoïde postérieur (EMG)."
        ),
        hint_en=(
            "Posterolateral corner of the acromion, where its lateral border meets "
            "the posterior border of the scapular spine. Follow the scapular spine "
            "laterally: the angle is the sharp change of direction felt under the "
            "finger. Reference for the posterior deltoid EMG site."
        ),
        body_side="left",
    ),
    Landmark(
        code="acromial_angle_right",
        category="BONE",
        name_fr="Angle acromial droit",
        name_en="Right acromial angle",
        hint_fr=(
            "Coin postéro-latéral de l'acromion, là où le bord latéral rejoint le "
            "bord postérieur de l'épine scapulaire. Suivre l'épine de la scapula "
            "vers le dehors : l'angle est le changement de direction net que l'on "
            "sent sous le doigt. Référence du deltoïde postérieur (EMG)."
        ),
        hint_en=(
            "Posterolateral corner of the acromion, where its lateral border meets "
            "the posterior border of the scapular spine. Follow the scapular spine "
            "laterally: the angle is the sharp change of direction felt under the "
            "finger. Reference for the posterior deltoid EMG site."
        ),
        body_side="right",
    ),
    Landmark(
        code="scapula_trigonum_left",
        category="BONE",
        name_fr="Trigone de l'épine scapulaire gauche",
        name_en="Left trigonum spinae (root of scapular spine)",
        hint_fr=(
            "Petite surface triangulaire à la jonction entre l'épine de la scapula "
            "et son bord médial (≈ niveau T3). Suivre l'épine vers le dedans "
            "jusqu'à ce qu'elle s'aplatisse contre le bord vertébral. Référence du "
            "trapèze moyen et du trapèze ascendant (EMG)."
        ),
        hint_en=(
            "Small triangular surface where the scapular spine meets the medial "
            "border of the scapula (≈ T3 level). Follow the spine medially until "
            "it flattens against the vertebral border. Reference for the middle "
            "and lower trapezius EMG sites."
        ),
        body_side="left",
    ),
    Landmark(
        code="scapula_trigonum_right",
        category="BONE",
        name_fr="Trigone de l'épine scapulaire droit",
        name_en="Right trigonum spinae (root of scapular spine)",
        hint_fr=(
            "Petite surface triangulaire à la jonction entre l'épine de la scapula "
            "et son bord médial (≈ niveau T3). Suivre l'épine vers le dedans "
            "jusqu'à ce qu'elle s'aplatisse contre le bord vertébral. Référence du "
            "trapèze moyen et du trapèze ascendant (EMG)."
        ),
        hint_en=(
            "Small triangular surface where the scapular spine meets the medial "
            "border of the scapula (≈ T3 level). Follow the spine medially until "
            "it flattens against the vertebral border. Reference for the middle "
            "and lower trapezius EMG sites."
        ),
        body_side="right",
    ),
    Landmark(
        code="scapula_inferior_angle_left",
        category="BONE",
        name_fr="Angle inférieur de la scapula gauche",
        name_en="Left inferior angle of the scapula",
        hint_fr=(
            "Pointe inférieure de la scapula, bras pendant le long du corps et "
            "épaule relâchée. Suivre le bord médial vers le bas jusqu'à la pointe. "
            "C'est le repère « subscapulare » qui sert à localiser le pli cutané "
            "sous-scapulaire (ISAK)."
        ),
        hint_en=(
            "Lowest tip of the scapula, with the arm hanging relaxed at the side. "
            "Follow the medial border downward to its point. This is the ISAK "
            "'subscapulare' landmark used to locate the subscapular skinfold."
        ),
        body_side="left",
    ),
    Landmark(
        code="scapula_inferior_angle_right",
        category="BONE",
        name_fr="Angle inférieur de la scapula droit",
        name_en="Right inferior angle of the scapula",
        hint_fr=(
            "Pointe inférieure de la scapula, bras pendant le long du corps et "
            "épaule relâchée. Suivre le bord médial vers le bas jusqu'à la pointe. "
            "C'est le repère « subscapulare » qui sert à localiser le pli cutané "
            "sous-scapulaire (ISAK)."
        ),
        hint_en=(
            "Lowest tip of the scapula, with the arm hanging relaxed at the side. "
            "Follow the medial border downward to its point. This is the ISAK "
            "'subscapulare' landmark used to locate the subscapular skinfold."
        ),
        body_side="right",
    ),
    # ══════════════════════════════════════════════════════════════════════
    # BONE — Membre supérieur
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="lateral_epicondyle_left",
        category="BONE",
        name_fr="Épicondyle latéral gauche",
        name_en="Left lateral epicondyle",
        hint_fr=(
            "Saillie osseuse latérale de l'extrémité distale de l'humérus gauche, "
            "coude fléchi à 90°. Elle donne insertion aux extenseurs du poignet. "
            "Avec l'épicondyle médial, elle définit le diamètre bi-épicondylien de "
            "l'humérus (ISAK)."
        ),
        hint_en=(
            "Lateral bony prominence at the distal end of the left humerus, elbow "
            "flexed to 90°. It is the common origin of the wrist extensors. With "
            "the medial epicondyle it defines the biepicondylar humerus breadth "
            "(ISAK)."
        ),
        body_side="left",
    ),
    Landmark(
        code="lateral_epicondyle_right",
        category="BONE",
        name_fr="Épicondyle latéral droit",
        name_en="Right lateral epicondyle",
        hint_fr=(
            "Saillie osseuse latérale de l'extrémité distale de l'humérus droit, "
            "coude fléchi à 90°. Elle donne insertion aux extenseurs du poignet. "
            "Avec l'épicondyle médial, elle définit le diamètre bi-épicondylien de "
            "l'humérus (ISAK)."
        ),
        hint_en=(
            "Lateral bony prominence at the distal end of the right humerus, elbow "
            "flexed to 90°. It is the common origin of the wrist extensors. With "
            "the medial epicondyle it defines the biepicondylar humerus breadth "
            "(ISAK)."
        ),
        body_side="right",
    ),
    Landmark(
        code="medial_epicondyle_left",
        category="BONE",
        name_fr="Épicondyle médial gauche",
        name_en="Left medial epicondyle",
        hint_fr=(
            "Saillie osseuse médiale de l'extrémité distale de l'humérus gauche, "
            "très proéminente coude fléchi. Le nerf ulnaire passe juste en arrière "
            "dans sa gouttière : palper avec douceur. Origine commune des "
            "fléchisseurs du poignet."
        ),
        hint_en=(
            "Medial bony prominence at the distal end of the left humerus, very "
            "prominent with the elbow flexed. The ulnar nerve runs in the groove "
            "just behind it, so palpate gently. Common origin of the wrist flexors."
        ),
        body_side="left",
    ),
    Landmark(
        code="medial_epicondyle_right",
        category="BONE",
        name_fr="Épicondyle médial droit",
        name_en="Right medial epicondyle",
        hint_fr=(
            "Saillie osseuse médiale de l'extrémité distale de l'humérus droit, "
            "très proéminente coude fléchi. Le nerf ulnaire passe juste en arrière "
            "dans sa gouttière : palper avec douceur. Origine commune des "
            "fléchisseurs du poignet."
        ),
        hint_en=(
            "Medial bony prominence at the distal end of the right humerus, very "
            "prominent with the elbow flexed. The ulnar nerve runs in the groove "
            "just behind it, so palpate gently. Common origin of the wrist flexors."
        ),
        body_side="right",
    ),
    Landmark(
        code="olecranon_left",
        category="BONE",
        name_fr="Olécrâne gauche",
        name_en="Left olecranon",
        hint_fr=(
            "Pointe osseuse postérieure de l'ulna, sommet du coude gauche, "
            "facilement palpable coude fléchi à 90°. Insertion du triceps brachial. "
            "Sert de référence distale pour le site EMG du triceps (mi-distance "
            "acromion-olécrâne)."
        ),
        hint_en=(
            "Posterior bony tip of the ulna at the point of the left elbow, easily "
            "palpated with the elbow flexed to 90°. Insertion of triceps brachii "
            "and distal reference for the triceps EMG site (midway "
            "acromion-olecranon)."
        ),
        body_side="left",
    ),
    Landmark(
        code="olecranon_right",
        category="BONE",
        name_fr="Olécrâne droit",
        name_en="Right olecranon",
        hint_fr=(
            "Pointe osseuse postérieure de l'ulna, sommet du coude droit, "
            "facilement palpable coude fléchi à 90°. Insertion du triceps brachial. "
            "Sert de référence distale pour le site EMG du triceps (mi-distance "
            "acromion-olécrâne)."
        ),
        hint_en=(
            "Posterior bony tip of the ulna at the point of the right elbow, easily "
            "palpated with the elbow flexed to 90°. Insertion of triceps brachii "
            "and distal reference for the triceps EMG site (midway "
            "acromion-olecranon)."
        ),
        body_side="right",
    ),
    Landmark(
        code="radiale_left",
        category="BONE",
        name_fr="Radiale (tête radiale) gauche",
        name_en="Left radiale (head of radius)",
        hint_fr=(
            "Point le plus proximal et latéral de la tête du radius gauche. Placer "
            "le doigt dans le creux sous l'épicondyle latéral et demander une "
            "prono-supination : la tête radiale roule sous la pulpe. Extrémité "
            "distale de la mesure acromiale-radiale (ISAK)."
        ),
        hint_en=(
            "Most proximal and lateral point of the left radial head. Place a "
            "finger in the dimple below the lateral epicondyle and ask for "
            "pronation-supination: the radial head rolls under the fingertip. "
            "Distal end of the ISAK acromiale-radiale length."
        ),
        body_side="left",
    ),
    Landmark(
        code="radiale_right",
        category="BONE",
        name_fr="Radiale (tête radiale) droite",
        name_en="Right radiale (head of radius)",
        hint_fr=(
            "Point le plus proximal et latéral de la tête du radius droit. Placer "
            "le doigt dans le creux sous l'épicondyle latéral et demander une "
            "prono-supination : la tête radiale roule sous la pulpe. Extrémité "
            "distale de la mesure acromiale-radiale (ISAK)."
        ),
        hint_en=(
            "Most proximal and lateral point of the right radial head. Place a "
            "finger in the dimple below the lateral epicondyle and ask for "
            "pronation-supination: the radial head rolls under the fingertip. "
            "Distal end of the ISAK acromiale-radiale length."
        ),
        body_side="right",
    ),
    Landmark(
        code="radial_styloid_left",
        category="BONE",
        name_fr="Processus styloïde radial gauche",
        name_en="Left radial styloid process",
        hint_fr=(
            "Saillie distale du radius sur la face latérale (côté pouce) du "
            "poignet gauche, au fond de la tabatière anatomique. Elle descend plus "
            "bas que la styloïde ulnaire d'environ 1 cm. Avec la styloïde ulnaire, "
            "elle définit l'axe du poignet."
        ),
        hint_en=(
            "Distal projection of the left radius on the lateral (thumb) side of "
            "the wrist, at the floor of the anatomical snuffbox. It extends about "
            "1 cm further distally than the ulnar styloid. With the ulnar styloid "
            "it defines the wrist axis."
        ),
        body_side="left",
    ),
    Landmark(
        code="radial_styloid_right",
        category="BONE",
        name_fr="Processus styloïde radial droit",
        name_en="Right radial styloid process",
        hint_fr=(
            "Saillie distale du radius sur la face latérale (côté pouce) du "
            "poignet droit, au fond de la tabatière anatomique. Elle descend plus "
            "bas que la styloïde ulnaire d'environ 1 cm. Avec la styloïde ulnaire, "
            "elle définit l'axe du poignet."
        ),
        hint_en=(
            "Distal projection of the right radius on the lateral (thumb) side of "
            "the wrist, at the floor of the anatomical snuffbox. It extends about "
            "1 cm further distally than the ulnar styloid. With the ulnar styloid "
            "it defines the wrist axis."
        ),
        body_side="right",
    ),
    Landmark(
        code="ulnar_styloid_left",
        category="BONE",
        name_fr="Processus styloïde ulnaire gauche",
        name_en="Left ulnar styloid process",
        hint_fr=(
            "Saillie distale de l'ulna sur la face médiale (côté auriculaire) du "
            "poignet gauche, avant-bras en pronation. Bien distinguer la styloïde "
            "(petite pointe postéro-médiale) de la tête de l'ulna, plus volumineuse "
            "et plus proximale."
        ),
        hint_en=(
            "Distal projection of the left ulna on the medial (little-finger) side "
            "of the wrist, forearm pronated. Distinguish the styloid (small "
            "posteromedial spike) from the ulnar head, which is bulkier and more "
            "proximal."
        ),
        body_side="left",
    ),
    Landmark(
        code="ulnar_styloid_right",
        category="BONE",
        name_fr="Processus styloïde ulnaire droit",
        name_en="Right ulnar styloid process",
        hint_fr=(
            "Saillie distale de l'ulna sur la face médiale (côté auriculaire) du "
            "poignet droit, avant-bras en pronation. Bien distinguer la styloïde "
            "(petite pointe postéro-médiale) de la tête de l'ulna, plus volumineuse "
            "et plus proximale."
        ),
        hint_en=(
            "Distal projection of the right ulna on the medial (little-finger) side "
            "of the wrist, forearm pronated. Distinguish the styloid (small "
            "posteromedial spike) from the ulnar head, which is bulkier and more "
            "proximal."
        ),
        body_side="right",
    ),
    # ══════════════════════════════════════════════════════════════════════
    # BONE — Bassin
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="ASIS_left",
        category="BONE",
        name_fr="Épine iliaque antéro-supérieure gauche (EIAS)",
        name_en="Left anterior superior iliac spine (ASIS)",
        hint_fr=(
            "Point osseux saillant à l'extrémité antérieure de la crête iliaque "
            "gauche. Poser la main à plat sur la crête et la suivre vers l'avant "
            "jusqu'à la butée osseuse. Repère « iliospinale » en ISAK et origine de "
            "presque tous les sites EMG du quadriceps."
        ),
        hint_en=(
            "Bony prominence at the anterior end of the left iliac crest. Lay the "
            "hand flat on the crest and follow it forward until it stops against "
            "bone. This is the ISAK 'iliospinale' and the origin of nearly all "
            "quadriceps EMG lines."
        ),
        body_side="left",
    ),
    Landmark(
        code="ASIS_right",
        category="BONE",
        name_fr="Épine iliaque antéro-supérieure droite (EIAS)",
        name_en="Right anterior superior iliac spine (ASIS)",
        hint_fr=(
            "Point osseux saillant à l'extrémité antérieure de la crête iliaque "
            "droite. Poser la main à plat sur la crête et la suivre vers l'avant "
            "jusqu'à la butée osseuse. Repère « iliospinale » en ISAK et origine de "
            "presque tous les sites EMG du quadriceps."
        ),
        hint_en=(
            "Bony prominence at the anterior end of the right iliac crest. Lay the "
            "hand flat on the crest and follow it forward until it stops against "
            "bone. This is the ISAK 'iliospinale' and the origin of nearly all "
            "quadriceps EMG lines."
        ),
        body_side="right",
    ),
    Landmark(
        code="PSIS_left",
        category="BONE",
        name_fr="Épine iliaque postéro-supérieure gauche (EIPS)",
        name_en="Left posterior superior iliac spine (PSIS)",
        hint_fr=(
            "Extrémité postérieure de la crête iliaque gauche, sous la fossette "
            "cutanée de Michaelis. Suivre la crête iliaque vers l'arrière jusqu'à "
            "la saillie terminale. Avec l'EIAS, elle donne l'inclinaison du bassin "
            "dans le plan sagittal."
        ),
        hint_en=(
            "Posterior end of the left iliac crest, under the visible skin dimple. "
            "Follow the iliac crest backward to its terminal prominence. Together "
            "with the ASIS it defines sagittal pelvic tilt."
        ),
        body_side="left",
    ),
    Landmark(
        code="PSIS_right",
        category="BONE",
        name_fr="Épine iliaque postéro-supérieure droite (EIPS)",
        name_en="Right posterior superior iliac spine (PSIS)",
        hint_fr=(
            "Extrémité postérieure de la crête iliaque droite, sous la fossette "
            "cutanée de Michaelis. Suivre la crête iliaque vers l'arrière jusqu'à "
            "la saillie terminale. Avec l'EIAS, elle donne l'inclinaison du bassin "
            "dans le plan sagittal."
        ),
        hint_en=(
            "Posterior end of the right iliac crest, under the visible skin dimple. "
            "Follow the iliac crest backward to its terminal prominence. Together "
            "with the ASIS it defines sagittal pelvic tilt."
        ),
        body_side="right",
    ),
    Landmark(
        code="iliac_crest_left",
        category="BONE",
        name_fr="Crête iliaque gauche (iliocristale)",
        name_en="Left iliac crest (iliocristale)",
        hint_fr=(
            "Point le plus latéral de la crête iliaque gauche, sujet debout, poids "
            "réparti également. Presser fermement vers le haut le long de la face "
            "latérale du bassin jusqu'au bord supérieur de l'os. Repère du pli "
            "cutané de la crête iliaque (ISAK)."
        ),
        hint_en=(
            "Most lateral point of the left iliac crest, subject standing with "
            "weight evenly distributed. Press firmly upward along the lateral "
            "pelvis until the superior border of the bone is reached. Landmark for "
            "the iliac crest skinfold (ISAK)."
        ),
        body_side="left",
    ),
    Landmark(
        code="iliac_crest_right",
        category="BONE",
        name_fr="Crête iliaque droite (iliocristale)",
        name_en="Right iliac crest (iliocristale)",
        hint_fr=(
            "Point le plus latéral de la crête iliaque droite, sujet debout, poids "
            "réparti également. Presser fermement vers le haut le long de la face "
            "latérale du bassin jusqu'au bord supérieur de l'os. Repère du pli "
            "cutané de la crête iliaque et du site supra-iliaque (ISAK)."
        ),
        hint_en=(
            "Most lateral point of the right iliac crest, subject standing with "
            "weight evenly distributed. Press firmly upward along the lateral "
            "pelvis until the superior border of the bone is reached. Landmark for "
            "the iliac crest and supraspinale skinfolds (ISAK)."
        ),
        body_side="right",
    ),
    Landmark(
        code="ischial_tuberosity_left",
        category="BONE",
        name_fr="Tubérosité ischiatique gauche",
        name_en="Left ischial tuberosity",
        hint_fr=(
            "Grosse saillie osseuse sur laquelle on s'assoit, palpable sous le pli "
            "fessier gauche, hanche fléchie (sujet en décubitus latéral ou debout "
            "penché). Origine des ischio-jambiers et point de départ des lignes EMG "
            "du biceps fémoral et du semi-tendineux."
        ),
        hint_en=(
            "Large bony prominence you sit on, palpable under the left gluteal fold "
            "with the hip flexed (side-lying or standing and bent forward). Origin "
            "of the hamstrings and starting point of the biceps femoris and "
            "semitendinosus EMG lines."
        ),
        body_side="left",
    ),
    Landmark(
        code="ischial_tuberosity_right",
        category="BONE",
        name_fr="Tubérosité ischiatique droite",
        name_en="Right ischial tuberosity",
        hint_fr=(
            "Grosse saillie osseuse sur laquelle on s'assoit, palpable sous le pli "
            "fessier droit, hanche fléchie (sujet en décubitus latéral ou debout "
            "penché). Origine des ischio-jambiers et point de départ des lignes EMG "
            "du biceps fémoral et du semi-tendineux."
        ),
        hint_en=(
            "Large bony prominence you sit on, palpable under the right gluteal "
            "fold with the hip flexed (side-lying or standing and bent forward). "
            "Origin of the hamstrings and starting point of the biceps femoris and "
            "semitendinosus EMG lines."
        ),
        body_side="right",
    ),
    Landmark(
        code="greater_trochanter_left",
        category="BONE",
        name_fr="Grand trochanter gauche",
        name_en="Left greater trochanter",
        hint_fr=(
            "Apophyse latérale du fémur, sur la face externe de la hanche gauche. "
            "Poser la paume sur la face latérale et demander une rotation interne/"
            "externe de hanche : le trochanter roule sous la main. Marquer son point "
            "le plus latéral et proximal."
        ),
        hint_en=(
            "Lateral process of the femur on the outer aspect of the left hip. Rest "
            "the palm on the lateral thigh and ask for internal/external hip "
            "rotation: the trochanter rolls under the hand. Mark its most lateral "
            "and proximal point."
        ),
        body_side="left",
    ),
    Landmark(
        code="greater_trochanter_right",
        category="BONE",
        name_fr="Grand trochanter droit",
        name_en="Right greater trochanter",
        hint_fr=(
            "Apophyse latérale du fémur, sur la face externe de la hanche droite. "
            "Poser la paume sur la face latérale et demander une rotation interne/"
            "externe de hanche : le trochanter roule sous la main. Repère "
            "« trochanterion » pour la circonférence mi-cuisse (ISAK)."
        ),
        hint_en=(
            "Lateral process of the femur on the outer aspect of the right hip. "
            "Rest the palm on the lateral thigh and ask for internal/external hip "
            "rotation: the trochanter rolls under the hand. ISAK 'trochanterion' "
            "for the mid-thigh girth."
        ),
        body_side="right",
    ),
    Landmark(
        code="sacrum_S2",
        category="BONE",
        name_fr="Sacrum (niveau S2)",
        name_en="Sacrum (S2 level)",
        hint_fr=(
            "Point médian de la face postérieure du sacrum, sur la ligne joignant "
            "les deux EIPS (≈ 2e vertèbre sacrée). Surface plane et lisse entre les "
            "deux masses fessières. Origine de la ligne EMG du grand fessier "
            "(sacrum → grand trochanter)."
        ),
        hint_en=(
            "Midpoint of the posterior sacrum, on the line joining both PSIS "
            "(≈ second sacral vertebra). Flat, smooth surface between the two "
            "gluteal masses. Origin of the gluteus maximus EMG line "
            "(sacrum → greater trochanter)."
        ),
    ),
    # ══════════════════════════════════════════════════════════════════════
    # BONE — Membre inférieur
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="patella_superior_left",
        category="BONE",
        name_fr="Bord supérieur de la patella gauche",
        name_en="Left superior border of the patella",
        hint_fr=(
            "Bord supérieur (base) de la rotule gauche, genou en extension et "
            "quadriceps relâché. Pousser la rotule vers le haut avec le pouce pour "
            "sentir nettement son bord. Extrémité distale de la ligne EMG du droit "
            "fémoral (EIAS → bord supérieur de la patella)."
        ),
        hint_en=(
            "Superior border (base) of the left patella, knee extended and "
            "quadriceps relaxed. Push the patella upward with the thumb to feel its "
            "border clearly. Distal end of the rectus femoris EMG line "
            "(ASIS → superior patellar border)."
        ),
        body_side="left",
    ),
    Landmark(
        code="patella_superior_right",
        category="BONE",
        name_fr="Bord supérieur de la patella droite",
        name_en="Right superior border of the patella",
        hint_fr=(
            "Bord supérieur (base) de la rotule droite, genou en extension et "
            "quadriceps relâché. Pousser la rotule vers le haut avec le pouce pour "
            "sentir nettement son bord. Sert aussi de repère « mid-patellare » pour "
            "le pli de la cuisse antérieure (ISAK)."
        ),
        hint_en=(
            "Superior border (base) of the right patella, knee extended and "
            "quadriceps relaxed. Push the patella upward with the thumb to feel its "
            "border clearly. Also the 'mid-patellare' reference for the front thigh "
            "skinfold (ISAK)."
        ),
        body_side="right",
    ),
    Landmark(
        code="lateral_knee_left",
        category="BONE",
        name_fr="Épicondyle latéral du fémur gauche",
        name_en="Left lateral femoral epicondyle",
        hint_fr=(
            "Saillie osseuse la plus latérale du condyle fémoral gauche, juste "
            "au-dessus de l'interligne articulaire. Genou fléchi à 90°, suivre le "
            "bord latéral de la trochlée vers l'arrière. Définit avec l'épicondyle "
            "médial l'axe de flexion du genou et le diamètre du fémur (ISAK)."
        ),
        hint_en=(
            "Most lateral bony prominence of the left femoral condyle, just above "
            "the joint line. With the knee flexed to 90°, follow the lateral edge "
            "of the trochlea backward. With the medial epicondyle it defines the "
            "knee flexion axis and the ISAK femur breadth."
        ),
        body_side="left",
    ),
    Landmark(
        code="lateral_knee_right",
        category="BONE",
        name_fr="Épicondyle latéral du fémur droit",
        name_en="Right lateral femoral epicondyle",
        hint_fr=(
            "Saillie osseuse la plus latérale du condyle fémoral droit, juste "
            "au-dessus de l'interligne articulaire. Genou fléchi à 90°, suivre le "
            "bord latéral de la trochlée vers l'arrière. Définit avec l'épicondyle "
            "médial l'axe de flexion du genou et le diamètre du fémur (ISAK)."
        ),
        hint_en=(
            "Most lateral bony prominence of the right femoral condyle, just above "
            "the joint line. With the knee flexed to 90°, follow the lateral edge "
            "of the trochlea backward. With the medial epicondyle it defines the "
            "knee flexion axis and the ISAK femur breadth."
        ),
        body_side="right",
    ),
    Landmark(
        code="medial_knee_left",
        category="BONE",
        name_fr="Épicondyle médial du fémur gauche",
        name_en="Left medial femoral epicondyle",
        hint_fr=(
            "Saillie osseuse la plus médiale du condyle fémoral gauche, au-dessus "
            "de l'interligne. C'est l'insertion proximale du ligament collatéral "
            "médial et la référence proximale du site EMG du soléaire (condyle "
            "médial → malléole médiale)."
        ),
        hint_en=(
            "Most medial bony prominence of the left femoral condyle, above the "
            "joint line. It is the proximal attachment of the medial collateral "
            "ligament and the proximal reference for the soleus EMG site (medial "
            "condyle → medial malleolus)."
        ),
        body_side="left",
    ),
    Landmark(
        code="medial_knee_right",
        category="BONE",
        name_fr="Épicondyle médial du fémur droit",
        name_en="Right medial femoral epicondyle",
        hint_fr=(
            "Saillie osseuse la plus médiale du condyle fémoral droit, au-dessus "
            "de l'interligne. C'est l'insertion proximale du ligament collatéral "
            "médial et la référence proximale du site EMG du soléaire (condyle "
            "médial → malléole médiale)."
        ),
        hint_en=(
            "Most medial bony prominence of the right femoral condyle, above the "
            "joint line. It is the proximal attachment of the medial collateral "
            "ligament and the proximal reference for the soleus EMG site (medial "
            "condyle → medial malleolus)."
        ),
        body_side="right",
    ),
    Landmark(
        code="fibular_head_left",
        category="BONE",
        name_fr="Tête de la fibula gauche",
        name_en="Left head of the fibula",
        hint_fr=(
            "Bosse osseuse arrondie sur la face latérale de la jambe gauche, "
            "environ 2 cm sous l'interligne du genou et en arrière du tubercule "
            "tibial. Le nerf fibulaire commun contourne le col juste en dessous : "
            "palper sans presser. Origine des lignes EMG tibial antérieur et fibulaires."
        ),
        hint_en=(
            "Rounded bony bump on the lateral aspect of the left leg, about 2 cm "
            "below the knee joint line and behind the tibial tubercle. The common "
            "fibular nerve wraps around the neck just below it, so palpate without "
            "pressing. Origin of the tibialis anterior and fibularis EMG lines."
        ),
        body_side="left",
    ),
    Landmark(
        code="fibular_head_right",
        category="BONE",
        name_fr="Tête de la fibula droite",
        name_en="Right head of the fibula",
        hint_fr=(
            "Bosse osseuse arrondie sur la face latérale de la jambe droite, "
            "environ 2 cm sous l'interligne du genou et en arrière du tubercule "
            "tibial. Le nerf fibulaire commun contourne le col juste en dessous : "
            "palper sans presser. Repère « tibiale laterale » pour la mi-cuisse (ISAK)."
        ),
        hint_en=(
            "Rounded bony bump on the lateral aspect of the right leg, about 2 cm "
            "below the knee joint line and behind the tibial tubercle. The common "
            "fibular nerve wraps around the neck just below it, so palpate without "
            "pressing. ISAK 'tibiale laterale' reference for mid-thigh."
        ),
        body_side="right",
    ),
    Landmark(
        code="tibial_tuberosity_left",
        category="BONE",
        name_fr="Tubérosité tibiale gauche",
        name_en="Left tibial tuberosity",
        hint_fr=(
            "Relief osseux triangulaire sur la face antérieure du tibia gauche, "
            "2 à 3 cm sous la pointe de la rotule. Descendre le long du tendon "
            "patellaire jusqu'à son insertion osseuse. Site d'insertion du "
            "quadriceps et repère clinique (Osgood-Schlatter)."
        ),
        hint_en=(
            "Triangular bony ridge on the anterior surface of the left tibia, 2 to "
            "3 cm below the apex of the patella. Follow the patellar tendon down to "
            "its bony insertion. Quadriceps insertion site and a clinical landmark "
            "(Osgood-Schlatter)."
        ),
        body_side="left",
    ),
    Landmark(
        code="tibial_tuberosity_right",
        category="BONE",
        name_fr="Tubérosité tibiale droite",
        name_en="Right tibial tuberosity",
        hint_fr=(
            "Relief osseux triangulaire sur la face antérieure du tibia droit, "
            "2 à 3 cm sous la pointe de la rotule. Descendre le long du tendon "
            "patellaire jusqu'à son insertion osseuse. Site d'insertion du "
            "quadriceps et repère clinique (Osgood-Schlatter)."
        ),
        hint_en=(
            "Triangular bony ridge on the anterior surface of the right tibia, 2 to "
            "3 cm below the apex of the patella. Follow the patellar tendon down to "
            "its bony insertion. Quadriceps insertion site and a clinical landmark "
            "(Osgood-Schlatter)."
        ),
        body_side="right",
    ),
    Landmark(
        code="lateral_malleolus_left",
        category="BONE",
        name_fr="Malléole latérale gauche",
        name_en="Left lateral malleolus",
        hint_fr=(
            "Extrémité distale de la fibula, saillante sur la face latérale de la "
            "cheville gauche. Marquer son point le plus distal : elle descend plus "
            "bas et se situe plus en arrière que la malléole médiale. Extrémité "
            "distale de la ligne EMG des fibulaires."
        ),
        hint_en=(
            "Distal end of the left fibula, prominent on the lateral ankle. Mark "
            "its most distal point: it sits lower and more posterior than the "
            "medial malleolus. Distal end of the fibularis EMG line."
        ),
        body_side="left",
    ),
    Landmark(
        code="lateral_malleolus_right",
        category="BONE",
        name_fr="Malléole latérale droite",
        name_en="Right lateral malleolus",
        hint_fr=(
            "Extrémité distale de la fibula, saillante sur la face latérale de la "
            "cheville droite. Marquer son point le plus distal : elle descend plus "
            "bas et se situe plus en arrière que la malléole médiale. Extrémité "
            "distale de la ligne EMG des fibulaires."
        ),
        hint_en=(
            "Distal end of the right fibula, prominent on the lateral ankle. Mark "
            "its most distal point: it sits lower and more posterior than the "
            "medial malleolus. Distal end of the fibularis EMG line."
        ),
        body_side="right",
    ),
    Landmark(
        code="medial_malleolus_left",
        category="BONE",
        name_fr="Malléole médiale gauche",
        name_en="Left medial malleolus",
        hint_fr=(
            "Extrémité distale du tibia, saillante sur la face médiale de la "
            "cheville gauche. Marquer son point le plus distal. Extrémité distale "
            "des lignes EMG du soléaire et du tibial antérieur."
        ),
        hint_en=(
            "Distal end of the left tibia, prominent on the medial ankle. Mark its "
            "most distal point. Distal end of the soleus and tibialis anterior EMG "
            "lines."
        ),
        body_side="left",
    ),
    Landmark(
        code="medial_malleolus_right",
        category="BONE",
        name_fr="Malléole médiale droite",
        name_en="Right medial malleolus",
        hint_fr=(
            "Extrémité distale du tibia, saillante sur la face médiale de la "
            "cheville droite. Marquer son point le plus distal. Extrémité distale "
            "des lignes EMG du soléaire et du tibial antérieur."
        ),
        hint_en=(
            "Distal end of the right tibia, prominent on the medial ankle. Mark its "
            "most distal point. Distal end of the soleus and tibialis anterior EMG "
            "lines."
        ),
        body_side="right",
    ),
    Landmark(
        code="heel_left",
        category="BONE",
        name_fr="Talon gauche (calcanéus)",
        name_en="Left heel (calcaneus)",
        hint_fr=(
            "Point le plus postérieur du calcanéus gauche, au niveau de l'insertion "
            "du tendon calcanéen, pied à plat au sol. Marquer sur la ligne médiane "
            "postérieure du talon. Référence distale de la ligne EMG du gastrocnémien "
            "latéral (tête fibulaire → talon)."
        ),
        hint_en=(
            "Most posterior point of the left calcaneus, at the level of the "
            "Achilles insertion, with the foot flat on the ground. Mark on the "
            "posterior midline of the heel. Distal reference of the lateral "
            "gastrocnemius EMG line (fibular head → heel)."
        ),
        body_side="left",
    ),
    Landmark(
        code="heel_right",
        category="BONE",
        name_fr="Talon droit (calcanéus)",
        name_en="Right heel (calcaneus)",
        hint_fr=(
            "Point le plus postérieur du calcanéus droit, au niveau de l'insertion "
            "du tendon calcanéen, pied à plat au sol. Marquer sur la ligne médiane "
            "postérieure du talon. Référence distale de la ligne EMG du gastrocnémien "
            "latéral (tête fibulaire → talon)."
        ),
        hint_en=(
            "Most posterior point of the right calcaneus, at the level of the "
            "Achilles insertion, with the foot flat on the ground. Mark on the "
            "posterior midline of the heel. Distal reference of the lateral "
            "gastrocnemius EMG line (fibular head → heel)."
        ),
        body_side="right",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # EMG — Quadriceps (SENIAM)
    # Le vaste intermédiaire est profond : pas de site EMG de surface.
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_rectus_femoris_left",
        category="EMG",
        name_fr="Droit fémoral gauche (EMG)",
        name_en="Left rectus femoris (EMG)",
        hint_fr=(
            "Tracer la ligne ASIS_left → patella_superior_left et marquer à 50 % "
            "de cette distance, sur la face antérieure de la cuisse. Orientation : "
            "paire d'électrodes parallèle à la ligne, dans l'axe longitudinal de la "
            "cuisse. Vérifier par une extension de genou contre résistance."
        ),
        hint_en=(
            "Draw the line ASIS_left → patella_superior_left and mark at 50 % of "
            "that distance on the anterior thigh. Orientation: electrode pair "
            "parallel to the line, along the long axis of the thigh. Confirm with a "
            "resisted knee extension."
        ),
        theme="knee_rehab",
        application_fr="Évalue la participation du droit fémoral dans le contrôle de l'extension du genou. Utilisé pour détecter le 'stiff-knee gait' en analyse de la marche neurologique et pour le suivi post-LCA.",
        application_en="Assesses rectus femoris contribution to knee extension control. Used to detect stiff-knee gait in neurological gait analysis and for post-ACL rehabilitation follow-up.",
        body_side="left",
    ),
    Landmark(
        code="EMG_rectus_femoris_right",
        category="EMG",
        name_fr="Droit fémoral droit (EMG)",
        name_en="Right rectus femoris (EMG)",
        hint_fr=(
            "Tracer la ligne ASIS_right → patella_superior_right et marquer à 50 % "
            "de cette distance, sur la face antérieure de la cuisse. Orientation : "
            "paire d'électrodes parallèle à la ligne, dans l'axe longitudinal de la "
            "cuisse. Vérifier par une extension de genou contre résistance."
        ),
        hint_en=(
            "Draw the line ASIS_right → patella_superior_right and mark at 50 % of "
            "that distance on the anterior thigh. Orientation: electrode pair "
            "parallel to the line, along the long axis of the thigh. Confirm with a "
            "resisted knee extension."
        ),
        theme="knee_rehab",
        application_fr="Évalue la participation du droit fémoral dans le contrôle de l'extension du genou. Utilisé pour détecter le 'stiff-knee gait' en analyse de la marche neurologique et pour le suivi post-LCA.",
        application_en="Assesses rectus femoris contribution to knee extension control. Used to detect stiff-knee gait in neurological gait analysis and for post-ACL rehabilitation follow-up.",
        body_side="right",
    ),
    Landmark(
        code="EMG_vastus_lateralis_left",
        category="EMG",
        name_fr="Vaste latéral gauche (EMG)",
        name_en="Left vastus lateralis (EMG)",
        hint_fr=(
            "Marquer à 2/3 de la distance entre ASIS_left et le bord latéral de la "
            "patella, sur la face antéro-latérale de la cuisse. Orientation : la "
            "paire d'électrodes forme un angle d'environ 20° avec cette ligne, dans "
            "la direction oblique des fibres. Rester en avant de la bandelette ilio-tibiale."
        ),
        hint_en=(
            "Mark at 2/3 of the distance between ASIS_left and the lateral border "
            "of the patella, on the anterolateral thigh. Orientation: the electrode "
            "pair sits at about 20° to that line, following the oblique fibre "
            "direction. Stay anterior to the iliotibial band."
        ),
        theme="knee_rehab",
        application_fr="Ratio VM/VL : marqueur clé du syndrome fémoro-patellaire. Un VL dominant (activé en avance) indique un déséquilibre latéral de la patella. Suivi du retour au sport post-LCA.",
        application_en="VM/VL ratio: key marker of patellofemoral pain syndrome. A dominant VL (earlier onset) indicates lateral patellar imbalance. Used in post-ACL return-to-sport monitoring.",
        body_side="left",
    ),
    Landmark(
        code="EMG_vastus_lateralis_right",
        category="EMG",
        name_fr="Vaste latéral droit (EMG)",
        name_en="Right vastus lateralis (EMG)",
        hint_fr=(
            "Marquer à 2/3 de la distance entre ASIS_right et le bord latéral de la "
            "patella, sur la face antéro-latérale de la cuisse. Orientation : la "
            "paire d'électrodes forme un angle d'environ 20° avec cette ligne, dans "
            "la direction oblique des fibres. Rester en avant de la bandelette ilio-tibiale."
        ),
        hint_en=(
            "Mark at 2/3 of the distance between ASIS_right and the lateral border "
            "of the patella, on the anterolateral thigh. Orientation: the electrode "
            "pair sits at about 20° to that line, following the oblique fibre "
            "direction. Stay anterior to the iliotibial band."
        ),
        theme="knee_rehab",
        application_fr="Ratio VM/VL : marqueur clé du syndrome fémoro-patellaire. Un VL dominant (activé en avance) indique un déséquilibre latéral de la patella. Suivi du retour au sport post-LCA.",
        application_en="VM/VL ratio: key marker of patellofemoral pain syndrome. A dominant VL (earlier onset) indicates lateral patellar imbalance. Used in post-ACL return-to-sport monitoring.",
        body_side="right",
    ),
    Landmark(
        code="EMG_vastus_medialis_left",
        category="EMG",
        name_fr="Vaste médial gauche (EMG)",
        name_en="Left vastus medialis (EMG)",
        hint_fr=(
            "Marquer à 80 % de la distance entre ASIS_left et l'interligne médial "
            "du genou (bord antérieur du ligament collatéral médial), sur le "
            "renflement musculaire juste au-dessus de la patella. Orientation : "
            "très oblique, environ 50-55° par rapport à cette ligne, presque "
            "horizontale vers le bas et l'avant."
        ),
        hint_en=(
            "Mark at 80 % of the distance between ASIS_left and the medial knee "
            "joint line (anterior border of the medial collateral ligament), on the "
            "muscle bulge just above the patella. Orientation: strongly oblique, "
            "about 50-55° to that line, almost horizontal and directed "
            "downward-forward."
        ),
        theme="knee_rehab",
        application_fr="Timing VMO-VL et verrouillage terminal du genou. L'orientation ~50-55° est critique : une erreur de placement change le signal. Douleur fémoro-patellaire, arthrose fémoro-tibiale.",
        application_en="VMO-VL timing and terminal knee lock. The ~50-55° orientation is critical: a placement error changes the signal. Patellofemoral pain, femorotibial osteoarthritis.",
        body_side="left",
    ),
    Landmark(
        code="EMG_vastus_medialis_right",
        category="EMG",
        name_fr="Vaste médial droit (EMG)",
        name_en="Right vastus medialis (EMG)",
        hint_fr=(
            "Marquer à 80 % de la distance entre ASIS_right et l'interligne médial "
            "du genou (bord antérieur du ligament collatéral médial), sur le "
            "renflement musculaire juste au-dessus de la patella. Orientation : "
            "très oblique, environ 50-55° par rapport à cette ligne, presque "
            "horizontale vers le bas et l'avant."
        ),
        hint_en=(
            "Mark at 80 % of the distance between ASIS_right and the medial knee "
            "joint line (anterior border of the medial collateral ligament), on the "
            "muscle bulge just above the patella. Orientation: strongly oblique, "
            "about 50-55° to that line, almost horizontal and directed "
            "downward-forward."
        ),
        theme="knee_rehab",
        application_fr="Timing VMO-VL et verrouillage terminal du genou. L'orientation ~50-55° est critique : une erreur de placement change le signal. Douleur fémoro-patellaire, arthrose fémoro-tibiale.",
        application_en="VMO-VL timing and terminal knee lock. The ~50-55° orientation is critical: a placement error changes the signal. Patellofemoral pain, femorotibial osteoarthritis.",
        body_side="right",
    ),
    # ══════════════════════════════════════════════════════════════════════
    # EMG — Ischio-jambiers (SENIAM)
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_biceps_femoris_left",
        category="EMG",
        name_fr="Biceps fémoral gauche (EMG)",
        name_en="Left biceps femoris (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne ischial_tuberosity_left → lateral_knee_left "
            "(épicondyle latéral), sur la face postéro-latérale de la cuisse. "
            "Orientation : électrodes parallèles à cette ligne. Le tendon distal se "
            "repère bien en fléchissant le genou contre résistance."
        ),
        hint_en=(
            "Mark at 50 % of the line ischial_tuberosity_left → lateral_knee_left "
            "(lateral epicondyle), on the posterolateral thigh. Orientation: "
            "electrodes parallel to that line. The distal tendon is easy to find "
            "with a resisted knee flexion."
        ),
        theme="knee_rehab",
        application_fr="Prévention des lésions des ischio-jambiers (Nordic hamstring). Co-activation ischio/quadriceps en protection du LCA. Analyse du sprint (phase de fin d'oscillation).",
        application_en="Hamstring injury prevention (Nordic hamstring). Hamstring/quadriceps co-activation for ACL protection. Sprint analysis (terminal swing phase).",
        body_side="left",
    ),
    Landmark(
        code="EMG_biceps_femoris_right",
        category="EMG",
        name_fr="Biceps fémoral droit (EMG)",
        name_en="Right biceps femoris (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne ischial_tuberosity_right → "
            "lateral_knee_right (épicondyle latéral), sur la face postéro-latérale "
            "de la cuisse. Orientation : électrodes parallèles à cette ligne. Le "
            "tendon distal se repère bien en fléchissant le genou contre résistance."
        ),
        hint_en=(
            "Mark at 50 % of the line ischial_tuberosity_right → "
            "lateral_knee_right (lateral epicondyle), on the posterolateral thigh. "
            "Orientation: electrodes parallel to that line. The distal tendon is "
            "easy to find with a resisted knee flexion."
        ),
        theme="knee_rehab",
        application_fr="Prévention des lésions des ischio-jambiers (Nordic hamstring). Co-activation ischio/quadriceps en protection du LCA. Analyse du sprint (phase de fin d'oscillation).",
        application_en="Hamstring injury prevention (Nordic hamstring). Hamstring/quadriceps co-activation for ACL protection. Sprint analysis (terminal swing phase).",
        body_side="right",
    ),
    Landmark(
        code="EMG_semitendinosus_left",
        category="EMG",
        name_fr="Semi-tendineux gauche (EMG)",
        name_en="Left semitendinosus (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne ischial_tuberosity_left → medial_knee_left "
            "(épicondyle médial), sur la face postéro-médiale de la cuisse. "
            "Orientation : électrodes parallèles à cette ligne. Attention au "
            "cross-talk avec le semi-membraneux, situé plus profondément et en dedans."
        ),
        hint_en=(
            "Mark at 50 % of the line ischial_tuberosity_left → medial_knee_left "
            "(medial epicondyle), on the posteromedial thigh. Orientation: "
            "electrodes parallel to that line. Beware of cross-talk with "
            "semimembranosus, which lies deeper and more medially."
        ),
        theme="knee_rehab",
        application_fr="Asymétrie médial/latéral après prélèvement de greffe DIDT (demi-tendineux + gracile). Évaluation de la rotation tibiale interne. Cross-talk avec le semi-membraneux (point de cours).",
        application_en="Medial/lateral asymmetry after DIDT graft harvest. Assessment of internal tibial rotation. Cross-talk with semimembranosus (teaching point).",
        body_side="left",
    ),
    Landmark(
        code="EMG_semitendinosus_right",
        category="EMG",
        name_fr="Semi-tendineux droit (EMG)",
        name_en="Right semitendinosus (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne ischial_tuberosity_right → "
            "medial_knee_right (épicondyle médial), sur la face postéro-médiale de "
            "la cuisse. Orientation : électrodes parallèles à cette ligne. Attention "
            "au cross-talk avec le semi-membraneux, plus profond et plus médial."
        ),
        hint_en=(
            "Mark at 50 % of the line ischial_tuberosity_right → "
            "medial_knee_right (medial epicondyle), on the posteromedial thigh. "
            "Orientation: electrodes parallel to that line. Beware of cross-talk "
            "with semimembranosus, which lies deeper and more medially."
        ),
        theme="knee_rehab",
        application_fr="Asymétrie médial/latéral après prélèvement de greffe DIDT (demi-tendineux + gracile). Évaluation de la rotation tibiale interne. Cross-talk avec le semi-membraneux (point de cours).",
        application_en="Medial/lateral asymmetry after DIDT graft harvest. Assessment of internal tibial rotation. Cross-talk with semimembranosus (teaching point).",
        body_side="right",
    ),
    # ══════════════════════════════════════════════════════════════════════
    # EMG — Triceps sural et loge antéro-latérale (SENIAM)
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_gastrocnemius_medialis_left",
        category="EMG",
        name_fr="Gastrocnémien médial gauche (EMG)",
        name_en="Left medial gastrocnemius (EMG)",
        hint_fr=(
            "Marquer sur la partie la plus bombée du chef médial, environ au tiers "
            "proximal de la jambe, sujet debout sur la pointe des pieds pour faire "
            "saillir le ventre musculaire. Orientation : électrodes parallèles à "
            "l'axe longitudinal de la jambe."
        ),
        hint_en=(
            "Mark on the most prominent bulge of the medial head, roughly at the "
            "proximal third of the leg, with the subject up on tiptoe so the muscle "
            "belly stands out. Orientation: electrodes parallel to the long axis of "
            "the leg."
        ),
        theme="gait",
        application_fr="Propulsion (push-off) à la marche et à la course. Tendinopathie d'Achille (comparaison bilatérale). Contrôle de l'équin spastique post-AVC.",
        application_en="Push-off during walking and running. Achilles tendinopathy (bilateral comparison). Spastic equinus control post-stroke.",
        body_side="left",
    ),
    Landmark(
        code="EMG_gastrocnemius_medialis_right",
        category="EMG",
        name_fr="Gastrocnémien médial droit (EMG)",
        name_en="Right medial gastrocnemius (EMG)",
        hint_fr=(
            "Marquer sur la partie la plus bombée du chef médial, environ au tiers "
            "proximal de la jambe, sujet debout sur la pointe des pieds pour faire "
            "saillir le ventre musculaire. Orientation : électrodes parallèles à "
            "l'axe longitudinal de la jambe."
        ),
        hint_en=(
            "Mark on the most prominent bulge of the medial head, roughly at the "
            "proximal third of the leg, with the subject up on tiptoe so the muscle "
            "belly stands out. Orientation: electrodes parallel to the long axis of "
            "the leg."
        ),
        theme="gait",
        application_fr="Propulsion (push-off) à la marche et à la course. Tendinopathie d'Achille (comparaison bilatérale). Contrôle de l'équin spastique post-AVC.",
        application_en="Push-off during walking and running. Achilles tendinopathy (bilateral comparison). Spastic equinus control post-stroke.",
        body_side="right",
    ),
    Landmark(
        code="EMG_gastrocnemius_lateralis_left",
        category="EMG",
        name_fr="Gastrocnémien latéral gauche (EMG)",
        name_en="Left lateral gastrocnemius (EMG)",
        hint_fr=(
            "Marquer à 1/3 de la ligne fibular_head_left → heel_left, sur le ventre "
            "du chef latéral. Orientation : électrodes parallèles à l'axe "
            "longitudinal de la jambe. Vérifier en demandant une flexion plantaire "
            "genou tendu."
        ),
        hint_en=(
            "Mark at 1/3 of the line fibular_head_left → heel_left, on the belly of "
            "the lateral head. Orientation: electrodes parallel to the long axis of "
            "the leg. Confirm with a plantarflexion performed with the knee extended."
        ),
        theme="gait",
        application_fr="Propulsion à la course et analyse du saut. Asymétrie du triceps sural (latéral vs médial). Comparaison bilatérale en réhabilitation de la cheville.",
        application_en="Propulsion during running and jump analysis. Triceps surae asymmetry (lateral vs medial). Bilateral comparison in ankle rehabilitation.",
        body_side="left",
    ),
    Landmark(
        code="EMG_gastrocnemius_lateralis_right",
        category="EMG",
        name_fr="Gastrocnémien latéral droit (EMG)",
        name_en="Right lateral gastrocnemius (EMG)",
        hint_fr=(
            "Marquer à 1/3 de la ligne fibular_head_right → heel_right, sur le "
            "ventre du chef latéral. Orientation : électrodes parallèles à l'axe "
            "longitudinal de la jambe. Vérifier en demandant une flexion plantaire "
            "genou tendu."
        ),
        hint_en=(
            "Mark at 1/3 of the line fibular_head_right → heel_right, on the belly "
            "of the lateral head. Orientation: electrodes parallel to the long axis "
            "of the leg. Confirm with a plantarflexion performed with the knee "
            "extended."
        ),
        theme="gait",
        application_fr="Propulsion à la course et analyse du saut. Asymétrie du triceps sural (latéral vs médial). Comparaison bilatérale en réhabilitation de la cheville.",
        application_en="Propulsion during running and jump analysis. Triceps surae asymmetry (lateral vs medial). Bilateral comparison in ankle rehabilitation.",
        body_side="right",
    ),
    Landmark(
        code="EMG_soleus_left",
        category="EMG",
        name_fr="Soléaire gauche (EMG)",
        name_en="Left soleus (EMG)",
        hint_fr=(
            "Marquer à 2/3 de la ligne medial_knee_left → medial_malleolus_left, "
            "sur la face postéro-médiale de la jambe, sous le ventre des "
            "gastrocnémiens. Orientation : électrodes parallèles à l'axe de la "
            "jambe. Le soléaire se distingue en fléchissant le genou (gastrocnémiens relâchés)."
        ),
        hint_en=(
            "Mark at 2/3 of the line medial_knee_left → medial_malleolus_left, on "
            "the posteromedial leg, below the gastrocnemius bellies. Orientation: "
            "electrodes parallel to the leg axis. Soleus is isolated by flexing the "
            "knee, which slackens the gastrocnemii."
        ),
        theme="gait",
        application_fr="Principal muscle antigravitaire du contrôle postural debout (oscillations sagittales). Tendinopathie d'Achille. Marche en montée. Cross-talk avec les gastrocnémiens (enseigner l'isolation par flexion du genou).",
        application_en="Main anti-gravity muscle in quiet standing balance (sagittal sway). Achilles tendinopathy. Uphill walking. Cross-talk with gastrocnemii (teach isolation via knee flexion).",
        body_side="left",
    ),
    Landmark(
        code="EMG_soleus_right",
        category="EMG",
        name_fr="Soléaire droit (EMG)",
        name_en="Right soleus (EMG)",
        hint_fr=(
            "Marquer à 2/3 de la ligne medial_knee_right → medial_malleolus_right, "
            "sur la face postéro-médiale de la jambe, sous le ventre des "
            "gastrocnémiens. Orientation : électrodes parallèles à l'axe de la "
            "jambe. Le soléaire se distingue en fléchissant le genou (gastrocnémiens relâchés)."
        ),
        hint_en=(
            "Mark at 2/3 of the line medial_knee_right → medial_malleolus_right, on "
            "the posteromedial leg, below the gastrocnemius bellies. Orientation: "
            "electrodes parallel to the leg axis. Soleus is isolated by flexing the "
            "knee, which slackens the gastrocnemii."
        ),
        theme="gait",
        application_fr="Principal muscle antigravitaire du contrôle postural debout (oscillations sagittales). Tendinopathie d'Achille. Marche en montée. Cross-talk avec les gastrocnémiens (enseigner l'isolation par flexion du genou).",
        application_en="Main anti-gravity muscle in quiet standing balance (sagittal sway). Achilles tendinopathy. Uphill walking. Cross-talk with gastrocnemii (teach isolation via knee flexion).",
        body_side="right",
    ),
    Landmark(
        code="EMG_tibialis_anterior_left",
        category="EMG",
        name_fr="Tibial antérieur gauche (EMG)",
        name_en="Left tibialis anterior (EMG)",
        hint_fr=(
            "Marquer à 1/3 de la ligne fibular_head_left → medial_malleolus_left, "
            "juste en dehors de la crête tibiale (sur le ventre musculaire, jamais "
            "sur l'os). Orientation : électrodes parallèles à cette ligne. Vérifier "
            "par une dorsiflexion-inversion contre résistance."
        ),
        hint_en=(
            "Mark at 1/3 of the line fibular_head_left → medial_malleolus_left, "
            "just lateral to the tibial crest (on the muscle belly, never on bone). "
            "Orientation: electrodes parallel to that line. Confirm with a resisted "
            "dorsiflexion-inversion."
        ),
        theme="gait",
        application_fr="Contrôle de l'attaque du talon et prévention du steppage (drop foot). Cible de la FES (stimulation électrique fonctionnelle). Périostite tibiale de stress.",
        application_en="Heel strike control and foot drop prevention. FES (functional electrical stimulation) target. Tibial stress syndrome (shin splints).",
        body_side="left",
    ),
    Landmark(
        code="EMG_tibialis_anterior_right",
        category="EMG",
        name_fr="Tibial antérieur droit (EMG)",
        name_en="Right tibialis anterior (EMG)",
        hint_fr=(
            "Marquer à 1/3 de la ligne fibular_head_right → medial_malleolus_right, "
            "juste en dehors de la crête tibiale (sur le ventre musculaire, jamais "
            "sur l'os). Orientation : électrodes parallèles à cette ligne. Vérifier "
            "par une dorsiflexion-inversion contre résistance."
        ),
        hint_en=(
            "Mark at 1/3 of the line fibular_head_right → medial_malleolus_right, "
            "just lateral to the tibial crest (on the muscle belly, never on bone). "
            "Orientation: electrodes parallel to that line. Confirm with a resisted "
            "dorsiflexion-inversion."
        ),
        theme="gait",
        application_fr="Contrôle de l'attaque du talon et prévention du steppage (drop foot). Cible de la FES (stimulation électrique fonctionnelle). Périostite tibiale de stress.",
        application_en="Heel strike control and foot drop prevention. FES (functional electrical stimulation) target. Tibial stress syndrome (shin splints).",
        body_side="right",
    ),
    Landmark(
        code="EMG_peroneus_longus_left",
        category="EMG",
        name_fr="Long fibulaire gauche (EMG)",
        name_en="Left fibularis (peroneus) longus (EMG)",
        hint_fr=(
            "Marquer à 25 % de la ligne fibular_head_left → lateral_malleolus_left, "
            "sur la loge latérale de la jambe. Orientation : électrodes parallèles "
            "à l'axe de la fibula. Vérifier par une éversion contre résistance ; ce "
            "site est sensible au cross-talk avec le court fibulaire."
        ),
        hint_en=(
            "Mark at 25 % of the line fibular_head_left → lateral_malleolus_left, "
            "over the lateral compartment of the leg. Orientation: electrodes "
            "parallel to the fibular axis. Confirm with a resisted eversion; this "
            "site is prone to cross-talk with fibularis brevis."
        ),
        theme="lower_limb",
        application_fr="Latence péronière après entorse latérale de cheville (instabilité chronique). Contrôle de l'éversion et de la proprioception. Cross-talk avec le court fibulaire.",
        application_en="Peroneal latency after lateral ankle sprain (chronic instability). Eversion control and proprioception. Cross-talk with fibularis brevis.",
        body_side="left",
    ),
    Landmark(
        code="EMG_peroneus_longus_right",
        category="EMG",
        name_fr="Long fibulaire droit (EMG)",
        name_en="Right fibularis (peroneus) longus (EMG)",
        hint_fr=(
            "Marquer à 25 % de la ligne fibular_head_right → "
            "lateral_malleolus_right, sur la loge latérale de la jambe. Orientation : "
            "électrodes parallèles à l'axe de la fibula. Vérifier par une éversion "
            "contre résistance ; site sensible au cross-talk avec le court fibulaire."
        ),
        hint_en=(
            "Mark at 25 % of the line fibular_head_right → "
            "lateral_malleolus_right, over the lateral compartment of the leg. "
            "Orientation: electrodes parallel to the fibular axis. Confirm with a "
            "resisted eversion; this site is prone to cross-talk with fibularis brevis."
        ),
        theme="lower_limb",
        application_fr="Latence péronière après entorse latérale de cheville (instabilité chronique). Contrôle de l'éversion et de la proprioception. Cross-talk avec le court fibulaire.",
        application_en="Peroneal latency after lateral ankle sprain (chronic instability). Eversion control and proprioception. Cross-talk with fibularis brevis.",
        body_side="right",
    ),
    # ══════════════════════════════════════════════════════════════════════
    # EMG — Hanche (SENIAM)
    # Le petit fessier est profond : pas de site EMG de surface.
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_gluteus_maximus_left",
        category="EMG",
        name_fr="Grand fessier gauche (EMG)",
        name_en="Left gluteus maximus (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne sacrum_S2 → greater_trochanter_left, au "
            "centre de la masse fessière. Orientation : électrodes parallèles à "
            "cette ligne, dans la direction oblique des fibres. Vérifier par une "
            "extension de hanche en procubitus, genou fléchi."
        ),
        hint_en=(
            "Mark at 50 % of the line sacrum_S2 → greater_trochanter_left, in the "
            "middle of the gluteal mass. Orientation: electrodes parallel to that "
            "line, along the oblique fibre direction. Confirm with a prone hip "
            "extension, knee flexed."
        ),
        theme="gait",
        application_fr="Timing du grand fessier en extension de hanche ('amnésie fessière'). Réathlétisation des ischio-jambiers. Lever de chaise et montée d'escalier chez l'aîné.",
        application_en="Gluteus maximus timing in hip extension ('gluteal amnesia'). Hamstring rehabilitation. Chair rise and stair ascent in older adults.",
        body_side="left",
    ),
    Landmark(
        code="EMG_gluteus_maximus_right",
        category="EMG",
        name_fr="Grand fessier droit (EMG)",
        name_en="Right gluteus maximus (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne sacrum_S2 → greater_trochanter_right, au "
            "centre de la masse fessière. Orientation : électrodes parallèles à "
            "cette ligne, dans la direction oblique des fibres. Vérifier par une "
            "extension de hanche en procubitus, genou fléchi."
        ),
        hint_en=(
            "Mark at 50 % of the line sacrum_S2 → greater_trochanter_right, in the "
            "middle of the gluteal mass. Orientation: electrodes parallel to that "
            "line, along the oblique fibre direction. Confirm with a prone hip "
            "extension, knee flexed."
        ),
        theme="gait",
        application_fr="Timing du grand fessier en extension de hanche ('amnésie fessière'). Réathlétisation des ischio-jambiers. Lever de chaise et montée d'escalier chez l'aîné.",
        application_en="Gluteus maximus timing in hip extension ('gluteal amnesia'). Hamstring rehabilitation. Chair rise and stair ascent in older adults.",
        body_side="right",
    ),
    Landmark(
        code="EMG_gluteus_medius_left",
        category="EMG",
        name_fr="Moyen fessier gauche (EMG)",
        name_en="Left gluteus medius (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne iliac_crest_left → greater_trochanter_left, "
            "sur la face latérale de la hanche. Orientation : électrodes parallèles "
            "à cette ligne. Vérifier par une abduction de hanche en décubitus "
            "latéral ; site connu pour une zone d'innervation variable."
        ),
        hint_en=(
            "Mark at 50 % of the line iliac_crest_left → greater_trochanter_left, "
            "on the lateral hip. Orientation: electrodes parallel to that line. "
            "Confirm with a side-lying hip abduction; this site is known for a "
            "variable innervation zone."
        ),
        theme="gait",
        application_fr="Contrôle frontal du bassin à la marche (signe de Trendelenburg). Prévention du valgus dynamique du genou (LCA, syndrome fémoro-patellaire). Coxarthrose.",
        application_en="Frontal plane pelvic control during gait (Trendelenburg sign). Dynamic knee valgus prevention (ACL, patellofemoral syndrome). Hip osteoarthritis.",
        body_side="left",
    ),
    Landmark(
        code="EMG_gluteus_medius_right",
        category="EMG",
        name_fr="Moyen fessier droit (EMG)",
        name_en="Right gluteus medius (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne iliac_crest_right → "
            "greater_trochanter_right, sur la face latérale de la hanche. "
            "Orientation : électrodes parallèles à cette ligne. Vérifier par une "
            "abduction de hanche en décubitus latéral ; zone d'innervation variable."
        ),
        hint_en=(
            "Mark at 50 % of the line iliac_crest_right → "
            "greater_trochanter_right, on the lateral hip. Orientation: electrodes "
            "parallel to that line. Confirm with a side-lying hip abduction; "
            "variable innervation zone."
        ),
        theme="gait",
        application_fr="Contrôle frontal du bassin à la marche (signe de Trendelenburg). Prévention du valgus dynamique du genou (LCA, syndrome fémoro-patellaire). Coxarthrose.",
        application_en="Frontal plane pelvic control during gait (Trendelenburg sign). Dynamic knee valgus prevention (ACL, patellofemoral syndrome). Hip osteoarthritis.",
        body_side="right",
    ),
    Landmark(
        code="EMG_tensor_fasciae_latae_left",
        category="EMG",
        name_fr="Tenseur du fascia lata gauche (EMG)",
        name_en="Left tensor fasciae latae (EMG)",
        hint_fr=(
            "Marquer au 1/6 proximal de la ligne ASIS_left → lateral_knee_left, sur "
            "la face antéro-latérale de la hanche. Orientation : électrodes "
            "parallèles à cette ligne. Le ventre se durcit nettement lors d'une "
            "flexion-abduction-rotation interne de hanche."
        ),
        hint_en=(
            "Mark at the proximal 1/6 of the line ASIS_left → lateral_knee_left, on "
            "the anterolateral hip. Orientation: electrodes parallel to that line. "
            "The belly hardens clearly during hip flexion-abduction with internal "
            "rotation."
        ),
        theme="gait",
        application_fr="Syndrome de la bandelette ilio-tibiale chez le coureur. Compensation TFL/moyen fessier (rapport d'activation). Analyse du pédalage en cyclisme.",
        application_en="Iliotibial band syndrome in runners. TFL/gluteus medius compensation ratio. Pedaling analysis in cycling.",
        body_side="left",
    ),
    Landmark(
        code="EMG_tensor_fasciae_latae_right",
        category="EMG",
        name_fr="Tenseur du fascia lata droit (EMG)",
        name_en="Right tensor fasciae latae (EMG)",
        hint_fr=(
            "Marquer au 1/6 proximal de la ligne ASIS_right → lateral_knee_right, "
            "sur la face antéro-latérale de la hanche. Orientation : électrodes "
            "parallèles à cette ligne. Le ventre se durcit nettement lors d'une "
            "flexion-abduction-rotation interne de hanche."
        ),
        hint_en=(
            "Mark at the proximal 1/6 of the line ASIS_right → lateral_knee_right, "
            "on the anterolateral hip. Orientation: electrodes parallel to that "
            "line. The belly hardens clearly during hip flexion-abduction with "
            "internal rotation."
        ),
        theme="gait",
        application_fr="Syndrome de la bandelette ilio-tibiale chez le coureur. Compensation TFL/moyen fessier (rapport d'activation). Analyse du pédalage en cyclisme.",
        application_en="Iliotibial band syndrome in runners. TFL/gluteus medius compensation ratio. Pedaling analysis in cycling.",
        body_side="right",
    ),
    # ══════════════════════════════════════════════════════════════════════
    # EMG — Épaule et bras (SENIAM)
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_deltoid_anterior_left",
        category="EMG",
        name_fr="Deltoïde antérieur gauche (EMG)",
        name_en="Left anterior deltoid (EMG)",
        hint_fr=(
            "Marquer à un travers de doigt en avant et en dessous de acromion_left, "
            "sur le faisceau claviculaire. Orientation : électrodes dirigées de "
            "l'acromion vers le pouce, bras le long du corps. Vérifier par une "
            "flexion d'épaule contre résistance."
        ),
        hint_en=(
            "Mark one finger width anterior and distal to acromion_left, over the "
            "clavicular head. Orientation: electrodes aligned from the acromion "
            "toward the thumb, arm at the side. Confirm with a resisted shoulder "
            "flexion."
        ),
        theme="shoulder",
        application_fr="Charge EMG en travail bras élevé (ergonomie). Ratio deltoïde antérieur/sus-épineux dans le conflit sous-acromial. Contrôle moteur du lancer.",
        application_en="EMG load during overhead work (ergonomics). Anterior deltoid/supraspinatus ratio in subacromial impingement. Throwing motor control.",
        body_side="left",
    ),
    Landmark(
        code="EMG_deltoid_anterior_right",
        category="EMG",
        name_fr="Deltoïde antérieur droit (EMG)",
        name_en="Right anterior deltoid (EMG)",
        hint_fr=(
            "Marquer à un travers de doigt en avant et en dessous de acromion_right, "
            "sur le faisceau claviculaire. Orientation : électrodes dirigées de "
            "l'acromion vers le pouce, bras le long du corps. Vérifier par une "
            "flexion d'épaule contre résistance."
        ),
        hint_en=(
            "Mark one finger width anterior and distal to acromion_right, over the "
            "clavicular head. Orientation: electrodes aligned from the acromion "
            "toward the thumb, arm at the side. Confirm with a resisted shoulder "
            "flexion."
        ),
        theme="shoulder",
        application_fr="Charge EMG en travail bras élevé (ergonomie). Ratio deltoïde antérieur/sus-épineux dans le conflit sous-acromial. Contrôle moteur du lancer.",
        application_en="EMG load during overhead work (ergonomics). Anterior deltoid/supraspinatus ratio in subacromial impingement. Throwing motor control.",
        body_side="right",
    ),
    Landmark(
        code="EMG_deltoid_medius_left",
        category="EMG",
        name_fr="Deltoïde moyen gauche (EMG)",
        name_en="Left middle deltoid (EMG)",
        hint_fr=(
            "Marquer sur la courbure la plus saillante du moignon de l'épaule, "
            "environ 3 travers de doigt sous acromion_left. Orientation : "
            "électrodes alignées sur la ligne acromion_left → "
            "lateral_epicondyle_left. Vérifier par une abduction à 90°."
        ),
        hint_en=(
            "Mark on the greatest curvature of the shoulder cap, about three finger "
            "widths below acromion_left. Orientation: electrodes aligned on the line "
            "acromion_left → lateral_epicondyle_left. Confirm with a 90° abduction."
        ),
        theme="shoulder",
        application_fr="Activité lors de l'abduction et du travail bras tendu. Évaluation de la fatigue statique en poste de travail. Rééducation post-luxation gléno-humérale.",
        application_en="Activity during abduction and sustained arm elevation. Static fatigue assessment at workstations. Rehabilitation after glenohumeral dislocation.",
        body_side="left",
    ),
    Landmark(
        code="EMG_deltoid_medius_right",
        category="EMG",
        name_fr="Deltoïde moyen droit (EMG)",
        name_en="Right middle deltoid (EMG)",
        hint_fr=(
            "Marquer sur la courbure la plus saillante du moignon de l'épaule, "
            "environ 3 travers de doigt sous acromion_right. Orientation : "
            "électrodes alignées sur la ligne acromion_right → "
            "lateral_epicondyle_right. Vérifier par une abduction à 90°."
        ),
        hint_en=(
            "Mark on the greatest curvature of the shoulder cap, about three finger "
            "widths below acromion_right. Orientation: electrodes aligned on the "
            "line acromion_right → lateral_epicondyle_right. Confirm with a 90° "
            "abduction."
        ),
        theme="shoulder",
        application_fr="Activité lors de l'abduction et du travail bras tendu. Évaluation de la fatigue statique en poste de travail. Rééducation post-luxation gléno-humérale.",
        application_en="Activity during abduction and sustained arm elevation. Static fatigue assessment at workstations. Rehabilitation after glenohumeral dislocation.",
        body_side="right",
    ),
    Landmark(
        code="EMG_deltoid_posterior_left",
        category="EMG",
        name_fr="Deltoïde postérieur gauche (EMG)",
        name_en="Left posterior deltoid (EMG)",
        hint_fr=(
            "Marquer à environ 2 travers de doigt en arrière et en dessous de "
            "acromial_angle_left. Orientation : électrodes dirigées de l'angle "
            "acromial vers l'auriculaire. Vérifier par une extension horizontale "
            "d'épaule contre résistance."
        ),
        hint_en=(
            "Mark about two finger widths behind and below acromial_angle_left. "
            "Orientation: electrodes aligned from the acromial angle toward the "
            "little finger. Confirm with a resisted horizontal shoulder extension."
        ),
        theme="shoulder",
        application_fr="Dyskinésie scapulaire et contrôle postérieur de l'épaule. Tirage/rowing en réathlétisation. Posture cyphotique prolongée (travail de bureau).",
        application_en="Scapular dyskinesis and posterior shoulder control. Rowing/pull-down in rehabilitation. Prolonged kyphotic posture (desk work).",
        body_side="left",
    ),
    Landmark(
        code="EMG_deltoid_posterior_right",
        category="EMG",
        name_fr="Deltoïde postérieur droit (EMG)",
        name_en="Right posterior deltoid (EMG)",
        hint_fr=(
            "Marquer à environ 2 travers de doigt en arrière et en dessous de "
            "acromial_angle_right. Orientation : électrodes dirigées de l'angle "
            "acromial vers l'auriculaire. Vérifier par une extension horizontale "
            "d'épaule contre résistance."
        ),
        hint_en=(
            "Mark about two finger widths behind and below acromial_angle_right. "
            "Orientation: electrodes aligned from the acromial angle toward the "
            "little finger. Confirm with a resisted horizontal shoulder extension."
        ),
        theme="shoulder",
        application_fr="Dyskinésie scapulaire et contrôle postérieur de l'épaule. Tirage/rowing en réathlétisation. Posture cyphotique prolongée (travail de bureau).",
        application_en="Scapular dyskinesis and posterior shoulder control. Rowing/pull-down in rehabilitation. Prolonged kyphotic posture (desk work).",
        body_side="right",
    ),
    Landmark(
        code="EMG_biceps_brachii_left",
        category="EMG",
        name_fr="Biceps brachial gauche (EMG)",
        name_en="Left biceps brachii (EMG)",
        hint_fr=(
            "Marquer à 1/3 de la distance entre le pli du coude (fosse cubitale) et "
            "acromion_left, sur la ligne médiane de la face antérieure du bras. "
            "Orientation : électrodes parallèles à l'axe du bras. Coude fléchi à 90° "
            "en supination pour faire ressortir le ventre musculaire."
        ),
        hint_en=(
            "Mark at 1/3 of the distance from the cubital fossa to acromion_left, "
            "on the midline of the anterior arm. Orientation: electrodes parallel to "
            "the arm axis. Flex the elbow to 90° in supination to make the belly "
            "stand out."
        ),
        theme="upper_limb",
        application_fr="Biofeedback de flexion du coude post-AVC. Charge du membre supérieur en manutention. Co-contraction biceps/triceps en évaluation de la spasticité.",
        application_en="Elbow flexion biofeedback post-stroke. Upper limb load during manual handling. Biceps/triceps co-contraction in spasticity assessment.",
        body_side="left",
    ),
    Landmark(
        code="EMG_biceps_brachii_right",
        category="EMG",
        name_fr="Biceps brachial droit (EMG)",
        name_en="Right biceps brachii (EMG)",
        hint_fr=(
            "Marquer à 1/3 de la distance entre le pli du coude (fosse cubitale) et "
            "acromion_right, sur la ligne médiane de la face antérieure du bras. "
            "Orientation : électrodes parallèles à l'axe du bras. Coude fléchi à 90° "
            "en supination pour faire ressortir le ventre musculaire."
        ),
        hint_en=(
            "Mark at 1/3 of the distance from the cubital fossa to acromion_right, "
            "on the midline of the anterior arm. Orientation: electrodes parallel to "
            "the arm axis. Flex the elbow to 90° in supination to make the belly "
            "stand out."
        ),
        theme="upper_limb",
        application_fr="Biofeedback de flexion du coude post-AVC. Charge du membre supérieur en manutention. Co-contraction biceps/triceps en évaluation de la spasticité.",
        application_en="Elbow flexion biofeedback post-stroke. Upper limb load during manual handling. Biceps/triceps co-contraction in spasticity assessment.",
        body_side="right",
    ),
    Landmark(
        code="EMG_triceps_brachii_long_left",
        category="EMG",
        name_fr="Triceps brachial, chef long gauche (EMG)",
        name_en="Left triceps brachii, long head (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne acromion_left → olecranon_left, puis "
            "décaler d'environ 2 travers de doigt vers le dedans, sur la face "
            "postérieure du bras. Orientation : électrodes parallèles à l'axe du "
            "bras. (Pour le chef latéral : même niveau, 2 cm en dehors de la ligne.)"
        ),
        hint_en=(
            "Mark at 50 % of the line acromion_left → olecranon_left, then shift "
            "about two finger widths medially, on the posterior arm. Orientation: "
            "electrodes parallel to the arm axis. (For the lateral head: same level, "
            "2 cm lateral to the line.)"
        ),
        theme="upper_limb",
        application_fr="Propulsion en fauteuil roulant manuel. Rééducation post-fracture de l'olécrâne. Analyse du développé couché et du push-up.",
        application_en="Manual wheelchair propulsion. Post-olecranon fracture rehabilitation. Bench press and push-up analysis.",
        body_side="left",
    ),
    Landmark(
        code="EMG_triceps_brachii_long_right",
        category="EMG",
        name_fr="Triceps brachial, chef long droit (EMG)",
        name_en="Right triceps brachii, long head (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne acromion_right → olecranon_right, puis "
            "décaler d'environ 2 travers de doigt vers le dedans, sur la face "
            "postérieure du bras. Orientation : électrodes parallèles à l'axe du "
            "bras. (Pour le chef latéral : même niveau, 2 cm en dehors de la ligne.)"
        ),
        hint_en=(
            "Mark at 50 % of the line acromion_right → olecranon_right, then shift "
            "about two finger widths medially, on the posterior arm. Orientation: "
            "electrodes parallel to the arm axis. (For the lateral head: same level, "
            "2 cm lateral to the line.)"
        ),
        theme="upper_limb",
        application_fr="Propulsion en fauteuil roulant manuel. Rééducation post-fracture de l'olécrâne. Analyse du développé couché et du push-up.",
        application_en="Manual wheelchair propulsion. Post-olecranon fracture rehabilitation. Bench press and push-up analysis.",
        body_side="right",
    ),
    Landmark(
        code="EMG_wrist_extensors_left",
        category="EMG",
        name_fr="Extenseurs du poignet gauche (EMG)",
        name_en="Left wrist extensors (EMG)",
        hint_fr=(
            "Masse commune des extenseurs radiaux du carpe : marquer sur le "
            "renflement musculaire situé 4 à 6 cm distalement à "
            "lateral_epicondyle_left, avant-bras en pronation. Orientation : "
            "électrodes parallèles à l'axe de l'avant-bras. Vérifier par une "
            "extension du poignet contre résistance."
        ),
        hint_en=(
            "Common extensor mass (extensor carpi radialis): mark on the muscle "
            "bulge 4 to 6 cm distal to lateral_epicondyle_left, forearm pronated. "
            "Orientation: electrodes parallel to the forearm axis. Confirm with a "
            "resisted wrist extension."
        ),
        theme="upper_limb",
        application_fr="Épicondylalgies (médiale : fléchisseurs, latérale : extenseurs). Ergonomie du poste informatique. Analyse de la préhension en réhabilitation.",
        application_en="Epicondylalgia (medial: flexors, lateral: extensors). Office workstation ergonomics. Grip analysis in rehabilitation.",
        body_side="left",
    ),
    Landmark(
        code="EMG_wrist_extensors_right",
        category="EMG",
        name_fr="Extenseurs du poignet droit (EMG)",
        name_en="Right wrist extensors (EMG)",
        hint_fr=(
            "Masse commune des extenseurs radiaux du carpe : marquer sur le "
            "renflement musculaire situé 4 à 6 cm distalement à "
            "lateral_epicondyle_right, avant-bras en pronation. Orientation : "
            "électrodes parallèles à l'axe de l'avant-bras. Vérifier par une "
            "extension du poignet contre résistance."
        ),
        hint_en=(
            "Common extensor mass (extensor carpi radialis): mark on the muscle "
            "bulge 4 to 6 cm distal to lateral_epicondyle_right, forearm pronated. "
            "Orientation: electrodes parallel to the forearm axis. Confirm with a "
            "resisted wrist extension."
        ),
        theme="upper_limb",
        application_fr="Épicondylalgies (médiale : fléchisseurs, latérale : extenseurs). Ergonomie du poste informatique. Analyse de la préhension en réhabilitation.",
        application_en="Epicondylalgia (medial: flexors, lateral: extensors). Office workstation ergonomics. Grip analysis in rehabilitation.",
        body_side="right",
    ),
    Landmark(
        code="EMG_wrist_flexors_left",
        category="EMG",
        name_fr="Fléchisseurs du poignet gauche (EMG)",
        name_en="Left wrist flexors (EMG)",
        hint_fr=(
            "Masse commune des fléchisseurs (fléchisseur radial du carpe) : marquer "
            "à 1/3 de la ligne medial_epicondyle_left → radial_styloid_left, sur la "
            "face antérieure de l'avant-bras en supination. Orientation : électrodes "
            "parallèles à cette ligne."
        ),
        hint_en=(
            "Common flexor mass (flexor carpi radialis): mark at 1/3 of the line "
            "medial_epicondyle_left → radial_styloid_left, on the anterior forearm "
            "in supination. Orientation: electrodes parallel to that line."
        ),
        theme="upper_limb",
        application_fr="Épicondylalgies (médiale : fléchisseurs, latérale : extenseurs). Ergonomie du poste informatique. Analyse de la préhension en réhabilitation.",
        application_en="Epicondylalgia (medial: flexors, lateral: extensors). Office workstation ergonomics. Grip analysis in rehabilitation.",
        body_side="left",
    ),
    Landmark(
        code="EMG_wrist_flexors_right",
        category="EMG",
        name_fr="Fléchisseurs du poignet droit (EMG)",
        name_en="Right wrist flexors (EMG)",
        hint_fr=(
            "Masse commune des fléchisseurs (fléchisseur radial du carpe) : marquer "
            "à 1/3 de la ligne medial_epicondyle_right → radial_styloid_right, sur "
            "la face antérieure de l'avant-bras en supination. Orientation : "
            "électrodes parallèles à cette ligne."
        ),
        hint_en=(
            "Common flexor mass (flexor carpi radialis): mark at 1/3 of the line "
            "medial_epicondyle_right → radial_styloid_right, on the anterior forearm "
            "in supination. Orientation: electrodes parallel to that line."
        ),
        theme="upper_limb",
        application_fr="Épicondylalgies (médiale : fléchisseurs, latérale : extenseurs). Ergonomie du poste informatique. Analyse de la préhension en réhabilitation.",
        application_en="Epicondylalgia (medial: flexors, lateral: extensors). Office workstation ergonomics. Grip analysis in rehabilitation.",
        body_side="right",
    ),
    # ══════════════════════════════════════════════════════════════════════
    # EMG — Tronc (SENIAM)
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_trapezius_descendens_left",
        category="EMG",
        name_fr="Trapèze supérieur gauche (EMG)",
        name_en="Left upper trapezius (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne C7_spinous → acromion_left, sur le bord "
            "supérieur de l'épaule. Orientation : électrodes alignées sur cette "
            "ligne. Vérifier par une élévation d'épaule contre résistance, bras le "
            "long du corps."
        ),
        hint_en=(
            "Mark at 50 % of the line C7_spinous → acromion_left, on the upper "
            "border of the shoulder. Orientation: electrodes aligned with that line. "
            "Confirm with a resisted shoulder shrug, arms at the side."
        ),
        theme="shoulder",
        application_fr="TMS du cou et de l'épaule au travail (amplitude, gap time, APD). Céphalées cervicogènes. Ratio trapèze supérieur/inférieur en dyskinésie scapulaire.",
        application_en="Neck and shoulder MSDs in occupational settings (amplitude, gap time, APD). Cervicogenic headaches. Upper/lower trapezius ratio in scapular dyskinesis.",
        body_side="left",
    ),
    Landmark(
        code="EMG_trapezius_descendens_right",
        category="EMG",
        name_fr="Trapèze supérieur droit (EMG)",
        name_en="Right upper trapezius (EMG)",
        hint_fr=(
            "Marquer à 50 % de la ligne C7_spinous → acromion_right, sur le bord "
            "supérieur de l'épaule. Orientation : électrodes alignées sur cette "
            "ligne. Vérifier par une élévation d'épaule contre résistance, bras le "
            "long du corps."
        ),
        hint_en=(
            "Mark at 50 % of the line C7_spinous → acromion_right, on the upper "
            "border of the shoulder. Orientation: electrodes aligned with that line. "
            "Confirm with a resisted shoulder shrug, arms at the side."
        ),
        theme="shoulder",
        application_fr="TMS du cou et de l'épaule au travail (amplitude, gap time, APD). Céphalées cervicogènes. Ratio trapèze supérieur/inférieur en dyskinésie scapulaire.",
        application_en="Neck and shoulder MSDs in occupational settings (amplitude, gap time, APD). Cervicogenic headaches. Upper/lower trapezius ratio in scapular dyskinesis.",
        body_side="right",
    ),
    Landmark(
        code="EMG_trapezius_transversalis_left",
        category="EMG",
        name_fr="Trapèze moyen gauche (EMG)",
        name_en="Left middle trapezius (EMG)",
        hint_fr=(
            "Marquer à mi-distance entre scapula_trigonum_left et l'épineuse de T3, "
            "sur la ligne horizontale reliant les deux. Orientation : électrodes "
            "horizontales, dirigées vers le rachis. Vérifier par une rétraction "
            "(adduction) des scapulas."
        ),
        hint_en=(
            "Mark midway between scapula_trigonum_left and the T3 spinous process, "
            "on the horizontal line joining them. Orientation: electrodes horizontal, "
            "pointing toward the spine. Confirm with scapular retraction."
        ),
        theme="shoulder",
        application_fr="Rétraction scapulaire et stabilisation de la scapula. Posture assise prolongée. Sport de lancer (décélération). Repère mobile : imposer une posture bras pendants.",
        application_en="Scapular retraction and stabilization. Prolonged sitting posture. Throwing deceleration. Mobile landmark: standardize posture with arms hanging.",
        body_side="left",
    ),
    Landmark(
        code="EMG_trapezius_transversalis_right",
        category="EMG",
        name_fr="Trapèze moyen droit (EMG)",
        name_en="Right middle trapezius (EMG)",
        hint_fr=(
            "Marquer à mi-distance entre scapula_trigonum_right et l'épineuse de T3, "
            "sur la ligne horizontale reliant les deux. Orientation : électrodes "
            "horizontales, dirigées vers le rachis. Vérifier par une rétraction "
            "(adduction) des scapulas."
        ),
        hint_en=(
            "Mark midway between scapula_trigonum_right and the T3 spinous process, "
            "on the horizontal line joining them. Orientation: electrodes horizontal, "
            "pointing toward the spine. Confirm with scapular retraction."
        ),
        theme="shoulder",
        application_fr="Rétraction scapulaire et stabilisation de la scapula. Posture assise prolongée. Sport de lancer (décélération). Repère mobile : imposer une posture bras pendants.",
        application_en="Scapular retraction and stabilization. Prolonged sitting posture. Throwing deceleration. Mobile landmark: standardize posture with arms hanging.",
        body_side="right",
    ),
    Landmark(
        code="EMG_trapezius_ascendens_left",
        category="EMG",
        name_fr="Trapèze inférieur gauche (EMG)",
        name_en="Left lower trapezius (EMG)",
        hint_fr=(
            "Marquer à 2/3 de la ligne scapula_trigonum_left → T8_spinous, dans "
            "l'aire triangulaire sous l'épine de la scapula. Orientation : électrodes "
            "obliques, alignées sur cette ligne (fibres ascendantes). Vérifier par "
            "une élévation du bras au-dessus de la tête."
        ),
        hint_en=(
            "Mark at 2/3 of the line scapula_trigonum_left → T8_spinous, in the "
            "triangular area below the scapular spine. Orientation: electrodes "
            "oblique, aligned with that line (ascending fibres). Confirm by raising "
            "the arm overhead."
        ),
        theme="shoulder",
        application_fr="Ratio TI/TS : marqueur clé du conflit sous-acromial (TI déficitaire). Rythme scapulo-huméral et rotation scapulaire. Natation (phase de récupération).",
        application_en="LT/UT ratio: key marker of subacromial impingement (LT deficit). Scapulohumeral rhythm and scapular upward rotation. Swimming (recovery phase).",
        body_side="left",
    ),
    Landmark(
        code="EMG_trapezius_ascendens_right",
        category="EMG",
        name_fr="Trapèze inférieur droit (EMG)",
        name_en="Right lower trapezius (EMG)",
        hint_fr=(
            "Marquer à 2/3 de la ligne scapula_trigonum_right → T8_spinous, dans "
            "l'aire triangulaire sous l'épine de la scapula. Orientation : électrodes "
            "obliques, alignées sur cette ligne (fibres ascendantes). Vérifier par "
            "une élévation du bras au-dessus de la tête."
        ),
        hint_en=(
            "Mark at 2/3 of the line scapula_trigonum_right → T8_spinous, in the "
            "triangular area below the scapular spine. Orientation: electrodes "
            "oblique, aligned with that line (ascending fibres). Confirm by raising "
            "the arm overhead."
        ),
        theme="shoulder",
        application_fr="Ratio TI/TS : marqueur clé du conflit sous-acromial (TI déficitaire). Rythme scapulo-huméral et rotation scapulaire. Natation (phase de récupération).",
        application_en="LT/UT ratio: key marker of subacromial impingement (LT deficit). Scapulohumeral rhythm and scapular upward rotation. Swimming (recovery phase).",
        body_side="right",
    ),
    Landmark(
        code="EMG_erector_spinae_longissimus_left",
        category="EMG",
        name_fr="Érecteur du rachis - longissimus gauche (EMG)",
        name_en="Left erector spinae - longissimus (EMG)",
        hint_fr=(
            "Marquer à environ 2 travers de doigt (2-3 cm) à gauche de l'épineuse "
            "de L1 (L1_spinous), sur la masse musculaire paravertébrale. "
            "Orientation : électrodes verticales, parallèles au rachis. Variante "
            "lombaire basse : 2-3 cm latéralement à L5_spinous."
        ),
        hint_en=(
            "Mark about two finger widths (2-3 cm) to the left of the L1 spinous "
            "process (L1_spinous), over the paravertebral muscle mass. Orientation: "
            "electrodes vertical, parallel to the spine. Low-lumbar variant: 2-3 cm "
            "lateral to L5_spinous."
        ),
        theme="core",
        application_fr="Phénomène de flexion-relaxation en lombalgie chronique. Charge vertébrale en levage (protocole NIOSH). Test d'endurance de Biering-Sørensen.",
        application_en="Flexion-relaxation phenomenon in chronic low back pain. Spinal load during lifting (NIOSH protocol). Biering-Sørensen endurance test.",
        body_side="left",
    ),
    Landmark(
        code="EMG_erector_spinae_longissimus_right",
        category="EMG",
        name_fr="Érecteur du rachis - longissimus droit (EMG)",
        name_en="Right erector spinae - longissimus (EMG)",
        hint_fr=(
            "Marquer à environ 2 travers de doigt (2-3 cm) à droite de l'épineuse "
            "de L1 (L1_spinous), sur la masse musculaire paravertébrale. "
            "Orientation : électrodes verticales, parallèles au rachis. Variante "
            "lombaire basse : 2-3 cm latéralement à L5_spinous."
        ),
        hint_en=(
            "Mark about two finger widths (2-3 cm) to the right of the L1 spinous "
            "process (L1_spinous), over the paravertebral muscle mass. Orientation: "
            "electrodes vertical, parallel to the spine. Low-lumbar variant: 2-3 cm "
            "lateral to L5_spinous."
        ),
        theme="core",
        application_fr="Phénomène de flexion-relaxation en lombalgie chronique. Charge vertébrale en levage (protocole NIOSH). Test d'endurance de Biering-Sørensen.",
        application_en="Flexion-relaxation phenomenon in chronic low back pain. Spinal load during lifting (NIOSH protocol). Biering-Sørensen endurance test.",
        body_side="right",
    ),
    Landmark(
        code="EMG_obliquus_externus_left",
        category="EMG",
        name_fr="Oblique externe gauche (EMG)",
        name_en="Left obliquus externus abdominis (EMG)",
        hint_fr=(
            "Marquer sur la ligne oblique joignant la dernière côte à ASIS_left, à "
            "mi-chemin — soit environ 15 cm latéralement à l'ombilic, au-dessus de "
            "l'EIAS. Orientation : électrodes obliques, dirigées vers le bas et vers "
            "l'avant (en direction de la symphyse pubienne), dans l'axe des fibres."
        ),
        hint_en=(
            "Mark on the oblique line joining the lowest rib to ASIS_left, about "
            "halfway — roughly 15 cm lateral to the umbilicus, above the ASIS. "
            "Orientation: electrodes oblique, pointing downward and forward (toward "
            "the pubic symphysis), along the fibre direction."
        ),
        theme="core",
        application_fr="Rotation du tronc (golf, lancer de balle). Stabilisation lombo-pelvienne en manutention asymétrique. Cross-talk avec l'oblique interne (point de discussion sur les limites de l'EMG de surface).",
        application_en="Trunk rotation (golf, throwing). Lumbopelvic stabilization in asymmetric lifting. Cross-talk with internal oblique (teaching point on sEMG limitations).",
        body_side="left",
    ),
    Landmark(
        code="EMG_obliquus_externus_right",
        category="EMG",
        name_fr="Oblique externe droit (EMG)",
        name_en="Right obliquus externus abdominis (EMG)",
        hint_fr=(
            "Marquer sur la ligne oblique joignant la dernière côte à ASIS_right, à "
            "mi-chemin — soit environ 15 cm latéralement à l'ombilic, au-dessus de "
            "l'EIAS. Orientation : électrodes obliques, dirigées vers le bas et vers "
            "l'avant (en direction de la symphyse pubienne), dans l'axe des fibres."
        ),
        hint_en=(
            "Mark on the oblique line joining the lowest rib to ASIS_right, about "
            "halfway — roughly 15 cm lateral to the umbilicus, above the ASIS. "
            "Orientation: electrodes oblique, pointing downward and forward (toward "
            "the pubic symphysis), along the fibre direction."
        ),
        theme="core",
        application_fr="Rotation du tronc (golf, lancer de balle). Stabilisation lombo-pelvienne en manutention asymétrique. Cross-talk avec l'oblique interne (point de discussion sur les limites de l'EMG de surface).",
        application_en="Trunk rotation (golf, throwing). Lumbopelvic stabilization in asymmetric lifting. Cross-talk with internal oblique (teaching point on sEMG limitations).",
        body_side="right",
    ),
    Landmark(
        code="EMG_rectus_abdominis_left",
        category="EMG",
        name_fr="Grand droit de l'abdomen gauche (EMG)",
        name_en="Left rectus abdominis (EMG)",
        hint_fr=(
            "Marquer à 2-3 cm à gauche de la ligne blanche, au niveau de l'ombilic "
            "(portion sus-ombilicale). Orientation : électrodes verticales, "
            "parallèles à l'axe du corps, entre deux intersections tendineuses. "
            "Variante sous-ombilicale : même décalage latéral, 3-5 cm sous l'ombilic."
        ),
        hint_en=(
            "Mark 2-3 cm to the left of the linea alba, at umbilicus level (upper "
            "portion). Orientation: electrodes vertical, parallel to the body axis, "
            "between two tendinous intersections. Lower variant: same lateral offset, "
            "3-5 cm below the umbilicus."
        ),
        theme="core",
        application_fr="Gainage et core stability. Lombalgie chronique (ratio fléchisseurs/extenseurs). Retour post-partum et diastasis des grands droits.",
        application_en="Core stability and trunk bracing. Chronic low back pain (flexor/extensor ratio). Postpartum return and rectus abdominis diastasis.",
        body_side="left",
    ),
    Landmark(
        code="EMG_rectus_abdominis_right",
        category="EMG",
        name_fr="Grand droit de l'abdomen droit (EMG)",
        name_en="Right rectus abdominis (EMG)",
        hint_fr=(
            "Marquer à 2-3 cm à droite de la ligne blanche, au niveau de l'ombilic "
            "(portion sus-ombilicale). Orientation : électrodes verticales, "
            "parallèles à l'axe du corps, entre deux intersections tendineuses. "
            "Variante sous-ombilicale : même décalage latéral, 3-5 cm sous l'ombilic."
        ),
        hint_en=(
            "Mark 2-3 cm to the right of the linea alba, at umbilicus level (upper "
            "portion). Orientation: electrodes vertical, parallel to the body axis, "
            "between two tendinous intersections. Lower variant: same lateral offset, "
            "3-5 cm below the umbilicus."
        ),
        theme="core",
        application_fr="Gainage et core stability. Lombalgie chronique (ratio fléchisseurs/extenseurs). Retour post-partum et diastasis des grands droits.",
        application_en="Core stability and trunk bracing. Chronic low back pain (flexor/extensor ratio). Postpartum return and rectus abdominis diastasis.",
        body_side="right",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # SKINFOLD — Plis cutanés ISAK (côté droit par convention)
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="SKINFOLD_triceps_right",
        category="SKINFOLD",
        name_fr="Pli tricipital (droit)",
        name_en="Triceps skinfold (right)",
        hint_fr=(
            "Face postérieure du bras droit, sur la ligne médiane, au niveau du "
            "point mi-acromiale-radiale (anthro_mid_acromiale_radiale_right). Bras "
            "pendant relâché, coude en extension. Le pli est vertical, parallèle à "
            "l'axe du bras."
        ),
        hint_en=(
            "Posterior surface of the right arm, on the midline, at the level of the "
            "mid-acromiale-radiale point (anthro_mid_acromiale_radiale_right). Arm "
            "hanging relaxed, elbow extended. The fold is vertical, parallel to the "
            "arm axis."
        ),
        body_side="right",
    ),
    Landmark(
        code="SKINFOLD_subscapular_right",
        category="SKINFOLD",
        name_fr="Pli sous-scapulaire (droit)",
        name_en="Subscapular skinfold (right)",
        hint_fr=(
            "À 2 cm de scapula_inferior_angle_right, le long d'une ligne descendant "
            "latéralement à 45° vers le bas et le dehors. Épaule relâchée, bras le "
            "long du corps. Le pli suit la même obliquité de 45°."
        ),
        hint_en=(
            "2 cm from scapula_inferior_angle_right, along a line running downward "
            "and laterally at 45°. Shoulder relaxed, arm at the side. The fold "
            "follows the same 45° obliquity."
        ),
        body_side="right",
    ),
    Landmark(
        code="SKINFOLD_biceps_right",
        category="SKINFOLD",
        name_fr="Pli bicipital (droit)",
        name_en="Biceps skinfold (right)",
        hint_fr=(
            "Face antérieure du bras droit, sur la ligne médiane, au même niveau "
            "horizontal que le pli tricipital (mi-acromiale-radiale). Bras pendant, "
            "paume tournée vers l'avant. Le pli est vertical, parallèle à l'axe du bras."
        ),
        hint_en=(
            "Anterior surface of the right arm, on the midline, at the same "
            "horizontal level as the triceps skinfold (mid-acromiale-radiale). Arm "
            "hanging with the palm facing forward. The fold is vertical, parallel to "
            "the arm axis."
        ),
        body_side="right",
    ),
    Landmark(
        code="SKINFOLD_iliac_crest_right",
        category="SKINFOLD",
        name_fr="Pli de la crête iliaque (droit)",
        name_en="Iliac crest skinfold (right)",
        hint_fr=(
            "Immédiatement au-dessus de iliac_crest_right (point le plus latéral de "
            "la crête), sur la ligne axillaire moyenne. Bras droit croisé sur la "
            "poitrine. Le pli est presque horizontal, légèrement descendant vers "
            "l'avant."
        ),
        hint_en=(
            "Immediately above iliac_crest_right (most lateral point of the crest), "
            "on the mid-axillary line. The right arm is crossed over the chest. The "
            "fold runs almost horizontally, angled slightly downward-forward."
        ),
        body_side="right",
    ),
    Landmark(
        code="SKINFOLD_supraspinale_right",
        category="SKINFOLD",
        name_fr="Pli supra-iliaque (supraspinale, droit)",
        name_en="Supraspinale skinfold (right)",
        hint_fr=(
            "Intersection de la ligne allant de ASIS_right vers le bord axillaire "
            "antérieur et de la ligne horizontale passant par iliac_crest_right : "
            "soit environ 5 à 7 cm au-dessus de l'EIAS. Le pli est oblique, dirigé "
            "vers le bas et vers le dedans (≈ 45°)."
        ),
        hint_en=(
            "Intersection of the line from ASIS_right toward the anterior axillary "
            "border with the horizontal line through iliac_crest_right — roughly 5 "
            "to 7 cm above the ASIS. The fold is oblique, running downward and "
            "medially (≈ 45°)."
        ),
        body_side="right",
    ),
    Landmark(
        code="SKINFOLD_abdominal_right",
        category="SKINFOLD",
        name_fr="Pli abdominal (droit)",
        name_en="Abdominal skinfold (right)",
        hint_fr=(
            "À 5 cm horizontalement à droite du centre de l'ombilic "
            "(anthro_omphalion). Le pli est vertical, parallèle à l'axe du corps. "
            "Ne jamais pincer dans l'ombilic lui-même ; demander au sujet de "
            "respirer normalement et de relâcher la paroi abdominale."
        ),
        hint_en=(
            "5 cm horizontally to the right of the midpoint of the umbilicus "
            "(anthro_omphalion). The fold is vertical, parallel to the body axis. "
            "Never pinch in the umbilicus itself; ask the subject to breathe "
            "normally and relax the abdominal wall."
        ),
        body_side="right",
    ),
    Landmark(
        code="SKINFOLD_front_thigh_right",
        category="SKINFOLD",
        name_fr="Pli de la cuisse antérieure (droit)",
        name_en="Front thigh skinfold (right)",
        hint_fr=(
            "Mi-distance entre le pli inguinal et le bord supérieur de la patella "
            "(patella_superior_right), sur la ligne médiane de la face antérieure de "
            "la cuisse. Sujet assis, genou fléchi à 90°, jambe relâchée. Le pli est "
            "vertical, parallèle à l'axe de la cuisse."
        ),
        hint_en=(
            "Midway between the inguinal fold and the superior border of the patella "
            "(patella_superior_right), on the midline of the anterior thigh. Subject "
            "seated, knee flexed to 90°, leg relaxed. The fold is vertical, parallel "
            "to the thigh axis."
        ),
        body_side="right",
    ),
    Landmark(
        code="SKINFOLD_medial_calf_right",
        category="SKINFOLD",
        name_fr="Pli du mollet médial (droit)",
        name_en="Medial calf skinfold (right)",
        hint_fr=(
            "Face médiale du mollet droit, exactement au niveau de la circonférence "
            "maximale (anthro_calf_girth_right). Sujet assis genou à 90° ou debout "
            "pied posé sur un tabouret. Le pli est vertical, parallèle à l'axe de la "
            "jambe."
        ),
        hint_en=(
            "Medial aspect of the right calf, exactly at the level of maximum girth "
            "(anthro_calf_girth_right). Subject seated with the knee at 90°, or "
            "standing with the foot on a step. The fold is vertical, parallel to the "
            "leg axis."
        ),
        body_side="right",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # ANTHRO — Circonférences et diamètres ISAK (côté droit)
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="anthro_mid_acromiale_radiale_right",
        category="ANTHRO",
        name_fr="Point mi-acromiale-radiale (droit)",
        name_en="Mid-acromiale-radiale point (right)",
        hint_fr=(
            "Milieu exact de la distance entre acromion_right et radiale_right, "
            "mesurée bras pendant le long du corps. Marquer le point sur la face "
            "latérale du bras, puis projeter le niveau en avant et en arrière : ce "
            "niveau définit les plis bicipital et tricipital ainsi que la "
            "circonférence du bras relâché."
        ),
        hint_en=(
            "Exact midpoint between acromion_right and radiale_right, measured with "
            "the arm hanging at the side. Mark the point on the lateral arm, then "
            "project the level anteriorly and posteriorly: this level defines the "
            "biceps and triceps skinfolds as well as the relaxed arm girth."
        ),
        body_side="right",
    ),
    Landmark(
        code="anthro_omphalion",
        category="ANTHRO",
        name_fr="Omphalion (centre de l'ombilic)",
        name_en="Omphalion (midpoint of the navel)",
        hint_fr=(
            "Point médian de l'ombilic, sujet debout et paroi abdominale relâchée. "
            "Il sert d'origine horizontale au pli abdominal (5 cm à droite) et de "
            "référence usuelle pour la circonférence de taille dans certains "
            "protocoles."
        ),
        hint_en=(
            "Midpoint of the navel, subject standing with the abdominal wall "
            "relaxed. It is the horizontal origin for the abdominal skinfold (5 cm "
            "to the right) and a common reference for waist girth in some protocols."
        ),
    ),
    Landmark(
        code="anthro_arm_flexed_girth_right",
        category="ANTHRO",
        name_fr="Circonférence du bras fléchi et contracté (droit)",
        name_en="Flexed and tensed arm girth (right)",
        hint_fr=(
            "Marquer le point le plus saillant du biceps contracté : épaule en "
            "abduction/flexion à ~45°, coude fléchi à 90°, poing serré en "
            "supination, contraction maximale. La circonférence est prise "
            "perpendiculairement à l'axe du bras, au niveau du pic marqué."
        ),
        hint_en=(
            "Mark the peak of the contracted biceps: shoulder abducted/flexed to "
            "about 45°, elbow flexed to 90°, fist clenched in supination, maximal "
            "contraction. The girth is taken perpendicular to the arm axis at the "
            "marked peak."
        ),
        body_side="right",
    ),
    Landmark(
        code="anthro_forearm_girth_right",
        category="ANTHRO",
        name_fr="Circonférence maximale de l'avant-bras (droit)",
        name_en="Maximum forearm girth (right)",
        hint_fr=(
            "Zone de plus grand périmètre de l'avant-bras droit, généralement à "
            "quelques centimètres sous le pli du coude. Bras pendant relâché, coude "
            "en extension, main en supination. Marquer le niveau du maximum, ruban "
            "perpendiculaire à l'axe de l'avant-bras."
        ),
        hint_en=(
            "Level of greatest perimeter of the right forearm, usually a few "
            "centimetres below the elbow crease. Arm hanging relaxed, elbow "
            "extended, hand supinated. Mark the level of the maximum; the tape stays "
            "perpendicular to the forearm axis."
        ),
        body_side="right",
    ),
    Landmark(
        code="anthro_thigh_girth_right",
        category="ANTHRO",
        name_fr="Circonférence de la mi-cuisse (droite)",
        name_en="Mid-thigh girth (right)",
        hint_fr=(
            "Milieu de la distance entre greater_trochanter_right (trochanterion) et "
            "fibular_head_right (tibiale laterale), mesurée sur la face latérale de "
            "la cuisse. Sujet debout, poids réparti également. Ruban perpendiculaire "
            "à l'axe long de la cuisse."
        ),
        hint_en=(
            "Midpoint between greater_trochanter_right (trochanterion) and "
            "fibular_head_right (tibiale laterale), measured on the lateral thigh. "
            "Subject standing with weight evenly distributed. Tape perpendicular to "
            "the long axis of the thigh."
        ),
        body_side="right",
    ),
    Landmark(
        code="anthro_calf_girth_right",
        category="ANTHRO",
        name_fr="Circonférence maximale du mollet (droite)",
        name_en="Maximum calf girth (right)",
        hint_fr=(
            "Niveau du plus grand périmètre du mollet droit, recherché en déplaçant "
            "le ruban vers le haut et vers le bas. Sujet debout, pieds écartés à "
            "largeur de bassin, poids réparti également. Marquer ce niveau sur la "
            "face médiale : il fixe aussi le site du pli du mollet médial."
        ),
        hint_en=(
            "Level of greatest perimeter of the right calf, found by sliding the "
            "tape up and down. Subject standing, feet pelvis-width apart, weight "
            "evenly distributed. Mark the level on the medial aspect: it also fixes "
            "the medial calf skinfold site."
        ),
        body_side="right",
    ),
    Landmark(
        code="anthro_humerus_breadth_right",
        category="ANTHRO",
        name_fr="Diamètre bi-épicondylien de l'humérus (droit)",
        name_en="Biepicondylar humerus breadth (right)",
        hint_fr=(
            "Mesure prise entre lateral_epicondyle_right et medial_epicondyle_right, "
            "épaule fléchie à 90° et coude fléchi à 90°, avant-bras en supination. "
            "Placer le compas obliquement (l'axe bi-épicondylien n'est pas "
            "horizontal) et comprimer fermement les tissus mous. Marquer le milieu "
            "de l'axe, sur la face postérieure du coude."
        ),
        hint_en=(
            "Measured between lateral_epicondyle_right and medial_epicondyle_right "
            "with the shoulder flexed to 90° and the elbow flexed to 90°, forearm "
            "supinated. Hold the caliper obliquely (the biepicondylar axis is not "
            "horizontal) and press firmly on the soft tissue. Mark the midpoint of "
            "the axis on the posterior elbow."
        ),
        body_side="right",
    ),
    Landmark(
        code="anthro_femur_breadth_right",
        category="ANTHRO",
        name_fr="Diamètre bi-épicondylien du fémur (droit)",
        name_en="Biepicondylar femur breadth (right)",
        hint_fr=(
            "Mesure prise entre lateral_knee_right et medial_knee_right (épicondyles "
            "fémoraux), sujet assis, genou fléchi à 90°. Placer le compas de biais "
            "vers le bas et comprimer fermement. Marquer le milieu de l'axe sur la "
            "face antérieure du genou, juste au-dessus de l'interligne."
        ),
        hint_en=(
            "Measured between lateral_knee_right and medial_knee_right (femoral "
            "epicondyles) with the subject seated and the knee flexed to 90°. Angle "
            "the caliper downward and press firmly. Mark the midpoint of the axis on "
            "the anterior knee, just above the joint line."
        ),
        body_side="right",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # EMG — Triceps chef latéral (SENIAM — complément au chef long)
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_triceps_brachii_lateral_left",
        category="EMG",
        name_fr="Triceps brachial chef latéral gauche (EMG)",
        name_en="Left triceps brachii lateral head (EMG)",
        hint_fr=(
            "Même niveau que le chef long : à 50 % de la ligne acromion_left → "
            "olecranon_left, puis 2 cm en **dehors** de cette ligne (face postéro-latérale "
            "du bras). Orientation : parallèle à l'axe huméral. Vérifier par extension du "
            "coude contre résistance, bras en abduction à 90°."
        ),
        hint_en=(
            "Same level as the long head: at 50 % of the line acromion_left → "
            "olecranon_left, then 2 cm **lateral** to this line (postero-lateral arm). "
            "Orientation: parallel to the humeral axis. Confirm with resisted elbow "
            "extension, arm abducted to 90°."
        ),
        body_side="left",
        theme="upper_limb",
        application_fr=(
            "Distinguer l'activité des chefs latéral et long du triceps (le chef latéral "
            "est plus actif en extension pure sans charge du grand pectoral). Fatigue lors "
            "du développé couché. Comparaison inter-chefs en réhabilitation du coude."
        ),
        application_en=(
            "Differentiate lateral vs long head triceps activity (lateral head more active "
            "in pure extension without pectoral load). Fatigue during bench press. "
            "Inter-head comparison in elbow rehabilitation."
        ),
    ),
    Landmark(
        code="EMG_triceps_brachii_lateral_right",
        category="EMG",
        name_fr="Triceps brachial chef latéral droit (EMG)",
        name_en="Right triceps brachii lateral head (EMG)",
        hint_fr=(
            "Même niveau que le chef long : à 50 % de la ligne acromion_right → "
            "olecranon_right, puis 2 cm en **dehors** de cette ligne (face postéro-latérale "
            "du bras). Orientation : parallèle à l'axe huméral. Vérifier par extension du "
            "coude contre résistance, bras en abduction à 90°."
        ),
        hint_en=(
            "Same level as the long head: at 50 % of the line acromion_right → "
            "olecranon_right, then 2 cm **lateral** to this line (postero-lateral arm). "
            "Orientation: parallel to the humeral axis. Confirm with resisted elbow "
            "extension, arm abducted to 90°."
        ),
        body_side="right",
        theme="upper_limb",
        application_fr=(
            "Distinguer l'activité des chefs latéral et long du triceps (le chef latéral "
            "est plus actif en extension pure sans charge du grand pectoral). Fatigue lors "
            "du développé couché. Comparaison inter-chefs en réhabilitation du coude."
        ),
        application_en=(
            "Differentiate lateral vs long head triceps activity (lateral head more active "
            "in pure extension without pectoral load). Fatigue during bench press. "
            "Inter-head comparison in elbow rehabilitation."
        ),
    ),

    # ══════════════════════════════════════════════════════════════════════
    # EMG — Brachioradial (pratique courante — non SENIAM officiel)
    # Cross-talk notable avec les extenseurs radiaux du poignet
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_brachioradialis_left",
        category="EMG",
        name_fr="Brachioradial gauche (EMG)",
        name_en="Left brachioradialis (EMG)",
        hint_fr=(
            "Tracer la ligne lateral_epicondyle_left → radial_styloid_left et marquer "
            "à 25 % depuis l'épicondyle, sur le relief antéro-radial de l'avant-bras "
            "(« mobile wad »). Coude à 90° en position neutre (pouce vers le haut). "
            "Orientation : axe longitudinal de l'avant-bras. ⚠️ Site non officiel SENIAM "
            "— cross-talk important avec les extenseurs radiaux (plus dorsaux). Vérifier "
            "par flexion du coude en position neutre."
        ),
        hint_en=(
            "Draw the line lateral_epicondyle_left → radial_styloid_left and mark at "
            "25 % from the epicondyle, on the antero-radial forearm prominence "
            "('mobile wad'). Elbow at 90° in neutral (thumb up). Orientation: "
            "forearm long axis. ⚠️ Not an official SENIAM site — significant cross-talk "
            "with the radial wrist extensors (more dorsal). Confirm by neutral forearm "
            "elbow flexion."
        ),
        body_side="left",
        theme="upper_limb",
        application_fr=(
            "Épicondylalgie latérale (différenciation brachioradial / extenseurs). "
            "Ergonomie du poste informatique : préhension et prono-supination répétées. "
            "Réhabilitation post-fracture de Colles. ⚠️ Non SENIAM : étiqueter explicitement "
            "dans le rapport EMG."
        ),
        application_en=(
            "Lateral epicondylalgia (brachioradialis vs extensor differentiation). "
            "Office workstation ergonomics: grip and repeated pronation-supination. "
            "Post-Colles fracture rehabilitation. ⚠️ Non-SENIAM: label explicitly in EMG report."
        ),
    ),
    Landmark(
        code="EMG_brachioradialis_right",
        category="EMG",
        name_fr="Brachioradial droit (EMG)",
        name_en="Right brachioradialis (EMG)",
        hint_fr=(
            "Tracer la ligne lateral_epicondyle_right → radial_styloid_right et marquer "
            "à 25 % depuis l'épicondyle, sur le relief antéro-radial de l'avant-bras "
            "(« mobile wad »). Coude à 90° en position neutre (pouce vers le haut). "
            "Orientation : axe longitudinal de l'avant-bras. ⚠️ Site non officiel SENIAM "
            "— cross-talk important avec les extenseurs radiaux (plus dorsaux). Vérifier "
            "par flexion du coude en position neutre."
        ),
        hint_en=(
            "Draw the line lateral_epicondyle_right → radial_styloid_right and mark at "
            "25 % from the epicondyle, on the antero-radial forearm prominence "
            "('mobile wad'). Elbow at 90° in neutral (thumb up). Orientation: "
            "forearm long axis. ⚠️ Not an official SENIAM site — significant cross-talk "
            "with the radial wrist extensors (more dorsal). Confirm by neutral forearm "
            "elbow flexion."
        ),
        body_side="right",
        theme="upper_limb",
        application_fr=(
            "Épicondylalgie latérale (différenciation brachioradial / extenseurs). "
            "Ergonomie du poste informatique : préhension et prono-supination répétées. "
            "Réhabilitation post-fracture de Colles. ⚠️ Non SENIAM : étiqueter explicitement "
            "dans le rapport EMG."
        ),
        application_en=(
            "Lateral epicondylalgia (brachioradialis vs extensor differentiation). "
            "Office workstation ergonomics: grip and repeated pronation-supination. "
            "Post-Colles fracture rehabilitation. ⚠️ Non-SENIAM: label explicitly in EMG report."
        ),
    ),

    # ══════════════════════════════════════════════════════════════════════
    # EMG — Dentelé antérieur (pratique courante — non SENIAM officiel)
    # Site difficile : zone axillaire, cross-talk grand dorsal / obliques
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_serratus_anterior_left",
        category="EMG",
        name_fr="Dentelé antérieur gauche (EMG)",
        name_en="Left serratus anterior (EMG)",
        hint_fr=(
            "Sur la ligne axillaire moyenne, au niveau de l'angle inférieur de la "
            "scapula_inferior_angle_left (≈ côtes 6-7), **en avant** du bord latéral du "
            "grand dorsal, sur une digitation costale visible à l'effort (bras élevé à "
            "90° en flexion). Orientation : le long de la digitation costale, parallèle à "
            "la côte. ⚠️ Non SENIAM officiel — cross-talk grand dorsal et obliques. "
            "Vérifier par protraction scapulaire contre résistance."
        ),
        hint_en=(
            "On the mid-axillary line, at the level of scapula_inferior_angle_left "
            "(≈ ribs 6-7), **anterior** to the lateral border of latissimus dorsi, over "
            "a visible costal digitation (arm at 90° flexion). Orientation: along the "
            "costal digitation, parallel to the rib. ⚠️ Not official SENIAM — cross-talk "
            "with latissimus dorsi and obliques. Confirm with resisted scapular protraction."
        ),
        body_side="left",
        theme="shoulder",
        application_fr=(
            "Dyskinésie scapulaire et décollement de la scapula (winging). Rééducation "
            "post-opératoire de l'épaule (push-up plus). Service au tennis et lancer. "
            "⚠️ Exige un bras en élévation pour visualiser la digitation."
        ),
        application_en=(
            "Scapular dyskinesis and winging scapula. Post-operative shoulder "
            "rehabilitation (push-up plus). Tennis serve and throwing. "
            "⚠️ Requires arm elevation to visualize the digitation."
        ),
    ),
    Landmark(
        code="EMG_serratus_anterior_right",
        category="EMG",
        name_fr="Dentelé antérieur droit (EMG)",
        name_en="Right serratus anterior (EMG)",
        hint_fr=(
            "Sur la ligne axillaire moyenne, au niveau de l'angle inférieur de la "
            "scapula_inferior_angle_right (≈ côtes 6-7), **en avant** du bord latéral du "
            "grand dorsal, sur une digitation costale visible à l'effort (bras élevé à "
            "90° en flexion). Orientation : le long de la digitation costale, parallèle à "
            "la côte. ⚠️ Non SENIAM officiel — cross-talk grand dorsal et obliques. "
            "Vérifier par protraction scapulaire contre résistance."
        ),
        hint_en=(
            "On the mid-axillary line, at the level of scapula_inferior_angle_right "
            "(≈ ribs 6-7), **anterior** to the lateral border of latissimus dorsi, over "
            "a visible costal digitation (arm at 90° flexion). Orientation: along the "
            "costal digitation, parallel to the rib. ⚠️ Not official SENIAM — cross-talk "
            "with latissimus dorsi and obliques. Confirm with resisted scapular protraction."
        ),
        body_side="right",
        theme="shoulder",
        application_fr=(
            "Dyskinésie scapulaire et décollement de la scapula (winging). Rééducation "
            "post-opératoire de l'épaule (push-up plus). Service au tennis et lancer. "
            "⚠️ Exige un bras en élévation pour visualiser la digitation."
        ),
        application_en=(
            "Scapular dyskinesis and winging scapula. Post-operative shoulder "
            "rehabilitation (push-up plus). Tennis serve and throwing. "
            "⚠️ Requires arm elevation to visualize the digitation."
        ),
    ),

    # ══════════════════════════════════════════════════════════════════════
    # EMG — Grand pectoral faisceau sterno-costal (pratique courante)
    # Site sensible : zone thoracique antérieure — note de consentement requise
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_pectoralis_major_sternal_left",
        category="EMG",
        name_fr="Grand pectoral sterno-costal gauche (EMG)",
        name_en="Left pectoralis major sternal head (EMG)",
        hint_fr=(
            "À mi-distance entre l'angle sternal (sternal_angle_louis) et le pli "
            "axillaire antérieur, au niveau de la 4e-5e côte, 2-3 cm en dedans du "
            "pli axillaire. Orientation : oblique vers le tubercule majeur de l'humérus "
            "(fibres convergentes). ⚠️ Non SENIAM officiel. Site thoracique antérieur : "
            "obtenir le consentement du sujet. Vérifier par adduction du bras contre "
            "résistance."
        ),
        hint_en=(
            "Midway between the sternal angle (sternal_angle_louis) and the anterior "
            "axillary fold, at the 4th-5th rib level, 2-3 cm medial to the fold. "
            "Orientation: oblique toward the greater tubercle of the humerus (converging "
            "fibers). ⚠️ Not official SENIAM. Anterior chest site: obtain subject consent. "
            "Confirm with resisted arm adduction."
        ),
        body_side="left",
        theme="shoulder",
        application_fr=(
            "Analyse du développé couché et du push-up. Réhabilitation post-mastectomie "
            "ou post-reconstruction. Phase de traction en natation. ⚠️ Obtenir "
            "explicitement le consentement (site thoracique antérieur)."
        ),
        application_en=(
            "Bench press and push-up analysis. Post-mastectomy or breast reconstruction "
            "rehabilitation. Swimming pull phase. ⚠️ Explicitly obtain consent "
            "(anterior chest site)."
        ),
    ),
    Landmark(
        code="EMG_pectoralis_major_sternal_right",
        category="EMG",
        name_fr="Grand pectoral sterno-costal droit (EMG)",
        name_en="Right pectoralis major sternal head (EMG)",
        hint_fr=(
            "À mi-distance entre l'angle sternal (sternal_angle_louis) et le pli "
            "axillaire antérieur, au niveau de la 4e-5e côte, 2-3 cm en dedans du "
            "pli axillaire. Orientation : oblique vers le tubercule majeur de l'humérus "
            "(fibres convergentes). ⚠️ Non SENIAM officiel. Site thoracique antérieur : "
            "obtenir le consentement du sujet. Vérifier par adduction du bras contre "
            "résistance."
        ),
        hint_en=(
            "Midway between the sternal angle (sternal_angle_louis) and the anterior "
            "axillary fold, at the 4th-5th rib level, 2-3 cm medial to the fold. "
            "Orientation: oblique toward the greater tubercle of the humerus (converging "
            "fibers). ⚠️ Not official SENIAM. Anterior chest site: obtain subject consent. "
            "Confirm with resisted arm adduction."
        ),
        body_side="right",
        theme="shoulder",
        application_fr=(
            "Analyse du développé couché et du push-up. Réhabilitation post-mastectomie "
            "ou post-reconstruction. Phase de traction en natation. ⚠️ Obtenir "
            "explicitement le consentement (site thoracique antérieur)."
        ),
        application_en=(
            "Bench press and push-up analysis. Post-mastectomy or breast reconstruction "
            "rehabilitation. Swimming pull phase. ⚠️ Explicitly obtain consent "
            "(anterior chest site)."
        ),
    ),

    # ══════════════════════════════════════════════════════════════════════
    # EMG — Iliocostal des lombes (SENIAM — niveau L1-L2)
    # Distinctinct du longissimus : plus latéral, ancré sur la ligne EIPS → 12e côte
    # ══════════════════════════════════════════════════════════════════════
    Landmark(
        code="EMG_erector_spinae_iliocostalis_left",
        category="EMG",
        name_fr="Iliocostal des lombes gauche (EMG)",
        name_en="Left erector spinae iliocostalis lumborum (EMG)",
        hint_fr=(
            "Tracer la ligne PSIS_left → point le plus bas de la 12e côte. Marquer "
            "à 1 travers de doigt **en dedans** de cette ligne, au niveau L1-L2 "
            "(≈ 6 cm latéralement à l'épineuse de L1). Plus latéral que le longissimus. "
            "Orientation : verticale, parallèle au rachis. Vérifier par extension du "
            "tronc contre résistance en DV."
        ),
        hint_en=(
            "Draw the line PSIS_left → lowest point of the 12th rib. Mark 1 finger-width "
            "**medial** to this line, at the L1-L2 level (~6 cm lateral to the L1 "
            "spinous process). More lateral than the longissimus. Orientation: vertical, "
            "parallel to the spine. Confirm with resisted trunk extension in prone."
        ),
        body_side="left",
        theme="core",
        application_fr=(
            "Asymétrie paravertébrale en scoliose (différence gauche/droite). Ergonomie "
            "du port de charge (comparaison iliocostal vs longissimus). Syndrome croisé "
            "inférieur (raccourcissement des érecteurs)."
        ),
        application_en=(
            "Paravertebral asymmetry in scoliosis (left/right difference). Manual "
            "handling ergonomics (iliocostalis vs longissimus comparison). Lower crossed "
            "syndrome (erector shortening)."
        ),
    ),
    Landmark(
        code="EMG_erector_spinae_iliocostalis_right",
        category="EMG",
        name_fr="Iliocostal des lombes droit (EMG)",
        name_en="Right erector spinae iliocostalis lumborum (EMG)",
        hint_fr=(
            "Tracer la ligne PSIS_right → point le plus bas de la 12e côte. Marquer "
            "à 1 travers de doigt **en dedans** de cette ligne, au niveau L1-L2 "
            "(≈ 6 cm latéralement à l'épineuse de L1). Plus latéral que le longissimus. "
            "Orientation : verticale, parallèle au rachis. Vérifier par extension du "
            "tronc contre résistance en DV."
        ),
        hint_en=(
            "Draw the line PSIS_right → lowest point of the 12th rib. Mark 1 finger-width "
            "**medial** to this line, at the L1-L2 level (~6 cm lateral to the L1 "
            "spinous process). More lateral than the longissimus. Orientation: vertical, "
            "parallel to the spine. Confirm with resisted trunk extension in prone."
        ),
        body_side="right",
        theme="core",
        application_fr=(
            "Asymétrie paravertébrale en scoliose (différence gauche/droite). Ergonomie "
            "du port de charge (comparaison iliocostal vs longissimus). Syndrome croisé "
            "inférieur (raccourcissement des érecteurs)."
        ),
        application_en=(
            "Paravertebral asymmetry in scoliosis (left/right difference). Manual "
            "handling ergonomics (iliocostalis vs longissimus comparison). Lower crossed "
            "syndrome (erector shortening)."
        ),
    ),
]


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

    ``include_midline`` also returns the non-lateralised ('bilateral') markers
    such as ``C7_spinous`` or ``sacrum_S2``.
    """
    return [
        lm
        for lm in LANDMARKS
        if lm.body_side == side or (include_midline and lm.body_side == "bilateral")
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
