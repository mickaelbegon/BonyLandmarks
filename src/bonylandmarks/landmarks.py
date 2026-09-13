"""Anatomical landmark definitions with bilingual names (FR/EN).

Each landmark maps the BodyLoop marker code to its anatomical name in French
and English, plus a short placement hint shown in the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Language = Literal["fr", "en"]


@dataclass(frozen=True)
class Landmark:
    code: str
    name_fr: str
    name_en: str
    hint_fr: str
    hint_en: str
    body_side: Literal["left", "right", "bilateral"] = "bilateral"

    def name(self, lang: Language = "fr") -> str:
        return self.name_fr if lang == "fr" else self.name_en

    def hint(self, lang: Language = "fr") -> str:
        return self.hint_fr if lang == "fr" else self.hint_en


LANDMARKS: list[Landmark] = [
    # ── Pelvis ──────────────────────────────────────────────────────────────
    Landmark(
        code="ASIS_left",
        name_fr="Épine iliaque antéro-supérieure gauche",
        name_en="Left anterior superior iliac spine",
        hint_fr="Point osseux saillant à l'avant de la crête iliaque, côté gauche.",
        hint_en="Bony prominence at the front of the left iliac crest.",
        body_side="left",
    ),
    Landmark(
        code="ASIS_right",
        name_fr="Épine iliaque antéro-supérieure droite",
        name_en="Right anterior superior iliac spine",
        hint_fr="Point osseux saillant à l'avant de la crête iliaque, côté droit.",
        hint_en="Bony prominence at the front of the right iliac crest.",
        body_side="right",
    ),
    Landmark(
        code="greater_trochanter_left",
        name_fr="Grand trochanter gauche",
        name_en="Left greater trochanter",
        hint_fr="Apophyse latérale du fémur, palpable sur la face externe de la cuisse gauche.",
        hint_en="Lateral bony prominence of the left femur, palpable on the outer thigh.",
        body_side="left",
    ),
    Landmark(
        code="greater_trochanter_right",
        name_fr="Grand trochanter droit",
        name_en="Right greater trochanter",
        hint_fr="Apophyse latérale du fémur, palpable sur la face externe de la cuisse droite.",
        hint_en="Lateral bony prominence of the right femur, palpable on the outer thigh.",
        body_side="right",
    ),
    # ── Épaule ──────────────────────────────────────────────────────────────
    Landmark(
        code="acromion_left",
        name_fr="Acromion gauche",
        name_en="Left acromion",
        hint_fr="Processus plat et large de la scapula formant le sommet de l'épaule gauche.",
        hint_en="Flat bony process of the left scapula forming the tip of the shoulder.",
        body_side="left",
    ),
    Landmark(
        code="acromion_right",
        name_fr="Acromion droit",
        name_en="Right acromion",
        hint_fr="Processus plat et large de la scapula formant le sommet de l'épaule droite.",
        hint_en="Flat bony process of the right scapula forming the tip of the shoulder.",
        body_side="right",
    ),
    # ── Coude ───────────────────────────────────────────────────────────────
    Landmark(
        code="lateral_epicondyle_left",
        name_fr="Épicondyle latéral gauche",
        name_en="Left lateral epicondyle",
        hint_fr="Saillie osseuse latérale de l'extrémité distale de l'humérus gauche.",
        hint_en="Lateral bony prominence at the distal end of the left humerus.",
        body_side="left",
    ),
    Landmark(
        code="medial_epicondyle_left",
        name_fr="Épicondyle médial gauche",
        name_en="Left medial epicondyle",
        hint_fr="Saillie osseuse médiale de l'extrémité distale de l'humérus gauche.",
        hint_en="Medial bony prominence at the distal end of the left humerus.",
        body_side="left",
    ),
    Landmark(
        code="lateral_epicondyle_right",
        name_fr="Épicondyle latéral droit",
        name_en="Right lateral epicondyle",
        hint_fr="Saillie osseuse latérale de l'extrémité distale de l'humérus droit.",
        hint_en="Lateral bony prominence at the distal end of the right humerus.",
        body_side="right",
    ),
    Landmark(
        code="medial_epicondyle_right",
        name_fr="Épicondyle médial droit",
        name_en="Right medial epicondyle",
        hint_fr="Saillie osseuse médiale de l'extrémité distale de l'humérus droit.",
        hint_en="Medial bony prominence at the distal end of the right humerus.",
        body_side="right",
    ),
    # ── Poignet ─────────────────────────────────────────────────────────────
    Landmark(
        code="ulnar_styloid_left",
        name_fr="Processus styloïde ulnaire gauche",
        name_en="Left ulnar styloid process",
        hint_fr="Saillie distale de l'ulna, palpable sur la face médiale du poignet gauche.",
        hint_en="Distal projection of the left ulna, palpable on the medial wrist.",
        body_side="left",
    ),
    Landmark(
        code="radial_styloid_left",
        name_fr="Processus styloïde radial gauche",
        name_en="Left radial styloid process",
        hint_fr="Saillie distale du radius, palpable sur la face latérale du poignet gauche.",
        hint_en="Distal projection of the left radius, palpable on the lateral wrist.",
        body_side="left",
    ),
    Landmark(
        code="ulnar_styloid_right",
        name_fr="Processus styloïde ulnaire droit",
        name_en="Right ulnar styloid process",
        hint_fr="Saillie distale de l'ulna, palpable sur la face médiale du poignet droit.",
        hint_en="Distal projection of the right ulna, palpable on the medial wrist.",
        body_side="right",
    ),
    Landmark(
        code="radial_styloid_right",
        name_fr="Processus styloïde radial droit",
        name_en="Right radial styloid process",
        hint_fr="Saillie distale du radius, palpable sur la face latérale du poignet droit.",
        hint_en="Distal projection of the right radius, palpable on the lateral wrist.",
        body_side="right",
    ),
    # ── Genou ───────────────────────────────────────────────────────────────
    Landmark(
        code="lateral_knee_left",
        name_fr="Condyle latéral du genou gauche",
        name_en="Left lateral knee condyle",
        hint_fr="Saillie osseuse latérale du condyle fémoral gauche, au niveau de l'interligne articulaire.",
        hint_en="Lateral bony prominence of the left femoral condyle at the joint line.",
        body_side="left",
    ),
    Landmark(
        code="medial_knee_left",
        name_fr="Condyle médial du genou gauche",
        name_en="Left medial knee condyle",
        hint_fr="Saillie osseuse médiale du condyle fémoral gauche, au niveau de l'interligne articulaire.",
        hint_en="Medial bony prominence of the left femoral condyle at the joint line.",
        body_side="left",
    ),
    Landmark(
        code="lateral_knee_right",
        name_fr="Condyle latéral du genou droit",
        name_en="Right lateral knee condyle",
        hint_fr="Saillie osseuse latérale du condyle fémoral droit, au niveau de l'interligne articulaire.",
        hint_en="Lateral bony prominence of the right femoral condyle at the joint line.",
        body_side="right",
    ),
    Landmark(
        code="medial_knee_right",
        name_fr="Condyle médial du genou droit",
        name_en="Right medial knee condyle",
        hint_fr="Saillie osseuse médiale du condyle fémoral droit, au niveau de l'interligne articulaire.",
        hint_en="Medial bony prominence of the right femoral condyle at the joint line.",
        body_side="right",
    ),
    # ── Cheville ────────────────────────────────────────────────────────────
    Landmark(
        code="lateral_malleolus_left",
        name_fr="Malléole latérale gauche",
        name_en="Left lateral malleolus",
        hint_fr="Extrémité distale du péroné, saillante sur la face latérale de la cheville gauche.",
        hint_en="Distal end of the left fibula, prominent on the lateral ankle.",
        body_side="left",
    ),
    Landmark(
        code="medial_malleolus_left",
        name_fr="Malléole médiale gauche",
        name_en="Left medial malleolus",
        hint_fr="Extrémité distale du tibia, saillante sur la face médiale de la cheville gauche.",
        hint_en="Distal end of the left tibia, prominent on the medial ankle.",
        body_side="left",
    ),
    Landmark(
        code="lateral_malleolus_right",
        name_fr="Malléole latérale droite",
        name_en="Right lateral malleolus",
        hint_fr="Extrémité distale du péroné, saillante sur la face latérale de la cheville droite.",
        hint_en="Distal end of the right fibula, prominent on the lateral ankle.",
        body_side="right",
    ),
    Landmark(
        code="medial_malleolus_right",
        name_fr="Malléole médiale droite",
        name_en="Right medial malleolus",
        hint_fr="Extrémité distale du tibia, saillante sur la face médiale de la cheville droite.",
        hint_en="Distal end of the right tibia, prominent on the medial ankle.",
        body_side="right",
    ),
    # ── Talon ────────────────────────────────────────────────────────────────
    Landmark(
        code="heel_left",
        name_fr="Talon gauche",
        name_en="Left heel",
        hint_fr="Face postérieure la plus distale du calcanéus gauche.",
        hint_en="Most posterior point of the left calcaneus.",
        body_side="left",
    ),
    Landmark(
        code="heel_right",
        name_fr="Talon droit",
        name_en="Right heel",
        hint_fr="Face postérieure la plus distale du calcanéus droit.",
        hint_en="Most posterior point of the right calcaneus.",
        body_side="right",
    ),
]

# Fast lookup by code
LANDMARK_BY_CODE: dict[str, Landmark] = {lm.code: lm for lm in LANDMARKS}


def get_landmark(code: str) -> Landmark:
    """Return the Landmark for the given BodyLoop marker code."""
    if code not in LANDMARK_BY_CODE:
        raise KeyError(f"Unknown landmark code: {code!r}")
    return LANDMARK_BY_CODE[code]
