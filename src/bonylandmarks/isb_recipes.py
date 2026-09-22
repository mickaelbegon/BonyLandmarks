"""Recettes de construction guidée des 7 repères locaux ISB.

Chaque recette est une ``list[Step]`` (voir :mod:`isb_step_engine`) qui conduit
l'étudiant, étape par étape, de la palpation des landmarks au repère orthonormé
direct du segment, conformément à Wu et al. 2002 (membre inférieur, rachis) et
Wu et al. 2005 (ceinture scapulaire, membre supérieur).

Depuis la migration vers les **templates déclaratifs**, ces recettes ne sont
plus écrites à la main : elles sont *dérivées* des :class:`FrameTemplate` de
:mod:`frame_template`.  Un template ne décrit que ce qui est propre au segment
(points, origine, axe primaire, direction guide, vocabulaire anatomique) ; la
convention du trièdre direct fait le reste.

Conventions
-----------
* **Repère ISB local** : X antérieur, Y supérieur (proximal pour les membres),
  Z vers la droite (latéral) du sujet ; trièdre direct, donc ``X = Y × Z``,
  ``Y = Z × X``, ``Z = X × Y``.
* **Repère global GLB** (coordonnées brutes des landmarks) : X droite, Y haut,
  −Z antérieur (convention Blender).  Les recettes ne s'appuient jamais sur ces
  axes globaux : tous les axes ISB sont déduits des seuls landmarks du segment,
  ce qui rend la construction indépendante de la pose du scan.
* Les produits vectoriels sont **ordonnés** : inverser A × B retourne l'axe.
  Le moteur détecte cette erreur et l'explique.
* L'orthogonalisation se fait par double produit vectoriel plutôt que par
  projection, afin que chaque étape reste une opération que l'étudiant peut
  exécuter à la main : l'axe primaire et la direction guide donnent d'abord
  l'axe secondaire, qui redonne ensuite le guide orthogonalisé.
* Les deux produits vectoriels finaux restent à la charge de l'étudiant ; seules
  les étapes reconstruisant un repère déjà construit ailleurs (la clavicule
  emprunte son Y au thorax) sont marquées ``auto=True``.

Correspondance des codes et proxies assumés : voir :mod:`frame_template`.
"""

from __future__ import annotations

import numpy as np

from .frame_template import ISB_TEMPLATES, template_to_steps
from .isb_step_engine import PickLandmark, StepEngine, Step

__all__ = [
    "ISB_RECIPES",
    "ISB_RECIPE_META",
    "RECIPE_KEYS",
    "get_recipe",
    "recipe_label",
    "required_landmarks",
    "make_engine",
]


# ─────────────────────────────────────────────────────────────────────────────
# Dérivation des recettes
# ─────────────────────────────────────────────────────────────────────────────

def _build_recipes() -> dict[str, list[Step]]:
    """Dérive les 7 recettes depuis les templates déclaratifs."""
    return {
        key: template_to_steps(tpl, ISB_TEMPLATES)
        for key, tpl in ISB_TEMPLATES.items()
    }


#: Recettes indexées par clé de segment (mêmes clés que ``isb_exercise``).
ISB_RECIPES: dict[str, list[Step]] = _build_recipes()

#: Ordre canonique d'affichage (du proximal/axial vers le distal).
RECIPE_KEYS: list[str] = list(ISB_RECIPES)


def _landmark_codes(steps: list[Step]) -> list[str]:
    """Codes des landmarks d'une recette, dans l'ordre et sans doublon."""
    out: list[str] = []
    for step in steps:
        if isinstance(step, PickLandmark) and step.expected_code not in out:
            out.append(step.expected_code)
    return out


#: Métadonnées d'affichage par segment (dérivées des templates).
ISB_RECIPE_META: dict[str, dict] = {
    key: {
        "name_fr": tpl.display_name("fr"),
        "name_en": tpl.display_name("en"),
        "label_fr": tpl.label_fr,
        "label_en": tpl.label("en"),
        "reference": tpl.reference,
        "n_steps": len(ISB_RECIPES[key]),
        "required_landmarks": _landmark_codes(ISB_RECIPES[key]),
    }
    for key, tpl in ISB_TEMPLATES.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────

def get_recipe(segment_key: str) -> list[Step]:
    """Recette d'un segment. Lève ``KeyError`` si la clé est inconnue."""
    if segment_key not in ISB_RECIPES:
        raise KeyError(
            f"Segment inconnu : {segment_key!r}. Clés valides : {RECIPE_KEYS}"
        )
    return ISB_RECIPES[segment_key]


def recipe_label(segment_key: str, lang: str = "fr") -> str:
    """Nom lisible d'un segment."""
    meta = ISB_RECIPE_META.get(segment_key, {})
    return meta.get("name_fr" if lang == "fr" else "name_en", segment_key)


def required_landmarks(segment_key: str) -> list[str]:
    """Codes des landmarks nécessaires à une recette, dans l'ordre de la recette."""
    return _landmark_codes(get_recipe(segment_key))


def make_engine(
    segment_key: str, ground_truth: dict[str, np.ndarray]
) -> StepEngine:
    """Instancie un :class:`~.isb_step_engine.StepEngine` pour un segment."""
    return StepEngine(get_recipe(segment_key), ground_truth, segment_key=segment_key)
