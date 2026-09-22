"""Templates déclaratifs de repères locaux ISB.

Ce module remplace les recettes procédurales écrites à la main par une
**description déclarative** de chaque repère (:class:`FrameTemplate`) et une
fonction de dérivation (:func:`template_to_steps`) qui en produit la séquence
pédagogique de :class:`~.isb_step_engine.Step`.

Pourquoi
--------
Un repère ISB se décrit entièrement par cinq informations :

1. les **points** à obtenir (palpations et milieux construits) ;
2. l'**origine** ;
3. l'axe **primaire** (porté exactement par une direction anatomique) ;
4. une direction **guide** (dans le plan voulu, mais pas encore orthogonale) ;
5. le **vocabulaire** anatomique du segment (``X`` = « avant », ``Y`` = « haut »
   ou « proximal », ``Z`` = « droite » ou « latéral »).

Tout le reste — l'ordre des produits vectoriels, l'axe qu'ils produisent, quel
axe est orthogonalisé en dernier — est **déduit** de la convention du trièdre
direct ISB :

    X × Y = Z        Y × Z = X        Z × X = Y

Ce qui reste à la charge de l'étudiant
--------------------------------------
La dérivation ne « court-circuite » jamais la pédagogie : les **deux** produits
vectoriels finaux (axe secondaire, puis orthogonalisation du guide) sont
toujours des étapes que l'étudiant exécute lui-même.  Seules sont marquées
``auto=True`` les étapes qui reconstruisent un repère *déjà construit par
ailleurs* (dépendance inter-segments : la clavicule emprunte son Y au thorax).

Expressions disponibles
-----------------------
=============  ==============================================================
:class:`LM`    un landmark à palper (``also`` = codes tolérés)
:class:`Mid`   milieu de deux points du workspace
:class:`Seg`   ``normalize(workspace[to] - workspace[frm])``
:class:`Normal` ``normalize((p1 - apex) × (p2 - apex))``
:class:`Borrow` un axe d'un autre segment déjà construit
=============  ==============================================================

Une expression :class:`Seg` produit une étape (``ComputeVector``) ; une
expression :class:`Normal` en produit trois (deux ``ComputeVector`` puis un
``ComputeCrossProduct``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .isb_step_engine import (
    ComputeCrossProduct,
    ComputeMidpoint,
    ComputeVector,
    PickLandmark,
    Step,
    ValidateFrame,
)

__all__ = [
    "LM",
    "Mid",
    "Seg",
    "Normal",
    "Borrow",
    "AxisSpec",
    "FrameTemplate",
    "ISB_TEMPLATES",
    "template_to_steps",
    "cross_order",
    "remaining",
]


# ─────────────────────────────────────────────────────────────────────────────
# Expressions de points
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LM:
    """Un landmark à palper.

    ``also`` liste les codes acceptés comme équivalents dégradés (transmis à
    ``PickLandmark.tolerance_ok``).
    """

    code: str
    also: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mid:
    """Milieu de deux points déjà nommés dans le workspace."""

    a: str
    b: str


# ─────────────────────────────────────────────────────────────────────────────
# Expressions de directions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Seg:
    """``normalize(workspace[to] - workspace[frm])``."""

    frm: str
    to: str


@dataclass(frozen=True)
class Normal:
    """Normale au plan ``(apex, p1, p2)`` : ``(p1 - apex) × (p2 - apex)``."""

    apex: str
    p1: str
    p2: str


@dataclass(frozen=True)
class Borrow:
    """Axe emprunté au repère d'un autre segment (dépendance inter-segments)."""

    segment: str
    axis: str


DirExpr = Seg | Normal | Borrow
PointExpr = LM | Mid


@dataclass
class AxisSpec:
    """Un axe (ou une direction guide) du repère."""

    label: str                  #: ``"X"``, ``"Y"`` ou ``"Z"``
    expr: DirExpr
    name: str = ""              #: nom dans le workspace (défaut : ``label``)

    def ws_name(self) -> str:
        """Nom sous lequel la direction est stockée dans le workspace."""
        return self.name or self.label


@dataclass
class FrameTemplate:
    """Description déclarative complète d'un repère local.

    Attributes
    ----------
    points
        ``dict`` **ordonné** : l'ordre de déclaration est l'ordre de palpation
        puis de construction présenté à l'étudiant.
    primary
        Axe porté *exactement* par une direction anatomique.
    guide
        Direction plaçant le plan du repère ; elle n'est pas orthogonale au
        primaire et sera orthogonalisée par le second produit vectoriel.
    words
        Vocabulaire anatomique FR : ``{"X": "avant", "Y": "haut", ...}``.
    notes
        Surcharges de prose, indexées par nom d'objet du workspace
        (``"O"``, ``"X_raw"``, ``"Y"``…).  La valeur remplace l'indice
        auto-généré de l'étape qui produit cet objet ; la clé suffixée
        ``"@en"`` fournit la version anglaise.
    """

    key: str
    label_fr: str                       #: adjectif : « thoracique », « pelvien »…
    reference: str
    points: dict[str, PointExpr]
    origin: str
    primary: AxisSpec
    guide: AxisSpec
    words: dict[str, str]
    notes: dict[str, str] = field(default_factory=dict)
    label_en: str = ""                  #: adjectif anglais (« thoracic »…)
    name_fr: str = ""                   #: nom court d'affichage (« Thorax »)
    name_en: str = ""
    words_en: dict[str, str] = field(default_factory=dict)

    # ── Accès localisés ──────────────────────────────────────────────────────

    def label(self, lang: str = "fr") -> str:
        return self.label_fr if lang == "fr" else (self.label_en or self.label_fr)

    def display_name(self, lang: str = "fr") -> str:
        if lang == "fr":
            return self.name_fr or self.label_fr
        return self.name_en or self.name_fr or self.label_fr

    def word(self, axis: str, lang: str = "fr") -> str:
        if lang == "fr":
            return self.words.get(axis, axis)
        if axis in self.words_en:
            return self.words_en[axis]
        return _EN_WORD.get(self.words.get(axis, ""), self.words.get(axis, axis))

    def note(self, name: str, lang: str = "fr") -> str:
        if lang == "fr":
            return self.notes.get(name, "")
        return self.notes.get(f"{name}@en", "") or self.notes.get(name, "")


#: Traduction du vocabulaire anatomique court FR → EN.
_EN_WORD: dict[str, str] = {
    "avant": "forward",
    "haut": "up",
    "droite": "right",
    "proximal": "proximal",
    "latéral": "lateral",
}


# ─────────────────────────────────────────────────────────────────────────────
# Convention du trièdre direct ISB
# ─────────────────────────────────────────────────────────────────────────────

#: ``(premier, second) → résultat`` pour la convention directe.
_CYCLE: dict[tuple[str, str], str] = {
    ("X", "Y"): "Z",
    ("Y", "Z"): "X",
    ("Z", "X"): "Y",
}


def cross_order(a: str, b: str) -> tuple[str, str, str]:
    """``(premier, second, résultat)`` tel que ``premier × second = résultat``.

    L'ordre des arguments est sans importance : c'est la convention directe qui
    impose lequel des deux axes vient en premier dans le produit vectoriel.
    """
    if (a, b) in _CYCLE:
        return a, b, _CYCLE[(a, b)]
    if (b, a) in _CYCLE:
        return b, a, _CYCLE[(b, a)]
    raise ValueError(f"axes non distincts ou inconnus : {a!r}, {b!r}")


def remaining(a: str, b: str) -> str:
    """Le troisième axe du trièdre."""
    rest = {"X", "Y", "Z"} - {a, b}
    if len(rest) != 1:
        raise ValueError(f"axes non distincts ou inconnus : {a!r}, {b!r}")
    return rest.pop()


# ─────────────────────────────────────────────────────────────────────────────
# Prose : noms et indices de palpation issus de la base de landmarks
# ─────────────────────────────────────────────────────────────────────────────

def _lm_name(code: str, lang: str = "fr") -> str:
    """Nom lisible d'un landmark, ou son code brut si la base est absente."""
    try:
        from .landmarks_extended import LANDMARK_BY_CODE
    except Exception:                                   # pragma: no cover
        return code
    lm = LANDMARK_BY_CODE.get(code)
    if lm is None:
        return code
    return lm.name_fr if lang == "fr" else (lm.name_en or lm.name_fr)


def _lm_hint(code: str, lang: str = "fr") -> str:
    """Indice de palpation d'un landmark (chaîne vide si indisponible)."""
    try:
        from .landmarks_extended import LANDMARK_BY_CODE
    except Exception:                                   # pragma: no cover
        return ""
    lm = LANDMARK_BY_CODE.get(code)
    if lm is None:
        return ""
    return lm.hint_fr if lang == "fr" else (lm.hint_en or lm.hint_fr)


def _q(name: str) -> str:
    """Nom d'objet du workspace, entre guillemets français (prose FR)."""
    return f"« {name} »"


def _qe(name: str) -> str:
    """Nom d'objet du workspace, entre guillemets anglais (prose EN)."""
    return f"“{name}”"


# ─────────────────────────────────────────────────────────────────────────────
# Dérivation : template → séquence pédagogique
# ─────────────────────────────────────────────────────────────────────────────

def template_to_steps(
    tpl: FrameTemplate,
    registry: dict[str, FrameTemplate] | None = None,
    auto: bool = False,
    upto_primary: bool = False,
) -> list[Step]:
    """Dérive la recette complète d'un :class:`FrameTemplate`.

    Parameters
    ----------
    tpl
        Le repère à construire.
    registry
        Index des templates, nécessaire dès qu'un axe est emprunté
        (:class:`Borrow`) à un autre segment.
    auto
        Marque toutes les étapes produites ``auto=True`` : le moteur les
        exécute sans intervention (utilisé pour les dépendances).
    upto_primary
        S'arrête après l'axe primaire (les étapes suffisantes pour qu'un autre
        repère puisse *emprunter* cet axe).

    La séquence produite est :

    1. les palpations (ordre de déclaration des ``points``) ;
    2. les points construits (milieux) ;
    3. l'axe **primaire** ;
    4. la direction **guide** (sauf si elle est empruntée : elle est alors
       préfixée en mode automatique au tout début) ;
    5. le produit vectoriel ``primaire × guide`` → axe **secondaire** ;
    6. le produit vectoriel ``primaire × secondaire`` → axe **guide
       orthogonalisé** ;
    7. la validation du repère.
    """
    registry = registry or {}
    steps: list[Step] = []

    # 0. Dépendance externe : l'axe guide est emprunté à un autre repère.
    #    On préfixe sa construction, en mode automatique (l'étudiant l'a déjà
    #    faite dans la recette du segment source).
    if isinstance(tpl.guide.expr, Borrow) and not upto_primary:
        steps.extend(_borrowed_steps(tpl, registry))

    # 1. Palpations, dans l'ordre de déclaration.
    for name, expr in tpl.points.items():
        if isinstance(expr, LM):
            steps.append(_pick_step(tpl, name, expr, auto))

    # 2. Points construits (milieux), dans l'ordre de déclaration.
    for name, expr in tpl.points.items():
        if isinstance(expr, Mid):
            steps.append(_midpoint_step(tpl, name, expr, auto))

    # 3. Axe primaire.
    steps.extend(_dir_steps(tpl, tpl.primary, "primary", auto))
    if upto_primary:
        return steps

    # 4. Direction guide (non orthogonale : elle place le plan du repère).
    if not isinstance(tpl.guide.expr, Borrow):
        steps.extend(_dir_steps(tpl, tpl.guide, "guide", auto))

    # 5. Axe secondaire = primaire × guide, dans l'ordre imposé par la
    #    convention directe.
    sec = remaining(tpl.primary.label, tpl.guide.label)
    first, second, _ = cross_order(tpl.primary.label, tpl.guide.label)
    ws = {tpl.primary.label: tpl.primary.ws_name(),
          tpl.guide.label: tpl.guide.ws_name()}
    steps.append(_cross_axis_step(
        tpl, sec, ws[first], ws[second], first, second, auto,
        tail_fr=(f"{sec} est ainsi exactement perpendiculaire à "
                 f"{tpl.primary.label}."),
        tail_en=(f"{sec} is then exactly perpendicular to "
                 f"{tpl.primary.label}."),
    ))

    # 6. Axe tertiaire : le guide, orthogonalisé par un second produit
    #    vectoriel (plutôt qu'une projection, pour rester calculable à la main).
    ter = tpl.guide.label
    f2, s2, _ = cross_order(tpl.primary.label, sec)
    ws2 = {tpl.primary.label: tpl.primary.ws_name(), sec: sec}
    steps.append(_cross_axis_step(
        tpl, ter, ws2[f2], ws2[s2], f2, s2, auto, orthogonalised=True,
        tail_fr=(f"{ter} est la version de {_q(tpl.guide.ws_name())} rendue "
                 f"perpendiculaire à {tpl.primary.label}."),
        tail_en=(f"{ter} is {_qe(tpl.guide.ws_name())} made perpendicular to "
                 f"{tpl.primary.label}."),
    ))

    # 7. Validation.
    steps.append(ValidateFrame(
        instruction_fr=(
            f"Valide le repère {tpl.label_fr} (origine {tpl.origin}, "
            f"X {tpl.word('X')}, Y {tpl.word('Y')}, Z {tpl.word('Z')})."
        ),
        instruction_en=(
            f"Validate the {tpl.label('en')} frame (origin {tpl.origin}, "
            f"X {tpl.word('X', 'en')}, Y {tpl.word('Y', 'en')}, "
            f"Z {tpl.word('Z', 'en')})."
        ),
        origin_name=tpl.origin,
        x_name="X", y_name="Y", z_name="Z",
        segment_key=tpl.key,
    ))
    return steps


# ── Sous-générateurs ─────────────────────────────────────────────────────────

def _borrowed_steps(
    tpl: FrameTemplate, registry: dict[str, FrameTemplate]
) -> list[Step]:
    """Étapes automatiques reconstruisant l'axe emprunté à un autre repère."""
    borrow = tpl.guide.expr
    assert isinstance(borrow, Borrow)
    try:
        dep = registry[borrow.segment]
    except KeyError:
        raise KeyError(
            f"{tpl.key} emprunte l'axe {borrow.axis} à {borrow.segment!r}, "
            "absent du registre de templates."
        ) from None
    if borrow.axis != dep.primary.label:
        raise NotImplementedError(
            f"{tpl.key} emprunte l'axe {borrow.axis} de {borrow.segment}, qui "
            f"n'est pas son axe primaire ({dep.primary.label}) : il faudrait "
            "déplier la construction complète du repère source."
        )
    steps = template_to_steps(dep, registry, auto=True, upto_primary=True)
    # La direction empruntée prend le nom attendu par le template courant.
    steps[-1].result_name = tpl.guide.ws_name()
    steps[-1].instruction_fr = (
        f"(auto) Axe {borrow.axis} {dep.label_fr}, emprunté comme direction de "
        f"référence → {_q(tpl.guide.ws_name())}."
    )
    steps[-1].instruction_en = (
        f"(auto) {dep.label('en').capitalize()} {borrow.axis} axis, borrowed as "
        f"the reference direction → {_qe(tpl.guide.ws_name())}."
    )
    steps[-1].hint_fr = (
        f"Ce repère n'a pas assez de landmarks palpables pour fixer son axe "
        f"{tpl.guide.label} : l'ISB l'emprunte au repère {dep.label_fr}."
    )
    steps[-1].hint_en = (
        f"This frame has too few palpable landmarks to fix its {tpl.guide.label} "
        f"axis: ISB borrows it from the {dep.label('en')} frame."
    )
    return steps


def _pick_step(
    tpl: FrameTemplate, name: str, expr: LM, auto: bool
) -> PickLandmark:
    """Palpation d'un landmark."""
    fr = f"Sélectionne : {_lm_name(expr.code, 'fr')} → {_q(name)}."
    en = f"Select: {_lm_name(expr.code, 'en')} → {_qe(name)}."
    if auto:
        fr = f"(auto) {_lm_name(expr.code, 'fr')} → {_q(name)}, repris du repère {tpl.label_fr}."
        en = f"(auto) {_lm_name(expr.code, 'en')} → {_qe(name)}, reused from the {tpl.label('en')} frame."
    return PickLandmark(
        instruction_fr=fr,
        instruction_en=en,
        expected_code=expr.code,
        result_name=name,
        tolerance_ok=list(expr.also),
        hint_fr=tpl.note(name, "fr") or _lm_hint(expr.code, "fr"),
        hint_en=tpl.note(name, "en") or _lm_hint(expr.code, "en"),
        auto=auto,
    )


def _midpoint_step(
    tpl: FrameTemplate, name: str, expr: Mid, auto: bool
) -> ComputeMidpoint:
    """Milieu de deux points du workspace."""
    fr = f"Calcule {_q(name)} = milieu de {_q(expr.a)} et {_q(expr.b)}."
    en = f"Compute {_qe(name)} = midpoint of {_qe(expr.a)} and {_qe(expr.b)}."
    if auto:
        fr = f"(auto) Milieu de {_q(expr.a)} et {_q(expr.b)} → {_q(name)}."
        en = f"(auto) Midpoint of {_qe(expr.a)} and {_qe(expr.b)} → {_qe(name)}."
    return ComputeMidpoint(
        instruction_fr=fr,
        instruction_en=en,
        expected_a=expr.a,
        expected_b=expr.b,
        result_name=name,
        hint_fr=tpl.note(name, "fr"),
        hint_en=tpl.note(name, "en"),
        auto=auto,
    )


def _dir_steps(
    tpl: FrameTemplate, spec: AxisSpec, role: str, auto: bool
) -> list[Step]:
    """Sous-étapes produisant la direction décrite par *spec*."""
    expr = spec.expr
    name = spec.ws_name()
    word = tpl.word(spec.label)
    word_en = tpl.word(spec.label, "en")

    if isinstance(expr, Seg):
        if role == "primary":
            fr = (f"Construis l'axe {spec.label} ({word}) : "
                  f"{_q(expr.frm)} → {_q(expr.to)}.")
            en = (f"Build the {spec.label} axis ({word_en}): "
                  f"{_qe(expr.frm)} → {_qe(expr.to)}.")
            hint_fr = (f"{spec.label} est l'axe PRIMAIRE du repère "
                       f"{tpl.label_fr} : il est porté directement par ce "
                       "vecteur, sans orthogonalisation.")
            hint_en = (f"{spec.label} is the PRIMARY axis of the "
                       f"{tpl.label('en')} frame: it lies directly along this "
                       "vector, with no orthogonalisation.")
        else:
            fr = f"Construis la direction guide {_q(name)} : {_q(expr.frm)} → {_q(expr.to)}."
            en = f"Build the guide direction {_qe(name)}: {_qe(expr.frm)} → {_qe(expr.to)}."
            hint_fr = (f"Elle pointe vers {word} mais n'est pas encore "
                       f"perpendiculaire à {tpl.primary.label}.")
            hint_en = (f"It points {word_en} but is not yet perpendicular to "
                       f"{tpl.primary.label}.")
        if auto:
            fr, en = f"(auto) {fr}", f"(auto) {en}"
        return [ComputeVector(
            instruction_fr=fr,
            instruction_en=en,
            expected_from=expr.frm,
            expected_to=expr.to,
            result_name=name,
            hint_fr=tpl.note(name, "fr") or hint_fr,
            hint_en=tpl.note(name, "en") or hint_en,
            auto=auto,
        )]

    if isinstance(expr, Normal):
        v1, v2 = f"V_{expr.p1}", f"V_{expr.p2}"
        plane = f"({expr.apex}, {expr.p1}, {expr.p2})"
        role_fr = (f"l'axe {spec.label} ({word})" if role == "primary"
                   else f"la direction guide {_q(name)}")
        role_en = (f"the {spec.label} axis ({word_en})" if role == "primary"
                   else f"the guide direction {_qe(name)}")
        steps: list[Step] = []
        for vname, target in ((v1, expr.p1), (v2, expr.p2)):
            fr = f"Construis {_q(vname)} : {_q(expr.apex)} → {_q(target)}."
            en = f"Build {_qe(vname)}: {_qe(expr.apex)} → {_qe(target)}."
            if auto:
                fr, en = f"(auto) {fr}", f"(auto) {en}"
            steps.append(ComputeVector(
                instruction_fr=fr,
                instruction_en=en,
                expected_from=expr.apex,
                expected_to=target,
                result_name=vname,
                hint_fr=f"Les deux vecteurs issus de {expr.apex} définissent le plan {plane}.",
                hint_en=f"Both vectors from {expr.apex} define the plane {plane}.",
                auto=auto,
            ))
        fr = (f"Calcule {role_fr} = {_q(v1)} × {_q(v2)}, normale au plan {plane}.")
        en = (f"Compute {role_en} = {_qe(v1)} × {_qe(v2)}, normal to the plane {plane}.")
        if auto:
            fr, en = f"(auto) {fr}", f"(auto) {en}"
        steps.append(ComputeCrossProduct(
            instruction_fr=fr,
            instruction_en=en,
            expected_a=v1,
            expected_b=v2,
            result_name=name,
            hint_fr=tpl.note(name, "fr") or (
                f"Cet ordre oriente la normale vers {word} ; l'ordre inverse la "
                "ferait pointer exactement à l'opposé."
            ),
            hint_en=tpl.note(name, "en") or (
                f"This order sends the normal {word_en}; the reverse order would "
                "make it point exactly the other way."
            ),
            auto=auto,
        ))
        return steps

    raise TypeError(f"expression de direction inconnue : {expr!r}")


def _cross_axis_step(
    tpl: FrameTemplate,
    result: str,
    a_name: str,
    b_name: str,
    a_axis: str,
    b_axis: str,
    auto: bool,
    *,
    orthogonalised: bool = False,
    tail_fr: str = "",
    tail_en: str = "",
) -> ComputeCrossProduct:
    """Produit vectoriel produisant un axe final du repère."""
    word = tpl.word(result)
    word_en = tpl.word(result, "en")
    qualif_fr = " orthogonalisé" if orthogonalised else ""
    qualif_en = "orthogonalised " if orthogonalised else ""
    fr = (f"Calcule l'axe {result}{qualif_fr} ({word}) = "
          f"{_q(a_name)} × {_q(b_name)}.")
    en = (f"Compute the {qualif_en}{result} axis ({word_en}) = "
          f"{_qe(a_name)} × {_qe(b_name)}.")
    if auto:
        fr, en = f"(auto) {fr}", f"(auto) {en}"
    mnemo_fr = (f"{tpl.word(a_axis)} × {tpl.word(b_axis)} = {word}.")
    mnemo_en = (f"{tpl.word(a_axis, 'en')} × {tpl.word(b_axis, 'en')} = {word_en}.")
    return ComputeCrossProduct(
        instruction_fr=fr,
        instruction_en=en,
        expected_a=a_name,
        expected_b=b_name,
        result_name=result,
        hint_fr=tpl.note(result, "fr") or f"{mnemo_fr} {tail_fr}".strip(),
        hint_en=tpl.note(result, "en") or f"{mnemo_en} {tail_en}".strip(),
        auto=auto,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Les 7 repères ISB
# ─────────────────────────────────────────────────────────────────────────────
# Correspondance des codes ISB → BonyLandmarks :
#
#   IJ  suprasternal_notch        C7  C7_spinous
#   PX  xiphoid_process           T8  T8_spinous
#   AA  acromial_angle_right      AI  scapula_inferior_angle_right
#   TS  scapula_trigonum_right    SC  sternoclavicular_joint_right
#   AC  acromioclavicular_joint_right
#   GH  greater_tubercle_right    (proxy palpable du centre gléno-huméral)
#   EL/EM  lateral|medial_epicondyle_right
#   HJC greater_trochanter_right  (proxy palpable du centre de hanche)
#   LC/MC  lateral|medial_knee_right
#   LM/MM  lateral|medial_malleolus_right
#   IC  milieu LC-MC              (proxy du point inter-condylaire tibial)

#: Vocabulaire du tronc (segments axiaux).
_TRUNK: dict[str, str] = {"X": "avant", "Y": "haut", "Z": "droite"}
#: Vocabulaire des membres (Y proximal, Z latéral du côté droit).
_LIMB: dict[str, str] = {"X": "avant", "Y": "proximal", "Z": "latéral"}


ISB_TEMPLATES: dict[str, FrameTemplate] = {
    "thorax": FrameTemplate(
        key="thorax",
        label_fr="thoracique", label_en="thoracic",
        name_fr="Thorax", name_en="Thorax",
        reference="Wu et al. 2005",
        points={
            "IJ": LM("suprasternal_notch"),
            "C7": LM("C7_spinous"),
            "PX": LM("xiphoid_process"),
            "T8": LM("T8_spinous"),
            "IJ_C7_mid": Mid("IJ", "C7"),
            "caudal": Mid("PX", "T8"),
        },
        origin="IJ",
        primary=AxisSpec("Y", Seg("caudal", "IJ_C7_mid")),
        guide=AxisSpec("Z", Normal("caudal", "IJ", "C7"), name="Z_raw"),
        words=_TRUNK,
        notes={
            "IJ_C7_mid": "Milieu crânial : il moyenne le sternum et le rachis, "
                         "ce qui rend l'axe long du thorax insensible à la cyphose locale.",
            "IJ_C7_mid@en": "Cranial midpoint: averaging sternum and spine makes the "
                            "thoracic long axis insensitive to local kyphosis.",
            "Z_raw": "V_IJ pointe vers l'avant-haut, V_C7 vers l'arrière-haut : leur "
                     "produit vectoriel sort du plan sagittal vers la droite du sujet.",
            "Z_raw@en": "V_IJ points up-forward, V_C7 up-backward: their cross product "
                        "leaves the sagittal plane towards the subject's right.",
        },
    ),
    "clavicle_right": FrameTemplate(
        key="clavicle_right",
        label_fr="claviculaire droit", label_en="right clavicular",
        name_fr="Clavicule D", name_en="Right clavicle",
        reference="Wu et al. 2005",
        points={
            "SC": LM("sternoclavicular_joint_right"),
            "AC": LM("acromioclavicular_joint_right"),
        },
        origin="SC",
        primary=AxisSpec("Z", Seg("SC", "AC")),
        guide=AxisSpec("Y", Borrow("thorax", "Y"), name="Y_thorax"),
        words=_LIMB,
        notes={
            "X": "haut × latéral = avant. La clavicule n'ayant que deux repères "
                 "palpables, sa rotation axiale est indéterminée : l'ISB lève "
                 "l'ambiguïté en empruntant la verticale au thorax.",
            "X@en": "up × lateral = forward. With only two palpable landmarks, the "
                    "clavicle's axial rotation is undetermined: ISB resolves it by "
                    "borrowing the vertical from the thorax.",
            "Y": "latéral × avant = proximal. Ce Y claviculaire diffère légèrement "
                 "du Y thoracique : il est rendu perpendiculaire à l'axe de la clavicule.",
            "Y@en": "lateral × forward = proximal. This clavicular Y differs slightly "
                    "from the thoracic Y: it is made perpendicular to the clavicle axis.",
        },
    ),
    "scapula_right": FrameTemplate(
        key="scapula_right",
        label_fr="scapulaire droit", label_en="right scapular",
        name_fr="Scapula D", name_en="Right scapula",
        reference="Wu et al. 2005",
        points={
            "AA": LM("acromial_angle_right", also=("acromion_right",)),
            "AI": LM("scapula_inferior_angle_right"),
            "TS": LM("scapula_trigonum_right"),
        },
        origin="AA",
        primary=AxisSpec("Z", Seg("TS", "AA")),
        guide=AxisSpec("X", Normal("AA", "AI", "TS"), name="X_raw"),
        words=_LIMB,
        notes={
            "Z": "Z est l'axe PRIMAIRE de la scapula : il suit l'épine scapulaire, "
                 "de médial (TS) vers latéral (AA).",
            "Z@en": "Z is the scapula PRIMARY axis: it follows the scapular spine, "
                    "medial (TS) to lateral (AA).",
            "X_raw": "Cet ordre fait sortir la normale vers l'AVANT du thorax (la "
                     "scapula est plaquée sur le gril costal) ; l'ordre inverse la "
                     "ferait pointer en arrière.",
            "X_raw@en": "This order makes the normal point ANTERIORLY (the scapula "
                        "lies on the rib cage); the reverse order would send it backwards.",
        },
    ),
    "humerus_right": FrameTemplate(
        key="humerus_right",
        label_fr="huméral droit", label_en="right humeral",
        name_fr="Humérus D", name_en="Right humerus",
        reference="Wu et al. 2005",
        points={
            "GH": LM("greater_tubercle_right"),
            "EL": LM("lateral_epicondyle_right"),
            "EM": LM("medial_epicondyle_right"),
            "elbow_mid": Mid("EL", "EM"),
        },
        origin="GH",
        primary=AxisSpec("Y", Seg("elbow_mid", "GH")),
        guide=AxisSpec("Z", Seg("EM", "EL"), name="EL_vec"),
        words=_LIMB,
        notes={
            "GH": "Proxy palpable du centre gléno-huméral, qui n'est pas accessible "
                  "à la palpation : juste en dehors et sous l'acromion, bras pendant. "
                  "L'ISB recommande une régression (Meskers 1998) ou les axes "
                  "hélicoïdaux (Stokdijk 2000).",
            "GH@en": "Palpable proxy for the glenohumeral centre, which cannot be "
                     "palpated: just lateral and below the acromion, arm hanging. ISB "
                     "recommends a regression (Meskers 1998) or helical axes "
                     "(Stokdijk 2000).",
            "EL_vec": "Ligne bi-épicondylienne, de médial vers latéral : elle place le "
                      "plan du coude mais n'est pas perpendiculaire à l'axe long.",
            "EL_vec@en": "Bi-epicondylar line, medial to lateral: it sets the elbow "
                         "plane but is not perpendicular to the long axis.",
            "X": "proximal × latéral = avant, pour le côté DROIT. X est la normale au "
                 "plan (GH, EL, EM).",
            "X@en": "proximal × lateral = forward, for the RIGHT side. X is the normal "
                    "to the plane (GH, EL, EM).",
        },
    ),
    "pelvis": FrameTemplate(
        key="pelvis",
        label_fr="pelvien", label_en="pelvic",
        name_fr="Bassin", name_en="Pelvis",
        reference="Wu et al. 2002",
        points={
            "ASIS_R": LM("ASIS_right"),
            "ASIS_L": LM("ASIS_left"),
            "PSIS_R": LM("PSIS_right"),
            "PSIS_L": LM("PSIS_left"),
            "O": Mid("ASIS_R", "ASIS_L"),
            "PSIS_mid": Mid("PSIS_R", "PSIS_L"),
        },
        origin="O",
        primary=AxisSpec("Z", Seg("ASIS_L", "ASIS_R")),
        guide=AxisSpec("X", Seg("PSIS_mid", "O"), name="AP_raw"),
        words=_TRUNK,
        notes={
            "O": "L'ISB place l'origine au centre articulaire de hanche, non "
                 "palpable : le milieu des EIAS en est le substitut pratique.",
            "O@en": "ISB places the origin at the hip joint centre, which is not "
                    "palpable: the ASIS midpoint is the practical substitute.",
            "Z": "Z est l'axe PRIMAIRE du bassin : il est directement porté par la "
                 "ligne bi-EIAS, de la gauche vers la droite du sujet.",
            "Z@en": "Z is the pelvis PRIMARY axis: it lies directly along the bi-ASIS "
                    "line, from the subject's left to their right.",
        },
    ),
    "femur_right": FrameTemplate(
        key="femur_right",
        label_fr="fémoral droit", label_en="right femoral",
        name_fr="Fémur D", name_en="Right femur",
        reference="Wu et al. 2002",
        points={
            "GT": LM("greater_trochanter_right"),
            "lat_knee": LM("lateral_knee_right"),
            "med_knee": LM("medial_knee_right"),
            "knee_mid": Mid("lat_knee", "med_knee"),
        },
        origin="GT",
        primary=AxisSpec("Y", Seg("knee_mid", "GT")),
        guide=AxisSpec("Z", Seg("med_knee", "lat_knee"), name="V_lat"),
        words=_LIMB,
        notes={
            "GT": "Proxy palpable du centre articulaire de hanche : la large saillie "
                  "latérale de la hanche, qui roule sous les doigts en rotation "
                  "interne/externe de la cuisse.",
            "GT@en": "Palpable proxy for the hip joint centre: the broad lateral bump "
                     "of the hip, rolling under the fingers during internal/external "
                     "thigh rotation.",
            "X": "proximal × latéral = avant, pour le côté DROIT. X est la normale au "
                 "plan (GT, épicondyle latéral, épicondyle médial).",
            "X@en": "proximal × lateral = forward, for the RIGHT side. X is the normal "
                    "to the plane (GT, lateral epicondyle, medial epicondyle).",
        },
    ),
    "tibia_right": FrameTemplate(
        key="tibia_right",
        label_fr="tibial droit", label_en="right tibial",
        name_fr="Tibia D", name_en="Right tibia",
        reference="Wu et al. 2002",
        points={
            "LM_": LM("lateral_malleolus_right"),
            "MM": LM("medial_malleolus_right"),
            "LC": LM("lateral_knee_right"),
            "MC": LM("medial_knee_right"),
            "IM": Mid("LM_", "MM"),
            "IC": Mid("LC", "MC"),
        },
        origin="IM",
        primary=AxisSpec("Z", Seg("MM", "LM_")),
        guide=AxisSpec("Y", Seg("IM", "IC"), name="V_IC"),
        words=_LIMB,
        notes={
            "LC": "Wu 2002 définit IC à partir des bords des condyles TIBIAUX ; on "
                  "utilise ici les épicondyles fémoraux comme proxy.",
            "LC@en": "Wu 2002 defines IC from the TIBIAL condylar borders; the femoral "
                     "epicondyles are used here as a proxy.",
            "IM": "Point inter-malléolaire : l'origine du repère tibial.",
            "IM@en": "Inter-malleolar point: the origin of the tibial frame.",
            "Z": "Z est l'axe PRIMAIRE du tibia selon Wu 2002 : la ligne malléolaire "
                 "elle-même, et non une composante orthogonalisée de l'axe long.",
            "Z@en": "Z is the tibia PRIMARY axis per Wu 2002: the malleolar line "
                    "itself, not an orthogonalised component of the long axis.",
            "Y": "latéral × avant = proximal. Attention : Y n'est PAS exactement "
                 "IM → IC, car Z est l'axe primaire et Y lui est rendu perpendiculaire.",
            "Y@en": "lateral × forward = proximal. Note: Y is NOT exactly IM → IC, "
                    "because Z is the primary axis and Y is made perpendicular to it.",
        },
    ),
}
