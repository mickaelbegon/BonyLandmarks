"""Catalogue des structures anatomiques (os et muscles) BodyParts3D.

Module pur Python (sans Qt ni pyvista).  Les données vivent dans deux fichiers
JSON sous ``data/`` :

* ``anatomy_bones.json``   — os individuels (meshes dans ``data/bones_full/``)
* ``anatomy_muscles.json`` — muscles (meshes dans ``data/muscles/``), optionnel

Chaque enregistrement suit le schéma ::

    {"id": "femur_left", "kind": "bone", "fma_id": "FMA24474",
     "name_fr": "Fémur gauche", "name_en": "Left femur",
     "side": "left", "region": "lower_limb", "tier": 1,
     "synonyms_fr": ["os de la cuisse"], "synonyms_en": [],
     "mesh_file": "bones_full/femur_left.ply"}

Format des meshes : PLY binaire (``.ply``), récupérés par
``python scripts/fetch_meshes.py`` ou régénérés par ``scripts/import_bp3d.py``.
Si le ``.ply`` est absent, un ``.stl``/``.obj`` de même nom est accepté.

Champs optionnels : ``layer`` (muscles : couche/plan, "" par défaut) et
``src_files`` (noms de fichiers sources BodyParts3D, sans extension, ex.
``["FJ3266"]``, pour lever une ambiguïté quand plusieurs meshes partagent le
même identifiant FMA).

Repère de coordonnées des meshes : celui de BodyParts3D (mm, X gauche = +X,
Y antérieur = -Y, Z supérieur = +Z), conservé tel quel — les structures sont
donc à leur place relative dans le corps.

Attribution : BodyParts3D, © The Database Center for Life Science, licensed
under CC Attribution-Share Alike 2.1 Japan.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

DATA_DIR: Path = Path(__file__).parent / "data"
BONES_JSON: Path = DATA_DIR / "anatomy_bones.json"
MUSCLES_JSON: Path = DATA_DIR / "anatomy_muscles.json"

KINDS: tuple[str, ...] = ("bone", "muscle")
SIDES: tuple[str, ...] = ("left", "right", "mid")
BONE_REGIONS: tuple[str, ...] = (
    "skull", "thorax", "spine", "upper_limb", "pelvis", "lower_limb",
)

_JSON_BY_KIND: dict[str, Path] = {"bone": BONES_JSON, "muscle": MUSCLES_JSON}


@dataclass(frozen=True)
class Structure:
    """Une structure anatomique identifiable (os ou muscle)."""

    id: str
    kind: str                      # "bone" | "muscle"
    fma_id: str                    # ex. "FMA24474"
    name_fr: str
    name_en: str
    side: str                      # "left" | "right" | "mid"
    region: str
    tier: int                      # 1 = majeur, 2 = détaillé
    synonyms_fr: tuple[str, ...] = ()
    synonyms_en: tuple[str, ...] = ()
    mesh_file: str = ""            # relatif à data/, ex. "bones_full/femur_left.ply"
    layer: str = ""                # muscles : couche / plan (optionnel)
    src_files: tuple[str, ...] = ()  # fichiers BP3D sources (optionnel)


_FIELD_NAMES = {f.name for f in fields(Structure)}
_TUPLE_FIELDS = ("synonyms_fr", "synonyms_en", "src_files")


def _from_dict(d: dict) -> Structure:
    kw = {k: v for k, v in d.items() if k in _FIELD_NAMES}  # clés inconnues ignorées
    for name in _TUPLE_FIELDS:
        if name in kw:
            kw[name] = tuple(kw[name] or ())
    kw["tier"] = int(kw.get("tier", 1))
    return Structure(**kw)


def _load_json(path: Path) -> list[Structure]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[Structure] = []
    for d in raw:
        try:
            out.append(_from_dict(d))
        except (TypeError, KeyError):
            continue  # entrée malformée : ignorée
    return out


@lru_cache(maxsize=1)
def _catalog() -> tuple[tuple[Structure, ...], dict[str, Structure]]:
    items: list[Structure] = []
    by_id: dict[str, Structure] = {}
    for kind in KINDS:
        for s in _load_json(_JSON_BY_KIND[kind]):
            if s.id in by_id:
                continue  # premier gagnant, pas de doublon d'id
            by_id[s.id] = s
            items.append(s)
    return tuple(items), by_id


def reload() -> None:
    """Vide le cache (après modification des JSON)."""
    _catalog.cache_clear()


def all_structures(kind: str | None = None) -> list[Structure]:
    """Toutes les structures du catalogue (meshes présents ou non)."""
    items, _ = _catalog()
    if kind is None:
        return list(items)
    return [s for s in items if s.kind == kind]


def get(id: str) -> Structure:
    """Structure par identifiant ; ``KeyError`` si absente."""
    _, by_id = _catalog()
    return by_id[id]


_MESH_EXTS = (".ply", ".stl", ".obj")


def mesh_path(s: Structure) -> Path | None:
    """Chemin absolu du mesh, ou ``None`` s'il n'est pas (encore) importé."""
    if not s.mesh_file:
        return None
    p = DATA_DIR / s.mesh_file
    if p.is_file():
        return p
    for ext in _MESH_EXTS:  # même nom, autre format (ex. anciens OBJ locaux)
        alt = p.with_suffix(ext)
        if alt.is_file():
            return alt
    return None


def available(kind: str | None = None, region: str | None = None,
              tier_max: int | None = None) -> list[Structure]:
    """Structures dont le mesh existe, filtrées par type/région/niveau."""
    out = []
    for s in all_structures(kind):
        if region is not None and s.region != region:
            continue
        if tier_max is not None and s.tier > tier_max:
            continue
        if mesh_path(s) is None:
            continue
        out.append(s)
    return out


def regions(kind: str | None = None) -> list[str]:
    """Régions présentes dans le catalogue (ordre de première apparition)."""
    seen: list[str] = []
    for s in all_structures(kind):
        if s.region not in seen:
            seen.append(s.region)
    return seen


def display_name(s: Structure, lang: str = "fr") -> str:
    """Nom affichable dans la langue demandée ("fr" ou "en"), avec repli."""
    if lang == "en":
        return s.name_en or s.name_fr
    return s.name_fr or s.name_en
