"""Tests du catalogue anatomique (os / muscles BodyParts3D).

Ne dépendent pas de la présence des meshes (gitignorés).
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from bonylandmarks import anatomy_catalog as ac
from bonylandmarks.anatomy_catalog import Structure

VALID_SIDES = {"left", "right", "mid"}
VALID_BONE_REGIONS = set(ac.BONE_REGIONS)
REQUIRED = ("id", "kind", "fma_id", "name_fr", "name_en", "side", "region",
            "tier", "synonyms_fr", "synonyms_en", "mesh_file")


def _raw(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def tmp_path():
    """Dossier temporaire (évite le tmp_path de pytest, capricieux sous Windows)."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="module")
def bones() -> list[Structure]:
    return ac.all_structures("bone")


def test_bones_json_exists_and_reasonable_size(bones):
    assert ac.BONES_JSON.is_file()
    assert 150 <= len(bones) <= 400


def test_raw_bone_records_have_full_schema():
    for d in _raw(ac.BONES_JSON):
        for key in REQUIRED:
            assert key in d, f"{d.get('id')}: clé manquante {key}"
        assert isinstance(d["synonyms_fr"], list)
        assert isinstance(d["synonyms_en"], list)
        assert isinstance(d["tier"], int)


def test_ids_unique_across_all_kinds():
    ids = [s.id for s in ac.all_structures()]
    assert len(ids) == len(set(ids))
    raw_ids = [d["id"] for k in ac.KINDS for d in _raw(ac._JSON_BY_KIND[k])]
    assert len(raw_ids) == len(set(raw_ids)), "id dupliqué dans les JSON"


def test_ids_are_ascii_snake_case():
    for s in ac.all_structures():
        assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", s.id), s.id


def test_bone_fields_valid(bones):
    for s in bones:
        assert s.kind == "bone"
        assert re.fullmatch(r"FMA\d+", s.fma_id), s.id
        assert s.side in VALID_SIDES, s.id
        assert s.region in VALID_BONE_REGIONS, s.id
        assert s.tier in (1, 2), s.id
        assert s.name_fr.strip(), s.id
        assert s.name_en.strip(), s.id
        assert s.mesh_file == f"bones_full/{s.id}.obj", s.id
        assert all(isinstance(x, str) and x.strip() for x in s.synonyms_fr + s.synonyms_en)


def test_other_kinds_minimal_validity():
    """Les muscles (s'ils existent) respectent au moins le schéma de base."""
    for s in ac.all_structures("muscle"):
        assert s.kind == "muscle"
        assert s.side in VALID_SIDES, s.id
        assert s.region.strip(), s.id
        assert s.tier >= 1, s.id
        assert s.name_fr.strip(), s.id
        assert isinstance(s.layer, str)


def test_side_consistent_with_id_suffix():
    for s in ac.all_structures():
        if s.id.endswith("_left"):
            assert s.side == "left", s.id
        elif s.id.endswith("_right"):
            assert s.side == "right", s.id
        else:
            assert s.side == "mid", s.id


def test_side_consistent_with_french_name(bones):
    for s in bones:
        fr = s.name_fr.lower()
        if s.side == "left":
            assert fr.endswith("gauche"), s.id
            assert s.name_en.startswith("Left"), s.id
        elif s.side == "right":
            assert fr.endswith("droite"), s.id
            assert s.name_en.startswith("Right"), s.id
        else:
            assert "gauche" not in fr and "droite" not in fr, s.id


def test_left_right_pairs_are_complete(bones):
    ids = {s.id for s in bones}
    for s in bones:
        if s.side == "left":
            assert s.id[:-5] + "_right" in ids, s.id
        elif s.side == "right":
            assert s.id[:-6] + "_left" in ids, s.id


def test_expected_major_bones_present(bones):
    ids = {s.id for s in bones}
    for must in ("femur_left", "femur_right", "tibia_left", "fibula_right", "patella_left",
                 "humerus_right", "scapula_left", "clavicle_right", "radius_left", "ulna_right",
                 "mandible", "sacrum", "hip_bone_left", "vertebra_l5", "vertebra_t12",
                 "atlas", "axis", "rib_1_left", "rib_12_right", "calcaneus_left", "talus_right"):
        assert must in ids, must


def test_counts_per_region_and_tier(bones):
    regions_found = {s.region for s in bones}
    assert regions_found == VALID_BONE_REGIONS
    tier1 = [s for s in bones if s.tier == 1]
    assert len(tier1) >= 50
    assert len([s for s in bones if s.tier == 2]) >= 50
    assert sum(1 for s in bones if s.id.startswith("rib_")) == 24
    assert sum(1 for s in bones if s.id.startswith("vertebra_")) == 22  # C3-C7, T1-T12, L1-L5


def test_common_synonyms_present():
    syn = lambda i: set(ac.get(i).synonyms_fr)  # noqa: E731
    assert "péroné" in syn("fibula_left")
    assert "omoplate" in syn("scapula_right")
    assert "rotule" in syn("patella_left")
    assert "astragale" in syn("talus_left")


def test_get_and_keyerror():
    s = ac.get("femur_left")
    assert isinstance(s, Structure)
    assert s.side == "left" and s.region == "lower_limb"
    with pytest.raises(KeyError):
        ac.get("does_not_exist")


def test_structure_is_frozen_with_defaults():
    s = ac.get("femur_left")
    with pytest.raises(Exception):
        s.id = "x"  # type: ignore[misc]
    assert s.layer == ""
    bare = Structure(id="t", kind="muscle", fma_id="FMA1", name_fr="T", name_en="T",
                     side="mid", region="trunk", tier=1)
    assert bare.synonyms_fr == () and bare.mesh_file == "" and bare.layer == ""


def test_all_structures_filter_by_kind():
    assert all(s.kind == "bone" for s in ac.all_structures("bone"))
    assert len(ac.all_structures()) >= len(ac.all_structures("bone"))
    assert ac.all_structures("nonexistent-kind") == []


def test_regions_and_display_name():
    assert set(ac.regions("bone")) == VALID_BONE_REGIONS
    s = ac.get("femur_left")
    assert ac.display_name(s, "fr") == "Fémur gauche"
    assert ac.display_name(s, "en") == "Left femur"
    assert ac.display_name(s) == s.name_fr


def test_mesh_path_none_when_missing():
    ghost = Structure(id="ghost", kind="bone", fma_id="FMA0", name_fr="x", name_en="x",
                      side="mid", region="skull", tier=1, mesh_file="bones_full/__nope__.obj")
    assert ac.mesh_path(ghost) is None
    assert ac.mesh_path(Structure(id="g2", kind="bone", fma_id="FMA0", name_fr="x",
                                  name_en="x", side="mid", region="skull", tier=1)) is None


def test_available_is_subset_and_filters_work():
    everything = {s.id for s in ac.all_structures("bone")}
    avail = ac.available("bone")
    assert {s.id for s in avail} <= everything
    for s in avail:
        p = ac.mesh_path(s)
        assert p is not None and p.is_file() and p.is_absolute()
    assert all(s.region == "lower_limb" for s in ac.available("bone", region="lower_limb"))
    assert all(s.tier == 1 for s in ac.available("bone", tier_max=1))
    assert ac.available("bone", region="not-a-region") == []


def test_available_tolerates_missing_meshes(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "DATA_DIR", tmp_path)  # aucun mesh sous tmp_path
    assert ac.available("bone") == []
    assert len(ac.all_structures("bone")) > 0      # le catalogue reste lisible


def test_loader_tolerates_missing_and_broken_json(monkeypatch, tmp_path):
    good = {"id": "a_left", "kind": "bone", "fma_id": "FMA1", "name_fr": "A gauche",
            "name_en": "Left a", "side": "left", "region": "skull", "tier": 1,
            "synonyms_fr": ["aa"], "synonyms_en": [], "mesh_file": "bones_full/a_left.obj",
            "unknown_key": 42}
    (tmp_path / "anatomy_bones.json").write_text(
        json.dumps([good, {"id": "broken"}]), encoding="utf-8")
    # muscles : fichier absent ; os : une entrée valide + une malformée
    monkeypatch.setattr(ac, "BONES_JSON", tmp_path / "anatomy_bones.json")
    monkeypatch.setattr(ac, "MUSCLES_JSON", tmp_path / "anatomy_muscles.json")
    monkeypatch.setattr(ac, "_JSON_BY_KIND", {"bone": ac.BONES_JSON, "muscle": ac.MUSCLES_JSON})
    ac.reload()
    try:
        assert [s.id for s in ac.all_structures()] == ["a_left"]
        assert ac.get("a_left").synonyms_fr == ("aa",)
        (tmp_path / "anatomy_muscles.json").write_text("{pas du json", encoding="utf-8")
        ac.reload()
        assert [s.id for s in ac.all_structures()] == ["a_left"]
    finally:
        monkeypatch.undo()
        ac.reload()
    assert len(ac.all_structures("bone")) > 100
