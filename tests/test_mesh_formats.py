"""Tests du format PLY des meshes BodyParts3D et des scripts de distribution.

Aucun test ne dépend des vrais meshes (non versionnés) : de mini-meshes sont
générés dans un dossier temporaire créé via ``tempfile.mkdtemp`` (le fixture
``tmp_path`` de pytest est refusé sous Windows dans cet environnement).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _meshio  # noqa: E402
import convert_meshes  # noqa: E402
import fetch_meshes  # noqa: E402
import pack_meshes  # noqa: E402

from bonylandmarks import anatomy_catalog as cat  # noqa: E402
from bonylandmarks import structure_scene as sc  # noqa: E402
from bonylandmarks.anatomy_catalog import Structure  # noqa: E402
from bonylandmarks.bone_map import find_bone_mesh  # noqa: E402

# Coordonnées volontairement grandes et non centrées (repère BodyParts3D).
_VERTS = np.array([[-62.221, -126.836, 1565.28], [-61.5, -124.9, 1562.18],
                   [-60.1, -128.7, 1567.72], [-63.0, -125.0, 1570.0]])
_FACES = np.array([[0, 1, 2], [0, 2, 3], [1, 3, 2]])


@pytest.fixture()
def workdir():
    d = Path(tempfile.mkdtemp(prefix="bony_test_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write_obj(path: Path, verts=_VERTS, faces=_FACES) -> None:
    lines = ["# Merged BodyParts3D mesh - CC BY-SA 2.1 JP (DBCLS)", ""]
    lines += [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─── Écriture / lecture PLY ───────────────────────────────────────────────────


def test_ply_roundtrip_pyvista(workdir):
    p = workdir / "m.ply"
    _meshio.write_ply(p, _VERTS, _FACES, ["BodyParts3D test"])
    assert p.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\n")
    m = pv.read(str(p))
    assert m.n_points == len(_VERTS) and m.n_cells == len(_FACES)
    assert m.is_all_triangles
    np.testing.assert_allclose(m.points, _VERTS, atol=1e-3)  # float32, pas de recentrage
    np.testing.assert_allclose(m.bounds, [_VERTS[:, 0].min(), _VERTS[:, 0].max(),
                                          _VERTS[:, 1].min(), _VERTS[:, 1].max(),
                                          _VERTS[:, 2].min(), _VERTS[:, 2].max()], atol=1e-3)
    assert "Normals" not in m.point_data


def test_convert_one_obj_to_ply_identical_geometry(workdir):
    obj = workdir / "bone.obj"
    _write_obj(obj)
    ply = obj.with_suffix(".ply")
    assert convert_meshes.convert_one(obj, ply) == (len(_VERTS), len(_FACES))
    a, b = pv.read(str(obj)), pv.read(str(ply))
    assert (a.n_points, a.n_cells) == (b.n_points, b.n_cells)
    np.testing.assert_allclose(a.bounds, b.bounds, atol=1e-3)
    assert obj.is_file()  # l'OBJ source n'est pas supprimé


def test_read_obj_handles_quads_and_negative_indices(workdir):
    obj = workdir / "q.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1//1 2//2 3//3 4//4\nf -4 -3 -2\n",
                   encoding="utf-8")
    v, f, _ = _meshio.read_obj(obj)
    assert v.shape == (4, 3) and f.shape == (3, 3)  # quad -> 2 triangles + 1 triangle


# ─── Chargeurs ────────────────────────────────────────────────────────────────


def test_find_bone_mesh_prefers_ply(workdir):
    assert find_bone_mesh(workdir, "skull") is None
    (workdir / "skull.obj").write_text("v 0 0 0\n")
    assert find_bone_mesh(workdir, "skull").suffix == ".obj"
    (workdir / "skull.stl").write_bytes(b"x")
    assert find_bone_mesh(workdir, "skull").suffix == ".stl"
    _meshio.write_ply(workdir / "skull.ply", _VERTS, _FACES)
    assert find_bone_mesh(workdir, "skull").suffix == ".ply"


def test_structure_scene_reads_ply(workdir, monkeypatch):
    _meshio.write_ply(workdir / "bones_full" / "x_left.ply", _VERTS, _FACES)
    monkeypatch.setattr(cat, "DATA_DIR", workdir)
    s = Structure(id="x_left", kind="bone", fma_id="FMA0", name_fr="x", name_en="x",
                  side="left", region="skull", tier=1, mesh_file="bones_full/x_left.ply")
    sc.clear_cache()
    try:
        m = sc.load_structure_mesh(s)
    finally:
        sc.clear_cache()
    assert m is not None and len(m.points) == len(_VERTS) and len(m.tris) == len(_FACES)
    np.testing.assert_allclose(m.points, _VERTS, atol=1e-3)


def test_catalog_mesh_path_ply_and_legacy_obj_fallback(workdir, monkeypatch):
    monkeypatch.setattr(cat, "DATA_DIR", workdir)
    s = Structure(id="y", kind="bone", fma_id="FMA0", name_fr="y", name_en="y",
                  side="mid", region="skull", tier=1, mesh_file="bones_full/y.ply")
    assert cat.mesh_path(s) is None
    (workdir / "bones_full").mkdir()
    _write_obj(workdir / "bones_full" / "y.obj")
    assert cat.mesh_path(s) == workdir / "bones_full" / "y.obj"  # ancien OBJ local
    _meshio.write_ply(workdir / "bones_full" / "y.ply", _VERTS, _FACES)
    assert cat.mesh_path(s) == workdir / "bones_full" / "y.ply"


# ─── Distribution : pack -> fetch ─────────────────────────────────────────────


def _fake_data(root: Path) -> None:
    for sub in ("bones", "bones_full", "muscles"):
        _meshio.write_ply(root / sub / f"{sub}_a.ply", _VERTS + len(sub), _FACES)


def test_pack_then_fetch_roundtrip_and_idempotence(workdir):
    data = workdir / "data"
    _fake_data(data)
    z = workdir / "out" / "bonylandmarks-meshes-v1.zip"
    manifest = pack_meshes.build_zip(data, z)
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
    assert {"MESHES_LICENSE.txt", "MANIFEST.json", "bones/bones_a.ply",
            "bones_full/bones_full_a.ply", "muscles/muscles_a.ply"} == names
    assert len(manifest["files"]) == 3
    assert "CC BY-SA" in zipfile.ZipFile(z).read("MESHES_LICENSE.txt").decode()

    dest = workdir / "dest"
    assert fetch_meshes.main(["--zip", str(z), "--dest", str(dest)]) == 0
    assert (dest / "muscles" / "muscles_a.ply").read_bytes() == \
        (data / "muscles" / "muscles_a.ply").read_bytes()
    assert (dest / "MESHES_LICENSE.txt").is_file()
    assert fetch_meshes.is_up_to_date(dest, "meshes-v1")
    assert fetch_meshes.install_zip(z, dest) == (0, 3)  # idempotent

    (dest / "bones" / "bones_a.ply").write_bytes(b"corrompu")
    assert not fetch_meshes.is_up_to_date(dest, "meshes-v1")
    assert fetch_meshes.install_zip(z, dest) == (1, 2)  # réparé


def test_fetch_rejects_tampered_zip(workdir):
    data = workdir / "data"
    _fake_data(data)
    z = workdir / "ok.zip"
    pack_meshes.build_zip(data, z)
    bad = workdir / "bad.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(bad, "w") as dst:
        for name in src.namelist():
            payload = src.read(name)
            if name == "bones/bones_a.ply":
                payload = payload[:-1] + b"\x00" if payload[-1] != 0 else payload[:-1] + b"\x01"
            dst.writestr(name, payload)
    with pytest.raises(SystemExit):
        fetch_meshes.install_zip(bad, workdir / "dest")
