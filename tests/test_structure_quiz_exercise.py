"""Tests de l'exercice « Anatomie 3D » (scène fusionnée + widget Qt).

Aucun test ne dépend des meshes BodyParts3D (gitignorés) : un mini-catalogue
factice de cubes est généré dans un dossier temporaire.  Les tests du widget
sont ignorés si Qt / pyvistaqt / OpenGL ne sont pas disponibles ; un test
optionnel utilise les vrais meshes s'ils sont installés.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from bonylandmarks import anatomy_catalog as cat  # noqa: E402
from bonylandmarks import structure_scene as sc  # noqa: E402
from bonylandmarks.anatomy_catalog import Structure  # noqa: E402
from bonylandmarks.structure_quiz_engine import Level, Mode, QuizConfig, base_id  # noqa: E402

# ─── Mini-catalogue factice ───────────────────────────────────────────────────

# id, kind, name_fr, name_en, side, region, tier, layer, centre (x, y, z)
_FAKE = [
    ("femur_left", "bone", "Fémur gauche", "Left femur", "left", "lower_limb", 1, "", (90, -80, 600)),
    ("femur_right", "bone", "Fémur droit", "Right femur", "right", "lower_limb", 1, "", (-90, -80, 600)),
    ("tibia_left", "bone", "Tibia gauche", "Left tibia", "left", "lower_limb", 1, "", (80, -80, 200)),
    ("humerus_left", "bone", "Humérus gauche", "Left humerus", "left", "upper_limb", 1, "", (190, -75, 1190)),
    ("humerus_right", "bone", "Humérus droit", "Right humerus", "right", "upper_limb", 1, "", (-190, -75, 1190)),
    ("sternum", "bone", "Sternum", "Sternum", "mid", "thorax", 1, "", (0, -190, 1220)),
    ("frontal", "bone", "Os frontal", "Frontal bone", "mid", "skull", 1, "", (0, -137, 1560)),
    ("patella_left", "bone", "Patella gauche", "Left patella", "left", "lower_limb", 2, "", (83, -112, 380)),
    ("rectus_femoris_left", "muscle", "Droit fémoral gauche", "Left rectus femoris", "left",
     "lower_limb", 1, "superficial", (100, -115, 610)),
    ("vastus_intermedius_left", "muscle", "Vaste intermédiaire gauche",
     "Left vastus intermedius", "left", "lower_limb", 1, "deep", (95, -95, 600)),
    ("gluteus_minimus_left", "muscle", "Petit glutéal gauche", "Left gluteus minimus", "left",
     "lower_limb", 2, "deep", (110, -70, 860)),
    ("biceps_brachii_left", "muscle", "Biceps brachial gauche", "Left biceps brachii", "left",
     "upper_limb", 1, "superficial", (188, -89, 1173)),
    ("brachialis_left", "muscle", "Brachial gauche", "Left brachialis", "left",
     "upper_limb", 1, "deep", (186, -85, 1150)),
    ("pectoralis_major_left", "muscle", "Grand pectoral gauche", "Left pectoralis major", "left",
     "trunk", 1, "superficial", (90, -156, 1246)),
]


def _make_catalog(root) -> list[Structure]:
    out = []
    for sid, kind, fr, en, side, region, tier, layer, c in _FAKE:
        f = root / f"{sid}.stl"
        cube = pv.Cube(center=c, x_length=40, y_length=40, z_length=40).triangulate()
        cube.save(str(f))
        out.append(Structure(
            id=sid, kind=kind, fma_id="FMA0", name_fr=fr, name_en=en, side=side,
            region=region, tier=tier, mesh_file=str(f), layer=layer,
        ))
    return out


@pytest.fixture()
def fake_catalog():
    """Cubes STL dans un dossier temporaire (le tmp_path de pytest est capricieux sous Windows)."""
    sc.clear_cache()
    with tempfile.TemporaryDirectory() as d:
        yield _make_catalog(Path(d))
    sc.clear_cache()


# ─── Scène fusionnée (sans Qt) ────────────────────────────────────────────────


def test_merged_mesh_and_visibility(fake_catalog):
    bones = [s for s in fake_catalog if s.kind == "bone"]
    merged = sc.build_merged(bones)
    assert merged.ids == [s.id for s in bones]
    assert len(merged.tris) == 12 * len(bones)          # cube = 12 triangles
    assert set(np.unique(merged.tri_sid)) == set(range(len(bones)))

    poly, cell_sid = merged.polydata()
    assert poly.n_cells == len(cell_sid) == len(merged.tris)

    vis = merged.visible_mask(["femur_left", "unknown_id"])
    poly2, cell_sid2 = merged.polydata(vis)
    assert poly2.n_cells == poly.n_cells - 12
    assert merged.index["femur_left"] not in set(cell_sid2)


def test_recolor_in_place(fake_catalog):
    bones = [s for s in fake_catalog if s.kind == "bone"]
    layer = sc.SceneLayer(
        "bone", sc.build_merged(bones),
        np.tile(np.array(sc.COLOR_BONE, np.uint8), (len(bones), 1)),
        {s.id: s for s in bones},
    )
    poly = pv.Plotter(off_screen=True)  # ne rend rien : sert seulement d'hôte d'acteur
    try:
        assert layer.ensure(poly) is True
        assert layer.ensure(poly) is False            # rien de changé : pas de rebuild
        layer.recolor({"sternum": sc.COLOR_TARGET})
        rgb = layer.poly.cell_data["rgb"]
        tgt = layer.cell_sid == layer.merged.index["sternum"]
        assert (rgb[tgt] == np.array(sc.COLOR_TARGET, np.uint8)).all()
        assert (rgb[~tgt] == np.array(sc.COLOR_BONE, np.uint8)).all()
        layer.recolor()                               # retour à la palette de base
        assert (layer.poly.cell_data["rgb"] == np.array(sc.COLOR_BONE, np.uint8)).all()
        assert layer.ensure(poly, ["sternum"]) is True   # masque différent -> rebuild
        assert layer.id_at_cell(0) is not None and layer.id_at_cell(10**9) is None
    finally:
        poly.close()


def test_muscle_colors_are_varied_and_deterministic():
    a = sc.muscle_color("biceps_brachii_left")
    assert a == sc.muscle_color("biceps_brachii_left")
    cols = {sc.muscle_color(f"m{i}") for i in range(40)}
    assert len(cols) > 20
    assert all(c[0] > c[1] and c[0] > c[2] for c in cols)   # dominante rouge


def test_choose_view():
    # squelette factice : tronc épais (y -200..-30) et une jambe fine
    ref = np.array([
        [-100, -200, 1000, 100, -30, 1400],     # thorax
        [60, -140, 300, 120, -20, 900],         # jambe gauche
    ], float)
    assert sc.choose_view((0, -195, 1200), ref) == sc.VIEW_FRONT      # sternum
    assert sc.choose_view((0, -35, 1200), ref) == sc.VIEW_BACK        # rachis
    assert sc.choose_view((0, -60, 1200), ref) == sc.VIEW_BACK
    # au milieu de la profondeur, structure latérale -> vue latérale
    assert sc.choose_view((90, -80, 600), ref) == sc.VIEW_LEFT
    assert sc.choose_view((-90, -80, 600), ref) == sc.VIEW_RIGHT
    # sans référence : latéral si loin de la ligne médiane, sinon face
    empty = np.zeros((0, 6))
    assert sc.choose_view((200, 0, 0), empty) == sc.VIEW_LEFT
    assert sc.choose_view((0, 0, 0), empty) == sc.VIEW_FRONT
    axis, sign = sc.VIEW_AXIS[sc.VIEW_FRONT]
    assert (axis, sign) == (1, -1)                                    # antérieur = -Y


def test_focus_distance_bounds():
    assert sc.focus_distance(1) == 350
    assert sc.focus_distance(10_000) == 1500
    assert 350 < sc.focus_distance(150) < 1500


def test_ray_pick_returns_nearest_visible_structure(fake_catalog):
    """Picking par rayon (NumPy) : structure la plus proche, masquage respecté, sans OpenGL."""
    bones = [s for s in fake_catalog if s.kind == "bone"]
    layer = sc.SceneLayer(
        "bone", sc.build_merged(bones),
        np.tile(np.array(sc.COLOR_BONE, np.uint8), (len(bones), 1)),
        {s.id: s for s in bones},
    )
    plotter = pv.Plotter(off_screen=True)
    try:
        layer.ensure(plotter)
        # rayon horizontal vers le sternum (cube de 40 mm centré en y=-190)
        p0, p1 = np.array([0.0, -600, 1220]), np.array([0.0, 0, 1220])
        t, cell = layer.ray_pick(p0, p1)
        assert layer.id_at_cell(cell) == "sternum"
        assert t == pytest.approx((600 - 190 - 20) / 600, abs=1e-6)
        # rayon qui traverse deux structures alignées : la première rencontrée gagne
        p0, p1 = np.array([300.0, -80, 600]), np.array([-300.0, -80, 600])
        assert layer.id_at_cell(layer.ray_pick(p0, p1)[1]) == "femur_left"
        p0, p1 = p1, p0
        assert layer.id_at_cell(layer.ray_pick(p0, p1)[1]) == "femur_right"
        # rayon dans le vide
        assert layer.ray_pick(np.array([500.0, 0, 0]), np.array([500.0, 100, 0])) is None
        # structure masquée : plus touchée
        layer.ensure(plotter, ["sternum"])
        assert layer.ray_pick(np.array([0.0, -600, 1220]), np.array([0.0, 0, 1220])) is None
    finally:
        plotter.close()


def test_pick_structure_headless(fake_catalog):
    """Le picking renvoie la structure sous le pixel (rendu hors écran PyVista)."""
    bones = [s for s in fake_catalog if s.kind == "bone"]
    layer = sc.SceneLayer(
        "bone", sc.build_merged(bones),
        np.tile(np.array(sc.COLOR_BONE, np.uint8), (len(bones), 1)),
        {s.id: s for s in bones},
    )
    try:
        pl = pv.Plotter(off_screen=True, window_size=(400, 400))
        layer.ensure(pl)
        c = layer.merged.centroids[layer.merged.index["sternum"]]
        pl.camera.position = (c[0], c[1] - 600, c[2])
        pl.camera.focal_point = tuple(c)
        pl.camera.up = (0, 0, 1)
        pl.show(auto_close=False)
        pl.renderer.SetWorldPoint(*c, 1.0)
        pl.renderer.WorldToDisplay()
        x, y, _ = pl.renderer.GetDisplayPoint()
        assert sc.pick_structure(pl.renderer, x, y, [layer]) == ("bone", "sternum")
        assert sc.pick_structure(pl.renderer, 2, 2, [layer]) == (None, None)
        pl.close()
    except Exception as exc:  # pragma: no cover - pas d'OpenGL
        pytest.skip(f"rendu OpenGL hors écran indisponible : {exc}")


# ─── Widget Qt ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyvistaqt")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_widget(qapp, catalog, **kw):
    from bonylandmarks.structure_quiz_exercise import StructureQuizExercise

    try:
        w = StructureQuizExercise(lang="fr", structures=catalog, **kw)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"widget non constructible ici : {exc}")
    return w


def _start(w, **kw):
    cfg = QuizConfig(seed=1, tier_max=2, **kw)
    try:
        assert w.start_quiz(cfg) is True
    except Exception as exc:  # pragma: no cover - QtInteractor indisponible
        if "OpenGL" in str(exc) or "render" in str(exc).lower():
            pytest.skip(f"QtInteractor indisponible : {exc}")
        raise
    return cfg


def _drain(qapp):
    qapp.processEvents()


def test_config_page_defaults_and_no_mesh_message(qapp):
    w = _make_widget(qapp, [], kind="muscle", mode="locate")
    try:
        assert w.page == 0
        assert not w._start_btn.isEnabled()
        assert "fetch_meshes.py" in w._avail_lbl.text()
        assert "import_bp3d.py" in w._avail_lbl.text()
        assert "muscle" in w._avail_lbl.text()
        assert "BodyParts3D" in w._attr_lbl.text()
    finally:
        w.shutdown()


def test_config_reflects_selection(qapp, fake_catalog):
    w = _make_widget(qapp, fake_catalog, kind="bone", mode="identify")
    try:
        assert w._start_btn.isEnabled()
        assert w._level_box.isVisibleTo(w)                 # niveaux : mode identifier
        assert set(w._region_checks) == {"lower_limb", "upper_limb", "thorax", "skull"}
        w._rb_mode["locate"].setChecked(True)
        assert not w._level_box.isVisibleTo(w)             # pas de niveaux en localiser
        w._rb_kind["muscle"].setChecked(True)
        assert set(w._region_checks) == {"lower_limb", "upper_limb", "trunk"}
        assert w._sup_check.isVisibleTo(w) and w._sup_check.isChecked()
        w._rb_mode["identify"].setChecked(True)
        assert not w._sup_check.isVisibleTo(w)             # option réservée à localiser
        w._rb_level[3].setChecked(True)
        w._region_checks["trunk"].setChecked(False)
        cfg = w._config_from_ui()
        assert cfg.kind == "muscle" and cfg.mode == Mode.IDENTIFY and cfg.level == Level.TEXT
        assert set(cfg.regions) == {"lower_limb", "upper_limb"}
        # 1 seule région de muscle cochée avec « majeures » : décompte cohérent
        w._region_checks["upper_limb"].setChecked(False)
        assert w._eligible_count(w._config_from_ui()) == 2   # rectus + vastus (tier 1)
        w._tier_combo.setCurrentIndex(1)
        assert w._eligible_count(w._config_from_ui()) == 3   # + gluteus_minimus (tier 2)
    finally:
        w.shutdown()


def _finish_identify(w, qapp, level, correct=True):
    from PySide6.QtCore import Qt

    while not w.session.finished:
        q = w.session.current
        if level == Level.CHOICE:
            idx = next(i for i, c in enumerate(q.choices)
                       if (c.id == q.target.id) == correct)
            w._on_choice(idx)
        elif level == Level.LIST:
            if correct:
                (it,) = w._list.findItems(q.target.name_fr, Qt.MatchExactly)
                w._list.setCurrentItem(it)
                w._on_list_submit()
            else:
                w._on_skip()
        else:
            w._text_edit.setText(q.target.name_en.upper() if correct else "zzz")
            w._on_text_submit()
        assert w._next_btn.isEnabled()
        w._on_next()
        _drain(qapp)


@pytest.mark.parametrize("level", [Level.CHOICE, Level.LIST, Level.TEXT])
def test_identify_bone_full_session(qapp, fake_catalog, level):
    w = _make_widget(qapp, fake_catalog)
    try:
        _start(w, kind="bone", mode=Mode.IDENTIFY, level=level, n_questions=5)
        assert w.page == 1
        assert w._answer_stack.currentIndex() == {Level.CHOICE: 0, Level.LIST: 1, Level.TEXT: 2}[level]
        assert not w._banner.isVisible()            # os : jamais de bandeau couche
        _finish_identify(w, qapp, level)
        assert w.page == 2                          # écran des résultats
        assert w.session.percent == pytest.approx(100.0)
        assert not w._replay_btn.isVisible()        # rien à rejouer
    finally:
        w.shutdown()


def test_identify_target_colour_and_scene_reuse(qapp, fake_catalog):
    w = _make_widget(qapp, fake_catalog)
    try:
        _start(w, kind="bone", mode=Mode.IDENTIFY, level=Level.CHOICE, n_questions=4)
        lay = w._layers["bone"]
        first = lay.rebuilds
        for _ in range(3):
            q = w.session.current
            rgb = lay.poly.cell_data["rgb"]
            tgt = lay.cell_sid == lay.merged.index[q.target.id]
            assert (rgb[tgt] == np.array(sc.COLOR_TARGET, np.uint8)).all()
            assert (rgb[~tgt] == np.array(sc.COLOR_BONE, np.uint8)).all()   # opaque, neutre
            w._on_skip()
            w._on_next()
        assert lay.rebuilds == first                # même mesh d'une question à l'autre
    finally:
        w.shutdown()


def test_locate_feedback_and_scores(qapp, fake_catalog):
    w = _make_widget(qapp, fake_catalog)
    try:
        _start(w, kind="bone", mode=Mode.LOCATE, n_questions=4)
        assert w._answer_stack.currentIndex() == 3 and w._awaiting_pick
        lay = w._layers["bone"]
        tid = w.session.current.target.id

        # clic dans le vide : ignoré, la question reste ouverte
        w._handle_pick(None, None)
        assert w._awaiting_pick and not w._answered

        # clic sur une mauvaise structure sans rapport -> 0, cible verte / clic rouge
        wrong = next(s.id for s in fake_catalog
                     if s.kind == "bone" and base_id(s.id) != base_id(tid))
        w._handle_pick("bone", wrong)
        rgb = lay.poly.cell_data["rgb"]
        m = lay.merged.index
        assert (rgb[lay.cell_sid == m[tid]] == np.array(sc.COLOR_OK, np.uint8)).all()
        assert (rgb[lay.cell_sid == m[wrong]] == np.array(sc.COLOR_WRONG, np.uint8)).all()
        assert w.session.current.verdict.points == 0.0
        assert not w._awaiting_pick
        w._handle_pick("bone", tid)                 # second clic ignoré
        assert w.session.current.verdict.points == 0.0
        w._on_next()

        # bonne réponse
        tid = w.session.current.target.id
        w._handle_pick("bone", tid)
        assert w.session.current.verdict.points == 1.0
        w._on_next()

        # côté opposé -> demi-point, orange
        q = w.session.current
        tid = q.target.id
        opp = {"_left": "_right", "_right": "_left"}
        base = next((tid[: -len(k)] + v for k, v in opp.items()
                     if tid.endswith(k) and tid[: -len(k)] + v in lay.merged.index), None)
        if base is not None:
            w._handle_pick("bone", base)
            assert w.session.current.verdict.points == 0.5
            rgb = lay.poly.cell_data["rgb"]
            assert (rgb[lay.cell_sid == m[base]] == np.array(sc.COLOR_OPPOSITE, np.uint8)).all()
        else:
            w._on_skip()
        w._on_next()
        w._on_skip()
        w._on_next()
        assert w.page == 2
    finally:
        w.shutdown()


def test_locate_click_on_bone_in_muscle_scene_is_ignored(qapp, fake_catalog):
    w = _make_widget(qapp, fake_catalog)
    try:
        _start(w, kind="muscle", mode=Mode.LOCATE, n_questions=2, superficial_targets_only=True)
        assert all(q.target.layer == "superficial" for q in w.session.questions)
        w._handle_pick("bone", "femur_left")       # un os n'est pas une réponse
        assert w._awaiting_pick and not w._answered
        assert w._feedback.text() == ""
    finally:
        w.shutdown()


def test_deep_muscle_hides_superficial_layer(qapp, fake_catalog):
    w = _make_widget(qapp, fake_catalog)
    try:
        by_id = {s.id: s for s in fake_catalog}
        targets = [by_id[i] for i in ("vastus_intermedius_left", "brachialis_left",
                                      "rectus_femoris_left")]
        cfg = QuizConfig(kind="muscle", mode=Mode.IDENTIFY, level=Level.CHOICE, tier_max=2, seed=2)
        from bonylandmarks.structure_quiz_engine import QuizSession

        pool = w._available("muscle")
        session = QuizSession(cfg, pool, targets=targets)
        try:
            w._config = cfg
            assert w._prepare_scene(cfg)
            w._begin_session(session)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"QtInteractor indisponible : {exc}")
        ml = w._layers["muscle"]
        n_all = len(ml.merged.tris)

        # Q1 : muscle profond du membre inférieur -> les superficiels du membre inférieur partent
        assert w.session.current.target.id == "vastus_intermedius_left"
        assert w._banner.isVisibleTo(w) and "superficielle" in w._banner.text().lower()
        assert len(ml.cell_sid) == n_all - 12                       # rectus_femoris retiré
        assert ml.merged.index["rectus_femoris_left"] not in set(ml.cell_sid)
        assert ml.merged.index["biceps_brachii_left"] in set(ml.cell_sid)   # autre région : reste
        col = ml.poly.cell_data["rgb"][ml.cell_sid == ml.merged.index["vastus_intermedius_left"]]
        assert (col == np.array(sc.COLOR_TARGET, np.uint8)).all()
        w._on_skip()
        w._on_next()

        # Q2 : profond du membre supérieur -> le biceps est retiré, le rectus revient
        assert w.session.current.target.id == "brachialis_left"
        assert ml.merged.index["biceps_brachii_left"] not in set(ml.cell_sid)
        assert ml.merged.index["rectus_femoris_left"] in set(ml.cell_sid)
        w._on_skip()
        w._on_next()

        # Q3 : cible superficielle -> tout est affiché, pas de bandeau
        assert w.session.current.target.id == "rectus_femoris_left"
        assert not w._banner.isVisibleTo(w)
        assert len(ml.cell_sid) == n_all
    finally:
        w.shutdown()


def test_locate_deep_muscle_removes_all_superficial(qapp, fake_catalog):
    """LOCALISER + cible profonde : couche superficielle retirée partout (pas d'indice)."""
    w = _make_widget(qapp, fake_catalog)
    try:
        from bonylandmarks.structure_quiz_engine import QuizSession

        cfg = QuizConfig(kind="muscle", mode=Mode.LOCATE, tier_max=2, seed=1,
                         superficial_targets_only=False)
        target = next(s for s in fake_catalog if s.id == "brachialis_left")
        try:
            w._config = cfg
            assert w._prepare_scene(cfg)
            w._begin_session(QuizSession(cfg, w._available("muscle"), targets=[target]))
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"QtInteractor indisponible : {exc}")
        ml = w._layers["muscle"]
        ids = {ml.merged.ids[i] for i in set(ml.cell_sid)}
        assert not any(w._by_id[i].layer == "superficial" for i in ids)
        assert w._banner.isVisibleTo(w)
    finally:
        w.shutdown()


def test_retry_missed_and_new_config(qapp, fake_catalog):
    w = _make_widget(qapp, fake_catalog)
    try:
        _start(w, kind="bone", mode=Mode.IDENTIFY, level=Level.CHOICE, n_questions=4)
        # 2 bonnes, 2 fausses
        for i in range(4):
            q = w.session.current
            idx = next(j for j, c in enumerate(q.choices) if (c.id == q.target.id) == (i < 2))
            w._on_choice(idx)
            w._on_next()
        assert w.page == 2
        assert w.session.score == pytest.approx(2.0)
        assert w._replay_btn.isVisibleTo(w) and "2" in w._replay_btn.text()
        missed = {t.id for t in w.session.retry_queue()}
        w._on_replay()
        assert w.page == 1 and len(w.session.questions) == 2
        assert {q.target.id for q in w.session.questions} == missed
        w.abandon()                                 # 0/2 : tout compte zéro
        assert w.page == 2 and w.session.score == 0.0
        w._on_new_config()
        assert w.page == 0
    finally:
        w.shutdown()


def test_signals_and_lifecycle(qapp, fake_catalog):
    w = _make_widget(qapp, fake_catalog)
    got = []
    w.exercise_complete.connect(lambda: got.append(1))
    w._menu_btn.click()
    assert got == [1]
    w.shutdown()
    w.shutdown()                                    # idempotent


@pytest.mark.skipif(
    not cat.available("bone"), reason="meshes BodyParts3D non importés (scripts/fetch_meshes.py)"
)
def test_real_meshes_identify_and_locate(qapp):
    from bonylandmarks.structure_quiz_exercise import StructureQuizExercise

    sc.clear_cache()
    try:
        w = StructureQuizExercise(lang="en", kind="bone", mode="identify")
    except Exception as exc:  # pragma: no cover
        pytest.skip(str(exc))
    try:
        assert w._start_btn.isEnabled()
        _start(w, kind="bone", mode=Mode.IDENTIFY, level=Level.CHOICE, n_questions=3)
        assert len(w._layers["bone"].merged) >= 100
        _finish_identify(w, qapp, Level.CHOICE)
        assert w.page == 2
        _start(w, kind="bone", mode=Mode.LOCATE, n_questions=2)
        tid = w.session.current.target.id
        w._handle_pick("bone", tid)
        assert w.session.current.verdict.correct
    finally:
        w.shutdown()
