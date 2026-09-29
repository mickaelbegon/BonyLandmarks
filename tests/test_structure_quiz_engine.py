"""Tests for the pure-Python structure quiz engine (no Qt, no catalogue import)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import pytest

from bonylandmarks.structure_quiz_engine import (
    Level,
    Mode,
    QuizConfig,
    QuizSession,
    Verdict,
    check_pick,
    check_text_answer,
    full_list,
    grade_letter,
    make_choices,
    normalize,
)


@dataclass(frozen=True)
class FakeStructure:
    id: str
    kind: str
    name_fr: str
    name_en: str
    side: str = "mid"
    region: str = "trunk"
    tier: int = 1
    synonyms_fr: tuple[str, ...] = ()
    synonyms_en: tuple[str, ...] = ()
    layer: str = ""


def _pair(stem, kind, fr, en, region, syn_fr=(), syn_en=(), layer="", tier=1):
    out = []
    for side, fr_s, en_s in (("left", "gauche", "left"), ("right", "droit", "right")):
        out.append(
            FakeStructure(
                id=f"{stem}_{side}", kind=kind, name_fr=f"{fr} {fr_s}",
                name_en=f"{en} {en_s}", side=side, region=region, tier=tier,
                synonyms_fr=tuple(syn_fr), synonyms_en=tuple(syn_en), layer=layer,
            )
        )
    return out


def _mid(stem, kind, fr, en, region, syn_fr=(), syn_en=(), layer="", tier=1):
    return FakeStructure(stem, kind, fr, en, "mid", region, tier, tuple(syn_fr),
                         tuple(syn_en), layer)


def build_pool() -> list[FakeStructure]:
    p: list[FakeStructure] = []
    p += _pair("femur", "bone", "Fémur", "Femur", "lower_limb", ("os de la cuisse",))
    p += _pair("tibia", "bone", "Tibia", "Tibia", "lower_limb")
    p += _pair("fibula", "bone", "Fibula", "Fibula", "lower_limb", ("péroné",), ("peroneal bone",))
    p += _pair("patella", "bone", "Patella", "Patella", "lower_limb", ("rotule",))
    p += _pair("humerus", "bone", "Humérus", "Humerus", "upper_limb")
    p += _pair("radius", "bone", "Radius", "Radius", "upper_limb")
    p += _pair("ulna", "bone", "Ulna", "Ulna", "upper_limb", ("cubitus",))
    p += _pair("scapula", "bone", "Scapula", "Scapula", "shoulder", ("omoplate",))
    p += _pair("clavicle", "bone", "Clavicule", "Clavicle", "shoulder")
    p.append(_mid("sacrum", "bone", "Sacrum", "Sacrum", "trunk"))
    p.append(_mid("sternum", "bone", "Sternum", "Sternum", "trunk"))
    p.append(_mid("l1", "bone", "Vertèbre L1", "L1 vertebra", "trunk"))
    p.append(_mid("l2", "bone", "Vertèbre L2", "L2 vertebra", "trunk"))
    p += _pair("pec_major", "muscle", "Grand pectoral", "Pectoralis major", "trunk",
               ("m. grand pectoral",), ("pectoralis major muscle",), layer="superficial")
    p += _pair("pec_minor", "muscle", "Petit pectoral", "Pectoralis minor", "trunk",
               layer="deep")
    p += _pair("fib_long", "muscle", "Fibulaire long", "Fibularis longus", "lower_limb",
               layer="superficial")
    p += _pair("fib_short", "muscle", "Fibulaire court", "Fibularis brevis", "lower_limb",
               layer="deep")
    p += _pair("rect_abd", "muscle", "Droit de l'abdomen", "Rectus abdominis", "trunk",
               layer="superficial")
    p += _pair("vast_med", "muscle", "Vaste médial", "Vastus medialis", "lower_limb",
               layer="superficial")
    p += _pair("vast_lat", "muscle", "Vaste latéral", "Vastus lateralis", "lower_limb",
               layer="superficial")
    p += _pair("deltoid", "muscle", "Deltoïde", "Deltoid", "shoulder", layer="superficial")
    return p


POOL = build_pool()
BY_ID = {s.id: s for s in POOL}


# ─── normalize ────────────────────────────────────────────────────────────────


def test_normalize_basic():
    assert normalize("Fémur") == "femur"
    assert normalize("  L'Os   de la Cuisse ") == "cuisse"
    assert normalize("Ligament, du genou!") == "ligament genou"
    assert normalize("The head of the Femur") == "head femur"


def test_normalize_optional_words():
    assert normalize("grand pectoral") == normalize("muscle grand pectoral")
    assert normalize("m. grand pectoral") == normalize("Grand Pectoral")
    assert normalize("os coxal") == normalize("coxal")
    assert normalize("femur bone") == normalize("femur")
    assert normalize("muscle of the biceps") == normalize("biceps")


def test_normalize_never_empty():
    assert normalize("os") == "os"
    assert normalize("muscle") == "muscle"
    assert normalize("") == ""


def test_normalize_ligatures_and_apostrophes():
    assert normalize("Cœur") == "coeur"
    assert normalize("l’omoplate") == "omoplate"
    assert normalize("d'abord") == "abord"


# ─── text answers ─────────────────────────────────────────────────────────────


def T(text, target_id, pool=POOL, lang="fr"):
    return check_text_answer(text, BY_ID[target_id], lang, pool)


def test_text_exact_with_side():
    v = T("Fémur gauche", "femur_left")
    assert v.correct and v.name_ok and v.side_ok and not v.close
    assert v.points == 1.0 and v.matched == "Fémur gauche"


def test_text_accepts_both_languages_and_synonyms():
    assert T("femur left", "femur_left", lang="fr").correct
    assert T("Left femur", "femur_left", lang="fr").correct
    assert T("omoplate gauche", "scapula_left").correct
    assert T("scapula g", "scapula_left").correct
    assert T("péroné droit", "fibula_right").correct
    assert T("Peroneal bone right", "fibula_right").correct
    assert T("os de la cuisse gauche", "femur_left").correct
    assert T("rotule (D)", "patella_right").correct
    assert T("pectoralis major muscle left", "pec_major_left").correct
    assert T("M. Grand pectoral gauche", "pec_major_left").correct
    assert T("muscle grand pectoral gauche", "pec_major_left").correct


def test_text_no_side_gives_half_point():
    v = T("fémur", "femur_left")
    assert v.name_ok and not v.side_ok and not v.correct and v.points == 0.5


def test_text_wrong_side_gives_half_point():
    v = T("fémur droit", "femur_left")
    assert v.name_ok and not v.side_ok and v.points == 0.5
    v = T("right femur", "femur_left")
    assert v.points == 0.5
    v = T("femur gauche droit", "femur_left")  # contradictory
    assert v.points == 0.5


def test_text_midline_no_side_needed():
    v = T("sacrum", "sacrum")
    assert v.correct and v.points == 1.0
    assert T("Sacrum", "sacrum").correct
    assert T("os sacrum", "sacrum").correct


def test_text_wrong_structure():
    v = T("humérus gauche", "femur_left")
    assert not v.name_ok and v.points == 0.0
    assert v.matched == "Humérus gauche"  # recognised, but another structure


def test_text_empty_or_garbage():
    assert T("", "femur_left").points == 0.0
    assert T("   ", "femur_left").matched is None
    assert T("blabla", "femur_left").points == 0.0
    assert T("gauche", "femur_left").points == 0.0
    assert T("os", "femur_left").points == 0.0


def test_text_typo_accepted_as_close():
    v = T("humerous gauche", "humerus_left")
    assert v.correct and v.close
    v = T("clavicul droite", "clavicle_right")
    assert v.correct and v.close
    v = T("scapulla gauche", "scapula_left")
    assert v.correct and v.close
    v = T("pectoralis majr left", "pec_major_left")
    assert not v.close or v.name_ok  # never a crash; see confusion tests below
    v = T("deltoide gauche", "deltoid_left")
    assert v.correct and not v.close  # accent only -> exact after normalisation


def test_text_typo_without_pool():
    v = check_text_answer("humerous gauche", BY_ID["humerus_left"])
    assert v.correct and v.close


@pytest.mark.parametrize("pool", [POOL, None])
def test_confusion_radius_ulna(pool):
    assert not check_text_answer("ulna gauche", BY_ID["radius_left"], "fr", pool).name_ok
    assert not check_text_answer("radius gauche", BY_ID["ulna_left"], "fr", pool).name_ok
    assert not check_text_answer("radios gauche", BY_ID["ulna_left"], "fr", pool).name_ok
    assert not check_text_answer("ulne gauche", BY_ID["radius_left"], "fr", pool).name_ok


@pytest.mark.parametrize("pool", [POOL, None])
def test_confusion_tibia_fibula(pool):
    assert not check_text_answer("tibia droit", BY_ID["fibula_right"], "fr", pool).name_ok
    assert not check_text_answer("fibula droit", BY_ID["tibia_right"], "fr", pool).name_ok
    assert not check_text_answer("fibulla droit", BY_ID["tibia_right"], "fr", pool).name_ok


@pytest.mark.parametrize("pool", [POOL, None])
def test_confusion_grand_petit_pectoral(pool):
    assert not check_text_answer("petit pectoral gauche", BY_ID["pec_major_left"], "fr", pool).name_ok
    assert not check_text_answer("grand pectoral gauche", BY_ID["pec_minor_left"], "fr", pool).name_ok
    assert not check_text_answer("grand pectoral", BY_ID["pec_minor_left"], "fr", pool).name_ok
    assert not check_text_answer("pectoralis minor left", BY_ID["pec_major_left"], "en", pool).name_ok
    assert not check_text_answer("pectoralis minr left", BY_ID["pec_major_left"], "en", pool).name_ok


@pytest.mark.parametrize("pool", [POOL, None])
def test_confusion_fibulaire_long_court(pool):
    assert not check_text_answer("fibulaire long gauche", BY_ID["fib_short_left"], "fr", pool).name_ok
    assert not check_text_answer("fibulaire court gauche", BY_ID["fib_long_left"], "fr", pool).name_ok
    assert not check_text_answer("fibularis brevis left", BY_ID["fib_long_left"], "en", pool).name_ok
    assert check_text_answer("fibulaire cort gauche", BY_ID["fib_short_left"], "fr", pool).correct


@pytest.mark.parametrize("pool", [POOL, None])
def test_confusion_medial_lateral_and_vertebrae(pool):
    assert not check_text_answer("vaste median gauche", BY_ID["vast_med_left"], "fr", pool).name_ok
    assert not check_text_answer("vaste lateral gauche", BY_ID["vast_med_left"], "fr", pool).name_ok
    assert check_text_answer("vaste medial gauche", BY_ID["vast_med_left"], "fr", pool).correct
    assert not check_text_answer("vertebre L2", BY_ID["l1"], "fr", pool).name_ok
    assert not check_text_answer("L1 vertebra", BY_ID["l2"], "en", pool).name_ok
    assert check_text_answer("vertèbre l1", BY_ID["l1"], "fr", pool).correct


def test_pool_attribution_beats_typo():
    """A typo equidistant to the target must go to the best pool structure."""
    v = T("radiuss gauche", "radius_left")
    assert v.correct  # only radius is close
    v = T("ulna gauche", "radius_left")
    assert not v.name_ok and v.matched == "Ulna gauche"


def test_droit_in_muscle_name_is_not_a_side():
    v = T("muscle droit de l'abdomen gauche", "rect_abd_left")
    assert v.correct
    v = T("droit de l'abdomen droit", "rect_abd_left")
    assert v.name_ok and not v.side_ok and v.points == 0.5
    v = T("droit de l'abdomen", "rect_abd_right")
    assert v.name_ok and not v.side_ok  # no side stated
    assert T("rectus abdominis right", "rect_abd_right").correct


def test_word_count_mismatch_not_fuzzy():
    assert not T("pectoral gauche", "pec_major_left").name_ok
    assert not T("grand pectoral inferieur gauche", "pec_major_left").name_ok


# ─── check_pick ───────────────────────────────────────────────────────────────


def test_check_pick():
    assert check_pick("femur_left", "femur_left").correct
    assert check_pick("femur_left", "femur_left").points == 1.0
    v = check_pick("femur_left", "femur_right")
    assert v.name_ok and not v.side_ok and not v.correct and v.points == 0.5
    v = check_pick("femur_left", "tibia_left")
    assert not v.name_ok and v.points == 0.0
    assert check_pick("femur_left", None).points == 0.0
    assert check_pick("sacrum", "sacrum").correct
    assert check_pick("sacrum", "sternum").points == 0.0
    assert not check_pick("femur_left", "femur_left_extra").name_ok


# ─── choices & lists ──────────────────────────────────────────────────────────


def test_make_choices_basic():
    rng = random.Random(1)
    target = BY_ID["femur_left"]
    ch = make_choices(target, POOL, 4, rng)
    assert len(ch) == 4
    assert target in ch
    assert len({c.name_fr for c in ch}) == 4
    assert len({c.id for c in ch}) == 4


def test_make_choices_same_region_and_kind_first():
    target = BY_ID["femur_left"]
    region_size = sum(1 for s in POOL if s.kind == "bone" and s.region == "lower_limb") - 1
    assert region_size >= 3
    for seed in range(30):
        ch = make_choices(target, POOL, 4, random.Random(seed))
        for c in ch:
            assert c.kind == "bone" and c.region == "lower_limb"


def test_make_choices_at_most_one_opposite_side():
    target = BY_ID["femur_left"]
    for seed in range(60):
        ch = make_choices(target, POOL, 4, random.Random(seed))
        opp = [c for c in ch if c.id == "femur_right"]
        assert len(opp) <= 1
    # with a pool made only of the two femurs + 1 other, opposite appears once
    small = [BY_ID["femur_left"], BY_ID["femur_right"], BY_ID["tibia_left"]]
    ch = make_choices(BY_ID["femur_left"], small, 4, random.Random(0))
    assert sorted(c.id for c in ch) == ["femur_left", "femur_right", "tibia_left"]


def test_make_choices_fills_from_other_regions_and_kinds():
    target = BY_ID["sternum"]  # trunk bones: sacrum, l1, l2 only + target
    ch = make_choices(target, POOL, 6, random.Random(3))
    assert len(ch) == 6
    trunk_bones = {"sacrum", "l1", "l2"}
    assert trunk_bones <= {c.id for c in ch}


def test_make_choices_small_pool_and_no_duplicate_names():
    target = BY_ID["sacrum"]
    assert [c.id for c in make_choices(target, [target], 4)] == ["sacrum"]
    assert len(make_choices(target, [], 4)) == 1
    dup = FakeStructure("dup", "bone", "Sacrum", "Sacrum copy")
    ch = make_choices(target, [target, dup], 4, random.Random(0))
    assert [c.id for c in ch] == ["sacrum"]


def test_make_choices_deterministic():
    target = BY_ID["radius_right"]
    a = make_choices(target, POOL, 4, random.Random(42))
    b = make_choices(target, list(reversed(POOL)), 4, random.Random(42))
    assert [c.id for c in a] == [c.id for c in b]


def test_full_list_sorted_alphabetically():
    bones = [s for s in POOL if s.kind == "bone"]
    lst = full_list(bones, "fr")
    assert len(lst) == len(bones)
    keys = [normalize(s.name_fr) for s in lst]
    assert keys == sorted(keys)
    # accents ignored: "Fémur" sorts among the f's, "Humérus" after "Fibula"
    names = [s.name_fr for s in lst]
    assert names.index("Fémur droit") < names.index("Humérus droit")
    en = full_list(bones, "en")
    ken = [normalize(s.name_en) for s in en]
    assert ken == sorted(ken)


# ─── grade ────────────────────────────────────────────────────────────────────


def test_grade_letter():
    assert grade_letter(100) == "A" and grade_letter(90) == "A"
    assert grade_letter(89.9) == "B" and grade_letter(75) == "B"
    assert grade_letter(74) == "C" and grade_letter(60) == "C"
    assert grade_letter(59.9) == "D" and grade_letter(0) == "D"


# ─── sessions ─────────────────────────────────────────────────────────────────


def cfg(**kw):
    base = dict(kind="bone", mode=Mode.IDENTIFY, level=Level.CHOICE, n_questions=5,
                seed=7)
    base.update(kw)
    return QuizConfig(**base)


def test_session_selects_distinct_targets_with_filters():
    s = QuizSession(cfg(n_questions=8, regions=("lower_limb",)), POOL)
    ids = [q.target.id for q in s.questions]
    assert len(ids) == len(set(ids)) == 8
    assert all(BY_ID[i].kind == "bone" and BY_ID[i].region == "lower_limb" for i in ids)


def test_session_caps_questions_to_available():
    s = QuizSession(cfg(n_questions=99, kind="muscle"), POOL)
    assert len(s.questions) == sum(1 for p in POOL if p.kind == "muscle")


def test_session_tier_filter():
    pool = POOL + [_mid("detail", "bone", "Détail", "Detail", "trunk", tier=2)]
    s = QuizSession(cfg(n_questions=99, tier_max=1), pool)
    assert all(q.target.tier == 1 for q in s.questions)
    s2 = QuizSession(cfg(n_questions=99, tier_max=2), pool)
    assert any(q.target.id == "detail" for q in s2.questions)


def test_session_superficial_only_locate():
    c = QuizConfig(kind="muscle", mode=Mode.LOCATE, n_questions=99, seed=1,
                   superficial_targets_only=True)
    s = QuizSession(c, POOL)
    assert s.questions and all(q.target.layer == "superficial" for q in s.questions)
    # not applied in identify mode
    c2 = QuizConfig(kind="muscle", mode=Mode.IDENTIFY, n_questions=99, seed=1,
                    superficial_targets_only=True)
    assert any(q.target.layer == "deep" for q in QuizSession(c2, POOL).questions)


def test_session_choice_level_full_run_deterministic():
    def run():
        s = QuizSession(cfg(), POOL)
        for q in s.questions:
            assert len(q.choices) == 4 and q.target in q.choices
        order = [q.target.id for q in s.questions]
        choices = [[c.id for c in q.choices] for q in s.questions]
        i = 0
        while not s.finished:
            q = s.current
            if i % 2 == 0:
                v = s.submit_choice(q.target.id)
                assert v.correct
            else:
                wrong = next(c for c in q.choices if c.id != q.target.id)
                v = s.submit_choice(wrong.id)
                assert v.points in (0.0, 0.5)
            s.next()
            i += 1
        return order, choices, s.score, s.summary()

    a, b = run(), run()
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
    assert a[2] >= 3.0  # 3 of 5 right (i = 0, 2, 4)
    other = QuizSession(cfg(seed=8), POOL)
    assert [q.target.id for q in other.questions] != a[0]


def test_session_list_level():
    s = QuizSession(cfg(level=Level.LIST, n_questions=3), POOL)
    assert all(q.choices == [] for q in s.questions)
    opts = s.options
    assert {o.id for o in opts} == {p.id for p in POOL if p.kind == "bone"}
    q = s.current
    assert s.submit_list(q.target.id).correct
    with pytest.raises(ValueError):
        s.submit_choice(q.target.id)
    with pytest.raises(ValueError):
        s.submit_text("x")


def test_session_text_level():
    s = QuizSession(cfg(level=Level.TEXT, n_questions=4, regions=("upper_limb",)), POOL)
    t0 = s.current.target
    v = s.submit_text(t0.name_en)
    assert v.correct
    assert s.submit_text("nonsense").correct  # second submit returns the recorded verdict
    s.next()
    t1 = s.current.target
    stem = t1.name_fr.rsplit(" ", 1)[0]
    v = s.submit_text(stem)  # side missing
    assert v.points == 0.5
    s.next()
    v = s.submit_text("zzz")
    assert v.points == 0.0
    s.next()
    s.skip()
    s.next()
    assert s.finished and s.current is None
    assert s.score == 1.5 and s.max_score == 4.0
    with pytest.raises(RuntimeError):
        s.submit_text("x")


def test_session_locate_mode():
    c = QuizConfig(kind="bone", mode=Mode.LOCATE, n_questions=4, seed=3,
                   level=Level.TEXT)  # level ignored
    s = QuizSession(c, POOL)
    assert all(q.choices == [] for q in s.questions)
    q = s.current
    assert s.submit_pick(q.target.id).correct
    s.next()
    q = s.current
    if q.target.side in ("left", "right"):
        opp = q.target.id.rsplit("_", 1)[0] + ("_right" if q.target.side == "left" else "_left")
        v = s.submit_pick(opp)
        assert v.points == 0.5 and v.name_ok and not v.side_ok
    else:
        s.submit_pick(None)
    s.next()
    assert s.submit_pick(None).points == 0.0
    with pytest.raises(ValueError):
        s.submit_text("femur")


def test_session_next_skips_unanswered_and_skip_verdict():
    s = QuizSession(cfg(n_questions=2), POOL)
    first = s.current
    s.next()
    assert first.skipped and first.verdict.points == 0.0
    assert s.skip().points == 0.0
    s.next()
    assert s.finished
    assert s.next() is None


def test_retry_queue_and_retry_session():
    s = QuizSession(cfg(n_questions=4), POOL)
    targets = [q.target for q in s.questions]
    s.submit_choice(targets[0].id)
    s.next()
    wrong = next(c for c in s.current.choices if c.id != targets[1].id)
    s.submit_choice(wrong.id)
    s.next()
    s.skip()
    s.next()
    # last question left unanswered
    retry = s.retry_queue()
    ids = [t.id for t in retry]
    assert targets[0].id not in ids
    assert set(ids) >= {targets[2].id, targets[3].id}
    r = s.retry_session(seed=1)
    assert [q.target.id for q in r.questions] == ids
    assert all(len(q.choices) == 4 for q in r.questions)
    assert r.config.mode == s.config.mode


def test_summary_is_json_serialisable_and_complete():
    s = QuizSession(cfg(level=Level.TEXT, n_questions=5, seed=11), POOL)
    answers = ["Fémur gauche", "blabla", None, "tibia", "humerus"]
    for a in answers:
        q = s.current
        if a is None:
            s.skip()
        elif a == "Fémur gauche":
            s.submit_text(q.target.name_fr)
        else:
            s.submit_text(a)
        s.next()
    sm = s.summary()
    text = json.dumps(sm)  # must not raise
    back = json.loads(text)
    assert back["n_questions"] == 5 and len(back["questions"]) == 5
    assert back["mode"] == "identify" and back["level"] == 3 and back["kind"] == "bone"
    assert back["max_score"] == 5.0
    assert back["score"] == pytest.approx(s.score)
    assert back["percent"] == pytest.approx(round(100 * s.score / 5, 1))
    assert back["grade"] == grade_letter(back["percent"])
    assert back["questions"][0]["points"] == 1.0
    assert back["questions"][2]["skipped"] is True
    assert set(back["error_types"]) == {"wrong_name", "wrong_side", "skipped"}
    assert isinstance(back["frequent_errors"], list)
    assert back["retry_ids"] == [t.id for t in s.retry_queue()]


def test_summary_choice_confusions():
    s = QuizSession(cfg(n_questions=3), POOL)
    for _ in range(3):
        q = s.current
        wrong = next(c for c in q.choices if c.id != q.target.id
                     and c.id.rsplit("_", 1)[0] != q.target.id.rsplit("_", 1)[0])
        s.submit_choice(wrong.id)
        s.next()
    sm = s.summary()
    json.dumps(sm)
    assert sm["score"] == 0.0 and sm["grade"] == "D"
    assert sm["error_types"]["wrong_name"] == 3
    assert len(sm["frequent_errors"]) >= 1
    assert all("target" in e and "given" in e for e in sm["frequent_errors"])
    assert sum(e["count"] for e in sm["errors_by_region"]) == 3


def test_verdict_default_and_to_dict():
    v = Verdict()
    assert not v.correct and v.points == 0.0 and v.matched is None
    assert json.dumps(v.to_dict())


def test_duck_typing_minimal_object():
    class Mini:
        id = "x"
        name_fr = "Machin"
        name_en = "Thing"

    v = check_text_answer("thing", Mini(), "fr", [Mini()])
    assert v.correct and v.points == 1.0
