"""Pure-Python engine for the "identify / locate a structure" anatomy quizzes.

This module contains **no Qt / PyVista / VTK import**.  It works on duck-typed
``Structure`` objects (see ``anatomy_catalog.Structure``): any object exposing
``id, kind, name_fr, name_en, side, region, tier, synonyms_fr, synonyms_en,
layer`` works (missing attributes get sensible defaults).

Two modes
---------
* ``Mode.IDENTIFY`` -- a structure is highlighted, the student names it
  (``Level.CHOICE`` multiple choice, ``Level.LIST`` pick in the full list,
  ``Level.TEXT`` free typing).
* ``Mode.LOCATE`` -- a name is given, the student clicks the structure.

Scoring per question: 1.0 (name and side right), 0.5 (name right, side wrong or
missing on a lateralised structure), 0.0 otherwise.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any, Iterable, Sequence

# ─── Enums ────────────────────────────────────────────────────────────────────


class Mode(str, Enum):
    """Quiz direction."""

    IDENTIFY = "identify"
    LOCATE = "locate"


class Level(IntEnum):
    """Answer format for ``Mode.IDENTIFY``."""

    CHOICE = 1  # multiple choice (4 answers)
    LIST = 2    # full alphabetical list of the pool
    TEXT = 3    # free typing


# ─── Text normalisation ───────────────────────────────────────────────────────

#: Articles / particles removed everywhere.
_STOPWORDS = frozenset(
    {"le", "la", "les", "du", "de", "des", "au", "aux", "the", "of"}
)  # elided "l'" / "d'" are removed by regex (lone "l"/"d" stay: side letters)
#: Words that are never mandatory to match (dropped when something remains).
_NOISE_ANYWHERE = frozenset({"m", "muscle", "muscles"})

_LEFT_WORDS = frozenset({"gauche", "gauches", "left"})
_RIGHT_WORDS = frozenset({"right"})
#: French "droit" is also part of muscle names (droit de l'abdomen): it only
#: counts as a side when it is the last word.
_RIGHT_FR_LAST = frozenset({"droit", "droite", "droits"})
_SIDE_LETTERS = {"g": "left", "l": "left", "d": "right", "r": "right"}

#: Discriminating tokens that must match exactly (never fuzzy-matched onto each other).
_EXACT_TOKENS = frozenset(
    {
        "grand", "petit", "moyen", "long", "longue", "court", "courte", "longus",
        "brevis", "major", "minor", "majeur", "mineur", "medial", "mediale",
        "median", "medians", "lateral", "laterale", "superieur", "inferieur",
        "anterieur", "posterieur", "superficiel", "profond", "interne",
        "externe", "proximal", "distal", "internus", "externus", "superior",
        "inferior", "anterior", "posterior", "magnus", "minimus", "medius",
        "maximus", "premier", "deuxieme", "troisieme", "quatrieme",
        "cinquieme", "first", "second", "third", "fourth", "fifth",
        "gauche", "left", "right", "droit", "droite",
    }
)

#: Similarity threshold for accepting a typo (spec: ratio >= 0.85).
CLOSE_RATIO = 0.85
_TOKEN_RATIO = 0.80


def _ascii_lower(text: str) -> str:
    text = (text or "").replace("œ", "oe").replace("Œ", "oe")
    text = text.replace("æ", "ae").replace("Æ", "ae")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    # l'os, d'un ... -> drop the elided article
    text = re.sub(r"\b[ld]'", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text)


def _stopped(text: str) -> list[str]:
    """Raw tokens without articles / particles."""
    raw = _ascii_lower(text).split()
    return [t for t in raw if t not in _STOPWORDS] or raw


def _drop_noise(toks: list[str]) -> list[str]:
    """Remove non-mandatory words (muscle, m, leading os, bone at an end)."""
    kept: list[str] = []
    for i, t in enumerate(toks):
        if t in _NOISE_ANYWHERE:
            continue
        if t == "os" and i == 0:
            continue
        if t == "bone" and i in (0, len(toks) - 1):
            continue
        kept.append(t)
    return kept or toks


def _tokens(text: str) -> list[str]:
    """Normalised tokens: articles and non-mandatory words removed."""
    return _drop_noise(_stopped(text))


def normalize(text: str) -> str:
    """Lower-case, accent-free, punctuation-free, without articles / "muscle" / "os".

    ``normalize("Muscle grand pectoral") == normalize("le Grand Pectoral")``.
    The words are never all removed: ``normalize("os") == "os"``.
    """
    return " ".join(_tokens(text))


@lru_cache(maxsize=8192)
def _parse(text: str) -> tuple[tuple[str, ...], str | None]:
    """Split a name/answer into (tokens without side words, side or None).

    ``side`` is ``"left"``, ``"right"``, ``"both"`` (contradictory) or ``None``.
    """
    toks = _stopped(text)
    if len(toks) <= 1:
        return tuple(_drop_noise(toks)), None
    sides: list[str] = []
    rest: list[str] = []
    last = len(toks) - 1
    for i, t in enumerate(toks):
        if t in _LEFT_WORDS:
            sides.append("left")
        elif t in _RIGHT_WORDS:
            sides.append("right")
        elif t in _RIGHT_FR_LAST and i == last:
            sides.append("right")
        elif t in _SIDE_LETTERS and i in (0, last):
            sides.append(_SIDE_LETTERS[t])
        else:
            rest.append(t)
    if not rest:  # only side words: keep them as the name
        return tuple(_drop_noise(toks)), None
    rest = _drop_noise(rest)
    if not sides:
        side = None
    elif len(set(sides)) == 1:
        side = sides[0]
    else:
        side = "both"
    return tuple(rest), side


# ─── Structure duck-typing helpers ────────────────────────────────────────────


def _g(s: Any, attr: str, default: Any = "") -> Any:
    return getattr(s, attr, default)


_SIDE_SUFFIX = re.compile(r"_(left|right)$")


def base_id(structure_id: str) -> str:
    """Id without its ``_left`` / ``_right`` suffix."""
    return _SIDE_SUFFIX.sub("", structure_id)


def is_opposite_side(id_a: str, id_b: str) -> bool:
    """True if both ids are the left/right versions of the same structure."""
    return id_a != id_b and base_id(id_a) == base_id(id_b)


def display_name(structure: Any, lang: str = "fr") -> str:
    """Name of the structure in ``lang`` (falls back to the other language)."""
    fr, en = _g(structure, "name_fr"), _g(structure, "name_en")
    return (fr or en) if lang == "fr" else (en or fr)


def _candidate_names(structure: Any) -> list[str]:
    names = [_g(structure, "name_fr"), _g(structure, "name_en")]
    names += list(_g(structure, "synonyms_fr", ()) or ())
    names += list(_g(structure, "synonyms_en", ()) or ())
    return [n for n in names if n]


# ─── Verdict ──────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    """Outcome of one answer."""

    correct: bool = False
    name_ok: bool = False
    side_ok: bool = False
    close: bool = False           # accepted despite a typo
    matched: str | None = None    # canonical name recognised in the answer
    points: float = 0.0           # 1.0 / 0.5 / 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "name_ok": self.name_ok,
            "side_ok": self.side_ok,
            "close": self.close,
            "matched": self.matched,
            "points": self.points,
        }


def _verdict(name_ok: bool, side_ok: bool, close: bool, matched: str | None) -> Verdict:
    correct = name_ok and side_ok
    points = 1.0 if correct else (0.5 if name_ok else 0.0)
    return Verdict(correct, name_ok, side_ok, close and name_ok, matched, points)


# ─── Text answer checking ─────────────────────────────────────────────────────


def _match_score(
    ans: tuple[str, ...], cand: tuple[str, ...], sm: SequenceMatcher
) -> float:
    """1.0 for an exact match, a ratio in [0.85, 1) for an accepted typo, else 0.

    ``sm`` must have the joined answer as ``seq2``.  Fuzzy matching requires the
    same number of words; short words, words with digits and discriminating
    words (grand/petit, long/court, medial/median...) must match exactly, so
    two neighbouring structures are never confused.
    """
    if not ans or not cand:
        return 0.0
    if ans == cand:
        return 1.0
    if len(ans) != len(cand):
        return 0.0
    for a, b in zip(ans, cand):
        if a == b:
            continue
        if (
            min(len(a), len(b)) < 4
            or (a in _EXACT_TOKENS and b in _EXACT_TOKENS)
            or any(c.isdigit() for c in a + b)
        ):
            return 0.0
        if SequenceMatcher(None, a, b).ratio() < _TOKEN_RATIO:
            return 0.0
    sm.set_seq1(" ".join(cand))
    if sm.real_quick_ratio() < CLOSE_RATIO or sm.quick_ratio() < CLOSE_RATIO:
        return 0.0
    ratio = sm.ratio()
    return ratio if ratio >= CLOSE_RATIO else 0.0


def _structure_score(
    ans: tuple[str, ...], structure: Any, sm: SequenceMatcher
) -> float:
    best = 0.0
    for name in _candidate_names(structure):
        cand, _ = _parse(name)
        sc = _match_score(ans, cand, sm)
        if sc > best:
            best = sc
            if best == 1.0:
                break
    return best


def check_text_answer(
    text: str,
    target: Any,
    lang: str = "fr",
    pool: Iterable[Any] | None = None,
) -> Verdict:
    """Check a free-text answer against ``target``.

    Accepts ``name_fr`` / ``name_en`` / synonyms (both languages whatever
    ``lang``), with or without the side (gauche/droit/left/right/g/d/l/r), and
    small typos (``close=True``).  When ``pool`` is given the answer is
    attributed to the pool structure with the best score and is correct only if
    that is the target: a typo can never turn a neighbouring structure into the
    target.
    """
    ans, ans_side = _parse(text or "")
    if not ans:
        return Verdict()
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(" ".join(ans))

    t_score = _structure_score(ans, target, sm)
    best_struct, best_score = target, t_score
    if pool is not None:
        t_id = _g(target, "id")
        for s in pool:
            if _g(s, "id") == t_id:
                continue
            sc = _structure_score(ans, s, sm)
            if sc > best_score + 1e-9:
                best_struct, best_score = s, sc

    name_ok = t_score > 0.0 and t_score >= best_score - 1e-9
    matched = display_name(best_struct, lang) if best_score > 0.0 else None

    t_side = _g(target, "side", "mid")
    side_ok = t_side not in ("left", "right") or ans_side == t_side
    return _verdict(name_ok, side_ok, name_ok and t_score < 1.0, matched)


# ─── Picking (choice / list / 3D click) ───────────────────────────────────────


def check_pick(target_id: str, picked_id: str | None) -> Verdict:
    """Check an id-based answer (LOCATE click, CHOICE, LIST).

    Equal ids -> correct.  Opposite side of the same structure -> name_ok,
    side_ok False, 0.5 point.  ``None`` (click on nothing) -> 0.
    """
    if not picked_id:
        return Verdict()
    if picked_id == target_id:
        return Verdict(True, True, True, False, target_id, 1.0)
    if is_opposite_side(target_id, picked_id):
        return Verdict(False, True, False, False, target_id, 0.5)
    return Verdict()


# ─── Choices & lists ──────────────────────────────────────────────────────────


def make_choices(
    target: Any,
    pool: Sequence[Any],
    n: int = 4,
    rng: random.Random | None = None,
) -> list[Any]:
    """Target + ``n-1`` distractors, shuffled.

    Distractors come first from the same region and kind, then the same kind,
    then anything.  At most one distractor is the opposite side of the target,
    and no two choices share the same ``name_fr``.  A too-small pool yields
    fewer than ``n`` choices.
    """
    rng = rng or random.Random()
    t_id = _g(target, "id")
    used_names = {_g(target, "name_fr")}
    chosen: list[Any] = []
    opposite_used = False

    cands = sorted((s for s in pool if _g(s, "id") != t_id), key=lambda s: _g(s, "id"))
    kind, region = _g(target, "kind"), _g(target, "region")
    tiers = [
        [s for s in cands if _g(s, "kind") == kind and _g(s, "region") == region],
        [s for s in cands if _g(s, "kind") == kind],
        cands,
    ]
    for tier in tiers:
        tier = list(tier)
        rng.shuffle(tier)
        for s in tier:
            if len(chosen) >= n - 1:
                break
            if s in chosen or _g(s, "name_fr") in used_names:
                continue
            opp = is_opposite_side(t_id, _g(s, "id"))
            if opp and opposite_used:
                continue
            chosen.append(s)
            used_names.add(_g(s, "name_fr"))
            opposite_used = opposite_used or opp
        if len(chosen) >= n - 1:
            break

    out = [target] + chosen
    rng.shuffle(out)
    return out


def full_list(pool: Iterable[Any], lang: str = "fr") -> list[Any]:
    """All structures sorted alphabetically (normalised key) for level LIST."""
    return sorted(
        pool, key=lambda s: (normalize(display_name(s, lang)), _g(s, "id"))
    )


# ─── Grade ────────────────────────────────────────────────────────────────────


def grade_letter(pct: float) -> str:
    """A >= 90, B >= 75, C >= 60, D otherwise."""
    if pct >= 90:
        return "A"
    if pct >= 75:
        return "B"
    if pct >= 60:
        return "C"
    return "D"


# ─── Session ──────────────────────────────────────────────────────────────────


@dataclass
class QuizConfig:
    """Parameters of one quiz.

    ``level`` is ignored in LOCATE mode.  ``superficial_targets_only`` only
    applies to muscles in LOCATE mode (deep muscles cannot be clicked).
    ``regions`` restricts targets *and* the answer pool (choices / list).
    """

    kind: str = "bone"                       # "bone" | "muscle"
    mode: Mode = Mode.IDENTIFY
    level: Level = Level.CHOICE
    n_questions: int = 10
    regions: tuple[str, ...] | None = None
    tier_max: int = 1
    superficial_targets_only: bool = False
    lang: str = "fr"
    seed: int | None = None
    n_choices: int = 4


@dataclass
class Question:
    """One question and its recorded answer."""

    index: int
    target: Any
    choices: list[Any] = field(default_factory=list)  # level CHOICE only
    given: str | None = None       # id (choice/list/pick) or raw text
    verdict: Verdict | None = None
    skipped: bool = False

    @property
    def answered(self) -> bool:
        return self.verdict is not None


class QuizSession:
    """Progression and scoring of a quiz (no Qt)."""

    def __init__(
        self,
        config: QuizConfig,
        pool: Sequence[Any],
        targets: Sequence[Any] | None = None,
    ) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        regions = set(config.regions) if config.regions else None

        self.answer_pool: list[Any] = [
            s
            for s in pool
            if _g(s, "kind") == config.kind
            and int(_g(s, "tier", 1)) <= config.tier_max
            and (regions is None or _g(s, "region") in regions)
        ]
        self._by_id = {_g(s, "id"): s for s in self.answer_pool}

        if targets is None:
            eligible = [
                s
                for s in self.answer_pool
                if not (
                    config.mode == Mode.LOCATE
                    and config.kind == "muscle"
                    and config.superficial_targets_only
                    and _g(s, "layer") != "superficial"
                )
            ]
            eligible.sort(key=lambda s: _g(s, "id"))
            k = min(config.n_questions, len(eligible))
            picked = self._rng.sample(eligible, k)
        else:
            picked = list(targets)
            for s in picked:
                self._by_id.setdefault(_g(s, "id"), s)

        use_choices = config.mode == Mode.IDENTIFY and config.level == Level.CHOICE
        self.questions: list[Question] = [
            Question(
                index=i,
                target=t,
                choices=(
                    make_choices(t, self.answer_pool, config.n_choices, self._rng)
                    if use_choices
                    else []
                ),
            )
            for i, t in enumerate(picked)
        ]
        self._index = 0

    # -- navigation -----------------------------------------------------------

    @property
    def current(self) -> Question | None:
        """Current question, or ``None`` once finished."""
        return self.questions[self._index] if self._index < len(self.questions) else None

    @property
    def finished(self) -> bool:
        return self._index >= len(self.questions)

    @property
    def all_answered(self) -> bool:
        return all(q.answered for q in self.questions)

    @property
    def options(self) -> list[Any]:
        """Full alphabetical list of the pool (level LIST)."""
        return full_list(self.answer_pool, self.config.lang)

    def next(self) -> Question | None:
        """Advance (an unanswered question counts as skipped); return the new current."""
        cur = self.current
        if cur is not None and not cur.answered:
            self.skip()
        if self._index < len(self.questions):
            self._index += 1
        return self.current

    # -- answers --------------------------------------------------------------

    def _need(self, mode: Mode, level: Level | None = None) -> Question:
        cfg = self.config
        if cfg.mode != mode or (level is not None and cfg.level != level):
            raise ValueError(f"not allowed in mode={cfg.mode.value} level={int(cfg.level)}")
        q = self.current
        if q is None:
            raise RuntimeError("quiz finished")
        return q

    def _record(self, q: Question, given: str | None, v: Verdict) -> Verdict:
        q.given, q.verdict = given, v
        return v

    def submit_choice(self, structure_id: str) -> Verdict:
        """Level CHOICE.  Re-submitting returns the recorded verdict."""
        q = self._need(Mode.IDENTIFY, Level.CHOICE)
        if q.verdict is not None:
            return q.verdict
        return self._record(q, structure_id, check_pick(_g(q.target, "id"), structure_id))

    def submit_list(self, structure_id: str) -> Verdict:
        """Level LIST."""
        q = self._need(Mode.IDENTIFY, Level.LIST)
        if q.verdict is not None:
            return q.verdict
        return self._record(q, structure_id, check_pick(_g(q.target, "id"), structure_id))

    def submit_text(self, text: str) -> Verdict:
        """Level TEXT."""
        q = self._need(Mode.IDENTIFY, Level.TEXT)
        if q.verdict is not None:
            return q.verdict
        v = check_text_answer(text, q.target, self.config.lang, self.answer_pool)
        return self._record(q, text, v)

    def submit_pick(self, picked_id: str | None) -> Verdict:
        """LOCATE mode: id of the clicked structure (``None`` = nothing)."""
        q = self._need(Mode.LOCATE)
        if q.verdict is not None:
            return q.verdict
        return self._record(q, picked_id, check_pick(_g(q.target, "id"), picked_id))

    def skip(self) -> Verdict:
        """Give up on the current question (0 point)."""
        q = self.current
        if q is None:
            raise RuntimeError("quiz finished")
        if q.verdict is not None:
            return q.verdict
        q.skipped = True
        return self._record(q, None, Verdict())

    # -- results --------------------------------------------------------------

    @property
    def score(self) -> float:
        return sum(q.verdict.points for q in self.questions if q.verdict)

    @property
    def max_score(self) -> float:
        return float(len(self.questions))

    @property
    def percent(self) -> float:
        return 100.0 * self.score / self.max_score if self.questions else 0.0

    def retry_queue(self) -> list[Any]:
        """Targets not fully right (partial, wrong, skipped, unanswered)."""
        return [
            q.target for q in self.questions if q.verdict is None or q.verdict.points < 1.0
        ]

    def retry_session(self, seed: int | None = None) -> "QuizSession":
        """New session on the missed targets, same settings and answer pool."""
        from dataclasses import replace

        retry = self.retry_queue()
        cfg = replace(self.config, n_questions=len(retry), seed=seed)
        return QuizSession(cfg, self.answer_pool, targets=retry)

    def _given_name(self, q: Question) -> str | None:
        if q.given is None:
            return None
        s = self._by_id.get(q.given)
        if s is not None:
            return display_name(s, self.config.lang)
        if self.config.mode == Mode.IDENTIFY and self.config.level == Level.TEXT:
            return (q.verdict.matched if q.verdict else None) or q.given
        return q.given

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable recap of the session."""
        cfg, lang = self.config, self.config.lang
        rows: list[dict[str, Any]] = []
        by_region: dict[str, int] = {}
        confusions: dict[tuple[str, str], int] = {}
        types = {"wrong_name": 0, "wrong_side": 0, "skipped": 0}
        for q in self.questions:
            v = q.verdict or Verdict()
            name = display_name(q.target, lang)
            given_name = self._given_name(q)
            rows.append(
                {
                    "index": q.index,
                    "target_id": _g(q.target, "id"),
                    "target": name,
                    "region": _g(q.target, "region"),
                    "given": q.given,
                    "given_name": given_name,
                    "skipped": q.skipped or q.verdict is None,
                    **v.to_dict(),
                }
            )
            if v.points >= 1.0:
                continue
            by_region[_g(q.target, "region")] = by_region.get(_g(q.target, "region"), 0) + 1
            if q.skipped or q.verdict is None:
                types["skipped"] += 1
            elif v.name_ok:
                types["wrong_side"] += 1
            else:
                types["wrong_name"] += 1
                if given_name:
                    key = (name, given_name)
                    confusions[key] = confusions.get(key, 0) + 1
        pct = self.percent
        return {
            "kind": cfg.kind,
            "mode": cfg.mode.value,
            "level": int(cfg.level),
            "lang": lang,
            "seed": cfg.seed,
            "n_questions": len(self.questions),
            "score": self.score,
            "max_score": self.max_score,
            "percent": round(pct, 1),
            "grade": grade_letter(pct),
            "questions": rows,
            "error_types": types,
            "errors_by_region": [
                {"region": r, "count": c}
                for r, c in sorted(by_region.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "frequent_errors": [
                {"target": t, "given": g, "count": c}
                for (t, g), c in sorted(
                    confusions.items(), key=lambda kv: (-kv[1], kv[0])
                )[:5]
            ],
            "retry_ids": [_g(t, "id") for t in self.retry_queue()],
        }
