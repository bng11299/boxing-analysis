"""Tests for punch combinations (src/boxingvi/combos.py, score_combinations.py; vault ADR-0019).

Synthetic fixtures only, every expected number worked by hand in the comments. No number here says anything
about boxing; the gate's real result comes from score_combinations.py on labelled own footage.

Run:  python tests/test_combos.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

from boxingvi import combos, report  # noqa: E402
from boxingvi.data import CLASSES, N_JOINTS, Span  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, exc=ValueError, match: str = "") -> None:
    """`match`: a phrase the error must contain, so the test cannot pass on another refusal."""
    try:
        fn()
    except exc as e:
        check(name, match in str(e), f"refused for another reason: {e}")
    else:
        check(name, False, "no exception")


FPS = 30.0
U = report.UNCLASSIFIED


def p(frame, t, unit=0):
    ok = isinstance(frame, (int, float)) and not isinstance(frame, bool)
    return {"frame": frame, "time_s": frame / FPS if ok else 0.0, "punch_type": t, "unit": unit}


def text_of(fn, *args, **kw):
    w = report._Writer()
    fn(w, *args, **kw)
    return " ".join(w.parts), w.numbers


def numbers_ok(text, nums) -> bool:
    """Every number in the text came through the writer, and nothing else did."""
    return Counter(report.numbers_in(text)) == Counter(t for n in nums for t in report.numbers_in(n))


print("== grouping ==")
# 1.0 s at 30 fps is 30 frames: 0 -> 30 joins, 0 -> 31 does not.
runs = combos.group([p(0, "jab"), p(30, "cross")], FPS)
check("two punches exactly 1.0 s apart are one combination", [len(r) for r in runs] == [2])
runs = combos.group([p(0, "jab"), p(31, "cross")], FPS)
check("one frame more than 1.0 s apart: two singles", [len(r) for r in runs] == [1, 1])
runs = combos.group([p(0, "jab"), p(3, "cross", unit=1)], FPS)
check("0.1 s apart but in different units (round, segment): never one combination", [len(r) for r in runs] == [1, 1])
runs = combos.group([p(0, "jab"), p(10, "cross", unit=None), p(20, "jab")], FPS)
check("a punch outside every unit is in no run and does not break the run around it",
      [[q["frame"] for q in r] for r in runs] == [[0, 20]])
runs = combos.group([p(60, "cross"), p(0, "jab"), p(20, "cross"), p(40, "jab")], FPS)
check("input order does not matter: runs are in time order", [[q["frame"] for q in r] for r in runs] == [[0, 20, 40, 60]])
# a chain: each within 1.0 s of the last, 0 to 90 is 3 s overall, still one run
runs = combos.group([p(0, "jab"), p(30, "jab"), p(60, "jab"), p(90, "jab")], FPS)
check("the gap is between consecutive punches, not from the first", [len(r) for r in runs] == [4])
check("half-frame label centres group exactly: 0 and 30.0 join, 0 and 30.5 do not",
      [len(r) for r in combos.group([p(0.0, "jab"), p(30.0, "cross")], FPS)] == [2]
      and [len(r) for r in combos.group([p(0.0, "jab"), p(30.5, "cross")], FPS)] == [1, 1])
check("a different gap is honoured (0.5 s = 15 frames)",
      [len(r) for r in combos.group([p(0, "jab"), p(16, "cross")], FPS, gap_s=0.5)] == [1, 1])
check("no punches: no runs", combos.group([], FPS) == [])
for bad_fps in (0, -30, float("nan"), float("inf"), True, "30"):
    raises(f"fps {bad_fps!r} is refused", lambda f=bad_fps: combos.group([p(0, "jab")], f), match="fps")
for bad_gap in (0, -1, float("nan"), float("inf"), True):
    raises(f"gap {bad_gap!r} is refused", lambda g=bad_gap: combos.group([p(0, "jab")], FPS, gap_s=g), match="gap")
for bad_frame in (-1, float("nan"), float("inf"), True, "5", None, 10 ** 13):
    raises(f"frame {bad_frame!r} is refused", lambda f=bad_frame: combos.group([p(f, "jab")], FPS), match="frame")
raises("an unknown punch type is refused", lambda: combos.group([p(0, "haymaker")], FPS), match="unknown punch type")
check("COMBO_GAP_S is the 1.0 s Brendan chose (changing it voids every gate result)", combos.COMBO_GAP_S == 1.0)

print("== names ==")
check("an unclassified punch is '?'", combos.name_of([p(0, "jab"), p(10, U)]) == ("jab", "?"))
check("spoken names read as words, with no digits",
      combos.spoken(("jab", "cross", "lead_hook")) == "jab-cross-lead hook"
      and not any(ch.isdigit() for t in CLASSES for ch in combos.spoken((t,))))
check("the sheet's boxing numbers: jab-cross-lead hook is 1-2-3, rear uppercut 6",
      combos.numbered(("jab", "cross", "lead_hook")) == "1-2-3" and combos.numbered(("rear_uppercut", "?")) == "6-?")

print("== facts ==")
# combo1 jab-cross (0, 15); combo2 jab-cross (100, 115); combo3 jab-jab-cross (200, 215, 230); single lead hook
# (400); combo4 ?-cross (500, 515). Punches 2+2+3+1+2 = 10; combinations 4 taking 9; longest 3.
# Named: jab-cross x2, jab-jab-cross x1 -> shown [jab-cross (2)] (only names seen twice). Starters (first punch
# named): jab, jab, jab -> jab, 3 of 3. Doubles: jab-jab once. One combination has a '?'.
FIX = [p(0, "jab"), p(15, "cross"), p(100, "jab"), p(115, "cross"), p(200, "jab"), p(215, "jab"), p(230, "cross"),
       p(400, "lead_hook"), p(500, U), p(515, "cross")]
runs = combos.group(FIX, FPS)
cf = combos.compute(runs, 0.4)
check("counts: 10 punches, 4 combinations taking 9, longest 3",
      (cf["n_punches"], cf["n_combos"], cf["n_in_combos"], cf["longest"]) == (10, 4, 9, 3), str(cf))
check("commonest: jab-cross (2) only; a name seen once is not listed", cf["shown"] == [(("jab", "cross"), 2)])
check("starter: the jab, 3 of the 3 combinations whose first punch is named",
      (cf["starter"], cf["starter_n"], cf["n_first_named"]) == (["jab"], 3, 3))
check("doubles: one, the jab", (cf["n_doubles"], cf["doubled"], cf["doubled_n"]) == (1, ["jab"], 1))
check("one combination with an unnamed punch; 3 named", (cf["n_unknown"], cf["n_named"]) == (1, 3))
check("lengths: two of 2... three of 2 and one of 3", cf["lengths"] == Counter({2: 3, 3: 1}))
check("the gate's favourite and starter on the same runs",
      combos.favourite(runs) == [("jab", "cross")] and combos.starter(runs) == ["jab"])
raises("a negative or non-number minimum gap is refused", lambda: combos.compute(runs, -0.1), match="minimum gap")
raises("... and a bool", lambda: combos.compute(runs, True), match="minimum gap")


def pairs_of(names_counts):
    """Runs of two punches, one per (name, count)."""
    out, f = [], 0
    for nm, k in names_counts:
        for _ in range(k):
            out += [p(f, nm[0]), p(f + 10, nm[1])]
            f += 100
    return combos.group(out, FPS)


cf4 = combos.compute(pairs_of([(("jab", "cross"), 2), (("cross", "jab"), 2), (("jab", "jab"), 2),
                               (("lead_hook", "cross"), 2)]), 0.4)
check("four names tie at the top (2 each): none listed, the tie stated instead",
      cf4["shown"] == [] and (cf4["tie_n"], cf4["tie_count"]) == (4, 2), str(cf4["shown"]))
cf3 = combos.compute(pairs_of([(("jab", "cross"), 2), (("cross", "jab"), 2), (("jab", "jab"), 2)]), 0.4)
check("three names tie at the top: all three listed", len(cf3["shown"]) == 3 and cf3["tie_n"] == 0)
cfc = combos.compute(pairs_of([(("jab", "cross"), 3), (("cross", "jab"), 2), (("jab", "jab"), 2),
                               (("lead_hook", "cross"), 2)]), 0.4)
check("a tie is never cut: 3 then three at 2 -> only the 3 is listed (1 + 3 > 3 names)",
      cfc["shown"] == [(("jab", "cross"), 3)], str(cfc["shown"]))
cfd = combos.compute(pairs_of([(("jab", "cross"), 3), (("cross", "jab"), 2), (("jab", "jab"), 2)]), 0.4)
check("... and 3 then two at 2 -> all three listed", len(cfd["shown"]) == 3)
st2 = combos.compute(pairs_of([(("jab", "cross"), 2), (("cross", "jab"), 2)]), 0.4)
check("starters tie: the jab and the cross, 2 each", (st2["starter"], st2["starter_n"]) == (["cross", "jab"], 2))

print("== summary sentences ==")
t, nums = text_of(combos.say_facts, cf)
check("facts: every number came through the writer", numbers_ok(t, nums), t)
check("facts: count, gap, share, longest",
      "It saw 4 combinations (two or more punches, each within 1.0 s of the last), taking 9 of the 10 punches "
      "(90%); the longest was 3 punches." in t, t)
check("facts: the commonest", "The commonest combination was jab-cross (2)." in t, t)
check("facts: the starter", "Of the 3 combinations whose first punch it could name, 3 (100%) started with the jab." in t, t)
check("facts: doubles", "The same punch came twice in a row 1 time inside combinations, most often the jab (1)." in t, t)
check("facts: the unnamed", "1 combination includes a punch it could not name, shown as ?, and is left out" in t, t)
check("facts: the per-boxer sentence only in the per-boxer report", "tracked person" not in t
      and "tracked person" in text_of(combos.say_facts, cf, per_boxer=True)[0])
t2, n2 = text_of(combos.say_facts, st2)
check("tied starters read 'or', with 'each'", "2 each (50% each) started with the cross or the jab." in t2, t2)
tc_, nc_ = text_of(combos.say_facts, cfd)
check("commonest is only the highest count (the gate's favourite); lower counts follow it",
      "The commonest combination was jab-cross (3), followed by cross-jab (2) and jab-jab (2)." in tc_
      and numbers_ok(tc_, nc_), tc_)
t4, n4 = text_of(combos.say_facts, cf4)
check("a tie too wide to list is stated with its numbers", "4 different combinations tie as the commonest, seen 2 times "
      "each." in t4 and numbers_ok(t4, n4), t4)
none = combos.compute(combos.group([p(0, "jab"), p(100, "cross")], FPS), 0.4)
tn, nn = text_of(combos.say_facts, none)
check("no combination: said, with the gap twice through the writer",
      "so the tool saw no combinations" in tn and numbers_ok(tn, nn) and nn.count("1.0") == 2, tn)
check("no combination: no caveat (nothing to qualify)", text_of(combos.say_reliability, none, None)[0] == "")
zero = combos.compute([], 0.4)
check("no punch: nothing at all", text_of(combos.say_facts, zero)[0] == "" and
      text_of(combos.say_reliability, zero, None)[0] == "")
once = combos.compute(pairs_of([(("jab", "cross"), 1), (("cross", "jab"), 1)]), 0.4)
check("named combinations each seen once: said so", "No named combination came up more than once." in
      text_of(combos.say_facts, once)[0])

print("== reliability sentences ==")
tr, nr = text_of(combos.say_reliability, cf, None)
check("unmeasured: the blind spot with the finder's own minimum gap, and 'not checked'",
      "at most one punch per 0.4 s" in tr and "have not been checked against labelled footage" in tr
      and numbers_ok(tr, nr), tr)
check("per-hand punches: the blind spot is per hand",
      "one punch per 0.4 s from each hand" in text_of(combos.say_reliability, combos.compute(runs, 0.4, True), None)[0])
raises("a finder mode that is not true/false is refused", lambda: combos.compute(runs, 0.4, None), match="minimum gap")
check("the minimum gap comes from the rule, not a constant",
      "one punch per 0.3 s" in text_of(combos.say_reliability, combos.compute(runs, 0.3), None)[0])
ta = text_of(combos.say_reliability, cf, {"status": "not_applicable"})[0]
check("another classifier / rule / gap: the check does not apply, and no number from it", "does not apply" in ta
      and "recovered" not in ta)
GOOD = {"status": "pass", "bars": {b: True for b in combos._BARS}, "recovered": 7, "n_label": 20, "recall": 0.35,
        "floor": 0.1}
tp, np_ = text_of(combos.say_reliability, cf, GOOD)
check("pass: the numbers and that the headline matched", "passed their pre-registered check: it recovered 7 of 20 "
      "labelled combinations exactly (35%), against 10% by chance with the punch types shuffled (97.5th percentile)" in tp
      and numbers_ok(tp, np_), tp)
check("pass, per boxer: says it was measured on solo footage",
      "solo own footage (not two boxers)" in text_of(combos.say_reliability, cf, GOOD, per_boxer=True)[0])
BADG = {**GOOD, "status": "fail", "bars": {"1_better_than_chance": False, "2_favourite_matches": True,
                                           "3_starter_matches": False}}
tf, nf = text_of(combos.say_reliability, cf, BADG)
check("fail: which bars failed, the numbers, 'rough'",
      "failed their pre-registered check (exact recovery was not above chance and the punch combinations start with "
      "did not match the labels in both clips)" in tf and "Treat them as rough." in tf and numbers_ok(tf, nf), tf)

print("== in report.summary ==")
facts = report.compute_facts([{**q, "lead_rear": "lead", "type_hand": "lead" if q["punch_type"] in
                               ("jab", "lead_hook", "lead_uppercut") else ("rear" if q["punch_type"] != U else None),
                               "hands_agree": None if q["punch_type"] == U else "yes", "hand": "left"} for q in FIX],
                             [(0.0, 20.0)], 20.0)
TYPES = [report.TypeReliability(c, 10, 10, 0.9, 0.9, 0.9) for c in CLASSES]
BENCH = {"types": TYPES, "side_accuracy": 0.95, "accuracy": 0.9, "n": 60, "videos": ["V2"]}
FINDER = {"recall": 0.74, "count_ratio": 1.6, "video": "V6"}
base, _ = report.summary(facts, BENCH, FINDER, None)
withc, wn = report.summary(facts, BENCH, FINDER, None, combinations=(cf, None, False))
check("without combinations the summary is unchanged (the argument is optional)", "combination" not in base)
check("with them: the facts sit after the punch types, the caveat at the end, every number checked",
      withc.index("It saw 4 combinations") > withc.index("could name") and withc.endswith(
          "have not been checked against labelled footage.") and numbers_ok(withc, wn), withc)
raises("combinations built from other punches than the summary's are refused",
       lambda: report.summary(facts, BENCH, FINDER, None, combinations=(combos.compute(runs[:2], 0.4), None, False)),
       match="other punches")

print("== units for the reports ==")
check("solo: the unit is the round; a punch outside every round has none",
      [q["unit"] for q in combos.solo_punches([p(0, "jab"), p(200, "jab"), p(600, "jab")],
                                              [(0.0, 5.0), (8.0, 25.0)])] == [0, None, 1])


class Seg:
    def __init__(self, track_id, first, last, boxer):
        self.track_id, self.first, self.last, self.boxer = track_id, first, last, boxer


SEGS = (Seg(1, 0, 99, "A"), Seg(2, 100, 300, "A"), Seg(3, 0, 300, "B"))
rows_a = [p(90, "jab"), p(105, "cross"), p(120, "jab")]
fin = [(1, ["90"]), (2, ["105"]), (2, ["120"])]
bp = combos.boxer_punches(rows_a, [(0.0, 20.0)], SEGS, "A", fin)
check("per boxer: the unit is (round, segment)", [q["unit"] for q in bp] == [(0, 0), (0, 1), (0, 1)])
check("per boxer: 0.5 s apart across a track change is NOT one combination (a join is never crossed)",
      [len(r) for r in combos.group(bp, FPS)] == [1, 2])
raises("per boxer: a punch on a track no segment of this boxer covers is refused",
       lambda: combos.boxer_punches([p(90, "jab")], [(0.0, 20.0)], SEGS, "A", [(3, ["90"])]), match="exactly one")
raises("per boxer: a punch with no finder row (no track) is refused",
       lambda: combos.boxer_punches([p(50, "jab")], [(0.0, 20.0)], SEGS, "A", []), match="exactly one")
raises("per boxer: two tracks at one frame for one boxer is refused",
       lambda: combos.boxer_punches(rows_a, [(0.0, 20.0)], SEGS, "A", fin + [(1, ["105"])]), match="two tracks")

print("== the gate: one clip by hand ==")
# Labels (spans, centre = middle): L1 jab 8-12 (10), cross 23-27 (25); L2 jab 98-102 (100), cross 113-117 (115);
# single lead hook 300; L3 jab 500, cross 505 (5 frames = 0.17 s apart: under the 0.4 s blind spot).
J_, X_, LH = CLASSES.index("jab"), CLASSES.index("cross"), CLASSES.index("lead_hook")   # CLASSES is alphabetical
SPANS = [Span(8, 12, J_), Span(23, 27, X_), Span(98, 102, J_), Span(113, 117, X_), Span(300, 300, LH),
         Span(500, 500, J_), Span(505, 505, X_)]
CEN = np.array([(s.start + s.end) / 2 for s in SPANS])
# Detections: 10 jab, 25 cross (recovers L1); 100 jab, 115 rear hook (wrong type: L2 not recovered); 300 lead hook;
# 505 cross (the finder saw one punch of L3).
DET = [p(10, "jab"), p(25, "cross"), p(100, "jab"), p(115, "rear_hook"), p(300, "lead_hook"), p(505, "cross")]
PAIRS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (6, 5)]
sc = combos.score_clip(DET, SPANS, CEN, PAIRS, FPS)
hit, n, precise = combos._recovered(sc, sc["det_types"])
check("exact: 1 of the 3 named label combinations recovered; 1 of 2 found named combinations recovers one",
      (hit, n, precise) == (1, 3, 1), str((hit, n, precise)))
check("structure (no types): L1 and L2 recovered, L3 not (the finder saw one of its two)",
      combos._structure(sc) == (2, 3))
sc_miss = combos.score_clip(DET[:1] + DET[2:], SPANS, CEN, [(0, 0), (2, 1), (3, 2), (4, 3), (6, 4)], FPS)
check("a missed punch: its combination cannot be recovered", combos._recovered(sc_miss, sc_miss["det_types"])[0] == 0)
DET_X = DET[:2] + [p(40, "jab")] + DET[2:]
sc_x = combos.score_clip(DET_X, SPANS, CEN, [(0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (6, 6)], FPS)
check("an extra (unmatched) punch inside a combination: length differs, not recovered",
      combos._recovered(sc_x, sc_x["det_types"])[0] == 0)
DET_U = [p(10, "jab"), p(25, U)] + DET[2:]
sc_u = combos.score_clip(DET_U, SPANS, CEN, PAIRS, FPS)
check("an unnamed found punch: that combination cannot be recovered", combos._recovered(sc_u, sc_u["det_types"])[0] == 0)
SPANS_U = [Span(8, 12, -1)] + SPANS[1:]
sc_lu = combos.score_clip(DET, SPANS_U, CEN, PAIRS, FPS)
check("an unclear label inside a label combination: not a named label combination (2 left)",
      combos._recovered(sc_lu, sc_lu["det_types"])[1] == 2)
raises("the matching must be one-to-one: a detection twice", lambda: combos.score_clip(DET, SPANS, CEN,
       [(0, 0), (1, 0)], FPS), match="one-to-one")
raises("... a label twice", lambda: combos.score_clip(DET, SPANS, CEN, [(0, 0), (0, 1)], FPS), match="one-to-one")
raises("... an index out of range", lambda: combos.score_clip(DET, SPANS, CEN, [(0, 9)], FPS), match="one-to-one")
raises("a label that is neither a type nor unclear is refused",
       lambda: combos.score_clip(DET, [Span(8, 12, 7)] + SPANS[1:], CEN, PAIRS, FPS), match="not a punch type")
raises("one centre per span", lambda: combos.score_clip(DET, SPANS, CEN[:3], PAIRS, FPS), match="one centre")

print("== the gate result ==")
RULE = {"variant": "relative", "threshold": 3.6238817996457016, "min_gap_s": 0.4}
SHA = "a" * 64
PROV = {"classified_csv_sha256": "c" * 64, "labels_sha256": "d" * 64, "fps": FPS, "tolerance_frames": 3, "point_tolerance_frames": 6}
res = combos.gate_result({"C1": (sc, PROV)}, RULE, SHA, {"C1": FPS}, n_perm=200)
d = res["clips"]["C1"]
check("per clip: label combinations 3, named 3, recovered 1, structure 2",
      (d["label_combinations"], d["named_label_combinations"], d["recovered_exactly"], d["structure_recovered"])
      == (3, 3, 1, 2), str(d))
check("the blind spot is counted: 1 label combination has a gap under 0.4 s",
      d["label_combinations_with_a_gap_under_min_gap"] == 1)
check("no name is seen twice in labels or detections: favourite empty, bar 2 fails",
      d["favourite_found"] == [] and res["bars"]["2_favourite_matches"] is False and res["PASS"] is False)
check("starter: found jab (both found combinations start with a jab), labels jab -> matches",
      d["starter_found"] == ["jab"] and d["starter_labels"] == ["jab"] and d["starter_matches"])
check("the floor is reproducible from its seed",
      combos.gate_result({"C1": (sc, PROV)}, RULE, SHA, {"C1": FPS}, n_perm=200)["pooled"]["floor_p975"]
      == res["pooled"]["floor_p975"])
check("PASS is exactly all three bars", res["PASS"] == all(res["bars"].values()))
# A classifier that names every punch "jab" on jab-only labels: shuffling changes nothing, so the floor equals the
# observed recall and bar 1 (strictly greater) fails. A floor that cannot be beaten cannot be passed.
JS = [Span(0, 0, J_), Span(10, 10, J_), Span(200, 200, J_), Span(210, 210, J_)]
JD = [p(0, "jab"), p(10, "jab"), p(200, "jab"), p(210, "jab")]
scj = combos.score_clip(JD, JS, np.array([0.0, 10.0, 200.0, 210.0]), [(i, i) for i in range(4)], FPS)
rj = combos.gate_result({"J": (scj, PROV)}, RULE, SHA, {"J": FPS}, n_perm=50)
check("all one type: recall 1.0 equals its floor, bar 1 fails (strictly greater)",
      rj["pooled"]["exact_recall"] == 1.0 and rj["pooled"]["floor_p975"] == 1.0
      and rj["bars"]["1_better_than_chance"] is False)
check("... while favourite (jab-jab twice) and starter (jab) match", rj["bars"]["2_favourite_matches"]
      and rj["bars"]["3_starter_matches"])
# Two clips that pass everything: alternating jab-cross labels named right; the floor shuffles the 8 types of each
# clip (4 jabs, 4 crosses), and a random arrangement rarely puts all four pairs in order.
AS = [Span(f, f, (J_, X_)[k % 2]) for k, f in enumerate((0, 10, 100, 110, 200, 210, 300, 310))]
AD = [p(s.start, CLASSES[s.label]) for s in AS]
sca = combos.score_clip(AD, AS, np.array([float(s.start) for s in AS]), [(i, i) for i in range(8)], FPS)
REG = {c: FPS for c in combos.GATE_CLIPS}
ra = combos.gate_result({c: (sca, PROV) for c in combos.GATE_CLIPS}, RULE, SHA, REG)
check("a clean pair of clips passes all three bars", ra["PASS"] and ra["pooled"]["exact_recall"] == 1.0
      and ra["pooled"]["floor_p975"] < 1.0, str(ra["pooled"]))

print("== reading a gate result back ==")
tmp = Path(tempfile.mkdtemp())


def write(d_, name="gate.json"):
    f = tmp / name
    f.write_text(json.dumps(d_), encoding="utf-8")
    return f


g = combos.read_gate(write(ra), RULE, SHA)
check("round trip: a passing result reads as pass with its numbers",
      g["status"] == "pass" and (g["recovered"], g["n_label"]) == (8, 8) and g["floor"] == ra["pooled"]["floor_p975"])
rf = combos.gate_result({c: (sc, PROV) for c in combos.GATE_CLIPS}, RULE, SHA, REG)
check("round trip: a failing result reads as fail", combos.read_gate(write(rf), RULE, SHA)["status"] == "fail"
      and rf["PASS"] is False)
check("exploratory run, other clips: does not apply (never the gate)",
      combos.read_gate(write(res), RULE, SHA)["status"] == "not_applicable")
check("exploratory run, other permutation count: does not apply",
      combos.read_gate(write({**ra, "pooled": {**ra["pooled"], "permutations": 200}}), RULE, SHA)["status"]
      == "not_applicable")
check("exploratory run, other seed: does not apply",
      combos.read_gate(write({**ra, "pooled": {**ra["pooled"], "seed": 1}}), RULE, SHA)["status"] == "not_applicable")
check("punches from the per-hand finder: the pooled gate does not apply",
      combos.read_gate(write(ra), RULE, SHA, per_hand=True)["status"] == "not_applicable")
check("punches whose finder mode is unknown (old sidecar): does not apply",
      combos.read_gate(write(ra), RULE, SHA, per_hand=None)["status"] == "not_applicable")
check("a result recorded for the per-hand finder: does not apply to pooled punches",
      combos.read_gate(write({**ra, "finder_per_hand": True}), RULE, SHA)["status"] == "not_applicable")
check("another classifier: does not apply", combos.read_gate(write(ra), RULE, "b" * 64)["status"] == "not_applicable")
check("another threshold: does not apply",
      combos.read_gate(write(ra), {**RULE, "threshold": 3.0}, SHA)["status"] == "not_applicable")
check("another minimum gap (a fast-combination finder): does not apply",
      combos.read_gate(write(ra), {**RULE, "min_gap_s": 0.2}, SHA)["status"] == "not_applicable")
check("measured with another combination gap: does not apply",
      combos.read_gate(write({**ra, "gap_s": 0.75}), RULE, SHA)["status"] == "not_applicable")


def bad(name, d_, match):
    raises(f"refused: {name}", lambda: combos.read_gate(write(d_), RULE, SHA), match=match)


bad("an extra key", {**ra, "note": "x"}, "not a boxingvi-combo-gate-1")
C0 = combos.GATE_CLIPS[0]


def with_clip(**kw):
    return {**ra, "clips": {**ra["clips"], C0: {**ra["clips"][C0], **kw}}}


bad("no finder mode", {k: v for k, v in ra.items() if k != "finder_per_hand"}, "not a boxingvi-combo-gate-1")
bad("a finder mode that is not true/false", {**ra, "finder_per_hand": "no"}, "finder_per_hand")
bad("clip evidence emptied", {**ra, "clips": {}}, "no clips")
bad("a clip block emptied", {**ra, "clips": {**ra["clips"], C0: {}}}, "not a gate clip block")
bad("bar 2 true while a clip's favourite did not match",
    with_clip(favourite_matches=False, favourite_found=["cross-jab"]), "bars 2 and 3")
bad("a clip's favourite match that contradicts its lists", with_clip(favourite_found=["cross-jab"]),
    "is not what its lists say")
bad("a clip's starter match that contradicts its lists", with_clip(starter_labels=["cross"]),
    "is not what its lists say")
bad("a list naming something that is not a punch type", with_clip(favourite_found=["jab-haymaker"],
                                                                  favourite_labels=["jab-haymaker"]),
    "not a punch type")
bad("clip counts that contradict each other", with_clip(recovered_exactly=99), "contradict")
bad("pooled counts that are not the sums of the clips'",
    {**ra, "pooled": {**ra["pooled"], "recovered_exactly": 7, "exact_recall": 7 / 8}}, "not the sums")
bad("a pooled block with a key missing", {**ra, "pooled": {k: v for k, v in ra["pooled"].items() if k != "seed"}},
    "no pooled result")
bad("a missing key", {k: v for k, v in ra.items() if k != "pooled"}, "not a boxingvi-combo-gate-1")
bad("another format", {**ra, "format": "boxingvi-combo-gate-0"}, "not a boxingvi-combo-gate-1")
bad("another gate document", {**ra, "document": "docs/gates/other.md"}, "not a boxingvi-combo-gate-1")
bad("PASS true over a failed bar", {**res, "PASS": True}, "PASS is not what its bars say")
bad("a bar that is not true/false", {**ra, "bars": {**ra["bars"], "2_favourite_matches": 1}}, "true or false")
bad("a bar missing", {**ra, "bars": {k: v for k, v in ra["bars"].items() if k != "3_starter_matches"}}, "true or false")
bad("more recovered than labelled", {**ra, "pooled": {**ra["pooled"], "recovered_exactly": 9}}, "not the sums")
bad("a recall that is not recovered / labelled", {**ra, "pooled": {**ra["pooled"], "exact_recall": 0.5}},
    "not recovered / labelled")
bad("bar 1 true while recall does not beat the floor", {**ra, "pooled": {**ra["pooled"], "floor_p975": 1.0}},
    "bar 1 is not what")
bad("a count that is a bool", {**ra, "pooled": {**ra["pooled"], "recovered_exactly": True}}, "whole number")
bad("a negative count", {**ra, "pooled": {**ra["pooled"], "named_label_combinations": -1}}, "whole number")
bad("a floor out of [0, 1]", {**ra, "pooled": {**ra["pooled"], "floor_p975": 2}}, "floor")
bad("a huge integer recall", {**ra, "pooled": {**ra["pooled"], "exact_recall": 10 ** 400}}, "exact recall")
bad("no classifier sha", {**ra, "classifier_sha256": "abc"}, "no classifier sha256")
bad("no rule", {**ra, "rule": None}, "no finder rule")
bad("a gap that is not a number", {**ra, "gap_s": "1.0"}, "gap")
(tmp / "nan.json").write_text(json.dumps(ra).replace('"exact_recall": 1.0', '"exact_recall": NaN'), encoding="utf-8")
raises("refused: NaN", lambda: combos.read_gate(tmp / "nan.json", RULE, SHA), match="NaN")
(tmp / "big.json").write_bytes(b" " * (combos.MAX_GATE_BYTES + 1))
raises("refused: an oversized file, read no further than the cap", lambda: combos.read_gate(tmp / "big.json", RULE, SHA),
       match="more than")
(tmp / "arr.json").write_text("[1]", encoding="utf-8")
raises("refused: not an object", lambda: combos.read_gate(tmp / "arr.json", RULE, SHA), match="not a boxingvi")
(tmp / "bin.json").write_bytes(b"\xff\xfe")
raises("refused: not UTF-8", lambda: combos.read_gate(tmp / "bin.json", RULE, SHA), match="not valid JSON")
raises("refused: the report's own rule is malformed", lambda: combos.read_gate(write(ra), {"variant": "relative"}, SHA),
       match="rule")

print("== score_combinations.py end to end (synthetic files) ==")
import score_combinations  # noqa: E402
from boxingvi import pose_yolo  # noqa: E402

n_fr = 600
d_ = tmp / "punches"
d_.mkdir()
lab_dir, pose_dir = tmp / "labels", tmp / "pose"
lab_dir.mkdir()
pose_dir.mkdir()


def classified_row(frame, t):
    lead = t in ("jab", "lead_hook", "lead_uppercut")
    side = "lead" if lead else "rear"
    return [str(frame), f"{frame / FPS:.3f}", "left" if lead else "right", "5.000", "no", side, t, "0.900", side,
            "yes", str(frame - 3), str(frame + 3), "7", ""]


def make_clip(clip, dets, spans, rule=RULE, per_hand=False, classifier_sha=None):
    pose = pose_yolo.PoseSequence(np.zeros((n_fr, N_JOINTS, 2), np.float32), np.zeros((n_fr, N_JOINTS), np.float32),
                                  np.zeros(n_fr, bool), FPS, 1080, 1920, f"{clip}.mov", "test")
    pose.save(pose_dir / f"{clip}.pose.npz")
    import csv as _csv
    import io as _io
    buf = _io.StringIO(newline="")
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(report.CLASSIFIED_HEADER)
    for f, t in dets:
        w.writerow(classified_row(f, t))
    data = buf.getvalue().encode("utf-8")
    (d_ / f"{clip}.classified.csv").write_bytes(data)
    (d_ / f"{clip}.classified.json").write_text(json.dumps({
        "clip": clip, "csv_sha256": hashlib.sha256(data).hexdigest(), "classifier_sha256": classifier_sha,
        "stance": "orthodox", "rule": rule, "finder_per_hand": per_hand, "extent": {}, "cut": None,
        "pose_frames": n_fr, "fps": FPS}), encoding="utf-8")
    (lab_dir / f"{clip}.labels.json").write_text(json.dumps({
        "source": f"{clip}.mov", "fps": FPS, "n_frames": n_fr,
        "spans": [{"start": a, "end": b, "label": lab} for a, b, lab in spans]}), encoding="utf-8")


from boxingvi import classify  # noqa: E402

ck = tmp / "clf.pt"
ck.write_bytes(b"not loaded by the scorer")
GEOM = classify.WindowGeometry(0.375, 0.125, 24.0, 200.0, 0.5, 0.55, "relative", 100)
mpath = tmp / "clf.json"
classify.write_manifest(mpath, ck, GEOM, 50, {})
MSHA = classify.read_manifest(mpath)["sha256"]
# Clips E1 and E2: labels jab 30 / cross 45 (L1), jab 300 / cross 315 (L2); detections at the same frames, all named
# right. Both recover 2 of 2; favourite jab-cross (2) and starter jab match the labels in both.
SP = [(30, 30, "jab"), (45, 45, "cross"), (300, 300, "jab"), (315, 315, "cross")]
DT = [(30, "jab"), (45, "cross"), (300, "jab"), (315, "cross")]
for c in ("E1", "E2"):
    make_clip(c, DT, SP, classifier_sha=MSHA)
out = tmp / "res" / "gate.json"
ARGS = ["--clip", "E1", "--clip", "E2", "--punches-dir", str(d_), "--labels-dir", str(lab_dir), "--pose-dir",
        str(pose_dir), "--classifier", str(mpath), "--out", str(out), "--permutations", "200"]
import contextlib  # noqa: E402
import io  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    rc = score_combinations.main(ARGS)
r = json.loads(out.read_text(encoding="utf-8"))
check("CLI: runs and writes a well-formed result; with other clips and 200 permutations it is exploratory, so a "
      "report does not apply it", rc == 0 and combos.read_gate(out, RULE, MSHA)["status"] == "not_applicable")
check("CLI: 4 of 4 named label combinations recovered across the two clips, favourite and starter match",
      r["pooled"]["recovered_exactly"] == 4 and r["pooled"]["named_label_combinations"] == 4
      and r["bars"]["2_favourite_matches"] and r["bars"]["3_starter_matches"], str(r["pooled"]))
check("CLI: the result records each clip's classified CSV and labels by sha256",
      all(len(r["clips"][c]["classified_csv_sha256"]) == 64 and len(r["clips"][c]["labels_sha256"]) == 64
          for c in ("E1", "E2")))
check("CLI: four label punches", r["clips"]["E1"]["label_punches"] == 4)
make_clip("E3", DT, SP + [(100, 100, "unclear")], classifier_sha=MSHA)
with contextlib.redirect_stdout(io.StringIO()):
    score_combinations.main(ARGS[:2] + ["--clip", "E3"] + ARGS[4:])
check("CLI: an unclear label is framing, not a punch (as RESULTS 6a/6b score it): still 4 label punches",
      json.loads(out.read_text(encoding="utf-8"))["clips"]["E3"]["label_punches"] == 4)


def cli_refused(name, args, match):
    with contextlib.redirect_stdout(io.StringIO()):
        raises(f"CLI refuses {name}", lambda: score_combinations.main(args), match=match)


cli_refused("a clip given twice", ARGS[:2] + ARGS[:2] + ARGS[4:], "each --clip once")
cli_refused("a clip name with a path in it", ["--clip", "../E1"] + ARGS[2:], "plain clip name")
make_clip("R1", DT, SP, rule={**RULE, "threshold": 3.0}, classifier_sha=MSHA)
cli_refused("clips found with different rules", ARGS[:2] + ["--clip", "R1"] + ARGS[4:], "one rule")
make_clip("P1", DT, SP, per_hand=True, classifier_sha=MSHA)
cli_refused("the per-hand finder", ["--clip", "P1"] + ARGS[4:], "pooled finder")
make_clip("P2", DT, SP, per_hand=None, classifier_sha=MSHA)
cli_refused("a sidecar that does not state the finder mode", ["--clip", "P2"] + ARGS[4:], "pooled finder")
make_clip("O1", DT, SP, classifier_sha="b" * 64)
cli_refused("punches named by another classifier", ["--clip", "O1"] + ARGS[4:], "another classifier")
make_clip("T1", DT, SP, classifier_sha=MSHA)
(d_ / "T1.classified.csv").write_bytes((d_ / "T1.classified.csv").read_bytes().replace(b"cross", b"jab"))
cli_refused("a classified CSV edited after its sidecar", ["--clip", "T1"] + ARGS[4:], "has changed")
make_clip("N1", DT, SP, classifier_sha=MSHA)
(d_ / "N1.classified.json").unlink()
cli_refused("a CSV with no sidecar", ["--clip", "N1"] + ARGS[4:], "no N1.classified.json")
make_clip("L1", DT, SP, classifier_sha=MSHA)
lab = json.loads((lab_dir / "L1.labels.json").read_text(encoding="utf-8"))
(lab_dir / "L1.labels.json").write_text(json.dumps({**lab, "n_frames": n_fr + 1}), encoding="utf-8")
cli_refused("labels for another video", ["--clip", "L1"] + ARGS[4:], "not the same video")
cli_refused("a permutation count out of range", ARGS[:-1] + ["0"], "permutations")
from boxingvi import boxers as _bx  # noqa: E402
(tmp / "participant").mkdir()
raises("outputs from participant inputs must stay under the participant folder (the CLI calls this check)",
       lambda: _bx.check_output_location([tmp / "participant" / "x"], tmp / "res", (tmp / "participant",)),
       match="participant")

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
