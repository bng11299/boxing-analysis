"""Tests for the report: reading classified punches, provenance, rounds,
reliability, the template summary and the spreadsheet, and report_punches.py end
to end.

Synthetic fixtures only: planted punches with every expected number worked by
hand in the comments. No number here says anything about boxing.

Run:  python tests/test_report.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from boxingvi import classify, pose_yolo, report, stats  # noqa: E402
from boxingvi.data import CLASSES, N_JOINTS  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc:
        check(name, True)
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


tmp = Path(tempfile.mkdtemp())
FPS = 60.0
WHY = "pose in only 1 of the window's 9 frames"


def row(t, ptype, hand="left", conf=0.9, lead="left", note=""):
    """A classify_punches.py row; `lead` is the stance's lead hand."""
    side = "lead" if hand == lead else "rear"
    f = int(t * FPS)
    if ptype == "unclassified":
        return [str(f), f"{t:.3f}", hand, "5.000", "no", side, "unclassified", "", "", "", str(f - 6), str(f + 17),
                "1", WHY]
    th = classify.class_side(ptype)
    return [str(f), f"{t:.3f}", hand, "5.000", "no", side, ptype, f"{conf:.3f}", th,
            "yes" if th == side else "no", str(f - 6), str(f + 17), "9", note]   # a 24-frame punch: 0.4 s at 60 fps


def write_csv(path, rows, header=report.CLASSIFIED_HEADER):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def provenance_ok(text: str, nums) -> bool:
    return Counter(report.numbers_in(text)) == Counter(t for n in nums for t in report.numbers_in(n))


print("the CSV contract with classify_punches.py")
import classify_punches  # noqa: E402

check("report reads exactly the columns classify_punches writes", report.CLASSIFIED_HEADER == classify_punches.OUT_HEADER)

# Round 1 (0-60 s): 6 jabs, 2 crosses, 1 lead hook, 1 unnamed = 10. Round 2 (90-150 s): 3 jabs, 1 cross = 4.
# One jab at 75 s, outside the rounds. One of round 1's jabs is thrown by the right (rear) wrist: a disagreement.
R1 = ([row(2 + 5 * i, "jab") for i in range(5)] + [row(27, "jab", "right"), row(40, "cross", "right"),
      row(45, "cross", "right"), row(50, "lead_hook"), row(55, "unclassified", "right")])
R2 = [row(95, "jab"), row(100, "jab"), row(110, "jab"), row(120, "cross", "right")]
GAP = [row(75, "jab")]
DUR = 160.0
good = write_csv(tmp / "IMG_1.classified.csv", R1 + GAP + R2)
rows = report.read_classified(good, DUR)
check("every row read, sorted by frame", len(rows) == 15 and [r["frame"] for r in rows] == sorted(r["frame"] for r in rows))
check("an unclassified row has no confidence", [r["confidence"] for r in rows if r["punch_type"] == "unclassified"] == [None])


def bad(name, rows_, header=report.CLASSIFIED_HEADER):
    p = write_csv(tmp / "bad.csv", rows_, header)
    raises(name, lambda: report.read_classified(p, DUR))


bad("another header is refused", [row(1, "jab")], header=report.CLASSIFIED_HEADER[:-1])
bad("a time past the clip is refused", [row(170, "jab")])
for t_, why in (("nan", "NaN"), ("1_0", "underscored"), (" 1.0", "padded"), ("-1.0", "negative")):
    r_ = row(1, "jab"); r_[1] = t_; bad(f"a {why} time is refused", [r_])
r_ = row(1, "jab"); r_[6] = "haymaker"; bad("an unknown punch type is refused", [r_])
r_ = row(1, "jab"); r_[7] = "1.500"; bad("a confidence above 1 is refused", [r_])
r_ = row(1, "jab"); r_[7] = ""; bad("a named punch without a confidence is refused", [r_])
r_ = row(1, "unclassified"); r_[7] = "0.5"; bad("an unclassified punch carrying a confidence is refused", [r_])
r_ = row(1, "unclassified"); r_[9] = "yes"; bad("an unclassified punch carrying an agreement flag is refused", [r_])
r_ = row(1, "unclassified"); r_[13] = ""; bad("an unclassified punch without its reason is refused", [r_])
r_ = row(1, "jab"); r_[10] = ""; r_[11] = ""; bad("a classified punch without the frames it was read from is refused", [r_])
r_ = row(1, "jab"); r_[10] = str(int(FPS) + 5); bad("punch frames that do not contain the peak are refused", [r_])
r_ = row(1, "jab"); r_[11] = "1e3"; bad("a punch frame that is not a whole number is refused", [r_])
r_ = row(0.05, "jab"); check("a punch whose estimated start falls before frame 0 is read (start -3)",
                              report.read_classified(write_csv(tmp / "neg.csv", [r_]), DUR)[0]["span"] == (-3, 20))
r_ = row(1, "jab"); r_[8] = "rear"; bad("a type hand that does not follow from the type is refused", [r_])
r_ = row(1, "jab"); r_[9] = "no"; bad("an agreement flag that does not follow from the hands is refused", [r_])
r_ = row(1, "jab"); r_[2] = "both"; bad("an unknown hand is refused", [r_])
r_ = row(1, "jab"); r_[0] = "²"; bad("a non-ASCII digit is refused", [r_])
r_ = row(1, "unclassified"); r_[13] = "=HYPERLINK(\"http://x\")"; bad("a note classify never writes is refused", [r_])
bad("a short row is refused", [row(1, "jab")[:5]])
bad("a named punch carrying an unnamed punch's reason is refused", [row(1, "jab", note=WHY)])
bad("a named punch carrying any note is refused: the pipeline writes none", [row(1, "jab", note="probably fine")])
r_ = row(1, "unclassified"); r_[13] = "probably fine"; bad("an unnamed punch with an unknown reason is refused", [r_])
bad("two stances in one file are refused", [row(1, "jab", lead="left"), row(2, "jab", lead="right")])
check("a file in one southpaw stance is read", len(report.read_classified(
    write_csv(tmp / "sp.csv", [row(1, "jab", "right", lead="right"), row(2, "cross", "left", lead="right")]), DUR)) == 2)
big = write_csv(tmp / "big.csv", [row(1, "jab")] * 3)
report.MAX_ROWS, keep = 2, report.MAX_ROWS
raises("more rows than the cap are refused", lambda: report.read_classified(big, DUR))
report.MAX_ROWS = keep

print("rounds")
check("default: the whole clip is one round", report.parse_rounds(None, DUR) == [(0.0, DUR)])
check("explicit rounds parse in seconds", report.parse_rounds("0-60, 90-150", DUR) == [(0.0, 60.0), (90.0, 150.0)])
for txt, why in (("0-60,50-120", "overlapping"), ("0-250", "past the clip"), ("60-60", "empty"),
                 ("a-b", "not numbers"), ("0-60;90-150", "wrong separator"), ("-5-10", "negative"),
                 (",".join(f"{i}-{i + 1}" for i in range(101)), "more than 100")):
    raises(f"rounds: {why} is refused", lambda txt=txt: report.parse_rounds(txt, 200.0))

print("facts, worked by hand")
ROUNDS = report.parse_rounds("0-60,90-150", DUR)
facts = report.compute_facts(rows, ROUNDS, DUR)
check("14 punches inside rounds, 1 outside", facts["n_found"] == 14 and facts["n_outside_rounds"] == 1)
check("13 named, 1 not", facts["n_named"] == 13 and facts["n_unnamed"] == 1)
check("types over both rounds: 9 jabs, 3 crosses, 1 lead hook",
      facts["types"] == {"cross": 3, "jab": 9, "lead_hook": 1, "lead_uppercut": 0, "rear_hook": 0, "rear_uppercut": 0})
check("rates: 10 a minute in round 1, 4 in round 2, 7 overall",
      [p["per_min"] for p in facts["rounds"]] == [10.0, 4.0] and facts["per_min"] == 7.0)
check("lead hand by TYPE: 10 of 13 named (9 jabs, 1 lead hook)", facts["n_type_lead"] == 10)
check("one punch where the type's hand and the wrist disagree", facts["n_hands_disagree"] == 1)
check("a punch exactly at the end of the last round counts in it",
      report.round_of(150.0, ROUNDS) == 1 and report.round_of(60.0, ROUNDS) is None)
check("shares add to 100 (largest remainder): 9/13, 3/13, 1/13 -> 69, 23, 8",
      report.shares({"a": 9, "b": 3, "c": 1}) == {"a": 69, "b": 23, "c": 8})
check("thirds add to 100 too", sum(report.shares({"a": 1, "b": 1, "c": 1}).values()) == 100)

print("reliability")
# Rows true, columns predicted, CLASSES order: cross, jab, lead_hook, lead_uppercut, rear_hook, rear_uppercut.
C = np.array([[90, 5, 0, 0, 5, 0],
              [2, 98, 0, 0, 0, 0],
              [0, 0, 80, 20, 0, 0],
              [0, 10, 40, 50, 0, 0],
              [30, 0, 0, 0, 20, 0],
              [10, 0, 0, 0, 0, 90]])
F1 = {c: 2 * int(C[i, i]) / int(C[i].sum() + C[:, i].sum()) for i, c in enumerate(CLASSES)}
ref = tmp / "stgcnpp.json"
ref.write_text(json.dumps({"val_videos": ["V2", "V5", "V9"], "results": [
    {"model": "st-gcn++ pretrained", "confusion": C.tolist(), "per_class_f1": F1}]}), encoding="utf-8")
ck = tmp / "clf.pt"
ck.write_bytes(b"not loaded by the report")
GEOM = classify.WindowGeometry(0.375, 0.125, 24.0, 200.0, 0.5, 0.55, "relative", 100)
mpath = tmp / "clf.json"
classify.write_manifest(mpath, ck, GEOM, 50, {"held_out": {"per_class_f1": F1}})
man = classify.read_manifest(mpath)
bench = report.benchmark_reliability(ref, man)
t = {x.cls: x for x in bench["types"]}
check("cross: 100 true, 132 counted (+32%), recall 0.90",
      t["cross"].true == 100 and t["cross"].predicted == 132 and abs(t["cross"].recall - 0.9) < 1e-12)
check("rear hook: 50 true, 25 counted, recall 0.40", t["rear_hook"].true == 50 and t["rear_hook"].predicted == 25
      and abs(t["rear_hook"].recall - 0.4) < 1e-12)
# Off-side cells: cross read as jab (5), jab read as cross (2); 550 punches in all.
check("type-implied hand right on 543 of 550", abs(bench["side_accuracy"] - 543 / 550) < 1e-12)


def manifest_f1(f1):
    p = tmp / "m2.json"
    classify.write_manifest(p, ck, GEOM, 50, {"held_out": {"per_class_f1": f1}})
    return classify.read_manifest(p)


raises("a manifest whose per-type F1 differs does not borrow this reliability",
       lambda: report.benchmark_reliability(ref, manifest_f1({**F1, "jab": 0.6})))
raises("a manifest with no held-out scores is refused", lambda: report.benchmark_reliability(ref, manifest_f1(None)))
raises("a non-numeric F1 is refused cleanly", lambda: report.benchmark_reliability(ref, manifest_f1({**F1, "jab": "x"})))


def ref_with(**kw):
    p = tmp / "ref2.json"
    r = {"model": "st-gcn++ pretrained", "confusion": C.tolist(), "per_class_f1": F1, **kw}
    p.write_text(json.dumps({"results": [r]}), encoding="utf-8")
    return p


raises("a confusion matrix of the wrong shape is refused",
       lambda: report.benchmark_reliability(ref_with(confusion=C[:5].tolist()), man))
raises("a fractional confusion cell is refused, not truncated",
       lambda: report.benchmark_reliability(ref_with(confusion=(C + 0.5).tolist()), man))
raises("an F1 that the matrix does not reproduce is refused (matrix and F1 from different runs)",
       lambda: report.benchmark_reliability(ref_with(per_class_f1={c: 0.5 for c in CLASSES}),
                                            manifest_f1({c: 0.5 for c in CLASSES})))
(tmp / "nan.json").write_text('{"results": NaN}', encoding="utf-8")
raises("NaN in a JSON input is refused", lambda: report.benchmark_reliability(tmp / "nan.json", man))
(tmp / "huge.json").write_text(json.dumps({"x": "y" * (report.MAX_JSON_BYTES + 10)}), encoding="utf-8")
raises("an oversized JSON input is refused", lambda: report.benchmark_reliability(tmp / "huge.json", man))
check("held-out videos come from the reference", bench["videos"] == ["V2", "V5", "V9"])
rule = tmp / "rule.json"
rule.write_text(json.dumps({"video": "V6", "chosen": {"variant": "relative", "test": {
    "recall": 0.741, "precision": 0.465, "count_ratio": 1.596, "n_labelled": 406}}}), encoding="utf-8")
finder = report.finder_reliability(rule)
check("finder reliability: held-out recall and count ratio", finder == {"recall": 0.741, "count_ratio": 1.596, "video": "V6"})
(tmp / "rule2.json").write_text(json.dumps({"chosen": {"variant": "relative"}}), encoding="utf-8")
raises("a finder rule with no held-out score is refused", lambda: report.finder_reliability(tmp / "rule2.json"))

print("summary: templates, filled")
text, nums = report.summary(facts, bench, finder, None)
print("    " + text)
check("the text holds exactly the numbers the templates inserted, and no other", provenance_ok(text, nums),
      f"{report.numbers_in(text)} vs {nums}")
# Worked by hand: column sums 132, 113, 120, 70, 25, 90 -> cross +32%, lead hook +20% (not over 20%: not reported),
# lead uppercut -30%, rear hook -50%; recall under 0.70: lead uppercut 0.50, rear hook 0.40.
# Rates 10 (10 punches / 1 min) and 4 (4 / 1 min): SE = sqrt(10 + 4) = 3.74, 2 SE = 7.5 > 6, so within chance.
for want in ("In 2 rounds (2:00 of footage) the tool found 14 punches, 7.0 a minute.",
             "Of the 13 it could name, 69% jabs (9), 23% crosses (3) and 8% lead hooks (1).",
             "By punch type, 77% were thrown with your lead hand.",
             "For 1 punch the punch type and the wrist that moved point to different hands",
             "1 it could not name: the pose was too incomplete to read.",
             "Your punch rate went from 10 a minute in round 1 to 4.0 in round 2: a difference chance alone could produce.",
             "1 punch fell outside the rounds and is not counted.",
             "The tool names these least reliably: lead uppercuts and rear hooks.",
             "it named only 50% of lead uppercuts and 40% of rear hooks correctly.",
             "On the benchmark, type counts ran high for crosses (by 32%), and low for lead uppercuts (by 30%) "
             "and rear hooks (by 50%).",
             "The hand a punch type implies was right 99% of the time on the benchmark; which wrist moved has not "
             "been checked against labels.",
             "On benchmark video V6, the only one with continuous labels, it found 74% of the labelled punches but "
             "flagged 1.6 times as many punches as were labelled",
             "how far the totals above are off is not known",
             "None of this has been measured on your own footage yet"):
    check(f"says: {want!r}", want in text, text)


def variant(facts_, own_=None, name=""):
    tx, nu = report.summary(facts_, bench, finder, own_)
    check(f"{name}: numbers only from the templates", provenance_ok(tx, nu), f"{report.numbers_in(tx)} vs {nu}")
    return tx


def facts_of(rows_, rounds_txt, dur=DUR):
    p = write_csv(tmp / "v.csv", rows_)
    return report.compute_facts(report.read_classified(p, dur), report.parse_rounds(rounds_txt, dur), dur)


busy = facts_of([row(i * 0.5, "jab") for i in range(1, 118)] + [row(90 + i * 1.0, "jab") for i in range(50)],
                "0-60,90-150")
tb = variant(busy, name="a real drop")
# 117 then 50 a minute: SE = sqrt(117 + 50) = 12.9, 2 SE = 25.8 < 67.
check("a change beyond chance is described from the printed rates: (50/117 - 1) = -57%",
      "Your punch rate dropped by 57%, from 117 a minute in round 1 to 50 in round 2." in tb, tb)
near = facts_of([row(i * 0.5 + 0.25, "jab") for i in range(104)] + [row(90 + i * 0.5 + 0.25, "jab") for i in range(96)],
                "0-60,90-150")
tn = variant(near, name="104 vs 96")
check("104 then 96 a minute is within chance, not 'dropped by 8%'", "chance alone" in tn and "dropped" not in tn, tn)
one_p = facts_of([row(30, "jab")], "0-180", dur=180.0)
t1 = variant(one_p, name="one punch")
check("1 punch in 3:00 is '0.3 a minute', never '0 a minute'", "found 1 punch, 0.3 a minute." in t1, t1)
lop = facts_of([row(0.1 + 0.05 * i, "jab") for i in range(299)] + [row(20, "rear_hook", "right")], None)
tl = variant(lop, name="299 to 1")
check("1 rear hook in 300 is 'under 1%', not '0%'", "under 1% rear hooks (1)" in tl and " 0% " not in tl, tl)
check("299 lead punches of 300 are 'over 99%', not '100%'", "over 99% were thrown with your lead hand" in tl, tl)
check("one round: no trend sentence, and 'one round' in words", "one round" in tl and "punch rate" not in tl)
rare = facts_of([row(0.1 + 0.05 * i, "cross", "right") for i in range(299)] + [row(20, "jab")], None)
tr_ = variant(rare, name="1 lead in 300")
check("1 lead-hand punch in 300 is 'under 1%' of the lead hand, not '0%'",
      "under 1% were thrown with your lead hand" in tr_, tr_)
t0 = variant(report.compute_facts([], ROUNDS, DUR), name="no punches")
check("no punches: says so, and still gives the caveats",
      "found no punches" in t0 and "least reliably" in t0 and "punch rate" not in t0)
to = variant(facts, {"n": 12, "unclassifiable": 2, "accuracy": 0.75, "floor": 0.5}, name="own labels")
check("with own labels: measured accuracy next to its floor, and the matched punches it could not name",
      "named 75% of 12 matched punches correctly; always guessing your commonest punch would score 50%." in to
      and "2 more matched punches could not be named." in to and "not been measured" not in to, to)
tz = variant(facts, {"n": 0, "unclassifiable": 0}, name="labelled, nothing matched")
check("labelled but nothing matched: not 'label a clip'", "You labelled this clip, but no labelled punch" in tz
      and "label a clip" not in tz, tz)

print("the hand check: one-sided exact binomial, worked by hand")
B = stats.binomial_greater_p
for (k_, n_, p_), want, why in (((4, 4, 0.5), 1 / 16, "4 of 4 at 1/2"), ((5, 5, 0.5), 1 / 32, "5 of 5 at 1/2"),
                               ((8, 10, 0.5), 56 / 1024, "8 of 10 at 1/2: (45 + 10 + 1) / 1024"),
                               ((9, 10, 0.5), 11 / 1024, "9 of 10 at 1/2: (10 + 1) / 1024"),
                               ((2, 3, 0.2), 0.104, "2 of 3 at 0.2: 3(0.04)(0.8) + 0.008"),
                               ((1, 2, 0.3), 0.51, "1 of 2 at 0.3: 1 - 0.7^2"),
                               ((0, 7, 0.4), 1.0, "0 of anything: certain"),
                               ((3, 3, 1.0), 1.0, "p = 1: certain"), ((1, 3, 0.0), 0.0, "p = 0, k > 0: impossible"),
                               ((0, 0, 0.5), 1.0, "0 of 0")):
    check(f"binomial P(X >= k): {why} = {want:.6f}", abs(B(k_, n_, p_) - want) < 1e-12, f"{B(k_, n_, p_)!r}")
big_p = B(50_000, 100_000, 0.5)   # 1/2 + half the central term, sqrt(2 / (pi * 100000)) / 2 = 0.00126
check("large n does not overflow: 50,000 of 100,000 at 1/2 = 0.50126", abs(big_p - 0.501262) < 1e-5, f"{big_p!r}")
for args_, why in (((5, 4, 0.5), "k > n"), ((-1, 4, 0.5), "negative k"), ((1, 4, 1.5), "p above 1"),
                   ((1, 4, float("nan")), "NaN p"), ((True, 4, 0.5), "a boolean k"), ((1.0, 4, 0.5), "a float k")):
    raises(f"binomial refuses {why}", lambda args_=args_: B(*args_))


def exact_tail(k_, n_, c):
    """The same tail with exact fractions and integer binomial coefficients: an oracle that shares no code."""
    c = Fraction(c)
    return float(sum(math.comb(n_, i) * c ** i * (1 - c) ** (n_ - i) for i in range(k_, n_ + 1)))


hc = facts["hand_check"]
# The 13 named punches above: type says lead for 10 (9 jabs, 1 lead hook), the wrist for 9 (one jab came from the
# right). Chance = (10*9 + 3*4) / 13^2 = 102/169 = 0.604; 12 of 13 agree.
check("hand check on the main fixture: 12 of 13 agree, chance 102/169",
      hc["n"] == 13 and hc["k"] == 12 and abs(hc["chance"] - 102 / 169) < 1e-12, str(hc))
check("p = 13 c^12 (1 - c) + c^13 = 0.0134 at c = 102/169: above chance, no warning",
      abs(hc["p"] - exact_tail(12, 13, Fraction(102, 169))) < 1e-12 and abs(hc["p"] - 0.01345) < 5e-5
      and not hc["warn"], str(hc))
check("no warning in a summary whose check passes", "Warning" not in text, text)


def hand_rows(ll, lr, rl, rr):
    """Named punches by (type's hand, wrist's hand): lead-lead, lead-rear, rear-lead, rear-rear. Orthodox: a jab is
    the lead (left) hand's type, a cross the rear (right) hand's."""
    kinds = [("jab", "left")] * ll + [("jab", "right")] * lr + [("cross", "left")] * rl + [("cross", "right")] * rr
    return [row(1 + 0.5 * i, t_, h_) for i, (t_, h_) in enumerate(kinds)]


def warned(name, rows_):
    f_ = facts_of(rows_, None)
    return f_, variant(f_, name=name)


print("the hand check: real-world facts (counts from the real clips; the clips themselves are not in the repo)")
# As stated in the brief, before classify's confidence fix: both must warn.
for name, k_, n_, c in (("IMG_4395, old 0.3 rule, as recorded", 22, 41, 0.482),
                        ("IMG_4406, old 0.3 rule, as recorded", 20, 41, 0.509)):
    check(f"{name}: {k_} of {n_} against {c}: p = {B(k_, n_, c):.3f}, warns",
          not B(k_, n_, c) < report.HAND_ALPHA and abs(B(k_, n_, c) - exact_tail(k_, n_, c)) < 1e-12)
# Joint tables with those agreements and chances, through the CSV -> facts -> summary path. The old clips' own
# tables were not kept; these reproduce the recorded n, agreement and chance. 48.2% is not reachable with 22 of 41
# agreeing (nearest 48.3%); iteration 3 recorded it with 42 punches (23 of 42, 54.8%), and both are run.
for name, table, n_, k_, c in (("old IMG_4395, 41 punches", (15, 15, 4, 7), 41, 22,
                                Fraction(30 * 19 + 11 * 22, 41 ** 2)),
                               ("old IMG_4395, 42 punches (iteration 3)", (19, 18, 1, 4), 42, 23,
                                Fraction(37 * 20 + 5 * 22, 42 ** 2)),
                               ("old IMG_4406", (18, 18, 3, 2), 41, 20, Fraction(36 * 21 + 5 * 20, 41 ** 2))):
    f_, tx = warned(name, hand_rows(*table))
    h_ = f_["hand_check"]
    check(f"{name}: {k_} of {n_}, chance {float(c):.4f}, p = {h_['p']:.3f}: the summary LEADS with the warning",
          h_["k"] == k_ and h_["n"] == n_ and abs(h_["chance"] - c) < 1e-12 and h_["warn"]
          and tx.startswith("Warning: the punch types in this summary may be wrong for this clip."), tx)
# Today's classified CSVs: their joint tables, read from results/punches/IMG_44*.classified.csv of 2026-10-13.
f95, t95 = warned("IMG_4395 today", hand_rows(15, 11, 5, 11))
h95 = f95["hand_check"]
check("IMG_4395 today: 26 of 42 (61.9%) against 49.4%: p = 0.0715, STILL WARNS",
      h95["k"] == 26 and h95["n"] == 42 and abs(h95["agree"] - 26 / 42) < 1e-12
      and abs(h95["chance"] - Fraction(26 * 20 + 16 * 22, 42 ** 2)) < 1e-12
      and abs(h95["p"] - exact_tail(26, 42, Fraction(872, 1764))) < 1e-12 and abs(h95["p"] - 0.0715) < 5e-4
      and h95["warn"] and t95.startswith("Warning:"), f"{h95} {t95[:300]}")
check("IMG_4395 today, the sentence as written",
      t95.startswith("Warning: the punch types in this summary may be wrong for this clip. For 26 of 42 named "
                     "punches the hand the punch type implies is the wrist that moved (62%), against 49% expected "
                     "by chance from how often each reading says lead hand; a one-sided exact binomial test gives "
                     "p = 0.071, not under 0.05, so that agreement is not shown to be better than chance. In one "
                     "round"), t95[:420])
f06, t06 = warned("IMG_4406 today", hand_rows(20, 7, 1, 13))
h06 = f06["hand_check"]
check("IMG_4406 today: 33 of 41 (80.5%) against 50.4%: p = 0.0001, no warning",
      h06["k"] == 33 and h06["n"] == 41
      and abs(h06["p"] - exact_tail(33, 41, Fraction(27 * 21 + 14 * 20, 41 ** 2))) < 1e-12
      and h06["p"] < 0.001 and not h06["warn"] and "Warning" not in t06, f"{h06}")

print("the hand check: the test is the only rule")
# Five jabs and five crosses: chance is 1/2 whatever the wrists say. 8 of 10 agree -> p = 56/1024 = 0.055, warns;
# 9 of 10 -> 11/1024 = 0.011, does not. Nothing but the test separates them.
f8, t8 = warned("8 of 10 at 1/2", hand_rows(4, 1, 1, 4))
f9, t9 = warned("9 of 10 at 1/2", hand_rows(5, 0, 1, 4))
check("8 of 10 at chance 1/2: p = 0.055, warns, printed so it does not read as a tie with 0.05",
      f8["hand_check"]["warn"] and "p = 0.055, not under 0.05" in t8,
      t8[:400])
check("9 of 10 at chance 1/2: p = 0.011, no warning", not f9["hand_check"]["warn"] and "Warning" not in t9)
check("the warning is the summary's first sentence and appears once",
      t8.startswith("Warning:") and t8.count("Warning") == 1)

print("the hand check: edge cases")
f0, t0n = warned("no named punch", [row(1 + i, "unclassified") for i in range(3)])
check("0 named punches: no agreement, no p, no warning (no punch type is stated)",
      f0["hand_check"] == {"n": 0, "agree": None, "chance": None, "k": 0, "p": None, "warn": False}
      and "Warning" not in t0n and "could name" not in t0n, str(f0["hand_check"]))
check("no punches at all: no warning", not report.compute_facts([], ROUNDS, DUR)["hand_check"]["warn"])
fa, ta = warned("all one side, agreeing", hand_rows(12, 0, 0, 0))
check("all lead by type AND by wrist: 100% agreement is exactly chance (100%), p = 1, warns",
      fa["hand_check"]["agree"] == 1.0 and fa["hand_check"]["chance"] == 1.0 and fa["hand_check"]["p"] == 1.0
      and "(100%), against 100% expected" in ta and "p = 1, not under" in ta, ta[:400])
fz, tz_ = warned("all one side, disagreeing", hand_rows(0, 9, 0, 0))
check("all lead by type, all rear by wrist: 0 of 9, chance 0, p = 1, warns",
      fz["hand_check"]["k"] == 0 and fz["hand_check"]["chance"] == 0.0 and fz["hand_check"]["warn"]
      and "For 0 of 9 named punches" in tz_, tz_[:300])
fm, tm = warned("one type hand, mixed wrists", hand_rows(7, 3, 0, 0))
check("types all lead, wrists 7 lead / 3 rear: agreement 70% = chance 70%, warns",
      abs(fm["hand_check"]["chance"] - 0.7) < 1e-12 and fm["hand_check"]["warn"], str(fm["hand_check"]))
f1_, t1_ = warned("one named punch", hand_rows(1, 0, 0, 0))
check("one named punch: singular, and it warns (1 of 1 at chance 1)",
      "For 1 of 1 named punch the hand" in t1_ and f1_["hand_check"]["warn"], t1_[:200])
fs, ts = warned("southpaw", [row(1 + 0.5 * i, "jab", "right", lead="right") for i in range(10)]
                + [row(10 + 0.5 * i, "cross", "left", lead="right") for i in range(10)])
check("southpaw: lead and rear follow the stance, 20 of 20 agree at chance 1/2, no warning",
      fs["hand_check"]["k"] == 20 and abs(fs["hand_check"]["chance"] - 0.5) < 1e-12 and not fs["hand_check"]["warn"])
check("p-values never round to 0.00 or 1.00 unless exact",
      [report._Writer().pval(x) for x in (1.0, 0.9999, 0.0001, 0.0, 0.2934, 0.0547, 0.0715)]
      == ["p = 1", "p over 0.99", "p under 0.001", "p = 0.000", "p = 0.29", "p = 0.055", "p = 0.071"])

print("provenance")
side = tmp / "IMG_1.classified.json"


CUT = {"per": "clip", "length_s": 0.4, "lead_share": 0.42, "n_used": 12, "n_left_out": 2, "capped": False,
       "source": "clip"}


def sidecar(**kw):
    d = {"clip": "IMG_1", "csv_sha256": hashlib.sha256(good.read_bytes()).hexdigest(), "cut": CUT,
         "classifier_sha256": man["sha256"], "stance": "orthodox", "pose_frames": int(DUR * FPS), "fps": FPS, **kw}
    side.write_text(json.dumps(d), encoding="utf-8")


sidecar()
check("the CSV, its sidecar and the manifest agree", report.verify_provenance(good, side, mpath, "IMG_1")["sha256"] == man["sha256"])
sidecar(classifier_sha256="0" * 64)
raises("punches named by another classifier are refused", lambda: report.verify_provenance(good, side, mpath, "IMG_1"))
sidecar(csv_sha256="1" * 64)
raises("a CSV edited after classify_punches wrote it is refused", lambda: report.verify_provenance(good, side, mpath, "IMG_1"))
sidecar(clip="IMG_2")
raises("a sidecar for another clip is refused", lambda: report.verify_provenance(good, side, mpath, "IMG_1"))
sidecar()
score = tmp / "IMG_1.classify_score.json"


def score_with(**kw):
    d = {"clip": "IMG_1", "classifier_sha256": man["sha256"], "matched_unclassifiable": 0,
         "matched_detections": {"n": 3, "accuracy": 1.0, "majority_floor_accuracy": 0.67}, **kw}
    score.write_text(json.dumps(d), encoding="utf-8")
    return lambda: report.own_reliability(score, "IMG_1", man["sha256"])


raises("a score for another clip is refused", score_with(clip="IMG_2"))
raises("a score of another classifier is refused", score_with(classifier_sha256="2" * 64))
raises("a score whose matched block is not an object is refused", score_with(matched_detections=[1, 2]))
raises("a score with a boolean count is refused", score_with(matched_detections={"n": True}))
check("a good score is read", score_with()()["accuracy"] == 1.0)

print("spreadsheet")
from openpyxl import load_workbook  # noqa: E402

xl = tmp / "IMG_1.report.xlsx"
HOW = report.cut_line({"cut": CUT})
check("how the clip was cut, in one sentence from the sidecar",
      HOW == "Every punch was read over 0.40 s around its wrist-speed peak: the median punch length in this clip, "
             "from 12 punches (2 left out: the wrist left view beside the peak).", HOW)
check("a capped cut says so", report.cut_line({"cut": {**CUT, "n_left_out": 0, "capped": True}}).endswith(
    "from 12 punches; capped at the longest the tool can read."))
check("a clip with no punch says so", "No punch was found" in report.cut_line({"cut": None}))
check("a clip with too few punches says its window came from the training clips",
      report.cut_line({"cut": {**CUT, "source": "training", "length_s": 0.375, "n_used": 0, "n_left_out": 0}})
      == "Too few punches in this clip to read its tempo: every punch was read over 0.38 s around its wrist-speed "
         "peak, the median punch length in the benchmark's training clips.")
for bad_cut, why in (({**CUT, "per": "punch"}, "a per-punch cut"), ({**CUT, "length_s": 60.0}, "a 60 s punch"),
                     ({**CUT, "length_s": True}, "a boolean length"), ({**CUT, "n_used": -1}, "a negative count"),
                     ({**CUT, "capped": "no"}, "a non-boolean flag"), ("clip", "a string"),
                     ({**CUT, "source": "guess"}, "an unknown source"),
                     ({k: v for k, v in CUT.items() if k != "source"}, "a cut with no source")):
    raises(f"sidecar cut: {why} is refused", lambda bad_cut=bad_cut: report.cut_line({"cut": bad_cut}))
report.write_workbook(xl, "IMG_1", rows, ROUNDS, facts, bench, text, FPS, HOW)
wb = load_workbook(xl)
check("four sheets", wb.sheetnames == ["Punches", "Rounds", "Reliability", "About"])
p = list(wb["Punches"].iter_rows(values_only=True))
check("one Punches row per punch, the one outside the rounds marked", len(p) == 1 + 15
      and sum(r[0] == "outside rounds" for r in p[1:]) == 1)
check("each punch shows its length: 24 frames at 60 fps is 0.4 s", all(r[8] == 0.4 for r in p[1:]), str(p[1]))
rr = list(wb["Rounds"].iter_rows(values_only=True))
check("Rounds: per round and total counts equal the facts", [r[3] for r in rr[1:]] == [10, 4, 14] and rr[-1][0] == "all")
rel = list(wb["Reliability"].iter_rows(values_only=True))
check("Reliability: one row per type, flagged by the rule",
      [r[0] for r in rel[1:7]] == [report.PLURAL[c] for c in CLASSES]
      and [r[0] for r in rel[1:7] if r[7] == "yes"] == ["lead uppercuts", "rear hooks"])
check("Reliability: then this clip's hand check, with the agreement, chance and p-value (12 of 13, 102/169)",
      len(rel) == 10 and all(v is None for v in rel[7]) and rel[8][3] == "agreement" and rel[8][5].startswith("p ")
      and rel[9][1:7] == (13, 12, 0.923, 0.604, 0.0135, "no"), str(rel[7:]))
for name, f_, want in (("a warned clip", f95, (42, 26, 0.619, 0.494, 0.0715, "yes")),
                       ("a clip far above chance: p = 6.83e-05 is not shown as 0", f06,
                        (41, 33, 0.805, 0.504, 6.83e-05, "no")),
                       ("a clip with no named punch", f0, (0, 0, None, None, None, "no: no named punch"))):
    p_ = tmp / "hc.xlsx"
    report.write_workbook(p_, "IMG_1", [], ROUNDS, f_, bench, "x", FPS, HOW)
    got = list(load_workbook(p_)["Reliability"].iter_rows(values_only=True))[9][1:7]
    check(f"Reliability hand-check row for {name}: {want}", got == want, str(got))
about = [r[0] for r in wb["About"].iter_rows(values_only=True)]
check("About holds the summary, how the clip was cut, and the held-out videos from the reference",
      text in about and HOW in about and any("V2/V5/V9" in (a or "") for a in about))
check("About states the warning's rule: the one-sided exact binomial test at 0.05",
      any("one-sided exact binomial test" in (a or "") and "at 0.05" in a for a in about))
check("About does not say a warned clip agrees 'no more often than chance' (IMG_4395: 62% vs 49%, warned)",
      any("not significantly more often than chance" in (a or "") for a in about)
      and not any("no more often than chance" in (a or "") for a in about))
check("no cell is a formula", all(c.data_type != "f" for ws in wb for r in ws.iter_rows() for c in r))
raises("a value that starts like a formula is refused at write time", lambda: report._cell("=1+1"))

print("report_punches CLI")
import report_punches  # noqa: E402

n = int(DUR * FPS)
pose = pose_yolo.PoseSequence(np.zeros((n, N_JOINTS, 2), np.float32), np.zeros((n, N_JOINTS), np.float32),
                              np.zeros(n, bool), FPS, 1080, 1920, "IMG_1.mov", "test")
pose.save(tmp / "IMG_1.pose.npz")
args = ["--pose", str(tmp / "IMG_1.pose.npz"), "--classified", str(good), "--classifier", str(mpath),
        "--reference", str(ref), "--rule", str(rule), "--rounds", "0-60,90-150", "--out-dir", str(tmp / "out")]
rc = report_punches.main(args)
check("end to end: workbook and summary written, summary equals the library's",
      rc == 0 and (tmp / "out" / "IMG_1.report.xlsx").is_file()
      and (tmp / "out" / "IMG_1.summary.txt").read_text(encoding="utf-8").strip() == text)
sidecar(pose_frames=n + 1)
raises("a CSV classified on a pose of another length is refused", lambda: report_punches.main(args))
side.unlink()
raises("no sidecar -> refused, with what to run", lambda: report_punches.main(args))
sidecar()
raises("a classified CSV named for another clip is refused",
       lambda: report_punches.main(["--pose", str(tmp / "IMG_1.pose.npz"), "--classified", str(tmp / "sp.csv")]
                                   + args[4:]))
score_with()
rc = report_punches.main(args + ["--score", str(score)])
check("with a matching score the summary reports it",
      rc == 0 and "named 100% of 3 matched punches" in (tmp / "out" / "IMG_1.summary.txt").read_text(encoding="utf-8"))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
