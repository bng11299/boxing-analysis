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
sys.path.insert(0, str(ROOT / "cli"))

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


def raises(name: str, fn, exc=ValueError, match: str = "") -> None:
    """`match`: a phrase the error must contain, so the test cannot pass on another refusal."""
    try:
        fn()
    except exc as e:
        check(name, match in str(e), f"refused for another reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


tmp = Path(tempfile.mkdtemp())
FPS = 60.0
WHY = "pose in only 1 of the window's 9 frames"


def row(t, ptype, hand="left", conf=0.9, lead="left", note="", fps=FPS):
    """A classify_punches.py row; `lead` is the stance's lead hand. The time is the
    frame's, as find_punches.py writes it: frame / fps to the millisecond."""
    side = "lead" if hand == lead else "rear"
    f = round(t * fps)
    ts = f"{f / fps:.3f}"
    if ptype == "unclassified":
        return [str(f), ts, hand, "5.000", "no", side, "unclassified", "", "", "", str(f - 6), str(f + 17),
                "1", WHY]
    th = classify.class_side(ptype)
    return [str(f), ts, hand, "5.000", "no", side, ptype, f"{conf:.3f}", th,
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
rows = report.read_classified(good, DUR, fps=FPS)
check("every row read, sorted by frame", len(rows) == 15 and [r["frame"] for r in rows] == sorted(r["frame"] for r in rows))
check("an unclassified row has no confidence", [r["confidence"] for r in rows if r["punch_type"] == "unclassified"] == [None])


def bad(name, rows_, header=report.CLASSIFIED_HEADER, match=""):
    p = write_csv(tmp / "bad.csv", rows_, header)
    raises(name, lambda: report.read_classified(p, DUR, fps=FPS), match=match)


bad("another header is refused", [row(1, "jab")], header=report.CLASSIFIED_HEADER[:-1])
bad("a time past the clip is refused", [row(170, "jab")])
# Frame 60 at 60 fps is 1.000 s; a CSV written from a 30 fps clip says 1.5 s for frame 45, which here is 0.75 s.
r_ = row(1, "jab"); r_[1] = "1.010"; bad("a time that is not its frame / fps is refused", [r_], match="is not frame")
bad("a CSV made at 30 fps is refused against a 60 fps clip, every frame inside it",
    [row(1.5, "jab", fps=30.0), row(2.25, "jab", fps=30.0)], match="is not frame 45 at 60.000 fps")
NF = int(DUR * FPS)   # 9600 frames, 0..9599
r_ = row(1, "jab"); r_[0], r_[1], r_[10], r_[11] = str(NF), f"{DUR:.3f}", str(NF - 6), str(NF + 17)
bad("frame 9600 of a 9600-frame clip is refused, though its time (160.000 s) is not past the end", [r_],
    match="past this 9600-frame clip")
check("a time rounded to the millisecond is read: frame 61 at 60 fps is 1.0167 s, written 1.017",
      report.read_classified(write_csv(tmp / "ms.csv", [row(61 / FPS, "jab")]), DUR, fps=FPS)[0]["time_s"] == 1.017)
raises("the clip's fps is a required argument", lambda: report.read_classified(good, DUR), exc=TypeError)
for f_, why in ((0.0, "zero"), (True, "a boolean"), (float("nan"), "NaN"), (10 ** 400, "a 400-digit integer"),
                (-(10 ** 400), "a negative 400-digit integer (Astra, 10-21)"), (-30, "a negative integer")):
    raises(f"an fps that is {why} is refused", lambda f_=f_: report.read_classified(good, DUR, fps=f_), match="fps")
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
                              report.read_classified(write_csv(tmp / "neg.csv", [r_]), DUR, fps=FPS)[0]["span"] == (-3, 20))
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
    write_csv(tmp / "sp.csv", [row(1, "jab", "right", lead="right"), row(2, "cross", "left", lead="right")]), DUR,
    fps=FPS)) == 2)
big = write_csv(tmp / "big.csv", [row(1, "jab")] * 3)
report.MAX_ROWS, keep = 2, report.MAX_ROWS
raises("more rows than the cap are refused", lambda: report.read_classified(big, DUR, fps=FPS))
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
classify.write_manifest(mpath, ck, GEOM, 50, {"held_out": {"per_class_f1": F1, "confusion": C.tolist()}})
man = classify.read_manifest(mpath)
bench = report.benchmark_reliability(ref, man)
t = {x.cls: x for x in bench["types"]}
check("cross: 100 true, 132 counted (+32%), recall 0.90",
      t["cross"].true == 100 and t["cross"].predicted == 132 and abs(t["cross"].recall - 0.9) < 1e-12)
check("rear hook: 50 true, 25 counted, recall 0.40", t["rear_hook"].true == 50 and t["rear_hook"].predicted == 25
      and abs(t["rear_hook"].recall - 0.4) < 1e-12)
# Off-side cells: cross read as jab (5), jab read as cross (2); 550 punches in all.
check("type-implied hand right on 543 of 550", abs(bench["side_accuracy"] - 543 / 550) < 1e-12)


def manifest_f1(f1, confusion=C.tolist()):
    p = tmp / "m2.json"
    held = {"per_class_f1": f1} if confusion is None else {"per_class_f1": f1, "confusion": confusion}
    classify.write_manifest(p, ck, GEOM, 50, {"held_out": held})
    return classify.read_manifest(p)


raises("a manifest whose per-type F1 differs does not borrow this reliability",
       lambda: report.benchmark_reliability(ref, manifest_f1({**F1, "jab": 0.6})), match="jab F1")
raises("a manifest with no held-out scores is refused", lambda: report.benchmark_reliability(ref, manifest_f1(None)),
       match="F1")
raises("a non-numeric F1 is refused cleanly", lambda: report.benchmark_reliability(ref, manifest_f1({**F1, "jab": "x"})),
       match="jab F1")
# Per-class F1 is 2 TP / (row sum + column sum): the same for a matrix and its transpose. Recall and the count
# warnings are not: transposed, cross is 132 true / 100 counted (-24%) instead of 100 true / 132 counted (+32%).
CT = C.T.tolist()
F1T = {c: 2 * CT[i][i] / (sum(CT[i]) + sum(r_[i] for r_ in CT)) for i, c in enumerate(CLASSES)}
check("fixture: the transposed matrix has exactly the same per-class F1", F1T == F1, str(F1T))
raises("a manifest written before the matrix was recorded is refused, with what to run",
       lambda: report.benchmark_reliability(ref, manifest_f1(F1, confusion=None)), match="re-run train_classifier.py")
refT = tmp / "refT.json"
refT.write_text(json.dumps({"results": [{"model": "st-gcn++ pretrained", "confusion": CT, "per_class_f1": F1}]}),
                encoding="utf-8")
raises("a transposed reference matrix is refused, though every F1 agrees",
       lambda: report.benchmark_reliability(refT, man), match="rows and columns swapped")
raises("a manifest that pins the transposed matrix is refused against the reference",
       lambda: report.benchmark_reliability(ref, manifest_f1(F1, confusion=CT)), match="rows and columns swapped")
moved = C.copy(); moved[0, 1] -= 1; moved[0, 2] += 1   # one cross read as lead hook instead of jab
raises("a matrix one cell away from the pinned one is refused",
       lambda: report.benchmark_reliability(ref, manifest_f1(F1, confusion=moved.tolist())), match="not the one")
raises("a pinned matrix of the wrong shape is refused",
       lambda: report.benchmark_reliability(ref, manifest_f1(F1, confusion=C[:5].tolist())), match="records no")
raises("a fractional pinned cell is refused",
       lambda: report.benchmark_reliability(ref, manifest_f1(F1, confusion=(C + 0.5).tolist())), match="cell")
huge = C.tolist(); huge[0][0] = 10 ** 400
raises("a 400-digit pinned cell is refused cleanly, not an int64 overflow",
       lambda: report.benchmark_reliability(ref, manifest_f1(F1, confusion=huge)), match="cell")

# A real fact, not a fixture: the reported run's held-out matrix (results/sweep-2026-09-22/seed0.json, V2/V5/V9,
# 976 punches; the matrix the project plan's "cross +28%, lead uppercut -41%" comes from). Transposed, the same
# F1 would have the report say lead uppercuts run HIGH by 69% (122 counted for 72 true).
REAL = np.array([[196, 2, 0, 0, 1, 1], [2, 281, 12, 0, 0, 0], [2, 7, 129, 3, 1, 0],
                 [4, 15, 32, 69, 2, 0], [18, 0, 1, 0, 12, 19], [34, 3, 0, 0, 22, 108]])
F1R = {c: 2 * int(REAL[i, i]) / int(REAL[i].sum() + REAL[:, i].sum()) for i, c in enumerate(CLASSES)}
check("real matrix: jab F1 0.932 and rear hook 0.273, the reported run's",
      abs(F1R["jab"] - 0.9320066334991708) < 1e-12 and abs(F1R["rear_hook"] - 0.2727272727272727) < 1e-12)
ref_real = tmp / "ref_real.json"
ref_real.write_text(json.dumps({"results": [{"model": "st-gcn++ pretrained", "confusion": REAL.tolist(),
                                             "per_class_f1": F1R}]}), encoding="utf-8")
tr = {x.cls: x for x in report.benchmark_reliability(ref_real, manifest_f1(F1R, confusion=REAL.tolist()))["types"]}
check("real matrix: cross 200 true, 256 counted (+28%); lead uppercut 122 true, 72 counted (-41%)",
      (tr["cross"].true, tr["cross"].predicted, tr["lead_uppercut"].true, tr["lead_uppercut"].predicted)
      == (200, 256, 122, 72) and round(100 * (tr["cross"].count_ratio - 1)) == 28
      and round(100 * (1 - tr["lead_uppercut"].count_ratio)) == 41)
ref_real_t = tmp / "ref_real_t.json"
ref_real_t.write_text(json.dumps({"results": [{"model": "st-gcn++ pretrained", "confusion": REAL.T.tolist(),
                                               "per_class_f1": F1R}]}), encoding="utf-8")
raises("real matrix: its transpose is refused against the model's manifest",
       lambda: report.benchmark_reliability(ref_real_t, manifest_f1(F1R, confusion=REAL.tolist())),
       match="rows and columns swapped")


def ref_with(**kw):
    p = tmp / "ref2.json"
    r = {"model": "st-gcn++ pretrained", "confusion": C.tolist(), "per_class_f1": F1, **kw}
    p.write_text(json.dumps({"results": [r]}), encoding="utf-8")
    return p


raises("a confusion matrix of the wrong shape is refused",
       lambda: report.benchmark_reliability(ref_with(confusion=C[:5].tolist()), man))
raises("a fractional confusion cell is refused, not truncated",
       lambda: report.benchmark_reliability(ref_with(confusion=(C + 0.5).tolist()), man))
raises("a 400-digit cell in the reference's matrix is refused cleanly, not an int64 overflow",
       lambda: report.benchmark_reliability(ref_with(confusion=huge), man), match="cell")
raises("an F1 that the matrix does not reproduce is refused (matrix and F1 from different runs)",
       lambda: report.benchmark_reliability(ref_with(per_class_f1={c: 0.5 for c in CLASSES}),
                                            manifest_f1({c: 0.5 for c in CLASSES})))
(tmp / "nan.json").write_text('{"results": NaN}', encoding="utf-8")
raises("NaN in a JSON input is refused", lambda: report.benchmark_reliability(tmp / "nan.json", man))
(tmp / "huge.json").write_text(json.dumps({"x": "y" * (report.MAX_JSON_BYTES + 10)}), encoding="utf-8")
raises("an oversized JSON input is refused", lambda: report.benchmark_reliability(tmp / "huge.json", man))
check("held-out videos come from the reference", bench["videos"] == ["V2", "V5", "V9"])
rule = tmp / "rule.json"
# The rule as find_punches.load_rule returns it and classify_punches.py records it in the sidecar.
RULE = {"variant": "relative", "threshold": 3.6238817996457016, "min_gap_s": 0.4}
rule.write_text(json.dumps({"video": "V6", "chosen": {**RULE, "threshold_quantile": 0.93, "test": {
    "recall": 0.741, "precision": 0.465, "count_ratio": 1.596, "n_labelled": 406}}}), encoding="utf-8")
finder = report.finder_reliability(rule, RULE)
check("finder reliability: held-out recall and count ratio", finder == {"recall": 0.741, "count_ratio": 1.596, "video": "V6"})
check("a rule recorded with a whole-number gap (1 vs 1.0) is the same rule",
      report._rule_of({**RULE, "min_gap_s": 1}, "x") == report._rule_of({**RULE, "min_gap_s": 1.0}, "x"))
(tmp / "rule2.json").write_text(json.dumps({"chosen": RULE}), encoding="utf-8")
raises("a finder rule with no held-out score is refused", lambda: report.finder_reliability(tmp / "rule2.json", RULE),
       match="no held-out test score")
for k_, v_, why in (("variant", "radial", "another wrist-speed variant"),
                    ("threshold", 3.6238817996457016 * 1.01, "a threshold 1% higher"),
                    ("min_gap_s", 0.25, "another minimum gap")):
    raises(f"punches found by a rule with {why} do not borrow this rule's score",
           lambda k_=k_, v_=v_: report.finder_reliability(rule, {**RULE, k_: v_}), match="another rule")
for used, why in ((None, "no rule in the sidecar"), ({"variant": "relative"}, "a rule with no threshold"),
                  ({**RULE, "threshold": True}, "a boolean threshold"),
                  ({**RULE, "min_gap_s": float("inf")}, "an infinite gap"),
                  ({**RULE, "threshold": 10 ** 400}, "a 400-digit integer threshold")):
    raises(f"the sidecar's rule is checked: {why} is refused",
           lambda used=used: report.finder_reliability(rule, used), match="sidecar")

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
    return report.compute_facts(report.read_classified(p, dur, fps=FPS), report.parse_rounds(rounds_txt, dur), dur)


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
check("own labels without a finder block: the finding sentence still says the error is not known",
      "how far the totals above are off is not known" in to, to)
# IMG_5835's real classify_score.json (results/repro5835/own_tol0.2): 149 labelled, 162 reported,
# 117 matched, 0.744 of 117 named right, commonest-punch floor 0.248. The poster quotes these.
own5835 = {"n": 117, "unclassifiable": 0, "accuracy": 0.7435897435897436, "floor": 0.24786324786324787,
           "finder": {"labelled": 149, "detected": 162, "matched": 117}}
tf = variant(facts, own5835, name="own labels with finding")
check("with a finder block: found / labelled and the unmatched detections, from the score",
      "Against your own labels for this clip it found 117 of your 149 punches (79%), and 45 of the 162 punches "
      "it reported matched none of them" in tf, tf)
check("with a finder block: the naming sentence follows without repeating 'Against your own labels'",
      "It named 74% of 117 matched punches correctly; always guessing your commonest punch would score 25%." in tf
      and tf.count("Against your own labels") == 1, tf)
check("with a finder block: no 'not known' and no 'not been measured'",
      "is not known" not in tf and "not been measured" not in tf, tf)
tp_ = variant(facts, {**own5835, "finder": {"labelled": 117, "detected": 117, "matched": 117}}, name="all matched")
check("every reported punch matched: says so, no '0 of the 117'", "every punch it reported matched" in tp_
      and "0 of the" not in tp_, tp_)
tnd = variant(facts, {"n": 0, "unclassifiable": 0, "finder": {"labelled": 5, "detected": 0, "matched": 0}},
              name="reported nothing")
check("nothing reported: 'found 0 of your 5 punches (0%), and it reported none at all', not 'every punch matched'",
      "found 0 of your 5 punches (0%), and it reported none at all." in tnd and "every punch" not in tnd, tnd)
tz = variant(facts, {"n": 0, "unclassifiable": 0}, name="labelled, nothing matched")
check("labelled but nothing matched: not 'label a clip'", "You labelled this clip, but no labelled punch" in tz
      and "label a clip" not in tz, tz)

print("the hand check: one-sided exact tests, worked by hand")
B, F = stats.binomial_greater_p, stats.fisher_greater_p
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
    """The binomial tail with exact fractions and integer binomial coefficients: an oracle that shares no code."""
    c = Fraction(c)
    return float(sum(math.comb(n_, i) * c ** i * (1 - c) ** (n_ - i) for i in range(k_, n_ + 1)))


def exact_fisher(a, b, c, d):
    """Fisher's one-sided tail as a hypergeometric sum of integer binomial coefficients: shares no code with scipy."""
    r1, c1, n_ = a + b, a + c, a + b + c + d
    return float(sum(Fraction(math.comb(c1, x) * math.comb(n_ - c1, r1 - x), math.comb(n_, r1))
                     for x in range(a, min(r1, c1) + 1)))


for table, want, why in (((3, 0, 0, 2), 1 / 10, "[[3,0],[0,2]]: 1 / C(5,3)"),
                         ((4, 1, 1, 4), 26 / 252, "[[4,1],[1,4]]: (C(5,4) C(5,1) + 1) / C(10,5)"),
                         ((9, 1, 0, 3), 4 / 286, "[[9,1],[0,3]]: C(4,1) / C(13,10)"),
                         ((3, 0, 1, 4), 4 / 56, "[[3,0],[1,4]]: C(4,3) / C(8,3)"),
                         ((2, 0, 1, 9), 3 / 66, "[[2,0],[1,9]]: C(3,2) / C(12,2)"),
                         ((15, 11, 5, 11), exact_fisher(15, 11, 5, 11), "IMG_4395 today, against the oracle"),
                         ((12, 0, 0, 0), 1.0, "an empty row and column: no evidence"),
                         ((0, 9, 0, 0), 1.0, "everything off the diagonal in one row: no evidence")):
    check(f"Fisher P: {why} = {want:.6f}", abs(F(*table) - want) < 1e-12, f"{F(*table)!r}")
for args_, why in (((-1, 0, 0, 0), "a negative cell"), ((1.0, 0, 0, 0), "a float cell"),
                   ((True, 0, 0, 0), "a boolean cell")):
    raises(f"Fisher refuses {why}", lambda args_=args_: F(*args_))

hc = facts["hand_check"]
# The 13 named punches above, (type's hand, wrist): 9 lead-lead, 1 lead-rear (the jab from the right), 0 rear-lead,
# 3 rear-rear. Chance = (10*9 + 3*4) / 13^2 = 102/169 = 0.604; 12 of 13 agree (92%).
check("hand check on the main fixture: table (9, 1, 0, 3), 12 of 13 agree, chance 102/169",
      hc["table"] == (9, 1, 0, 3) and hc["n"] == 13 and hc["k"] == 12 and abs(hc["chance"] - 102 / 169) < 1e-12,
      str(hc))
check("binomial 13 c^12 (1 - c) + c^13 = 0.0135, Fisher 4/286 = 0.0140, 92% >= 70%: passes",
      abs(hc["p_binomial"] - exact_tail(12, 13, Fraction(102, 169))) < 1e-12 and abs(hc["p_fisher"] - 4 / 286) < 1e-12
      and hc["verdict"] == "pass" and not hc["warn"] and not hc["below_floor"], str(hc))
check("a passing clip: no warning or caution, and a sentence after the hand sentences with both p and the hand-only "
      "limit", "Warning" not in text and "Caution" not in text
      and "one of the two readings is wrong there. Across the clip the hand each punch type implies is the wrist "
          "that moved for 12 of 13 (92%) against 60% by chance (p = 0.013 binomial, p = 0.014 Fisher), and at least "
          "the 70% this report requires. That checks the hand only: it cannot tell a jab from a lead hook or "
          "uppercut, or a cross from a rear hook or uppercut." in text, text)


def hand_rows(ll, lr, rl, rr):
    """Named punches by (type's hand, wrist's hand): lead-lead, lead-rear, rear-lead, rear-rear. Orthodox: a jab is
    the lead (left) hand's type, a cross the rear (right) hand's."""
    kinds = [("jab", "left")] * ll + [("jab", "right")] * lr + [("cross", "left")] * rl + [("cross", "right")] * rr
    return [row(1 + 0.5 * i, t_, h_) for i, (t_, h_) in enumerate(kinds)]


def warned(name, rows_):
    f_ = facts_of(rows_, None)
    return f_, variant(f_, name=name)


WARN = "Warning: the punch types in this summary may be wrong for this clip."
print("the hand check: real-world facts (counts from the real clips; the clips themselves are not in the repo)")
# As stated in the brief, before classify's confidence fix: both must warn (only n, agreement and chance were kept).
for name, k_, n_, c in (("IMG_4395, old 0.3 rule, as recorded", 22, 41, 0.482),
                        ("IMG_4406, old 0.3 rule, as recorded", 20, 41, 0.509)):
    check(f"{name}: {k_} of {n_} against {c}: binomial p = {B(k_, n_, c):.3f}, and under 70% agreement: warns",
          not B(k_, n_, c) < report.HAND_ALPHA and k_ / n_ < report.HAND_FLOOR
          and abs(B(k_, n_, c) - exact_tail(k_, n_, c)) < 1e-12)
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
    check(f"{name}: {k_} of {n_}, chance {float(c):.4f}, p = {h_['p_binomial']:.3f} / {h_['p_fisher']:.3f}: the "
          f"summary LEADS with the warning", h_["k"] == k_ and h_["n"] == n_ and abs(h_["chance"] - c) < 1e-12
          and h_["warn"] and h_["below_floor"] and abs(h_["p_fisher"] - exact_fisher(*table)) < 1e-12
          and tx.startswith(WARN), tx)
# Today's classified CSVs: their joint tables, read from results/punches/IMG_44*.classified.csv of 2026-10-13.
f95, t95 = warned("IMG_4395 today", hand_rows(15, 11, 5, 11))
h95 = f95["hand_check"]
check("IMG_4395 today: 26 of 42 (61.9%) against 49.4%: binomial 0.0715, Fisher 0.0883, under 70%: STILL WARNS",
      h95["k"] == 26 and h95["n"] == 42 and abs(h95["agree"] - 26 / 42) < 1e-12
      and abs(h95["chance"] - Fraction(26 * 20 + 16 * 22, 42 ** 2)) < 1e-12
      and abs(h95["p_binomial"] - exact_tail(26, 42, Fraction(872, 1764))) < 1e-12
      and abs(h95["p_binomial"] - 0.0715) < 5e-4 and abs(h95["p_fisher"] - 0.0883) < 5e-4
      and h95["below_floor"] and h95["warn"], f"{h95} {t95[:300]}")
check("IMG_4395 today, the sentences as written",
      t95.startswith(WARN + " For 26 of 42 named punches the hand the punch type implies is the wrist that moved "
                     "(62%), against 49% expected by chance from how often each reading says lead hand. One-sided "
                     "exact tests of whether that is above chance give p = 0.071 (binomial) and p = 0.088 (Fisher), "
                     "neither under 0.05, so that agreement is not shown to be better than chance. The 62% agreement "
                     "is under the 70% this report requires.\n\nIn short: ")
      and "\n\nWhat the tool saw: In one round" in t95 and "\n\nHow much to trust it: " in t95, t95[:900])
check("IMG_4395 today: the hand check warned, so the headline names no punch type",
      "most often" not in t95.split("\n\n")[1], t95.split("\n\n")[1])
f06, t06 = warned("IMG_4406 today", hand_rows(20, 7, 1, 13))
h06 = f06["hand_check"]
check("IMG_4406 today: 33 of 41 (80.5%) against 50.4%: binomial and Fisher under 0.001, at least 70%: passes",
      h06["k"] == 33 and h06["n"] == 41 and h06["verdict"] == "pass"
      and abs(h06["p_binomial"] - exact_tail(33, 41, Fraction(27 * 21 + 14 * 20, 41 ** 2))) < 1e-12
      and abs(h06["p_fisher"] - exact_fisher(20, 7, 1, 13)) < 1e-12 and h06["p_fisher"] < 0.001
      and "Warning" not in t06 and "for 33 of 41 (80%) against 50% by chance (p under 0.001 binomial, p under 0.001 "
                                  "Fisher), and at least the 70%" in t06, f"{h06}")
keep_floor = report.HAND_FLOOR
try:
    report.HAND_FLOOR = 0.80
    f80, _ = warned("IMG_4406 at the 80% target", hand_rows(20, 7, 1, 13))
    f80b, t80b = warned("IMG_4406 with one punch fewer agreeing, at 80%", hand_rows(19, 8, 1, 13))
finally:
    report.HAND_FLOOR = keep_floor
check("at the 80% target IMG_4406 still passes (33 of 41 = 80.5%), by one punch: 32 of 41 (78%) warns",
      f80["hand_check"]["verdict"] == "pass" and f80b["hand_check"]["below_floor"] and f80b["hand_check"]["warn"]
      and "The 78% agreement is under the 80% this report requires." in t80b, t80b[:600])

print("the hand check: what decides")
# Five jabs and five crosses: chance is 1/2 whatever the wrists say. 8 of 10 agree (80%, over the floor) ->
# binomial 56/1024 = 0.055, Fisher 26/252 = 0.103: warns. 9 of 10 -> 11/1024 and 6/252: passes.
f8, t8 = warned("8 of 10 at 1/2", hand_rows(4, 1, 1, 4))
f9, t9 = warned("9 of 10 at 1/2", hand_rows(5, 0, 1, 4))
check("8 of 10 at chance 1/2: both tests fail, warns, p printed so 0.055 does not read as a tie with 0.05",
      f8["hand_check"]["warn"] and not f8["hand_check"]["below_floor"]
      and "give p = 0.055 (binomial) and p = 0.10 (Fisher), neither under 0.05" in t8
      and "requires" not in t8.split("In one round")[0], t8[:500])
check("9 of 10 at chance 1/2: both tests pass, no warning", f9["hand_check"]["verdict"] == "pass" and "Warning" not in t9)
check("the warning is the summary's first sentence and appears once", t8.startswith(WARN) and t8.count("Warning") == 1)
fb, tb_ = warned("binomial passes, Fisher fails", hand_rows(3, 0, 1, 4))
check("7 of 8, chance 1/2: binomial 9/256 = 0.035 passes, Fisher 4/56 = 0.071 does not: warns, and says which passed",
      fb["hand_check"]["warn"] and abs(fb["hand_check"]["p_binomial"] - 9 / 256) < 1e-12
      and "give p = 0.035 (binomial) and p = 0.071 (Fisher); only the binomial test is under 0.05" in tb_, tb_[:500])
ff, tf = warned("Fisher passes, binomial fails", hand_rows(2, 0, 1, 9))
check("11 of 12, chance 2/3: binomial 28672/531441 = 0.054 fails, Fisher 3/66 = 0.045 passes: warns",
      ff["hand_check"]["warn"] and abs(ff["hand_check"]["p_binomial"] - 28672 / 531441) < 1e-12
      and "give p = 0.054 (binomial) and p = 0.045 (Fisher); only the Fisher test is under 0.05" in tf, tf[:500])
fl, tl_ = warned("68 of 100 at 1/2", hand_rows(34, 16, 16, 34))
check("68 of 100 at 1/2: both tests far under 0.05, but under 70%: warns on the floor alone",
      fl["hand_check"]["p_binomial"] < 0.001 and fl["hand_check"]["p_fisher"] < 0.001 and fl["hand_check"]["warn"]
      and "(Fisher). The 68% agreement is under the 70% this report requires." in tl_
      and "wrong stance" not in tl_, tl_[:600])
fe, te = warned("70 of 100 at 1/2", hand_rows(35, 15, 15, 35))
check("70 of 100 is not under 70%: passes", not fe["hand_check"]["below_floor"] and fe["hand_check"]["verdict"] == "pass")
keep_floor = report.HAND_FLOOR
try:
    report.HAND_FLOOR = 0.68   # 0.68 * 75 = 51.00000000000001 in floats
    f68, _ = warned("51 of 75 at a 0.68 floor", hand_rows(26, 12, 12, 25))
finally:
    report.HAND_FLOOR = keep_floor
check("the floor is compared exactly: 51 of 75 is not under 0.68, though 0.68 * 75 in floats is just over 51",
      not f68["hand_check"]["below_floor"], str(f68["hand_check"]))
fq, tq = warned("Fisher exactly at alpha", hand_rows(12, 2, 0, 2))
check("Fisher's exact p of 1/20 (scipy: 0.04999999999999999) counts as NOT under 0.05; never '0.050 ... under'",
      abs(fq["hand_check"]["p_fisher"] - 0.05) < 1e-12 and fq["hand_check"]["warn"]
      and "p = 0.050 (Fisher), neither under 0.05" in tq and "only the Fisher test" not in tq, tq[:500])
check("a p under alpha that would print as 0.050 says 'just under'",
      report._Writer().pval(0.0497) == "p just under 0.05" and report._Writer().pval(0.05) == "p = 0.050"
      and report._Writer().pval(0.04999999999999999) == "p = 0.050")

print("the hand check: when it cannot confirm the types")
fa, ta = warned("all one side, agreeing", hand_rows(12, 0, 0, 0))
check("12 jabs, all from the lead wrist: 100% agreement is chance itself; a caution, not a warning, and first",
      fa["hand_check"]["verdict"] == "concentrated" and not fa["hand_check"]["warn"] and fa["hand_check"]["p_binomial"] == 1.0
      and ta.startswith("Caution: there appears to be an extreme concentration of lead-hand punch types: 100% of the "
                        "12 named punches, most often jabs (12). On a split like this the hand check cannot "
                        "confirm the types even if every punch agreed with the wrist that moved. If this is not "
                        "expected, checking the footage is recommended.") and "Warning" not in ta, ta[:400])
f91, t91 = warned("9 lead, 1 rear, all agreeing", hand_rows(9, 0, 0, 1))
check("90/10 over 10 punches, all agreeing: full agreement cannot pass (0.82^10 = 0.137): concentrated",
      f91["hand_check"]["verdict"] == "concentrated" and "90% of the 10 named punches" in t91, t91[:300])
f92, _ = warned("18 lead, 2 rear, all agreeing", hand_rows(18, 0, 0, 2))
check("90/10 over 20 punches can be checked (0.82^20 = 0.019, Fisher 1/190): passes",
      f92["hand_check"]["verdict"] == "pass")
fm, tm = warned("types all lead, wrists mixed", hand_rows(7, 3, 0, 0))
check("types all lead, wrists 7 lead / 3 rear: agreement 70% = chance 70%: concentrated",
      abs(fm["hand_check"]["chance"] - 0.7) < 1e-12 and fm["hand_check"]["verdict"] == "concentrated", str(fm["hand_check"]))
fr, tr2 = warned("all rear", [row(1 + 0.5 * i, "cross", "right") for i in range(6)]
                 + [row(5 + 0.5 * i, "rear_uppercut", "right") for i in range(4)])
check("a rear-hand concentration names the rear hand and its commonest type",
      fr["hand_check"]["verdict"] == "concentrated"
      and "rear-hand punch types: 100% of the 10 named punches, most often crosses (6)" in tr2, tr2[:300])
ft, tt = warned("a tie", [row(1 + 0.5 * i, "jab") for i in range(6)] + [row(5 + 0.5 * i, "lead_hook") for i in range(6)])
check("tied types are all named, not the first in the alphabet: 'most often jabs and lead hooks (6 each)'",
      ft["hand_check"]["verdict"] == "concentrated" and "most often jabs and lead hooks (6 each)" in tt, tt[:300])
fz, tz_ = warned("all one side, disagreeing", hand_rows(0, 9, 0, 0))
check("all jabs, all from the rear wrist: 0 of 9 is under 70%, so a WARNING (a wrong stance looks like this)",
      fz["hand_check"]["warn"] and fz["hand_check"]["below_floor"]
      and tz_.startswith(WARN + " For 0 of 9 named punches") and "p = 1 (binomial) and p = 1 (Fisher), neither" in tz_
      and "can also mean the clip was read with the wrong stance (orthodox or southpaw)" in tz_, tz_[:700])
check("agreement above chance never mentions stance", "stance" not in t95 and "stance" not in t8)
f1_, t1_ = warned("one named punch", hand_rows(1, 0, 0, 0))
check("one named punch: too few to check, singular", f1_["hand_check"]["verdict"] == "too_few"
      and t1_.startswith("Caution: only 1 named punch, too few to check the punch types against the wrist that "
                         "moved."), t1_[:200])
f4, t4 = warned("4 balanced, all agreeing", hand_rows(2, 0, 0, 2))
check("4 punches, 2/2, all agreeing: even a 50/50 split this small cannot pass (0.5^4, 1/6): too few",
      f4["hand_check"]["verdict"] == "too_few" and t4.startswith("Caution: only 4 named punches"), t4[:200])
f5, t5 = warned("5 punches, 3/2, all agreeing", hand_rows(3, 0, 0, 2))
check("5 punches, 3/2, all agreeing: full agreement passes the binomial ((13/25)^5 = 0.038) but not Fisher (1/10), "
      "so the check cannot pass: too few, not a warning",
      abs(f5["hand_check"]["p_binomial"] - (13 / 25) ** 5) < 1e-12 and abs(f5["hand_check"]["p_fisher"] - 0.1) < 1e-12
      and f5["hand_check"]["verdict"] == "too_few" and t5.startswith("Caution: only 5 named punches"), t5[:200])
f0, t0n = warned("no named punch", [row(1 + i, "unclassified") for i in range(3)])
check("0 named punches: no agreement, no p, no warning or caution (no punch type is stated)",
      f0["hand_check"]["verdict"] == "none" and f0["hand_check"]["p_binomial"] is None
      and f0["hand_check"]["p_fisher"] is None and not f0["hand_check"]["warn"]
      and "Warning" not in t0n and "Caution" not in t0n and "could name" not in t0n, str(f0["hand_check"]))
check("no punches at all: nothing said", report.compute_facts([], ROUNDS, DUR)["hand_check"]["verdict"] == "none")
fs, ts = warned("southpaw", [row(1 + 0.5 * i, "jab", "right", lead="right") for i in range(10)]
                + [row(10 + 0.5 * i, "cross", "left", lead="right") for i in range(10)])
check("southpaw: lead and rear follow the stance, 20 of 20 agree at chance 1/2: passes",
      fs["hand_check"]["k"] == 20 and abs(fs["hand_check"]["chance"] - 0.5) < 1e-12
      and fs["hand_check"]["verdict"] == "pass")
check("p-values never round to 0.00 or 1.00 unless exact",
      [report._Writer().pval(x) for x in (1.0, 0.9999, 0.0001, 0.0, 0.2934, 0.0547, 0.0715)]
      == ["p = 1", "p over 0.99", "p under 0.001", "p = 0.000", "p = 0.29", "p = 0.055", "p = 0.071"])

print("provenance")
side = tmp / "IMG_1.classified.json"


CUT = {"per": "clip", "length_s": 0.4, "lead_share": 0.42, "n_used": 12, "n_left_out": 2, "capped": False,
       "source": "clip"}


def sidecar(**kw):
    d = {"clip": "IMG_1", "csv_sha256": hashlib.sha256(good.read_bytes()).hexdigest(), "cut": CUT,
         "classifier_sha256": man["sha256"], "stance": "orthodox", "rule": RULE, "pose_frames": int(DUR * FPS),
         "fps": FPS, **kw}
    side.write_text(json.dumps(d), encoding="utf-8")
    return d


sidecar()
check("the CSV, its sidecar and the manifest agree", report.verify_provenance(good, side, mpath, "IMG_1")["sha256"] == man["sha256"])
sidecar(classifier_sha256="0" * 64)
raises("punches named by another classifier are refused", lambda: report.verify_provenance(good, side, mpath, "IMG_1"))
sidecar(csv_sha256="1" * 64)
raises("a CSV edited after classify_punches wrote it is refused", lambda: report.verify_provenance(good, side, mpath, "IMG_1"))
sidecar(clip="IMG_2")
raises("a sidecar for another clip is refused", lambda: report.verify_provenance(good, side, mpath, "IMG_1"))
sidecar()
GOOD_SHA = hashlib.sha256(good.read_bytes()).hexdigest()
print("the pose the report reads is the one the CSV was classified on")
SIDE = {"pose_frames": NF, "fps": FPS}
report.check_pose(SIDE, NF, FPS)
check("same frames, same fps: accepted", True)
for kw_, why in (({"fps": 30.0}, "30 fps against a 60 fps pose"), ({"fps": 59.94}, "59.94 against 60"),
                 ({"fps": None}, "no fps"), ({"fps": True}, "a boolean fps"), ({"fps": "60"}, "a string fps"),
                 ({"pose_frames": NF + 1}, "one frame more"), ({"pose_frames": float(NF)}, "a float frame count"),
                 ({"pose_frames": True}, "a boolean frame count"), ({"fps": 10 ** 400}, "a 400-digit integer fps"),
                 ({"fps": float("nan")}, "a NaN fps")):
    raises(f"a sidecar for {why} is refused", lambda kw_=kw_: report.check_pose({**SIDE, **kw_}, NF, FPS),
           match="classified on")
score = tmp / "IMG_1.classify_score.json"


def score_with(**kw):
    d = {"clip": "IMG_1", "classifier_sha256": man["sha256"], "csv_sha256": GOOD_SHA, "matched_unclassifiable": 0,
         "matched_detections": {"n": 3, "accuracy": 1.0, "majority_floor_accuracy": 0.67}, **kw}
    score.write_text(json.dumps(d), encoding="utf-8")
    return lambda: report.own_reliability(score, "IMG_1", man["sha256"], GOOD_SHA)


raises("a score for another clip is refused", score_with(clip="IMG_2"))
raises("a score of another classifier is refused", score_with(classifier_sha256="2" * 64))
raises("a score whose matched block is not an object is refused", score_with(matched_detections=[1, 2]))
raises("a score with a boolean count is refused", score_with(matched_detections={"n": True}))
raises("a score of another classified CSV (same clip, same model) is refused", score_with(csv_sha256="3" * 64),
       match="another classified CSV")
s_ = score_with()
old = {k: v for k, v in json.loads(score.read_text(encoding="utf-8")).items() if k != "csv_sha256"}
score.write_text(json.dumps(old), encoding="utf-8")
raises("a score written before it recorded the CSV it scored is refused, and says so", s_,
       match="does not say which classified CSV")
score_with()
raises("no CSV hash to check against is refused, not skipped",
       lambda: report.own_reliability(score, "IMG_1", man["sha256"], None), match="no classified-CSV sha256")
check("a good score is read", score_with()()["accuracy"] == 1.0)
check("a score without a finder block is read, with no finding numbers", "finder" not in score_with()())
FIND = {"labelled_punches": 10, "detections": 12, "matched": 8}
check("a score with a finder block is read", score_with(finder=FIND)()["finder"]
      == {"labelled": 10, "detected": 12, "matched": 8})
raises("a finder block that is not an object is refused", score_with(finder=[10, 12, 8]), match="not an object")
raises("a finder block missing its matched count is refused",
       score_with(finder={"labelled_punches": 10, "detections": 12}))
raises("a negative labelled count is refused", score_with(finder={**FIND, "labelled_punches": -1}))
raises("a boolean detection count is refused", score_with(finder={**FIND, "detections": True}))
raises("a float matched count is refused", score_with(finder={**FIND, "matched": 8.5}))
raises("more matched than labelled is refused", score_with(finder={**FIND, "matched": 11}), match="more than")
raises("more matched than detected is refused",
       score_with(finder={"labelled_punches": 20, "detections": 9, "matched": 10}), match="more than")
raises("more punches named than matched is refused (3 named, 2 matched)",
       score_with(finder={"labelled_punches": 10, "detections": 12, "matched": 2}), match="were named")

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
check("Reliability: then this clip's hand check, both p-values, the floor and the result (12 of 13, 102/169)",
      len(rel) == 10 and all(v is None for v in rel[7]) and rel[8][3] == "agreement"
      and rel[8][5].startswith("p, binomial") and rel[8][6].startswith("p, Fisher") and rel[8][8] == "result"
      and rel[9][1:9] == (13, 12, 0.923, 0.604, 0.0135, 0.014, 0.7, "passed"), str(rel[7:]))
for name, f_, want in (("a warned clip", f95, (42, 26, 0.619, 0.494, 0.0715, 0.0883, 0.7,
                                               "warning: under 70%; binomial not under 0.05; Fisher not under 0.05")),
                       ("a clip far above chance: tiny p is not shown as 0", f06,
                        (41, 33, 0.805, 0.504, 6.83e-05, float(f"{h06['p_fisher']:.3g}"), 0.7, "passed")),
                       ("a split verdict", fb, (8, 7, 0.875, 0.5, 0.0352, 0.0714, 0.7,
                                                "warning: Fisher not under 0.05")),
                       ("a concentrated clip", fa, (12, 12, 1.0, 1.0, 1.0, 1.0, 0.7,
                                                    "cannot check: types concentrated on one hand")),
                       ("too few", f4, (4, 4, 1.0, 0.5, 0.0625, 0.167, 0.7, "cannot check: too few named punches")),
                       ("both tests pass, under the floor", fl, (100, 68, 0.68, 0.5, float(f"{fl['hand_check']['p_binomial']:.3g}"),
                                                                 float(f"{fl['hand_check']['p_fisher']:.3g}"), 0.7,
                                                                 "warning: under 70%")),
                       ("Fisher exactly at alpha", fq, (16, 14, 0.875, 0.688, float(f"{exact_tail(14, 16, Fraction(176, 256)):.3g}"), 0.05, 0.7,
                                                        "warning: binomial not under 0.05; Fisher not under 0.05")),
                       ("a clip with no named punch", f0, (0, 0, None, None, None, None, 0.7, "no named punch"))):
    p_ = tmp / "hc.xlsx"
    report.write_workbook(p_, "IMG_1", [], ROUNDS, f_, bench, "x", FPS, HOW)
    got = list(load_workbook(p_)["Reliability"].iter_rows(values_only=True))[9][1:9]
    check(f"Reliability hand-check row for {name}: {want}", got == want, str(got))
about = [r[0] for r in wb["About"].iter_rows(values_only=True)]
check("About holds the summary, how the clip was cut, and the held-out videos from the reference",
      text in about and HOW in about and any("V2/V5/V9" in (a or "") for a in about))
check("About states the warning's rule: under 70%, or either one-sided exact test (binomial, Fisher) at 0.05",
      any("under 70% of the time" in (a or "") and "either of two one-sided exact tests at 0.05" in a
          and "Fisher" in a and "cannot confirm them" in a for a in about))
check("About says the check sees the hand only, and how stance enters",
      any("cannot tell a jab from a lead hook" in (a or "") and "wrong stance swaps them" in a for a in about))
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
        "--reference", str(ref), "--rule", str(rule), "--rounds", "0-60,90-150", "--out-dir", str(tmp / "out"),
        "--combo-gate", str(tmp / "no-combo-gate.json")]   # pinned: a real gate result on disk must not leak in
rc = report_punches.main(args)
from boxingvi import combos  # noqa: E402
cf_e2e = combos.compute(combos.group(combos.solo_punches(rows, ROUNDS), FPS), RULE["min_gap_s"])
text_e2e, _ = report.summary(facts, bench, finder, None, combinations=(cf_e2e, None, False))
check("end to end: workbook and summary written, summary equals the library's (with combinations)",
      rc == 0 and (tmp / "out" / "IMG_1.report.xlsx").is_file()
      and (tmp / "out" / "IMG_1.summary.txt").read_text(encoding="utf-8").strip() == text_e2e, text_e2e)
check("end to end: the combinations add to the summary, they do not replace it", text_e2e.startswith(text[:200]))
import contextlib  # noqa: E402
import io as _io  # noqa: E402
with contextlib.redirect_stdout(_io.StringIO()):
    report_punches.main(args + ["--prompt-out"])
_pp = tmp / "out" / "IMG_1.narration-prompt.json"
_pr = json.loads(_pp.read_text(encoding="utf-8")) if _pp.is_file() else {}
check("end to end: --prompt-out writes a narration prompt whose fallback is the summary itself, outside any "
      "participant folder so it may leave the machine",
      _pr.get("format") == "boxingvi-narration-prompt-1" and _pr.get("may_leave_machine") is True
      and _pr.get("template_text") == text_e2e and _pr.get("facts"))
check("end to end: the workbook has a Combinations sheet",
      "Combinations" in load_workbook(tmp / "out" / "IMG_1.report.xlsx").sheetnames)
sidecar(pose_frames=n + 1)
raises("a CSV classified on a pose of another length is refused", lambda: report_punches.main(args),
       match="9601-frame pose")
sidecar()
# Only the fps changes: same clip name, same 9600 frames, same CSV and sidecar; the pose is at 30 fps.
pose30 = pose_yolo.PoseSequence(np.zeros((n, N_JOINTS, 2), np.float32), np.zeros((n, N_JOINTS), np.float32),
                                np.zeros(n, bool), 30.0, 1080, 1920, "IMG_1.mov", "test")
(tmp / "p30").mkdir()
pose30.save(tmp / "p30" / "IMG_1.pose.npz")
args30 = ["--pose", str(tmp / "p30" / "IMG_1.pose.npz")] + args[2:]
raises("CLI: a CSV classified at 60 fps is refused against a same-named, same-length pose at 30 fps",
       lambda: report_punches.main(args30), match="classified on a pose at 60.0 fps")
sidecar(fps=30.0)
raises("... and still refused when the sidecar is made to say 30 fps: every row's time is frame / 60",
       lambda: report_punches.main(args30), match="is not frame 120 at 30.000 fps")
sidecar(rule={**RULE, "threshold": 3.0})
raises("CLI: punches found with another threshold than --rule's are refused", lambda: report_punches.main(args),
       match="another rule")
sidecar(rule={**RULE, "variant": "radial"})
raises("CLI: punches found with another variant than --rule's are refused", lambda: report_punches.main(args),
       match="another rule")
sidecar()
raises("CLI: a reference whose matrix is the manifest's transposed is refused",
       lambda: report_punches.main(args[:6] + ["--reference", str(refT)] + args[8:]), match="rows and columns swapped")
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
# The clip is classified again (here: the jab at 75 s is gone), same clip and same model; the score on disk is the
# old run's. Its accuracy described punches that are no longer the ones in the report.
(tmp / "re").mkdir()
re_csv = write_csv(tmp / "re" / "IMG_1.classified.csv", R1 + R2)
RE_SHA = hashlib.sha256(re_csv.read_bytes()).hexdigest()
(tmp / "re" / "IMG_1.classified.json").write_text(json.dumps({**sidecar(), "csv_sha256": RE_SHA}), encoding="utf-8")
args_re = args[:2] + ["--classified", str(re_csv)] + args[4:]
check("control: the re-classified CSV reports on its own", report_punches.main(args_re) == 0)
raises("CLI: the old score is refused for the re-classified CSV (same clip, same classifier)",
       lambda: report_punches.main(args_re + ["--score", str(score)]), match="another classified CSV")
score_with(csv_sha256=RE_SHA)
check("... and a score of the re-classified CSV is read", report_punches.main(args_re + ["--score", str(score)]) == 0)
raises("... which in turn is refused for the first CSV", lambda: report_punches.main(args + ["--score", str(score)]),
       match="another classified CSV")

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
