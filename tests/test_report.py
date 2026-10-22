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
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from boxingvi import classify, pose_yolo, report  # noqa: E402
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
tz = variant(facts, {"n": 0, "unclassifiable": 0}, name="labelled, nothing matched")
check("labelled but nothing matched: not 'label a clip'", "You labelled this clip, but no labelled punch" in tz
      and "label a clip" not in tz, tz)

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
      len(rel) == 7 and [r[0] for r in rel[1:] if r[7] == "yes"] == ["lead uppercuts", "rear hooks"])
about = [r[0] for r in wb["About"].iter_rows(values_only=True)]
check("About holds the summary, how the clip was cut, and the held-out videos from the reference",
      text in about and HOW in about and any("V2/V5/V9" in (a or "") for a in about))
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
