"""Tests for the report: reading classified punches, rounds, reliability, the
template summary and the spreadsheet, and report_punches.py end to end.

Synthetic fixtures only: planted punches with counts worked by hand. No number
here says anything about boxing.

Run:  python tests/test_report.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from boxingvi import pose_yolo, report  # noqa: E402
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
LEAD = {"jab", "lead_hook", "lead_uppercut"}


def row(t, ptype, hand="left", conf=0.9, note=""):
    """A classify_punches.py row, orthodox stance (left leads)."""
    side = "lead" if hand == "left" else "rear"
    if ptype == "unclassified":
        return [str(int(t * FPS)), f"{t:.3f}", hand, "5.000", "no", side, "unclassified", "", "", "",
                "1", note or "pose in only 1 of the window's 9 frames"]
    th = "lead" if ptype in LEAD else "rear"
    return [str(int(t * FPS)), f"{t:.3f}", hand, "5.000", "no", side, ptype, f"{conf:.3f}", th,
            "yes" if th == side else "no", "9", note]


def write_csv(path, rows, header=report.CLASSIFIED_HEADER):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


print("the CSV contract with classify_punches.py")
import classify_punches  # noqa: E402

check("report reads exactly the columns classify_punches writes", report.CLASSIFIED_HEADER == classify_punches.OUT_HEADER)

# Round 1 (0-60 s): 6 jabs, 2 crosses, 1 lead hook, 1 unnamed = 10. Round 2 (90-150 s): 3 jabs, 1 cross = 4.
# One punch at 75 s, between rounds.
R1 = ([row(2 + 5 * i, "jab") for i in range(6)] + [row(40, "cross", "right"), row(45, "cross", "right"),
      row(50, "lead_hook"), row(55, "unclassified", "right")])
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
r_ = row(1, "jab"); r_[1] = "nan"; bad("a NaN time is refused", [r_])
r_ = row(1, "jab"); r_[6] = "haymaker"; bad("an unknown punch type is refused", [r_])
r_ = row(1, "jab"); r_[7] = "1.500"; bad("a confidence above 1 is refused", [r_])
r_ = row(1, "jab"); r_[7] = ""; bad("a named punch without a confidence is refused", [r_])
r_ = row(1, "unclassified"); r_[7] = "0.5"; bad("an unclassified punch carrying a confidence is refused", [r_])
r_ = row(1, "jab"); r_[8] = "rear"; bad("a type hand that does not follow from the type is refused", [r_])
r_ = row(1, "jab"); r_[9] = "no"; bad("an agreement flag that does not follow from the hands is refused", [r_])
r_ = row(1, "jab"); r_[2] = "both"; bad("an unknown hand is refused", [r_])
r_ = row(1, "jab"); r_[0] = "²"; bad("a non-ASCII digit is refused", [r_])
r_ = row(1, "jab", note="=HYPERLINK(\"http://x\")"); bad("a note that would be a spreadsheet formula is refused", [r_])
bad("a short row is refused", [row(1, "jab")[:5]])

print("rounds")
check("default: the whole clip is one round", report.parse_rounds(None, DUR) == [(0.0, DUR)])
check("explicit rounds parse in seconds", report.parse_rounds("0-60, 90-150", DUR) == [(0.0, 60.0), (90.0, 150.0)])
for txt, why in (("0-60,50-120", "overlapping"), ("0-200", "past the clip"), ("60-60", "empty"),
                 ("a-b", "not numbers"), ("0-60;90-150", "wrong separator"), ("-5-10", "negative")):
    raises(f"rounds: {why} is refused", lambda txt=txt: report.parse_rounds(txt, DUR))

print("facts, worked by hand")
ROUNDS = report.parse_rounds("0-60,90-150", DUR)
facts = report.compute_facts(rows, ROUNDS, DUR)
check("14 punches inside rounds, 1 between", facts["n_found"] == 14 and facts["n_outside_rounds"] == 1)
check("13 named, 1 not", facts["n_named"] == 13 and facts["n_unnamed"] == 1)
check("types over both rounds: 9 jabs, 3 crosses, 1 lead hook",
      facts["types"] == {"cross": 3, "jab": 9, "lead_hook": 1, "lead_uppercut": 0, "rear_hook": 0, "rear_uppercut": 0})
check("rates: 10 a minute in round 1, 4 in round 2, 7 overall",
      [p["per_min"] for p in facts["rounds"]] == [10.0, 4.0] and facts["per_min"] == 7.0)
check("lead share: 10 of 14 from the left (lead) hand", abs(facts["lead_share"] - 10 / 14) < 1e-12)
check("a punch exactly at the end of the last round counts in it",
      report.round_of(150.0, ROUNDS) == 1 and report.round_of(60.0, ROUNDS) is None)

print("reliability")
# Hand-made confusion (rows true, columns predicted), CLASSES order:
# cross, jab, lead_hook, lead_uppercut, rear_hook, rear_uppercut
C = np.array([[90, 5, 0, 0, 5, 0],
              [2, 98, 0, 0, 0, 0],
              [0, 0, 80, 20, 0, 0],
              [0, 10, 40, 50, 0, 0],
              [30, 0, 0, 0, 20, 0],
              [10, 0, 0, 0, 0, 90]])
f1 = {c: 0.5 for c in CLASSES}
ref = tmp / "stgcnpp.json"
ref.write_text(json.dumps({"results": [{"model": "st-gcn++ pretrained", "confusion": C.tolist(),
                                        "per_class_f1": f1}]}), encoding="utf-8")
man = tmp / "clf.json"
man.write_text(json.dumps({"held_out": {"per_class_f1": f1}}), encoding="utf-8")
bench = report.benchmark_reliability(ref, man)
t = {x.cls: x for x in bench["types"]}
check("cross: 100 true, 132 counted (+32%), recall 0.90",
      t["cross"].true == 100 and t["cross"].predicted == 132 and abs(t["cross"].recall - 0.9) < 1e-12)
check("rear hook: 50 true, 25 counted, recall 0.40", t["rear_hook"].true == 50 and t["rear_hook"].predicted == 25
      and abs(t["rear_hook"].recall - 0.4) < 1e-12)
# Off-side cells: cross read as jab (5), jab read as cross (2). Everything else stays on its hand.
check("hand right: same-side cells over all, 543 of 550", abs(bench["side_accuracy"] - 543 / 550) < 1e-12,
      str(bench["side_accuracy"]))
man.write_text(json.dumps({"held_out": {"per_class_f1": {**f1, "jab": 0.6}}}), encoding="utf-8")
raises("a classifier whose per-type F1 differs from the reported run does not borrow its reliability",
       lambda: report.benchmark_reliability(ref, man))
man.write_text(json.dumps({}), encoding="utf-8")
raises("a manifest with no held-out scores is refused", lambda: report.benchmark_reliability(ref, man))
man.write_text(json.dumps({"held_out": {"per_class_f1": f1}}), encoding="utf-8")
bad_ref = tmp / "bad_ref.json"
bad_ref.write_text(json.dumps({"results": [{"model": "st-gcn++ pretrained", "confusion": C[:5].tolist(),
                                            "per_class_f1": f1}]}), encoding="utf-8")
raises("a confusion matrix of the wrong shape is refused", lambda: report.benchmark_reliability(bad_ref, man))
(tmp / "nan.json").write_text('{"results": NaN}', encoding="utf-8")
raises("NaN in a JSON input is refused", lambda: report.benchmark_reliability(tmp / "nan.json", man))
rule = tmp / "rule.json"
rule.write_text(json.dumps({"chosen": {"variant": "relative", "test": {"recall": 0.741, "precision": 0.465,
                                                                       "n_labelled": 406}}}), encoding="utf-8")
finder = report.finder_reliability(rule)
check("finder reliability is the rule's held-out score", finder["recall"] == 0.741)

print("summary: templates, filled")
text, nums = report.summary(facts, bench, finder, None)
print("    " + text)
check("the text holds exactly the numbers the templates inserted, and no other",
      Counter(report.numbers_in(text)) == Counter(t_ for n in nums for t_ in report.numbers_in(n)),
      f"{report.numbers_in(text)} vs {nums}")
# Worked by hand from the planted rows and the matrix above:
# column sums 132, 113, 120, 70, 25, 90 -> cross +32%, lead hook +20% (not over 20%: not flagged),
# lead uppercut -30%, rear hook -50% (25 of 50); recall under 0.70 for lead uppercut (0.50) and rear hook (0.40).
for want in ("In 2 rounds (2:00 of footage) the tool found 14 punches, 7 a minute.",
             "Of the 13 it could name, 69% jabs (9), 23% crosses (3) and 8% lead hooks (1).",
             "1 it could not name: too little of you was in view.",
             "71% came from your lead hand.",
             "Your punch rate dropped by 60%, from 10 a minute in round 1 to 4 in round 2.",
             "1 punch fell between rounds and is not counted.",
             "Trust these counts least: lead uppercuts and rear hooks.",
             "named only 50% of lead uppercuts and 40% of rear hooks correctly.",
             "On the benchmark, counts ran high for crosses (by 32%), and low for lead uppercuts (by 30%) "
             "and rear hooks (by 50%).",
             "right 99% of the time on the benchmark.",
             "found 74% of the labelled punches",
             "None of this has been measured on your own footage yet"):
    check(f"says: {want!r}", want in text, text)
own = {"n": 12, "accuracy": 0.75, "side_accuracy": 1.0}
text_own, _ = report.summary(facts, bench, finder, own)
check("with own labels: the measured accuracy replaces 'not measured'",
      "named 75% of 12 matched punches correctly" in text_own and "not been measured" not in text_own)
one = report.compute_facts(rows, report.parse_rounds(None, DUR), DUR)
text1, _ = report.summary(one, bench, finder, None)
check("one round: no trend sentence, and 'one round' in words", "one round" in text1 and "punch rate" not in text1)
empty = report.compute_facts([], report.parse_rounds("0-60,90-150", DUR), DUR)
text0, n0 = report.summary(empty, bench, finder, None)
check("no punches: says so, and still gives the reliability caveats",
      "found no punches" in text0 and "Trust these counts least" in text0 and "punch rate" not in text0)
steady = report.compute_facts(report.read_classified(write_csv(tmp / "s.csv", [row(10, "jab"), row(100, "jab")]), DUR),
                              ROUNDS, DUR)
check("equal rates are 'held steady'", "Your punch rate held steady" in report.summary(steady, bench, finder, None)[0])

print("spreadsheet")
from openpyxl import load_workbook  # noqa: E402

xl = tmp / "IMG_1.report.xlsx"
report.write_workbook(xl, "IMG_1", rows, ROUNDS, facts, bench, text)
wb = load_workbook(xl)
check("four sheets", wb.sheetnames == ["Punches", "Rounds", "Reliability", "About"])
p = list(wb["Punches"].iter_rows(values_only=True))
check("one Punches row per punch, the between-rounds one marked", len(p) == 1 + 15
      and sum(r[0] == "between rounds" for r in p[1:]) == 1)
rr = list(wb["Rounds"].iter_rows(values_only=True))
check("Rounds: per round and total counts equal the facts",
      [r[3] for r in rr[1:]] == [10, 4, 14] and rr[-1][0] == "all")
rel = list(wb["Reliability"].iter_rows(values_only=True))
check("Reliability: one row per type, trust-least flagged by the rule",
      len(rel) == 7 and [r[0] for r in rel[1:] if r[7] == "yes"] == ["lead uppercuts", "rear hooks"])
check("About holds the summary", any(r[0] == text for r in wb["About"].iter_rows(values_only=True)))
check("no cell is a formula", all(not (isinstance(c.value, str) and c.data_type == "f")
                                  for ws in wb for r in ws.iter_rows() for c in r))
raises("a value that starts like a formula is refused at write time", lambda: report._cell("=1+1"))

print("report_punches CLI")
import report_punches  # noqa: E402

n = int(DUR * FPS)
pose = pose_yolo.PoseSequence(np.zeros((n, N_JOINTS, 2), np.float32), np.zeros((n, N_JOINTS), np.float32),
                              np.zeros(n, bool), FPS, 1080, 1920, "IMG_1.mov", "test")
pose.save(tmp / "IMG_1.pose.npz")
args = ["--pose", str(tmp / "IMG_1.pose.npz"), "--classified", str(good), "--classifier", str(man),
        "--reference", str(ref), "--rule", str(rule), "--rounds", "0-60,90-150", "--out-dir", str(tmp / "out")]
rc = report_punches.main(args)
check("end to end: workbook and summary written, summary equals the library's",
      rc == 0 and (tmp / "out" / "IMG_1.report.xlsx").is_file()
      and (tmp / "out" / "IMG_1.summary.txt").read_text(encoding="utf-8").strip() == text)
raises("a classified CSV named for another clip is refused",
       lambda: report_punches.main(["--pose", str(tmp / "IMG_1.pose.npz"), "--classified", str(tmp / "s.csv"),
                                    "--classifier", str(man), "--reference", str(ref), "--rule", str(rule),
                                    "--out-dir", str(tmp / "out")]))
score = tmp / "IMG_1.classify_score.json"
score.write_text(json.dumps({"clip": "IMG_2", "matched_detections": {"n": 3, "accuracy": 1.0, "side_accuracy": 1.0}}),
                 encoding="utf-8")
raises("a score file for another clip is refused", lambda: report_punches.main(args + ["--score", str(score)]))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
