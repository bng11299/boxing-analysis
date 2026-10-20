"""Tests for eval_pipeline_v5v9.py: the matching, the scoring arithmetic, the
block bootstrap, and every file it reads between tools (classified CSV and
sidecar, annotation rows, audit judgments).

Synthetic fixtures only. No number here says anything about boxing.

Run:  python tests/test_e2e.py
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import eval_pipeline_v5v9 as ev  # noqa: E402
from boxingvi import punchfind  # noqa: E402
from boxingvi.data import CLASSES, Span  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, exc=(ValueError,)) -> None:
    try:
        fn()
    except exc:
        check(name, True)
        return
    except Exception as e:  # noqa: BLE001 - the wrong exception is a failure, reported
        check(name, False, f"raised {type(e).__name__}: {e}")
        return
    check(name, False, "did not raise")


JAB, CROSS = CLASSES.index("jab"), CLASSES.index("cross")
TMP = Path(tempfile.mkdtemp())

print("distance to labels")
d = ev.distance_to([0, 5, 12, 20, 31], [(5, 10), (25, 30)])
check("inside is 0, before / between / after measured to the nearer edge, worked by hand",
      d.tolist() == [5, 0, 2, 5, 1], str(d.tolist()))
check("no labels at all: every detection is infinitely far", (ev.distance_to([3], []) > 10 ** 9).all())

print("matching")
# The matcher itself is punchfind.match_nearest, tested in tests/test_hands.py. Pinned here: this script uses
# it and keeps no copy of its own that could drift from what classify_punches and eval_classify_v6 use.
combo = [Span(10, 18, JAB), Span(19, 26, CROSS)]
check("the script keeps no matcher of its own", not hasattr(ev, "match_nearest"))
check("split_detections attributes with punchfind.match_nearest by default: the jab-cross detection is the cross's",
      ev.split_detections([22], combo, [], 4, 30)["pairs"] == [(1, 0)], str(ev.split_detections([22], combo, [], 4, 30)))

rng = np.random.default_rng(3)   # the later random cases' generator (it was seeded here before the move)

print("splitting detections")
spans = [Span(100, 110, JAB), Span(200, 210, CROSS)]
s = ev.split_detections([105, 160, 213, 300, 420, 404], spans, [(400, 402)], tol=3, far=9)
check("hit, unmatched far, unmatched near, unmatched far, far, on an unusable row",
      s["hit"].tolist() == [True, False, True, False, False, False]
      and s["ignored"].tolist() == [False, False, False, False, False, True]
      and s["far"].tolist() == [False, True, False, True, True, False], str(s))
s2 = ev.split_detections([215], spans, [], tol=3, far=9)
check("5 frames past a label is unmatched but NOT far (within the audit distance)",
      not s2["hit"][0] and not s2["far"][0])
s3 = ev.split_detections([395], spans, [(400, 402)], tol=3, far=9)
check("near an unusable row but outside its tolerance: not ignored, and not far either",
      not s3["ignored"][0] and not s3["far"][0])

print("confusion-matrix metrics")
from sklearn.metrics import f1_score  # noqa: E402

ok = True
for k in range(100):
    n = int(rng.integers(1, 60))
    y = rng.integers(0, 4 if k % 2 else 6, n)        # half the draws miss classes entirely
    p = rng.integers(0, 6, n)
    cm = ev.confusion(y, p)
    ok &= abs(ev.cm_macro_f1(cm) - f1_score(y, p, labels=range(6), average="macro", zero_division=0)) < 1e-12
    ok &= abs(ev.cm_accuracy(cm) - (y == p).mean()) < 1e-12
    ok &= abs(ev.cm_hand(cm) - (ev.SIDE[y] == ev.SIDE[p]).mean()) < 1e-12
check("macro F1, accuracy and hand accuracy from the confusion matrix equal sklearn / direct, 100 draws", bool(ok))
check("empty confusion: undefined, not 0", ev.cm_accuracy(np.zeros((6, 6))) is None
      and ev.cm_macro_f1(np.zeros((6, 6))) is None and ev.cm_hand(np.zeros((6, 6))) is None)
check("jab named cross is the wrong type and the wrong hand",
      ev.cm_hand(ev.confusion([JAB], [CROSS])) == 0.0 and ev.cm_accuracy(ev.confusion([JAB], [CROSS])) == 0.0)

print("block bootstrap")
const = {"A": np.tile([3, 4], (10, 1)), "B": np.tile([1, 2], (7, 1))}
r = ev.block_ci(const, lambda t: t[0] / t[1])
check("identical blocks: the interval collapses onto the point", r["ci_low"] == r["ci_high"] == r["point"] == 37 / 54,
      str(r))
strata = {"X": np.tile([1, 0], (8, 1)), "Y": np.tile([0, 1], (3, 1))}
r = ev.block_ci(strata, lambda t: t[0] / (t[0] + t[1]))
check("blocks are resampled inside their own video: every draw keeps 8 of X and 3 of Y",
      r["ci_low"] == r["ci_high"] == 8 / 11, str(r))
var = {"X": np.array([[1, 1], [0, 1], [1, 1], [0, 1]])}
r = ev.block_ci(var, lambda t: t[0] / t[1])
check("varying blocks: a real interval around the point", r["ci_low"] < r["point"] < r["ci_high"], str(r))
sparse = {"X": np.array([[1, 1]] + [[0, 0]] * 3)}
r = ev.block_ci(sparse, lambda t: t[0] / t[1] if t[1] else None, n_boot=400)
check("draws where the statistic is undefined are dropped and counted, not scored 0",
      r["n_undefined_dropped"] > 0 and r["ci_low"] == 1.0, str(r))
raises("undefined on the data itself: refused", lambda: ev.block_ci({"X": np.zeros((3, 2))}, lambda t: None))
raises("undefined on most draws: refused",
       lambda: ev.block_ci({"X": np.array([[1]] + [[0]] * 30)}, lambda t: 1.0 if t[0] == 1 else None))  # ~63% undefined

print("Wilson interval")
w0, w5 = ev.wilson(0, 10), ev.wilson(5, 10)
check("0 of 10: [0, 0.278]", abs(w0[0]) < 1e-12 and abs(w0[1] - 0.27753) < 1e-4, str(w0))
check("5 of 10: [0.237, 0.763]", abs(w5[0] - 0.23659) < 1e-4 and abs(w5[1] - 0.76341) < 1e-4, str(w5))
check("nothing sampled: no interval", ev.wilson(0, 0) is None)

print("tally")
spans = [Span(0, 9, JAB), Span(30, 39, CROSS), Span(70, 79, JAB)]
rows = [{"frame": 5, "hand": "left", "lead_rear": "lead", "type": JAB},
        {"frame": 35, "hand": "left", "lead_rear": "lead", "type": JAB},
        {"frame": 50, "hand": "right", "lead_rear": "rear", "type": CROSS},
        {"frame": 72, "hand": "left", "lead_rear": "lead", "type": None}]
det = np.array([r["frame"] for r in rows])
refs = {("V", 0): (JAB, JAB), ("V", 1): (CROSS, JAB), ("V", 2): (JAB, JAB)}
t = ev.tally("V", punchfind.match_nearest(det, spans, 2), det, det, spans, rows, refs, block=40, n_blocks=2)
check("per-block detections, labels, hits", t["fin"][:, :4].tolist() == [[2, 2, 2, 2], [2, 1, 1, 1]], str(t["fin"][:, :4]))
rbt = ev.recall_by_type({"V": t["fin"]}, False)
check("recall by type: both jabs found, the cross found", rbt["jab"] == {"labelled": 2, "found": 2, "recall": 1.0}
      and rbt["cross"]["found"] == 1)
ts = ev.type_summary({"V": t["typ"]})
check("type on matched and classified only: pipeline 1 of 2, Y_clip 1 of 2 (the cross read as jab), Y_all 2 of 2",
      ts["n"] == 2 and ts["pipeline"]["accuracy"] == 0.5 and ts["Y_clip"]["accuracy"] == 0.5
      and ts["Y_all"]["accuracy"] == 1.0, str(ts))
check("the matched punch the classifier could not read is counted, not scored", t["unclassified"] == 1)
check("finder hand under the stance against the label's side", t["stance"] == [True, False, True])
ti = ev.tally("V", [(0, 0)], np.array([5, 7]), np.array([5]), spans, rows[:1] + [rows[0]],
              {}, block=40, n_blocks=2)
check("a detection left out of the scored set (on an unusable row) is not counted as a detection",
      ti["fin"][:, 0].tolist() == [1, 0])
check("a matched span with no perfect-finder reading is counted, not paired and not a crash",
      ti["no_reference"] == 1 and ti["typ"].sum() == 0)
# A hit that straddles a block edge: the span starts in block 0, its detection lands in block 1.
edge = [Span(35, 45, JAB)]
er = [{"frame": 44, "hand": "left", "lead_rear": "lead", "type": JAB}]
te = ev.tally("V", punchfind.match_nearest([44], edge, 2), np.array([44]), np.array([44]), edge, er,
              {("V", 0): (JAB, JAB)}, block=40, n_blocks=2)
check("across a block edge: precision's hit goes with the detection, recall's and the type's with the span",
      te["fin"][:, 2].tolist() == [0, 1] and te["fin"][:, 3].tolist() == [1, 0]
      and te["fin"][:, 4 + len(CLASSES) + JAB].tolist() == [1, 0]
      and ev.cm_from(te["typ"][0], 0).sum() == 1 and te["typ"][1].sum() == 0, str(te["fin"]))

print("annotation rows the clip loader left out")
import pandas as pd  # noqa: E402

(TMP / "Annotation_files").mkdir()
pd.DataFrame({"start": [10, 30, 50, 70, np.nan], "end": [18, 25, 58, 78, np.nan],
              "class": ["jab", "cross", None, "lead hook", "jab"]}).iloc[:4].to_excel(
    TMP / "Annotation_files" / "V5.xlsx", index=False)
check("a labelled row not kept comes back with its frames in order (30..25 -> 25..30); unlabelled skipped",
      ev.unusable_rows(TMP, "V5", {0, 3}) == [(1, 25, 30)], str(ev.unusable_rows(TMP, "V5", {0, 3})))
pd.DataFrame({"start": [10, np.nan, 50], "end": [18, np.nan, 58],
              "class": ["jab", "cross", "jab"]}).to_excel(TMP / "Annotation_files" / "V9.xlsx", index=False)
raises("a labelled punch with no frames at all stops the run (it cannot be placed)",
       lambda: ev.unusable_rows(TMP, "V9", {0, 2}))

print("gates")
raises("spans aligned at another offset: stop", lambda: ev.check_alignment("V5", {"sharp": True, "best": 2, "ratio": 3.0}),
       (SystemExit,))
raises("no sharp offset: stop", lambda: ev.check_alignment("V5", {"sharp": False, "best": 0, "ratio": 1.1}),
       (SystemExit,))
ev.check_alignment("V5", {"sharp": True, "best": 0, "ratio": 3.0})
check("aligned at 0, sharply: passes", True)
REF = {"classifier_sha256": "ab" * 32, "window": {"Y_all": {"n": 742, "accuracy": 0.889487870619946},
                                                   "Y_clip": {"n": 742, "accuracy": 0.8261455525606469}}}
ev.check_reference(REF, 742, 660 / 742, 613 / 742, "ab" * 32)
check("the readings reproduce the reference to the bit: passes", True)
raises("Y_clip off by one punch: stop", lambda: ev.check_reference(REF, 742, 660 / 742, 612 / 742, "ab" * 32),
       (SystemExit,))
raises("another number of punches: stop", lambda: ev.check_reference(REF, 741, 660 / 742, 613 / 742, "ab" * 32),
       (SystemExit,))
raises("another classifier: stop", lambda: ev.check_reference(REF, 742, 660 / 742, 613 / 742, "cd" * 32),
       (SystemExit,))
raises("a reference missing a reading: stop",
       lambda: ev.check_reference({**REF, "window": {"Y_all": REF["window"]["Y_all"]}}, 742, 660 / 742, 613 / 742,
                                  "ab" * 32), (SystemExit,))

print("the classified CSV and its sidecar")
import classify_punches  # noqa: E402

FPS, NF, SHA = 30.0, 1000, "ef" * 32


def classified(rows_, header=None, sidecar=None, name="V5"):
    p, j = TMP / f"{name}.classified.csv", TMP / f"{name}.classified.json"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header or classify_punches.OUT_HEADER)
        w.writerows(rows_)
    side = {"clip": name, "csv_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "stance": "orthodox",
            "classifier_sha256": SHA, "pose_frames": NF, **(sidecar or {})}
    j.write_text(json.dumps(side), encoding="utf-8")
    return p, j


def row(frame=30, hand="left", lr="lead", typ="jab", time=None):
    return [str(frame), f"{(frame / FPS) if time is None else time:.3f}", hand, "4.2", "no", lr, typ, "0.9", "lead",
            "yes", str(frame - 5), str(frame + 5), "11", ""]


def read(p, j, clip="V5", stance="orthodox", sha=SHA):
    return ev.read_classified(p, j, clip, NF, FPS, stance, sha)


got = read(*classified([row(), row(60, "right", "rear", "unclassified")]))
check("a well-formed CSV: frames, hands and types read; unclassified is None",
      [g["frame"] for g in got] == [30, 60] and got[0]["type"] == JAB and got[1]["type"] is None)
p, j = classified([row()])
p.write_bytes(p.read_bytes().replace(b"jab", b"cross"))
raises("a CSV edited after classify_punches.py wrote it (hash): refused", lambda: read(p, j))
raises("another clip's sidecar: refused", lambda: read(*classified([row()], sidecar={"clip": "V9"})))
raises("classified under another stance: refused", lambda: read(*classified([row()]), stance="southpaw"))
raises("classified by another classifier: refused", lambda: read(*classified([row()]), sha="00" * 32))
raises("sidecar for a pose of another length: refused", lambda: read(*classified([row()], sidecar={"pose_frames": 999})))
raises("not classify_punches.py's header: refused",
       lambda: read(*classified([row()], header=classify_punches.OUT_HEADER[:-1] + ["notes"])))
raises("a frame outside the clip: refused", lambda: read(*classified([row(frame=1000)])))
raises("a negative or non-ASCII frame: refused", lambda: read(*classified([["-3"] + row()[1:]])))
raises("time that does not match the frame at this fps: refused", lambda: read(*classified([row(time=1.5)])))
raises("a time of NaN: refused", lambda: read(*classified([row(30)[:1] + ["nan"] + row(30)[2:]])))
RULE = {"variant": "relative", "threshold": 3.6, "min_gap_s": 0.4}
check("the sidecar's finder rule matches: read",
      len(ev.read_classified(*classified([row()], sidecar={"rule": RULE}), "V5", NF, FPS, "orthodox", SHA, RULE)) == 1)
raises("found with another finder rule: refused", lambda: ev.read_classified(
    *classified([row()], sidecar={"rule": {**RULE, "threshold": 2.0}}), "V5", NF, FPS, "orthodox", SHA, RULE))
raises("a punch type the classifier does not have: refused", lambda: read(*classified([row(typ="hook")])))
raises("lead/rear that contradicts the hand under this stance: refused", lambda: read(*classified([row(lr="rear")])))
raises("a short row: refused", lambda: read(*classified([row()[:-1]])))
raises("a sidecar that is not a JSON object: refused",
       lambda: (lambda pj: (pj[1].write_text("[1]", encoding="utf-8"), read(*pj)))(classified([row()])))
old = ev.MAX_CSV_BYTES
ev.MAX_CSV_BYTES = 50
raises("an oversized CSV: refused before parsing", lambda: read(*classified([row()])))
ev.MAX_CSV_BYTES = old

print("audit sample and judgments")
pools = {"V5": {"far": [(i, "left") for i in range(100, 140)], "matched": [(i, "right") for i in range(10)]},
         "V9": {"far": [(i, "left") for i in range(3)], "matched": [(i, "left") for i in range(500, 520)]}}
items = ev.audit_sample(pools, n_far=25, n_controls=5)
kinds = {(v, k): sum(it["video"] == v and it["kind"] == k for it in items) for v in ("V5", "V9") for k in ("far", "control")}
check("25 far and 5 controls per video, or all there are", kinds == {("V5", "far"): 25, ("V5", "control"): 5,
                                                                     ("V9", "far"): 3, ("V9", "control"): 5}, str(kinds))
check("ids are unique and sequential, and say nothing about the kind",
      [it["id"] for it in items] == [f"s{i:03d}" for i in range(len(items))])
check("kinds are mixed in id order, not grouped",
      [it["kind"] for it in items] != sorted([it["kind"] for it in items])
      and [it["kind"] for it in items] != sorted([it["kind"] for it in items], reverse=True))
check("same pools, same sample (seeded); the digest changes when any item does",
      ev.audit_sample(pools) == ev.audit_sample(pools)
      and ev.key_digest(items) != ev.key_digest(items[:-1] + [{**items[-1], "frame": items[-1]["frame"] + 1}]))


def judg(lines, header="id,sample_digest,judgment,note", bom=False):
    p = TMP / "j.csv"
    p.write_bytes((codecs.BOM_UTF8 if bom else b"") + ("\n".join([header] + lines) + "\n").encode("utf-8"))
    return p


small = items[:3]
D = ev.key_digest(small)
good = [f"{it['id']},{D},punch," for it in small]
check("one judgment per id, on this sample: read",
      ev.read_judgments(judg(good), small) == {it["id"]: "punch" for it in small})
check("a file saved by Excel (UTF-8 with a BOM): read", ev.read_judgments(judg(good, bom=True), small) is not None)
check("a quoted note with a comma is one field", ev.read_judgments(
    judg(good[:2] + [f'{small[2]["id"]},{D},unclear,"jab, then cross"']), small)[small[2]["id"]] == "unclear")
raises("an unquoted comma in a note (five fields): refused",
       lambda: ev.read_judgments(judg(good[:2] + [f"{small[2]['id']},{D},punch,jab, then cross"]), small))
other = [{**small[0], "frame": small[0]["frame"] + 1}] + small[1:]
raises("judgments made on another sample with the same ids (digest): refused",
       lambda: ev.read_judgments(judg(good), other))
raises("one row from another sample among good ones: refused",
       lambda: ev.read_judgments(judg(good[:2] + [f"{small[2]['id']},{'0' * 64},punch,"]), small))
raises("a sampled strip left unjudged: refused", lambda: ev.read_judgments(judg(good[:2]), small))
raises("a judgment for a strip not in the sample: refused",
       lambda: ev.read_judgments(judg(good + [f"s999,{D},punch,"]), small))
raises("the same strip judged twice: refused", lambda: ev.read_judgments(judg(good + [good[0]]), small))
try:
    ev.read_judgments(judg(good + [f"x{i},{D},punch," for i in range(10)]), small)
    check("far more rows than strips: refused by the row cap, before reading them all", False, "did not raise")
except ValueError as e:
    check("far more rows than strips: refused by the row cap, before reading them all", "more rows than" in str(e), str(e))
raises("a judgment outside punch / not_punch / unclear: refused",
       lambda: ev.read_judgments(judg(good[:2] + [f"{small[2]['id']},{D},yes,"]), small))
raises("the old header without the digest: refused",
       lambda: ev.read_judgments(judg([f"{it['id']},punch," for it in small], header="id,judgment,note"), small))

print("PNG writer")
img = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
ev.write_png(TMP / "t.png", img)
b = (TMP / "t.png").read_bytes()
w_, h_ = struct.unpack(">II", b[16:24])
ihdr_ok = zlib.crc32(b[12:29]) & 0xFFFFFFFF == struct.unpack(">I", b[29:33])[0]
n = struct.unpack(">I", b[33:37])[0]
raw = zlib.decompress(b[41:41 + n])
back = np.frombuffer(raw, np.uint8).reshape(7, 5 * 3 + 1)[:, 1:].reshape(7, 5, 3)
check("signature, size, CRC and pixels round-trip", b[:8] == b"\x89PNG\r\n\x1a\n" and (w_, h_) == (5, 7)
      and ihdr_ok and (back == img).all())

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
