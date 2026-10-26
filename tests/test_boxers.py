"""Tests for the per-boxer report: the identity file, punches per boxer, the report, and analyse_clip --two-boxer.

Synthetic only: two hand-made bodies with planted punches (boxer A at the left, B at the right), a still referee,
a randomly initialised classifier and hand-written rule files. No number here says anything about boxing.
Weighted toward the trust boundaries: every file the report reads is forged, swapped or mismatched once, and
each refusal is checked. Two equivalences pin the new path to the tested old ones: on a linker identity the
per-boxer finder gives exactly completeness.py's bursts, and on one unbroken person it names exactly what
find_punches.py + classify_punches.py name.

Run:  python tests/test_boxers.py
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from openpyxl import load_workbook

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import analyse_clip  # noqa: E402
import boxer_punches  # noqa: E402
import classify_punches  # noqa: E402
import find_punches  # noqa: E402
import report_boxers  # noqa: E402
from boxingvi import boxers, classify, completeness, punchfind, report, video  # noqa: E402
from boxingvi.data import CLASSES, N_JOINTS  # noqa: E402
from boxingvi.linking import Linking  # noqa: E402
from boxingvi.pose_yolo import PoseSequence, PoseTable  # noqa: E402
from boxingvi.stgcnpp import STGCNPP  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def refused(name: str, fn, needle: str) -> None:
    """fn() must raise ValueError whose message contains `needle`."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except ValueError as e:
        check(name, needle in str(e), f"refused, but with: {e}")
        return
    except Exception as e:   # noqa: BLE001 -- a crash is not a refusal
        check(name, False, f"{type(e).__name__}: {e}")
        return
    check(name, False, "was accepted")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


tmp = Path(tempfile.mkdtemp())
W, H, FPS, SECONDS = 320, 120, 60, 20
N = FPS * SECONDS
HANDOVER = 750              # boxer A: track 1 before, track 4 from here (no punch within 0.4 s of it)
B_HOLE = (900, 929)         # boxer B (track 2) undetected: a hole longer than the bridge
B_UNSURE = (300, 449)       # B's frames the method cannot name
REF = 9                     # a referee: tracked, never anybody's identity

POSE = np.array([
    [0, -150], [-8, -158], [8, -158], [-16, -154], [16, -154],
    [-50, -100], [50, -100], [-60, -50], [60, -50], [-55, -10], [55, -10],
    [-30, 0], [30, 0], [-32, 90], [32, 90], [-34, 180], [34, 180]], dtype=np.float64)


def body(cx: float, left=(), right=(), n: int = N) -> np.ndarray:
    t = np.arange(n) / FPS
    xy = np.broadcast_to(POSE * 0.25 + (cx, 60), (n, N_JOINTS, 2)).copy()
    for hand, times in ((0, left), (1, right)):
        for p in times:
            ph = np.clip((t - p) / 0.25, 0, 1)
            xy[:, 9 + hand, 0] += (-1 if hand == 0 else 1) * 30 * np.sin(np.pi * ph)
    return xy.astype(np.float32).astype(np.float64)   # what a saved pose holds


A_LEFT, A_RIGHT = (2, 5, 8, 11, 14), (3.5, 9.5, 16)
B_LEFT, B_RIGHT = (3, 6.5, 12), (4, 7.2, 13, 17.5)   # 6.5 s and 7.2 s fall in B's unsure frames
XA, XB, XR = body(80, A_LEFT, A_RIGHT), body(240, B_LEFT, B_RIGHT), body(160)
XA_AFTER = XA.copy()
XA_AFTER[:, 9:11, 0] += 15.0     # track 4 is another detection of A: the wrists sit 15 px off track 1's
CONF = np.float64(np.float32(0.9))


def make_table(people, source: str, n_frames: int = N, fps: float = float(FPS), fp: str = "") -> PoseTable:
    """people: [(track id, frames, xy (n_frames, 17, 2))]."""
    fr, tid, xy = [], [], []
    for t, frames, pxy in people:
        frames = np.asarray(frames, dtype=np.int64)
        fr.append(frames)
        tid.append(np.full(len(frames), t, dtype=np.int64))
        xy.append(pxy[frames])
    fr, tid, xy = np.concatenate(fr), np.concatenate(tid), np.concatenate(xy)
    box = np.stack([xy[:, :, 0].min(1), xy[:, :, 1].min(1), xy[:, :, 0].max(1), xy[:, :, 1].max(1)], axis=1)
    return PoseTable(frame=fr, track_id=tid, box=box, xy=xy, conf=np.full((len(fr), N_JOINTS), CONF),
                     n_frames=n_frames, fps=fps, width=W, height=H, source=source, model="fake", video_fp=fp)


ALL = np.arange(N)
B_FRAMES = np.r_[0:B_HOLE[0], B_HOLE[1] + 1:N]


def bout_people():
    return [(1, np.arange(HANDOVER), XA), (4, np.arange(HANDOVER, N), XA_AFTER), (2, B_FRAMES, XB), (REF, ALL, XR)]


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_video(path: Path, colour: str = "gray") -> Path:
    ffmpeg("-f", "lavfi", "-i", f"color=c={colour}:s={W}x{H}:r={FPS}", "-frames:v", str(N), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", str(path))
    return path


data = tmp / "data"
data.mkdir()
vid = make_video(data / "BOUT01.mp4")
table = make_table(bout_people(), "BOUT01.mp4", fp=video.fingerprint(vid))
tracks = data / "BOUT01.tracks.npz"
table.save(tracks)
table = PoseTable.load(tracks)


def glove_like_names(t: PoseTable) -> list:
    """What a per-row method gives: A for tracks 1 and 4, B for track 2 except its unsure frames, nobody else."""
    out = []
    for f, i in zip(t.frame.tolist(), t.track_id.tolist()):
        if i in (1, 4):
            out.append("A")
        elif i == 2:
            out.append(boxers.UNSURE if B_UNSURE[0] <= f <= B_UNSURE[1] else "B")
        else:
            out.append(None)
    return out


UNGATED = {"verdict": "ungated"}
ident = data / "BOUT01.identity.csv"
boxers.write_identity(ident, table, tracks, boxers.segments_from_row_names(table, glove_like_names(table)),
                      "glove-colour", UNGATED)

# The model files (as tests/test_analyse_clip.py): random weights, the plumbing is under test, not the types.
torch.manual_seed(0)
ck = tmp / "clf.pt"
torch.save(STGCNPP().state_dict(), ck)
C = np.array([[90, 5, 0, 0, 5, 0], [2, 98, 0, 0, 0, 0], [0, 0, 80, 20, 0, 0],
              [0, 10, 40, 50, 0, 0], [30, 0, 0, 0, 20, 0], [10, 0, 0, 0, 0, 90]])
F1 = {c: 2 * int(C[i, i]) / int(C[i].sum() + C[:, i].sum()) for i, c in enumerate(CLASSES)}
GEOM = classify.WindowGeometry(0.375, 0.125, 24.0, 200.0, 0.5, 0.55, "relative", 100)
manifest = tmp / "clf.json"
classify.write_manifest(manifest, ck, GEOM, 50, {"held_out": {"per_class_f1": F1, "confusion": C.tolist()}})
reference = tmp / "ref.json"
reference.write_text(json.dumps({"val_videos": ["V2", "V5", "V9"], "results": [
    {"model": "st-gcn++ pretrained", "confusion": C.tolist(), "per_class_f1": F1}]}), encoding="utf-8")
rule_p = tmp / "rule.json"
rule_p.write_text(json.dumps({"video": "V6", "chosen": {"variant": "relative", "threshold": 1.0, "min_gap_s": 0.4,
                  "test": {"recall": 0.741, "precision": 0.465, "count_ratio": 1.596, "n_labelled": 406}}}),
                  encoding="utf-8")
extent = tmp / "extent.json"
extent.write_text(json.dumps({"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8, "variant": "relative",
                              "beats_fixed_window": True, "gated_cut": classify.GATED_CUT}), encoding="utf-8")
RULE = find_punches.load_rule(rule_p)
MODEL_FILES = ["--classifier", str(manifest), "--rule", str(rule_p)]


def punches_args(out: Path, tr=tracks, idf=ident, extra=()) -> list:
    return (["--tracks", str(tr), "--identity", str(idf), "--stance-a", "orthodox", "--stance-b", "southpaw",
             "--extent", str(extent), "--out-dir", str(out)] + MODEL_FILES + list(extra))


def report_args(out: Path, csv_p: Path, tr=tracks, idf=ident, extra=()) -> list:
    return (["--tracks", str(tr), "--boxers", str(csv_p), "--identity", str(idf), "--reference", str(reference),
             "--out-dir", str(out)] + MODEL_FILES + list(extra))


def fresh(name: str) -> Path:
    p = tmp / name
    p.mkdir()
    return p


print("== the identity file: written, read back, lossless ==")
idn = boxers.read_identity(ident, table, tracks)
segs = sorted((s.track_id, s.first, s.last, s.boxer) for s in idn.segments)
check("segments are maximal runs of one name within one track id",
      segs == [(1, 0, HANDOVER - 1, "A"), (2, 0, B_UNSURE[0] - 1, "B"), (2, B_UNSURE[0], B_UNSURE[1], "unsure"),
               (2, B_UNSURE[1] + 1, N - 1, "B"), (4, HANDOVER, N - 1, "A")], str(segs))
check("the referee, whom the method names nobody, is in no segment", all(s.track_id != REF for s in idn.segments))
back = [None] * len(table.frame)
by = boxers._rows_by_track(table)
for s in idn.segments:
    for r in boxers.segment_rows(table, s, by).tolist():
        back[r] = s.boxer
check("per-row names -> segments -> per-row names is lossless", back == glove_like_names(table))
check("method and verdict carried", idn.method == "glove-colour" and idn.verdict == "ungated")
side = json.loads(boxers.sidecar_path(ident).read_text(encoding="utf-8"))
check("the sidecar binds the tracks file by hash, frames, fps, source and fingerprint",
      side["tracks"] == {"file": "BOUT01.tracks.npz", "sha256": sha(tracks), "source": "BOUT01.mp4",
                         "n_frames": N, "fps": float(FPS), "video_fp": table.video_fp})

print("== adapters: the linker (#11) ==")
mem = np.full((N, 2), -1, dtype=np.int64)
mem[:HANDOVER, 0], mem[HANDOVER:, 0] = 1, 4
mem[B_FRAMES, 1] = 2
lk = Linking(identities=((1, 4), (2,)), membership=mem, n_frames=N, fps=float(FPS), source="BOUT01.mp4",
             region=None)
lnames = boxers.row_names_from_linking(table, lk)
check("linker rows: track 1 and 4 are A, 2 is B, the referee nobody",
      Counter(lnames) == Counter({"A": N, "B": len(B_FRAMES), None: N}))
lnames_u = boxers.row_names_from_linking(table, lk, unsure_tracks=[REF])
check("a credited track the linker left out can be marked unsure", lnames_u.count("unsure") == N)
refused("a linking from another table is refused",
        lambda: boxers.row_names_from_linking(table, Linking(((1,), (2,)), mem, N, float(FPS), "OTHER.mp4", None)),
        "different pose table")
lk_ident = data / "LINK" / "BOUT01.identity.csv"
lk_ident.parent.mkdir()
boxers.write_identity(lk_ident, table, tracks, boxers.segments_from_row_names(table, lnames), "linker", UNGATED)
lk_id = boxers.read_identity(lk_ident, table, tracks)

print("== punches per boxer: segments, handovers, equivalence with completeness.py ==")
bridge, min_rows = boxers.finder_params(table.fps)


def spans(rows_list):
    return [(int(table.frame[r[0]]), int(table.frame[r[-1]]), int(table.track_id[r[0]])) for r in rows_list]


check("boxer A is cut at the handover", spans(boxers.scan_segments_of(table, idn, "A", bridge))
      == [(0, HANDOVER - 1, 1), (HANDOVER, N - 1, 4)])
check("boxer B is cut at the hole longer than the bridge and around the unsure frames",
      spans(boxers.scan_segments_of(table, idn, "B", bridge))
      == [(0, B_UNSURE[0] - 1, 2), (B_UNSURE[1] + 1, B_HOLE[0] - 1, 2), (B_HOLE[1] + 1, N - 1, 2)])
check("the unsure frames are their own segment", spans(boxers.scan_segments_of(table, idn, "unsure", bridge))
      == [B_UNSURE + (2,)])
found_lk = boxers.find(table, lk_id, RULE)
for k, b in enumerate("AB"):
    old = completeness._scan_segments(table, completeness.boxer_segments(table, lk, k, max_gap=bridge), RULE,
                                      "x", bridge, min_rows)
    new = [(int(f), int(h)) for _s, _t, _g, fr, hs in found_lk[b]["kept"] for f, h in zip(fr, hs)]
    check(f"linker identity: boxer {b}'s punches are exactly completeness.find_bursts' scan",
          sorted(new) == sorted((f, h) for f, h, _p in old["raw"]) and len(new) > 0, f"{new} vs {old['raw']}")
    check(f"linker identity: boxer {b}'s exposure is completeness's",
          found_lk[b]["stats"]["exposure_frames"] == int(old["mask"].sum()))

found = boxers.find(table, idn, RULE)


def peak_frames(who):
    return sorted(int(f) for _s, _t, _g, fr, _h in found[who]["kept"] for f in fr)


def near(frames, times):
    """Every planted punch has a found peak within its 0.25 s (the speed peaks inside the movement)."""
    return all(any(round(t * FPS) <= f <= round((t + 0.25) * FPS) for f in frames) for t in times)


check("boxer A's planted punches are found", near(peak_frames("A"), A_LEFT + A_RIGHT), str(peak_frames("A")))
check("boxer B's planted punches outside the unsure frames are found, inside them not",
      near(peak_frames("B"), (3, 12, 4, 13, 17.5)) and len(peak_frames("B")) == 5 and not any(B_UNSURE[0] <= f <= B_UNSURE[1]
                                                                for f in peak_frames("B")))
check("the punches on B's unsure frames are found as unsure", near(peak_frames("unsure"), (6.5, 7.2))
      and len(peak_frames("unsure")) == 2, str(peak_frames("unsure")))
check("no punch at A's handover (the 15 px jump between two detections is never differenced)",
      not any(abs(f - HANDOVER) <= 8 for f in peak_frames("A")), str(peak_frames("A")))
# control: the same rows stitched into ONE track do read the jump as a punch, so the test above can fail
rows = np.flatnonzero(np.isin(table.track_id, (1, 4)))
rows = rows[np.argsort(table.frame[rows], kind="stable")]
st = punchfind.bridge(completeness._track_of(table, rows, "stitched"), bridge)
sf, _ = punchfind.detect(st, punchfind.wrist_signal(st, RULE["variant"]), RULE["threshold"], RULE["min_gap_s"])
check("control: stitched across the handover, the finder fires at it", any(abs(int(f) - HANDOVER) <= 8 for f in sf))
check("the referee is never scanned", all(int(table.track_id[s[0]]) != REF
                                          for w in boxers.NAMES for s, *_ in found[w]["kept"]))

print("== naming: on one unbroken person, exactly the solo path ==")
solo = fresh("solo")
s_tracks = solo / "SOLO01.tracks.npz"
s_table = make_table([(1, ALL, XA), (2, ALL, XB)], "SOLO01.mp4")
s_table.save(s_tracks)
s_table = PoseTable.load(s_tracks)
s_ident = solo / "SOLO01.identity.csv"
boxers.write_identity(s_ident, s_table, s_tracks, boxers.segments_from_row_names(
    s_table, ["A" if t == 1 else "B" for t in s_table.track_id.tolist()]), "ground-truth", UNGATED)
pose = PoseSequence(XA.astype(np.float32), np.full((N, N_JOINTS), CONF, dtype=np.float32), np.ones(N, bool),
                    float(FPS), W, H, "SOLO01.mp4", "fake")
pose.save(solo / "SOLO01.pose.npz")
quiet(find_punches.main, ["--pose", str(solo / "SOLO01.pose.npz"), "--rule", str(rule_p), "--out-dir", str(solo)])
quiet(classify_punches.main, ["--pose", str(solo / "SOLO01.pose.npz"), "--punches", str(solo / "SOLO01.punches.csv"),
                              "--classifier", str(manifest), "--stance=orthodox", "--rule", str(rule_p),
                              "--extent", str(extent), "--out-dir", str(solo)])
quiet(boxer_punches.main, punches_args(solo, s_tracks, s_ident))
with (solo / "SOLO01.classified.csv").open(newline="", encoding="utf-8") as f:
    solo_rows = list(csv.reader(f))
with (solo / "SOLO01.boxers.csv").open(newline="", encoding="utf-8") as f:
    two = list(csv.reader(f))
two_a = [r[2:] for r in two[1:] if r[0] == "A"]
check("boxer A's rows are byte for byte the solo classified rows (finder, cut, window, type, confidence)",
      two_a == solo_rows[1:] and len(two_a) == 8, f"{len(two_a)} vs {len(solo_rows) - 1}")
s_side = json.loads((solo / "SOLO01.boxers.json").read_text(encoding="utf-8"))
c_side = json.loads((solo / "SOLO01.classified.json").read_text(encoding="utf-8"))
check("boxer A's cut is the solo clip's cut", s_side["cuts"]["A"] == c_side["cut"])
check("boxer B is named with B's own stance (southpaw: the right hand leads)",
      all(r[7] == ("lead" if r[4] == "right" else "rear") for r in two[1:] if r[0] == "B")
      and any(r[0] == "B" for r in two[1:]))

print("== the end-to-end report, ungated ==")
run = fresh("run")
rc = quiet(boxer_punches.main, punches_args(run))
check("boxer_punches runs", rc == 0)
box_csv = run / "BOUT01.boxers.csv"
rc = quiet(report_boxers.main, report_args(run, box_csv))
check("report_boxers runs", rc == 0)
texts = {b: (run / f"BOUT01.{b}.summary.txt").read_text(encoding="utf-8").strip() for b in "AB"}
WARN = "Warning: which boxer threw each punch is unverified in this report:"
check("ungated: the FIRST sentence of both reports is the unverified-split warning",
      all(t.startswith(WARN) for t in texts.values()))
check("the header names the method and says it has no gate",
      "told apart by glove-colour, which has not been tested against a gate" in texts["A"])
n_a, n_b = len(peak_frames("A")), len(peak_frames("B"))
check("the unassigned share counts the unsure punches against all found",
      f"Of the {n_a + n_b + 2} punches found on the two boxers, 2 ({round(200 / (n_a + n_b + 2))}%)" in texts["B"],
      texts["B"][:400])
check("each report states that boxer's own count", f"the tool found {n_a} punches" in texts["A"]
      and f"the tool found {n_b} punches" in texts["B"])
check("a boxer's report says 'this boxer's', never 'your', for the boxer's hand and rate",
      all("your lead hand" not in t and "Your punch rate" not in t for t in texts.values())
      and all("this boxer's lead hand" in t for t in texts.values() if "lead hand" in t), texts["A"][:600])
check("each report also gives the rate over the time that boxer was identified",
      all(f"a minute of the time boxer {b} was identified." in texts[b] for b in "AB"), texts["A"][:600])
try:
    report.summary(None, None, None, None, who="their")
    check("summary refuses a possessive outside its fixed set", False)
except ValueError:
    check("summary refuses a possessive outside its fixed set", True)
wb = load_workbook(run / "BOUT01.A.report.xlsx")
check("the workbook has the solo sheets plus Identity", wb.sheetnames == ["Punches", "Rounds", "Reliability", "About",
                                                                          "Identity"])
idrows = {r[0]: r[1] for r in wb["Identity"].iter_rows(values_only=True)}
check("the Identity sheet records method, verdict and the identity file's hash",
      idrows["identity method"] == "glove-colour" and idrows["gate verdict"] == "ungated"
      and idrows["identity file sha256"] == sha(ident))
check("A's Punches sheet holds exactly A's punches", wb["Punches"].max_row - 1 == n_a)
rc = quiet(report_boxers.main, report_args(fresh("rounds"), box_csv, extra=["--rounds", "0-10"]))
rt = (tmp / "rounds" / "BOUT01.B.summary.txt").read_text(encoding="utf-8")
b_in = sum(f < 600 for f in peak_frames("B"))
check("with rounds, the share counts only punches inside them", rc == 0 and
      f"Of the {sum(f < 600 for f in peak_frames('A')) + b_in + 2} punches" in rt, rt[:300])

print("== header: fixed templates, every number traced ==")


def traced(text, nums):
    return Counter(report.numbers_in(text)) == Counter(t for n in nums for t in report.numbers_in(n))


for v, first in (("pass", "This is boxer A's report."), ("fail", WARN), ("ungated", WARN)):
    t, nums = boxers.header("A", "glove-colour", v, 3, 40, 95.0, 1200.0)
    check(f"{v}: first sentence", t.startswith(first), t[:80])
    check(f"{v}: numbers only from the templates", traced(t, nums), f"{report.numbers_in(t)} vs {nums}")
t, _ = boxers.header("B", "linker", "fail", 1, 7, 10.0, 20.0)
check("fail names the failed gate; one unsure punch 'is' counted",
      "(linker) failed its pre-registered gate" in t and "and is counted in neither report" in t)
t, _ = boxers.header("A", "linker", "pass", 0, 0, 10.0, 20.0)
check("nothing found: says so, no share", "found no punch on either boxer" in t and "%" not in t)
t, n = boxers.header("A", "linker", "pass", 0, 12, 10.0, 20.0)
check("nothing unsure: every punch assigned", "Every one of the 12 punches" in t and traced(t, n))
for full in texts.values():
    i = full.index("This is boxer")
    check("full summary: header then solo summary, all numbers traced", full.startswith(WARN) and i > 0)
for bad in (("C", "linker", "pass"), ("A", "linker2", "pass"), ("A", "linker", "maybe")):
    refused(f"header refuses {bad}", lambda bad=bad: boxers.header(*bad, 0, 0, 1.0, 1.0), "out of range")

print("== gated identities ==")
GATE_DOC = boxers.GATES["glove-colour"]["document"]
REAL_ROOT = boxers.ROOT
if not (REAL_ROOT / GATE_DOC).is_file():   # on feat/glove-identity (#31) until it merges
    refused("a registered gate whose document is not in this repository is refused", lambda: boxers._check_gate(
        {"verdict": "fail", "document": GATE_DOC, "result_sha256": "0" * 64}, None, "glove-colour"),
        "no pre-registered gate")
else:
    check("the glove gate document is in this repository", True)
# The rest stands a repository root in which both registered gate documents exist.
fake_root = tmp / "root"
(fake_root / "docs" / "gates").mkdir(parents=True)
for g in boxers.GATES.values():
    (fake_root / g["document"]).write_text("# a pre-registered gate\n", encoding="utf-8")
boxers.ROOT = fake_root


def gated(name: str, verdict: str, result: "bytes | None", method: str = "glove-colour",
          doc: str = GATE_DOC) -> "tuple[Path, Path | None]":
    d = data / name
    d.mkdir()
    res = None
    if result is not None:
        res = d / "gate.json"
        res.write_bytes(result)
    p = d / "BOUT01.identity.csv"
    boxers.write_identity(p, table, tracks, idn.segments, method,
                          {"verdict": verdict, "document": doc,
                           "result_sha256": hashlib.sha256(result or b"none").hexdigest()})
    return p, res


def result_json(passed: bool, bars=(True, True, True, True)) -> bytes:
    """What glove_identity.py gate writes (glovescore.gate_verdict)."""
    keys = boxers.GATES["glove-colour"]["bars"]
    return json.dumps({**dict(zip(keys, bars)), "baseline_best_correct_share": 0.5, "PASS": passed}).encode()


PASSED, FAILED = result_json(True), result_json(False, (True, False, True, True))
p_pass, r_pass = gated("pass", "pass", PASSED)
check("a pass with its gate result is read", boxers.read_identity(p_pass, table, tracks, r_pass).verdict == "pass")
out = fresh("passrun")
quiet(boxer_punches.main, punches_args(out, idf=p_pass, extra=["--gate-result", str(r_pass)]))
rc = quiet(report_boxers.main, report_args(out, out / "BOUT01.boxers.csv", idf=p_pass,
                                           extra=["--gate-result", str(r_pass)]))
pt = (out / "BOUT01.A.summary.txt").read_text(encoding="utf-8")
check("pass: no warning, the header leads", rc == 0 and pt.startswith("This is boxer A's report.")
      and "Warning: which boxer" not in pt)
refused("pass without its gate result", lambda: boxers.read_identity(p_pass, table, tracks), "give its gate result")
other = data / "pass" / "other.json"
other.write_bytes(b'{"PASS": true}')
refused("pass with another gate result file", lambda: boxers.read_identity(p_pass, table, tracks, other),
        "is not the gate result the identity recorded")
p_lie, r_lie = gated("lie", "pass", FAILED)
refused("pass whose gate result says FAIL", lambda: boxers.read_identity(p_lie, table, tracks, r_lie),
        "its gate result says FAIL")
p_fail, r_fail = gated("fail", "fail", FAILED)
check("a fail is read without its result", boxers.read_identity(p_fail, table, tracks).verdict == "fail")
check("a fail is read with its result", boxers.read_identity(p_fail, table, tracks, r_fail).verdict == "fail")
p_fl, r_fl = gated("faillie", "fail", PASSED)
refused("fail whose gate result says PASS", lambda: boxers.read_identity(p_fl, table, tracks, r_fl),
        "its gate result says PASS")
p_np, r_np = gated("nopass", "fail", b'{"verdict": "pass"}')
refused("a gate result with no PASS boolean", lambda: boxers.read_identity(p_np, table, tracks, r_np),
        "not this gate's result")
p_bare, r_bare = gated("bare", "pass", b'{"PASS": true}')
refused("a PASS that is only an assertion (no bars) is not evidence",
        lambda: boxers.read_identity(p_bare, table, tracks, r_bare), "not this gate's result")
p_inc, r_inc = gated("inconsistent", "pass", result_json(True, (True, False, True, True)))
refused("a PASS its own bars contradict", lambda: boxers.read_identity(p_inc, table, tracks, r_inc),
        "PASS is not what its bars say")
p_str, r_str = gated("strbars", "pass", result_json(True, ("yes", True, True, True)))
refused("a bar that is not true or false", lambda: boxers.read_identity(p_str, table, tracks, r_str),
        "not this gate's result")
refused("a gated verdict for a method with no registered gate (the linker)", lambda: gated(
    "linkergate", "fail", FAILED, method="linker"), "no gate is registered for the method linker")
refused("a gate document that is not the method's own", lambda: gated(
    "wrongdoc", "fail", FAILED, doc=boxers.GATES["headgear-colour"]["document"]), "is not the gate pre-registered")
p_hg, r_hg = gated("headgear", "fail", FAILED, method="headgear-colour",
                   doc=boxers.GATES["headgear-colour"]["document"])
check("headgear colour (#29) can record its failed gate", boxers.read_identity(p_hg, table, tracks).verdict == "fail")
refused("... but no result file for it can be checked", lambda: boxers.read_identity(p_hg, table, tracks, r_hg),
        "no result format is registered")
refused("... and it can never record a pass", lambda: boxers.read_identity(
    gated("headgearpass", "pass", PASSED, method="headgear-colour",
          doc=boxers.GATES["headgear-colour"]["document"])[0], table, tracks, data / "headgearpass" / "gate.json"),
    "no result format is registered")
refused("a gate result for an ungated identity", lambda: boxers.read_identity(ident, table, tracks, r_fail),
        "records no gate")
quiet(boxer_punches.main, punches_args(fresh("failrun"), idf=p_fail))
rc = quiet(report_boxers.main, report_args(tmp / "failrun", tmp / "failrun" / "BOUT01.boxers.csv", idf=p_fail))
ft = (tmp / "failrun" / "BOUT01.A.summary.txt").read_text(encoding="utf-8")
check("fail: the first sentence warns, naming the failed gate", rc == 0 and ft.startswith(WARN)
      and "(glove-colour) failed its pre-registered gate" in ft)

print("== refusals: the identity CSV and its sidecar ==")


def variant(name: str, csv_text: "str | None" = None, side_edit=None, rehash: bool = True) -> Path:
    """A copy of the good identity with its CSV and/or sidecar changed; the sidecar re-hashed over the new CSV
    unless rehash=False (a forger who knows the format)."""
    d = data / f"v_{name}"
    d.mkdir()
    p = d / "BOUT01.identity.csv"
    raw = ident.read_bytes() if csv_text is None else csv_text.encode("utf-8")
    p.write_bytes(raw)
    s = json.loads(boxers.sidecar_path(ident).read_text(encoding="utf-8"))
    if rehash:
        s["csv_sha256"] = hashlib.sha256(raw).hexdigest()
    if side_edit:
        side_edit(s)
    boxers.sidecar_path(p).write_text(json.dumps(s), encoding="utf-8")
    return p


good_csv = ident.read_text(encoding="utf-8")
body_lines = good_csv.splitlines()[1:]
HEAD = ",".join(boxers.IDENTITY_HEADER) + "\n"


def csv_of(lines):
    return HEAD + "".join(line + "\n" for line in lines)


def read(p, gate=None):
    return lambda: boxers.read_identity(p, table, tracks, gate)


refused("CSV edited after its sidecar was written", read(variant("edited", good_csv.replace(",B\n", ",A\n", 1),
                                                                 rehash=False)), "not the CSV its sidecar describes")
refused("no sidecar", lambda: (boxers.sidecar_path(variant("noside")).unlink(), read(data / "v_noside" /
                                                                                       "BOUT01.identity.csv")())[1],
        "needs its provenance sidecar")
refused("forged sidecar: an extra key", read(variant("extra", side_edit=lambda s: s.update(note="trust me"))),
        "expected exactly the keys")
refused("forged sidecar: unknown format", read(variant("fmt", side_edit=lambda s: s.update(format="identity-2"))),
        "not a boxingvi-identity-1 sidecar")
refused("forged sidecar: a method name with digits", read(variant("meth", side_edit=lambda s: s.update(method="v2"))),
        "lower-case letters and hyphens")
refused("forged sidecar: an unknown verdict", read(variant("verd", side_edit=lambda s: s.update(
    gate={"verdict": "probably"}))), "gate verdict must be one of")
refused("forged sidecar: ungated with a document", read(variant("ungdoc", side_edit=lambda s: s.update(
    gate={"verdict": "ungated", "document": GATE_DOC}))), "carries no gate document")
refused("forged sidecar: a gate document outside docs/gates", read(variant("trav", side_edit=lambda s: s.update(
    gate={"verdict": "fail", "document": "docs/gates/../../README.md", "result_sha256": "0" * 64}))),
    "must be docs/gates/<name>.md")
refused("forged sidecar: a gate that was never pre-registered", read(variant("nodoc", side_edit=lambda s: s.update(
    gate={"verdict": "fail", "document": "docs/gates/made-up.md", "result_sha256": "0" * 64}))),
    "is not the gate pre-registered for glove-colour")
refused("forged sidecar: wrong tracks hash", read(variant("hash", side_edit=lambda s: s["tracks"].update(
    sha256="0" * 64))), "with other content")
refused("forged sidecar: tracks binding missing a field", read(variant("bind", side_edit=lambda s: s["tracks"].pop(
    "video_fp"))), "exactly the keys")
refused("sidecar not JSON", lambda: (boxers.sidecar_path(variant("nj")).write_text("{oops", encoding="utf-8"),
                                     read(data / "v_nj" / "BOUT01.identity.csv")())[1], "not valid JSON")
refused("sidecar holding NaN", read(variant("nan", side_edit=lambda s: s["tracks"].update(fps=float("nan")))),
        "not valid JSON")
refused("CSV: wrong header", read(variant("hdr", "track,first,last,boxer\n" + "\n".join(body_lines) + "\n")),
        "not an identity CSV")
refused("CSV: an unknown boxer", read(variant("boxc", csv_of(body_lines + ["9,0,10,C"]))), "boxer must be one of")
refused("CSV: a negative frame", read(variant("neg", csv_of(body_lines + ["9,-5,10,unsure"]))), "whole numbers")
refused("CSV: first after last", read(variant("order", csv_of(body_lines + ["9,10,5,unsure"]))), "first frame after")
refused("CSV: a row with a missing field", read(variant("short", csv_of(body_lines + ["9,10,unsure"]))), "3 fields")
refused("CSV: not UTF-8", lambda: (variant("bin"), (data / "v_bin" / "BOUT01.identity.csv").write_bytes(
    b"\xff\xfe" + good_csv.encode()), boxers.sidecar_path(data / "v_bin" / "BOUT01.identity.csv").write_text(
    json.dumps({**json.loads(boxers.sidecar_path(ident).read_text()), "csv_sha256": hashlib.sha256(
        b"\xff\xfe" + good_csv.encode()).hexdigest()}), encoding="utf-8"),
    read(data / "v_bin" / "BOUT01.identity.csv")())[3], "not a readable UTF-8 CSV")

print("== refusals: segments against the tracks ==")
refused("a track the tracks file does not have", read(variant("ghost", csv_of(body_lines + ["77,0,10,unsure"]))),
        "has no such person")
refused("frames where the track is absent (B's hole)", read(variant("hole", csv_of(
    body_lines + [f"2,{B_HOLE[0]},{B_HOLE[1]},unsure"]))), "has no such person")
refused("frames past the video", read(variant("past", csv_of(body_lines + [f"{REF},0,{N + 5},unsure"]))),
        "not inside this")
refused("one track in two overlapping segments", read(variant("twice", csv_of(body_lines + ["1,10,20,unsure"]))),
        "is in two segments")
refused("boxer A two people in one frame", read(variant("twoA", csv_of(body_lines + [f"{REF},0,50,A"]))),
        "two people in one frame")
refused("an empty boxer: nobody is B", read(variant("noB", csv_of([ln for ln in body_lines if not ln.endswith(",B")]))),
        "names nobody as boxer B")
refused("every segment unsure", read(variant("allu", csv_of([ln.rsplit(",", 1)[0] + ",unsure" for ln in body_lines]))),
        "every segment is unsure")
refused("an identity with no segment at all", read(variant("empty", HEAD)), "names nobody as boxer A")
refused("the writer refuses what the reader would", lambda: boxers.write_identity(
    data / "w" / "BOUT01.identity.csv", table, tracks, [boxers.Segment(1, 0, 10, "A")], "linker", UNGATED),
    "names nobody as boxer B")
refused("the writer refuses a method with digits", lambda: boxers.write_identity(
    data / "BOUT01.x.csv", table, tracks, idn.segments, "glove2", UNGATED), "lower-case letters")
refused("segments_from_row_names: a wrong name", lambda: boxers.segments_from_row_names(table, ["Z"] * len(table.frame)),
        "must be one of")
refused("segments_from_row_names: wrong length", lambda: boxers.segments_from_row_names(table, ["A"]),
        "one name per pose-table row")

print("== refusals: the identity against another tracks file ==")


def other_tracks(name: str, **kw) -> "tuple[PoseTable, Path]":
    d = data / f"t_{name}"
    d.mkdir()
    t = make_table(bout_people(), kw.pop("source", "BOUT01.mp4"), fp=kw.pop("fp", table.video_fp), **kw)
    p = d / "BOUT01.tracks.npz"
    t.save(p)
    return PoseTable.load(p), p


t2, p2 = other_tracks("frames", n_frames=N + 30)
refused("frame-count mismatch", lambda: boxers.read_identity(ident, t2, p2), f"of {N} frames")
t3, p3 = other_tracks("fps", fps=50.0)
refused("fps mismatch", lambda: boxers.read_identity(ident, t3, p3), "at 60.0 fps")
t4, p4 = other_tracks("video", source="BOUT02.mp4")
refused("an identity for another video", lambda: boxers.read_identity(ident, t4, p4), "tracks of another video")
t5, p5 = other_tracks("fp", fp="1" * 64)
refused("an identity for another recording (fingerprint)", lambda: boxers.read_identity(ident, t5, p5),
        "tracks of another video")
retracked = make_table(bout_people(), "BOUT01.mp4", fp=table.video_fp)
retracked.conf[0, 0] = 0.5
p6 = data / "t_retrack.tracks"
p6.mkdir()
retracked.save(p6 / "BOUT01.tracks.npz")
refused("the same video re-tracked (other content, same everything else)",
        lambda: boxers.read_identity(ident, PoseTable.load(p6 / "BOUT01.tracks.npz"), p6 / "BOUT01.tracks.npz"),
        "with other content")
renamed = data / "t_name"
renamed.mkdir()
shutil.copy(ident, renamed / "BOUT02.identity.csv")
shutil.copy(boxers.sidecar_path(ident), renamed / "BOUT02.identity.json")
refused("an identity file named for another clip",
        lambda: boxers.read_identity(renamed / "BOUT02.identity.csv", table, tracks), "does not name the clip")

print("== refusals: the per-boxer punches ==")
man = classify.read_manifest(manifest)


def punch_variant(name: str, csv_edit=None, side_edit=None, rehash: bool = True, unsure_edit=None) -> Path:
    d = fresh(f"p_{name}")
    for f in ("BOUT01.boxers.csv", "BOUT01.boxers.json", "BOUT01.unsure.csv"):
        shutil.copy(run / f, d / f)
    s = json.loads((d / "BOUT01.boxers.json").read_text(encoding="utf-8"))
    if csv_edit:
        (d / "BOUT01.boxers.csv").write_text(csv_edit((d / "BOUT01.boxers.csv").read_text(encoding="utf-8")),
                                             encoding="utf-8", newline="")
        if rehash:
            s["csv_sha256"] = sha(d / "BOUT01.boxers.csv")
    if unsure_edit:
        (d / "BOUT01.unsure.csv").write_text(unsure_edit((d / "BOUT01.unsure.csv").read_text(encoding="utf-8")),
                                             encoding="utf-8", newline="")
        if rehash:
            s["unsure_csv_sha256"] = sha(d / "BOUT01.unsure.csv")
    if side_edit:
        side_edit(s)
    (d / "BOUT01.boxers.json").write_text(json.dumps(s), encoding="utf-8")
    return d / "BOUT01.boxers.csv"


def full_check(p, idt=None, tbl=table, tr=tracks):
    def go():
        idt_ = idt or idn
        got = boxers.read_boxers(p, tbl, tr, idt_, man)
        report.finder_reliability(rule_p, got["side"].get("rule"))   # as report_boxers.py does
        boxers.check_against_finder(got, boxers.find(tbl, idt_, RULE), tbl, man)
    return go


box_text = box_csv.read_text(encoding="utf-8")
lines = box_text.splitlines()
a_line = next(i for i, ln in enumerate(lines) if ln.startswith("A,"))
b_line = next(i for i, ln in enumerate(lines) if ln.startswith("B,"))
full_check(box_csv)()   # the untouched set passes
check("the untouched punches pass every check", True)


def swap_line(i, new):
    return lambda t: "\n".join(lines[:i] + [new] + lines[i + 1:]) + "\n"


refused("boxers CSV edited, sidecar not", full_check(punch_variant("edit", lambda t: t.replace(
    lines[a_line], lines[a_line].replace(",0.", ",1.", 1)), rehash=False)), "has changed since")
refused("unsure CSV edited, sidecar not", full_check(punch_variant("uedit", unsure_edit=lambda t: t + "2,400,6.667,"
                                                                   "left,3.000,no\n", rehash=False)), "has changed")
refused("forged: a punch moved from A to B, sidecar re-hashed", full_check(punch_variant(
    "move", swap_line(a_line, "B" + lines[a_line][1:]))), "does not name that track as boxer B")
def invented() -> str:
    """A's first punch copied to frame 700 (track 1, inside A's segment), its time and window moved with it."""
    c = lines[a_line].split(",")
    c[2], c[3], c[12], c[13] = "700", "11.667", "690", "710"
    return ",".join(c)


refused("forged: an invented punch inside A's segments, sidecar re-hashed", full_check(punch_variant(
    "invent", lambda t: t + invented() + "\n")), "not what the finder finds")
refused("forged: a punch deleted, sidecar re-hashed", full_check(punch_variant("delete", swap_line(
    a_line, lines[a_line + 1]))), "not what the finder finds")
refused("forged: unsure punches dropped, sidecar re-hashed", full_check(punch_variant(
    "udrop", unsure_edit=lambda t: t.splitlines()[0] + "\n")), "not what the finder finds")
refused("forged: an unsure punch moved off its unsure segment", full_check(punch_variant(
    "uoff", unsure_edit=lambda t: t + "1,700,11.667,left,3.000,no\n")), "not on an unsure segment")
refused("forged sidecar: the scan statistics", full_check(punch_variant(
    "scan", side_edit=lambda s: s["scan"]["A"].update(identified_frames=1))), "scan the sidecar records")
refused("forged sidecar: the cut", full_check(punch_variant(
    "cut", side_edit=lambda s: s["cuts"]["A"].update(n_used=1))), "cut the sidecar records")
refused("forged sidecar: B's stance", full_check(punch_variant(
    "stance", side_edit=lambda s: s["stances"].update(B="orthodox"))), "for a orthodox boxer")
refused("forged sidecar: a stance that does not exist", full_check(punch_variant(
    "stance2", side_edit=lambda s: s["stances"].update(B="switch"))), "a stance per boxer")
refused("forged sidecar: another classifier", full_check(punch_variant(
    "clf", side_edit=lambda s: s.update(classifier_sha256="0" * 64))), "another classifier")
refused("forged sidecar: an extra key", full_check(punch_variant(
    "key", side_edit=lambda s: s.update(note=1))), "not a boxingvi-boxers-1 sidecar")
refused("forged sidecar: another clip", full_check(punch_variant(
    "clip", side_edit=lambda s: s.update(clip="BOUT02"))), "describes another clip")
refused("forged sidecar: frame count", full_check(punch_variant(
    "frames", side_edit=lambda s: s.update(pose_frames=N + 1))), "-frame pose")
refused("forged sidecar: fps", full_check(punch_variant(
    "fpsx", side_edit=lambda s: s.update(fps=50.0))), "fps")
refused("forged sidecar: tracks hash", full_check(punch_variant(
    "thash", side_edit=lambda s: s["tracks"].update(sha256="f" * 64))), "with other content")
refused("punches made with another identity (the linker's)", full_check(box_csv, idt=lk_id),
        "another identity file")
refused("punches made with the same segments but a failed-gate identity", full_check(
    box_csv, idt=boxers.read_identity(p_fail, table, tracks)), "another identity file")
refused("punches against a re-tracked tracks file", full_check(
    box_csv, tbl=PoseTable.load(p6 / "BOUT01.tracks.npz"), tr=p6 / "BOUT01.tracks.npz"), "with other content")
refused("punches with no sidecar", lambda: ((tmp / "p_edit" / "BOUT01.boxers.json").unlink(),
                                            full_check(tmp / "p_edit" / "BOUT01.boxers.csv")())[1], "no BOUT01.boxers.json")
refused("report_boxers stops on a forged set", lambda: report_boxers.main(report_args(
    fresh("forgedrun"), tmp / "p_invent" / "BOUT01.boxers.csv")), "not what the finder finds")
refused("report_boxers refuses another rule than the punches were found with", lambda: report_boxers.main(
    report_args(fresh("rulerun"), box_csv)[:-2] + ["--rule", str(tmp / "rule2.json")]) if (tmp / "rule2.json").write_text(
    json.dumps({"video": "V6", "chosen": {"variant": "relative", "threshold": 2.0, "min_gap_s": 0.4, "test": {
        "recall": 0.5, "precision": 0.5, "count_ratio": 1.0}}}), encoding="utf-8") else None, "another rule")

print("== refusals: unreadable CSVs fail closed, never with a traceback ==")
# Python 3.13's csv module accepts NUL bytes; the field checks refuse them.
refused("identity CSV with a NUL byte", read(variant("nul", good_csv.replace("A\n", "A\x00\n", 1))),
        "boxer must be one of")
refused("identity CSV with a field over the csv module's limit", read(variant("huge", csv_of(
    body_lines + ["9,0,10," + "x" * 200_000]))), "not a readable UTF-8 CSV")
refused("boxers CSV with a NUL byte, sidecar re-hashed", full_check(punch_variant(
    "pnul", lambda t: t.replace(",no,", ",no\x00,", 1))), "interpolated flag")
refused("unsure CSV with a NUL byte, sidecar re-hashed", full_check(punch_variant(
    "unul", unsure_edit=lambda t: t + "\x00\n")), "not a punch on an unsure segment")

print("== every sidecar field, mutated: refused with a ValueError, never accepted, never a crash ==")
BAD = ([], {}, None, "x", 10 ** 400, -1, True, 1.5)


def leaves(d, path=()):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from leaves(v, path + (k,))
        else:
            yield path + (k,)


def mutate(s, path, value):
    for k in path[:-1]:
        s = s[k]
    s[path[-1]] = value


def sweep(name: str, sidecar: dict, attempt, valid=()) -> None:
    """`valid`: (path, repr(value)) pairs that are well formed by the format's own rules and rightly accepted."""
    accepted, crashed, n = [], [], 0
    for path in leaves(sidecar):
        for val in BAD:
            if ("/".join(path), repr(val)) in valid:
                continue
            n += 1
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    attempt(path, val)
                accepted.append(("/".join(path), repr(val)[:12]))
            except ValueError:
                pass
            except Exception as e:   # noqa: BLE001
                crashed.append(("/".join(path), repr(val)[:12], type(e).__name__))
    check(f"{name}: {n} mutations, none accepted", not accepted, str(accepted[:6]))
    check(f"{name}: none escapes as another exception", not crashed, str(crashed[:6]))


id_side = json.loads(boxers.sidecar_path(ident).read_text(encoding="utf-8"))
_k = {"n": 0}


def try_identity(path, val):
    _k["n"] += 1
    boxers.read_identity(variant(f"m{_k['n']}", side_edit=lambda s: mutate(s, path, val)), table, tracks)


# An ungated method's name is its writer's label: "x" is a valid one. Renaming it changes the sidecar's hash, which
# the punches made with the old identity refuse ("punches made with the same segments but a failed-gate identity").
sweep("identity sidecar", id_side, try_identity, valid={("method", "'x'")})
bx_side = json.loads((run / "BOUT01.boxers.json").read_text(encoding="utf-8"))


def try_boxers(path, val):
    _k["n"] += 1
    full_check(punch_variant(f"m{_k['n']}", side_edit=lambda s: mutate(s, path, val)))()


sweep("boxers sidecar", bx_side, try_boxers)

print("== an identity segment's edge is always a scan boundary ==")
edge_names = glove_like_names(table)
for r in np.flatnonzero((table.track_id == 1) & (table.frame >= 100) & (table.frame <= 102)).tolist():
    edge_names[r] = boxers.UNSURE           # three frames inside A's track 1: shorter than the bridge
edge = data / "EDGE" / "BOUT01.identity.csv"
edge.parent.mkdir()
boxers.write_identity(edge, table, tracks, boxers.segments_from_row_names(table, edge_names), "glove-colour", UNGATED)
edge_id = boxers.read_identity(edge, table, tracks)
check("the method's three unsure frames are never bridged back into A",
      spans(boxers.scan_segments_of(table, edge_id, "A", bridge)) == [(0, 99, 1), (103, HANDOVER - 1, 1),
                                                                       (HANDOVER, N - 1, 4)],
      str(spans(boxers.scan_segments_of(table, edge_id, "A", bridge))))
check("... and no A row of the scan lies in them", not any(
    100 <= int(table.frame[r]) <= 102 for s in boxers.scan_segments_of(table, edge_id, "A", bridge) for r in s))
eo = fresh("edgerun")
rc = quiet(boxer_punches.main, punches_args(eo, idf=edge))
rc2 = quiet(report_boxers.main, report_args(eo, eo / "BOUT01.boxers.csv", idf=edge))
check("what the punch stage writes on it, the report stage accepts", rc == 0 and rc2 == 0)

print("== every extent rule the punch stage accepts, the report stage accepts ==")
ext0 = tmp / "extent0.json"
ext0.write_text(json.dumps({**json.loads(extent.read_text(encoding="utf-8")), "lead_widths": 0}), encoding="utf-8")
eo0 = fresh("extent0")
a0 = punches_args(eo0)
a0[a0.index("--extent") + 1] = str(ext0)
rc = quiet(boxer_punches.main, a0)
rc2 = quiet(report_boxers.main, report_args(eo0, eo0 / "BOUT01.boxers.csv"))
check("a zero lead width (legal for classify.read_extent) passes both stages", rc == 0 and rc2 == 0)

print("== outputs replace a planted link, never write through it ==")
outside = tmp / "outside-root.txt"
outside.write_text("must not change", encoding="utf-8")
lo = fresh("linked")
os.link(outside, lo / "BOUT01.boxers.csv")            # a hard link needs no privilege on NTFS
try:
    os.symlink(outside, lo / "BOUT01.A.summary.txt")
    symlinked = True
except OSError:
    symlinked = False
rc = quiet(boxer_punches.main, punches_args(lo))
rc2 = quiet(report_boxers.main, report_args(lo, lo / "BOUT01.boxers.csv"))
check("hard link planted at an output: the file outside is unchanged, the output is the real one",
      rc == 0 and outside.read_text(encoding="utf-8") == "must not change"
      and (lo / "BOUT01.boxers.csv").read_bytes() == box_csv.read_bytes())
if symlinked:
    check("symbolic link planted at an output: replaced, not followed",
          rc2 == 0 and not (lo / "BOUT01.A.summary.txt").is_symlink()
          and outside.read_text(encoding="utf-8") == "must not change")
else:
    print("  (symbolic links cannot be created without privilege here; the hard-link case covers the same write)")
check("no temporary file is left beside the outputs", not any(p.name.endswith(".tmp") for p in lo.iterdir()))


def boom(p):
    p.write_bytes(b"half")
    raise RuntimeError("disk full")


try:
    boxers.replace_file(lo / "BOUT01.unsure.csv", boom)
except RuntimeError:
    pass
check("a failed write leaves the old output and no temporary file",
      (lo / "BOUT01.unsure.csv").read_bytes() == (run / "BOUT01.unsure.csv").read_bytes()
      and not any(p.name.endswith(".tmp") for p in lo.iterdir()))

print("== participant data never leaves its folder ==")
proot = data   # stands in for ~/comps-participant-data
refused("boxer_punches: participant inputs, --out-dir outside their folder", lambda: boxer_punches.main(
    punches_args(fresh("outside")), participant_roots=(proot,)), "is participant data")
refused("report_boxers: participant inputs, --out-dir outside their folder", lambda: report_boxers.main(
    report_args(fresh("outside2"), box_csv), participant_roots=(proot,)), "is participant data")
refused("an --out-dir that only looks inside (..) is resolved first", lambda: boxer_punches.main(
    punches_args(proot / ".." / "escaped"), participant_roots=(proot,)), "is participant data")
inside = proot / "derived"
check("participant inputs, --out-dir inside their folder: allowed", quiet(
    boxer_punches.main, punches_args(inside), participant_roots=(proot,)) == 0
    and (inside / "BOUT01.boxers.csv").read_bytes() == box_csv.read_bytes())
check("inputs outside every participant folder: no restriction", boxers.check_output_location(
    (tracks, ident), tmp / "anywhere", (tmp / "elsewhere",)) is None)

print("== analyse_clip --two-boxer ==")
out = fresh("analyse")
solo_marker = out / "BOUT01.report.xlsx"   # a solo run's output in the same folder must be left alone
solo_marker.write_bytes(b"solo report")
TWO = ["--two-boxer", "--video", str(vid), "--tracks", str(tracks), "--identity", str(ident), "--stance-a",
       "orthodox", "--stance-b", "southpaw", "--out-dir", str(out), "--classifier", str(manifest), "--rule",
       str(rule_p), "--extent", str(extent), "--reference", str(reference)]
rc = quiet(analyse_clip.main, TWO)
names = sorted(p.name for p in out.iterdir())
check("two-boxer mode publishes its seven outputs", rc == 0 and all(
    f"BOUT01.{s}" in names for s, _ in analyse_clip.TWO_BOXER_OUTPUTS), str(names))
check("no staging folder is left", not any(p.name.startswith(".") for p in out.iterdir()))
check("a solo run's outputs in the same folder are untouched", solo_marker.read_bytes() == b"solo report")
check("the published punches are the ones the stages make",
      (out / "BOUT01.boxers.csv").read_bytes() == box_csv.read_bytes())
check("its summaries lead with the warning", (out / "BOUT01.A.summary.txt").read_text(encoding="utf-8").startswith(WARN))


def exits(name: str, argv, needle: str = "") -> None:
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc_ = analyse_clip.main(argv)
    except SystemExit as e:
        rc_ = e.code
    check(name, rc_ not in (0, None) and needle in err.getvalue(), f"rc {rc_}: {err.getvalue()[-300:]}")


def without(flag: str, n: int = 2):
    i = TWO.index(flag)
    return TWO[:i] + TWO[i + n:]


exits("a missing --stance-a is refused (never guessed)", without("--stance-a"), "--stance-a")
exits("a missing --stance-b is refused", without("--stance-b"), "--stance-b")
exits("--stance is not a two-boxer option", TWO + ["--stance", "orthodox"], "unrecognized")
exits("a missing --identity is refused", without("--identity"), "--identity")
before = sorted((p.name, sha(p)) for p in out.iterdir())
exits("an identity for another tracks file stops before any stage",
      TWO[:TWO.index("--tracks") + 1] + [str(p6 / "BOUT01.tracks.npz")] + TWO[TWO.index("--tracks") + 2:],
      "Nothing was run or written")
exits("a pass without its gate result stops before any stage",
      TWO[:TWO.index("--identity") + 1] + [str(p_pass)] + TWO[TWO.index("--identity") + 2:], "give its gate result")
exits("tracks named for another clip are refused", TWO[:TWO.index("--tracks") + 1] + [str(p4.parent / "X.tracks.npz")]
      + TWO[TWO.index("--tracks") + 2:], "does not name the clip")
other_vid = make_video(data / "t_video" / "BOUT01.mp4", "navy")
exits("tracks of another recording of the same name are refused", TWO[:TWO.index("--video") + 1] + [str(other_vid)]
      + TWO[TWO.index("--video") + 2:], "fingerprint")
check("after every refusal the out-dir is exactly as it was", sorted((p.name, sha(p)) for p in out.iterdir()) == before)
rc = quiet(analyse_clip.main, TWO[:TWO.index("--identity") + 1] + [str(p_pass)] + TWO[TWO.index("--identity") + 2:]
           + ["--gate-result", str(r_pass)])
check("a pass with its gate result runs through, and replaces the previous two-boxer set",
      rc == 0 and (out / "BOUT01.A.summary.txt").read_text(encoding="utf-8").startswith("This is boxer A")
      and solo_marker.read_bytes() == b"solo report")

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
