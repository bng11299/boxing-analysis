"""Tests for the completeness test on CREDITED fragments (ADR-0012 Option 2): completeness.find_bursts_credited,
completeness.box_at / mss, and `olympic_completeness.py --credited`.

No model, no GPU, no Olympic data: every scene is planted. People stand (or drift) at known places with known
track ids and known wrist bursts; the labelled punches are real `olympic.Event`s whose glove box sits on a
wrist, credited by the real `olympic.attribute`, so "who threw it" has a known answer. Scenes are separated by
at least 350 px so a punch of one person never reaches another's wrist.

Run:  python tests/test_completeness_credited.py
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import olympic_completeness  # noqa: E402
from boxingvi import completeness, olympic, punchfind  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, pattern: str = "", exc=ValueError) -> None:
    try:
        fn()
    except exc as e:
        check(name, pattern in str(e), f"wrong reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


tmp = Path(tempfile.mkdtemp())
W, H, FPS = 640, 480, 50.0
RULE = {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.4}
BOX_H = 108.0        # pose(): the box runs cy-52 .. cy+56


def pose(cx, cy, ext=(0.0, 0.0)):
    """A person centred at (cx, cy): torso 40 px, wrists 24 px out to each side, `ext` further out."""
    xy = np.tile([cx, cy], (N_JOINTS, 1)).astype(float)
    xy[:, 0] += np.linspace(-6, 6, N_JOINTS)
    xy[5], xy[6] = (cx - 12, cy - 20), (cx + 12, cy - 20)
    xy[11], xy[12] = (cx - 8, cy + 20), (cx + 8, cy + 20)
    xy[9], xy[10] = (cx - 24 - ext[0], cy - 4), (cx + 24 + ext[1], cy - 4)
    return xy, np.array([cx - 28, cy - 52, cx + 28, cy + 56], float)


def person(tid, path, frames, bursts=(), drop=()):
    """One track id: `path(f) -> (cx, cy)`, the frame ranges it exists in, wrist bursts [(frame, hand)] (hand 0 = left),
    frames missing from the table."""
    return {"tid": tid, "path": path, "frames": frames, "bursts": list(bursts), "drop": set(drop)}


def still(x, y):
    return lambda f: (x, y)


def ext_of(p, f):
    e = [0.0, 0.0]
    for b, hand in p["bursts"]:
        e[hand] = max(e[hand], 120.0 * max(0.0, 1 - abs(f - b) / 4))      # out and back in 8 frames
    return tuple(e)


def build(people, n, source="GH000001.mp4"):
    rows = []
    for p in people:
        for lo, hi in p["frames"]:
            for f in range(lo, hi + 1):
                if f in p["drop"]:
                    continue
                xy, box = pose(*p["path"](f), ext_of(p, f))
                rows.append((f, p["tid"], box, xy))
    rows.sort(key=lambda r: (r[0], r[1]))
    k = len(rows)
    return PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                     np.array([r[2] for r in rows], np.float32).reshape(k, 4),
                     np.array([r[3] for r in rows], np.float32).reshape(k, N_JOINTS, 2),
                     np.full((k, N_JOINTS), 0.9, np.float32), n, FPS, W, H, source=source)


def label(table, tid, first, last, hand):
    """A labelled punch whose glove box covers the person's `hand` wrist over first..last (+-10 px)."""
    pts = []
    for f in range(first, last + 1):
        r = np.flatnonzero((table.frame == f) & (table.track_id == tid))
        if len(r):
            pts.append(table.xy[r[0], 9 + hand])
    assert pts, f"no rows for track {tid} in frames {first}-{last}"
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - 10, pts.max(axis=0) + 10
    return olympic.Event(first, last, "head", ("left", "right")[hand],
                         (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), 1)


def free_label(first, last):
    """A labelled punch nobody is near (an empty corner): unattributed."""
    return olympic.Event(first, last, "head", "left", (4, 4, 20, 20), 1)


def run(table, events, **kw):
    attrs = olympic.attribute(events, table)
    return attrs, completeness.find_bursts_credited(table, events, attrs, RULE, video="GH000001", **kw)


def near(frame, want, tol=6):
    return abs(frame - want) <= tol


def where(res, tid):
    return sorted((b.frame, b.hand, b.match) for b in res["bursts"] if b.track_id == tid)


def close(got, want):
    return len(got) == len(want) and all(near(g[0], w[0]) and g[1:] == w[1:] for g, w in zip(got, want))


# ---------------------------------------------------------------------------------------------------------------------
print("two co-present credited fragments, a referee and a 1-punch fragment: each label explains only its own fragment's burst")
# P (id 1, left) and Q (id 2, right) throw at the SAME moment (100); only P's punch is labelled. R (id 3, a referee) has a
# burst and no credited punch. S (id 4) has exactly one credited punch.
P = person(1, still(150, 150), [(0, 399)], [(100, 1), (250, 0), (338, 1)])
Q = person(2, still(500, 150), [(0, 399)], [(100, 0), (300, 0)])
R = person(3, still(325, 290), [(0, 399)], [(200, 0)])
S = person(4, still(325, 400), [(0, 399)], [(340, 0)])
tA = build([P, Q, R, S], 400)
evA = [label(tA, 1, 98, 102, 1), label(tA, 1, 20, 22, 0), label(tA, 2, 298, 302, 0), label(tA, 2, 30, 32, 1),
       label(tA, 4, 338, 342, 0)]
atA, resA = run(tA, evA)
check("setup: the five labels are credited to ids 1, 1, 2, 2, 4 (the referee throws nothing labelled)",
      [a.track_id for a in atA] == [1, 1, 2, 2, 4], str([a.track_id for a in atA]))
fa = resA["fragments"]
check("the population is exactly the credited ids {1, 2, 4}; the uncredited referee (3) is not in it",
      sorted(fa) == [1, 2, 4] and 3 not in resA["exposure"], str(sorted(fa)))
check("P: its labelled punch at 100 is explained by its OWN label; 250 and 338 are unexplained",
      close(where(resA, 1), [(100, "right", "own"), (250, "left", "none"), (338, "right", "none")]), str(where(resA, 1)))
check("Q: its burst at 100 is NOT explained by P's label (same moment, other person); its labelled one at 300 is, by its own",
      close(where(resA, 2), [(100, "left", "none"), (300, "left", "own")]), str(where(resA, 2)))
check("S (one credited punch): its burst at 340 is explained by its own label; P's burst at 338 is not (S's label is not P's)",
      close(where(resA, 4), [(340, "left", "own")]), str(where(resA, 4)))
check("the referee's burst at 200 is not counted anywhere (an uncredited fragment is not in the population)",
      not any(190 <= b.frame <= 210 for b in resA["bursts"]) and len(resA["bursts"]) == 6, str(len(resA["bursts"])))
check("per-fragment counts: P 3 bursts / 2 unexplained / 1 own match; Q 2 / 1 / 1; S 1 / 0 / 1",
      (fa[1]["bursts"], fa[1]["unexplained"], fa[1]["labels_matched_own"]) == (3, 2, 1)
      and (fa[2]["bursts"], fa[2]["unexplained"], fa[2]["labels_matched_own"]) == (2, 1, 1)
      and (fa[4]["bursts"], fa[4]["unexplained"], fa[4]["labels_matched_own"]) == (1, 0, 1)
      and (fa[1]["credited_punches"], fa[2]["credited_punches"], fa[4]["credited_punches"]) == (2, 2, 1), str(fa))
check("per-fragment identity: track id, first and last frame of the fragment",
      (fa[1]["track_id"], fa[1]["first_frame"], fa[1]["last_frame"], fa[1]["rows"]) == (1, 0, 399, 400))
check("the label counts: 5 labels all owned, none unattributed, none credited to an excluded fragment; 3 matched, all own",
      resA["labels"] == {"total": 5, "owned": 5, "owned_in_exposure": 5, "owned_outside_exposure": 0, "credited_to_excluded": 0,
                         "unattributed": 0, "unattributed_in_exposure": 0, "matched_own": 3, "matched_unattributed": 0},
      str(resA["labels"]))
check("the matched label's hand is recorded: P's label says right, P's burst is the right hand",
      [b.hand_agrees for b in resA["bursts"] if b.track_id == 1 and near(b.frame, 100)] == [True])
check("each burst carries its fragment's track id, and 'boxer' is that id as text (there is no boxer A / B here)",
      all(b.track_id is not None and b.boxer == str(b.track_id) for b in resA["bursts"]))
check("the exposure of a plain fragment is all of its frames (no hole, no short segment)",
      fa[1]["exposure_frames"] == 400 and abs(fa[1]["exposure_min"] - 400 / FPS / 60) < 1e-12
      and int(resA["exposure"][1].sum()) == 400)

print("min_punches: a 1-punch fragment is excluded at 2 and included at 1; its label then explains nobody")
_, res2 = run(tA, evA, min_punches=2)
f2 = res2["fragments"]
check("min_punches=2 leaves exactly the two 2-punch fragments, and S's burst is gone",
      sorted(f2) == [1, 2] and not any(b.track_id == 4 for b in res2["bursts"]) and len(res2["bursts"]) == 5, str(sorted(f2)))
check("... S's label is credited to a person this test does not follow: counted as credited_to_excluded, owned by nobody",
      res2["labels"]["credited_to_excluded"] == 1 and res2["labels"]["owned"] == 4 and res2["labels"]["unattributed"] == 0
      and res2["labels"]["matched_own"] == 2 and res2["labels"]["matched_unattributed"] == 0, str(res2["labels"]))
check("... and it does NOT explain P's burst at 338 (it is not unattributed): still unexplained",
      close(where(res2, 1), [(100, "right", "own"), (250, "left", "none"), (338, "right", "none")]), str(where(res2, 1)))
check("min_punches=1 (the default) and an explicit 1 agree", [(b.frame, b.track_id, b.match) for b in resA["bursts"]]
      == [(b.frame, b.track_id, b.match) for b in run(tA, evA, min_punches=1)[1]["bursts"]])
check("min_punches=3 leaves nobody: no fragment, no burst, and every label is credited to an excluded fragment",
      run(tA, evA, min_punches=3)[1]["fragments"] == {} and run(tA, evA, min_punches=3)[1]["bursts"] == []
      and run(tA, evA, min_punches=3)[1]["labels"]["credited_to_excluded"] == 5)
for bad in (0, -1, True, 1.5, "2", None):
    raises(f"min_punches={bad!r} is refused", lambda b=bad: completeness.find_bursts_credited(tA, evA, atA, RULE, min_punches=b),
           "min_punches")

print("an unattributed (ambiguous) label explains ONE burst once, across fragments")
P2 = person(1, still(150, 150), [(0, 399)], [(150, 1)])
Q2 = person(2, still(500, 150), [(0, 399)], [(150, 0)])
tB = build([P2, Q2], 400)
evB = [label(tB, 1, 20, 22, 0), label(tB, 2, 30, 32, 1), free_label(148, 152)]
atB, resB = run(tB, evB)
check("setup: the third label is credited to nobody", [a.track_id for a in atB] == [1, 2, None], str([a.track_id for a in atB]))
check("both fragments burst at the same moment: exactly ONE burst is explained, by the unattributed label, and the tie goes "
      "to the lower track id (1)",
      close(where(resB, 1), [(150, "right", "unattributed")]) and close(where(resB, 2), [(150, "left", "none")]),
      str((where(resB, 1), where(resB, 2))))
check("the label is used once: matched_unattributed is 1, not 2; one unattributed label, inside the exposure",
      resB["labels"]["matched_unattributed"] == 1 and resB["labels"]["unattributed"] == 1
      and resB["labels"]["unattributed_in_exposure"] == 1 and resB["labels"]["matched_own"] == 0, str(resB["labels"]))
check("an unattributed label out of every fragment's reach explains nothing",
      all(b.match == "none" for b in run(tB, [evB[0], evB[1], free_label(300, 302)])[1]["bursts"]))
# the unattributed label serves a burst NOT matched by an own label (own first, then unattributed)
P3 = person(1, still(150, 150), [(0, 399)], [(100, 1)])
tC = build([P3], 400)
evC = [label(tC, 1, 98, 102, 1), free_label(98, 102)]
_, resC = run(tC, evC)
check("own labels first: the burst takes its own label and the unattributed label stays unused",
      close(where(resC, 1), [(100, "right", "own")]) and resC["labels"]["matched_unattributed"] == 0, str(resC["labels"]))

print("exposure: a bridged-frame peak is counted and exposed; a short segment adds none")
# a 2-frame hole (101, 102) is bridged inside one id (the peak lands on frame 102, which has no row); frames 404-406 are a
# 3-row segment after a 4-frame hole, under the 5-row smoothing window: skipped.
Pe = person(1, still(150, 150), [(0, 406)], [(100, 1)], drop=[101, 102, 400, 401, 402, 403])
tE = build([Pe], 410)
evE = [label(tE, 1, 20, 22, 0), label(tE, 1, 404, 406, 0)]
atE, resE = run(tE, evE)
fe = resE["fragments"][1]
check("setup: the burst is at frame 102, a frame with no row in the table",
      [b.frame for b in resE["bursts"]] == [102] and 102 not in set(tE.frame.tolist()), str([b.frame for b in resE["bursts"]]))
check("the bridged-frame peak is in the numerator AND the exposure: 399 rows (0-399 less 2) + 3 tail rows, 400 exposure frames",
      fe["rows"] == 401 and fe["bursts"] == 1 and fe["unexplained"] == 1 and fe["exposure_frames"] == 400
      and bool(resE["exposure"][1][102]) and abs(fe["exposure_min"] - 400 / FPS / 60) < 1e-12, str(fe))
check("the 3-row tail is a short segment: skipped and counted, adds no exposure (403 would be 'all rows')",
      fe["segments"] == 2 and fe["short_segments"] == 1 and not resE["exposure"][1][404:407].any()
      and int(resE["exposure"][1].sum()) == 400, str(fe))
check("a label wholly inside the skipped segment is OUTSIDE the exposure, the other inside",
      fe["own_labels_in_exposure"] == 1 and fe["own_labels_outside_exposure"] == 1
      and resE["labels"]["owned_in_exposure"] == 1 and resE["labels"]["owned_outside_exposure"] == 1, str(fe))
check("the burst box is interpolated across the hole: the person is still, so (150, 152, 108) exactly",
      resE["bursts"][0].box == (150.0, 152.0, BOX_H), str(resE["bursts"][0].box))
tE3 = build([person(1, still(150, 150), [(0, 399)], [(100, 1)], drop=[101, 102, 103])], 400)
r3 = run(tE3, [label(tE3, 1, 20, 22, 0)])[1]["fragments"][1]
tE2 = build([person(1, still(150, 150), [(0, 399)], [(100, 1)], drop=[101, 102])], 400)
r2 = run(tE2, [label(tE2, 1, 20, 22, 0)])[1]["fragments"][1]
check("a 2-frame hole stays one segment (the bridge), a 3-frame hole ends it, in the SAME fragment (one person)",
      r2["segments"] == 1 and r3["segments"] == 2 and r2["exposure_frames"] == 400, str((r2, r3)))
real_sf = punchfind.signal_frames
punchfind.signal_frames = lambda track, *a, **k: np.zeros((len(track.frame_ids), 2), dtype=bool)
try:
    run(tE, evE)
    raised = None
except RuntimeError as e:
    raised = str(e)
finally:
    punchfind.signal_frames = real_sf
check("a burst outside the exposure it is counted against is a RuntimeError, not a quiet mismatch",
      raised is not None and "outside the exposure" in raised and "fragment1" in raised, str(raised))
t_bl = dataclasses.replace(tE, conf=np.zeros_like(tE.conf))
resBl = completeness.find_bursts_credited(t_bl, evE, olympic.attribute(evE, t_bl), RULE)
check("with no confident body, a fragment never enters: no credited punch (so no fragment), no crash", resBl["fragments"] == {})

print("negative: inputs at the boundary fail closed")
raises("attributions that are not one per event are refused",
       lambda: completeness.find_bursts_credited(tA, evA, atA[:-1], RULE), "one attribution per event")
raises("... or are for other events", lambda: completeness.find_bursts_credited(tA, evA[::-1], atA, RULE), "one attribution per event")
raises("a negative tolerance is refused", lambda: completeness.find_bursts_credited(tA, evA, atA, RULE, tol_s=-1), "tol_s")
t_no4 = build([P, Q, R], 400)
raises("a punch credited to a track id that is not in the table (attributions from another video) is refused",
       lambda: completeness.find_bursts_credited(t_no4, evA, atA, RULE), "not in this pose table")
empty = completeness.find_bursts_credited(tA, [], [], RULE)
check("no events at all: no fragment, no burst, no label, no crash",
      empty["fragments"] == {} and empty["bursts"] == [] and empty["labels"]["total"] == 0)
raises("box_at outside the fragment's span is refused",
       lambda: completeness.box_at(tA, np.flatnonzero(tA.track_id == 1), 400), "outside")

print("box_at and mss")
tm = build([person(1, lambda f: (100 + 0.5 * f, 150), [(0, 10)], drop=[5, 6])], 11)
rows_m = np.flatnonzero(tm.track_id == 1)
check("box_at: a frame with a row is its own box; a dropped frame is the linear interpolation (centre x 100 + 0.5 f, y 152, height 108)",
      completeness.box_at(tm, rows_m, 3) == (101.5, 152.0, BOX_H) and completeness.box_at(tm, rows_m, 6) == (103.0, 152.0, BOX_H)
      and completeness.box_at(tm, rows_m, 5) == (102.5, 152.0, BOX_H))
check("mss: 62.0 s is 1:02.0, 2.04 s is 0:02.0, 0 is 0:00.0, 125.34 is 2:05.3",
      (completeness.mss(62.0), completeness.mss(2.04), completeness.mss(0), completeness.mss(125.34))
      == ("1:02.0", "0:02.0", "0:00.0", "2:05.3"))
check("mss rounds to a tenth FIRST: 59.96 s is 1:00.0 (never 0:60.0), 119.95 is 2:00.0",
      completeness.mss(59.96) == "1:00.0" and completeness.mss(119.96) == "2:00.0" and completeness.mss(59.94) == "0:59.9")
for bad in (-1.0, float("nan"), float("inf"), "5", None):
    raises(f"mss({bad!r}) is refused", lambda b=bad: completeness.mss(b), "seconds")

# ---------------------------------------------------------------------------------------------------------------------
print("olympic_completeness.py --credited end to end")
N2 = 3400
Pm = person(1, lambda f: (100 + 0.1 * f, 150), [(0, N2 - 1)], [(100, 1), (1500, 0), (3100, 1)], drop=[101, 102])
tM = build([Pm], N2)
evM = [label(tM, 1, 300 + 30 * i, 302 + 30 * i, 0) for i in range(11)]    # 11 labels (the linking needs 10 to see a ring), clear of the bursts
ds = tmp / "ds"
task = ds / "task_kam2_gh000001"
(task / "data").mkdir(parents=True)
clip = task / "data" / "GH000001.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=50", "-frames:v", str(N2),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
pl = {v: k for k, v in olympic.TARGETS.items()}
ph = {v: k for k, v in olympic.HANDS.items()}


def cvat(evs):
    tracks = []
    for e in evs:
        x1, y1, x2, y2 = e.box
        shp = lambda frame, out: {"type": "rectangle", "frame": frame, "outside": out, "points": [x1, y1, x2, y2]}  # noqa: E731
        tracks.append({"label": f"{pl[e.target]} {ph[e.hand]}", "shapes": [shp(e.first, False), shp(e.last + 1, True)]})
    return [{"tracks": tracks}]


(task / "annotations.json").write_text(json.dumps(cvat(evM)), encoding="utf-8")
trk = tmp / "tracks"
trk.mkdir()
tM.save(trk / "GH000001.tracks.npz")
rule_f = tmp / "rule.json"
rule_f.write_text(json.dumps({"chosen": RULE}), encoding="utf-8")
cj, cc = tmp / "res" / "c.json", tmp / "res" / "c.csv"
base = ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited"]


def cli(extra, quiet=True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = olympic_completeness.main(base + extra)
    return rc, buf.getvalue()


rc, out1 = cli(["--out", str(cj), "--sample-out", str(cc)])
js = json.loads(cj.read_text(encoding="utf-8"))
ja = js["all"]
check("exit 0; one credited fragment, three bursts (all unexplained: the 11 labels are all far from them), no one-punch fragment",
      rc == 0 and ja["fragments_n"] == 1 and ja["bursts"] == 3 and ja["unexplained"] == 3 and ja["fragments_one_punch"] == 0 and ja["labels"]["owned"] == 11, str(ja))
check("REAL FACT: the rates are real numbers over real minutes: 3 unexplained over 3400 frames (68 s) of exposure",
      abs(ja["exposure_min"] - N2 / FPS / 60) < 0.001 and abs(ja["unexplained_per_exposure_min"] - 3 / (N2 / FPS / 60)) < 0.01
      and abs(ja["bursts_per_exposure_min"] - 3 / (N2 / FPS / 60)) < 0.01
      and abs(ja["own_labels_in_exposure_per_exposure_min"] - 11 / (N2 / FPS / 60)) < 0.01, str(ja))
check("the whole-video density stays a separate number: 11 labels over 2 x the video's minutes",
      abs(ja["labels_per_video_boxer_min"] - 11 / (2 * N2 / FPS / 60)) < 0.01)
check("the json states the mode, the min_punches and the bias; no linking",
      js["mode"] == "credited_fragments" and js["linking"] is None and js["min_punches"] == 1
      and "UNDERCOUNT" in js["bias"] and "weak evidence" in js["bias"] and "decisive" in js["bias"], js["bias"])
check("the CLI print carries the bias and the one-punch count",
      "UNDERCOUNT" in out1 and "weak evidence" in out1 and "exactly one credited punch" in out1, out1)
check("the per-video row lists the fragment with its track id and its rates",
      js["videos"]["task_kam2_gh000001"]["fragments"][0]["track_id"] == 1
      and js["videos"]["task_kam2_gh000001"]["fragments"][0]["unexplained_per_exposure_min"] > 0)
check("the output does not call the count a bound or a recall", "upper bound" not in json.dumps(js) and "recall" not in json.dumps(js))
rows = list(csv.reader(cc.read_text(encoding="utf-8").splitlines()))
check("the sample csv has the columns that find the person: track id, box centre and height, time_s and m:ss.s",
      rows[0] == ["video", "frame", "time_s", "time_mss", "track_id", "box_cx", "box_cy", "box_h", "hand", "inclusion_prob",
                  "user_label"], str(rows[0]))
by_frame = {int(r[1]): r for r in rows[1:]}
check("setup: all three unexplained bursts are sampled (N=3 <= 40), near 102 (a bridged frame), 1500 and 3100",
      len(rows) == 4 and sorted(abs(f - w) <= 6 for f, w in zip(sorted(by_frame), (102, 1500, 3100))) == [True] * 3, str(sorted(by_frame)))
exact = True
for fr_, r in by_frame.items():
    sec = fr_ / FPS
    tenths = round(sec * 10)
    want = ["GH000001", str(fr_), f"{sec:.2f}", f"{tenths // 600}:{(tenths % 600) / 10:04.1f}", "1",
            f"{100 + 0.1 * fr_:.1f}", "152.0", "108.0", r[8], "1.000000", ""]
    exact &= r == want
    if r != want:
        print("    got ", r, "\n    want", want)
check("every value is exact for the planted burst: time_s = frame/50, m:ss.s, track id 1, box centre (100 + 0.1 f, 152) and height 108",
      exact)
f3100 = [f for f in by_frame if near(f, 3100)][0]
check("a burst after the first minute reads m:ss.s with minutes (about 1:02.0)", by_frame[f3100][3].startswith("1:0"), str(by_frame[f3100]))
check("the burst on a BRIDGED frame (no row) still has a box: interpolated between its neighbours",
      by_frame[[f for f in by_frame if near(f, 102)][0]][5] == f"{100 + 0.1 * [f for f in by_frame if near(f, 102)][0]:.1f}")
check("the hand column names the planted hands (right at ~102, left at ~1500, right at ~3100)",
      [by_frame[f][8] for f in sorted(by_frame)] == ["right", "left", "right"], str(sorted(by_frame.items())))
rc, _ = cli(["--out", str(cj), "--sample-out", str(cc), "--sample-size", "2"])
rows2 = list(csv.reader(cc.read_text(encoding="utf-8").splitlines()))
js2 = json.loads(cj.read_text(encoding="utf-8"))
check("inclusion_prob = n / N: a sample of 2 of the 3 unexplained bursts is 0.666667 in every row, and the json agrees",
      len(rows2) == 3 and all(r[9] == "0.666667" for r in rows2[1:]) and js2["sample"]["n"] == 2
      and js2["sample"]["population_unexplained"] == 3 and abs(js2["sample"]["inclusion_prob"] - 2 / 3) < 1e-9, str(rows2))
cli(["--out", str(cj), "--sample-out", str(cc), "--sample-size", "2"])
check("the same seed reproduces the identical sample", list(csv.reader(cc.read_text(encoding="utf-8").splitlines())) == rows2)

print("olympic_completeness.py --credited: refusals and the default file names")


def refused(argv):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            olympic_completeness.main(argv)
    except SystemExit as e:
        return e.code, err.getvalue()
    return None, err.getvalue()


c_ra, e_ra = refused(base + ["--reacquire", "--out", str(tmp / "no1.json"), "--sample-out", str(tmp / "no1.csv")])
check("--credited together with --reacquire is refused (exit 2) before anything is written",
      c_ra == 2 and "--reacquire" in e_ra and not (tmp / "no1.json").exists() and not (tmp / "no1.csv").exists(), e_ra)
c_mp, e_mp = refused(["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--min-punches", "2",
                      "--out", str(tmp / "no2.json"), "--sample-out", str(tmp / "no2.csv")])
check("--min-punches without --credited is refused rather than silently ignored",
      c_mp == 2 and "--credited" in e_mp and not (tmp / "no2.json").exists(), e_mp)
for bad in ("0", "-3"):
    c_b, e_b = refused(base + ["--min-punches", bad, "--out", str(tmp / "no3.json"), "--sample-out", str(tmp / "no3.csv")])
    check(f"--min-punches {bad} is refused", c_b == 2 and "min-punches" in e_b and not (tmp / "no3.json").exists(), e_b)
c_same, e_same = refused(base + ["--out", str(tmp / "same"), "--sample-out", str(tmp / "same")])
check("the json and the sample may not be the same file", c_same == 2 and "same file" in e_same, e_same)
check("the credited defaults are their own files, different from the linking-based ones",
      olympic_completeness.DEFAULT_OUT_CREDITED != olympic_completeness.DEFAULT_OUT
      and olympic_completeness.DEFAULT_SAMPLE_CREDITED != olympic_completeness.DEFAULT_SAMPLE
      and olympic_completeness.DEFAULT_OUT_CREDITED.name == "olympic_completeness_credited.json"
      and olympic_completeness.DEFAULT_SAMPLE_CREDITED.name == "olympic_completeness_credited_sample.csv")
work = tmp / "cwd"
(work / "results").mkdir(parents=True)
sent = {"results/olympic_completeness.json": "SENTINEL-JSON", "results/olympic_completeness_sample.csv": "SENTINEL-CSV"}
for k_, v_ in sent.items():
    (work / k_).write_text(v_, encoding="utf-8")
here = os.getcwd()
os.chdir(work)
try:
    rc_def, _ = cli([])
finally:
    os.chdir(here)
check("with default output names, --credited writes the credited files and leaves the linking-based ones untouched",
      rc_def == 0 and (work / "results/olympic_completeness_credited.json").is_file()
      and (work / "results/olympic_completeness_credited_sample.csv").is_file()
      and all((work / k_).read_text(encoding="utf-8") == v_ for k_, v_ in sent.items()))
rc_lk = None
os.chdir(work)
try:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_lk = olympic_completeness.main(["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f)])
finally:
    os.chdir(here)
check("... and without --credited the linking run still writes the linking-based names (and not the credited ones again)",
      rc_lk == 0 and (work / "results/olympic_completeness.json").read_text(encoding="utf-8") != "SENTINEL-JSON"
      and (work / "results/olympic_completeness_credited.json").is_file()
      and json.loads((work / "results/olympic_completeness_credited.json").read_text(encoding="utf-8"))["mode"] == "credited_fragments")

print("olympic_completeness.py --credited: a video with no credited punch runs to the end")
(task / "annotations.json").write_text(json.dumps(cvat([free_label(100, 102)])), encoding="utf-8")
cj0, cc0 = tmp / "res" / "c0.json", tmp / "res" / "c0.csv"
rc0, out0 = cli(["--out", str(cj0), "--sample-out", str(cc0)])
j0 = json.loads(cj0.read_text(encoding="utf-8"))["all"]
check("one label credited to nobody: exit 0, no fragment, exact zero counts, the rates absent (null), printed as n/a",
      rc0 == 0 and j0["fragments_n"] == 0 and j0["bursts"] == 0 and j0["unexplained_per_exposure_min"] is None
      and j0["own_labels_in_exposure_per_exposure_min"] is None and "n/a per exposure minute" in out0 and j0["labels"]["unattributed"] == 1,
      str(j0))
check("... and the sample is the header only, with a null inclusion probability",
      cc0.read_text(encoding="utf-8").strip().split(",")[0] == "video" and len(cc0.read_text(encoding="utf-8").strip().splitlines()) == 1
      and json.loads(cj0.read_text(encoding="utf-8"))["sample"]["inclusion_prob"] is None)

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
