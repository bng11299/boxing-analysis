"""Tests for the per-hand wrist-speed signal and dropout bridging (punchfind).

The contract that matters most: on a track WITHOUT joint confidence (BoxingVI's
V6) the new signal is bit-for-bit the old one, so every V6 number stands. A
frozen copy of the old implementation is kept here as the reference.

Run:  python tests/test_hands.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from boxingvi import pose_yolo, punchfind  # noqa: E402
from boxingvi.data import N_JOINTS, Span, Track  # noqa: E402

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


def old_wrist_signal(track, variant="relative", smooth_s=0.1):
    """punchfind.wrist_signal as it was at fcdc42d, before per-hand validity."""
    L_S, R_S, L_W, R_W, L_H, R_H = 5, 6, 9, 10, 11, 12
    sh = (track.xy[:, L_S] + track.xy[:, R_S]) / 2
    hp = (track.xy[:, L_H] + track.xy[:, R_H]) / 2
    scale = float(np.median(np.linalg.norm(sh - hp, axis=1)))
    rel = np.stack([track.xy[:, L_W] - track.xy[:, L_S], track.xy[:, R_W] - track.xy[:, R_S]], axis=1) / scale
    w = max(1, int(round(smooth_s * track.fps)))
    w += (w + 1) % 2
    out = np.zeros((len(rel), 2))
    for lo, hi in track.segments():
        if hi - lo < 2:
            continue
        seg = punchfind._smooth(rel[lo:hi], w)
        if variant == "relative":
            out[lo:hi] = np.linalg.norm(np.gradient(seg, axis=0) * track.fps, axis=-1)
        else:
            out[lo:hi] = np.clip(np.gradient(np.linalg.norm(seg, axis=-1), axis=0) * track.fps, 0, None)
    return out


def body(n, fps, ids=None, punches=(), hand=1, dur_s=0.25, amp=1.0, conf=None):
    """A still body (torso 100px) whose wrist moves out and back for each punch."""
    t = np.arange(n) / fps
    xy = np.zeros((n, N_JOINTS, 2))
    xy[:, 5] = (400, 300); xy[:, 6] = (500, 300)
    xy[:, 11] = (400, 400); xy[:, 12] = (500, 400)
    xy[:, 9] = (400, 350); xy[:, 10] = (500, 350)
    for p in punches:
        phase = np.clip((t - p) / dur_s, 0, 1)
        xy[:, 9 + hand, 0] += amp * 100 * np.sin(np.pi * phase)
    f = np.arange(n) if ids is None else np.asarray(ids)
    return Track(frame_ids=f.astype(np.int64), xy=xy, fps=fps, conf=conf)


rng = np.random.default_rng(0)

print("no confidence: bit-for-bit the old signal (V6's numbers must not move)")
cases = [body(240, 24.0, punches=[2.0, 5.0]), body(600, 60.0, punches=[1.0, 4.0], hand=0),
         body(40, 24.0, ids=np.r_[np.arange(20), np.arange(100, 120)], punches=[0.3])]
for k in range(4):  # random walks with random gaps, at both frame rates
    n = int(rng.integers(50, 400)); fps = [23.976, 59.946][k % 2]
    ids = np.cumsum(rng.choice([1, 1, 1, 1, 2, 3, 50], size=n))
    tr = body(n, fps, ids=ids)
    tr.xy[:, [9, 10]] += rng.normal(0, 8, (n, 2, 2)).cumsum(axis=0)
    cases.append(tr)
same = all(np.array_equal(punchfind.wrist_signal(c, v), old_wrist_signal(c, v))
           for c in cases for v in punchfind.VARIANTS)
check("no-confidence tracks: new signal equals the old, bitwise, both variants", same)
full = [Track(c.frame_ids, c.xy, c.fps, conf=np.ones((len(c.frame_ids), N_JOINTS))) for c in cases]
check("all-confident tracks: also equal to the old, bitwise",
      all(np.array_equal(punchfind.wrist_signal(f, v), old_wrist_signal(f, v))
          for f in full for v in punchfind.VARIANTS))

print("per-hand validity")
n, fps = 600, 60.0
conf = np.ones((n, N_JOINTS)); conf[:, 10] = 0.05   # the right (far) wrist hidden all clip
hidden = body(n, fps, punches=[2.0, 6.0], hand=0, conf=conf)
sig = punchfind.wrist_signal(hidden)
check("a hand whose wrist is never trustworthy has no signal", sig[:, 1].max() == 0)
det, hands = punchfind.detect(hidden, sig, 1.0, 0.4)
gt = [Span(int(s * fps), int((s + 0.25) * fps), 0) for s in (2.0, 6.0)]
check("... and the visible hand's punches are still found (the old whole-frame rule found none)",
      len(punchfind.match(det, gt, 0)) == 2 and set(hands.tolist()) == {0}, f"{det} {hands}")
check("control: the old whole-frame rule leaves no usable frame on this clip",
      not punchfind.hand_valid(hidden).all(axis=1).any())
cov = punchfind.signal_coverage(hidden)
check("coverage reports the left hand whole and the right hand absent",
      cov["left"] == 1.0 and cov["right"] == 0.0, str(cov))
few = np.zeros((n, N_JOINTS)); few[:, [5, 6, 9, 10, 11, 12]] = 0.9   # wrists and shoulders solid, 6 joints in all
check("a hand does not count in a frame with fewer than 8 solid joints (no body)",
      not punchfind.hand_valid(body(n, fps, conf=few)).any())

print("bridging short holes in a hand")
base = punchfind.wrist_signal(body(n, fps, punches=[3.0], hand=1)).max()
c2 = np.ones((n, N_JOINTS)); c2[188:190, 10] = 0.05   # a 2-frame wrist dropout mid-punch (3.0-3.25 s)
holed = body(n, fps, punches=[3.0], hand=1, conf=c2)
with_bridge = punchfind.wrist_signal(holed)
no_bridge = punchfind.wrist_signal(holed, bridge_s=0.0)
check("control: without bridging the dropout zeroes the hand at those frames",
      no_bridge[188:190, 1].max() == 0)
check("with bridging the 2-frame dropout is filled: speed there, and the peak within 10% of no dropout",
      with_bridge[188:190, 1].min() > 0 and abs(with_bridge[:, 1].max() / base - 1) < 0.10,
      f"{with_bridge[:, 1].max():.3f} vs {base:.3f}")
c4 = np.ones((n, N_JOINTS)); c4[186:190, 10] = 0.05   # 4 frames > round(0.05 * 60) = 3
check("a 4-frame dropout at 60 fps is NOT bridged",
      punchfind.wrist_signal(body(n, fps, punches=[3.0], conf=c4))[186:190, 1].max() == 0)
ce = np.ones((n, N_JOINTS)); ce[:2, 10] = 0.05
check("a hole at the start of a run (nothing on one side) is not filled",
      punchfind.wrist_signal(body(n, fps, conf=ce))[:2, 1].max() == 0)
ok, filled = punchfind._fill_holes(np.array([1, 0, 0, 1, 0, 0, 0, 1, 0], bool), 2)
check("_fill_holes: interior 2-hole filled, 3-hole and trailing hole kept",
      ok.tolist() == [1, 1, 1, 1, 0, 0, 0, 1, 0] and filled.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0])

print("bridge(): short frame-id gaps")
ids = np.r_[np.arange(0, 100), np.arange(102, 200), np.arange(206, 300)]  # 2 missing, then 5 missing
cg = np.full((len(ids), N_JOINTS), 0.9); cg[99, 9] = 0.2; cg[100, 10] = 0.15   # low on the left (id 99) for joint 9, on the right (id 102) for 10
g = body(len(ids), 60.0, ids=ids, conf=cg)
g.xy[:, 9, 0] = ids  # a wrist moving one pixel per frame, so interpolation is checkable
b = punchfind.bridge(g, 3)
check("a 2-frame gap is filled, a 5-frame gap is not",
      {100, 101}.issubset(set(b.frame_ids.tolist())) and not {201, 202}.intersection(b.frame_ids.tolist()))
check("inserted frames are linearly interpolated",
      np.allclose(b.xy[np.isin(b.frame_ids, [100, 101]), 9, 0], [100, 101]))
check("an inserted frame takes the lower confidence of its neighbours, from either side",
      np.isclose(b.conf[b.frame_ids == 100][0, 9], 0.2) and np.isclose(b.conf[b.frame_ids == 100][0, 10], 0.15))
check("original frames are unchanged",
      np.array_equal(b.xy[np.isin(b.frame_ids, ids)], g.xy) and np.array_equal(b.conf[np.isin(b.frame_ids, ids)], g.conf))
v6like = body(30, 23.976, ids=np.r_[np.arange(10), np.arange(14, 34)])  # 4 missing at 24 fps
check("at 24 fps the bridge is 1 frame, so a V6-style 3-plus-frame gap stays a gap",
      punchfind.bridge(v6like, int(round(punchfind.BRIDGE_S * 23.976))) is v6like)
check("max_gap 0 returns the track untouched", punchfind.bridge(g, 0) is g)
raises("a negative max_gap is refused", lambda: punchfind.bridge(g, -1))
jump = body(40, 60.0, ids=np.r_[np.arange(20), np.arange(30, 50)])
jump.xy[20:, 10, 0] += 300
check("a long gap is still never differenced across (no phantom punch)",
      punchfind.wrist_signal(punchfind.bridge(jump, 3)).max() < 1e-9)

print("body scale with confidence")
# The unsure frames are the MAJORITY: a median shrugs off a minority whether or not
# confidence is used, so only this version can tell the two apart.
cs = np.ones((n, N_JOINTS)); cs[:400, [11, 12]] = 0.05
garb = body(n, fps, conf=cs); garb.xy[:400, [11, 12]] = (450, 2000)   # hips flung far away, but unsure
check("body scale ignores frames with unsure hips (torso stays 100 px)",
      abs(punchfind.body_scale(garb) - 100) < 1e-9, str(punchfind.body_scale(garb)))
raises("no frame with confident shoulders and hips -> refused",
       lambda: punchfind.body_scale(body(n, fps, conf=np.zeros((n, N_JOINTS)))))

print("constants agree with the pose stage")
check("MIN_CONF and BODY_MIN_JOINTS equal pose_yolo's usable-frame rule",
      punchfind.MIN_CONF == pose_yolo.USABLE_MIN_CONF and punchfind.BODY_MIN_JOINTS == pose_yolo.USABLE_MIN_JOINTS)

print("find_punches CLI")
import csv  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

sys.path.insert(0, str(ROOT))
import find_punches  # noqa: E402

tmp = Path(tempfile.mkdtemp())
n, fps = 1200, 59.946
tr = body(n, fps, punches=[3.0, 9.0], hand=0)
tr2 = body(n, fps, punches=[14.0], hand=1)
xy = tr.xy.copy(); xy[:, 10] = tr2.xy[:, 10]                  # left punches at 3 and 9 s, right at 14 s
cf = np.ones((n, N_JOINTS), np.float32); cf[:, 10] = 0.9; cf[700:702, 9] = 0.05   # a left-wrist flicker
pose = pose_yolo.PoseSequence(xy.astype(np.float32), cf, np.ones(n, bool), fps, 1080, 1920, "IMG_7777.mov", "test")
pose.save(tmp / "IMG_7777.pose.npz")
rule = tmp / "rule.json"
rule.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": 1.0, "min_gap_s": 0.4}}), encoding="utf-8")
rc = find_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o")])
rows = list(csv.DictReader((tmp / "o" / "IMG_7777.punches.csv").open(encoding="utf-8")))
check("end to end: one CSV row per planted punch, at the right time and hand",
      rc == 0 and [(r["hand"], round(float(r["time_s"]))) for r in rows] == [("left", 3), ("left", 9), ("right", 14)],
      str(rows))
lab = tmp / "IMG_7777.labels.json"
spans = [{"start": int(s * fps), "end": int((s + 0.25) * fps), "label": l} for s, l in ((3.0, "jab"), (9.0, "unclear"), (14.0, "cross"))]
lab.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": spans}), encoding="utf-8")
find_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o"),
                   "--labels", str(lab)])
sc = json.loads((tmp / "o" / "IMG_7777.score.json").read_text(encoding="utf-8"))
check("scored against labels: all 3 planted punches found, nothing else (unclear counts as a punch)",
      sc["score"]["tp"] == 3 and sc["score"]["recall"] == 1.0 and sc["score"]["precision"] == 1.0, str(sc["score"]))
check("a 20 s clip gets a chance floor", "shift_floor" in sc)
rows = list(csv.DictReader((tmp / "o" / "IMG_7777.punches.csv").open(encoding="utf-8")))
check("the interpolated-frame flag agrees with the confidence at each peak",
      all((r["peak_on_interpolated_frame"] == "yes") == (cf[int(r["frame"]), 9 if r["hand"] == "left" else 10] < 0.3)
          for r in rows))


def labels_with(spans_, n_=n):
    lab.write_text(json.dumps({"n_frames": n_, "fps": fps, "spans": spans_}), encoding="utf-8")
    return lambda: find_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--rule", str(rule),
                                      "--out-dir", str(tmp / "o"), "--labels", str(lab)])


raises("labels: an unknown label is refused (not a KeyError)", labels_with([{"start": 1, "end": 2, "label": "haymaker"}]))
raises("labels: a fractional frame is refused, not truncated", labels_with([{"start": 1.7, "end": 9, "label": "jab"}]))
raises("labels: a boolean frame is refused", labels_with([{"start": True, "end": 9, "label": "jab"}]))
raises("labels: end before start is refused", labels_with([{"start": 9, "end": 1, "label": "jab"}]))
raises("labels: a span past the clip is refused", labels_with([{"start": 1195, "end": 1205, "label": "jab"}]))
raises("labels: a file with no punches is refused (nothing to score against)", labels_with([]))
short = pose_yolo.PoseSequence(xy[:600].astype(np.float32), cf[:600], np.ones(600, bool), fps, 1080, 1920, "s", "t")
short.save(tmp / "IMG_6666.pose.npz")
lab.write_text(json.dumps({"n_frames": 600, "fps": fps, "spans": spans[:1]}), encoding="utf-8")
find_punches.main(["--pose", str(tmp / "IMG_6666.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o"),
                   "--labels", str(lab)])
check("a 10 s clip gets no chance floor (too few distinct shifts to mean anything)",
      "shift_floor" not in json.loads((tmp / "o" / "IMG_6666.score.json").read_text(encoding="utf-8")))

print("coverage is exactly where the signal is")
walk = body(900, 59.946)
walk.xy[:, [9, 10]] += np.random.default_rng(3).normal(0, 6, (900, 2, 2)).cumsum(axis=0)  # never still
cw = np.ones((900, N_JOINTS))
for a_, b_ in ((100, 102), (300, 304), (500, 501), (700, 760)):
    cw[a_:b_, 9] = 0.05
cw[400:403, 10] = 0.05
walk = Track(walk.frame_ids, walk.xy, walk.fps, conf=cw)
sw, cvw = punchfind.wrist_signal(walk), punchfind.signal_coverage(walk)
check("signal_coverage equals the share of frames where wrist_signal is non-zero, per hand",
      abs(cvw["left"] - (sw[:, 0] > 0).mean()) < 1e-12 and abs(cvw["right"] - (sw[:, 1] > 0).mean()) < 1e-12,
      f"{cvw} vs {(sw > 0).mean(axis=0)}")
lab.write_text(json.dumps({"n_frames": n + 5, "fps": fps, "spans": spans}), encoding="utf-8")
raises("labels for a different video (frame count) are refused",
       lambda: find_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--rule", str(rule),
                                  "--out-dir", str(tmp / "o"), "--labels", str(lab)]))
raises("no tuned rule file -> refused with what to run",
       lambda: find_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--rule", str(tmp / "none.json")]))
rule.write_text(json.dumps({"chosen": {"variant": "sideways", "threshold": 1.0, "min_gap_s": 0.4}}), encoding="utf-8")
raises("a rule file naming an unknown variant -> refused",
       lambda: find_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--rule", str(rule)]))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
