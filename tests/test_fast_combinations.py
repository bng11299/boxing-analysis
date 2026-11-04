"""Tests for the two fast-combination arms (docs/gates/fast-combinations.md).

Arm 1: per-hand peaks, a peak a candidate only while its wrist moves AWAY from its
shoulder (punchfind.extension_mask). Arm 2: the pooled rule with a shorter minimum
gap between peaks of different hands (detect(..., cross_gap_s=)). Most cases are
negative: missing joints, a one-handed clip, a retraction with no punch, malformed
rule files.

Run:  python tests/test_fast_combinations.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import classify_punches  # noqa: E402
import find_punches  # noqa: E402
from boxingvi import pose_yolo, punchfind, report  # noqa: E402
from boxingvi.data import N_JOINTS, Track  # noqa: E402

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


def ramp(t: np.ndarray, t0: float, dur: float) -> np.ndarray:
    """0 before t0, 1 after t0 + dur, a raised cosine between: its speed peaks mid-ramp."""
    p = np.clip((t - t0) / dur, 0, 1)
    return (1 - np.cos(np.pi * p)) / 2


def body(n: int, fps: float, conf=None) -> Track:
    """A still body, torso 100 px, each wrist 50 px straight below its shoulder (0.5 body lengths)."""
    xy = np.zeros((n, N_JOINTS, 2))
    xy[:, 5] = (400, 300); xy[:, 6] = (500, 300)
    xy[:, 11] = (400, 400); xy[:, 12] = (500, 400)
    xy[:, 9] = (400, 350); xy[:, 10] = (500, 350)
    return Track(frame_ids=np.arange(n, dtype=np.int64), xy=xy, fps=fps, conf=conf)


def punch(tr: Track, hand: int, t0: float, ext: float = 0.12, hold: float = 0.05, ret: float = 0.2,
          reach: float = 100.0) -> None:
    """The wrist goes straight out `reach` px in `ext` s (away from the shoulder), holds, comes back in `ret` s."""
    t = np.arange(len(tr.frame_ids)) / tr.fps
    u = ramp(t, t0, ext) - ramp(t, t0 + ext + hold, ret)
    tr.xy[:, 9 + hand, 0] += (-1 if hand == 0 else 1) * reach * u


def frames_of(f, h, hand):
    return [int(x) for x, y in zip(f, h) if y == hand]


FPS = 30.0
THR, GAP = 2.0, 0.4

print("extension_mask: out is True, back and still are False")
tr = body(300, FPS)
punch(tr, 1, 3.0)                                 # right: out 3.00-3.12 s, back 3.17-3.37 s
m = punchfind.extension_mask(tr, 0.1)
check("mid-extension (3.06 s) is eligible", bool(m[int(3.06 * FPS), 1]))
check("mid-retraction (3.27 s) is not", not m[int(3.27 * FPS), 1])
check("standing still (1.0 s) is not: no rise is not a rise", not m[30, 1] and not m[30, 0])
check("the other hand, which never moves, is never eligible", not m[:, 0].any())
check("the shape is (frames, 2), bool", m.shape == (300, 2) and m.dtype == bool)
k = max(1, int(round(0.1 * FPS / 2)))
check("a run's first and last k frames are not eligible (their window leaves the run)",
      not m[:k].any() and not m[-k:].any())

print("extension_mask: missing joints are never eligible")
conf = np.ones((300, N_JOINTS))
conf[85:95, 10] = 0.0                              # right wrist lost 2.83-3.17 s, across the punch's extension
trm = body(300, FPS, conf=conf)
punch(trm, 1, 3.0)
mm = punchfind.extension_mask(trm, 0.1)
check("no frame with a missing wrist is eligible", not mm[85:95, 1].any())
check("... nor any frame whose window reaches into the gap", not mm[85 - k:95 + k, 1].any(), str(np.flatnonzero(mm[:, 1])))
c1 = np.ones((300, N_JOINTS)); c1[91, 10] = 0.0    # one frame lost mid-extension
tb = body(300, FPS, conf=c1); punch(tb, 1, 3.0)
check("a one-frame dropout is bridged (BRIDGE_S fills up to 2 frames at 30 fps), so the extension still counts",
      bool(punchfind.extension_mask(tb, 0.1)[91, 1]))
nobody = np.ones((300, N_JOINTS)); nobody[:, 6] = 0.0            # right SHOULDER never seen
trs = body(300, FPS, conf=nobody); punch(trs, 1, 3.0)
raises("a shoulder never seen leaves no body scale: refused, exactly as wrist_signal refuses it",
       lambda: punchfind.extension_mask(trs, 0.1))
raises("... (control: wrist_signal on the same track)", lambda: punchfind.wrist_signal(trs))
nowr = np.ones((300, N_JOINTS)); nowr[:, 10] = 0.1                # right WRIST never trustworthy
trw = body(300, FPS, conf=nowr); punch(trw, 1, 3.0)
check("a hand whose wrist is never trustworthy is never eligible",
      not punchfind.extension_mask(trw, 0.1)[:, 1].any())
check("no confidence at all (V6): every frame is valid, the extension still counts",
      bool(punchfind.extension_mask(tr, 0.1)[int(3.06 * FPS), 1]))
for bad in (0, -0.1, float("nan"), float("inf"), True):
    raises(f"window_s={bad!r} is refused", lambda b=bad: punchfind.extension_mask(tr, b))
short = body(4, FPS)
check("a run shorter than the window gives no eligible frame and no error",
      not punchfind.extension_mask(short, 0.3).any())

print("retraction only: a guard pulled back fast is not a punch")
rt = body(300, FPS)
t_ = np.arange(300) / FPS
rt.xy[:, 9, 0] -= 100 * (ramp(t_, 0.5, 1.5) - ramp(t_, 4.0, 0.12))  # out over 1.5 s (slow), back in 0.12 s (fast)
rsig = punchfind.wrist_signal(rt)
check("control: the slow extension stays under the threshold, the snap back is over it",
      rsig[int(1.25 * FPS), 0] < THR < rsig[int(4.06 * FPS), 0], f"{rsig[int(1.25 * FPS), 0]:.2f} {rsig[int(4.06 * FPS), 0]:.2f}")
pf, ph = punchfind.detect(rt, rsig, THR, GAP, per_hand=True)
check("control: per hand, the retraction is a detection", len(pf) == 1 and abs(pf[0] - 4.06 * FPS) <= 3, str(pf))
af, ah = punchfind.detect(rt, rsig, THR, GAP, per_hand=True, eligible=punchfind.extension_mask(rt, 0.1))
check("arm 1: no detection at all", len(af) == 0, str(af))

print("jab-cross faster than 0.4 s: pooled merges, both arms split")
jc = body(300, FPS)
punch(jc, 0, 3.0, hold=0.0, ret=0.12)              # jab, left, out 3.00-3.12, back 3.12-3.24
punch(jc, 1, 3.2, hold=0.0, ret=0.12)              # cross, right, out 3.20-3.32, back 3.32-3.44
jsig = punchfind.wrist_signal(jc)
pooled = punchfind.detect(jc, jsig, THR, GAP)
check("pooled (the V6 rule's gap): one detection", len(pooled[0]) == 1, str(pooled))
a1 = punchfind.detect_rule(jc, jsig, {"variant": "relative", "threshold": THR, "min_gap_s": GAP,
                                       "finder": {"mode": "per_hand_extension", "extension_window_s": 0.1}})
check("arm 1: two detections, one per hand", sorted(a1[1].tolist()) == [0, 1], str(a1))
check("arm 1: the left detection is on the jab's way OUT (3.00-3.12 s), not its way back",
      all(3.0 * FPS <= f <= 3.12 * FPS for f in frames_of(*a1, 0)), str(a1))
check("arm 1: the right detection is on the cross's way out (3.20-3.32 s)",
      all(3.2 * FPS <= f <= 3.32 * FPS for f in frames_of(*a1, 1)), str(a1))
a2 = punchfind.detect_rule(jc, jsig, {"variant": "relative", "threshold": THR, "min_gap_s": GAP,
                                       "finder": {"mode": "cross_hand_gap", "cross_gap_s": 0.15}})
check("arm 2 (cross-hand gap 0.15 s): two detections, one per hand", sorted(a2[1].tolist()) == [0, 1], str(a2))
a2w = punchfind.detect(jc, jsig, THR, GAP, cross_gap_s=0.35)
check("arm 2 with a cross-hand gap wider than the punches' spacing merges them again", len(a2w[0]) == 1, str(a2w))
fast_back = body(300, FPS)
punch(fast_back, 0, 3.0, ext=0.25, ret=0.1)        # a jab that comes back faster than it went out
fsig = punchfind.wrist_signal(fast_back)
pfb = punchfind.detect(fast_back, fsig, THR, GAP, per_hand=True)
afb = punchfind.detect(fast_back, fsig, THR, GAP, per_hand=True, eligible=punchfind.extension_mask(fast_back, 0.1))
check("control: per hand alone, a jab that returns faster than it went out is detected on its way BACK",
      len(pfb[0]) == 1 and pfb[0][0] > (3.0 + 0.25) * FPS, str(pfb))
check("arm 1: the same jab is detected on its way OUT (the taller retraction is removed before the gap)",
      len(afb[0]) == 1 and 3.0 * FPS <= afb[0][0] <= (3.0 + 0.25) * FPS, str(afb))
rfb = punchfind.detect_rule(fast_back, fsig, {"variant": "relative", "threshold": THR, "min_gap_s": GAP,
                                              "finder": {"mode": "per_hand_extension", "extension_window_s": 0.1}})
check("detect_rule applies the gate for arm 1: the rule file's arm, not plain per hand",
      np.array_equal(rfb[0], afb[0]) and not np.array_equal(rfb[0], pfb[0]), f"{rfb} {pfb}")
rrt = punchfind.detect_rule(rt, rsig, {"variant": "relative", "threshold": THR, "min_gap_s": GAP,
                                       "finder": {"mode": "per_hand_extension", "extension_window_s": 0.1}})
check("detect_rule, arm 1, on the retraction-only clip: nothing", len(rrt[0]) == 0, str(rrt))

print("a one-handed clip: the hidden hand never fires, the seen hand still does")
one = np.ones((300, N_JOINTS)); one[:, 10] = 0.05   # right wrist never trustworthy
oh = body(300, FPS, conf=one)
punch(oh, 0, 2.0); punch(oh, 1, 2.2); punch(oh, 0, 5.0)
osig = punchfind.wrist_signal(oh)
for name, rule in (("arm 1", {"mode": "per_hand_extension", "extension_window_s": 0.1}),
                   ("arm 2", {"mode": "cross_hand_gap", "cross_gap_s": 0.15})):
    f_, h_ = punchfind.detect_rule(oh, osig, {"variant": "relative", "threshold": THR, "min_gap_s": GAP,
                                              "finder": rule})
    check(f"{name}: no right-hand detection", not (h_ == 1).any(), str(list(zip(f_, h_))))
    check(f"{name}: both left punches found", len(frames_of(f_, h_, 0)) == 2, str(list(zip(f_, h_))))

print("detect with cross_gap_s: equal to the pooled rule when the gaps are equal")
rng = np.random.default_rng(7)
same = True
for trial in range(300):
    n = int(rng.integers(5, 400))
    t2 = Track(np.arange(n, dtype=np.int64), np.zeros((n, N_JOINTS, 2)), float(rng.choice([10.0, 24.0, 30.0, 60.0])))
    s2 = rng.gamma(1.5, 1.5, (n, 2))
    if trial % 5 == 0:
        s2 = np.round(s2, 1)          # ties and plateaus
    el = rng.random((n, 2)) > 0.3 if trial % 2 else None
    gap = float(rng.choice([0.1, 0.25, 0.4, 1.0]))
    thr = float(rng.uniform(0.5, 4.0))
    want = punchfind.detect(t2, s2, thr, gap, eligible=el)
    have = punchfind.detect(t2, s2, thr, gap, eligible=el, cross_gap_s=gap)
    if not (np.array_equal(want[0], have[0]) and np.array_equal(want[1], have[1])):
        same = False
        print(f"    trial {trial}: {want} != {have}")
        break
check("300 random signals (ties, plateaus, eligibility masks, 4 frame rates): identical output", same)

print("detect with cross_gap_s: which gap applies to which pair")
N_ = 40
tq = Track(np.arange(N_, dtype=np.int64), np.zeros((N_, N_JOINTS, 2)), 10.0)    # 10 fps: 0.4 s = 4 frames
sq = np.zeros((N_, 2))
sq[10, 0], sq[12, 1] = 5.0, 4.0        # different hands, 2 frames apart
sq[25, 1], sq[27, 1] = 5.0, 4.0        # same hand, 2 frames apart


def gq(**kw):
    f_, h_ = punchfind.detect(tq, sq, 1.0, 0.4, **kw)
    return list(zip(f_.tolist(), h_.tolist()))


check("cross gap 0.1 s (1 frame): the other hand 2 frames later survives; the same hand 2 frames later does not",
      gq(cross_gap_s=0.1) == [(10, 0), (12, 1), (25, 1)], str(gq(cross_gap_s=0.1)))
check("cross gap 0.2 s (2 frames): a pair exactly the gap apart both survive, as find_peaks",
      gq(cross_gap_s=0.2) == [(10, 0), (12, 1), (25, 1)], str(gq(cross_gap_s=0.2)))
check("cross gap 0.3 s (3 frames): the shorter of the two is suppressed",
      gq(cross_gap_s=0.3) == [(10, 0), (25, 1)], str(gq(cross_gap_s=0.3)))
check("control: pooled", gq() == [(10, 0), (25, 1)], str(gq()))
raises("cross_gap_s with per_hand is refused", lambda: gq(cross_gap_s=0.1, per_hand=True))
for bad in (0.5, -0.1, float("nan"), True):
    raises(f"cross_gap_s={bad!r} is refused (outside [0, min_gap_s] or not a number)", lambda b=bad: gq(cross_gap_s=b))

print("every arm's detection passes classify_punches's row check")
ok1 = ok2 = True
for trial in range(150):
    n = int(rng.integers(30, 300))
    tr2 = body(n, 30.0)
    t2_ = np.arange(n) / 30.0
    for h in (0, 1):
        for t0 in rng.uniform(0, n / 30.0, size=int(rng.integers(0, 5))):
            punch(tr2, h, float(t0), ext=float(rng.uniform(0.08, 0.3)), ret=float(rng.uniform(0.08, 0.4)))
    tr2.xy[:, [9, 10]] += rng.normal(0, 2, (n, 2, 2))
    s_ = punchfind.wrist_signal(tr2)
    f1, h1 = punchfind.detect(tr2, s_, THR, GAP, per_hand=True, eligible=punchfind.extension_mask(tr2, 0.1))
    ok1 &= all(classify_punches.own_hand_peak(s_, int(f), int(h), THR) for f, h in zip(f1, h1))
    f2, h2 = punchfind.detect(tr2, s_, THR, GAP, cross_gap_s=0.12)
    ok2 &= all(classify_punches.own_hand_peak(s_, int(f), int(h), THR)
               and classify_punches.pooled_peak(s_, int(f), int(h), THR) for f, h in zip(f2, h2))
check("arm 1: every detection is its own hand's peak (the per-hand check)", ok1)
check("arm 2: every detection is a peak of the faster hand (the pooled check, which is stricter)", ok2)

print("parse_finder: the rule file's finder block is untrusted input")
check("arm 1 parses", punchfind.parse_finder({"mode": "per_hand_extension", "extension_window_s": 0.1}, 0.4)
      == {"mode": "per_hand_extension", "extension_window_s": 0.1})
check("arm 2 parses, an int becomes a float", punchfind.parse_finder({"mode": "cross_hand_gap", "cross_gap_s": 0.15}, 0.4)
      == {"mode": "cross_hand_gap", "cross_gap_s": 0.15})
bad_blocks = {
    "not an object": ["per_hand_extension"],
    "no mode": {"extension_window_s": 0.1},
    "unknown mode": {"mode": "learned", "extension_window_s": 0.1},
    "a mode that is not a string": {"mode": 1, "extension_window_s": 0.1},
    "the other arm's parameter": {"mode": "per_hand_extension", "cross_gap_s": 0.1},
    "an extra key": {"mode": "cross_hand_gap", "cross_gap_s": 0.1, "threshold": 1.0},
    "a missing parameter": {"mode": "cross_hand_gap"},
    "a bool": {"mode": "per_hand_extension", "extension_window_s": True},
    "a string": {"mode": "per_hand_extension", "extension_window_s": "0.1"},
    "NaN": {"mode": "per_hand_extension", "extension_window_s": float("nan")},
    "infinity": {"mode": "cross_hand_gap", "cross_gap_s": float("inf")},
    "zero": {"mode": "per_hand_extension", "extension_window_s": 0},
    "negative": {"mode": "cross_hand_gap", "cross_gap_s": -0.1},
    "a window over 1 s": {"mode": "per_hand_extension", "extension_window_s": 1.5},
    "a cross gap over the minimum gap": {"mode": "cross_hand_gap", "cross_gap_s": 0.5},
}
for name, blk in bad_blocks.items():
    raises(f"refused: {name}", lambda b=blk: punchfind.parse_finder(b, 0.4))

print("load_rule and find_punches: the pooled rule is unchanged, an arm is carried through")
tmp = Path(tempfile.mkdtemp())
v6 = {"chosen": {"variant": "relative", "threshold": THR, "min_gap_s": GAP, "test": {"recall": 0.7, "count_ratio": 1.5}},
      "video": "V6"}
(tmp / "pooled.json").write_text(json.dumps(v6), encoding="utf-8")
check("a rule with no finder block loads to exactly the three keys it always had",
      find_punches.load_rule(tmp / "pooled.json") == {"variant": "relative", "threshold": THR, "min_gap_s": GAP})
arm1 = json.loads(json.dumps(v6)); arm1["chosen"]["finder"] = {"mode": "per_hand_extension", "extension_window_s": 0.1}
arm2 = json.loads(json.dumps(v6)); arm2["chosen"]["finder"] = {"mode": "cross_hand_gap", "cross_gap_s": 0.15}
(tmp / "arm1.json").write_text(json.dumps(arm1), encoding="utf-8")
(tmp / "arm2.json").write_text(json.dumps(arm2), encoding="utf-8")
check("an arm's rule carries its finder", find_punches.load_rule(tmp / "arm1.json")["finder"]["mode"] == "per_hand_extension")
badrule = json.loads(json.dumps(v6)); badrule["chosen"]["finder"] = {"mode": "cross_hand_gap", "cross_gap_s": 2.0}
(tmp / "bad.json").write_text(json.dumps(badrule), encoding="utf-8")
try:
    find_punches.load_rule(tmp / "bad.json")
    check("a bad finder block is refused, naming the file", False, "did not raise")
except ValueError as e:
    check("a bad finder block is refused, naming the file", "bad.json" in str(e), str(e))

n = 300
src = body(n, FPS)
punch(src, 0, 3.0, hold=0.0, ret=0.12); punch(src, 1, 3.2, hold=0.0, ret=0.12); punch(src, 0, 6.0)
cf = np.ones((n, N_JOINTS), np.float32)
pose = pose_yolo.PoseSequence(src.xy.astype(np.float32), cf, np.ones(n, bool), FPS, 1080, 1920, "IMG_8888.mov", "test")
pose.save(tmp / "IMG_8888.pose.npz")
outs = {}
for name in ("pooled", "arm1", "arm2"):
    rc = find_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--rule", str(tmp / f"{name}.json"),
                            "--out-dir", str(tmp / name)])
    rows = list(csv.DictReader((tmp / name / "IMG_8888.punches.csv").open(encoding="utf-8")))
    side = json.loads((tmp / name / "IMG_8888.punches.json").read_text(encoding="utf-8"))
    outs[name] = (rc, rows, side)
check("find_punches: the pooled rule finds 2 (the jab-cross merged), each arm 3",
      [len(outs[k][1]) for k in ("pooled", "arm1", "arm2")] == [2, 3, 3], str([len(outs[k][1]) for k in outs]))
check("the sidecar records per_hand: False pooled, True arm 1, False arm 2 (pooled peaks)",
      [outs[k][2]["per_hand"] for k in ("pooled", "arm1", "arm2")] == [False, True, False])
check("the sidecar records the arm's finder in its rule", outs["arm2"][2]["rule"].get("finder", {}).get("cross_gap_s") == 0.15)
check("classify_punches reads arm 1's sidecar as per hand, arm 2's as pooled",
      classify_punches.finder_per_hand(tmp / "arm1" / "IMG_8888.punches.csv", "IMG_8888",
                                       find_punches.load_rule(tmp / "arm1.json")) is True
      and classify_punches.finder_per_hand(tmp / "arm2" / "IMG_8888.punches.csv", "IMG_8888",
                                           find_punches.load_rule(tmp / "arm2.json")) is False)
check("... and refuses to lend an arm's sidecar to the pooled rule (rule mismatch: unknown)",
      classify_punches.finder_per_hand(tmp / "arm2" / "IMG_8888.punches.csv", "IMG_8888",
                                       find_punches.load_rule(tmp / "pooled.json")) is None)
raises("--per-hand with a rule that names its own finder is refused",
       lambda: find_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--rule", str(tmp / "arm2.json"),
                                  "--out-dir", str(tmp / "x"), "--per-hand"]))

print("report: the pooled rule's held-out score is never lent to an arm")
check("control: the pooled rule's own score applies to the pooled rule",
      report.finder_reliability(tmp / "pooled.json", find_punches.load_rule(tmp / "pooled.json"))["recall"] == 0.7)
raises("an arm with the SAME threshold and gap is another finder: refused",
       lambda: report.finder_reliability(tmp / "pooled.json", find_punches.load_rule(tmp / "arm1.json")))
raises("... and the other way round", lambda: report.finder_reliability(tmp / "arm2.json",
                                                                          find_punches.load_rule(tmp / "pooled.json")))
raises("a malformed finder in a sidecar's rule is refused, not ignored",
       lambda: report.finder_reliability(tmp / "pooled.json", {"variant": "relative", "threshold": THR,
                                                                "min_gap_s": GAP, "finder": {"mode": "x"}}))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
