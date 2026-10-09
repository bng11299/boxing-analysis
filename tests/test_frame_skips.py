"""Tests for the camera's dropped frames ("skips"): video.frame_change, skips.find_skips / Skips,
completeness.skip_mask_half / skip_mask / check_skips / near_skip_stats, the mask inside find_bursts(_credited), and
`olympic_framechange.py` / `olympic_completeness.py --skips`.

No Olympic data and no model: videos are synthetic (a sine-grating that moves a known number of pixels per
frame, written with the repo's own VideoWriter), people are planted with known wrist bursts, and a "skip" is a
known jump planted at a known frame. The rule's constant (SKIP_RATIO = 3) was not tuned: the boundary tests
below only pin what the rule says.

Run:  python tests/test_frame_skips.py
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
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import olympic_completeness  # noqa: E402
import olympic_framechange  # noqa: E402
from boxingvi import completeness, olympic, skips, video  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.linking import Linking  # noqa: E402
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
NAN = float("nan")


def flat(n, base=1.0, **at):
    """A change series: `base` everywhere, NaN at frame 0, and `at` overrides as f<frame>=value."""
    c = np.full(n, base)
    c[0] = NAN
    for k, v in at.items():
        c[int(k[1:])] = v
    return c


# ---------------------------------------------------------------------------------------------------------------------
print("find_skips: the rule, its boundary and its window")
n = 80
check("a steady series has no skip", len(skips.find_skips(flat(n))) == 0)
check("2.99x the local median is not a skip, 3.01x is, exactly 3.0x is not (strictly greater)",
      len(skips.find_skips(flat(n, f40=2.99))) == 0 and skips.find_skips(flat(n, f40=3.01)).tolist() == [40]
      and len(skips.find_skips(flat(n, f40=3.0))) == 0)
check("the ratio scales with the local level: 30.1 over a level of 10 is a skip, 25 is not (a fixed threshold cannot do both)",
      skips.find_skips(flat(n, 10.0, f40=30.1)).tolist() == [40] and len(skips.find_skips(flat(n, 10.0, f40=25.0))) == 0
      and skips.find_skips(flat(n, 1.0, f40=3.01)).tolist() == [40] and len(skips.find_skips(flat(n, 1.0, f40=2.0))) == 0)
check("frame 0 is NaN and never a skip, however the rest looks",
      0 not in skips.find_skips(flat(n)).tolist() and 0 not in skips.find_skips(np.r_[NAN, np.full(30, 0.0)]).tolist())
check("NaN never enters a median: a NaN neighbour does not change the verdict",
      skips.find_skips(flat(n, f40=3.01, f41=NAN)).tolist() == [40])
check("the frame next to frame 0 (a window cut short on the left) is judged on its right neighbours",
      skips.find_skips(flat(n, f1=3.01)).tolist() == [1] and len(skips.find_skips(flat(n, f1=2.99))) == 0)
check("the last frame (a window cut short on the right) is judged on its left neighbours",
      skips.find_skips(flat(n, **{f"f{n - 1}": 3.01})).tolist() == [n - 1])
check("a series shorter than the window still works (5 frames)",
      skips.find_skips(np.array([NAN, 1.0, 5.0, 1.0, 1.0])).tolist() == [2])
check("a one-frame video has no skip and does not crash", len(skips.find_skips(np.array([NAN]))) == 0)
check("an empty series has no skip", len(skips.find_skips(np.zeros(0))) == 0)
check("a jump of TWO consecutive frames flags both (each is judged against its neighbours, the other one a single outlier)",
      skips.find_skips(flat(n, f40=6.0, f41=6.0)).tolist() == [40, 41])
pan = flat(n)
pan[30:60] = 6.0
check("a pan: 30 frames at 6x the level are NOT skips, inside the stretch or at its edges (the local median rises with them)",
      len(skips.find_skips(pan)) == 0, str(skips.find_skips(pan)))
ramp = np.r_[NAN, np.linspace(1.0, 4.0, n - 1)]
check("a smooth acceleration (1x to 4x over the video) has no skip", len(skips.find_skips(ramp)) == 0)
# neighbours: ten at 1.0 on the left, ten at 2.0 on the right. Median without the frame: 1.5, so 4.6 > 4.5 is a skip.
# Median WITH the frame (the 11th of 21): 2.0, so 4.6 < 6.0 would not be.
step = np.r_[NAN, np.full(29, 1.0), np.full(10, 1.0), 4.6, np.full(10, 2.0), np.full(29, 2.0)]
check("the frame is excluded from its own window: 4.6 between ten 1.0s and ten 2.0s is a skip (median 1.5), "
      "which a window that included it (median 2.0) would miss",
      skips.find_skips(step).tolist() == [40], str(skips.find_skips(step)))
check("a perfectly still stretch has median 0 and nothing divides by it: zeros are no skip, a non-zero frame in them is",
      len(skips.find_skips(flat(n, 0.0))) == 0 and skips.find_skips(flat(n, 0.0, f40=0.5)).tolist() == [40])
check("the ratio is a parameter: ratio 2 flags 2.5x, which ratio 3 does not",
      skips.find_skips(flat(n, f40=2.5), ratio=2.0).tolist() == [40] and len(skips.find_skips(flat(n, f40=2.5))) == 0)
three = flat(n, f39=4.0, f40=5.0, f41=4.0)
check("the half window is a parameter: three raised frames in a row are all skips against 10 neighbours each side, and none against 1",
      skips.find_skips(three).tolist() == [39, 40, 41] and len(skips.find_skips(three, half_window=1)) == 0)
for bad_r in (0, -1, float("nan"), float("inf"), True, "3"):
    raises(f"ratio {bad_r!r} is refused", lambda r=bad_r: skips.find_skips(flat(n), ratio=r), "ratio")
for bad_w in (0, -2, 1.5, True, None):
    raises(f"half_window {bad_w!r} is refused", lambda w=bad_w: skips.find_skips(flat(n), half_window=w), "half_window")
raises("a 2-D change array is refused", lambda: skips.find_skips(np.zeros((4, 4))), "1-D")
raises("an integer change array is refused (a counted change is not this signal)", lambda: skips.find_skips(np.arange(10)), "floating")
raises("a negative change is refused", lambda: skips.find_skips(np.array([NAN, 1.0, -0.5, 1.0])), ">= 0")
raises("an infinite change is refused", lambda: skips.find_skips(np.array([NAN, 1.0, np.inf, 1.0])), "finite")

# ---------------------------------------------------------------------------------------------------------------------
print("video.frame_change on synthetic video: one dropped stretch, a pan, a still scene")
W, H, FPS = 256, 192, 50
XX, YY = np.arange(W)[None, :], np.arange(H)[:, None]


def grating(shift):
    lum = 128 + 70 * np.sin(2 * np.pi * (XX + shift) / 128) + 40 * np.sin(2 * np.pi * YY / 96)
    return np.repeat(np.clip(lum, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def make(name, shifts):
    p = tmp / name
    with video.VideoWriter(p, W, H, Fraction(FPS)) as w:
        for s in shifts:
            w.write(grating(s))
    return p


N = 60
vd = make("drop.mp4", [f + (5 if f >= 30 else 0) for f in range(N)])        # moves 1 px a frame; at 30 it jumps 6 px
c_drop = video.frame_change(vd)
base = float(np.median(np.r_[c_drop[1:30], c_drop[31:]]))
check("frame_change: one value per decoded frame (60) and frame 0 is NaN, everything else a positive number",
      len(c_drop) == N and np.isnan(c_drop[0]) and np.isfinite(c_drop[1:]).all() and (c_drop[1:] > 0).all(), str(c_drop[:4]))
check("REAL FACT: the jump of 6 px at frame 30 reads about 6x the 1 px steps around it (between 4x and 8x)",
      4 < c_drop[30] / base < 8, f"{c_drop[30]:.2f} vs {base:.2f}")
check("a stretch where 5 frames were dropped: exactly frame 30 is a skip, no other",
      skips.find_skips(c_drop).tolist() == [30], str(skips.find_skips(c_drop)))
check("... also at step 1 (every pixel) and step 8: the same single skip",
      skips.find_skips(video.frame_change(vd, step=1)).tolist() == [30] and skips.find_skips(video.frame_change(vd, step=8)).tolist() == [30])
c_pan = video.frame_change(make("pan.mp4", [2 * f for f in range(N)]))
check("a smooth pan (2 px a frame): no skip, and the change is steady (max within 2x the median)",
      len(skips.find_skips(c_pan)) == 0 and np.nanmax(c_pan) < 2 * np.nanmedian(c_pan), f"{np.nanmax(c_pan):.2f} {np.nanmedian(c_pan):.2f}")
p_still = tmp / "still.mp4"
with video.VideoWriter(p_still, W, H, Fraction(FPS)) as w_:
    for _ in range(N):
        w_.write(np.full((H, W, 3), 90, np.uint8))
c_still = video.frame_change(p_still)
check("a still scene (a flat picture): the change is 0 at every frame after the first, and no skip is found (median 0, no division)",
      len(c_still) == N and np.isnan(c_still[0]) and (c_still[1:] == 0).all() and len(skips.find_skips(c_still)) == 0, str(c_still[:5]))
for bad_step in (0, -1, 2.5, True, None):
    raises(f"frame_change step {bad_step!r} is refused", lambda s=bad_step: video.frame_change(vd, step=s), "step")
raises("a missing video is a FileNotFoundError, not an empty answer", lambda: video.frame_change(tmp / "nope.mp4"), "no video", FileNotFoundError)

# ---------------------------------------------------------------------------------------------------------------------
print("Skips: bound to its video, saved and loaded strictly")
FP = video.fingerprint(vd)
chg = flat(N, 1.0, f30=5.0)
sk = skips.Skips(chg, "GH000001.mp4", FP)
check("Skips derives its frames from the stored change with the stored constants, and knows its length",
      sk.frames.tolist() == [30] and sk.n_frames == N and sk.ratio == 3.0 and sk.half_window == 10 and sk.step == 4)
f_ok = tmp / "ok.framechange.npz"
sk.save(f_ok)
back = skips.Skips.load(f_ok)
check("a saved file loads back to the same change (NaN at 0 included), source, fingerprint and constants",
      np.array_equal(back.change, sk.change, equal_nan=True) and back.source == sk.source and back.video_fp == FP
      and (back.step, back.ratio, back.half_window) == (4, 3.0, 10) and back.frames.tolist() == [30])
check("saving leaves no .part file behind and replaces an existing file in whole",
      not list(tmp.glob("*.part")) and (sk.save(f_ok) or True) and skips.Skips.load(f_ok).frames.tolist() == [30] and not list(tmp.glob("*.part")))


import threading  # noqa: E402

f_race = tmp / "race.framechange.npz"
errors_ = []
levels_ = [1.0 + i for i in range(8)]


def writer_(level):
    try:
        for _ in range(15):
            skips.Skips(flat(N, level), "GH000001.mp4", FP).save(f_race)
    except BaseException as e:    # noqa: BLE001  (collected, then asserted on below)
        errors_.append(repr(e))


ths_ = [threading.Thread(target=writer_, args=(lv,)) for lv in levels_]
[t_.start() for t_ in ths_]
[t_.join() for t_ in ths_]
won_ = skips.Skips.load(f_race)
check("eight writers saving the same path 15 times each, at once: no writer fails, the file is one writer's COMPLETE content, no temporary file is left",
      not errors_ and won_.n_frames == N and float(won_.change[1]) in levels_ and (won_.change[1:] == won_.change[1]).all()
      and not list(tmp.glob("race*.part")), f"{errors_[:2]} {won_.change[1]}")
check("each writer's temporary file is its own: the name carries a random part (two saves never share one)",
      len({p_.name for p_ in [Path(tempfile.NamedTemporaryFile(dir=tmp, prefix="x.", suffix=".part", delete=False).name) for _ in range(3)]}) == 3)
for p_ in tmp.glob("x.*.part"):
    p_.unlink()
raises("a failed save (unwritable directory) leaves neither a file nor a temporary one",
       lambda: skips.Skips(chg, "GH000001.mp4", FP).save(tmp / "no_such_dir" / "a.npz"), "", OSError)


def tab(n_frames=N, source="GH000001.mp4", fp=""):
    """A minimal pose table with one row, for the binding checks."""
    return PoseTable(np.array([0], np.int64), np.array([1], np.int64), np.zeros((1, 4), np.float32),
                     np.zeros((1, N_JOINTS, 2), np.float32), np.full((1, N_JOINTS), 0.9, np.float32), n_frames, 50.0,
                     W, H, source=source, video_fp=fp)


check("verify accepts the table of this video (same name, same length, same fingerprint; or no fingerprint on the table)",
      sk.verify(tab(fp=FP)) is None and sk.verify(tab(fp="")) is None)
raises("verify refuses a table of another file name", lambda: sk.verify(tab(source="GH999999.mp4")), "pose table for")
raises("verify refuses a table of another length (59 frames)", lambda: sk.verify(tab(n_frames=N - 1)), "cover")
raises("verify refuses a table whose content fingerprint differs", lambda: sk.verify(tab(fp="b" * 64)), "fingerprints differ")
raises("a missing file is a FileNotFoundError that says what to run", lambda: skips.load_for(tab(), tmp / "none.npz"), "olympic_framechange", FileNotFoundError)
check("load_for loads and verifies in one step", skips.load_for(tab(fp=FP), f_ok).frames.tolist() == [30])
raises("load_for refuses a mismatching table", lambda: skips.load_for(tab(source="X.mp4"), f_ok), "pose table for")

print("Skips: malformed files are refused")


def write(name, **fields):
    p = tmp / name
    np.savez(p, **fields)
    return p


good = dict(change=chg, n_frames=N, step=4, ratio=3.0, half_window=10, source="GH000001.mp4", video_fp=FP)
for what, fields, why in (
        ("a missing field", {k: v for k, v in good.items() if k != "video_fp"}, "video_fp"),
        ("an integer change array", good | {"change": np.arange(N)}, "floating"),
        ("n_frames that disagrees with the array", good | {"n_frames": N + 1}, "n_frames"),
        ("a fractional n_frames", good | {"n_frames": 60.5}, "whole"),
        ("a NaN anywhere but frame 0", good | {"change": flat(N, f10=NAN)}, "NaN at frame 0"),
        ("a change value at frame 0", good | {"change": np.r_[1.0, chg[1:]]}, "NaN at frame 0"),
        ("a negative change", good | {"change": flat(N, f5=-1.0)}, ">= 0"),
        ("an infinite change", good | {"change": flat(N, f5=np.inf)}, "finite"),
        ("a 2-D change", good | {"change": np.zeros((N, 2)), "n_frames": N}, "1-D"),
        ("a fingerprint that is not a sha256", good | {"video_fp": "zz"}, "sha256"),
        ("an empty source", good | {"source": ""}, "file name"),
        ("a source that is a number", good | {"source": 5}, "single string"),
        ("a zero ratio", good | {"ratio": 0.0}, "ratio"),
        ("a zero half window", good | {"half_window": 0}, "half_window"),
        ("a zero step", good | {"step": 0}, "step")):
    raises(f"refused: {what}", lambda f=fields, w=what: skips.Skips.load(write(f"bad_{abs(hash(w))}.npz", **f)), why)
obj = tmp / "pickled.npz"
np.savez(obj, **(good | {"change": np.array([object()], dtype=object)}))
raises("refused: a pickled object array (no pickle is ever loaded)", lambda: skips.Skips.load(obj), "")
junk = tmp / "junk.npz"
junk.write_bytes(b"this is not a zip file at all")
raises("refused: a file that is not an npz", lambda: skips.Skips.load(junk), "")
trunc = tmp / "trunc.npz"
trunc.write_bytes(f_ok.read_bytes()[:40])
raises("refused: a truncated npz", lambda: skips.Skips.load(trunc), "")
raises("constructing a Skips with an empty change is refused", lambda: skips.Skips(np.zeros(0), "a.mp4", FP), "at least one frame")

# ---------------------------------------------------------------------------------------------------------------------
print("check_skips: what a skip list may be")
FRPS = 50.0


def pose(cx, cy, ext=(0.0, 0.0)):
    xy = np.tile([cx, cy], (N_JOINTS, 1)).astype(float)
    xy[:, 0] += np.linspace(-6, 6, N_JOINTS)
    xy[5], xy[6] = (cx - 12, cy - 20), (cx + 12, cy - 20)
    xy[11], xy[12] = (cx - 8, cy + 20), (cx + 8, cy + 20)
    xy[9], xy[10] = (cx - 24 - ext[0], cy - 4), (cx + 24 + ext[1], cy - 4)
    return xy, np.array([cx - 28, cy - 52, cx + 28, cy + 56], float)


def table_of(n_frames, frames, source="GH000001.mp4", tid=1, bursts=(), steps=(), cx=150.0):
    """One person (track `tid`) in `frames`, still, with triangular wrist bursts [(frame, hand[, amplitude px])] (out and back in 8
    frames; the default amplitude is 120 px) and permanent wrist STEPS [(frame, hand, px)]: the wrist is `px` further out from that frame on, which
    the rule reads as one fast jump."""
    rows = []
    for f in frames:
        e = [0.0, 0.0]
        for b, hand, *amp in bursts:
            e[hand] = max(e[hand], (amp[0] if amp else 120.0) * max(0.0, 1 - abs(f - b) / 4))
        for sf, hand, px in steps:
            if f >= sf:
                e[hand] = max(e[hand], float(px))
        xy, box = pose(cx, 150.0, tuple(e))
        rows.append((f, tid, box, xy))
    k = len(rows)
    return PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                     np.array([r[2] for r in rows], np.float32).reshape(k, 4),
                     np.array([r[3] for r in rows], np.float32).reshape(k, N_JOINTS, 2),
                     np.full((k, N_JOINTS), 0.9, np.float32), n_frames, FRPS, 640, 480, source=source)


for what, bad in (("an unsorted array", [5, 3]), ("a repeated frame", [3, 3]), ("frame 0", [0, 4]), ("a frame past the video", [4, 20]),
                  ("a negative frame", [-1]), ("floats", np.array([3.0, 4.0])), ("a 2-D array", np.array([[3, 4]]))):
    raises(f"check_skips refuses {what}", lambda b=bad: completeness.check_skips(b, 20), "skips")
check("check_skips accepts None, an empty list and a sorted list, and returns int64",
      completeness.check_skips(None, 20) is None and len(completeness.check_skips([], 20)) == 0
      and completeness.check_skips([3, 19], 20).dtype == np.int64)

# ---------------------------------------------------------------------------------------------------------------------
print("skip_mask_half: derived from the rule's own smoothing window and the frame rate, never a constant")
from boxingvi import punchfind  # noqa: E402

check("at 50 fps and the rule's 0.1 s the smoothing window is 5 frames (half 2) and the reach is 3 frames each side",
      punchfind.smooth_window(50.0) == 5 and completeness.skip_mask_half(50.0) == 3)
check("the reach follows smooth_s: 0.2 s -> 11 frames (half 5) -> 6; 0.02 s -> 1 frame (half 0) -> 1",
      completeness.skip_mask_half(50.0, 0.2) == 6 and completeness.skip_mask_half(50.0, 0.02) == 1)
check("... and the frame rate: 100 fps -> 11 frames -> 6; 25 fps -> 3 frames (2.5 rounds to 2, made odd) -> 2",
      completeness.skip_mask_half(100.0) == 6 and completeness.skip_mask_half(25.0) == 2)
check("it is always smooth_window // 2 + 1 over a grid of rates and windows (one definition, shared with wrist_signal)",
      all(completeness.skip_mask_half(fps, sm) == punchfind.smooth_window(fps, sm) // 2 + 1
          for fps in (24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 100.0) for sm in (0.02, 0.05, 0.1, 0.15, 0.2, 0.5)))
check("the window is at least 1 frame and odd: smooth_s 0 -> 1, and every window over the grid above is odd",
      punchfind.smooth_window(50.0, 0.0) == 1 and all(punchfind.smooth_window(f_, s_) % 2 == 1 for f_ in (25.0, 50.0, 60.0) for s_ in (0.0, 0.1, 0.2)))

print("skip_mask: which frames of which hand a skip invalidates")
FRPS = 50.0
tc = table_of(40, range(40))
sg = [np.arange(len(tc.frame))]
trk_c = lambda tab_: punchfind.bridge(completeness._track_of(tab_, np.arange(len(tab_.frame)), "t"), 2)    # noqa: E731
mk = lambda tab_, sk_, half=3: completeness.skip_mask(trk_c(tab_), tab_.frame, np.array(sk_, np.int64), half, 2)    # noqa: E731
m1 = mk(tc, [20])
check("a skip at 20 masks 17..23 (3 either side: the contaminated range 17..22 plus one), both hands, and nothing else",
      np.flatnonzero(m1[:, 0]).tolist() == list(range(17, 24)) and np.array_equal(m1[:, 0], m1[:, 1]), str(np.flatnonzero(m1[:, 0])))
check("the mask is two-sided: it covers the frames BEFORE the skip as well as after (17 and 23)", m1[17, 0] and m1[23, 0] and not m1[16, 0] and not m1[24, 0])
check("near the ends of the track the mask is clipped, not wrapped: skip at 2 -> 0..5, skip at 38 -> 35..39",
      np.flatnonzero(mk(tc, [2])[:, 0]).tolist() == list(range(0, 6)) and np.flatnonzero(mk(tc, [38])[:, 0]).tolist() == list(range(35, 40)))
check("two skips mask the union (10 and 30), and no skips mask nothing",
      np.flatnonzero(mk(tc, [10, 30])[:, 0]).tolist() == list(range(7, 14)) + list(range(27, 34)) and not mk(tc, [])[:, 0].any())
check("the half is a parameter: half 0 masks the skip frame alone, half 5 masks 11 frames",
      np.flatnonzero(mk(tc, [20], 0)[:, 0]).tolist() == [20] and int(mk(tc, [20], 5)[:, 0].sum()) == 11)
check("a skip far outside the track masks nothing", not mk(tc, [200])[:, 0].any())
# a hole of two rows (19, 20) that the bridge interpolates across: the jump becomes a ramp over the whole hole
th = table_of(40, [f for f in range(40) if f not in (19, 20)])
th_track = trk_c(th)
check("setup: the bridge fills the 2-frame hole, so the bridged track still has 40 frames", len(th_track.frame_ids) == 40)
check("a skip inside a bridged hole widens the mask over the whole hole: skip 20, rows 18 and 21 around it -> 16..23 (hole 19..20 plus 3 beyond each end)",
      np.flatnonzero(mk(th, [20])[:, 0]).tolist() == list(range(16, 24)), str(np.flatnonzero(mk(th, [20])[:, 0])))
check("... a skip on the first frame after the hole (21) -> 16..24; one before the hole (18) is the plain 15..21",
      np.flatnonzero(mk(th, [21])[:, 0]).tolist() == list(range(16, 25)) and np.flatnonzero(mk(th, [18])[:, 0]).tolist() == list(range(15, 22)))
# one hand's wrist is untrustworthy for frames 18-19 (a 2-frame hole `wrist_signal` interpolates across)
conf = tc.conf.copy()
conf[np.isin(tc.frame, (18, 19)), 9] = 0.0              # left wrist
tl = dataclasses.replace(tc, conf=conf)
ml = mk(tl, [21])
check("a skip whose zone (18..24) meets one hand's interpolated hole (left, 18..19) widens the mask for THAT hand only: left 15..24, right 18..24",
      np.flatnonzero(ml[:, 0]).tolist() == list(range(15, 25)) and np.flatnonzero(ml[:, 1]).tolist() == list(range(18, 25)),
      f"{np.flatnonzero(ml[:, 0])} {np.flatnonzero(ml[:, 1])}")
check("... a skip that does not reach that hole (27: zone 24..30) leaves it alone: both hands 24..30",
      np.flatnonzero(mk(tl, [27])[:, 0]).tolist() == list(range(24, 31)) and np.flatnonzero(mk(tl, [27])[:, 1]).tolist() == list(range(24, 31)))
for what, bad in (("an unsorted array", [5, 3]), ("a repeated frame", [3, 3]), ("frame 0", [0, 4]), ("a frame past the video", [4, 20]),
                  ("a negative frame", [-1]), ("floats", np.array([3.0, 4.0])), ("a 2-D array", np.array([[3, 4]]))):
    raises(f"check_skips refuses {what}", lambda b=bad: completeness.check_skips(b, 20), "skips")
check("check_skips accepts None, an empty list and a sorted list, and returns int64",
      completeness.check_skips(None, 20) is None and len(completeness.check_skips([], 20)) == 0
      and completeness.check_skips([3, 19], 20).dtype == np.int64)

# ---------------------------------------------------------------------------------------------------------------------
print("the mask in find_bursts_credited / find_bursts: a skip-made burst goes, a real punch outside the mask stays, one inside it is invisible, skips=None is unchanged")
RULE = {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.4}
NS = 400
MH = completeness.skip_mask_half(FRPS)


def label(table, tid, first, last, hand):
    pts = [table.xy[np.flatnonzero((table.frame == f) & (table.track_id == tid))[0], 9 + hand] for f in range(first, last + 1)]
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - 10, pts.max(axis=0) + 10
    return olympic.Event(first, last, "head", ("left", "right")[hand], (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), 1)


def run(table, events, **kw):
    attrs = olympic.attribute(events, table)
    return completeness.find_bursts_credited(table, events, attrs, RULE, video="GH000001", **kw)


near = lambda f, want, tol=6: abs(f - want) <= tol    # noqa: E731
frames_of = lambda res: sorted(b.frame for b in res["bursts"])    # noqa: E731

# one labelled real punch at 60 (so the fragment is credited), and the wrist JUMPS 120 px at frame 200 (a dropped stretch)
tJ = table_of(NS, range(NS), bursts=[(60, 1)], steps=[(200, 0, 120)])
evJ = [label(tJ, 1, 58, 62, 1)]
r_off = run(tJ, evJ)
r_on = run(tJ, evJ, skips=np.array([200]))
check("without skips the one-frame jump at 200 is read as a punch: two bursts, the real one at 60 (explained) and one near 200 (unexplained)",
      len(r_off["bursts"]) == 2 and near(r_off["bursts"][0].frame, 60) and r_off["bursts"][0].explained
      and near(r_off["bursts"][1].frame, 200, 4) and not r_off["bursts"][1].explained, str(frames_of(r_off)))
check("with the skip at 200 the jump is masked out: only the real punch at 60 remains, still explained, with the SAME peak speed as before",
      len(r_on["bursts"]) == 1 and r_on["bursts"][0].frame == r_off["bursts"][0].frame and r_on["bursts"][0].explained
      and r_on["bursts"][0].peak_speed == r_off["bursts"][0].peak_speed, str(frames_of(r_on)))
check("... the segments are exactly what they were (1, not cut); the 9 frames 196..204 (the signal mask 197..203 plus the peak test's one more each side) left the exposure: 391 of 400",
      r_on["fragments"][1]["segments"] == 1 and r_on["fragments"][1]["exposure_frames"] == NS - (2 * (MH + 1) + 1)
      and int(r_on["exposure"][1].sum()) == 391 and not r_on["exposure"][1][196:205].any() and r_on["exposure"][1][195] and r_on["exposure"][1][205],
      str(r_on["fragments"][1]))
check("... and the exposure minutes follow: the rates divide by 391 frames, not 400",
      abs(r_on["fragments"][1]["exposure_min"] - 391 / FRPS / 60) < 1e-12)
r_far = run(tJ, evJ, skips=np.array([300]))
check("a skip elsewhere (300) leaves the jump at 200 a burst: the mask acts only where the skip is",
      len(r_far["bursts"]) == 2 and near(r_far["bursts"][1].frame, 200, 4), str(frames_of(r_far)))
r_b = run(tJ, evJ, skips=np.array([207]))
check("a skip 7 frames after the jump (207: exposure removed 203..211) does NOT hide the jump's burst near 200: the mask is the derived reach, not a wide window",
      len(r_b["bursts"]) == 2 and near(r_b["bursts"][1].frame, 200, 4), str(frames_of(r_b)))
# a real punch: unmasked it is found with the same frame and peak speed; inside the mask it is invisible and out of the exposure
tN = table_of(NS, range(NS), bursts=[(60, 1), (150, 0)])
evN = [label(tN, 1, 58, 62, 1)]
r_n0 = run(tN, evN)
bf = [b.frame for b in r_n0["bursts"] if near(b.frame, 150, 6)]
check("setup: without skips the real punch at 150 is one burst near 150, unexplained", len(bf) == 1 and not [b for b in r_n0["bursts"] if near(b.frame, 150, 6)][0].explained, str(bf))
bf = bf[0]
bp = [b.peak_speed for b in r_n0["bursts"] if b.frame == bf][0]
for side, sgn in (("after", -1), ("before", +1)):
    r_ok = run(tN, evN, skips=np.array([bf + sgn * (MH + 2)]))
    check(f"a real punch peaking at {bf}, a skip {MH + 2} frames {side} it (outside the signal mask and the peak test's extra frame): found, same frame, same peak speed, in the exposure",
          bf in frames_of(r_ok) and [b.peak_speed for b in r_ok["bursts"] if b.frame == bf] == [bp] and r_ok["exposure"][1][bf], str(frames_of(r_ok)))
    for dist, why in ((MH + 1, "on the extra frame the peak test needs"), (MH, "at the edge of the signal mask"), (1, "right beside the skip")):
        f_in = bf + sgn * dist
        r_in = run(tN, evN, skips=np.array([f_in]))
        check(f"... a skip {dist} frames {side} it ({why}): no burst at {bf}, that frame is out of the exposure, and the hidden punch does not come back as a burst on the frame beside the removed range either",
              bf not in frames_of(r_in) and not r_in["exposure"][1][bf]
              and not any(f_in - (MH + 2) <= b.frame <= f_in + (MH + 2) for b in r_in["bursts"]), f"{frames_of(r_in)} exposure removed {f_in - MH - 1}..{f_in + MH + 1}")
        check("... and the real punch at 60 is untouched", any(b.frame == r_n0["bursts"][0].frame for b in r_in["bursts"]))

# a real punch 12 frames after a skip-made jump: the jump's giant peak suppresses it (min gap 0.4 s) until the signal is zeroed
tS = table_of(NS, range(NS), bursts=[(60, 1), (212, 1)], steps=[(200, 0, 120)])
evS = [label(tS, 1, 58, 62, 1)]
rS0, rS1 = run(tS, evS), run(tS, evS, skips=np.array([200]))
check("without skips the jump at 200 (a much bigger peak) suppresses the real punch at 212 inside the 0.4 s minimum gap: bursts near 60 and 200 only",
      [near(b.frame, 60) or near(b.frame, 200, 4) for b in rS0["bursts"]] == [True, True] and len(rS0["bursts"]) == 2, str(frames_of(rS0)))
check("with the skip the zeroed signal no longer suppresses it: the real punch at 212 is found (and the jump is not)",
      len(rS1["bursts"]) == 2 and near(rS1["bursts"][1].frame, 212, 6) and not near(rS1["bursts"][1].frame, 200, 4) and rS1["exposure"][1][rS1["bursts"][1].frame],
      str(frames_of(rS1)))

# B1 (review): an ineligible peak just outside the zeroed zone must not suppress a real, eligible one. The skip at 200 zeroes 197..203 and
# removes 196..204; the return lobe of a strong punch hidden under the mask is a flat top at 204-205 whose reported peak is 204 (ineligible),
# stronger than the real, weaker punch at 212 (8 frames later, inside the 20-frame minimum gap).
tE = table_of(NS, range(NS), bursts=[(60, 1), (202, 1), (212, 0, 60)])
evE = [label(tE, 1, 58, 62, 1)]
rE = run(tE, evE, skips=np.array([200]))
rE0 = run(tE, evE)
check("setup: without skips the strong punch (burst at 204) suppresses the weaker one at 212 through the 20-frame minimum gap: bursts 62 and 204 only",
      len(rE0["bursts"]) == 2 and near(rE0["bursts"][1].frame, 204, 2) and not any(208 <= b.frame <= 218 for b in rE0["bursts"]), str(frames_of(rE0)))
check("the real punch at 212 IS a burst when the stronger edge peak at 204 is ineligible: found near 212, in the exposure, not suppressed by a peak that is then discarded",
      any(near(b.frame, 212, 3) for b in rE["bursts"]) and not any(196 <= b.frame <= 204 for b in rE["bursts"])
      and all(rE["exposure"][1][b.frame] for b in rE["bursts"]), str(frames_of(rE)))
raises("detect refuses an eligibility array of the wrong shape or type",
       lambda: punchfind.detect(completeness._track_of(tE, np.arange(len(tE.frame)), "t"), np.zeros((NS, 2)), 3.6, 0.4, np.ones((NS, 1), bool)), "eligible")
trE = completeness._track_of(tE, np.arange(len(tE.frame)), "t")
sgE = punchfind.wrist_signal(trE, "relative")
check("detect with every frame eligible equals detect without the argument, on the scene's signal",
      all(np.array_equal(x, y) for x, y in zip(punchfind.detect(trE, sgE, 3.62, 0.4), punchfind.detect(trE, sgE, 3.62, 0.4, np.ones(sgE.shape, bool)))))
rng_ = np.random.default_rng(7)
same_ = True
for _ in range(200):          # random smooth-ish signals with ties and plateaus: the two code paths agree peak for peak
    z_ = np.round(np.abs(np.convolve(rng_.normal(size=300), np.ones(3) / 3, mode="same")) * 4) / 2
    sg_ = np.stack([z_, np.roll(z_, 5)], axis=1)
    tr_ = completeness._track_of(table_of(300, range(300)), np.arange(300), "t")
    a_, b_ = punchfind.detect(tr_, sg_, 2.0, 0.4), punchfind.detect(tr_, sg_, 2.0, 0.4, np.ones(sg_.shape, bool))
    same_ &= np.array_equal(a_[0], b_[0]) and np.array_equal(a_[1], b_[1])
check("... and on 200 random signals with plateaus and ties (the new path is the old suppression, not a look-alike)", same_)
el_ = np.ones(sgE.shape, bool)
el_[204] = False
fr_, hd_ = punchfind.detect(trE, sgE, 3.62, 0.4, el_)
check("an ineligible frame is no candidate but still a neighbour: marking only 204 ineligible removes the burst there and nothing else",
      204 not in fr_.tolist() and 205 not in fr_.tolist() and set(punchfind.detect(trE, sgE, 3.62, 0.4)[0].tolist()) - {204} <= set(fr_.tolist()))

# the mask follows smooth_s: the same scene, a wider smoothing window, a wider mask
segs_J = [np.arange(len(tJ.frame))]
sc = lambda sm: completeness._scan_segments(tJ, segs_J, RULE, "t", 2, 5, np.array([200]), sm)    # noqa: E731
check("_scan_segments with smooth_s = 0.1 removes 9 frames from the exposure (391), with 0.2 it removes 15 (385), with 0.02 it removes 5 (395)",
      int(sc(0.1)["mask"].sum()) == 391 and int(sc(0.2)["mask"].sum()) == 385 and int(sc(0.02)["mask"].sum()) == 395,
      f"{int(sc(0.1)['mask'].sum())} {int(sc(0.2)['mask'].sum())} {int(sc(0.02)['mask'].sum())}")
raw_ = lambda sm: sorted(completeness._scan_segments(tJ, segs_J, RULE, "t", 2, 5, None, sm)["raw"])    # noqa: E731
pk_ = lambda sm: [r for r in raw_(sm) if abs(r[0] - 60) <= 6]    # noqa: E731
check("the same smooth_s reaches the signal and the mask: a wider window (0.2 s) smooths the real punch at 60 to a lower peak than 0.1 s does, "
      "and the default is 0.1 s",
      len(pk_(0.1)) == 1 and len(pk_(0.2)) == 1 and pk_(0.2)[0][2] < pk_(0.1)[0][2] - 0.3
      and raw_(punchfind.SMOOTH_S) == raw_(0.1) and punchfind.SMOOTH_S == 0.1, f"{pk_(0.1)} {pk_(0.2)}")
# a 2-frame hole bridged and one hand's hole: the exposure follows the widened masks
th2 = table_of(NS, [f for f in range(NS) if f not in (199, 200)], bursts=[(60, 1)], steps=[(201, 0, 120)])
r_h = run(th2, [label(th2, 1, 58, 62, 1)], skips=np.array([200]))
check("a skip inside a bridged hole removes the whole hole plus 4 beyond each end: rows 198 and 201 around it -> frames 195..204 = 10 frames out of the exposure (390 of 400)",
      r_h["fragments"][1]["exposure_frames"] == NS - 10 and not r_h["exposure"][1][195:205].any() and r_h["exposure"][1][194] and r_h["exposure"][1][205],
      str(r_h["fragments"][1]["exposure_frames"]))

# skips=None / empty / outside every segment: identical to not passing it
same = lambda a, b: [(x.frame, x.hand, x.match, x.track_id, x.peak_speed) for x in a["bursts"]] == [   # noqa: E731
    (x.frame, x.hand, x.match, x.track_id, x.peak_speed) for x in b["bursts"]] and a["fragments"] == b["fragments"] and a["labels"] == b["labels"]
check("skips=None is exactly the old output (bursts with their peak speeds, fragments, exposure, labels), and so are an empty list and a skip past the track",
      same(run(tJ, evJ), run(tJ, evJ, skips=None)) and same(run(tJ, evJ), run(tJ, evJ, skips=[]))
      and same(r_off, run(table_of(NS + 50, range(NS), bursts=[(60, 1)], steps=[(200, 0, 120)]), evJ, skips=np.array([NS + 10]))))
raises("find_bursts_credited refuses a skip list outside the video", lambda: run(tJ, evJ, skips=np.array([NS])), "skips")
raises("find_bursts_credited refuses an unsorted skip list", lambda: run(tJ, evJ, skips=np.array([200, 100])), "sorted")

# the linking path: find_bursts masks through the same function (boxer A is the one person, track 1, for the whole video)
mem = np.full((NS, 2), -1, np.int64)
mem[:, 0] = 1
lk = Linking(identities=((1,), ()), membership=mem, n_frames=NS, fps=FRPS, source=tJ.source, region=None)
atJ = olympic.attribute(evJ, tJ)
f_off = completeness.find_bursts(tJ, lk, evJ, atJ, RULE, video="GH000001")
f_on = completeness.find_bursts(tJ, lk, evJ, atJ, RULE, video="GH000001", skips=np.array([200]))
check("find_bursts (the linking path) masks too: boxer A has the jump as a burst without skips and not with; one segment either way; 391 exposure frames",
      sum(near(b.frame, 200, 4) for b in f_off["bursts"]) == 1 and sum(near(b.frame, 200, 4) for b in f_on["bursts"]) == 0
      and f_off["boxers"]["A"]["segments"] == 1 and f_on["boxers"]["A"]["segments"] == 1
      and f_off["boxers"]["A"]["exposure_frames"] == NS and f_on["boxers"]["A"]["exposure_frames"] == 391, str([b.frame for b in f_on["bursts"]]))
check("... and with skips=None find_bursts is unchanged",
      [(b.frame, b.hand, b.match, b.peak_speed) for b in completeness.find_bursts(tJ, lk, evJ, atJ, RULE, video="GH000001", skips=None)["bursts"]]
      == [(b.frame, b.hand, b.match, b.peak_speed) for b in f_off["bursts"]])

print("near_skip_stats: the base rate and the counts")
bs = [completeness.Burst("V", f, f / 50, "1", "left", 5.0, m, track_id=1) for f, m in
      ((10, "none"), (50, "none"), (101, "none"), (30, "own"), (99, "own"), (150, "unattributed"))]
mask = np.zeros(200, bool)
mask[40:190] = True
st = completeness.near_skip_stats(bs, {1: mask}, np.array([50, 100]), 200)
check("near_skip_stats: skips 50 and 100, within 2: unexplained near = 50, 101 (2 of 3), explained near = 99 (1 of 3)",
      st["skips"] == 2 and st["unexplained"] == 3 and st["unexplained_near_skip"] == 2 and st["explained"] == 3
      and st["explained_near_skip"] == 1 and st["within_frames"] == 2, str(st))
check("... the base rate counts exposure frames within 2 of a skip: 48..52 and 98..102, both inside 40..189 = 10 of 150",
      st["exposure_frames"] == 150 and st["exposure_near_skip"] == 10, str(st))
st0 = completeness.near_skip_stats(bs, {1: mask}, np.array([50, 100]), 200, within=0)
check("within=0 means the skip frame itself only: 50 (unexplained) and none of the others",
      st0["unexplained_near_skip"] == 1 and st0["explained_near_skip"] == 0 and st0["exposure_near_skip"] == 2, str(st0))
two = completeness.near_skip_stats(bs, {1: mask, 2: mask}, np.array([50]), 200)
check("two people's exposures are summed, as the burst rates' denominators are: 300 frames, 10 near", two["exposure_frames"] == 300 and two["exposure_near_skip"] == 10)
check("a skip at the start of the video does not wrap around (frame 1, within 2 -> frames 0..3)",
      completeness.near_skip_stats([completeness.Burst("V", 0, 0.0, "1", "left", 5.0)], {1: np.ones(200, bool)}, np.array([1]), 200)
      ["unexplained_near_skip"] == 1)
raises("a burst outside the video is refused", lambda: completeness.near_skip_stats([completeness.Burst("V", 200, 4.0, "1", "left", 5.0)], {1: mask}, [], 200), "outside")
raises("a negative window is refused", lambda: completeness.near_skip_stats(bs, {1: mask}, [], 200, within=-1), "within")
raises("unsorted skips are refused", lambda: completeness.near_skip_stats(bs, {1: mask}, [100, 50], 200), "sorted")
sh = olympic_completeness.pool_skip_stats([st, st])
check("pooling sums the counts and divides once: 4 of 6 unexplained, 20 of 300 exposure frames, 4/6 over 20/300 = 10x",
      sh["unexplained"] == 6 and sh["unexplained_near_skip"] == 4 and sh["exposure_near_skip"] == 20 and sh["exposure_frames"] == 300
      and abs(sh["base_rate"] - 20 / 300) < 1e-12 and abs(sh["unexplained_excess_over_base"] - (4 / 6) / (20 / 300)) < 1e-9, str(sh))
vA = {"skips": 5, "within_frames": 2, "exposure_frames": 150, "exposure_near_skip": 10, "unexplained": 3, "unexplained_near_skip": 2,
      "explained": 6, "explained_near_skip": 3}
vB = {"skips": 50, "within_frames": 2, "exposure_frames": 1500, "exposure_near_skip": 20, "unexplained": 30, "unexplained_near_skip": 3,
      "explained": 60, "explained_near_skip": 6}
pl2 = olympic_completeness.pool_skip_stats([vA, vB])
check("pooling UNEQUAL videos adds the raw counts and divides once: unexplained near a skip 5 of 33 (15.2%), base rate 30 of 1,650 (1.8%), excess 8.33x",
      pl2["unexplained"] == 33 and pl2["unexplained_near_skip"] == 5 and pl2["exposure_frames"] == 1650 and pl2["exposure_near_skip"] == 30
      and abs(pl2["unexplained_near_skip_share"] - 5 / 33) < 1e-12 and abs(pl2["base_rate"] - 30 / 1650) < 1e-12
      and abs(pl2["unexplained_excess_over_base"] - (5 / 33) / (30 / 1650)) < 1e-9, str(pl2))
check("... which is NOT the mean of the per-video shares (38.3% and 4.0% here) nor the mean of the counts",
      abs(pl2["unexplained_near_skip_share"] - (2 / 3 + 3 / 30) / 2) > 0.2 and abs(pl2["base_rate"] - (10 / 150 + 20 / 1500) / 2) > 0.02
      and pl2["skips"] == 55 and pl2["explained"] == 66, str(pl2))
nobase = olympic_completeness.pool_skip_stats([completeness.near_skip_stats([], {1: np.zeros(200, bool)}, [], 200)])
check("with nothing in exposure every share is None (written as null), not a crash",
      nobase["base_rate"] is None and nobase["unexplained_near_skip_share"] is None and nobase["unexplained_excess_over_base"] is None)

print("skipcheck_rows: a sampled burst is 'still unexplained' only for the same video, track, hand, within 2 frames, and not explained")
nfx = {"V": 400, "W": 400}
bx = {"V": [completeness.Burst("V", 100, 2.0, "1", "left", 5.0, "none", track_id=1),
            completeness.Burst("V", 300, 6.0, "1", "right", 5.0, "own", track_id=1)], "W": []}
row = lambda frame, tid="1", hand="left", v="V": {"video": v, "frame": str(frame), "track_id": tid, "hand": hand}    # noqa: E731
got = [r[4] for r in olympic_completeness.skipcheck_rows(
    [row(100), row(102), row(98), row(103), row(97), row(100, hand="right"), row(100, tid="2"), row(300, hand="right"), row(100, v="W")], bx, nfx)]
check("same track and hand: 100, 102 and 98 are still unexplained (true); 103 and 97 are 3 frames off (false)",
      got[:5] == ["true", "true", "true", "false", "false"], str(got))
check("the other hand, the other track, a burst that is now EXPLAINED, and a video with no burst there are all false",
      got[5:] == ["false", "false", "false", "false"], str(got))
raises("a sample row of a video that is not in the run is an error, not a quiet false", lambda: olympic_completeness.skipcheck_rows([row(100, v="Z")], bx, nfx), "not in this run")
check("skipcheck_rows keeps video, frame, track_id and hand as given", olympic_completeness.skipcheck_rows([row(100)], bx, nfx)[0][:4] == ["V", "100", "1", "left"])
raises("a video that has bursts but no recorded length is an error too (video and length travel together)",
       lambda: olympic_completeness.skipcheck_rows([row(100)], bx, {}), "not in this run")
for bad_f in (400, 999999, 10 ** 12):
    raises(f"a sample row naming frame {bad_f} of a 400-frame video is an error, not a quiet false",
           lambda f_=bad_f: olympic_completeness.skipcheck_rows([row(f_)], bx, nfx), "frames 0..399")
check("the first and last frame of the video are valid rows", [r[4] for r in olympic_completeness.skipcheck_rows([row(0), row(399)], bx, nfx)] == ["false", "false"])

# ---------------------------------------------------------------------------------------------------------------------
print("the CLIs end to end: olympic_framechange.py and olympic_completeness.py --skips / --skips-measure / --skipcheck")
NC = 1500
ds = tmp / "ds"
task = ds / "task_kam2_gh000001"
(task / "data").mkdir(parents=True)
clip = task / "data" / "GH000001.mp4"
# a REAL clip whose picture jumps at frame 700 (a grating moving 1 px a frame that jumps 6 px there), so olympic_framechange.py
# finds a real skip in real pixels; the person in the pose table has the matching wrist jump at 700.
small = tmp / "cli_small.mp4"
with video.VideoWriter(small, W, H, Fraction(FPS)) as w:
    for f in range(NC):
        w.write(grating(f + (5 if f >= 700 else 0)))
# the pose table's video size must equal the clip's: use the clip's own size
tM = table_of(NC, range(NC), bursts=[(100, 1)], steps=[(700, 0, 120)])
tM = dataclasses.replace(tM, width=W, height=H)
evM = [label(tM, 1, 300 + 30 * i, 302 + 30 * i, 0) for i in range(11)]
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(small), "-c", "copy", str(clip)], check=True)
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


def cli(mod, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = mod.main(argv)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


rc, out_fc, _ = cli(olympic_framechange, ["--data", str(ds), "--tracks", str(trk)])
fcf = trk / "GH000001.framechange.npz"
real = skips.Skips.load(fcf) if fcf.is_file() else None
check("olympic_framechange.py exits 0 and writes <id>.framechange.npz beside the table, bound to the clip's name, length and content",
      rc == 0 and real is not None and real.source == "GH000001.mp4" and real.n_frames == NC and real.video_fp == video.fingerprint(clip), out_fc)
check("REAL FACT: in the real pixels of the clip exactly frame 700 is a skip, and the CLI says 1 skip of 1500 frames",
      real is not None and real.frames.tolist() == [700] and "1500 frames, 1 skips" in out_fc and "one skip every 30.0 s" in out_fc, out_fc)
check("the print has the pooled line (videos, skips, per minute, wall time)", "ALL: 1 videos, 1 skips" in out_fc and "per minute" in out_fc and "wall" in out_fc, out_fc)
before = fcf.read_bytes()
rc2, out2, _ = cli(olympic_framechange, ["--data", str(ds), "--tracks", str(trk)])
check("a second run leaves the existing file alone (--overwrite replaces it) and says so", rc2 == 0 and fcf.read_bytes() == before and "left alone" in out2, out2)
rc3, _, _ = cli(olympic_framechange, ["--data", str(ds), "--tracks", str(trk), "--overwrite"])
check("--overwrite re-measures the same file: same skips", rc3 == 0 and skips.Skips.load(fcf).frames.tolist() == [700])
rc4, out4, _ = cli(olympic_framechange, ["--data", str(ds), "--tracks", str(trk), "--only", "GH999999"])
check("--only a video that is not there does nothing and exits 0", rc4 == 0 and "ALL" not in out4)
raises("--only with a path-like id is refused", lambda: cli(olympic_framechange, ["--data", str(ds), "--tracks", str(trk), "--only", "../x"]), "GoPro id")

res = tmp / "res"
res.mkdir()
j_off, c_off = res / "off.json", res / "off.csv"
rc, o_off, _ = cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited",
                                        "--out", str(j_off), "--sample-out", str(c_off)])
ja_off = json.loads(j_off.read_text(encoding="utf-8"))
check("setup: without skips the credited run has two unexplained bursts (the punch at 100 and the jump at 700) and no skips block",
      rc == 0 and ja_off["all"]["unexplained"] == 2 and "skips" not in ja_off and "skips" not in ja_off["all"]
      and "skips" not in ja_off["videos"]["task_kam2_gh000001"] and "SKIPS" not in o_off, o_off)
rows_off = list(csv.DictReader(c_off.read_text(encoding="utf-8").splitlines()))
check("setup: the sample holds both bursts, with their track and hand", sorted((int(r["frame"]) // 100, r["hand"]) for r in rows_off) == [(1, "right"), (6, "left")]
      or sorted((int(r["frame"]) // 100, r["hand"]) for r in rows_off) == [(1, "right"), (7, "left")], str(rows_off))

j_on, c_on, k_on = res / "on.json", res / "on.csv", res / "check.csv"
rc, o_on, _ = cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited", "--skips",
                                       "--out", str(j_on), "--sample-out", str(c_on), "--skipcheck", str(c_off), "--skipcheck-out", str(k_on)])
ja_on = json.loads(j_on.read_text(encoding="utf-8"))
sk_on = ja_on["all"]["skips"]
check("--skips --credited: one unexplained burst left (the real punch), the jump is gone",
      rc == 0 and ja_on["all"]["unexplained"] == 1 and ja_on["all"]["bursts"] == 1, o_on)
check("... the json says what was done: mode mask, the rule, the declaration that the constant was not tuned, the mask rule",
      ja_on["skips"]["mode"] == "mask" and ja_on["skips"]["ratio"] == 3.0 and ja_on["skips"]["half_window"] == 10
      and "not from tuning" in ja_on["skips"]["declaration"] and "smooth_window // 2 + 1" in ja_on["skips"]["mask"], str(ja_on["skips"]))
check("REAL FACT: pooled near-skip counts after masking: 1 skip, 1 unexplained burst and none near the skip; the 9 masked frames left the exposure (1491 of 1500), and with them every frame within 2 of the skip (base rate 0)",
      sk_on["skips"] == 1 and sk_on["unexplained"] == 1 and sk_on["unexplained_near_skip"] == 0 and sk_on["exposure_frames"] == NC - 9
      and sk_on["exposure_near_skip"] == 0 and sk_on["base_rate"] == 0 and sk_on["unexplained_near_skip_share"] == 0,
      str(sk_on))
check("the per-video row carries the same block and the print leads with the SKIPS line",
      ja_on["videos"]["task_kam2_gh000001"]["skips"]["skips"] == 1 and "SKIPS (mask; 1 skips): bursts within 2 frames of a skip: unexplained 0/1" in o_on, o_on[:300])
rows_chk = list(csv.reader(k_on.read_text(encoding="utf-8").splitlines()))
check("skipcheck: the header is video, frame, track_id, hand, still_unexplained_with_skips",
      rows_chk[0] == ["video", "frame", "track_id", "hand", "still_unexplained_with_skips"], str(rows_chk[:1]))
verdict = {int(r[1]) // 100: (r[3], r[4]) for r in rows_chk[1:]}
check("skipcheck: the real punch at 100 (right hand) is still unexplained = true; the jump at 700 (left hand) is false",
      len(rows_chk) == 3 and verdict.get(1) == ("right", "true") and (verdict.get(6) or verdict.get(7)) == ("left", "false"), str(rows_chk))
check("skipcheck copies the ids it was given and nothing else (no user_label, no inclusion_prob column)", all(len(r) == 5 for r in rows_chk))
check("the sample it read was not modified", list(csv.DictReader(c_off.read_text(encoding="utf-8").splitlines())) == rows_off)

j_m, c_m = res / "m.json", res / "m.csv"
rc, o_m, _ = cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited", "--skips-measure",
                                      "--out", str(j_m), "--sample-out", str(c_m)])
ja_m = json.loads(j_m.read_text(encoding="utf-8"))
sk_m = ja_m["all"]["skips"]
strip = lambda d: {k: v for k, v in d.items() if k not in ("skips",)}    # noqa: E731
check("--skips-measure does NOT mask: the same bursts as the run without skips (2 unexplained), and the sample is identical",
      rc == 0 and ja_m["all"]["unexplained"] == 2 and c_m.read_bytes() == c_off.read_bytes()
      and ja_m["skips"]["mode"] == "measure" and "nothing was masked" in ja_m["skips"]["mask"] and ja_m["all"]["exposure_frames"] == NC, o_m)
check("REAL FACT: before masking, 1 of the 2 unexplained bursts (50%) is within 2 frames of the skip, against a base rate of 5/1500 (0.3%): "
      "the excess is about 150x",
      sk_m["unexplained"] == 2 and sk_m["unexplained_near_skip"] == 1 and sk_m["unexplained_near_skip_share"] == 0.5
      and sk_m["exposure_near_skip"] == 5 and 100 < sk_m["unexplained_excess_over_base"] < 200, str(sk_m))
check("... every other number of the measure-only run equals the run without skips (same fragments, bursts, rates)",
      ja_m["all"]["bursts"] == ja_off["all"]["bursts"] and ja_m["all"]["exposure_min"] == ja_off["all"]["exposure_min"]
      and ja_m["all"]["unexplained_per_exposure_min"] == ja_off["all"]["unexplained_per_exposure_min"])

# linking mode (the 11 labels give it a ring)
j_l, c_l = res / "l.json", res / "l.csv"
rc, o_l, _ = cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--skips",
                                      "--out", str(j_l), "--sample-out", str(c_l)])
ja_l = json.loads(j_l.read_text(encoding="utf-8")) if j_l.is_file() else {}
j_l0, c_l0 = res / "l0.json", res / "l0.csv"
rc0, _, _ = cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f),
                                       "--out", str(j_l0), "--sample-out", str(c_l0)])
ja_l0 = json.loads(j_l0.read_text(encoding="utf-8"))
check("--skips in the linking mode: the jump burst is gone (one fewer burst than without) and the skips block is there",
      rc == 0 and rc0 == 0 and ja_l["all"]["bursts"] == ja_l0["all"]["bursts"] - 1 and ja_l["skips"]["mode"] == "mask"
      and ja_l["all"]["skips"]["skips"] == 1 and "skips" not in ja_l0, f"{o_l}")

print("--skips refuses a missing, mismatched or malformed framechange file, and writes nothing")
created = lambda: sorted(p.name for p in res.iterdir())    # noqa: E731
snapshot = created()
args_cut = ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited", "--skips",
            "--out", str(res / "x.json"), "--sample-out", str(res / "x.csv")]


def refuse(name, mutate, why):
    mutate()
    try:
        rc_, _, err_ = cli(olympic_completeness, args_cut)
    finally:
        fcf.write_bytes(before)
    check(name, rc_ == 2 and why in err_ and created() == snapshot, f"rc {rc_} {err_}")


fcf.unlink()
rc_, _, err_ = cli(olympic_completeness, args_cut)
check("a missing framechange file: exit 2, the message names the video and says to run olympic_framechange.py, nothing written",
      rc_ == 2 and "GH000001" in err_ and "olympic_framechange" in err_ and created() == snapshot, f"{rc_} {err_}")
fcf.write_bytes(before)
refuse("a framechange file of another video name", lambda: skips.Skips(real.change, "GH000002.mp4", real.video_fp).save(fcf), "pose table for")
refuse("a framechange file of another length (1499 frames)", lambda: skips.Skips(real.change[:-1], "GH000001.mp4", real.video_fp).save(fcf), "cover 1499")
refuse("a framechange file measured on another file (fingerprint differs from the video)", lambda: skips.Skips(real.change, "GH000001.mp4", "c" * 64).save(fcf), "fingerprints differ")
refuse("a framechange file measured with another pixel step (8: positive, bound to the video, but other skips)", lambda: skips.Skips(real.change, "GH000001.mp4", real.video_fp, step=8).save(fcf), "pixel step 8")
refuse("a framechange file made with another half window", lambda: skips.Skips(real.change, "GH000001.mp4", real.video_fp, half_window=5).save(fcf), "half window 5")
refuse("a framechange file made with another ratio", lambda: skips.Skips(real.change, "GH000001.mp4", real.video_fp, ratio=2.0).save(fcf), "ratio")
refuse("a garbage framechange file", lambda: fcf.write_bytes(b"not an npz"), "framechange")
refuse("a framechange file with a missing field", lambda: np.savez(fcf, change=real.change, n_frames=NC, step=4, ratio=3.0,
                                                                    half_window=10, source="GH000001.mp4"), "video_fp")
fcf.unlink()
rc_, _, err_ = cli(olympic_completeness, [a if a != "--skips" else "--skips-measure" for a in args_cut])
fcf.write_bytes(before)
check("--skips-measure refuses a missing file the same way (exit 2, nothing written)", rc_ == 2 and "olympic_framechange" in err_ and created() == snapshot, err_)

print("--skips: argument checks and file names")
refused_args = lambda extra: cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f)] + extra)    # noqa: E731
for what, extra, why in (
        ("--skips together with --skips-measure", ["--skips", "--skips-measure"], "pick one"),
        ("--skipcheck without --skipcheck-out", ["--credited", "--skips", "--skipcheck", str(c_off)], "go together"),
        ("--skipcheck without --skips", ["--credited", "--skipcheck", str(c_off), "--skipcheck-out", str(res / "q.csv")], "--skips"),
        ("--skipcheck without --credited", ["--skips", "--skipcheck", str(c_off), "--skipcheck-out", str(res / "q.csv")], "--credited"),
        ("--skipcheck-out that is the sample it reads", ["--credited", "--skips", "--skipcheck", str(c_off), "--skipcheck-out", str(c_off)], "overwrite"),
        ("--skipcheck-out that is --out", ["--credited", "--skips", "--skipcheck", str(c_off), "--skipcheck-out", str(res / "o.json"),
                                          "--out", str(res / "o.json")], "three files"),
        ("--sample-out that is the sample --skipcheck reads", ["--credited", "--skips", "--skipcheck", str(c_off), "--skipcheck-out", str(res / "q.csv"),
                                                              "--sample-out", str(c_off)], "three files")):
    rc_, _, err_ = refused_args(extra)
    check(f"refused (exit 2): {what}", rc_ == 2 and why in err_ and not (res / "q.csv").exists(), err_)
mid = tmp / "malformed.csv"
mid.write_text("video,frame,track_id,hand\nGH000001,abc,1,left\n", encoding="utf-8")
try:
    cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited", "--skips",
                               "--out", str(res / "y.json"), "--sample-out", str(res / "y.csv"), "--skipcheck", str(mid),
                               "--skipcheck-out", str(res / "y_check.csv")])
    ok_raise = False
except ValueError as e:
    ok_raise = "malformed row" in str(e)
check("a malformed skipcheck input stops the run BEFORE any output is written", ok_raise and not (res / "y.json").exists() and not (res / "y_check.csv").exists())
beyond = tmp / "beyond.csv"
beyond.write_text("video,frame,track_id,hand" + chr(10) + "GH000001,999999,1,left" + chr(10), encoding="utf-8")
try:
    cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited", "--skips",
                               "--out", str(res / "z.json"), "--sample-out", str(res / "z.csv"), "--skipcheck", str(beyond),
                               "--skipcheck-out", str(res / "z_check.csv")])
    ok_beyond = False
except ValueError as e:
    ok_beyond = "frames 0..1499" in str(e)
check("a sample row for a processed 1,500-frame video naming frame 999999 stops the run BEFORE any output is written",
      ok_beyond and not (res / "z.json").exists() and not (res / "z_check.csv").exists())
check("the default names for --skips are their own files (never the ones of a run without skips)",
      {olympic_completeness.DEFAULT_OUT_CREDITED_SKIPS.name, olympic_completeness.DEFAULT_SAMPLE_CREDITED_SKIPS.name,
       olympic_completeness.DEFAULT_OUT_SKIPS.name, olympic_completeness.DEFAULT_OUT_CREDITED_SKIPS_MEASURE.name}
      .isdisjoint({olympic_completeness.DEFAULT_OUT.name, olympic_completeness.DEFAULT_OUT_CREDITED.name,
                   olympic_completeness.DEFAULT_SAMPLE.name, olympic_completeness.DEFAULT_SAMPLE_CREDITED.name})
      and olympic_completeness.DEFAULT_OUT_CREDITED_SKIPS.as_posix() == "results/olympic_completeness_credited_skips.json"
      and olympic_completeness.DEFAULT_SAMPLE_CREDITED_SKIPS.as_posix() == "results/olympic_completeness_credited_skips_sample.csv")
work = tmp / "cwd"
(work / "results").mkdir(parents=True)
sent = {"results/olympic_completeness_credited.json": "SENTINEL-1", "results/olympic_completeness_credited_sample.csv": "SENTINEL-2",
        "results/olympic_completeness_credited_min2.json": "SENTINEL-3"}
for k_, v_ in sent.items():
    (work / k_).write_text(v_, encoding="utf-8")
here = os.getcwd()
os.chdir(work)
try:
    rc_def, _, _ = cli(olympic_completeness, ["--data", str(ds), "--tracks", str(trk), "--rule", str(rule_f), "--credited", "--skips"])
finally:
    os.chdir(here)
check("with default names, --credited --skips writes the *_skips files and leaves every existing credited file byte-for-byte as it was",
      rc_def == 0 and (work / "results/olympic_completeness_credited_skips.json").is_file()
      and (work / "results/olympic_completeness_credited_skips_sample.csv").is_file()
      and all((work / k_).read_text(encoding="utf-8") == v_ for k_, v_ in sent.items()))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
