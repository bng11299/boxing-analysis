"""Tests for the ring-floor filter: boxingvi.ring and olympic_ring.py.

No Olympic data: floors, tables and videos are synthetic. Most checks are refusals, because the floor file and
the stillness file are read back from disk (trust boundaries) and a wrong polygon silently removes boxers. The
end-to-end section plants two boxers, a referee and two spectators in a dataset named like a development video,
and asserts REAL FACTS: the spectators and only they are dropped, and the evaluator's baseline equals what
olympic_completeness.py itself reports on the same files (the gate's "same code" claim).

Run:  python tests/test_ring.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import olympic_completeness  # noqa: E402
import olympic_ring  # noqa: E402
from boxingvi import olympic, ring  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402
from boxingvi.video import VideoWriter, fingerprint, probe  # noqa: E402

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
FP = "a" * 64
SQ = [[20, 20], [80, 20], [80, 80], [20, 80]]


def entry(**over):
    e = {"points": SQ, "width": 100, "height": 100, "video_fp": FP, "click_s": 12.5}
    e.update(over)
    return e


def write_doc(name, doc=None, raw=None):
    p = tmp / name
    if raw is not None:
        p.write_bytes(raw)
    else:
        p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def floors_doc(videos):
    return {"format": ring.FORMAT, "videos": videos}


# ======================================================================== the floor file
print("floor file: a valid file loads, and round-trips through dump_floors")
good = write_doc("good.json", floors_doc({"GH078416": entry(), "GH128416": entry(points=[[0, 0], [100, 0], [100, 100]], click_s=None)}))
fl = ring.load_floors(good)
check("both floors load, with their points, size, fingerprint and timing",
      set(fl) == {"GH078416", "GH128416"} and fl["GH078416"].points.tolist() == [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]
      and fl["GH078416"].width == 100 and fl["GH078416"].video_fp == FP and fl["GH078416"].click_s == 12.5
      and fl["GH128416"].click_s is None)
check("points on the frame's edge (a floor that runs off the picture) are accepted", len(fl["GH128416"].points) == 3)
check("the loaded points are read-only", not fl["GH078416"].points.flags.writeable)
ring.dump_floors(fl, tmp / "out" / "rt.json")
check("dump then load gives the same floors", {k: (v.points.tolist(), v.width, v.height, v.video_fp, v.click_s)
                                                for k, v in ring.load_floors(tmp / "out" / "rt.json").items()}
      == {k: (v.points.tolist(), v.width, v.height, v.video_fp, v.click_s) for k, v in fl.items()})
raises("dump refuses an entry filed under another video's name",
       lambda: ring.dump_floors({"GH999999": fl["GH078416"]}, tmp / "out" / "x.json"), "holds the floor of")

print("floor file: refusals (each names what is wrong; nothing is repaired)")
raises("a missing file is an OSError, not a silent empty set", lambda: ring.load_floors(tmp / "nope.json"), exc=OSError)
raises("a file over 1 MiB", lambda: ring.load_floors(write_doc("big.json", raw=b" " * (ring.MAX_FILE_BYTES + 1))), "larger than")
raises("non-UTF-8 bytes", lambda: ring.load_floors(write_doc("bin.json", raw=b"\xff\xfe{}")), "UTF-8 JSON")
raises("broken JSON", lambda: ring.load_floors(write_doc("broken.json", raw=b'{"format": ')), "UTF-8 JSON")
raises("deep nesting (a recursion bomb under the size cap)",
       lambda: ring.load_floors(write_doc("deep.json", raw=b"[" * 200000 + b"]" * 200000)), "")
raises("a duplicated top-level key",
       lambda: ring.load_floors(write_doc("dup.json", raw=('{"format": "%s", "videos": {}, "videos": {}}' % ring.FORMAT).encode())), "twice")
raises("a duplicated video id (the second would silently win)",
       lambda: ring.load_floors(write_doc("dupv.json", raw=('{"format": "%s", "videos": {"GH1": %s, "GH1": %s}}'
                                                           % (ring.FORMAT, json.dumps(entry()), json.dumps(entry()))).encode())), "twice")
raises("NaN", lambda: ring.load_floors(write_doc("nan.json", raw=json.dumps(floors_doc({"GH1": entry()})).replace("12.5", "NaN").encode())),
       "not a number")
raises("Infinity", lambda: ring.load_floors(write_doc("inf.json", raw=json.dumps(floors_doc({"GH1": entry()})).replace("12.5", "Infinity").encode())),
       "not a number")
raises("1e400 (parses to inf)", lambda: ring.load_floors(write_doc("big_num.json", raw=json.dumps(floors_doc({"GH1": entry()})).replace("12.5", "1e400").encode())),
       "not finite")
raises("an extra top-level key", lambda: ring.load_floors(write_doc("extra.json", floors_doc({}) | {"note": 1})), "exactly the keys")
raises("a missing 'videos'", lambda: ring.load_floors(write_doc("miss.json", {"format": ring.FORMAT})), "exactly the keys")
raises("a different format string", lambda: ring.load_floors(write_doc("fmt.json", {"format": "v0", "videos": {}})), "format")
raises("'videos' that is a list", lambda: ring.load_floors(write_doc("vl.json", {"format": ring.FORMAT, "videos": []})), "object")
raises("more than MAX_VIDEOS entries",
       lambda: ring.load_floors(write_doc("many.json", floors_doc({f"GH{i}": entry() for i in range(ring.MAX_VIDEOS + 1)}))), "at most")
for bad_id in ("../GH1", "", "x" * 33, "GH 1", "GH1.mp4"):
    raises(f"a video id that is not a GoPro id: {bad_id!r}", lambda b=bad_id: ring.load_floors(write_doc("id.json", floors_doc({b: entry()}))),
           "GoPro id")
raises("an entry with an extra key", lambda: ring.load_floors(write_doc("ek.json", floors_doc({"GH1": entry(colour="red")}))), "exactly the keys")
raises("an entry missing click_s", lambda: ring.load_floors(write_doc("mk.json", floors_doc({"GH1": {k: v for k, v in entry().items() if k != "click_s"}}))),
       "exactly the keys")
raises("an entry that is a list", lambda: ring.load_floors(write_doc("el.json", floors_doc({"GH1": [1, 2]}))), "exactly the keys")
CASES = [
    ("two points", dict(points=[[0, 0], [50, 50]]), "3 to 32"),
    ("33 points", dict(points=[[50 + 40 * math.cos(k / 33 * 6.283), 50 + 40 * math.sin(k / 33 * 6.283)] for k in range(33)]), "3 to 32"),
    ("points that are not a list", dict(points="20,20 80,20 80,80"), "3 to 32"),
    ("a point with three numbers", dict(points=[[20, 20, 0], [80, 20], [80, 80]]), "pair"),
    ("a boolean coordinate", dict(points=[[True, 20], [80, 20], [80, 80], [20, 80]]), "not a number"),
    ("a string coordinate", dict(points=[["20", 20], [80, 20], [80, 80], [20, 80]]), "not a number"),
    ("a point left of the frame", dict(points=[[-1, 20], [80, 20], [80, 80], [20, 80]]), "outside"),
    ("a point below the frame", dict(points=[[20, 20], [80, 20], [80, 100.5], [20, 80]]), "outside"),
    ("the same point twice in a row", dict(points=[[20, 20], [20, 20], [80, 80], [20, 80]]), "same point"),
    ("the last point equal to the first", dict(points=SQ + [[20, 20]]), "same point"),
    ("an outline that crosses itself (a bow tie)", dict(points=[[20, 20], [80, 80], [80, 20], [20, 80]]), "crosses itself"),
    ("an outline that doubles back on itself (collinear overlap)", dict(points=[[10, 50], [90, 50], [50, 50], [50, 90]]), "crosses itself"),
    ("a sliver under 1% of the frame", dict(points=[[20, 20], [80, 20], [80, 21]]), "under 1%"),
    ("a zero-area line", dict(points=[[10, 10], [50, 50], [90, 90]]), ""),
    ("a width given as 100.0", dict(width=100.0), "whole number"),
    ("a width given as true", dict(width=True), "whole number"),
    ("a zero height", dict(height=0), "whole number"),
    ("a frame larger than 4K", dict(width=16000, height=9000, points=[[0, 0], [16000, 0], [16000, 9000]]), "larger than"),
    ("an uppercase fingerprint", dict(video_fp="A" * 64), "sha256"),
    ("a 63-character fingerprint", dict(video_fp="a" * 63), "sha256"),
    ("a fingerprint that is a number", dict(video_fp=7), "sha256"),
    ("a negative click time", dict(click_s=-1), "click_s"),
    ("a click time over a day", dict(click_s=ring.MAX_CLICK_S + 1), "click_s"),
    ("a click time given as text", dict(click_s="12"), "not a number"),
    ("a click time given as true", dict(click_s=True), "not a number"),
]
for name, over, pat in CASES:
    raises(f"a floor with {name}", lambda o=over: ring.load_floors(write_doc("case.json", floors_doc({"GH1": entry(**o)}))), pat)
big_int = "1" + "0" * 400
raises("ASTRA 1.7: an integer too large for a float (10**400) is a ValueError, not an OverflowError",
       lambda: ring.load_floors(write_doc("bigint.json", raw=json.dumps(floors_doc({"GH1": entry()})).replace("12.5", big_int).encode())),
       "not finite")
raises("ASTRA 1.7: ... also as a coordinate",
       lambda: ring.load_floors(write_doc("bigint2.json", raw=json.dumps(floors_doc({"GH1": entry(points=[[1, 1], [80, 20], [80, 80]])}))
                                          .replace("[1, 1]", f"[{big_int}, 1]").encode())), "not finite")
print("floor file: every field x 8 wrong-type values (LESSONS 2026-10-25: hand-picked refusals miss what == forgives)")
WRONG = [("a list", [1]), ("an object", {"a": 1}), ("null", None), ("text", "x"), ("10**400", 10 ** 400), ("-1", -1),
         ("true", True), ("1.5", 1.5)]


def set_path(doc, path, value):
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return doc


def sweep(name, base_doc, paths, load, legal):
    """Every (field, value) must raise ValueError, except the combinations listed in `legal`, which must load."""
    bad = []
    for path in paths:
        for label, value in WRONG:
            doc = set_path(json.loads(json.dumps(base_doc)), path, value)
            p = tmp / "sweep.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            key = (".".join(map(str, path)), label)
            try:
                load(p)
                if key not in legal:
                    bad.append(f"{key} ACCEPTED")
            except ValueError:
                if key in legal:
                    bad.append(f"{key} refused but legal")
            except Exception as e:
                bad.append(f"{key} raised {type(e).__name__}")
    check(f"{name}: {len(paths)} fields x {len(WRONG)} values, ValueError for every illegal one, never another exception",
          not bad, "; ".join(bad))


FPATHS = [("format",), ("videos",), ("videos", "GH1"), ("videos", "GH1", "points"), ("videos", "GH1", "points", 0),
          ("videos", "GH1", "points", 0, 0), ("videos", "GH1", "points", 0, 1), ("videos", "GH1", "width"),
          ("videos", "GH1", "height"), ("videos", "GH1", "video_fp"), ("videos", "GH1", "click_s")]
sweep("floor file", floors_doc({"GH1": entry()}), FPATHS, ring.load_floors,
      legal={("videos.GH1.click_s", "null"), ("videos.GH1.click_s", "1.5"),
             ("videos.GH1.points.0.0", "1.5"), ("videos.GH1.points.0.1", "1.5")})
raises("check_floor refuses the same things the reader does (the click tool saves through it)",
       lambda: ring.check_floor("GH1", [[20, 20], [80, 80], [80, 20], [20, 80]], 100, 100, FP), "crosses itself")

# ======================================================================== geometry
print("signed distance: positive inside, negative outside, zero on the outline")
sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
sd = ring.signed_distance([[5, 5], [15, 5], [10, 5], [5, 0], [20, 20], [9, 1]], sq)
check("centre +5, outside -5, on an edge 0, on the bottom edge 0, a corner diagonal -14.14, near a corner +1",
      np.allclose(sd, [5, -5, 0, 0, -math.hypot(10, 10), 1]), str(sd))
L = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
check("a concave (L-shaped) floor: the notch is OUTSIDE, 3 px from the outline", np.isclose(ring.signed_distance([[7, 7]], L)[0], -3))
check("... and the arm of the L is inside", ring.signed_distance([[2, 8], [8, 2]], L).min() > 0)
dia = np.array([[5, 0], [10, 5], [5, 10], [0, 5]], float)
sdd = ring.signed_distance([[2, 5], [-1, 5], [5, 5]], dia)
check("a horizontal ray through two vertices (the even-odd edge case) still classifies correctly",
      sdd[0] > 0 and sdd[1] < 0 and sdd[2] > 0, str(sdd))
check("no points in, no distances out", ring.signed_distance(np.zeros((0, 2)), sq).shape == (0,))
raises("ASTRA 3.3: two clicks a sub-pixel apart (1e-200 px) are one point, refused",
       lambda: ring.check_floor("GH1", [[0, 0], [1e-200, 0], [100, 0], [100, 100], [0, 100]], 100, 100, FP), "same point")
zl = np.array([[0, 0], [0, 0], [10, 0], [10, 10], [0, 10]], float)
sdz = ring.signed_distance([[0, 5], [5, 5]], zl)
check("ASTRA 3.3: even a zero-length edge (bypassing the check) gives finite distances, 0 on the outline",
      np.isfinite(sdz).all() and sdz[0] == 0 and sdz[1] == 5, str(sdz))


def tbl(rows, w=100, h=100, fp=""):
    """rows: (frame, track id, feet x, feet y); a 10 x 30 box standing on the feet."""
    k = len(rows)
    fr = np.array([r[0] for r in rows], np.int64)
    ti = np.array([r[1] for r in rows], np.int64)
    box = np.array([[r[2] - 5, r[3] - 30, r[2] + 5, r[3]] for r in rows], np.float32).reshape(k, 4)
    xy = np.zeros((k, N_JOINTS, 2), np.float32)
    xy[:, :, 0], xy[:, :, 1] = box[:, :1], box[:, 1:2]
    return PoseTable(fr, ti, box, xy, np.full((k, N_JOINTS), 0.9, np.float32), max(int(fr.max(initial=0)) + 1, 1), 50.0, w, h,
                     source="GH1.mp4", model="yolo11m-pose.pt", video_fp=fp)


print("ids_outside: whole ids, by the share of rows whose feet stand on the (grown) floor")
floor = ring.check_floor("GH1", SQ, 100, 100, FP)
rows = ([(f, 1, 50, 50) for f in range(4)]                                   # always inside
        + [(f, 2, 50, 95) for f in range(4)]                                 # 15 px below the floor
        + [(0, 3, 50, 50), (1, 3, 50, 50), (2, 3, 5, 5), (3, 3, 5, 5)]       # exactly half inside
        + [(0, 4, 50, 50), (1, 4, 50, 50), (2, 4, 5, 5), (3, 4, 5, 5), (4, 4, 5, 5)]   # 40% inside
        + [(f, 5, 50, 70) for f in range(4)])                                # 10 px inside the edge
t = tbl(rows)
check("margin 0: the id 15 px outside and the 40%-inside id are dropped; the exactly-half id is kept",
      ring.ids_outside(t, floor, 0) == {2, 4}, str(ring.ids_outside(t, floor, 0)))
check("margin 15 grows the floor: the id exactly 15 px outside now counts as inside (boundary inclusive)",
      ring.ids_outside(t, floor, 15) == {4})
check("margin -15 shrinks it: the id 10 px inside the edge is now outside too", ring.ids_outside(t, floor, -15) == {2, 4, 5})
check("an empty table drops nothing", ring.ids_outside(tbl([]), floor, 0) == frozenset())
check("feet below the frame are held on its edge (a boxer cut off at the bottom stands on the near floor)",
      ring.feet(tbl([(0, 1, 50, 130)])).tolist() == [[50.0, 100.0]])
raises("a table of another size is refused", lambda: ring.ids_outside(tbl(rows, w=200), floor, 0), "floor drawn on")
raises("a table of another recording (fingerprint) is refused", lambda: ring.ids_outside(tbl(rows, fp="b" * 64), floor, 0), "different recording")
check("a table without a fingerprint (legacy) is accepted", ring.ids_outside(tbl(rows), floor, 0) == {2, 4})
raises("a NaN margin is refused", lambda: ring.ids_outside(t, floor, float("nan")), "not finite")
raises("a boolean margin is refused", lambda: ring.ids_outside(t, floor, True), "not a number")
raises("a share of 0 is refused", lambda: ring.ids_outside(t, floor, 0, 0.0), "(0, 1]")
raises("a share over 1 is refused", lambda: ring.ids_outside(t, floor, 0, 1.5), "(0, 1]")

print("without_ids: removes exactly those ids; everything else is unchanged")
tf = tbl(rows, fp=FP)
ft = ring.without_ids(tf, {2, 4}, "+ringfloor margin=0px")
check("the dropped ids' rows are gone and every other row is identical",
      set(ft.track_id.tolist()) == {1, 3, 5} and len(ft.frame) == len(tf.frame) - 9
      and np.array_equal(ft.box, tf.box[~np.isin(tf.track_id, [2, 4])]))
check("frames, rate, size, source and FINGERPRINT are kept; the model string says it was filtered",
      ft.n_frames == tf.n_frames and ft.fps == tf.fps and ft.video_fp == FP and ft.source == tf.source
      and ft.model == "yolo11m-pose.pt +ringfloor margin=0px")
ft.save(tmp / "ft.npz")
check("the filtered table saves and loads through the strict loader", len(PoseTable.load(tmp / "ft.npz").frame) == len(ft.frame))
check("dropping nothing changes nothing", np.array_equal(ring.without_ids(tf, frozenset()).frame, tf.frame))
long_model = PoseTable(tf.frame, tf.track_id, tf.box, tf.xy, tf.conf, tf.n_frames, tf.fps, tf.width, tf.height,
                       source=tf.source, model="m" * 4090, video_fp=FP)
raises("ASTRA 2.6: a note that would push the model string past the loader's 4096-character cap is refused before writing",
       lambda: ring.without_ids(long_model, {2}, "+ringfloor margin=0px"), "4096")
check("... while the same table without a note still filters", len(ring.without_ids(long_model, {2}).frame) == len(tf.frame) - 4)

# ======================================================================== camera stillness
print("camera stillness: phase correlation on pixels, no labels")
rng = np.random.default_rng(1)
tex = rng.integers(0, 255, (120, 160)).astype(float)
dx, dy = ring.phase_shift(tex, np.roll(tex, (-3, 7), axis=(0, 1)))
check("a known shift of 7 px across and 3 down is recovered (magnitude; the sign is not used)",
      abs(dx) == 7 and abs(dy) == 3, str((dx, dy)))
check("an image against itself: no shift", ring.phase_shift(tex, tex) == (0.0, 0.0))
raises("two images of different sizes are refused", lambda: ring.phase_shift(tex, tex[:-1]), "one size")
rgb = np.repeat(rng.integers(0, 255, (120, 160, 1)).astype(np.uint8), 3, axis=2)
still = [rgb.copy() for _ in range(24)]
m0 = ring.camera_motion(still)
check("24 identical frames: 0 px motion", m0["motion_px"] == 0 and m0["max_px"] == 0 and m0["samples"] == 24)
two_bad = still[:22] + [np.roll(rgb, 20, axis=1)] * 2
m2 = ring.camera_motion(two_bad)
check("two odd frames out of 24 do not condemn a video (90th percentile) but show in the maximum (20 px)",
      m2["motion_px"] == 0 and m2["max_px"] == 20, str(m2))
panned = still[:16] + [np.roll(rgb, 20, axis=1)] * 8
mp = ring.camera_motion(panned)
check("a camera that pans 20 px for the last third is caught", mp["motion_px"] == 20, str(mp))
halves = still[:12] + [np.roll(rgb, 30, axis=1)] * 12
mh = ring.camera_motion(halves)
check("REGRESSION (GH208416): a camera in two positions for half the video each is caught, not averaged to 0 px",
      mh["motion_px"] == 30, str(mh))
raises("fewer frames than the starting reference needs is refused", lambda: ring.camera_motion(still[:2]), "at least 3")
sq_frames = []
for k in range(24):
    f = np.zeros((120, 160, 3), np.uint8)
    x = (20, 60, 100)[k % 3]
    f[50:70, x:x + 20] = 255
    sq_frames.append(f)
msq = ring.camera_motion(sq_frames)
check("ASTRA 4.4: a textureless starting reference (a white square moving over black) is UNMEASURABLE and reads as the "
      "largest move, never as 0 px", msq["motion_px"] > olympic_ring.STILL_MAX_PX and msq["motion_px"] == math.hypot(120, 160), str(msq))
flat_one = still[:23] + [np.zeros_like(rgb)]
check("one black frame among 23 textured ones is one unmeasurable sample, which the 90th percentile tolerates",
      ring.camera_motion(flat_one)["motion_px"] == 0 and ring.camera_motion(flat_one)["max_px"] == math.hypot(120, 160))
moved = np.roll(rgb, 25, axis=1)
early2 = [rgb, rgb] + [moved] * 22
check("a camera that moves after the second sample (the first ~8% elsewhere) is caught: 22 of 24 frames off the start",
      24 <= ring.camera_motion(early2)["motion_px"] <= 26 > olympic_ring.STILL_MAX_PX, str(ring.camera_motion(early2)))
early1 = [rgb] + [moved] * 23
check("ASTRA 1.4, the stated limit: a position held for ONE sample of 24 (~4%) passes unseen, wherever it is",
      ring.camera_motion(early1)["motion_px"] == 0, str(ring.camera_motion(early1)))
blob = []
for k in range(9):
    f = rgb.copy()
    f[10:30, 10 + 15 * k:30 + 15 * k] = 255
    blob.append(f)
check("the median background removes a person who moves", np.array_equal(ring.background(blob), rgb))
raises("background of no frames is refused", lambda: ring.background([]), "no frames")

# ======================================================================== olympic_ring.py: small parts
print("olympic_ring.py: pre-registered numbers are the gate document's")
check("the stillness bar, the margin grid, the dev cap and every gate bar are as registered",
      olympic_ring.STILL_MAX_PX == 20.0 and olympic_ring.MARGIN_GRID == (-30.0, -15.0, 0.0, 15.0, 30.0, 60.0)
      and olympic_ring.DEV_MAX_DROPPED_CREDITED == 0.005 and olympic_ring.G1_MAX_RATIO == 0.90
      and olympic_ring.G1_MAX_UPPER == 1.00 and olympic_ring.G2_MIN_REMOVED == 0.20 and olympic_ring.G3A_MAX_DROPPED == 0.01
      and olympic_ring.G3A_MAX_DROPPED_VIDEO == 0.05 and olympic_ring.G3B_MAX_FALL == 0.01
      and (olympic_ring.BOOT_N, olympic_ring.BOOT_SEED) == (2000, 20261027) and ring.MIN_INSIDE_SHARE == 0.5
      and ring.STILL_SAMPLES == 24 and ring.STILL_QUANTILE == 90 and ring.STILL_REF_SAMPLES == 3)
gate_doc = (ROOT / "docs" / "gates" / "ring-floor-filter.md").read_text(encoding="utf-8")
check("the gate document states the same bars",
      all(s in gate_doc for s in ("<= **0.90**", "< **1.00**", ">= **20%**", "<= **1.0%** pooled", "<= **5%** in every video",
                                  "- **1.0 pp**", "`motion_px > 20`", "{-30, -15, 0, 15, 30, 60}", "at\n  most 0.5%")),
      "a bar in the code is not in docs/gates/ring-floor-filter.md")

print("olympic_ring.py: click helpers")
check("a click maps back to frame pixels by the shrink factor", olympic_ring.display_to_frame(100, 50, 2, 1920, 1080) == (200.0, 100.0))
check("a click within 10 px of the left/top edge snaps onto it", olympic_ring.display_to_frame(8, 3, 2, 1920, 1080) == (0.0, 0.0))
check("a click within 10 px of the right/bottom edge snaps onto it", olympic_ring.display_to_frame(955, 535, 2, 1920, 1080) == (1920.0, 1080.0))
check("a click past the image is held inside the frame", olympic_ring.display_to_frame(2000, -50, 1, 1920, 1080) == (1920.0, 0.0))
check("development videos are clicked first", olympic_ring.click_order(["GH088416", "GH159681", "GH078416", "GH199681"])[:2]
      == ["GH078416", "GH159681"])

print("olympic_ring.py: where derived files may go")
raises("inside the repo but not under data/ or results/ (docs/) is refused",
       lambda: olympic_ring._inside_repo_ok(ROOT / "docs" / "x.json"), "never elsewhere")
raises("the repo root itself is refused", lambda: olympic_ring._inside_repo_ok(ROOT), "never elsewhere")
raises("a path that climbs out of results/ and back into src/ is refused",
       lambda: olympic_ring._inside_repo_ok(ROOT / "results" / ".." / "src" / "x.json"), "never elsewhere")
for okp in (ROOT / "results" / "x.json", ROOT / "data" / "olympic_ring" / "floors.json", tmp / "x.json"):
    try:
        olympic_ring._inside_repo_ok(okp)
        check(f"allowed: {okp.relative_to(ROOT) if ROOT in okp.parents else 'outside the repo'}", True)
    except ValueError as e:
        check(f"allowed: {okp}", False, str(e))
import os  # noqa: E402

link_dir = ROOT / "docs" / ".ring-test-link"          # removed below, whatever happens
link_name = link_dir / "leak.json"
try:
    link_dir.mkdir(exist_ok=True)
    try:
        os.symlink(tmp / "outside.json", link_name)
        made = True
    except OSError:                                    # Windows without developer mode cannot make links
        made = False
    if made:
        raises("ASTRA 3.6: a real link planted at docs/...json pointing outside the repo is refused (the name is checked)",
               lambda: olympic_ring._inside_repo_ok(link_name), "never elsewhere")
    else:
        print("  note  no symlink privilege here: the link cases below are simulated through Path.resolve")
finally:
    if link_name.is_symlink() or link_name.exists():
        link_name.unlink()
    if link_dir.exists():
        link_dir.rmdir()

# Links simulated without privilege (ASTRA 4.5): a Path whose resolve() follows a planted link.
LINKS = {}


class _LinkedPath(type(Path())):
    def resolve(self, strict=False):
        return LINKS.get(str(self), super().resolve(strict))


real_path = olympic_ring.Path
olympic_ring.Path = _LinkedPath
try:
    LINKS.clear()
    LINKS[str(ROOT / "docs" / "leak.json")] = tmp / "outside.json"            # a file link: docs -> outside
    raises("ASTRA 3.6 / 4.5: a FILE link at docs/leak.json pointing outside is refused (the old full-resolve-only check passed it)",
           lambda: olympic_ring._inside_repo_ok(_LinkedPath(ROOT / "docs" / "leak.json")), "never elsewhere")
    LINKS.clear()
    LINKS[str(ROOT / "results" / "jn" / "leak.json")] = ROOT / "docs" / "leak.json"   # a folder junction: results/jn -> docs
    raises("ASTRA 4.2: a path under results/ that resolves into docs/ (a junction) is refused",
           lambda: olympic_ring._inside_repo_ok(_LinkedPath(ROOT / "results" / "jn" / "leak.json")), "never elsewhere")
finally:
    olympic_ring.Path = real_path
    LINKS.clear()
raises("ASTRA 4.2: an output folder 'results/..' (the repo root) is refused",
       lambda: olympic_ring._inside_repo_ok(Path("results") / ".."), "not '..'")
raises("... and so is a bare '.'", lambda: olympic_ring._inside_repo_ok(Path(".")), "")
raises("write_png refuses a float image", lambda: olympic_ring.write_png(tmp / "x.png", np.zeros((4, 4, 3))), "uint8 RGB")
raises("write_png refuses a grey image", lambda: olympic_ring.write_png(tmp / "x.png", np.zeros((4, 4), np.uint8)), "uint8 RGB")

print("olympic_ring.py: the stillness file is checked when read back")
SE = {"video_fp": FP, "width": 100, "height": 100, "motion_px": 1.5, "max_px": 3.0, "samples": 24}
sd_ok = write_doc("still.json", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE}})
check("a valid stillness file loads", olympic_ring.load_stillness(sd_ok)["GH1"]["motion_px"] == 1.5)
for name, doc in [("the wrong format", {"format": "x", "videos": {}}),
                  ("an extra key", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE | {"x": 1}}}),
                  ("a bad fingerprint", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE | {"video_fp": "zz"}}}),
                  ("a negative motion", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE | {"motion_px": -1}}}),
                  ("a motion given as text", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE | {"motion_px": "1"}}}),
                  ("a bad video id", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"../x": SE}})]:
    raises(f"a stillness file with {name}", lambda d=doc: olympic_ring.load_stillness(write_doc("s.json", d)))
for name, raw in [("ASTRA 1.2: motion_px twice (the second, 0, would win)",
                   ('{"format": "%s", "videos": {"GH1": {"video_fp": "%s", "width": 100, "height": 100, "motion_px": 100, '
                    '"max_px": 100, "samples": 24, "motion_px": 0}}}' % (olympic_ring.STILLNESS_FORMAT, FP)).encode()),
                  ("a video id twice", ('{"format": "%s", "videos": {"GH1": %s, "GH1": %s}}'
                                        % (olympic_ring.STILLNESS_FORMAT, json.dumps(SE), json.dumps(SE))).encode()),
                  ("ASTRA 1.7: motion_px of 10**400", json.dumps({"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE}})
                   .replace("1.5", big_int).encode())]:
    raises(f"a stillness file with {name}", lambda r=raw: olympic_ring.load_stillness(write_doc("sd.json", raw=r)))
for name, over in [("ASTRA 1.2: 1 sample (the rule measures 24)", {"samples": 1}), ("samples given as true", {"samples": True}),
                   ("ASTRA 1.2: motion above its own maximum", {"motion_px": 30, "max_px": 0})]:
    raises(f"a stillness file with {name}",
           lambda o=over: olympic_ring.load_stillness(write_doc("so.json", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE | o}})))
SPATHS = [("format",), ("videos",), ("videos", "GH1")] + [("videos", "GH1", k) for k in sorted(olympic_ring.STILL_KEYS)]
sweep("stillness file", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE}}, SPATHS, olympic_ring.load_stillness,
      legal={("videos.GH1.motion_px", "1.5"), ("videos.GH1.max_px", "1.5")})
raises("ASTRA 2.5: a stillness entry for a 16384x16384 frame (no video this project reads)",
       lambda: olympic_ring.load_stillness(write_doc("sbig.json", {"format": olympic_ring.STILLNESS_FORMAT,
                                                                   "videos": {"GH1": SE | {"width": 16384, "height": 16384}}})), "larger than")
raises("ASTRA 2.5: samples written as 24.0 (prepare writes a whole number)",
       lambda: olympic_ring.load_stillness(write_doc("s24.json", raw=json.dumps({"format": olympic_ring.STILLNESS_FORMAT, "videos": {"GH1": SE}})
                                                     .replace('"samples": 24', '"samples": 24.0').encode())), "whole number")
raises("a stillness file with an extra top-level key",
       lambda: olympic_ring.load_stillness(write_doc("sx.json", {"format": olympic_ring.STILLNESS_FORMAT, "videos": {}, "x": 1})))
raises("a stillness file with NaN",
       lambda: olympic_ring.load_stillness(write_doc("sn.json", raw=json.dumps({"format": olympic_ring.STILLNESS_FORMAT,
                                                                                 "videos": {"GH1": SE}}).replace("1.5", "NaN").encode())))

print("olympic_ring.py: margin selection (development videos only)")
S = lambda m, lost, rem: {"margin_px": m, "credited_dropped_share": lost, "noncredited_removed_share": rem}  # noqa: E731
check("the margin removing the most non-boxers among those losing <= 0.5% of credited punches",
      olympic_ring.select_margin([S(-30, 0.02, 0.6), S(0, 0.004, 0.5), S(30, 0.001, 0.4)]) == 0)
check("ties go to the larger (more forgiving) margin", olympic_ring.select_margin([S(0, 0.0, 0.5), S(15, 0.0, 0.5)]) == 15)
check("exactly 0.5% lost still qualifies", olympic_ring.select_margin([S(0, 0.005, 0.5)]) == 0)
check("no margin qualifies: None (the experiment stops)", olympic_ring.select_margin([S(0, 0.01, 0.5), S(60, 0.006, 0.1)]) is None)
check("a margin with no credited punches at all (no measurement) never qualifies", olympic_ring.select_margin([S(0, None, 0.5)]) is None)

print("olympic_ring.py: the video bootstrap of the rate ratio")
base = [(10, 1.0), (20, 2.0), (5, 1.0), (40, 3.0)]
lo, hi = olympic_ring.bootstrap_ratio(base, base)
check("identical arms: the ratio is exactly 1 in every resample", lo == 1.0 and hi == 1.0)
half = [(u / 2, e) for u, e in base]
lo, hi = olympic_ring.bootstrap_ratio(base, half)
check("half the unexplained bursts in every video: 0.5 in every resample", math.isclose(lo, 0.5) and math.isclose(hi, 0.5))
mixed = [(5, 1.0), (20, 2.0), (5, 1.0), (40, 3.0)]
r1, r2 = olympic_ring.bootstrap_ratio(base, mixed), olympic_ring.bootstrap_ratio(base, mixed)
check("the same seed gives the same interval, and a mixed effect gives an interval around the point ratio",
      r1 == r2 and r1[0] < (70 / 7) / (75 / 7) <= r1[1] <= 1.0, str(r1))
raises("unpaired lists are refused", lambda: olympic_ring.bootstrap_ratio(base, base[:2]), "paired")
raises("empty lists are refused", lambda: olympic_ring.bootstrap_ratio([], []), "paired")
zero_f = [(1, 0.0)] * 4
check("a filtered arm with no exposure gives an infinite (failing) bound, not a crash",
      olympic_ring.bootstrap_ratio(base, zero_f)[1] == math.inf)

print("olympic_ring.py: the verdict needs every gate bar; secondary bars never change it")
ARM = {"linked_unexplained_per_exposure_min": 40.0, "credited_path_unexplained_per_exposure_min": 45.0,
       "credited_share": 0.53, "link_coverage": 0.65, "ab_noncredited_share": 0.3, "median_both_identified": 0.57}
PASS_S = {"videos": 23, "videos_compared": 23, "ratio": 0.8, "ratio_ci95": (0.7, 0.95), "noncredited_removed_share": 0.3, "credited_dropped_share": 0.005,
          "credited_dropped_share_worst_video": 0.02, "newly_failed": [], "baseline": dict(ARM),
          "filtered": ARM | {"linked_unexplained_per_exposure_min": 32.0, "ab_noncredited_share": 0.2}}
check("all bars met: PASS", olympic_ring.verdict(PASS_S)["pass"] is True)
FAILS = [("G1: ratio 0.91", {"ratio": 0.91}), ("G1: upper bound 1.00 (not below)", {"ratio_ci95": (0.7, 1.0)}),
         ("G2: removes 19.9% of non-boxers", {"noncredited_removed_share": 0.199}),
         ("G2: no measurement", {"noncredited_removed_share": None}),
         ("G3a: loses 1.01% of credited punches", {"credited_dropped_share": 0.0101}),
         ("G3a: one video loses 5.1%", {"credited_dropped_share_worst_video": 0.051}),
         ("G3b: credited share falls 1.1 pp", {"filtered": PASS_S["filtered"] | {"credited_share": 0.519}}),
         ("G3c: a video fails only when filtered", {"newly_failed": ["GH088416"]})]
for name, over in FAILS:
    check(f"FAIL when {name}", olympic_ring.verdict(PASS_S | over)["pass"] is False)
sec_bad = PASS_S | {"filtered": PASS_S["filtered"] | {"link_coverage": 0.5, "ab_noncredited_share": 0.4,
                                                      "credited_path_unexplained_per_exposure_min": 50.0,
                                                      "median_both_identified": 0.4}}
vb = olympic_ring.verdict(sec_bad)
check("every secondary bar failing still leaves a PASS (they are reported, not gating)",
      vb["pass"] is True and not any(vb["secondary"].values()), str(vb))

# ======================================================================== end to end on a planted dataset
print("end to end: a planted dataset named like a development video")
Wv, Hv, FPS, N = 640, 360, 50.0, 1000


def pose(cx, cy, ext=(0.0, 0.0)):
    xy = np.tile([cx, cy], (N_JOINTS, 1)).astype(float)
    xy[:, 0] += np.linspace(-6, 6, N_JOINTS)
    xy[5], xy[6] = (cx - 12, cy - 20), (cx + 12, cy - 20)
    xy[11], xy[12] = (cx - 8, cy + 20), (cx + 8, cy + 20)
    xy[9], xy[10] = (cx - 24 - ext[0], cy - 4), (cx + 24 + ext[1], cy - 4)
    return xy, np.array([cx - 28, cy - 52, cx + 28, cy + 56], float)


PEOPLE = {   # name: (path, fragments, wrist bursts (frame, hand))
    "A": (lambda f: (230 + 40 * np.sin(2 * np.pi * f / 200), 200.0), [(1, 0, 199), (2, 215, 420), (3, 431, 640), (4, 655, 999)], [(100, 1)]),
    "B": (lambda f: (420 + 40 * np.sin(2 * np.pi * f / 230 + 1.0), 190.0), [(11, 0, 139), (12, 150, 399), (13, 420, 700), (14, 712, 999)], []),
    "ref": (lambda f: (325 + 50 * np.sin(f / 150), 215.0), [(21, 0, 499), (22, 520, 999)], []),
    # spectators behind the ring: feet above the far edge of the floor; one waves (an unexplained burst if absorbed)
    "spec1": (lambda f: (60.0, 60.0), [(31, 0, 999)], [(300, 0), (600, 1)]),
    "spec2": (lambda f: (600.0, 60.0), [(32, 0, 999)], []),
}


def ext(name, f):
    e = [0.0, 0.0]
    for b, hand in PEOPLE[name][2]:
        e[hand] = max(e[hand], 120.0 * max(0.0, 1 - abs(f - b) / 4))
    return tuple(e)


def scene_table(source, fp=""):
    rows = []
    for name, (path, frags, _) in PEOPLE.items():
        for tid, f0, f1 in frags:
            for f in range(f0, f1 + 1):
                xy, box = pose(*path(f), ext(name, f))
                rows.append((f, tid, box, xy))
    rows.sort(key=lambda r: (r[0], r[1]))
    k = len(rows)
    return PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                     np.array([r[2] for r in rows], np.float32).reshape(k, 4),
                     np.array([r[3] for r in rows], np.float32).reshape(k, N_JOINTS, 2),
                     np.full((k, N_JOINTS), 0.9, np.float32), N, FPS, Wv, Hv, source=source, video_fp=fp)


def punches(table):
    """Labels on the two boxers' wrists, every 25 frames, away from the planted bursts and from other wrists."""
    out = []
    for name in ("A", "B"):
        path, frags, _ = PEOPLE[name]
        for tid, f0, f1 in frags:
            got = 0
            for f in range(f0 + 10, f1 - 10, 25):
                if got >= 4 or any(abs(f - b) < 15 for q in PEOPLE.values() for b, _ in q[2]):
                    continue
                hand = got % 2
                w = pose(*path(f), ext(name, f))[0][9 + hand]
                out.append(olympic.Event(f, f + 2, "head", ("left", "right")[hand], (w[0] - 8, w[1] - 8, w[0] + 8, w[1] + 8), 1))
                got += 1
    return out


pl = {v: k for k, v in olympic.TARGETS.items()}
ph = {v: k for k, v in olympic.HANDS.items()}


def cvat(evs):
    tracks = []
    for e in evs:
        x1, y1, x2, y2 = e.box
        shp = lambda frame, out: {"type": "rectangle", "frame": frame, "outside": out, "points": [x1, y1, x2, y2]}  # noqa: E731
        tracks.append({"label": f"{pl[e.target]} {ph[e.hand]}", "shapes": [shp(e.first, False), shp(e.last + 1, True)]})
    return [{"tracks": tracks}]


def make_video(path, pan_from=None, seed=7):
    """A static random texture (one per seed: distinct recordings); with pan_from, the camera pans 60 px from then on."""
    tex = np.repeat(np.random.default_rng(seed).integers(0, 255, (Hv, Wv + 80, 1)).astype(np.uint8), 3, axis=2)
    from fractions import Fraction
    with VideoWriter(path, Wv, Hv, Fraction(50)) as vw:
        for f in range(N):
            off = 60 if pan_from is not None and f >= pan_from else 0
            vw.write(np.ascontiguousarray(tex[:, off:off + Wv]))
    return path


ds = tmp / "ds"
trk = tmp / "tracks"
trk.mkdir()
STILL_DEV = ["GH078416", "GH128416", "GH159681"]      # the panning one, GH069681, is the fourth development video
for task_name, vid, pan, seed in (("task_kam2_gh078416", "GH078416", None, 7), ("task_kam4_gh069681", "GH069681", 500, 8),
                                  ("task_kam2_gh128416", "GH128416", None, 9), ("task_kam4_gh159681", "GH159681", None, 10)):
    d = ds / task_name / "data"
    d.mkdir(parents=True)
    v = make_video(d / f"{vid}.mp4", pan, seed)
    tb = scene_table(f"{vid}.mp4", fingerprint(v))
    tb.save(trk / f"{vid}.tracks.npz")
    (ds / task_name / "annotations.json").write_text(json.dumps(cvat(punches(tb))), encoding="utf-8")
check("setup: the four development ids, each a whole decodable video of N frames and a distinct recording",
      len(olympic.find_tasks(ds)) == 4 and all(probe(t.video).timing.n_frames == N for t in olympic.find_tasks(ds))
      and len({fingerprint(t.video) for t in olympic.find_tasks(ds)}) == 4)

prep = tmp / "prep"
with contextlib.redirect_stdout(io.StringIO()) as out_prep:
    rc_p = olympic_ring.main(["prepare", "--data", str(ds), "--out-dir", str(prep)])
st = olympic_ring.load_stillness(prep / "stillness.json")
check("prepare: exit 0, a background PNG per video, and a stillness entry bound to each video's fingerprint",
      rc_p == 0 and (prep / "GH078416.background.png").stat().st_size > 0 and (prep / "GH069681.background.png").is_file()
      and st["GH078416"]["video_fp"] == fingerprint(ds / "task_kam2_gh078416" / "data" / "GH078416.mp4"), out_prep.getvalue())
check("REAL FACT: the still camera measures under the 20 px bar; the one that pans 60 px for half the video, over it",
      st["GH078416"]["motion_px"] <= 20 < st["GH069681"]["motion_px"], str({k: v["motion_px"] for k, v in st.items()}))
raises("prepare refuses to write derived images into docs/",
       lambda: olympic_ring.cmd_prepare(type("A", (), {"out_dir": ROOT / "docs" / "ring", "data": ds, "only": None})()),
       "never elsewhere")

print("the click window, driven on a hidden Tk root (what Brendan's clicks do)")
import tkinter as tk  # noqa: E402

root = tk.Tk()
root.withdraw()
gui_floors = tmp / "gui" / "floors.json"
ticks = iter([100.0, 107.25, 200.0, 260.0, 300.0])
fc = olympic_ring.FloorClicker(root, ["GH078416", "GH069681"], st, {}, gui_floors, prep, clock=lambda: next(ticks))
check("the first video is shown at full size on a wide screen (640 px image, scale 1)", fc.video == "GH078416" and fc.scale == 1)
for x, y in [(100, 100), (500, 100), (500, 300), (5, 300)]:
    fc.add(x, y)
check("a click near the left edge snaps onto it", fc.pts[-1] == (0.0, 300.0))
fc.undo()
fc.add(100, 300)
saved = fc.save()
gf = ring.load_floors(gui_floors)
check("Enter saves a valid outline: the file holds it, bound to the video, with the time taken (7.2 s)",
      saved and list(gf) == ["GH078416"] and gf["GH078416"].points.tolist() == [[100, 100], [500, 100], [500, 300], [100, 300]]
      and gf["GH078416"].video_fp == st["GH078416"]["video_fp"] and gf["GH078416"].click_s == 7.2, str(gf))
check("... and the next video comes up with no points", fc.video == "GH069681" and fc.pts == [])
for x, y in [(100, 100), (500, 300), (500, 100), (100, 300)]:
    fc.add(x, y)
check("a crossing outline is NOT saved: a red reason, the file unchanged, the same video still up",
      fc.save() is False and "crosses itself" in fc.status.cget("text") and list(ring.load_floors(gui_floors)) == ["GH078416"]
      and fc.video == "GH069681")
fc.skip()
check("Esc skips the video; after the last one the window is done and the file still holds one floor",
      fc.done and fc.video is None and list(ring.load_floors(gui_floors)) == ["GH078416"])
check("actions after the last video do nothing", fc.save() is False and fc.add(1, 1) is None)
root.destroy()

# The floor: the canvas from y = 120 down to the bottom, nearly full width. Boxers' and referee's feet are at
# y 246-271; the spectators' at y 116 (4 px above the far edge).
FLOOR_PTS = [[20, 120], [620, 120], [640, 360], [0, 360]]
floors_p = tmp / "floors.json"
ring.dump_floors({vid: ring.check_floor(vid, FLOOR_PTS, Wv, Hv, st[vid]["video_fp"], 30.0) for vid in sorted(st)}, floors_p)
rule_f = tmp / "rule.json"
rule_f.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": 3.62, "min_gap_s": 0.4}}), encoding="utf-8")
rule_reg = tmp / "rule_registered.json"
rule_reg.write_text(json.dumps({"chosen": olympic_ring.REGISTERED_RULE}), encoding="utf-8")
check("the registered rule is the tuned V6 rule rows 7c/7d use, as load_rule returns it",
      __import__("find_punches").load_rule(rule_reg) == {"variant": "relative", "threshold": 3.6238817996457016, "min_gap_s": 0.4})
ev_args = ["evaluate", "--data", str(ds), "--tracks", str(trk), "--floors", str(floors_p), "--prep-dir", str(prep),
           "--rule", str(rule_f)]
dev_out = tmp / "res" / "dev.json"
with contextlib.redirect_stdout(io.StringIO()) as o, contextlib.redirect_stderr(io.StringIO()) as e_:
    rc_d = olympic_ring.main(ev_args + ["--split", "dev", "--out", str(dev_out)])
if rc_d != 0:
    print(o.getvalue(), e_.getvalue())
dev = json.loads(dev_out.read_text(encoding="utf-8"))
check("dev: exit 0; the panning video is LEFT OUT (both arms) and listed; the three still ones are evaluated",
      rc_d == 0 and dev["left_out_camera_moves"] == ["GH069681"] and sorted(dev["videos"]) == STILL_DEV
      and dev["rule"]["threshold"] == 3.62, o.getvalue())
by_m = {s["margin_px"]: s for s in dev["margins"]}
check("REAL FACT: at margin 0 the two spectators (and nobody else) are dropped: every non-boxer row of theirs is removed",
      by_m[0.0]["credited_dropped"] == 0 and by_m[0.0]["noncredited_rows"] == 3 * (2 * N + 500 + 480)  # 2 spectators + referee, x3
      and math.isclose(by_m[0.0]["noncredited_removed_share"], 2 * N / (2 * N + 980), rel_tol=1e-9), str(by_m[0.0]))
check("REAL FACT: at margin +15 the spectators 4 px behind the edge count as on the floor: nothing removed",
      by_m[15.0]["noncredited_removed_share"] == 0.0)
check("the dev rule picks the most removal with no boxer lost: 0 px (ties between -30, -15 and 0 go to the larger)",
      dev["selected_margin_px"] == 0.0, str(dev["selected_margin_px"]))

# The same code: the evaluator's baseline is what olympic_completeness.py says about the same files.
with contextlib.redirect_stdout(io.StringIO()):
    olympic_completeness.main(["--data", str(ds), "--tracks", str(trk), "--only", *STILL_DEV, "--rule", str(rule_f),
                               "--out", str(tmp / "res" / "cl.json"), "--sample-out", str(tmp / "res" / "cl.csv")])
    olympic_completeness.main(["--data", str(ds), "--tracks", str(trk), "--only", *STILL_DEV, "--rule", str(rule_f), "--credited",
                               "--out", str(tmp / "res" / "cc.json"), "--sample-out", str(tmp / "res" / "cc.csv")])
cli_l = json.loads((tmp / "res" / "cl.json").read_text(encoding="utf-8"))["all"]
cli_c = json.loads((tmp / "res" / "cc.json").read_text(encoding="utf-8"))["all"]
b0 = by_m[0.0]["baseline"]
check("REAL FACT (gate self-check): the evaluator's baseline linked rate equals olympic_completeness.py's, to its 3 decimals",
      round(b0["linked_unexplained_per_exposure_min"], 3) == cli_l["unexplained_per_exposure_min"]
      and b0["linked_unexplained"] == cli_l["unexplained"], f"{b0} vs {cli_l['unexplained_per_exposure_min']}")
check("REAL FACT (gate self-check): ... and its credited-fragment rate equals olympic_completeness.py --credited's",
      round(b0["credited_path_unexplained_per_exposure_min"], 3) == cli_c["unexplained_per_exposure_min"],
      f"{b0['credited_path_unexplained_per_exposure_min']} vs {cli_c['unexplained_per_exposure_min']}")

# Filtered tables on disk, then the same CLI: equal to the evaluator's filtered arm.
filt_dir = tmp / "filtered"
with contextlib.redirect_stdout(io.StringIO()):
    rc_w = olympic_ring.main(ev_args + ["--split", "dev", "--margin", "0", "--out", str(tmp / "res" / "d0.json"),
                                        "--write-filtered", str(filt_dir)])
    olympic_completeness.main(["--data", str(ds), "--tracks", str(filt_dir), "--only", *STILL_DEV, "--rule", str(rule_f),
                               "--out", str(tmp / "res" / "fl.json"), "--sample-out", str(tmp / "res" / "fl.csv")])
fw = PoseTable.load(filt_dir / "GH078416.tracks.npz")
cli_f = json.loads((tmp / "res" / "fl.json").read_text(encoding="utf-8"))["all"]
f0 = json.loads((tmp / "res" / "d0.json").read_text(encoding="utf-8"))["margins"][0]["filtered"]
check("the written filtered table lacks exactly the spectators, keeps the fingerprint, and says it was filtered",
      rc_w == 0 and set(np.unique(fw.track_id).tolist()) == {1, 2, 3, 4, 11, 12, 13, 14, 21, 22}
      and fw.video_fp == fingerprint(ds / "task_kam2_gh078416" / "data" / "GH078416.mp4") and "+ringfloor margin=0px" in fw.model)
check("REAL FACT: olympic_completeness.py on the written filtered tables equals the evaluator's filtered arm",
      round(f0["linked_unexplained_per_exposure_min"], 3) == cli_f["unexplained_per_exposure_min"],
      f"{f0['linked_unexplained_per_exposure_min']} vs {cli_f['unexplained_per_exposure_min']}")
check("the planted boxers keep every credited punch (the floor holds them)",
      math.isclose(f0["credited_share"], b0["credited_share"]) and b0["credited_share"] > 0.9, str((b0["credited_share"], f0["credited_share"])))

print("pooling: every metric is over the same videos (a baseline failure is in no arm)")
import copy  # noqa: E402

from find_punches import load_rule  # noqa: E402

task78 = next(t for t in olympic.find_tasks(ds) if t.video.stem == "GH078416")
real = olympic_ring.evaluate_video(task78, trk, ring.load_floors(floors_p)["GH078416"], (0.0,), load_rule(rule_f))
broken = copy.deepcopy(real)
broken["video"] = "GH128416"
broken["base"] = {"events": 10, "credited": 10, "rows": 1, "noncredited_rows": 1, "credited_path_error": "planted"}
broken["margins"][0.0].update(credited=10, credited_dropped=10, noncredited_rows=999, noncredited_rows_dropped=0)
sm = olympic_ring.summarise_margin([real, broken], 0.0)
solo = olympic_ring.summarise_margin([real], 0.0)
check("a video whose baseline fails is listed and left out of G2 and G3a too, not only of G1",
      sm["baseline_failed"] == ["GH128416"] and sm["credited_dropped_share"] == solo["credited_dropped_share"] == 0.0
      and sm["credited_dropped_share_worst_video"] == 0.0 and sm["noncredited_removed_share"] == solo["noncredited_removed_share"]
      and sm["videos_compared"] == 1 and sm["ratio"] == solo["ratio"], str({k: sm[k] for k in ("baseline_failed", "credited_dropped_share")}))
newly = copy.deepcopy(real)
newly["video"] = "GH138416"
newly["margins"][0.0]["arm"]["linked_error"] = "planted"
newly["margins"][0.0].update(credited_dropped=3)
sn = olympic_ring.summarise_margin([real, newly], 0.0)
killed = copy.deepcopy(real)
killed["video"] = "GH159681"
killed["margins"][0.0]["arm"]["credited"] = 0                      # every credited person removed ...
killed["margins"][0.0]["arm"]["linked_error"] = "no fragment has a usable body"   # ... so its linker fails
sk = olympic_ring.summarise_margin([real, killed], 0.0)
want = real["margins"][0.0]["arm"]["credited"] / (2 * real["margins"][0.0]["arm"]["events"])
check("ASTRA 4.1: G3b counts a video that fails only when filtered: its lost credits halve the filtered share and G3b FAILS",
      sk["newly_failed"] == ["GH159681"] and math.isclose(sk["filtered"]["credited_share"], want)
      and math.isclose(sk["baseline"]["credited_share"], real["base"]["credited"] / real["base"]["events"])
      and olympic_ring.verdict(sk)["gate"]["G3b_credited_share"] is False, str(sk["filtered"]["credited_share"]))
check("a video that fails only when filtered is G3c's failure AND stays in the filter's cost (G3a)",
      sn["newly_failed"] == ["GH138416"] and sn["credited_dropped"] == 3 and sn["videos_compared"] == 1
      and olympic_ring.verdict(sn)["gate"]["G3c_no_new_failures"] is False)
only_new = olympic_ring.summarise_margin([newly], 0.0)
vn = olympic_ring.verdict(only_new)
check("ASTRA 1.5: every filtered arm failing is a FAIL verdict (G1, G3b unmeasured -> fail; G3c fails), not a crash",
      vn["pass"] is False and vn["gate"]["G3c_no_new_failures"] is False and vn["gate"]["G1_linked_unexplained"] is False
      and vn["gate"]["G3b_credited_share"] is False and vn["secondary"] == {}, str(vn))
zero = copy.deepcopy(real)
for arm_ in (zero["base"], zero["margins"][0.0]["arm"]):
    arm_["linked"]["unexplained"] = 0
    for n_ in "AB":
        arm_["linked"]["boxers"][n_]["unexplained"] = 0
sz = olympic_ring.summarise_margin([zero], 0.0)
check("ASTRA 1.6: a baseline with 0 unexplained bursts gives no ratio and a failing G1, not a ZeroDivisionError",
      sz["ratio"] is None and olympic_ring.verdict(sz)["gate"]["G1_linked_unexplained"] is False, str(sz.get("ratio")))
try:
    json.dumps(sz, allow_nan=False)
    ok_json = sz["ratio_ci95"][1] is None
except ValueError as e:
    ok_json = False
check("ASTRA 2.1: an unbounded bootstrap (zero baseline) is written as null, so the FAIL report serialises", ok_json,
      str(sz.get("ratio_ci95")))
s_empty = olympic_ring.summarise_margin([], 0.0)
v_empty = olympic_ring.verdict(s_empty)
check("ASTRA 2.2: no video to compare is a FAIL on every gate bar, not an IndexError",
      v_empty["pass"] is False and not any(v_empty["gate"].values()) and s_empty["credited_dropped_share"] is None, str(v_empty))
all_base_fail = copy.deepcopy(broken)
all_base_fail["video"] = "GH078416"
v_abf = olympic_ring.verdict(olympic_ring.summarise_margin([all_base_fail], 0.0))["gate"]
check("ASTRA 2.3 / 3.4: with every baseline failed, G3a and G3c (nothing measured) FAIL instead of passing by default",
      v_abf["G3a_boxers_kept"] is False and v_abf["G3c_no_new_failures"] is False, str(v_abf))
check("the local report carries per-video counts, with the linked (unexplained, minutes) of both arms when both ran",
      set(sm["per_video"]) == {"GH078416", "GH128416"} and "linked_ue_filtered" in sm["per_video"]["GH078416"]
      and "linked_ue_baseline" not in sm["per_video"]["GH128416"])

print("end to end: refusals")


def run_quiet(argv):
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        rc = olympic_ring.main(argv)
    return rc, err.getvalue()


rc, err = run_quiet(ev_args + ["--split", "heldout"])
check("held-out is refused while the margin is not frozen (exit 1, says why)", rc == 1 and "not frozen" in err, err)
real_frozen = olympic_ring.FROZEN_MARGIN_PX
olympic_ring.FROZEN_MARGIN_PX = 15.0
try:
    rc, err = run_quiet(ev_args + ["--split", "heldout", "--margin", "0"])
    check("held-out at any margin but the frozen one is refused", rc == 1 and "frozen margin" in err, err)
    rc, err = run_quiet(ev_args + ["--split", "heldout"])
    check("ASTRA 3.2: held-out with any punch rule but the registered one is refused", rc == 1 and "registered punch rule" in err, err)
    rc, err = run_quiet(ev_args[:-1] + [str(rule_reg), "--split", "heldout"])
    check("held-out without the 25 held-out tables is refused (no partial verdict)", rc == 1 and "held-out tables" in err, err)
    # ASTRA 4.3: membership, not a count. The manifest names GH999998; the data offers GH999999 instead.
    ds_sub, trk_sub = tmp / "ds_sub", tmp / "trk_sub"
    shutil.copytree(ds, ds_sub)
    shutil.copytree(trk, trk_sub)
    (ds_sub / "task_kam2_gh999999" / "data").mkdir(parents=True)
    v_sub = make_video(ds_sub / "task_kam2_gh999999" / "data" / "GH999999.mp4", None, 11)
    scene_table("GH999999.mp4", fingerprint(v_sub)).save(trk_sub / "GH999999.tracks.npz")
    shutil.copy(ds / "task_kam2_gh078416" / "annotations.json", ds_sub / "task_kam2_gh999999" / "annotations.json")
    real_manifest = olympic_ring.HELD_OUT_VIDEOS
    olympic_ring.HELD_OUT_VIDEOS = frozenset({"GH999998"})
    try:
        rc, err = run_quiet(["evaluate", "--data", str(ds_sub), "--tracks", str(trk_sub), "--floors", str(floors_p),
                             "--prep-dir", str(prep), "--rule", str(rule_reg), "--split", "heldout"])
    finally:
        olympic_ring.HELD_OUT_VIDEOS = real_manifest
    check("ASTRA 4.3: a substitute recording cannot stand in for a missing held-out video (the manifest, not a count)",
          rc == 1 and "GH999998" in err and "GH999999" in err, err)
finally:
    olympic_ring.FROZEN_MARGIN_PX = real_frozen
check("the held-out manifest is ADR-0013's: the 29 Olympic ids minus the 4 development ones, 25 ids",
      len(olympic_ring.HELD_OUT_VIDEOS) == 25 and not (olympic_ring.HELD_OUT_VIDEOS & olympic_ring.DEV_VIDEOS)
      and olympic_ring.HELD_OUT_VIDEOS == olympic_ring.DEV_VIDEOS.__class__(
          {"GH0%d8416" % c if c < 10 else "GH%d8416" % c for c in range(7, 22)}
          | {"GH0%d9681" % c if c < 10 else "GH%d9681" % c for c in range(6, 20)}) - olympic_ring.DEV_VIDEOS)
rc, err = run_quiet(ev_args + ["--split", "dev", "--margin", "0", "--write-filtered", str(trk)])
check("--write-filtered onto the baseline tables is refused", rc == 1 and "overwrite the baseline" in err, err)
rc, err = run_quiet(ev_args + ["--split", "dev", "--write-filtered", str(tmp / "wf")])
check("--write-filtered over the whole grid is refused (it writes one margin)", rc == 1 and "one margin" in err, err)
rc, err = run_quiet(ev_args + ["--split", "dev", "--out", str(ROOT / "docs" / "leak.json")])
check("an output inside docs/ is refused before anything runs", rc == 1 and "never elsewhere" in err and not (ROOT / "docs" / "leak.json").exists(), err)
only_one = tmp / "floors_one.json"
ring.dump_floors({"GH069681": ring.load_floors(floors_p)["GH069681"]}, only_one)
rc, err = run_quiet(["evaluate", "--data", str(ds), "--tracks", str(trk), "--floors", str(only_one), "--prep-dir", str(prep), "--rule", str(rule_f)])
check("a video without a floor is refused (click it first), never silently skipped", rc == 1 and "GH078416" in err, err)
swapped = tmp / "floors_swapped.json"
fs = ring.load_floors(floors_p)
ring.dump_floors(fs | {"GH078416": ring.check_floor("GH078416", FLOOR_PTS, Wv, Hv, fs["GH128416"].video_fp)}, swapped)
rc, err = run_quiet(["evaluate", "--data", str(ds), "--tracks", str(trk), "--floors", str(swapped), "--prep-dir", str(prep), "--rule", str(rule_f)])
check("a floor clicked on another recording (fingerprint) is refused", rc == 1 and "different recordings" in err, err)
stale_prep = tmp / "prep_stale"
stale_prep.mkdir()
st_doc = json.loads((prep / "stillness.json").read_text(encoding="utf-8"))
st_doc["videos"]["GH069681"]["video_fp"] = "b" * 64        # measured on another recording, and "moving"
(stale_prep / "stillness.json").write_text(json.dumps(st_doc), encoding="utf-8")
rc, err = run_quiet(["evaluate", "--data", str(ds), "--tracks", str(trk), "--floors", str(floors_p), "--prep-dir", str(stale_prep),
                     "--rule", str(rule_f)])
check("ASTRA 1.3: a stale stillness entry (another recording) stops the run BEFORE it can quietly leave a video out",
      rc == 1 and "GH069681" in err and "different recording" in err, err)
moving_prep = tmp / "prep_moving"
moving_prep.mkdir()
mv_doc = json.loads((prep / "stillness.json").read_text(encoding="utf-8"))
for v_ in mv_doc["videos"].values():
    v_["motion_px"] = v_["max_px"] = 100.0
(moving_prep / "stillness.json").write_text(json.dumps(mv_doc), encoding="utf-8")
mv_out = tmp / "res" / "moving.json"
rc, err = run_quiet(ev_args[:8] + [str(moving_prep), "--rule", str(rule_f), "--split", "dev", "--out", str(mv_out)])
mv = json.loads(mv_out.read_text(encoding="utf-8")) if mv_out.is_file() else {}
check("ASTRA 2.2: every video left out still writes a report (no video compared, no margin chosen), not an IndexError",
      rc == 0 and mv.get("videos") == [] and mv.get("left_out_camera_moves") == sorted(st)
      and mv.get("selected_margin_px") is None, err)
trk3 = tmp / "tracks_three"
shutil.copytree(trk, trk3)
(trk3 / "GH159681.tracks.npz").unlink()
rc, err = run_quiet(["evaluate", "--data", str(ds), "--tracks", str(trk3), "--floors", str(floors_p), "--prep-dir", str(prep),
                     "--rule", str(rule_f)])
check("ASTRA 3.1: a development run missing one of the four dev tables is refused (no margin from a subset)",
      rc == 1 and "needs all" in err and "GH159681" in err, err)
DUP_REASON = {"GH078416": "more than one task", "GH128416": "same recording"}
for label, second_task, second_name in (("the same video id in two tasks", "task_kam4_gh078416", "GH078416"),
                                        ("the same recording under two ids", "task_kam2_gh128416", "GH128416")):
    dsd, trd = tmp / f"ds_dup_{second_name}", tmp / f"trk_dup_{second_name}"
    shutil.copytree(ds / "task_kam2_gh078416", dsd / "task_kam2_gh078416")
    (dsd / second_task / "data").mkdir(parents=True)
    shutil.copy(ds / "task_kam2_gh078416" / "data" / "GH078416.mp4", dsd / second_task / "data" / f"{second_name}.mp4")
    shutil.copy(ds / "task_kam2_gh078416" / "annotations.json", dsd / second_task / "annotations.json")
    trd.mkdir()
    for nm in {"GH078416", second_name}:
        shutil.copy(trk / "GH078416.tracks.npz", trd / f"{nm}.tracks.npz")
    rc, err = run_quiet(["evaluate", "--data", str(dsd), "--tracks", str(trd), "--floors", str(floors_p), "--prep-dir", str(prep),
                         "--rule", str(rule_f)])
    check(f"ASTRA 2.4 / 3.7: {label} is refused FOR THAT REASON (it would be pooled twice)",
          rc == 1 and DUP_REASON[second_name] in err, err)
rc, err = run_quiet(["evaluate", "--data", str(ds), "--tracks", str(trk), "--floors", str(tmp / "bin.json"), "--prep-dir", str(prep), "--rule", str(rule_f)])
check("a malformed floor file stops the run with a reason, not a traceback", rc == 1 and "UTF-8 JSON" in err, err)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
