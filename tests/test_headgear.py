"""Tests for the headgear corner colour (headgear.py, kit colour v3) and olympic_headgear.py.

Synthetic only: every frame is painted in memory (a red-vest boxer in red headgear, a red-vest boxer in
blue headgear, a referee, skin faces). No Olympic video and no model; the only pixels written to disk are the
tiny painted clip the CLI test builds. Most checks are negative: the counts file is an input file, and the
self-check must be able to fail.

Run:  python tests/test_headgear.py
"""

from __future__ import annotations

import colorsys
import contextlib
import io
import json
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import olympic_headgear  # noqa: E402
from boxingvi import headgear, olympic, video  # noqa: E402
from boxingvi.appearance import MIN_REGION_PX  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.headgear import HeadgearCounts  # noqa: E402
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


def raises(name: str, fn, pattern: str = "") -> None:
    try:
        fn()
    except ValueError as e:
        check(name, pattern in str(e), f"wrong reason: {e}")
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


tmp = Path(tempfile.mkdtemp())
W, H, FPS = 320, 240, 50.0
BG = (110, 110, 110)
SKIN = (225, 170, 140)
RED, BLUE, WHITE = (220, 30, 30), (30, 60, 230), (245, 245, 245)


def person_xy(cx, cy):
    """COCO-17, torso length 60 at (cx, cy): ears 16 px apart, face centre (cx, cy - 51.2), shoulders at
    cy - 30. The crown is then centred at (cx, cy - 59.2) with radius 7.2."""
    xy = np.tile([cx, cy], (N_JOINTS, 1)).astype(np.float64)
    xy[0] = (cx, cy - 50)
    xy[1], xy[2] = (cx - 4, cy - 53), (cx + 4, cy - 53)
    xy[3], xy[4] = (cx - 8, cy - 50), (cx + 8, cy - 50)
    xy[5], xy[6] = (cx - 20, cy - 30), (cx + 20, cy - 30)
    xy[7], xy[8] = (cx - 35, cy - 10), (cx + 35, cy - 10)
    xy[9], xy[10] = (cx - 45, cy), (cx + 45, cy)
    xy[11], xy[12] = (cx - 15, cy + 30), (cx + 15, cy + 30)
    return xy


def disc(frame, cx, cy, r, colour):
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = colour


def paint(frame, cx, cy, vest, gear):
    """Torso in the vest colour, a headgear disc (radius 12) over the crown, and a skin face below it."""
    frame[max(int(cy - 30), 0):int(cy + 30) + 1, max(int(cx - 20), 0):int(cx + 20) + 1] = vest
    if gear is not None:
        disc(frame, cx, cy - 59.2, 12, gear)
    disc(frame, cx, cy - 47, 5, SKIN)


def bg():
    f = np.empty((H, W, 3), np.uint8)
    f[:] = BG
    return f


def table_of(people, n_frames, conf=0.9, source="GH000001.mp4", fp=""):
    """people: list of (tid, first, last, cx, cy)."""
    rows = []
    for tid, f0, f1, cx, cy in people:
        for f in range(f0, f1 + 1):
            rows.append((f, tid, person_xy(cx, cy)))
    rows.sort(key=lambda r: (r[0], r[1]))
    k = len(rows)
    xy = np.array([r[2] for r in rows], np.float32).reshape(k, N_JOINTS, 2)
    box = np.stack([xy[:, :, 0].min(1), xy[:, :, 1].min(1), xy[:, :, 0].max(1), xy[:, :, 1].max(1)], axis=1)
    return PoseTable(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64),
                     box.astype(np.float32), xy, np.full((k, N_JOINTS), conf, np.float32), n_frames, FPS, W, H,
                     source=source, video_fp=fp)


def run(people, look, n_frames, **kw):
    """look(tid, frame) -> (vest, gear) or None (not painted)."""
    tb = table_of(people, n_frames, **kw)

    def frames():
        for i in range(n_frames):
            fr = bg()
            for tid, f0, f1, cx, cy in people:
                if f0 <= i <= f1:
                    paint(fr, cx, cy, *look(tid, i))
            yield fr
    return tb, headgear.compute_headgear(None, tb, frames=frames())


def attrs_of(*tids):
    return [SimpleNamespace(track_id=t) for t in tids] + [SimpleNamespace(track_id=None)]


# ---------------------------------------------------------------------------------------------
print("head size and the crown disc")
xy0 = person_xy(100, 120)
c9 = np.full(N_JOINTS, 0.9)
check("head size from the ears: 1.0 x ear distance (16 px)", headgear.head_size(xy0, c9) == 16.0)
c = c9.copy(); c[[3, 4]] = 0.1
check("... no ears: 2.5 x the eye distance (20 px)", headgear.head_size(xy0, c) == 20.0)
c[[1, 2]] = 0.1
check("... no eyes either: 0.45 x the torso length (27 px)", abs(headgear.head_size(xy0, c) - 27.0) < 1e-9)
c[[5, 6, 11, 12]] = 0.1
check("... nothing usable: NaN", math.isnan(headgear.head_size(xy0, c)))
small = xy0.copy(); small[3], small[4] = (99, 70), (102, 70)
check("a head under MIN_HEAD_PX (3 px) is not read: NaN", math.isnan(headgear.head_size(small, c9)))
cx_, cy_, r_ = headgear.crown_disc(xy0, c9)
check("crown centre = face centre + 0.5 head sizes along the neck axis (100, 60.8), radius 0.45 x 16 = 7.2",
      abs(cx_ - 100) < 1e-9 and abs(cy_ - 60.8) < 1e-9 and abs(r_ - 7.2) < 1e-9, str((cx_, cy_, r_)))
side = xy0.copy(); side[5], side[6] = (140, 69), (140, 69)        # shoulders to the RIGHT of the face: lying down
sx, sy, _ = headgear.crown_disc(side, c9)
check("the crown follows the neck axis, not the image: shoulders to the right -> crown to the LEFT of the face",
      sx < 100 - 7 and abs(sy - 68.8) < 1.0, str((sx, sy)))
c = c9.copy(); c[[5, 6]] = 0.1
ux, uy, _ = headgear.crown_disc(xy0, c)
check("without both shoulders the crown is straight up in the image", abs(ux - 100) < 1e-9 and abs(uy - 60.8) < 1e-9)
c = c9.copy(); c[list(headgear.FACE)] = 0.1
check("no confident face keypoint: no crown", headgear.crown_disc(xy0, c) is None)

print("colour bands: both sides of every edge")


def hue_of(rgb):
    return colorsys.rgb_to_hsv(*(np.array(rgb) / 255.0))


cases = [((255, 0, 0), "red"), ((255, 0, 63), "red"), ((255, 0, 70), "-"), ((255, 100, 40), "red"),
         ((255, 128, 0), "-"), (SKIN, "-"), ((255, 147, 89), "-"), ((128, 40, 40), "-"), ((128, 38, 38), "red"), ((50, 10, 10), "-"),
         ((30, 60, 230), "blue"), ((6, 12, 40), "blue"), ((3, 6, 18), "-"), ((0, 200, 255), "-"),
         ((0, 190, 255), "blue"), ((190, 0, 255), "blue"), ((195, 0, 255), "-"), ((128, 160, 255), "-"),
         ((127, 160, 255), "blue"), (WHITE, "-"), (BG, "-")]
px = np.array([p for p, _ in cases], np.uint8)
rm, bm = headgear.kit_masks(px)
got = ["red" if r else ("blue" if b else "-") for r, b in zip(rm, bm)]
for (p, want), g in zip(cases, got):
    h, s, v = hue_of(p)
    check(f"RGB {p} (hue {h * 360:.1f}, sat {s:.3f}, val {v:.3f}) is {want}", g == want, f"got {g}")
check("no pixel is both red and blue", not (rm & bm).any())

print("crown counts on painted frames")
fr = bg(); paint(fr, 100, 120, RED, BLUE)
rc = headgear.crown_counts(fr, xy0, c9)
check("blue headgear on a RED vest reads blue: every crown pixel blue, none red", rc is not None and rc[1] == rc[2]
      and rc[0] == 0 and rc[2] >= MIN_REGION_PX, str(rc))
fr = bg(); paint(fr, 100, 120, RED, None)
rc = headgear.crown_counts(fr, xy0, c9)
check("a red vest and a skin face with NO headgear: the crown holds no kit (the vest and face are not read)",
      rc is not None and rc[0] == 0 and rc[1] == 0, str(rc))
fr = bg(); fr[:] = RED
check("a red background is read if it is under the crown (stated residual, ring pads)",
      headgear.crown_counts(fr, xy0, c9)[0] > 0)
edge = person_xy(100, 62)                    # crown centre at y = 2.8: mostly above the frame
rc = headgear.crown_counts(bg(), edge, c9)
check("a crown partly outside the frame counts only the pixels inside", rc is None or rc[2] < 163, str(rc))
out = person_xy(100, -40)
check("a crown wholly outside the frame is not read", headgear.crown_counts(bg(), out, c9) is None)
check("a crown under MIN_REGION_PX pixels is not read",
      headgear.crown_counts(bg(), xy0, c9, min_px=10_000) is None)

print("decisions over a set of rows")
n = np.full(10, 100)
check("ten all-red rows: red", headgear.decide_rows(np.full(10, 90), np.zeros(10, int), n) == "red")
check("ten all-blue rows: blue", headgear.decide_rows(np.zeros(10, int), np.full(10, 90), n) == "blue")
check("two rows only: unsure (too few)", headgear.decide_rows([90, 90], [0, 0], [100, 100]) == "unsure")
check("ten rows without kit: none", headgear.decide_rows(np.zeros(10, int), np.zeros(10, int), n) == "none")
check("half red rows, half blue rows: unsure", headgear.decide_rows([90] * 5 + [0] * 5, [0] * 5 + [90] * 5, n) == "unsure")
check("no rows at all: unsure", headgear.decide_rows([], [], []) == "unsure")

print("compute_headgear: which rows, and refusals")
NF = 60
tb, cnt = run([(1, 0, NF - 1, 80, 120), (2, 0, NF - 1, 240, 120)], lambda t, i: (RED, RED if t == 1 else BLUE), NF)
check("rows read are exactly those on frames divisible by 5 (12 frames x 2 people)",
      len(cnt.rows) == 24 and (tb.frame[cnt.rows] % 5 == 0).all(), str(len(cnt.rows)))
check("person 1 (red headgear) counts red, person 2 (blue headgear) counts blue",
      (cnt.red[tb.track_id[cnt.rows] == 1] > 0).all() and (cnt.blue[tb.track_id[cnt.rows] == 2] > 0).all()
      and (cnt.blue[tb.track_id[cnt.rows] == 1] == 0).all() and (cnt.red[tb.track_id[cnt.rows] == 2] == 0).all())
raises("a frame of the wrong size is refused",
       lambda: headgear.compute_headgear(None, tb, frames=iter([np.zeros((10, 10, 3), np.uint8)] * NF)), "frame 0")
raises("a video shorter than the table is refused",
       lambda: headgear.compute_headgear(None, tb, frames=iter([bg()] * 30)), "different video")
for bad in (0, -1, 2.5, True):
    raises(f"every={bad!r} is refused", lambda b=bad: headgear.compute_headgear(None, tb, every=b, frames=iter(())), "every")

print("the counts file is an input: round trip and refusals")
p = tmp / "a.headgear.npz"
cnt.save(p)
back = HeadgearCounts.load(p)
check("save -> load is exact", all(np.array_equal(getattr(back, k), getattr(cnt, k)) for k in ("rows", "red", "blue", "n"))
      and (back.source, back.n_frames, back.n_table_rows, back.video_fp) == (cnt.source, cnt.n_frames, cnt.n_table_rows, cnt.video_fp))
back.check_table(tb)
check("... and binds to its table", True)


def write(name, **over):
    d = {"format": np.array(headgear.FORMAT), "source": np.array(cnt.source), "video_fp": np.array(""),
         "table_sha256": np.array(cnt.table_sha256),
         "meta": np.array([cnt.n_frames, cnt.n_table_rows, cnt.every], np.int64), "rows": cnt.rows, "red": cnt.red,
         "blue": cnt.blue, "n": cnt.n}
    for k, v in over.items():
        if v is None:
            d.pop(k)
        else:
            d[k] = v
    q = tmp / name
    np.savez(q, **d)
    return q


bad_files = [
    ("a missing array", dict(n=None), "wrong arrays"),
    ("an extra array", dict(extra=np.zeros(1)), "wrong arrays"),
    ("an unknown format", dict(format=np.array("boxingvi-headgear-0")), "unknown format"),
    ("a format that is not text", dict(format=np.array(1)), "text"),
    ("a malformed fingerprint", dict(video_fp=np.array("ABC")), "video_fp"),
    ("float rows", dict(rows=cnt.rows.astype(float)), "not allowed"),
    ("2-D counts", dict(red=cnt.red.reshape(-1, 1)), "1-D"),
    ("arrays of different lengths", dict(red=cnt.red[:-1]), "differ in length"),
    ("rows out of order", dict(rows=cnt.rows[::-1].copy()), "strictly increasing"),
    ("a repeated row", dict(rows=np.r_[cnt.rows[:1], cnt.rows[:-1]]), "strictly increasing"),
    ("a row beyond the table", dict(rows=np.r_[cnt.rows[:-1], cnt.n_table_rows]), "strictly increasing"),
    ("a negative row", dict(rows=np.r_[-1, cnt.rows[1:]]), "strictly increasing"),
    ("more kit than pixels", dict(red=cnt.n + 1), "impossible"),
    ("a negative count", dict(blue=np.r_[-1, cnt.blue[1:]]), "impossible"),
    ("a region under the minimum", dict(n=np.r_[MIN_REGION_PX - 1, cnt.n[1:]]), "impossible"),
    ("meta of the wrong shape", dict(meta=np.array([1, 2], np.int64)), "meta"),
    ("zero frames", dict(meta=np.array([0, cnt.n_table_rows, 5], np.int64)), "meta"),
    ("a sampling step of 0", dict(meta=np.array([cnt.n_frames, cnt.n_table_rows, 0], np.int64)), "meta"),
    ("a missing table digest", dict(table_sha256=None), "wrong arrays"),
    ("a malformed table digest", dict(table_sha256=np.array("xyz")), "table_sha256"),
    ("counts that overflow int64 when added (red 2^63-1, blue 2^61)",
     dict(red=np.full(len(cnt.n), 2 ** 63 - 1, np.int64), blue=np.full(len(cnt.n), 2 ** 61, np.int64)), "impossible"),
    ("a crown bigger than a 4K frame", dict(n=np.r_[headgear.MAX_REGION_PX + 1, cnt.n[1:]]), "impossible"),
]
for i, (what, over, why) in enumerate(bad_files):
    q = write(f"bad{i}.npz", **over)
    raises(f"refused: {what}", lambda q=q: HeadgearCounts.load(q), why)
q = tmp / "pickle.npz"
np.savez(q, format=np.array([{"x": 1}], dtype=object))
raises("refused: an object (pickle) array, never unpickled", lambda: HeadgearCounts.load(q), "")
q = tmp / "junk.npz"
q.write_bytes(b"not a zip at all")
raises("refused: a file that is not an npz", lambda: HeadgearCounts.load(q), "not a readable npz")
old = headgear.MAX_FILE_BYTES
headgear.MAX_FILE_BYTES = 10
try:
    raises("refused: a file over the size cap, before it is read", lambda: HeadgearCounts.load(p), "bytes")
finally:
    headgear.MAX_FILE_BYTES = old
old_u, old_load = headgear.MAX_UNCOMPRESSED_BYTES, headgear.np.load


def _no_load(*a, **k):
    raise AssertionError("np.load was called before the declared sizes were checked")


headgear.MAX_UNCOMPRESSED_BYTES = 100
headgear.np.load = _no_load
try:
    raises("refused: a file whose zip directory DECLARES too many bytes, before any member is read",
           lambda: HeadgearCounts.load(p), "uncompressed")
finally:
    headgear.MAX_UNCOMPRESSED_BYTES, headgear.np.load = old_u, old_load
import zipfile  # noqa: E402

bomb = tmp / "bomb.npz"
with zipfile.ZipFile(bomb, "w") as zf:
    for name in sorted(headgear._KEYS):
        b = io.BytesIO()
        if name == "rows":                   # a header declaring 10^10 int64 rows over a few bytes of data
            np.lib.format.write_array_header_1_0(b, {"descr": "<i8", "fortran_order": False, "shape": (10 ** 10,)})
            b.write(b"\0" * 16)
        else:
            np.lib.format.write_array(b, {"format": np.array(headgear.FORMAT), "source": np.array(cnt.source),
                                          "video_fp": np.array(""), "table_sha256": np.array(cnt.table_sha256),
                                          "meta": np.array([cnt.n_frames, cnt.n_table_rows, cnt.every], np.int64),
                                          "red": cnt.red, "blue": cnt.blue, "n": cnt.n}[name])
        zf.writestr(name + ".npy", b.getvalue())
headgear.np.load = _no_load
try:
    raises("refused: a tiny member whose NPY header declares 10^10 elements, before numpy allocates anything",
           lambda: HeadgearCounts.load(bomb), "declares")
finally:
    headgear.np.load = old_load
ta = table_of([(1, 0, NF - 1, 80, 120)], NF)
tb_ = table_of([(1, 0, NF - 1, 80, 120)], NF)
ta.conf = ta.conf.astype(np.float64); tb_.conf = tb_.conf.astype(np.float64)
ta.conf[0, 0], tb_.conf[0, 0] = 0.299999999, 0.300000001
check("the content digest is lossless: confidences 0.299999999 and 0.300000001 (either side of MIN_CONF) differ",
      headgear.table_digest(ta, 5) != headgear.table_digest(tb_, 5))
check("... while a float32 table and its float64 copy (exact) hash the same",
      headgear.table_digest(table_of([(1, 0, 9, 80, 120)], 10), 5) == headgear.table_digest(
          (lambda t: (setattr(t, "xy", t.xy.astype(np.float64)), setattr(t, "conf", t.conf.astype(np.float64)), t)[2])(
              table_of([(1, 0, 9, 80, 120)], 10)), 5))
re_t = table_of([(1, 0, NF - 1, 80, 120), (2, 0, NF - 1, 240, 120)], NF)
re_t.xy[:] = re_t.xy[::-1]                  # same video, same metadata and row count: other CONTENT per row
raises("counts are refused against a regenerated table with the same metadata (content digest)",
       lambda: cnt.check_table(re_t), "different content")
for what, kw in (("another source", dict(source="GH999999.mp4")), ("another length", dict(n_frames=NF + 1)),
                 ("another fingerprint", dict(fp="0" * 64))):
    other = table_of([(1, 0, NF - 1, 80, 120), (2, 0, NF - 1, 240, 120)], kw.get("n_frames", NF),
                     source=kw.get("source", "GH000001.mp4"), fp=kw.get("fp", ""))
    raises(f"counts are refused against a table from {what}", lambda o=other: cnt.check_table(o), "")
fewer = table_of([(1, 0, NF - 1, 80, 120), (2, 0, NF - 2, 240, 120)], NF)
raises("counts are refused against a table with another row count", lambda: cnt.check_table(fewer), "table rows")

print("the self-check reads each pair over its shared frames")
c1 = headgear.headgear_check(tb, attrs_of(1, 2), cnt)
check("two red-VEST boxers, red and blue headgear, co-present: one decided pair, opposite (the 10-22 eye check)",
      c1["copresent_pairs"] == 1 and c1["decided_pairs"] == 1 and c1["opposite"] == 1 and c1["opposite_share"] == 1.0, str(c1))
tb_s, cnt_s = run([(1, 0, NF - 1, 80, 120), (2, 0, NF - 1, 240, 120)], lambda t, i: (RED, RED), NF)
c2 = headgear.headgear_check(tb_s, attrs_of(1, 2), cnt_s)
check("both in red headgear: a decided SAME pair, opposite share 0.0 (the check can fail)",
      c2["same"] == 1 and c2["opposite_share"] == 0.0, str(c2))
c3 = headgear.headgear_check(tb, attrs_of(1), cnt)
check("a fragment without a credited punch is never paired", c3["copresent_pairs"] == 0 and c3["opposite_share"] is None)
NF2 = 300
# id 2 is the blue boxer while co-present with id 1 (frames 0-99), then HOPS to a red-headgear person for 200 frames
hop = [(1, 0, 99, 80, 120), (2, 0, NF2 - 1, 240, 120)]
tb_h, cnt_h = run(hop, lambda t, i: (RED, RED) if (t == 1 or i >= 100) else (RED, BLUE), NF2)
c4 = headgear.headgear_check(tb_h, attrs_of(1, 2), cnt_h)
whole = headgear.decide_rows(cnt_h.red[tb_h.track_id[cnt_h.rows] == 2], cnt_h.blue[tb_h.track_id[cnt_h.rows] == 2],
                             cnt_h.n[tb_h.track_id[cnt_h.rows] == 2])
check("an id that hops to someone else after the shared frames: the WHOLE fragment reads red ...", whole == "red", whole)
check("... but read over the shared frames the pair is opposite (why v3 reads the window)", c4["opposite"] == 1, str(c4))
for f1, want in ((23, 0), (24, 1)):        # 24 or 25 shared frames: 0.5 s at 50 fps is 25
    tb_c, cnt_c = run([(1, 0, f1, 80, 120), (2, 0, f1, 240, 120)], lambda t, i: (RED, RED if t == 1 else BLUE), 60)
    cc = headgear.headgear_check(tb_c, attrs_of(1, 2), cnt_c)
    check(f"{f1 + 1} shared frames: {want} co-present pair(s) (the bar is 0.5 s = 25 frames)", cc["copresent_pairs"] == want, str(cc))
tb_u, cnt_u = run([(1, 0, NF - 1, 80, 120), (2, 0, NF - 1, 240, 120)], lambda t, i: (RED, None), NF)
c5 = headgear.headgear_check(tb_u, attrs_of(1, 2), cnt_u)
check("no headgear on either: the pair is co-present but undecided, the share is None (no measurement, not 0%)",
      c5["copresent_pairs"] == 1 and c5["decided_pairs"] == 0 and c5["opposite_share"] is None, str(c5))
raises("the check refuses counts from another table", lambda: headgear.headgear_check(tb_s, attrs_of(1, 2), cnt_u.__class__(
    cnt_u.rows, cnt_u.red, cnt_u.blue, cnt_u.n, "GH999999.mp4", cnt_u.n_frames, cnt_u.n_table_rows, cnt_u.every, cnt_u.table_sha256)), "")
star = [(1, 0, 99, 160, 120), (2, 0, 99, 40, 120), (3, 0, 99, 240, 120), (4, 0, 99, 300, 120)]
tb_x, cnt_x = run(star, lambda t, i: (RED, RED if t == 1 else BLUE), 100)
c6 = headgear.headgear_check(tb_x, attrs_of(1, 2, 3, 4), cnt_x)
check("one red fragment with three blue partners: fragment counts read each fragment ONCE (red 1, blue 3) ...",
      c6["credited_by_decision"]["red"] == 1 and c6["credited_by_decision"]["blue"] == 3, str(c6["credited_by_decision"]))
check("... while pair sides count the red fragment once per partner (red 3)", c6["window_decisions"]["red"] == 3
      and c6["copresent_pairs"] == 6 and c6["opposite"] == 3 and c6["same"] == 3, str(c6))
pl = headgear.pool([c1, c2, c5])
check("pooling adds integer counts and divides once: 1 opposite of 2 decided, 2 of 3 co-present decided",
      pl["opposite"] == 1 and pl["decided_pairs"] == 2 and pl["opposite_share"] == 0.5
      and abs(pl["decided_share"] - 2 / 3) < 1e-12 and pl["videos"] == 3, str(pl))

print("intervals")
lo, hi = headgear.wilson(98, 108)
check("Wilson 98/108 = [0.8379, 0.9489] (the dev number, recomputed)", abs(lo - 0.83790) < 1e-4 and abs(hi - 0.94893) < 1e-4, str((lo, hi)))
check("Wilson n = 0 is None; k = n has upper bound 1", headgear.wilson(0, 0) is None and headgear.wilson(5, 5)[1] == 1.0)
for k_, n_ in ((-1, 5), (6, 5), (1.5, 5), (True, 5)):
    raises(f"Wilson refuses k={k_!r}, n={n_!r}", lambda a=k_, b=n_: headgear.wilson(a, b), "whole numbers")
b1 = headgear.cluster_bootstrap([(9, 10), (8, 10), (10, 10)], draws=2000)
b2 = headgear.cluster_bootstrap([(9, 10), (8, 10), (10, 10)], draws=2000)
check("the video bootstrap is repeatable for a fixed seed", b1 == b2)
check("... and its interval contains the pooled share 0.9", b1["lo"] <= 0.9 <= b1["hi"], str(b1))
check("all videos perfect: the interval is [1, 1]", headgear.cluster_bootstrap([(5, 5), (7, 7)], draws=500)["lo"] == 1.0)
bb = headgear.cluster_bootstrap([(5, 5), (0, 0)], draws=500)
check("draws with no decided pair are dropped and counted", bb["dropped"] > 0 and bb["lo"] == 1.0, str(bb))
check("no decided pair anywhere: None", headgear.cluster_bootstrap([(0, 0)]) is None and headgear.cluster_bootstrap([]) is None)
raises("the bootstrap refuses opposite > decided", lambda: headgear.cluster_bootstrap([(3, 2)]), "opposite")
raises("the bootstrap refuses zero draws", lambda: headgear.cluster_bootstrap([(1, 2)], draws=0), "draws")
one_bad = headgear.cluster_bootstrap([(100, 100)] * 22 + [(0, 50)], draws=4000)
check("one video of 50 same pairs among 22 perfect ones pulls the lower bound well under the point estimate",
      one_bad["lo"] < (2200 / 2250) - 0.02, str(one_bad))

print("the split and the gate constants")
ALL29 = ["GH078416", "GH088416", "GH098416", "GH108416", "GH118416", "GH128416", "GH138416", "GH148416", "GH158416",
         "GH168416", "GH178416", "GH188416", "GH198416", "GH208416", "GH218416", "GH069681", "GH079681", "GH089681",
         "GH099681", "GH109681", "GH119681", "GH129681", "GH139681", "GH149681", "GH159681", "GH169681", "GH179681",
         "GH189681", "GH199681"]
sp = [olympic_headgear.split_of(v) for v in ALL29]
check("29 videos: 4 dev, 2 excluded (eye-checked GH089681, GH118416), 23 held-out",
      sp.count("dev") == 4 and sp.count("excluded") == 2 and sp.count("heldout") == 23
      and {v for v, s in zip(ALL29, sp) if s == "excluded"} == {"GH089681", "GH118416"})
check("the frozen held-out manifest is exactly the 29 minus dev minus the 2 eye-checked",
      olympic_headgear.HELD_OUT == frozenset(ALL29) - olympic_headgear.DEV_VIDEOS - olympic_headgear.EYE_CHECKED
      and len(olympic_headgear.HELD_OUT) == 23)
check("a video outside the 29 is 'unknown', never held-out", olympic_headgear.split_of("GH000002") == "unknown")
check("the robustness row drops exactly the 3 chapter twins of dev videos: 20 videos",
      len([v for v, s in zip(ALL29, sp) if s == "heldout" and v not in olympic_headgear.CHAPTER_TWINS]) == 20
      and olympic_headgear.CHAPTER_TWINS <= {v for v, s in zip(ALL29, sp) if s == "heldout"})
check("the gate is 0.90 over >= 100 decided pairs with a bootstrap lower bound >= 0.80, 20 pairs per camera",
      (olympic_headgear.GATE_SHARE, olympic_headgear.GATE_MIN_PAIRS, olympic_headgear.GATE_BOOT_LO,
       olympic_headgear.CAMERA_MIN_PAIRS) == (0.90, 100, 0.80, 20))
ok = {"opposite_share": 0.90, "decided_pairs": 100, "bootstrap95": {"lo": 0.80, "draws": 10_000, "dropped": 100}}
check("gate: exactly at every bar passes (including 1% of draws dropped)", olympic_headgear.gate(ok))
for what, ch in (("share 0.8999", {"opposite_share": 0.8999}), ("99 pairs", {"decided_pairs": 99}),
                 ("lower bound 0.7999", {"bootstrap95": {"lo": 0.7999, "draws": 10_000, "dropped": 0}}),
                 ("101 of 10,000 draws dropped", {"bootstrap95": {"lo": 0.95, "draws": 10_000, "dropped": 101}}),
                 ("no bootstrap", {"bootstrap95": None}), ("no share", {"opposite_share": None})):
    check(f"gate: {what} fails", not olympic_headgear.gate(ok | ch))

print("the CLI on a painted clip")
NFV = 120
two = [(1, 0, NFV - 1, 80, 120), (2, 0, NFV - 1, 240, 120)]
ds = tmp / "ds" / "task_kam4_gh000002"
(ds / "data").mkdir(parents=True)
with video.VideoWriter(ds / "data" / "GH000002.mp4", W, H, Fraction(50)) as vw:
    for i in range(NFV):
        fr = bg()
        paint(fr, 80, 120, RED, RED)
        paint(fr, 240, 120, RED, BLUE)
        vw.write(fr)
tab = table_of(two, NFV, source="GH000002.mp4", fp=video.fingerprint(ds / "data" / "GH000002.mp4"))
labels = {v: k for k, v in olympic.TARGETS.items()}
hands = {v: k for k, v in olympic.HANDS.items()}
tracks_json = []
for f0, cx in ((20, 80), (60, 240), (90, 80)):            # a glove box at each person's left wrist (cx - 45, 120)
    box = [cx - 53, 112, cx - 37, 128]
    tracks_json.append({"label": f"{labels['head']} {hands['left']}",
                        "shapes": [{"type": "rectangle", "frame": f0, "outside": False, "points": box},
                                   {"type": "rectangle", "frame": f0 + 3, "outside": True, "points": box}]})
(ds / "annotations.json").write_text(json.dumps([{"tracks": tracks_json}]), encoding="utf-8")
trk = tmp / "trk"
trk.mkdir()
tab.save(trk / "GH000002.tracks.npz")
root, cdir = tmp / "ds", tmp / "counts"
base = ["--data", str(root), "--tracks", str(trk), "--counts-dir", str(cdir)]
with contextlib.redirect_stdout(io.StringIO()):
    raises("a video outside the frozen manifest is never read as held-out", lambda: olympic_headgear.main(
        base + ["--split", "heldout", "--check-out", str(tmp / "r" / "u.json")]), "could be read")
check("... and no counts file was written for it", not (cdir / "GH000002.headgear.npz").exists())
REAL_HELD_OUT = olympic_headgear.HELD_OUT
olympic_headgear.HELD_OUT = frozenset({"GH000002"})          # the painted clip stands in for the manifest
with contextlib.redirect_stdout(io.StringIO()):
    raises("the default split is dev: a (manifest) held-out video is not read", lambda: olympic_headgear.main(
        base + ["--check-out", str(tmp / "r" / "d.json")]), "could be read")
check("... and no counts file was written for it either", not (cdir / "GH000002.headgear.npz").exists())
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = olympic_headgear.main(base + ["--split", "heldout", "--check-out", str(tmp / "r" / "h.json")])
res = json.loads((tmp / "r" / "h.json").read_text(encoding="utf-8"))
v = res["videos"]["GH000002"]
check("held-out run: one opposite decided pair on the painted clip", v["opposite"] == 1 and v["decided_pairs"] == 1, str(v))
check("... but one pair is not 100: G1 FAILS, exit 2 (the gate can fail)", rc == 2 and res["passes"] is False, str((rc, res["passes"])))
check("... the counts went to --counts-dir, not next to the tracks",
      (cdir / "GH000002.headgear.npz").exists() and not (trk / "GH000002.headgear.npz").exists())
check("... and the pooled block carries Wilson, the bootstrap and the camera's 'not established'",
      res["heldout"]["all"]["wilson95"] is not None and res["heldout"]["all"]["bootstrap95"] is not None
      and res["heldout"]["kam4"]["established"] is False)
with contextlib.redirect_stdout(io.StringIO()):
    rc2 = olympic_headgear.main(base + ["--split", "heldout", "--check-only", "--check-out", str(tmp / "r" / "h2.json")])
res2 = json.loads((tmp / "r" / "h2.json").read_text(encoding="utf-8"))
check("--check-only recomputes the same numbers from the counts file", rc2 == 2
      and res2["heldout"]["all"]["opposite"] == res["heldout"]["all"]["opposite"])
with contextlib.redirect_stdout(io.StringIO()):
    rc3 = olympic_headgear.main(base + ["--split", "heldout", "--only", "GH000002", "--check-out", str(tmp / "r" / "h3.json")])
check("--only gives no gate verdict (a subset is not the gate)", rc3 == 0
      and json.loads((tmp / "r" / "h3.json").read_text(encoding="utf-8"))["passes"] is None)
wrong = tmp / "trk_wrong"
wrong.mkdir()
table_of(two, NFV, source="GH000002.mp4", fp="0" * 64).save(wrong / "GH000002.tracks.npz")
with contextlib.redirect_stdout(io.StringIO()):
    rc4 = olympic_headgear.main(["--data", str(root), "--tracks", str(wrong), "--counts-dir", str(tmp / "c2"),
                                 "--split", "all", "--check-out", str(tmp / "r" / "h4.json")])
res4 = json.loads((tmp / "r" / "h4.json").read_text(encoding="utf-8")) if (tmp / "r" / "h4.json").exists() else None
check("a tracks table bound to another video's content fails that video (never read against the wrong footage)",
      rc4 == 1 and res4 is not None and res4["failed"] == ["GH000002"] and "error" in res4["videos"]["GH000002"],
      str(rc4))
with contextlib.redirect_stdout(io.StringIO()):
    raises("--only refuses a malformed video id (no path tricks)", lambda: olympic_headgear.main(
        base + ["--split", "heldout", "--only", "..\\x", "--check-out", str(tmp / "r" / "h5.json")]), "")
(cdir / "GH000002.headgear.npz").write_bytes(b"garbage")
with contextlib.redirect_stdout(io.StringIO()):
    rc6 = olympic_headgear.main(base + ["--split", "heldout", "--check-only", "--check-out", str(tmp / "r" / "h6.json")])
check("--check-only on a corrupt counts file fails that video, exit 1", rc6 == 1, str(rc6))
olympic_headgear.HELD_OUT = frozenset({"GH000002", "GH000003"})   # one manifest video is absent from the data
with contextlib.redirect_stdout(io.StringIO()):
    rc7 = olympic_headgear.main(base + ["--split", "heldout", "--check-out", str(tmp / "r" / "h7.json")])
res7 = json.loads((tmp / "r" / "h7.json").read_text(encoding="utf-8"))
check("a held-out set with a video missing is NOT evaluated (passes None, exit 1), even though the video read passes "
      "its own check", rc7 == 1 and res7["passes"] is None and res7["heldout_missing"] == ["GH000003"], str((rc7, res7.get("passes"))))
olympic_headgear.HELD_OUT = frozenset({"GH000002"})
dup = tmp / "dup"
for task in ("task_kam2_gh000002", "task_kam4_gh000002"):
    (dup / task / "data").mkdir(parents=True)
    (dup / task / "data" / "GH000002.mp4").write_bytes((ds / "data" / "GH000002.mp4").read_bytes())
    (dup / task / "annotations.json").write_text((ds / "annotations.json").read_text(encoding="utf-8"), encoding="utf-8")
with contextlib.redirect_stdout(io.StringIO()):
    raises("one video id in two task folders is refused, not overwritten", lambda: olympic_headgear.main(
        ["--data", str(dup), "--tracks", str(trk), "--counts-dir", str(tmp / "c3"), "--split", "heldout",
         "--check-out", str(tmp / "r" / "h8.json")]), "two task folders")
olympic_headgear.HELD_OUT = REAL_HELD_OUT

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
