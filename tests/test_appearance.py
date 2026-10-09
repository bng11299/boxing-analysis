"""Tests for kit colour per fragment (appearance.py) and olympic_colour.py.

Synthetic only: every frame is generated in memory from painted shapes (a red-vest figure, a blue-vest
figure, a white-shirt referee, a skin-only person, ...). No Olympic video, no model, no GPU, and nothing
here ever writes a pixel to disk except the tiny synthetic clip the CLI test builds from painted shapes.

Run:  python tests/test_appearance.py
"""

from __future__ import annotations

import colorsys
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from boxingvi import appearance  # noqa: E402
from boxingvi.appearance import Colours, FragmentColour  # noqa: E402
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
BG = (110, 110, 110)                 # ring canvas: grey, no saturation
SKIN = (225, 170, 140)               # saturation 0.38, hue 20 degrees: skin, not kit
RED, BLUE, WHITE = (220, 30, 30), (30, 60, 230), (245, 245, 245)


def person_xy(cx, cy):
    """COCO-17 for a figure of torso length 60 centred at (cx, cy): shoulders +-20, hips +-15, a face
    cluster whose mean is (cx, cy - 51.2), wrists 45 px out to each side."""
    xy = np.tile([cx, cy], (N_JOINTS, 1)).astype(np.float64)
    xy[0] = (cx, cy - 50)
    xy[1], xy[2] = (cx - 4, cy - 53), (cx + 4, cy - 53)
    xy[3], xy[4] = (cx - 8, cy - 50), (cx + 8, cy - 50)
    xy[5], xy[6] = (cx - 20, cy - 30), (cx + 20, cy - 30)
    xy[11], xy[12] = (cx - 15, cy + 30), (cx + 15, cy + 30)
    xy[9], xy[10] = (cx - 45, cy), (cx + 45, cy)
    return xy


def disc(frame, cx, cy, r, colour):
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = colour


def paint(frame, cx, cy, vest, head=SKIN, gloves=SKIN):
    """Paint a figure bigger than its sampled regions, so each region sits wholly inside its colour."""
    x0, x1, y0, y1 = int(cx - 20), int(cx + 20), int(cy - 30), int(cy + 30)
    frame[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = vest
    disc(frame, cx, cy - 51.2, 22, head)
    disc(frame, cx - 45, cy, 22, gloves)
    disc(frame, cx + 45, cy, 22, gloves)


def bg():
    f = np.empty((H, W, 3), np.uint8)
    f[:] = BG
    return f


def table_of(people, n_frames, conf=0.9, source="GH000001.mp4", fp=""):
    """people: list of (tid, frame_first, frame_last, cx, cy)."""
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


def run_colours(people, looks, n_frames, every=appearance.SAMPLE_EVERY, bad_frames=None, **kw):
    """Colours of a synthetic video. looks: {tid: (vest, head, gloves)}; bad_frames: {frame: {tid: vest}} override."""
    tb = table_of(people, n_frames, **kw)

    def frames():
        for i in range(n_frames):
            fr = bg()
            for tid, f0, f1, cx, cy in people:
                if f0 <= i <= f1:
                    vest, head, gloves = looks[tid]
                    if bad_frames and i in bad_frames and tid in bad_frames[i]:
                        vest, head, gloves = bad_frames[i][tid]
                    paint(fr, cx, cy, vest, head, gloves)
            yield fr
    return tb, appearance.compute_colours(None, tb, frames=frames(), every=every)


print("HSV: against the standard library, hue wrap, and the thresholds' edges")
rng = np.random.default_rng(7)
rgb = rng.integers(0, 256, size=(3000, 3), dtype=np.uint8)
hh, ss, vv = appearance.hsv(rgb)
ref = np.array([colorsys.rgb_to_hsv(*(p / 255.0)) for p in rgb.astype(float)])
check("hue (degrees), saturation and value agree with colorsys on 3,000 random pixels",
      np.abs(hh - ref[:, 0] * 360).max() < 1e-7 and np.abs(ss - ref[:, 1]).max() < 1e-12
      and np.abs(vv - ref[:, 2]).max() < 1e-12, str((np.abs(hh - ref[:, 0] * 360).max(),)))


def kit(*px):
    r, b = appearance.kit_masks(np.array(px, np.uint8))
    return r.tolist(), b.tolist()


check("hue wrap: pure red (hue 0), hue 359 and hue 340.2 are red; hue 339.8 and hue 330 are not",
      kit((255, 0, 0), (255, 0, 4), (255, 0, 84), (255, 0, 86), (255, 0, 128))[0] == [True, True, True, False, False])
check("red upper edge: hue 11.8 is red, hue 12.5 is not (a yellow-orange)",
      kit((255, 50, 0), (255, 53, 0))[0] == [True, False])
check("blue band: hue 200.2 and 254 are blue; 199.5 (cyan-ish) and 256 (violet) are not",
      kit((0, 169, 255), (0, 172, 255), (60, 0, 255), (68, 0, 255))[1] == [True, False, True, False])
check("a red pixel is never also blue, and pure blue is not red",
      kit((255, 0, 0), (0, 0, 255)) == ([True, False], [False, True]))
check("saturation edge: (255,140,140) has saturation 0.451 and is red, (255,141,141) has 0.447 and is not",
      kit((255, 140, 140), (255, 141, 141))[0] == [True, False])
check("value edge: (51,0,0) has value exactly 0.20 and is red, (50,0,0) is too dark to have a hue",
      kit((51, 0, 0), (50, 0, 0))[0] == [True, False])
check("skin (225,170,140), white, grey, black and the canvas are neither red nor blue",
      kit(SKIN, WHITE, (128, 128, 128), (0, 0, 0), BG) == ([False] * 5, [False] * 5))
check("skin in strong light (255,140,100: saturation 0.61 but hue 15.5) is not red; pinkish skin (255,170,160) is "
      "red-hued but too pale (saturation 0.37)", kit((255, 140, 100), (255, 170, 160))[0] == [False, False])
r_floor, _ = appearance.kit_masks(np.array([(255, 170, 160)], np.uint8), sat_min=0.0)
check("... and without the saturation floor that pale pixel WOULD be red (the floor is what excludes it)", r_floor.tolist() == [True])

print("regions: quadrilateral shrink, discs, clipping, too few pixels")
sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
check("the torso quadrilateral shrinks 20% towards its centroid: (0,0)-(100,100) becomes (10,10)-(90,90)",
      np.allclose(appearance.shrink_quad(sq), [[10, 10], [90, 10], [90, 90], [10, 90]]))
X, Y = np.meshgrid(np.arange(0, 100, dtype=float), np.arange(0, 100, dtype=float))
check("... which holds exactly 80 x 80 pixels", int(appearance._in_quad(X, Y, appearance.shrink_quad(sq)).sum()) == 6400)
bow = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], float)         # a self-crossing 'bow tie': even-odd, no crash
check("a self-crossing vertex order (shoulders swapped) still gives a finite region",
      0 < int(appearance._in_quad(X, Y, bow).sum()) < 10000)
xy0, conf1 = person_xy(160, 120), np.full(N_JOINTS, 0.9)
box0 = (90, 48, 230, 150)
x0, y0, mask, parts = appearance.region_masks(xy0, conf1, box0, W, H)
check("a standing figure gives all four regions: torso, head and both gloves", sorted(parts) == ["glove_l", "glove_r", "head", "torso"],
      str(parts))
check("the torso is the shrunk trapezoid: area 0.64 x (40 + 30) / 2 x 60 = 1344 (+-3%)", abs(parts["torso"] - 1344) < 0.03 * 1344,
      str(parts))
check("each disc is 0.35 torso lengths (21 px) in radius: area pi x 21^2 = 1385 (+-3%)",
      all(abs(parts[k] - 1385) < 0.03 * 1385 for k in ("head", "glove_l", "glove_r")), str(parts))
check("the regions do not overlap here, so the union is their sum", int(mask.sum()) == sum(parts.values()), str((mask.sum(), parts)))
face_mean = xy0[:5].mean(axis=0)
yy, xx = np.nonzero(mask)
head_hit = ((xx + x0 - face_mean[0]) ** 2 + (yy + y0 - face_mean[1]) ** 2 <= 21.0 ** 2).sum()
check("the head disc is centred on the MEAN of the face keypoints (160, 108.8), not on the nose",
      head_hit >= parts["head"], str((head_hit, parts)))
lowc = conf1.copy()
lowc[[3, 4]] = 0.1
_, _, _, parts_f = appearance.region_masks(xy0, lowc, box0, W, H)
check("a low-confidence ear is left out of the face mean, the head stays", "head" in parts_f)
check("a low-confidence wrist has no glove, the other glove stays",
      (lambda p_: "glove_l" not in p_ and "glove_r" in p_)(
          appearance.region_masks(xy0, np.where(np.arange(N_JOINTS) == 9, 0.1, 0.9), box0, W, H)[3]))
nohead = np.where(np.isin(np.arange(N_JOINTS), [0, 1, 2, 3, 4]), 0.1, 0.9)
check("no confident face keypoint: no head region", "head" not in appearance.region_masks(xy0, nohead, box0, W, H)[3])
notorso = np.where(np.isin(np.arange(N_JOINTS), [5, 6, 11, 12]), 0.1, 0.9)
parts_nt = appearance.region_masks(xy0, notorso, box0, W, H)[3]
check("an unconfident hip or shoulder: no torso region, but the head and gloves remain (sized by the box height x 0.3 = 30.6)",
      "torso" not in parts_nt and "head" in parts_nt and abs(parts_nt["head"] - np.pi * (0.35 * 30.6) ** 2) < 0.1 * parts_nt["head"],
      str(parts_nt))
edge = appearance.region_masks(person_xy(5, 120), conf1, (0, 48, 60, 150), W, H)
full = appearance.region_masks(person_xy(160, 120), conf1, box0, W, H)
check("a figure at the left edge is clipped to the frame: the window starts at 0 and fewer pixels are counted",
      edge is not None and edge[0] == 0 and edge[2].shape[1] <= W and sum(edge[3].values()) < sum(full[3].values()), str(edge and edge[3]))
check("every pixel of the window is inside the frame at all four edges",
      all((lambda r: r is not None and r[0] >= 0 and r[1] >= 0 and r[0] + r[2].shape[1] <= W and r[1] + r[2].shape[0] <= H)(
          appearance.region_masks(person_xy(cx, cy), conf1, box0, W, H)) for cx, cy in ((5, 120), (315, 120), (160, 20), (160, 235))))
check("a figure wholly outside the frame has no region (None), not an empty window",
      appearance.region_masks(person_xy(-500, 120), conf1, box0, W, H) is None
      and appearance.region_masks(person_xy(160, 9000), conf1, box0, W, H) is None)
wild = person_xy(160, 120)
wild[5] = (1e7, -1e7)
rw = appearance.region_masks(wild, conf1, box0, W, H)
check("a wild keypoint (10 million pixels away) cannot allocate a huge window: the window stays inside the frame",
      rw is None or (rw[2].shape[0] <= H and rw[2].shape[1] <= W))
thin = person_xy(160, 120)
thin[5], thin[6], thin[11], thin[12] = (140, 116), (180, 116), (145, 124), (175, 124)     # wide but only 8 px tall: torso length 8
parts_t = appearance.region_masks(thin, conf1, box0, W, H)[3]
check("a region under MIN_REGION_PX = 30 pixels is skipped: the torso (area ~ 0.64 x 35 x 8 = 180) stays, the discs (r = 2.8, "
      "24 px) are dropped", list(parts_t) == ["torso"], str(parts_t))
check("... and with a higher floor nothing is left, which is None",
      appearance.region_masks(person_xy(160, 120), conf1, box0, W, H, min_px=10 ** 6) is None)
small = appearance.region_masks(person_xy(160, 120), conf1, box0, W, H, min_px=1360)
check("min_px is per region: with 1,360 the 1,344-pixel torso falls away and the three 1,385-pixel discs stay",
      small is not None and sorted(small[3]) == ["glove_l", "glove_r", "head"], str(small and small[3]))
check("MIN_REGION_PX is 30 and MIN_CONF is 0.3", appearance.MIN_REGION_PX == 30 and appearance.MIN_CONF == 0.3)

print("one row: the shares of painted figures (exact where the figure is one colour)")
f = bg()
paint(f, 160, 120, RED, RED, RED)
sh = appearance.row_shares(f, xy0, conf1, box0)
check("an all-red figure: red share 1.0, blue 0.0 exactly", sh[0] == 1.0 and sh[1] == 0.0 and sh[2] > 3000, str(sh))
f = bg()
paint(f, 160, 120, BLUE, BLUE, BLUE)
sh = appearance.row_shares(f, xy0, conf1, box0)
check("an all-blue figure: blue 1.0, red 0.0", sh[1] == 1.0 and sh[0] == 0.0, str(sh))
f = bg()
paint(f, 160, 120, RED, SKIN, RED)
sh = appearance.row_shares(f, xy0, conf1, box0)
want = (1344 + 2 * 1385) / (1344 + 3 * 1385)
check(f"red vest and red gloves, skin head: red share = (torso + 2 gloves) / all regions = {want:.3f}",
      abs(sh[0] - want) < 0.02 and sh[1] == 0.0, str(sh))
f = bg()
paint(f, 160, 120, WHITE, SKIN, SKIN)
sh = appearance.row_shares(f, xy0, conf1, box0)
check("a white-shirt referee with bare hands and face: no red, no blue", sh[0] == 0.0 and sh[1] == 0.0, str(sh))
f = bg()
paint(f, 160, 120, SKIN, SKIN, SKIN)
sh = appearance.row_shares(f, xy0, conf1, box0)
check("a skin-only person: no red, no blue", sh[0] == 0.0 and sh[1] == 0.0, str(sh))
f = bg()
paint(f, 160, 120, (255, 170, 160), SKIN, SKIN)
sh = appearance.row_shares(f, xy0, conf1, box0)
sh_nofloor = appearance.row_shares(f, xy0, conf1, box0, sat_min=0.0)
check("pale pink on skin (red-hued, saturation 0.37): not red with the floor; with the floor removed the torso reads red",
      sh[0] == 0.0 and sh_nofloor[0] > 0.2, str((sh, sh_nofloor)))
f = bg()
f[:, :160] = RED
f[:, 160:] = BLUE
sh = appearance.row_shares(f, xy0, conf1, box0)
check("a figure half in red, half in blue pixels reads both: neither share is small",
      sh[0] > 0.3 and sh[1] > 0.3, str(sh))
check("a frame with no readable region gives None (nothing to count)",
      appearance.row_shares(bg(), person_xy(-500, 120), conf1, box0) is None)

print("the decision from medians and rows (boundaries)")
d = appearance.decide
check("red: 3 rows, red 0.10, blue 0 is red; red 0.0999 is not", d(0.10, 0.0, 3) == "red" and d(0.0999, 0.0, 3) == "unsure")
check("blue is symmetric", d(0.0, 0.10, 3) == "blue" and d(0.0, 0.0999, 3) == "unsure")
check("the margin is 3x: 0.375 vs 0.125 is red, 0.375 vs 0.1251 is not", d(0.375, 0.125, 5) == "red" and d(0.375, 0.1251, 5) == "unsure")
check("both colours present (0.3 and 0.2): unsure, whatever the rows", d(0.3, 0.2, 50) == "unsure")
check("2 rows is never a colour: a clear red with 2 rows is unsure; with 3 it is red", d(0.9, 0.0, 2) == "unsure" and d(0.9, 0.0, 3) == "red")
check("none needs 5 rows: 4 rows with nothing is unsure, 5 is none", d(0.0, 0.0, 4) == "unsure" and d(0.0, 0.0, 5) == "none")
check("none needs BOTH medians under 0.02: 0.0199 is none, 0.02 is not", d(0.0199, 0.0199, 9) == "none" and d(0.02, 0.0, 9) == "unsure"
      and d(0.0, 0.02, 9) == "unsure")
check("a weak signal (0.05) with plenty of rows is unsure: neither a colour nor nothing", d(0.05, 0.0, 30) == "unsure")
check("no rows at all is unsure", d(0.0, 0.0, 0) == "unsure")

print("a fragment through the pipeline: synthetic video, rows, medians")
P = [(1, 0, 30, 80, 120), (2, 0, 30, 240, 120)]
tb7, col7 = run_colours(P, {1: (RED, SKIN, RED), 2: (BLUE, SKIN, BLUE)}, 31)
check("a red-vest figure and a blue-vest figure side by side: exactly red and blue, 7 sampled rows each (frames 0, 5 ... 30)",
      col7.decision(1) == "red" and col7.decision(2) == "blue" and col7.by_id[1].rows == 7 and col7.by_id[2].rows == 7,
      str(col7.by_id))
check("... with the medians the painting implies (red ~0.75 / blue 0 and the reverse)",
      0.70 < col7.by_id[1].red < 0.80 and col7.by_id[1].blue == 0.0 and 0.70 < col7.by_id[2].blue < 0.80 and col7.by_id[2].red == 0.0,
      str(col7.by_id))
tbr, colr = run_colours([(1, 0, 30, 80, 120)], {1: (WHITE, SKIN, SKIN)}, 31)
check("a white-shirt referee seen on 7 rows is none (well seen, no kit colour)", colr.decision(1) == "none" and colr.by_id[1].rows == 7)
tbs, cols = run_colours([(1, 0, 30, 80, 120)], {1: (SKIN, SKIN, SKIN)}, 31)
check("a skin-only person is none too", cols.decision(1) == "none")
tbp, colp = run_colours([(1, 0, 30, 80, 120)], {1: ((255, 170, 160), SKIN, SKIN)}, 31)
check("the red-on-skin edge case (pale pink, saturation 0.37) is none, not red", colp.decision(1) == "none", str(colp.by_id))
# median robustness: frames 10 and 20 (2 of the 7 sampled rows) are completely blue around the red figure
tbm, colm = run_colours([(1, 0, 30, 80, 120)], {1: (RED, SKIN, RED)}, 31, bad_frames={10: {1: (BLUE, BLUE, BLUE)}, 20: {1: (BLUE, BLUE, BLUE)}})
check("median robustness: a red figure with 2 of its 7 sampled rows full of blue stays red (median blue 0)",
      colm.decision(1) == "red" and colm.by_id[1].rows == 7 and colm.by_id[1].blue == 0.0, str(colm.by_id))
mean_red = np.mean([0.748] * 5 + [0.0] * 2)
mean_blue = np.mean([0.0] * 5 + [1.0] * 2)
check("... and the MEAN would not have: mean red 0.53 vs mean blue 0.29 is under the 3x margin, so unsure",
      appearance.decide(mean_red, mean_blue, 7) == "unsure", str((mean_red, mean_blue)))
tbm3, colm3 = run_colours([(1, 0, 30, 80, 120)], {1: (RED, SKIN, RED)}, 31,
                          bad_frames={f_: {1: (BLUE, BLUE, BLUE)} for f_ in (5, 10, 15, 20)})
check("... but with 4 of 7 rows blue the median IS blue: the test follows the majority of the evidence",
      colm3.decision(1) == "blue", str(colm3.by_id))
# rows thresholds through the pipeline
for last, want_rows, want_dec in ((5, 2, "unsure"), (10, 3, "red")):
    _, c_ = run_colours([(1, 0, last, 80, 120)], {1: (RED, SKIN, RED)}, last + 1)
    check(f"a red figure on frames 0..{last} has {want_rows} rows (the ends are always read) and is {want_dec}",
          c_.by_id[1].rows == want_rows and c_.decision(1) == want_dec, str(c_.by_id))
for last, want_rows, want_dec in ((15, 4, "unsure"), (20, 5, "none")):
    _, c_ = run_colours([(1, 0, last, 80, 120)], {1: (WHITE, SKIN, SKIN)}, last + 1)
    check(f"a white-shirt figure on frames 0..{last} has {want_rows} rows and is {want_dec}",
          c_.by_id[1].rows == want_rows and c_.decision(1) == want_dec, str(c_.by_id))
_, cone = run_colours([(1, 3, 3, 80, 120)], {1: (RED, SKIN, RED)}, 4)
check("a one-frame fragment has one row and is unsure", cone.by_id[1].rows == 1 and cone.decision(1) == "unsure")
tb_nr = table_of([(1, 0, 30, 80, 120)], 31, conf=0.0)
col_nr = appearance.compute_colours(None, tb_nr, frames=iter([bg()] * 31))
check("a fragment whose joints are all unconfident has no readable row: rows 0, unsure (never 'none')",
      col_nr.by_id[1].rows == 0 and col_nr.decision(1) == "unsure", str(col_nr.by_id))

print("which rows are read")
tsel = table_of([(1, 3, 23, 80, 120), (2, 0, 9, 240, 120)], 30)
tsel.conf[(tsel.track_id == 1) & (tsel.frame == 3), 5] = 0.1          # frame 3 of id 1: a shoulder is unsure
sel = appearance.select_rows(tsel)
got = sorted(zip(tsel.track_id[sel].tolist(), tsel.frame[sel].tolist()))
check("every 5th frame, plus each fragment's first and last row with a confident torso (id 1's frame 3 has an unsure shoulder, "
      "so its first is 4)", got == [(1, 4), (1, 5), (1, 10), (1, 15), (1, 20), (1, 23), (2, 0), (2, 5), (2, 9)], str(got))
raises("a stride of 0 is refused", lambda: appearance.select_rows(tsel, 0), "whole number")
check("a different stride changes which rows: every 10th frame", sorted(set(tsel.frame[appearance.select_rows(tsel, 10)].tolist()))
      == [0, 4, 9, 10, 20, 23])
raises("a frame of the wrong size is refused, not misread",
       lambda: appearance.compute_colours(None, tb7, frames=iter([np.zeros((H + 1, W, 3), np.uint8)] * 31)), "frame 0")
raises("a video shorter than the table is refused",
       lambda: appearance.compute_colours(None, tb7, frames=iter([bg()] * 10)), "ended after 10 frames")

print("Colours: save, load and refusals")
col_path = tmp / "x.colours.json"
col7.save(col_path)
back = Colours.load(col_path)
check("save then load gives back exactly the same colours", back == col7 and back.counts() == {"red": 1, "blue": 1, "none": 0, "unsure": 0})
check("the file holds only numbers and the decision (no pixel data of any kind)",
      set(json.loads(col_path.read_text())["fragments"]["1"]) == {"decision", "red", "blue", "rows"})
col7.check_table(tb7)
doc = json.loads(col_path.read_text())


def loads(d, name="m.colours.json"):
    p = tmp / name
    p.write_text(json.dumps(d) if not isinstance(d, str) else d, encoding="utf-8")
    return Colours.load(p)


raises("malformed JSON is refused", lambda: loads("{not json"), "not valid JSON")
assert '"blue": 0.0' in json.dumps(doc)
raises("NaN in a share is refused", lambda: loads(json.dumps(doc).replace('"blue": 0.0', '"blue": NaN', 1)), "not a number")
raises("Infinity is refused", lambda: loads('{"x": Infinity}'), "not a number")
raises("a bare list is refused", lambda: loads("[1, 2]"), "wrong keys")
bad = json.loads(json.dumps(doc))
bad["fragments"]["1"]["decision"] = "purple"
raises("an unknown decision is refused", lambda: loads(bad), "unknown decision")
bad = json.loads(json.dumps(doc))
bad["fragments"]["1"]["decision"] = "blue"
raises("a decision its own medians do not imply (red numbers, 'blue') is refused", lambda: loads(bad), "imply")
bad = json.loads(json.dumps(doc))
bad["fragments"]["1"]["red"] = 1.5
raises("a share above 1 is refused", lambda: loads(bad), "share in [0, 1]")
bad = json.loads(json.dumps(doc))
bad["fragments"]["1"]["rows"] = -1
raises("a negative row count is refused", lambda: loads(bad), "rows")
bad = json.loads(json.dumps(doc))
bad["fragments"]["1"]["rows"] = True
raises("a boolean row count is refused", lambda: loads(bad), "rows")
bad = json.loads(json.dumps(doc))
bad["fragments"]["1"]["extra"] = 1
raises("an extra key in a fragment is refused", lambda: loads(bad), "wrong keys")
bad = json.loads(json.dumps(doc))
bad["fragments"]["x1"] = bad["fragments"].pop("1")
raises("a track id that is not a whole number is refused", lambda: loads(bad), "not a whole number")
bad = json.loads(json.dumps(doc))
bad["n_frames"] = 0
raises("n_frames 0 is refused", lambda: loads(bad), "n_frames")
bad = json.loads(json.dumps(doc))
bad["video_fp"] = "ABC"
raises("a malformed fingerprint is refused", lambda: loads(bad), "video_fp")
bad = json.loads(json.dumps(doc))
bad["format"] = "other"
raises("an unknown format is refused", lambda: loads(bad), "format")
bad = json.loads(json.dumps(doc))
bad["surprise"] = 1
raises("an extra top-level key is refused", lambda: loads(bad), "wrong keys")
_real_max = appearance.MAX_FILE_BYTES
appearance.MAX_FILE_BYTES = 20
raises("a file over the size cap is refused before it is parsed", lambda: Colours.load(col_path), "bytes")
appearance.MAX_FILE_BYTES = _real_max
raises("a colours file cannot be saved with a NaN share (allow_nan=False)",
       lambda: Colours({1: FragmentColour("unsure", float("nan"), 0.0, 3)}, "a.mp4", 10).save(tmp / "nan.json"))
# binding to the table
raises("colours for another source (file name) are refused", lambda: dataclasses.replace(col7, source="GH999999.mp4").check_table(tb7), "GH999999")
raises("colours for another frame count are refused", lambda: dataclasses.replace(col7, n_frames=99).check_table(tb7), "99 frames")
fp_a, fp_b = "a" * 64, "b" * 64
raises("colours with another video fingerprint are refused", lambda: dataclasses.replace(col7, video_fp=fp_a).check_table(tb7), "fingerprint")
tb_fp = dataclasses.replace(tb7, video_fp=fp_a)
raises("a table with a fingerprint against colours made without one is refused", lambda: col7.check_table(tb_fp), "fingerprint")
raises("... and a different one", lambda: dataclasses.replace(col7, video_fp=fp_b).check_table(tb_fp), "fingerprint")
dataclasses.replace(col7, video_fp=fp_a).check_table(tb_fp)
check("... while the same fingerprint is accepted", True)
raises("colours naming other track ids than the table are refused",
       lambda: Colours({1: col7.by_id[1], 99: col7.by_id[2]}, col7.source, col7.n_frames, "").check_table(tb7), "another table")
raises("colours missing a track id of the table are refused",
       lambda: Colours({1: col7.by_id[1]}, col7.source, col7.n_frames, "").check_table(tb7), "another table")

print("the self-check: do co-present credited fragments get opposite colours?")


def mk(colour_of, frags):
    """A table with the given (tid, first, last) fragments and a Colours with the given decisions."""
    tb = table_of([(t, a, b, 80 + 10 * t, 120) for t, a, b in frags], 400)
    nums = {"red": (0.5, 0.0, 10), "blue": (0.0, 0.5, 10), "none": (0.0, 0.0, 10), "unsure": (0.0, 0.0, 1)}
    by = {t: FragmentColour(dd, *nums[dd]) for t, dd in colour_of.items()}
    return tb, Colours(by, tb.source, tb.n_frames, "")


def attrs_of(*tids):
    return [SimpleNamespace(track_id=t) for t in tids] + [SimpleNamespace(track_id=None)]


tb_a, col_a = mk({1: "red", 2: "blue"}, [(1, 0, 99), (2, 0, 99)])
c_a = appearance.colour_check(tb_a, attrs_of(1, 2, 1), col_a)
check("two co-present credited fragments of opposite colour: one decided pair, opposite share exactly 1.0",
      c_a["copresent_pairs"] == 1 and c_a["decided_pairs"] == 1 and c_a["opposite"] == 1 and c_a["opposite_share"] == 1.0
      and c_a["same_share"] == 0.0, str(c_a))
tb_b, col_b = mk({1: "red", 2: "red"}, [(1, 0, 99), (2, 0, 99)])
c_b = appearance.colour_check(tb_b, attrs_of(1, 2), col_b)
check("the same colour: the opposite share is exactly 0.0 (a test that calls two people alike fails the check)",
      c_b["decided_pairs"] == 1 and c_b["opposite_share"] == 0.0 and c_b["same_share"] == 1.0, str(c_b))
tb_c, col_c = mk({1: "red", 2: "blue"}, [(1, 0, 99), (2, 100, 199)])
c_c = appearance.colour_check(tb_c, attrs_of(1, 2), col_c)
check("fragments that are never on screen together are not a pair at all", c_c["copresent_pairs"] == 0 and c_c["decided_pairs"] == 0
      and c_c["opposite_share"] is None, str(c_c))
tb_d, col_d = mk({1: "red", 2: "blue"}, [(1, 0, 99), (2, 75, 199)])        # overlap frames 75..99 = 25 frames = 0.5 s
tb_e, col_e = mk({1: "red", 2: "blue"}, [(1, 0, 99), (2, 76, 199)])        # 24 frames
check("co-present for exactly 0.5 s (25 frames at 50 fps) is a pair; 24 frames is not",
      appearance.colour_check(tb_d, attrs_of(1, 2), col_d)["copresent_pairs"] == 1
      and appearance.colour_check(tb_e, attrs_of(1, 2), col_e)["copresent_pairs"] == 0)
tb_f, col_f = mk({1: "red", 2: "none", 3: "unsure", 4: "blue"}, [(1, 0, 99), (2, 0, 99), (3, 0, 99), (4, 0, 99)])
c_f = appearance.colour_check(tb_f, attrs_of(1, 2, 3, 4), col_f)
check("a none or unsure fragment makes a co-present pair but not a DECIDED one: 6 pairs, 1 decided (red vs blue)",
      c_f["copresent_pairs"] == 6 and c_f["decided_pairs"] == 1 and c_f["opposite"] == 1
      and c_f["credited_by_decision"] == {"red": 1, "blue": 1, "none": 1, "unsure": 1}, str(c_f))
c_g = appearance.colour_check(tb_f, attrs_of(1, 4), col_f)
check("only fragments credited with a punch are paired: id 2 and 3 carry none, so just (1, 4)",
      c_g["copresent_pairs"] == 1 and c_g["credited_fragments"] == 2, str(c_g))
c_h = appearance.colour_check(tb_f, [SimpleNamespace(track_id=None)], col_f)
check("no credited punch: no pairs and the shares are None, never 0%", c_h["copresent_pairs"] == 0 and c_h["opposite_share"] is None
      and c_h["same_share"] is None and c_h["credited_fragments"] == 0)
raises("the check refuses colours made from another table", lambda: appearance.colour_check(tb_f, attrs_of(1), col_a), "another table")
pool = appearance.pool_checks([c_a, c_b, c_c])
check("pooling divides the integer counts once: 2 decided pairs, 1 opposite, 1 same, share 0.5",
      pool["decided_pairs"] == 2 and pool["opposite"] == 1 and pool["same"] == 1 and pool["opposite_share"] == 0.5
      and pool["videos"] == 3, str(pool))
check("pooling nothing decided gives None", appearance.pool_checks([c_c])["opposite_share"] is None)

print("olympic_colour.py end to end, on a synthetic dataset (painted shapes, written once to a temp clip)")
import io  # noqa: E402
import contextlib  # noqa: E402
from fractions import Fraction  # noqa: E402

import olympic_colour  # noqa: E402
from boxingvi import olympic, video  # noqa: E402

NF = 200
two = [(1, 0, NF - 1, 80, 120), (2, 0, NF - 1, 240, 120), (3, 0, NF - 1, 160, 200)]     # red, blue, white referee
LOOK = {1: (RED, SKIN, RED), 2: (BLUE, SKIN, BLUE), 3: (WHITE, SKIN, SKIN)}
ds = tmp / "ds" / "task_kam2_gh000002"
(ds / "data").mkdir(parents=True)
with video.VideoWriter(ds / "data" / "GH000002.mp4", W, H, Fraction(50)) as vw:
    for i in range(NF):
        fr = bg()
        for tid, f0, f1, cx, cy in two:
            paint(fr, cx, cy, *LOOK[tid])
        vw.write(fr)
tab2 = table_of(two, NF, source="GH000002.mp4", fp=video.fingerprint(ds / "data" / "GH000002.mp4"))
labels = {v: k for k, v in olympic.TARGETS.items()}
hands = {v: k for k, v in olympic.HANDS.items()}
evs = []
for f0, cx in ((20, 80), (60, 240), (100, 80), (140, 240)):      # a glove box on the person's left wrist (cx - 45, 120)
    evs.append(olympic.Event(f0, f0 + 2, "head", "left", (cx - 53, 112, cx - 37, 128), 1))
tracks_json = []
for e in evs:
    x1, y1, x2, y2 = e.box
    shp = lambda frame, out: {"type": "rectangle", "frame": frame, "outside": out, "points": [x1, y1, x2, y2]}  # noqa: E731
    tracks_json.append({"label": f"{labels[e.target]} {hands[e.hand]}", "shapes": [shp(e.first, False), shp(e.last + 1, True)]})
(ds / "annotations.json").write_text(json.dumps([{"tracks": tracks_json}]), encoding="utf-8")
trk = tmp / "trk"
trk.mkdir()
tab2.save(trk / "GH000002.tracks.npz")
root = tmp / "ds"
chk_out = tmp / "res" / "check.json"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = olympic_colour.main(["--data", str(root), "--tracks", str(trk), "--check-out", str(chk_out)])
col2 = Colours.load(trk / "GH000002.colours.json")
check("exit 0; the colours file is bound to the table (name, length, fingerprint, ids)",
      rc == 0 and col2.source == "GH000002.mp4" and col2.n_frames == NF and col2.video_fp == tab2.video_fp
      and set(col2.by_id) == {1, 2, 3}, buf.getvalue())
check("REAL FACT: read from the clip's pixels, id 1 is red, id 2 is blue, the white-shirt referee is none",
      [col2.decision(t) for t in (1, 2, 3)] == ["red", "blue", "none"], str(col2.by_id))
chk = json.loads(chk_out.read_text(encoding="utf-8"))
check("the self-check: the one co-present credited pair is red vs blue; opposite share 1.0; the gate passes",
      chk["passes"] is True and chk["all"]["copresent_pairs"] == 1 and chk["all"]["opposite"] == 1
      and chk["all"]["opposite_share"] == 1.0 and chk["all"]["credited_by_decision"] == {"red": 1, "blue": 1, "none": 0, "unsure": 0},
      str(chk["all"]))
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    rc2 = olympic_colour.main(["--data", str(root), "--tracks", str(trk), "--check-out", str(tmp / "res" / "check2.json"), "--check-only"])
check("--check-only recomputes the same numbers from the colours file, without the video",
      rc2 == 0 and json.loads((tmp / "res" / "check2.json").read_text(encoding="utf-8"))["all"] == chk["all"])
Colours({1: FragmentColour("red", 0.5, 0.0, 10), 2: FragmentColour("red", 0.5, 0.0, 10), 3: col2.by_id[3]},
        col2.source, col2.n_frames, col2.video_fp).save(trk / "GH000002.colours.json")
with contextlib.redirect_stdout(io.StringIO()):
    rc_bad = olympic_colour.main(["--data", str(root), "--tracks", str(trk), "--check-out", str(tmp / "res" / "bad.json"), "--check-only"])
bad = json.loads((tmp / "res" / "bad.json").read_text(encoding="utf-8"))
check("REAL FACT, the gate can FAIL: two co-present boxers both called red give opposite share 0.0, passes false, exit code 2",
      rc_bad == 2 and bad["passes"] is False and bad["all"]["opposite_share"] == 0.0 and bad["all"]["same_share"] == 1.0, str(bad["all"]))
col2.save(trk / "GH000002.colours.json")
ren = tmp / "trk_renamed"
ren.mkdir()
dataclasses.replace(tab2, source="GH999999.mp4").save(ren / "GH000002.tracks.npz")
with contextlib.redirect_stdout(io.StringIO()):
    raises("a renamed (unbound) table is refused, nothing is coloured",
           lambda: olympic_colour.main(["--data", str(root), "--tracks", str(ren), "--check-out", str(tmp / "res" / "c3.json")]), "could be read")
check("... and no colours file was written for it", not (ren / "GH000002.colours.json").exists())
wrongfp = tmp / "trk_fp"
wrongfp.mkdir()
dataclasses.replace(tab2, video_fp="d" * 64).save(wrongfp / "GH000002.tracks.npz")
with contextlib.redirect_stdout(io.StringIO()):
    raises("a table with another recording's fingerprint is refused too",
           lambda: olympic_colour.main(["--data", str(root), "--tracks", str(wrongfp), "--check-out", str(tmp / "res" / "c4.json")]),
           "could be read")
for bad_id in ("../GH000002", "GH 2", "", "GH000002.mp4"):
    raises(f"--only {bad_id!r} is not a GoPro id: refused before anything is read or written",
           lambda b=bad_id: olympic_colour.main(["--data", str(root), "--tracks", str(trk), "--only", b,
                                                 "--check-out", str(tmp / "res" / "c5.json")]), "GoPro id")
nocol = tmp / "trk_nocol"
nocol.mkdir()
tab2.save(nocol / "GH000002.tracks.npz")
with contextlib.redirect_stdout(io.StringIO()):
    raises("--check-only with no colours file is refused, not treated as an empty pass",
           lambda: olympic_colour.main(["--data", str(root), "--tracks", str(nocol), "--check-only",
                                        "--check-out", str(tmp / "res" / "c6.json")]), "could be read")
check("the colours path stays inside the tracks folder", olympic_colour.colours_path(trk, "GH000002").parent == trk.resolve())
raises("colours_path refuses a path-like id", lambda: olympic_colour.colours_path(trk, "..\\x"), "GoPro id")

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
