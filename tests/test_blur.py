"""Tests for the face blur (a privacy control: it must fail closed).

Synthetic frames carry planted "faces" -- 2-pixel checkerboards, the
highest-frequency texture there is -- at known places, and an injected predictor
reports people at them. A face counts as blurred only if the PIXELS say so: the
checkerboard's high-frequency energy is gone, not merely the plan claiming a
region. Weighted toward the failure paths (each must blur more, never less) and
the output-location refusals, each of which asserts WHICH rule refused (a test
that passes because an earlier rule fired proves nothing). No model needed;
ffmpeg is.

Run:  python tests/test_blur.py
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))

import blur_faces  # noqa: E402
from boxingvi import faceblur, pose_yolo, video  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.faceblur import FramePlan, PredictorError, UnsafeOutput  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def refused(name: str, reason: str, fn) -> None:
    """`fn` must raise UnsafeOutput for exactly `reason`."""
    try:
        fn()
    except UnsafeOutput as e:
        check(name, e.reason == reason, f"refused for {e.reason!r}, expected {reason!r}")
    except Exception as e:  # wrong exception type is a failure too
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


W, H = 320, 240
FACE = 12  # planted face half-size at scale 1: a 24-px head on a 170-px body


def person(cx, cy, conf=0.9, s=1.0, cut=None):
    """Box, keypoints and confidences of a person whose face is centred at (cx, cy).
    `s` scales the body; `cut` crops the box: "shoulders" (head and shoulders only,
    head 0.4 of the box width) or "head" (a close-up, head 0.67 of the box width)."""
    if cut == "head":
        box = np.array([cx - 18 * s, cy - 20 * s, cx + 18 * s, cy + 30 * s], float)
    elif cut == "shoulders":
        box = np.array([cx - 30 * s, cy - 20 * s, cx + 30 * s, cy + 40 * s], float)
    else:
        box = np.array([cx - 30 * s, cy - 20 * s, cx + 30 * s, cy + 150 * s], float)
    off = {0: (0, 2), 1: (-5, -3), 2: (5, -3), 3: (-10, 0), 4: (10, 0),
           5: (-25, 30), 6: (25, 30), 7: (-28, 60), 8: (28, 60), 9: (-20, 40), 10: (20, 40),
           11: (-15, 100), 12: (15, 100), 13: (-15, 125), 14: (15, 125), 15: (-15, 148),
           16: (15, 148)}
    kxy = np.array([(cx + s * off[j][0], cy + s * off[j][1]) for j in range(N_JOINTS)], float)
    kc = np.full(N_JOINTS, conf, float)
    return box, kxy, kc


def pred_of(people):
    if not people:
        return np.zeros((0, 4)), np.zeros((0, N_JOINTS, 2)), np.zeros((0, N_JOINTS))
    return (np.stack([p[0] for p in people]), np.stack([p[1] for p in people]),
            np.stack([p[2] for p in people]))


def scene(faces, w=W, h=H):
    """A grey frame, a flat body under each face, and a checkerboard face at each
    centre. A face is (cx, cy) or (cx, cy, half-size)."""
    f = np.full((h, w, 3), 100, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // 2) + (xx // 2)) % 2 * 255).astype(np.uint8)
    for face in faces:
        cx, cy, r = (*face, FACE)[:3]
        f[max(0, cy + r + 3):cy + 150, max(0, cx - 30):cx + 30] = (60, 90, 140)
        ys, xs = slice(max(0, cy - r), cy + r), slice(max(0, cx - r), cx + r)
        f[ys, xs] = checker[ys, xs, None]
    return f


def hf(a) -> float:
    g = a.astype(np.float64).mean(axis=2)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def face_px(frame, cx, cy, r=FACE):
    return frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]


def destroyed(before, after, cx, cy, r=FACE, limit=0.2) -> bool:
    """The planted face's high-frequency energy fell below `limit` of what it was."""
    return hf(face_px(after, cx, cy, r)) < limit * hf(face_px(before, cx, cy, r))


def covered(plan, cx, cy, r=FACE, h=H, w=W) -> bool:
    return bool(faceblur.blurred_mask(plan, h, w)[max(0, cy - r):cy + r,
                                                  max(0, cx - r):cx + r].all())


tmp = Path(tempfile.mkdtemp())
junctions = []


@atexit.register
def _cleanup():  # junctions first, so removing the tree never follows one
    for j in junctions:
        try:
            os.rmdir(j)
        except OSError:
            pass
    shutil.rmtree(tmp, ignore_errors=True)


A, B = (80, 60), (240, 70)
two = scene([A, B])

print("keypoint patches")
p = faceblur.plan_frame(pred_of([person(*A), person(*B)]), W, H)
out = faceblur.apply_plan(two, p)
check("two separate people with faces -> keypoint mode, both counted, box floor + face patch each",
      p.mode == "keypoints" and p.persons == 2 and len(p.regions) == 4, str(p))
check("both planted faces lie wholly inside the blur", covered(p, *A) and covered(p, *B))
check("both faces' texture is destroyed in the pixels",
      destroyed(two, out, *A) and destroyed(two, out, *B),
      f"{hf(face_px(out, *A)):.1f} vs {hf(face_px(two, *A)):.1f}")
check("apply_plan leaves its input untouched", hf(face_px(two, *A)) > 100)
check("pixels outside every region are unchanged",
      np.array_equal(out[~faceblur.blurred_mask(p, H, W)], two[~faceblur.blurred_mask(p, H, W)]))

side = person(*A)  # bladed stance: shoulders overlap, far eye and ear hidden
side[1][5] = side[1][6] = (A[0], A[1] + 30)
side[2][[2, 4]] = 0.05
p = faceblur.plan_frame(pred_of([side]), W, H)
check("side view (shoulder width ~0, one eye) still covers the face",
      p.mode == "keypoints" and covered(p, *A), str(p))
lone = person(*A, conf=0.05)  # only the nose is confident; no shoulders, no hips
lone[2][0] = 0.9
p = faceblur.plan_frame(pred_of([lone]), W, H)
check("a single confident nose, no body joints: sized from the box, still covers",
      p.mode == "keypoints" and covered(p, *A), str(p))
for ear in (faceblur.L_EAR, faceblur.R_EAR):  # regression: PAD 0.6 left a 2-px sliver
    one_ear = person(*A, conf=0.05)
    one_ear[2][ear] = 0.9
    p = faceblur.plan_frame(pred_of([one_ear]), W, H)
    out = faceblur.apply_plan(scene([A]), p)
    check(f"only one ear confident (joint {ear}): the patch reaches across the whole face",
          covered(p, *A) and destroyed(scene([A]), out, *A), str(p))
# Regression (review B1): a head-and-shoulders box, legs out of frame, one confident
# nose. The box-height floor assumed a whole body and covered 31% of the head.
bw_, bh_ = 400, 300
big = np.full((bh_, bw_, 3), 100, np.uint8)
yy, xx = np.mgrid[0:bh_, 0:bw_]
big[110:190, 117:182] = ((((yy // 2) + (xx // 2)) % 2) * 255).astype(np.uint8)[110:190, 117:182, None]
kxy = np.zeros((N_JOINTS, 2))
kxy[:] = (150, 150)
kc = np.zeros(N_JOINTS)
kc[faceblur.NOSE] = 0.9
p = faceblur.plan_frame((np.array([[100, 100, 200, 220]]), kxy[None], kc[None]), bw_, bh_)
m = faceblur.blurred_mask(p, bh_, bw_)
check("head-and-shoulders box, nose only: the whole 65x80 head is covered (was 31%)",
      bool(m[110:190, 117:182].all()), f"{m[110:190, 117:182].mean():.0%} {p}")
x0, y0, x1, y1 = faceblur.plan_frame(pred_of([person(*A)]), W, H).regions[1]  # the face patch
check("face points confident: the patch is HEAD-sized, 1.6 to 2.4 head widths across",
      1.6 * 2 * FACE <= min(x1 - x0, y1 - y0) and max(x1 - x0, y1 - y0) <= 2.4 * 2 * FACE,
      f"{x1 - x0}x{y1 - y0} for a {2 * FACE}-px head")
# Utility (Brendan, 2026-10-20): the guard and chest below a whole boxer's head stay
# visible -- the body-sized patch and top-third floor hid them, and they are where
# feints show.
p = faceblur.plan_frame(pred_of([person(*A)]), W, H)
m = faceblur.blurred_mask(p, H, W)
chest = m[A[1] + 40:A[1] + 120, A[0] - 30:A[0] + 30]
check("a whole boxer's chest and guard (40-120 px below the eyes) are not mosaicked",
      not chest.any(), f"{chest.mean():.0%} of the chest mosaicked")
for (j1, j2), name in (((faceblur.L_EYE, faceblur.R_EYE), "both eyes"),
                       ((faceblur.NOSE, faceblur.L_EYE), "nose + one eye"),
                       ((faceblur.L_EYE, faceblur.L_EAR), "eye + ear, same side"),
                       ((faceblur.NOSE, faceblur.R_EAR), "nose + ear")):
    two_pts = person(*A, conf=0.05)
    two_pts[2][[j1, j2]] = 0.9
    p = faceblur.plan_frame(pred_of([two_pts]), W, H)
    out = faceblur.apply_plan(scene([A]), p)
    check(f"two face points ({name}): the head-sized patch still covers the whole face",
          covered(p, *A) and destroyed(scene([A]), out, *A), str(p))
huge = person(*A, s=1.4)  # a bigger head: the patch grows with the face points
hx0, hy0, hx1, hy1 = faceblur.plan_frame(pred_of([huge]), W, H).regions[1]
check("the head-sized patch scales with the face (1.4x the person, wider patch)",
      hx1 - hx0 > x1 - x0, f"{hx1 - hx0} vs {x1 - x0}")
clump = person(*A, conf=0.05)  # two face points almost on top of each other: no size info
clump[1][[faceblur.L_EYE, faceblur.R_EYE]] = [(A[0], A[1]), (A[0] + 0.5, A[1])]
clump[2][[faceblur.L_EYE, faceblur.R_EYE]] = 0.9
f = faceblur.face_patch(clump[0], clump[1], clump[2], W, H)
check("two face points on top of each other: no head size, the generous body-sized patch",
      faceblur.head_width([faceblur.L_EYE, faceblur.R_EYE], clump[1], 170) is None
      and f.region[2] - f.region[0] >= 2.5 * 2 * FACE, str(f))

print("the box is the floor: keypoints only add blur (blocker report, option 1)")
chest = person(*A)  # confident face points, all on the chest: the skeleton is wrong
chest[1][list(faceblur.FACE_JOINTS)] = (A[0], A[1] + 70)
p = faceblur.plan_frame(pred_of([chest]), W, H)
out = faceblur.apply_plan(scene([A]), p)
check("face keypoints confidently on the chest: the head is still covered (by the box floor)",
      p.mode == "keypoints" and covered(p, *A) and destroyed(scene([A]), out, *A), str(p))
# Review P3: a tall crop (passes the 2:1 test), nose only, bladed shoulders, hips and
# a knee imagined inside the box. Three layers of wrong joints left 80% covered.
tall_k, tall_c = np.zeros((N_JOINTS, 2)), np.zeros(N_JOINTS)
tall_k[:] = (150, 150)
tall_c[faceblur.NOSE] = 0.9
tall_k[[5, 6]] = [(148, 200), (152, 200)]
tall_c[[5, 6]] = 0.5
tall_k[[11, 12]] = [(149, 240), (151, 240)]
tall_c[[11, 12]] = 0.35
tall_k[13] = (150, 290)
tall_c[13] = 0.35
p = faceblur.plan_frame((np.array([[100, 100, 200, 300]]), tall_k[None], tall_c[None]), bw_, bh_)
m = faceblur.blurred_mask(p, bh_, bw_)
check("tall crop with three layers of wrong joints: the whole head is covered (was 80%)",
      bool(m[110:190, 117:182].all()), f"{m[110:190, 117:182].mean():.0%} {p}")
# Review H1: a head-and-neck close-up at exactly 2:1, head as wide as the box
# (100x125), no confident face points (turned away). One box width left 96%.
big2 = np.full((bh_, bw_, 3), 100, np.uint8)
yy2, xx2 = np.mgrid[0:bh_, 0:bw_]
big2[100:225, 100:200] = ((((yy2 // 2) + (xx2 // 2)) % 2) * 255).astype(np.uint8)[100:225, 100:200, None]
p = faceblur.plan_frame((np.array([[100, 100, 200, 300]]), np.full((1, N_JOINTS, 2), 150.0),
                         np.zeros((1, N_JOINTS))), bw_, bh_)
out = faceblur.apply_plan(big2, p)
m = faceblur.blurred_mask(p, bh_, bw_)
check("(the H1 and P3 crops touch the frame's bottom edge, as a close-up crop does)",
      faceblur._touches_edge((100, 100, 200, 300), bw_, bh_))
check("box-wide head in a 2:1 crop at the frame edge, no face points: whole head under the floor (was 96%)",
      p.mode == "box" and bool(m[100:225, 100:200].all())
      and hf(out[100:225, 100:200]) < 0.2 * hf(big2[100:225, 100:200]),
      f"{m[100:225, 100:200].mean():.1%} {p}")
# Option (c), then the top fifth (Brendan, 2026-10-20): away from the frame edge a
# whole boxer's floor is the top fifth (+5%), not 1.5 box widths or a third -- the
# guard and torso stay visible for feint labels.
mid, mk, mc_ = person(*A)  # box (50, 40, 110, 210) in a 320x240 frame: touches no edge
sh_y = mk[[5, 6], 1].mean()
fx0, fy0, fx1, fy1 = faceblur.box_floor_region(mid, W, H, mk, mc_)
check("mid-frame whole boxer: the floor stops at the shoulder line + 5% (guard and torso visible)",
      not faceblur._touches_edge(mid, W, H)
      and fy1 <= np.ceil(max(mid[1] + faceblur.FLOOR_FRAC * (mid[3] - mid[1]), sh_y)
                         + faceblur.FLOOR_BELOW * (mid[3] - mid[1])),
      f"floor ends at y={fy1}, shoulders at {sh_y}, box {mid}")
# Review R1: arms raised, so the box top is the gloves and the head sits at ~0.3-0.4
# of the box -- below the top fifth. Face points on a glove (wrong skeleton): the
# head was 0% covered. The floor now reaches the shoulder line.
W3, H3 = 320, 420
up_box = np.array([100.0, 40.0, 160.0, 400.0])  # gloves at y=40, head at 145-175
up_k = np.zeros((N_JOINTS, 2))
up_k[:] = (130, 300)
up_k[list(faceblur.FACE_JOINTS)] = [(130, 52), (125, 48), (135, 48), (120, 50), (140, 50)]  # on the glove
up_k[[5, 6]] = [(115, 190), (145, 190)]  # shoulders, correctly placed
up_c = np.full(N_JOINTS, 0.9)
up_scene = scene([(130, 160, 15)], W3, H3)
p = faceblur.plan_frame((up_box[None], up_k[None], up_c[None]), W3, H3)
out = faceblur.apply_plan(up_scene, p)
check("arms up, face points on a glove: the real head is under the shoulder-anchored floor (was 0%)",
      p.mode != "full" and covered(p, 130, 160, 15, H3, W3) and destroyed(up_scene, out, 130, 160, 15),
      str(p))
no_sh = up_c.copy()
no_sh[[5, 6]] = 0.1
fx0, fy0, fx1, fy1 = faceblur.box_floor_region(up_box, W3, H3, up_k, no_sh)
check("... and with no confident shoulder the floor falls back to a third + 10%",
      fy1 >= up_box[1] + (up_box[3] - up_box[1]) * (1 / 3 + 0.10) - 1, f"floor ends at {fy1}")
# Review of R1: a skeleton fitted up the raised arm -- face on the glove AND shoulders
# on the elbows -- is ONE failure, and brought the floor back to the top fifth (0%).
for name, ubox in (("arms up", up_box), ("referee raising one arm", np.array([100.0, 40.0, 177.0, 400.0]))):
    shifted = up_k.copy()
    shifted[[5, 6]] = [(115, 80), (145, 80)]  # shoulders on the elbows
    p = faceblur.plan_frame((ubox[None], shifted[None], up_c[None]), W3, H3)
    out = faceblur.apply_plan(up_scene, p)
    check(f"{name} ({(ubox[3] - ubox[1]) / (ubox[2] - ubox[0]):.1f}:1), whole skeleton shifted up the "
          "arm: the head is still covered (was 0%)",
          covered(p, 130, 160, 15, H3, W3) and destroyed(up_scene, out, 130, 160, 15), str(p))
one_high = up_k.copy()
one_high[5] = (115, 80)  # one shoulder on an elbow, the other correct
fx0, fy0, fx1, fy1 = faceblur.box_floor_region(np.array([100.0, 40.0, 200.0, 400.0]), W3, H3, one_high, up_c)
check("one shoulder wrongly high: the floor follows the LOWER shoulder (the mean gave 47%)",
      fy1 >= 190, f"floor ends at {fy1}")
guard = np.array([50.0, 40.0, 100.0, 210.0])  # 3.4:1, a side-on guard
gx0, gy0, gx1, gy1 = faceblur.box_floor_region(guard, W, H, mk, mc_)
check("a side-on guard box (3.4:1) keeps the shallow floor (the arms-up rule starts at 4:1)",
      gy1 <= np.ceil(max(guard[1] + 0.2 * 170, sh_y) + 0.05 * 170), f"floor ends at {gy1}")
edge = mid.copy()
edge[3] = H  # the same boxer, feet cut off by the frame's bottom edge
ex0, ey0, ex1, ey1 = faceblur.box_floor_region(edge, W, H, mk, mc_)
check("... and the same box touching the frame edge reaches 1.5 box widths",
      ey1 >= edge[1] + 1.5 * (edge[2] - edge[0]), f"floor ends at y={ey1}, box {edge}")
# Review of option (c): the model places edges to ~1 model pixel (imgsz 640), so a
# crop's box can stop a few frame pixels short of the border. With a fixed 2 px band
# a crop 3 px short lost the extension (70% of the head covered).
for fw, fh, short in ((1080, 1920, 3), (1920, 1080, 3), (3840, 2160, 6), (2160, 3840, 6)):
    hx0, hy0 = fw // 2 - 50, fh - 200 - short  # box-wide head in a 2:1 crop, bottom edge
    crop = np.array([[hx0, hy0, hx0 + 100, fh - short]], float)
    pc = faceblur.plan_frame((crop, np.full((1, N_JOINTS, 2), 1.0) * crop[0, :2],
                              np.zeros((1, N_JOINTS))), fw, fh)
    mc = faceblur.blurred_mask(pc, fh, fw)
    head = mc[hy0:hy0 + 125, hx0:hx0 + 100]
    check(f"{fw}x{fh}: a crop stopping {short} px short of the edge still gets the whole head",
          bool(head.all()), f"{head.mean():.1%} {pc}")
rng_f = np.random.default_rng(7)
floor_bad = []
for t in range(500):
    people = []
    for i in range(int(rng_f.integers(1, 4))):
        b, k, _ = person(60 + 100 * i, int(40 + rng_f.integers(0, 30)),
                         cut=rng_f.choice([None, "shoulders", "head"]))
        k = k + rng_f.normal(0, 25, k.shape)  # any skeleton, however wrong
        people.append((b, k, rng_f.uniform(0, 1, N_JOINTS)))
    p = faceblur.plan_frame(pred_of(people), W, H)
    if p.mode == "full":
        continue
    m = faceblur.blurred_mask(p, H, W)
    for b, k, c in people:
        x0, y0, x1, y1 = faceblur.box_floor_region(b, W, H, k, c)
        bwi, bhi = b[2] - b[0], b[3] - b[1]  # the floor no keypoint can shrink: top fifth + 5%
        mx0, my0 = max(0, int(np.floor(b[0] - 0.1 * bwi))), max(0, int(np.floor(b[1] - 0.1 * bhi)))
        mx1, my1 = min(W, int(np.ceil(b[2] + 0.1 * bwi))), min(H, int(np.ceil(b[1] + 0.25 * bhi)))
        if not (m[y0:y1, x0:x1].all() and m[my0:my1, mx0:mx1].all()):
            floor_bad.append(t)
check("keypoints only add: every person's box floor, and at least its top fifth + 5%, is blurred "
      "in every non-full plan (500 scenes)", not floor_bad, str(floor_bad[:5]))

print("box fallback (no confident face keypoints)")
noface = person(*A)
noface[2][list(faceblur.FACE_JOINTS)] = 0.2
p = faceblur.plan_frame(pred_of([noface, person(*B)]), W, H)
out = faceblur.apply_plan(two, p)
check("a person without face keypoints -> box mode", p.mode == "box", str(p))
check("... the top third of their box covers the face, in the pixels",
      covered(p, *A) and destroyed(two, out, *A))
check("... and the other person is still blurred from keypoints",
      covered(p, *B) and destroyed(two, out, *B))
closeup = person(*A, cut="head")  # box barely wider than the head, no face keypoints
closeup[2][list(faceblur.FACE_JOINTS)] = 0.2
p = faceblur.plan_frame(pred_of([closeup]), W, H)
out = faceblur.apply_plan(scene([A]), p)
check("a cut-off (not body-shaped) box without face keypoints: the WHOLE box, chin included",
      p.mode == "box" and covered(p, *A) and destroyed(scene([A]), out, *A)
      and p.regions[0][3] >= closeup[0][3], str(p))


class _Tensor:  # just enough of a torch tensor: .cpu().numpy()
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, a):
        self.xyxy = _Tensor(a)

    def __len__(self):
        return len(self.xyxy.a)


# An ultralytics-shaped model whose keypoints carry NO confidences
fake_model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(
    boxes=_Boxes(person(*A)[0][None]),
    keypoints=SimpleNamespace(xy=_Tensor(person(*A)[1][None]), conf=None))])
bxs, kx, kcs = pose_yolo.yolo_predictor(fake_model, missing_conf=0.0)(two)
p = faceblur.plan_frame((bxs, kx, kcs), W, H)
check("a model reporting no keypoint confidences: blur treats them as unconfident (box mode)",
      p.mode == "box" and covered(p, *A), str(p))
check("... while ingest keeps its old default (confidences of 1)",
      bool((pose_yolo.yolo_predictor(fake_model)(two)[2] == 1).all()))

print("whole-frame fallbacks (each must blur MORE)")
full_cases = {
    "nobody found": pred_of([]),
    "predictor raised": PredictorError(),
    "not a 3-tuple": (np.zeros((1, 4)),),
    "shape mismatch": (np.zeros((2, 4)), np.zeros((1, N_JOINTS, 2)), np.zeros((1, N_JOINTS))),
    "N=0 boxes with keypoints": (np.zeros((0, 4)), np.zeros((1, N_JOINTS, 2)), np.zeros((1, N_JOINTS))),
    "NaN keypoint": (lambda b, k, c: (b[None], np.where(np.arange(N_JOINTS)[:, None] == 0, np.nan, k)[None], c[None]))(*person(*A)),
    "huge finite keypoint (overflow)": (lambda b, k, c: (b[None], np.where(np.arange(N_JOINTS)[:, None] == 5, 1e308, k)[None], c[None]))(*person(*A)),
    "confidence above 1": (lambda b, k, c: (b[None], k[None], c[None] * 2))(*person(*A)),
    "inverted box": (lambda b, k, c: (b[None, [2, 3, 0, 1]], k[None], c[None]))(*person(*A)),
    "box outside the frame": (lambda b, k, c: (b[None] + 1000, k[None], c[None]))(*person(*A)),
    "string garbage": "not a prediction",
}
for name, pred in full_cases.items():
    p = faceblur.plan_frame(pred, W, H)
    out = faceblur.apply_plan(two, p)
    check(f"{name} -> whole frame, faces destroyed",
          p.mode == "full" and faceblur.blurred_mask(p, H, W).all()
          and destroyed(two, out, *A) and destroyed(two, out, *B), str(p))

clinch_a, clinch_b = person(*A), person(A[0] + 40, A[1] + 10)
clinch_b[2][list(faceblur.FACE_JOINTS)] = 0.1
p = faceblur.plan_frame(pred_of([clinch_a, clinch_b]), W, H)
check("overlapping boxes, one without face keypoints -> whole frame",
      p.mode == "full" and p.reason == "overlap_without_face_keypoints", str(p))
# Review B2: in a clinch both skeletons put their face points on A's head; B's
# real head is bare unless the frame is blurred whole.
shared_a, shared_b = person(*A), person(A[0] + 40, A[1] + 10)
shared_b[1][list(faceblur.FACE_JOINTS)] = shared_a[1][list(faceblur.FACE_JOINTS)] + 2
p = faceblur.plan_frame(pred_of([shared_a, shared_b]), W, H)
both = scene([A, (A[0] + 40, A[1] + 10)])
check("clinch, both skeletons on ONE head -> whole frame, the other head destroyed too",
      p.mode == "full" and p.reason == "ambiguous_faces"
      and destroyed(both, faceblur.apply_plan(both, p), A[0] + 40, A[1] + 10), str(p))
p = faceblur.plan_frame(pred_of([person(*A), person(A[0] + 40, A[1] + 10)]), W, H)
near = scene([A, (A[0] + 40, A[1] + 10)])
outn = faceblur.apply_plan(near, p)
check("overlapping boxes, two distinct heads 40 px apart: whatever the mode, BOTH heads destroyed",
      covered(p, *A) and covered(p, A[0] + 40, A[1] + 10)
      and destroyed(near, outn, *A) and destroyed(near, outn, A[0] + 40, A[1] + 10), str(p))
# Review N1: side by side, boxes NOT touching, but B's skeleton puts its face on A's
# head (16 px outside B's box, within the slack). B's real head was 0% covered.
W2, H2 = 320, 420
nA = person(145, 122)
nA[0][:] = (100, 100, 160, 400)
nB = person(225, 125)
nB[0][:] = (162, 105, 282, 405)
nB[1][list(faceblur.FACE_JOINTS)] = nA[1][list(faceblur.FACE_JOINTS)] + 1
p = faceblur.plan_frame(pred_of([nA, nB]), W2, H2)
side_by_side = scene([(145, 122), (225, 125)], W2, H2)
check("boxes apart, one skeleton on the neighbour's head -> whole frame, B's head destroyed",
      p.mode == "full" and p.reason == "ambiguous_faces"
      and destroyed(side_by_side, faceblur.apply_plan(side_by_side, p), 225, 125), str(p))
# Review N2: the B1 crop again, but the model imagines hips just under the shoulders
# inside the crop, so a (tiny) torso is "measured". Was 31% covered.
kxy2, kc2 = kxy.copy(), kc.copy()
kxy2[[faceblur.L_SHOULDER, faceblur.R_SHOULDER]] = [(148, 200), (152, 200)]
kc2[[faceblur.L_SHOULDER, faceblur.R_SHOULDER]] = 0.5
kxy2[[faceblur.L_HIP, faceblur.R_HIP]] = [(149, 216), (151, 216)]
kc2[[faceblur.L_HIP, faceblur.R_HIP]] = 0.35
p = faceblur.plan_frame((np.array([[100, 100, 200, 220]]), kxy2[None], kc2[None]), bw_, bh_)
m = faceblur.blurred_mask(p, bh_, bw_)
check("crop with imagined in-box hips: the whole head is still covered (was 31%)",
      bool(m[110:190, 117:182].all()), f"{m[110:190, 117:182].mean():.0%} {p}")
# Review P1: A's skeleton claims B's head (28 px outside A's box, inside the slack);
# B, head turned, has no face points. Boxes apart. A's real head was 0% covered.
pA = person(100, 120)
pA[0][:] = (40, 100, 160, 400)
pA[1][list(faceblur.FACE_JOINTS)] += (88, 5)
pB = person(195, 125)
pB[0][:] = (165, 105, 225, 405)
pB[2][list(faceblur.FACE_JOINTS)] = 0.1
p = faceblur.plan_frame(pred_of([pA, pB]), W2, H2)
claimed = scene([(100, 120), (195, 125)], W2, H2)
check("one face, on the OTHER person's head, boxes apart -> whole frame, its owner's head destroyed",
      p.mode == "full" and p.reason == "ambiguous_faces"
      and destroyed(claimed, faceblur.apply_plan(claimed, p), 100, 120), str(p))
# Review P2: the B1 crop, mid-frame, with a knee imagined BELOW the box: a "whole
# body" by the joint chain alone. Was 31% covered.
kxy3, kc3 = kxy2.copy(), kc2.copy()
kxy3[13] = (150, 260)
kc3[13] = 0.35
p = faceblur.plan_frame((np.array([[100, 100, 200, 220]]), kxy3[None], kc3[None]), bw_, bh_)
m = faceblur.blurred_mask(p, bh_, bw_)
check("crop with a knee imagined below the box: the whole head is still covered (was 31%)",
      bool(m[110:190, 117:182].all()), f"{m[110:190, 117:182].mean():.0%} {p}")
x0n, y0n, x1n, y1n = faceblur.plan_frame(pred_of([person(*A)]), W, H).regions[0]
check("... while a whole body (legs confident) keeps a body-sized patch, not a box-sized one",
      x1n - x0n < 0.9 * 60 * 1.25 * 2, f"{x1n - x0n}")
reach_a, reach_b = person(*A), person(150, 70)
reach_b[0][0] = 95  # B's jab reaches into A's box; the heads stay apart
p = faceblur.plan_frame(pred_of([reach_a, reach_b]), W, H)
apart = scene([A, (150, 70)])
out = faceblur.apply_plan(apart, p)
check("overlapping boxes, two distinct heads apart -> per-face patches, BOTH faces destroyed",
      p.mode == "keypoints" and covered(p, *A) and covered(p, 150, 70)
      and destroyed(apart, out, *A) and destroyed(apart, out, 150, 70), str(p))
stray = person(*A)
stray[1][list(faceblur.FACE_JOINTS)] += (150, 0)
p = faceblur.plan_frame(pred_of([stray]), W, H)
check("face keypoints far outside the person's own box -> whole frame",
      p.mode == "full" and p.reason == "face_outside_box", str(p))
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, expected_persons=2)
check("fewer people than --expect-persons -> whole frame",
      p.mode == "full" and p.reason == "fewer_persons_than_expected", str(p))
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, neighbour_missing=True)
check("a person missing against the neighbours -> whole frame",
      p.mode == "full" and p.reason == "person_missing_vs_neighbours", str(p))
raises_ok = False
try:
    faceblur.apply_plan(two, FramePlan("none", (), 0))
except ValueError:
    raises_ok = True
check("an unknown mode is refused, not silently passed through", raises_ok)

print("neighbour matching (review B3: by position, one to one)")
bA, bB = person(*A)[0], person(*B)[0]
false_box = np.array([5.0, 5.0, 25.0, 60.0])  # a "person" on the ropes
miss = faceblur.missing_vs_neighbours
check("steady two people: nothing missing",
      not miss([np.stack([bA, bB])] * 5, 2).any())
check("A missed, a false box in its place: the COUNT is unchanged but A is flagged",
      list(miss([np.stack([bA, bB]), np.stack([bB, false_box]), np.stack([bA, bB])], 1))
      == [True, True, True])
check("two people merged into one box that overlaps both: flagged (one box vouches for one)",
      bool(miss([np.stack([bA, bB]), np.array([[bA[0], bA[1], bB[2], bB[3]]])], 1)[1]))
legs_only = np.array([bA[0], bA[1] + 0.55 * (bA[3] - bA[1]), bA[2], bA[3]])
check("a legs-only box does not vouch for a person whose head it lost (top edges disagree)",
      bool(miss([bA[None], legs_only[None]], 1)[1]))
check("a person moving a little between frames is still matched",
      not miss([bA[None], (bA + 6)[None], (bA + 12)[None]], 1).any())
check("a miss is flagged across the window, and only within it",
      list(miss([bA[None]] + [np.zeros((0, 4))] * 4, 2)) == [False, True, True, False, False])
check("one-frame detection (maybe a false person, maybe a real one mostly missed): its "
      "neighbours are blurred whole -- fail closed",
      list(miss([np.zeros((0, 4)), bA[None], np.zeros((0, 4))], 1)) == [True, False, True])

print("never less: random sizes, crops and confidence loss on every joint (1000 scenes)")
rng = np.random.default_rng(0)
bad = []
for t in range(1000):
    s = float(rng.uniform(0.6, 1.6))
    cut = rng.choice([None, "shoulders", "head"])
    k = int(rng.integers(1, 3 if s > 1.2 else 4))
    centres = [(int(55 + (300 // k) * i + rng.integers(-6, 6)),
                int(30 * s + rng.integers(0, 30))) for i in range(k)]
    people = []
    for c in centres:
        b, kx, _ = person(*c, s=s, cut=cut)
        if cut is not None and rng.random() < 0.6:  # the model imagines hips and legs in or below the crop
            lo = b[3] if rng.random() < 0.5 else b[3] + 80 * s
            kx[11:] = np.column_stack([rng.uniform(b[0], b[2], 6), rng.uniform(c[1] + 10 * s, lo, 6)])
        conf = rng.uniform(0, 1, N_JOINTS)  # any subset may drop out
        if rng.random() < 0.3:  # ... and the face may be confidently WRONG, anywhere in the box
            kx[:5] = np.column_stack([rng.uniform(b[0], b[2], 5), rng.uniform(b[1], b[3], 5)])
            conf[:5] = rng.uniform(0.5, 1, 5)
        people.append((b, kx, conf))
    r = int(round(FACE * s))
    p = faceblur.plan_frame(pred_of(people), W, H, int(rng.integers(0, 4)),
                            bool(rng.random() < 0.1))
    frame = scene([(*c, r) for c in centres])
    out = faceblur.apply_plan(frame, p)
    if not all(covered(p, *c, r) and destroyed(frame, out, *c, r) for c in centres):
        bad.append((t, p.mode, cut, round(s, 2), centres))
check("every planted face is covered and destroyed under every degradation", not bad,
      f"{len(bad)} bad, e.g. {bad[:3]}")

print("hybrid: a face detector's tight box only where it AGREES with the keypoints")


def fdet(cx, cy, r=FACE, score=0.9):
    """A CenterFace-like detection hugging a planted face (cheek to cheek, brow to chin)."""
    return [cx - r + 2, cy - r, cx + r - 2, cy + r, score]


one = scene([A])
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, face_dets=[fdet(*A)])
out = faceblur.apply_plan(one, p)
m = faceblur.blurred_mask(p, H, W)
floor_only = faceblur.plan_frame(pred_of([person(*A)]), W, H)
check("agreed face: mode 'detected', the face destroyed",
      p.mode == "detected" and p.agreed == 1 and destroyed(one, out, *A), str(p))
check("... only the face box is mosaicked: the neck, shoulders and guard stay visible",
      not m[A[1] + FACE + 6:A[1] + 120, A[0] - 30:A[0] + 30].any()
      and m.sum() < 0.6 * faceblur.blurred_mask(floor_only, H, W).sum(),
      f"{m.sum()} px vs {faceblur.blurred_mask(floor_only, H, W).sum()} px with the floor")
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, face_dets=[fdet(A[0] + 25, A[1] + 45)])
out = faceblur.apply_plan(scene([A]), p)
check("a decoy box on the glove (nose and eyes not inside it): fallback, face destroyed",
      p.mode == "keypoints" and p.agreed == 0 and destroyed(scene([A]), out, *A), str(p))
check("... and the decoy is still blurred (an unclaimed face box only adds)",
      faceblur.blurred_mask(p, H, W)[A[1] + 45, A[0] + 25])
noface_kp = person(*A)
noface_kp[2][list(faceblur.FACE_JOINTS)] = 0.1
p = faceblur.plan_frame(pred_of([noface_kp]), W, H, face_dets=[fdet(*A)])
check("a detection but no confident face keypoints: no agreement, the box floor",
      p.mode == "box" and p.agreed == 0 and covered(p, *A), str(p))
one_pt = person(*A, conf=0.05)
one_pt[2][faceblur.NOSE] = 0.9
p = faceblur.plan_frame(pred_of([one_pt]), W, H, face_dets=[fdet(*A)])
check("one confident face point (head not measurable): no agreement, fallback",
      p.agreed == 0 and covered(p, *A), str(p))
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, face_dets=[[A[0] - 3, A[1] - 6, A[0] + 3, A[1] + 4, 0.9]])
check("a box hugging only the nose and eyes (narrower than half the head): fallback",
      p.agreed == 0 and covered(p, *A), str(p))
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, face_dets=[fdet(A[0] + 14, A[1])])
out = faceblur.apply_plan(scene([A]), p)
check("a box beside the face (eyes not all inside it): fallback, face destroyed",
      p.agreed == 0 and destroyed(scene([A]), out, *A), str(p))
p = faceblur.plan_frame(pred_of([person(*A), person(*B)]), W, H, face_dets=[fdet(*A)])
out = faceblur.apply_plan(two, p)
check("two people, only A detected: A gets the tight box, B the floor, both destroyed",
      p.agreed == 1 and p.mode == "keypoints" and destroyed(two, out, *A) and destroyed(two, out, *B),
      str(p))
C = (240, 190)
three = scene([A, C])
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, face_dets=[fdet(*A), fdet(*C, r=10)])
check("a detected face with no person around it (YOLO missed them) is blurred too",
      destroyed(three, faceblur.apply_plan(three, p), *C, r=10), str(p))
for name, junk in (("NaN", [[np.nan, 0, 10, 10, 0.9]]), ("score above 1", [fdet(*A, score=1.5)]),
                  ("wrong shape", [[1, 2, 3]]), ("garbage", "not detections")):
    p = faceblur.plan_frame(pred_of([person(*A)]), W, H, face_dets=junk)
    check(f"malformed detections ({name}): ignored -- the fallback, never less",
          p.agreed == 0 and p.mode == "keypoints" and covered(p, *A), str(p))
# Review D1: arms up, YOLO's face points AND a CenterFace box both on the raised glove,
# the real face undetected. Agreement dropped the arms-up floor: 0% of the head.
glove_det = [118.0, 36.0, 142.0, 66.0, 0.6]
p = faceblur.plan_frame((up_box[None], up_k[None], up_c[None]), W3, H3, face_dets=[glove_det])
out = faceblur.apply_plan(up_scene, p)
check("arms up, face points and a face box both on the glove: the real head is still covered (was 0%)",
      covered(p, 130, 160, 15, H3, W3) and destroyed(up_scene, out, 130, 160, 15), str(p))
# Review D2: in guard the glove at the cheek fools both models; the real face is
# missed. A confident wrist inside the face box vetoes the agreement.
decoy = person(*A)
gx, gy = A[0] + 16, A[1] + 6  # glove beside the face
decoy[1][list(faceblur.FACE_JOINTS)] = [(gx, gy + 2), (gx - 5, gy - 3), (gx + 5, gy - 3), (gx - 10, gy), (gx + 10, gy)]
decoy[1][faceblur.R_WRIST] = (gx, gy)
p = faceblur.plan_frame(pred_of([decoy]), W, H, face_dets=[fdet(gx, gy)])
out = faceblur.apply_plan(scene([A]), p)
check("guard, face points and a face box on the glove, wrist inside the box: no agreement, head covered",
      p.agreed == 0 and destroyed(scene([A]), out, *A), str(p))
profile = person(*A)
profile[2][faceblur.R_EYE] = 0.1  # three-quarter view: nose, one eye, ears
p = faceblur.plan_frame(pred_of([profile]), W, H, face_dets=[fdet(*A)])
check("a three-quarter face (one eye) never drops the floor, even with a matching box",
      p.agreed == 0 and covered(p, *A), str(p))
# The "trust" policy (the middle ground): any detection on the person replaces the
# floor -- deface's look -- and only a person with NO detection gets the floor.
p = faceblur.plan_frame(pred_of([person(*A)]), W, H, face_dets=[fdet(*A)], face_policy="trust")
m = faceblur.blurred_mask(p, H, W)
check("trust: a detected face gets only its box -- face destroyed, the rest of the body visible",
      p.mode == "detected" and destroyed(one, faceblur.apply_plan(one, p), *A)
      and not m[A[1] + FACE + 6:A[1] + 120, A[0] - 30:A[0] + 30].any(), str(p))
p = faceblur.plan_frame(pred_of([profile]), W, H, face_dets=[fdet(*A)], face_policy="trust")
check("trust: a three-quarter face with a box is trusted too (no keypoint agreement needed)",
      p.agreed == 1 and covered(p, *A), str(p))
p = faceblur.plan_frame(pred_of([person(*A), person(*B)]), W, H, face_dets=[fdet(*A)], face_policy="trust")
check("trust: the person deface MISSED still gets the floor, both faces destroyed",
      p.agreed == 1 and destroyed(two, faceblur.apply_plan(two, p), *A)
      and destroyed(two, faceblur.apply_plan(two, p), *B), str(p))
p = faceblur.plan_frame((up_box[None], up_k[None], up_c[None]), W3, H3, face_dets=[glove_det],
                        face_policy="trust")
check("trust: arms up keeps the deep floor even when the box is trusted",
      covered(p, 130, 160, 15, H3, W3), str(p))
try:
    faceblur.plan_frame(pred_of([person(*A)]), W, H, face_policy="nope")
    check("an unknown face policy is refused", False, "accepted")
except ValueError:
    check("an unknown face policy is refused", True)
rng_h = np.random.default_rng(11)
hy_bad = []
for t in range(1000):
    centres = [(int(60 + 100 * i + rng_h.integers(-6, 6)), int(40 + rng_h.integers(0, 30)))
               for i in range(int(rng_h.integers(1, 4)))]
    people, dets = [], []
    for c in centres:
        b, kx, _ = person(*c)
        people.append((b, kx, rng_h.uniform(0, 1, N_JOINTS)))  # keypoints right, confidence random
        kind = rng_h.choice(["face", "decoy", "none", "tiny", "shifted"])
        if kind == "face":
            dets.append(fdet(*c, score=rng_h.uniform(0.2, 1)))
        elif kind == "decoy":
            dets.append(fdet(c[0] + rng_h.integers(-30, 30), c[1] + rng_h.integers(20, 120)))
        elif kind == "tiny":
            dets.append([c[0] - 2, c[1] - 2, c[0] + 2, c[1] + 2, 0.9])
        elif kind == "shifted":
            dx, dy = rng_h.integers(-14, 15, 2)
            dets.append(fdet(c[0] + dx, c[1] + dy))
    p = faceblur.plan_frame(pred_of(people), W, H, int(rng_h.integers(0, 4)),
                            bool(rng_h.random() < 0.1), dets)
    frame = scene(centres)
    outf = faceblur.apply_plan(frame, p)
    if not all(covered(p, *c) and destroyed(frame, outf, *c) for c in centres):
        hy_bad.append((t, p.mode, p.agreed, centres))
check("hybrid, 1000 random scenes (detections right, decoy, missing, tiny or shifted): "
      "every planted face covered and destroyed", not hy_bad, f"{len(hy_bad)} bad, e.g. {hy_bad[:3]}")

print("where output may go")
safe = tmp / "safe"
safe.mkdir()
iso = {"allowed_root": None, "forbidden_roots": [], "syncthing_configs": []}
check("positive control: a plain folder is accepted (allowlist lifted)",
      faceblur.check_output_location(safe / "x.mp4", **iso) == (safe / "x.mp4").resolve())
proot = tmp / "comps-participant-data"
(proot / "blurred").mkdir(parents=True)
inroot = {"allowed_root": proot, "forbidden_roots": [], "syncthing_configs": []}
check("positive control: under the participant root is accepted",
      faceblur.check_output_location(proot / "blurred" / "x.mp4", **inroot).name == "x.mp4")
refused("deny by default: outside the participant root", "outside_participant_root",
        lambda: faceblur.check_output_location(safe / "x.mp4", **inroot))
refused("deny by default, with the real default root", "outside_participant_root",
        lambda: faceblur.check_output_location(safe / "x.mp4", forbidden_roots=[],
                                               syncthing_configs=[]))
if os.name == "nt":
    import _winapi
    esc = proot / "escape"
    _winapi.CreateJunction(str(safe), str(esc))
    junctions.append(esc)
    refused("a junction inside the root that leads out of it", "outside_participant_root",
            lambda: faceblur.check_output_location(esc / "x.mp4", **inroot))
no_allow = {"allowed_root": None}
refused("inside this repo, even with the allowlist lifted", "forbidden_root",
        lambda: faceblur.check_output_location(ROOT / "x.mp4", **no_allow))
vault_dir = Path.home() / "ai-os-vault"
if vault_dir.is_dir():
    refused("inside the vault, even with the allowlist lifted", "forbidden_root",
            lambda: faceblur.check_output_location(vault_dir / "x.mp4", **no_allow))
else:
    print("  SKIP  no vault on this machine")
for marker, reason in ((".git", "git_checkout"), (".obsidian", "obsidian_vault"),
                       (".stfolder", "syncthing_marker"),
                       (".stfolder.removed-20261007", "syncthing_marker")):
    d = tmp / f"m{marker}"
    (d / "sub").mkdir(parents=True)
    (d / marker).mkdir()
    refused(f"below a folder holding {marker}", reason,
            lambda d=d: faceblur.check_output_location(d / "sub" / "x.mp4", **iso))
wt = tmp / "worktree"
wt.mkdir()
(wt / ".git").write_text("gitdir: elsewhere")  # a worktree's .git is a FILE
refused("inside a git worktree (.git is a file)", "git_checkout",
        lambda: faceblur.check_output_location(wt / "x.mp4", **iso))
st = tmp / "synced"
st.mkdir()
cfg = tmp / "config.xml"
cfg.write_text(f'<configuration><folder id="a" path="{st}"></folder></configuration>')
refused("inside a folder the Syncthing config shares", "syncthing_folder",
        lambda: faceblur.check_output_location(st / "x.mp4", allowed_root=None,
                                               forbidden_roots=[], syncthing_configs=[cfg]))
broken = tmp / "broken.xml"
broken.write_text("<configuration><folder")
refused("an unreadable Syncthing config refuses everything", "syncthing_config_unreadable",
        lambda: faceblur.check_output_location(safe / "x.mp4", allowed_root=None,
                                               forbidden_roots=[], syncthing_configs=[broken]))
check("a missing Syncthing config is not an error",
      faceblur.check_output_location(safe / "x.mp4", allowed_root=None, forbidden_roots=[],
                                     syncthing_configs=[tmp / "nope.xml"]).name == "x.mp4")
vault = tmp / "vault"
vault.mkdir()
fv = {"allowed_root": None, "forbidden_roots": [vault], "syncthing_configs": []}
refused("inside an explicitly forbidden root", "forbidden_root",
        lambda: faceblur.check_output_location(vault / "x.mp4", **fv))
refused("inside a forbidden root, spelled with .. and different case", "forbidden_root",
        lambda: faceblur.check_output_location(Path(str(safe / ".." / "VAULT" / "x.mp4")), **fv))
for cloud_name in ("OneDrive - Personal", "Shared drives", "Box", "Nextcloud", "pCloud Drive"):
    cloud = tmp / cloud_name / "clips"
    cloud.mkdir(parents=True)
    refused(f"inside a cloud-synced folder ({cloud_name})", "cloud_folder",
            lambda c=cloud: faceblur.check_output_location(c / "x.mp4", **iso))
boxing = tmp / "boxing-clips"
boxing.mkdir()
check("... but a folder merely starting with 'box' is fine",
      faceblur.check_output_location(boxing / "x.mp4", **iso).name == "x.mp4")
refused("not an .mp4", "not_mp4", lambda: faceblur.check_output_location(safe / "x.mkv", **iso))
refused("folder does not exist (never created silently)", "no_folder",
        lambda: faceblur.check_output_location(safe / "new" / "x.mp4", **iso))
refused("a UNC network path (checked before the folder exists)", "network_path",
        lambda: faceblur.check_output_location(Path(r"\\server\share\x.mp4"), **iso))
refused("a verbatim UNC network path", "network_path",
        lambda: faceblur.check_output_location(Path(r"\\?\UNC\server\share\x.mp4"), **iso))
(safe / "exists.mp4").write_bytes(b"x")
refused("an existing file without --overwrite", "exists",
        lambda: faceblur.check_output_location(safe / "exists.mp4", **iso))
check("... replaced with --overwrite",
      faceblur.check_output_location(safe / "exists.mp4", overwrite=True, **iso).name == "exists.mp4")
refused("the input itself", "is_input",
        lambda: faceblur.check_output_location(safe / "exists.mp4", input_path=safe / "exists.mp4",
                                               overwrite=True, **iso))
if os.name == "nt":
    link = tmp / "innocent"
    _winapi.CreateJunction(str(vault), str(link))
    junctions.append(link)
    refused("a junction from a plain folder INTO a forbidden one (resolved path checked)",
            "forbidden_root", lambda: faceblur.check_output_location(link / "x.mp4", **fv))
    out_link = vault / "exit"
    _winapi.CreateJunction(str(safe), str(out_link))
    junctions.append(out_link)
    refused("a junction from a forbidden folder OUT to a plain one (given path checked)",
            "forbidden_root", lambda: faceblur.check_output_location(out_link / "x.mp4", **fv))

print("end to end: real encode, audio, metadata and names stripped")
N, FPS = 24, 30
clip = tmp / "Jane-Doe-round1.mp4"  # a clip named after its participant
centres = [((A[0] + 2 * i, A[1]), (B[0] - 2 * i, B[1])) for i in range(N)]
frames_in = [scene(list(c)) for c in centres]
SECRETS = (b"SECRET-TITLE-7731", b"SECRET-COMMENT-4410", b"+33.6789-112.0123/", b"Jane")
enc = subprocess.run(
    ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
     "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
     "-c:a", "aac", "-shortest", "-metadata", f"title={SECRETS[0].decode()}",
     "-metadata", f"comment={SECRETS[1].decode()}", "-metadata", f"location={SECRETS[2].decode()}",
     "-metadata", "artist=Jane Doe",
     "-metadata", "creation_time=2026-10-12T10:00:00Z", "-movflags", "use_metadata_tags",
     str(clip)], input=b"".join(f.tobytes() for f in frames_in), capture_output=True)
check("fixture encoded", enc.returncode == 0, enc.stderr.decode(errors="replace")[-300:])


def ffprobe_json(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                        str(path)], capture_output=True, check=True)
    return json.loads(r.stdout)


src = ffprobe_json(clip)
raw_in = clip.read_bytes()
check("positive control: the input HAS an audio track, the planted tags and the name",
      any(s["codec_type"] == "audio" for s in src["streams"])
      and all(s in raw_in for s in SECRETS) and "creation_time" in src["format"].get("tags", {}),
      str(src["format"].get("tags")))


def make_predict(drop_at=10, raise_at=15):
    state = {"i": 0, "calls": 0}

    def predict(frame):
        i = state["i"]
        state["i"] += 1
        state["calls"] += 1
        if i == raise_at:
            raise RuntimeError("model fell over")
        people = [person(*c) for c in centres[i]]
        return pred_of(people[:1] if i == drop_at else people)
    return predict, state


handed = []


class SpyWriter(video.VideoWriter):
    """Records every frame handed to the encoder -- the only way pixels leave memory."""
    fail_at = None

    def write(self, frame):
        if SpyWriter.fail_at is not None and len(handed) == SpyWriter.fail_at:
            raise RuntimeError("disk full")
        handed.append(frame.copy())
        super().write(frame)


outdir = tmp / "out"
outdir.mkdir()
out = outdir / "P01-R1.mp4"
pred, state = make_predict()
rep = faceblur.blur_video(clip, out, pred, writer_factory=SpyWriter)
check("every frame handed to the encoder has every face destroyed (none unblurred, ever)",
      len(handed) == N and all(destroyed(frames_in[i], handed[i], *c)
                               for i in range(N) for c in centres[i]))
check("frame where a person was missed and frame where the model raised are whole-frame",
      rep["per_frame"]["mode"][10] == "f" and rep["per_frame"]["mode"][15] == "f",
      rep["per_frame"]["mode"])
check("report: modes add up, reasons named, fully blurred ranges listed",
      rep["frames"] == N and sum(rep["frames_by_mode"].values()) == N
      and rep["frames_fully_blurred"] == 2 and rep["fully_blurred_ranges"] == [[10, 10], [15, 15]]
      and rep["full_blur_reasons"] == {"person_missing_vs_neighbours": 1, "predictor_error": 1},
      json.dumps({k: rep[k] for k in ("frames_by_mode", "full_blur_reasons")}))
check("report: persons per frame", rep["persons_per_frame"] == {"0": 1, "1": 1, "2": 22},
      str(rep["persons_per_frame"]))
report_text = (outdir / "P01-R1.blur.json").read_text()
check("report written next to the video and equal to the returned one",
      json.loads(report_text) == rep)
check("report names no file: the source is its sha256, and 'Jane' appears nowhere",
      "Jane" not in report_text and "source" not in rep
      and rep["source_sha256"] == faceblur.sha256_file(clip))
check("only the video and its report are left in the folder (no partials)",
      sorted(p.name for p in outdir.iterdir()) == ["P01-R1.blur.json", "P01-R1.mp4"])
dst = ffprobe_json(out)
raw_out = out.read_bytes()
check("output: one stream, H.264 video, no audio",
      [s["codec_type"] for s in dst["streams"]] == ["video"]
      and dst["streams"][0]["codec_name"] == "h264", str([s["codec_name"] for s in dst["streams"]]))
check("output: none of the planted metadata or the name survives, anywhere in its bytes",
      not any(s in raw_out for s in SECRETS) and b"2026-10-12" not in raw_out)
check("output: no creation time, location, title, comment or artist tag, no Lavf version",
      not any(k in t for t in [dst["format"].get("tags", {})] + [s.get("tags", {}) for s in dst["streams"]]
              for k in ("creation_time", "location", "title", "comment", "artist"))
      and b"Lavf6" not in raw_out, str(dst["format"].get("tags")))
decoded = list(video.iter_frames(out))
check("output decodes to the same frame count", len(decoded) == N, str(len(decoded)))
check("after H.264 encoding every planted face is still destroyed",
      all(destroyed(frames_in[i], decoded[i], *c, limit=0.25) for i in range(N) for c in centres[i]))
check("report hash is the hash of the file on disk", rep["output_sha256"] == faceblur.sha256_file(out))

xdir = tmp / "extra"
xdir.mkdir()
xrep = faceblur.blur_video(clip, xdir / "x.mp4", make_predict()[0],
                           report_extra={"audio": "yes", "source_sha256": "forged", "note": 1})
check("report_extra can add fields but never overwrite a core one",
      xrep["audio"] == "none" and xrep["source_sha256"] == rep["source_sha256"] and xrep["note"] == 1)

print("failures leave nothing behind")


def leaves_nothing(name, run, exc):
    d = tmp / name.replace(" ", "_")[:20]
    d.mkdir()
    try:
        run(d / "x.mp4")
        check(f"{name}: raises", False, "did not raise")
    except exc:
        check(f"{name}: raises", True)
    check(f"{name}: no video, report or partial is left", list(d.iterdir()) == [],
          str(list(d.iterdir())))


handed.clear()
SpyWriter.fail_at = 5
leaves_nothing("writer fails mid-encode",
               lambda o: faceblur.blur_video(clip, o, make_predict()[0], writer_factory=SpyWriter),
               RuntimeError)
SpyWriter.fail_at = None


def _late_refusal(o):
    raise UnsafeOutput("syncthing_folder", "became a Syncthing folder during the run")


leaves_nothing("output folder turned unsafe during the run (recheck)",
               lambda o: faceblur.blur_video(clip, o, make_predict()[0], recheck=_late_refusal),
               UnsafeOutput)
real_sha = faceblur.sha256_file
calls = {"n": 0}


def _shifty_sha(path):
    calls["n"] += 1
    return real_sha(path) if calls["n"] == 1 else "0" * 64


faceblur.sha256_file = _shifty_sha
leaves_nothing("input replaced between the two passes",
               lambda o: faceblur.blur_video(clip, o, make_predict()[0]), RuntimeError)
faceblur.sha256_file = real_sha

print("hybrid end to end")
hdir = tmp / "hybrid"
hdir.mkdir()
fstate = {"i": 0}


def good_faces(frame):
    i = fstate["i"]
    fstate["i"] += 1
    return [fdet(*c) for c in centres[i]]


def broken_faces(frame):
    raise RuntimeError("detector fell over")


handed.clear()
hrep = faceblur.blur_video(clip, hdir / "h.mp4", make_predict(drop_at=-1, raise_at=-1)[0],
                           detect_faces=good_faces, face_model="fake", writer_factory=SpyWriter)
check("agreed faces are counted, and every frame handed to the encoder has both faces destroyed",
      hrep["persons_face_agreed"] == 2 * N and hrep["persons_fallback"] == 0
      and hrep["frames_by_mode"]["detected"] == N
      and all(destroyed(frames_in[i], handed[i], *c) for i in range(N) for c in centres[i]),
      json.dumps({k: hrep[k] for k in ("persons_face_agreed", "persons_fallback", "frames_by_mode")}))
brep = faceblur.blur_video(clip, hdir / "b.mp4", make_predict(drop_at=-1, raise_at=-1)[0],
                           detect_faces=broken_faces, face_model="fake")
check("a face detector that raises on every frame: errors counted, everyone falls back",
      brep["face_detector_errors"] == N and brep["persons_face_agreed"] == 0
      and brep["frames_by_mode"]["detected"] == 0 and brep["frames_by_mode"]["full"] == 0,
      json.dumps({k: brep[k] for k in ("face_detector_errors", "frames_by_mode")}))

try:
    faceblur.load_centerface(sha256="0" * 64)
    check("the face-detector model is refused unless its sha256 matches the pin", False, "loaded")
except ValueError:
    check("the face-detector model is refused unless its sha256 matches the pin", True)
except RuntimeError:  # deface not installed here: the refusal path is unreachable, say so
    print("  SKIP  deface not installed")

print("the deface policy (the default): the face detector alone decides")
p = faceblur.plan_faces_only([fdet(*A), fdet(*B)], W, H)
out = faceblur.apply_plan(two, p)
check("deface: each detected face is mosaicked, nothing else touched",
      p.mode == "faces" and destroyed(two, out, *A) and destroyed(two, out, *B)
      and np.array_equal(out[~faceblur.blurred_mask(p, H, W)], two[~faceblur.blurred_mask(p, H, W)]),
      str(p))
p = faceblur.plan_faces_only([], W, H)
check("deface: a frame with NO detected face is left as it is (the accepted residual)",
      p.mode == "no_face" and np.array_equal(faceblur.apply_plan(two, p), two), str(p))
for name, junk in (("a detector error", faceblur.DetectorError()), ("NaN", [[np.nan, 0, 10, 10, 0.9]]),
                   ("garbage", "not detections")):
    p = faceblur.plan_faces_only(junk, W, H)
    check(f"deface: {name} is a failure, not a miss -- the whole frame is blurred",
          p.mode == "full" and destroyed(two, faceblur.apply_plan(two, p), *A), str(p))
dfdir = tmp / "deface"
dfdir.mkdir()
dstate = {"i": 0}


def some_faces(frame):  # finds both faces except in frames 5-7, where it finds none
    i = dstate["i"]
    dstate["i"] += 1
    if i == 9:
        raise RuntimeError("detector fell over")
    return [] if 5 <= i <= 7 else [fdet(*c) for c in centres[i]]


handed.clear()
drep = faceblur.blur_video(clip, dfdir / "d.mp4", None, detect_faces=some_faces, face_model="fake",
                           face_policy="deface", writer_factory=SpyWriter)
codes = drep["per_frame"]["mode"]
check("deface end to end: no YOLO needed; no-face frames counted and listed; an error frame blurred whole",
      drep["frames_without_face"] == 3 and drep["no_face_ranges"] == [[5, 7]]
      and codes[9] == "f" and drep["face_detector_errors"] == 1
      and codes.count("a") == N - 4, json.dumps({k: drep[k] for k in ("frames_by_mode", "no_face_ranges")}))
check("... the faces it found are destroyed in every frame it found them",
      all(destroyed(frames_in[i], handed[i], *c) for i in range(N) if codes[i] == "a" for c in centres[i]))
check("... and a frame it missed reaches the encoder unblurred (the residual, measured not hidden)",
      not destroyed(frames_in[6], handed[6], *centres[6][0]))
check("deface report: face threshold recorded, faces (not persons) counted, deface params only",
      "faces_per_frame" in drep and "persons_per_frame" not in drep and "box_floor" not in drep["params"]
      and "face_thresh" in drep["params"], json.dumps(drep["params"]))
# Review F1: a detector that silently finds nothing (broken, or mis-set) wrote a fully
# unblurred clip with exit 0. A clip with no face anywhere is refused.
nodir = tmp / "noface"
nodir.mkdir()
try:
    faceblur.blur_video(clip, nodir / "n.mp4", None, detect_faces=lambda f: np.zeros((0, 5)),
                        face_model="fake", face_policy="deface")
    check("deface: a detector that never fires is refused", False, "wrote it")
except faceblur.NoFacesFound:
    check("deface: a detector that never fires is refused, and nothing is written",
          list(nodir.iterdir()) == [], str(list(nodir.iterdir())))
def one_hit():  # fires on exactly one frame (the near-miss variant of a broken detector)
    st = {"i": 0}

    def det(frame):
        st["i"] += 1
        return [fdet(*A)] if st["i"] == 1 else []
    return det


def raise_once():  # raises on one frame, empty otherwise
    st = {"i": 0}

    def det(frame):
        st["i"] += 1
        if st["i"] == 1:
            raise RuntimeError("boom")
        return []
    return det


for name, det in (("fires on one frame only", one_hit()), ("raises once, else empty", raise_once())):
    try:
        faceblur.blur_video(clip, nodir / "n.mp4", None, detect_faces=det, face_model="fake",
                            face_policy="deface")
        check(f"deface: a detector that {name} is refused (over half the frames faceless)", False, "wrote it")
    except faceblur.NoFacesFound:
        check(f"deface: a detector that {name} is refused (over half the frames faceless)",
              list(nodir.iterdir()) == [])
quarter = {"i": 0}


def misses_a_quarter(frame):  # 25% no-face frames: inside the measured range, passes
    quarter["i"] += 1
    return [] if quarter["i"] % 4 == 0 else [fdet(*c) for c in centres[quarter["i"] - 1]]


qrep = faceblur.blur_video(clip, nodir / "q.mp4", None, detect_faces=misses_a_quarter, face_model="fake",
                           face_policy="deface")
check("deface: a clip with 25% no-face frames passes without the flag",
      qrep["frames_without_face"] == N // 4 and qrep["allow_no_faces"] is False)
for p_ in nodir.iterdir():
    p_.unlink()
nrep = faceblur.blur_video(clip, nodir / "n.mp4", None, detect_faces=lambda f: np.zeros((0, 5)),
                           face_model="fake", face_policy="deface", allow_no_faces=True)
check("... unless --allow-no-faces is passed, which the report records",
      nrep["allow_no_faces"] is True and nrep["frames_without_face"] == N)
try:
    faceblur.blur_video(clip, dfdir / "e.mp4", None, face_policy="deface")
    check("the deface policy without a face detector is refused", False, "ran")
except ValueError:
    check("the deface policy without a face detector is refused", True)

print("the CLI")
cli_dir = tmp / "cli"
cli_dir.mkdir()
pred, state = make_predict(drop_at=-1, raise_at=-1)
rc = blur_faces.main(["--video", str(clip), "--out", str(cli_dir / "c.mp4")], predict=pred,
                     location_kwargs={"forbidden_roots": [], "syncthing_configs": []})
check("CLI refuses a folder outside the participant root by default (exit 2, nothing decoded)",
      rc == 2 and state["calls"] == 0 and not (cli_dir / "c.mp4").exists())
rc = blur_faces.main(["--video", str(clip), "--out", str(cli_dir / "c.mp4"),
                      "--allow-outside-participant-root", "--face-policy", "agree"], predict=pred,
                     location_kwargs={"forbidden_roots": [], "syncthing_configs": []})
cli_rep = json.loads((cli_dir / "c.blur.json").read_text()) if rc == 0 else {}
check("CLI with --allow-outside-participant-root succeeds, and the report says so",
      rc == 0 and (cli_dir / "c.mp4").is_file() and cli_rep.get("outside_participant_root") is True)
pred, state = make_predict()
rc = blur_faces.main(["--video", str(clip), "--out", str(vault / "c.mp4"),
                      "--allow-outside-participant-root", "--face-policy", "agree"], predict=pred,
                     location_kwargs={"forbidden_roots": [vault], "syncthing_configs": []})
check("CLI override never lifts the denylist (exit 2 BEFORE decoding a frame)",
      rc == 2 and state["calls"] == 0 and not (vault / "c.mp4").exists())
over = ["--allow-outside-participant-root", "--face-policy", "agree"]
rc = blur_faces.main(["--video", str(clip), "--out", str(cli_dir / "c.mp4"), *over], predict=pred,
                     location_kwargs=iso)
check("CLI refuses to replace an existing output", rc == 2 and state["calls"] == 0)
(cli_dir / "c.mp4").unlink()
rc = blur_faces.main(["--video", str(clip), "--out", str(cli_dir / "c.mp4"), *over], predict=pred,
                     location_kwargs=iso)
check("CLI refuses to replace an existing report", rc == 2 and state["calls"] == 0)
rc = blur_faces.main(["--video", str(clip), "--out", str(clip), *over], predict=pred,
                     location_kwargs=iso)
check("CLI refuses to write over its input", rc == 2 and state["calls"] == 0)
rc = blur_faces.main(["--video", str(clip), "--out", str(cli_dir / "z.mp4"), *over, "--face-policy", "deface"],
                     detect_faces=lambda f: np.zeros((0, 5)), location_kwargs=iso)
check("CLI (deface default): a clip with no face found anywhere is refused, exit 2, nothing written",
      rc == 2 and not (cli_dir / "z.mp4").exists())
try:
    blur_faces.main(["--video", str(clip), "--out", str(cli_dir / "y.mp4"), *over, "--face-policy", "deface", "--face-thresh", "0.9"],
                    detect_faces=lambda f: np.zeros((0, 5)), location_kwargs=iso)
    check("CLI: a deface face threshold above 0.5 is refused", False, "accepted")
except SystemExit as e:
    check("CLI: a deface face threshold above 0.5 is refused", e.code == 2 and not (cli_dir / "y.mp4").exists())
os.environ.pop("YOLO_OFFLINE", None)
fake_w = tmp / "fake.pt"
fake_w.write_bytes(b"not a checkpoint")
try:
    blur_faces.main(["--video", str(clip), "--out", str(cli_dir / "d.mp4"), *over, "--face-policy", "agree", "--weights",
                     str(fake_w), "--weights-sha256", "0" * 64], location_kwargs=iso)
except ValueError:
    pass  # the pinned-hash check refuses it; we only need what happened before
check("CLI puts ultralytics in offline mode (no analytics) before loading a model",
      os.environ.get("YOLO_OFFLINE") == "1")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
