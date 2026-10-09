"""Tests for the pose-switch measurement: clip selection, YOLO clips in corpus
space, the alignment and identity checks, and eval_pose_switch.py's gates.

Synthetic fixtures only. No number here says anything about boxing.

Run:  python tests/test_poseswitch.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from boxingvi import poseswitch  # noqa: E402
from boxingvi.data import N_JOINTS  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


W, H, FPS = 1920, 1080, 24.0
POSE = np.array([[0, -160], [-8, -168], [8, -168], [-16, -164], [16, -164], [-50, -100], [50, -100],
                 [-60, -50], [60, -50], [-55, -10], [55, -10], [-30, 0], [30, 0], [-32, 90], [32, 90],
                 [-34, 180], [34, 180]], dtype=np.float64)


def video_pose(n, hx=900.0, hy=600.0, seed=0):
    """A person whose wrists wander, so every frame differs from its neighbours."""
    rng = np.random.default_rng(seed)
    xy = np.broadcast_to(POSE + (hx, hy), (n, N_JOINTS, 2)).copy()
    xy[:, [9, 10]] += rng.normal(0, 25, (n, 2, 2)).cumsum(axis=0) * 0.3
    xy[:, :, 0] += np.sin(np.arange(n) / 7)[:, None] * 40
    return xy


def corpus_clip(xy, start, end):
    kp = np.zeros((25, N_JOINTS, 3), np.float32)
    k = end - start + 1
    kp[:k, :, 0] = xy[start:end + 1, :, 0] / W
    kp[:k, :, 1] = xy[start:end + 1, :, 1] / H
    kp[:k, :, 2] = 1
    return kp


n = 600
xy = video_pose(n)
pose = PoseSequence(xy.astype(np.float32), np.full((n, N_JOINTS), 0.9, np.float32), np.ones(n, bool), FPS, W, H)
spans = [(40 + 30 * i, 48 + 30 * i) for i in range(15)]
clips = [poseswitch.HeldOutClip("V2", i, s, e, 1, corpus_clip(xy, s, e)) for i, (s, e) in enumerate(spans)]

print("YOLO clips in corpus space")
y = poseswitch.yolo_clip(pose, *spans[0])
check("same frames, same normalisation: a YOLO clip of the AlphaPose person equals the corpus clip",
      np.allclose(y, clips[0].alphapose, atol=1e-5))
check("padding after the real frames, validity 0", (y[9:] == 0).all() and (y[:9, :, 2] == 1).all())
low = PoseSequence(pose.xy, pose.conf.copy(), pose.detected, FPS, W, H)
low.conf[:, 15] = 0.1
check("the pipeline rule drops a joint under 0.3; keep-all keeps it",
      (poseswitch.yolo_clip(low, *spans[0])[:9, 15, 2] == 0).all()
      and (poseswitch.yolo_clip(low, *spans[0], None)[:9, 15, 2] == 1).all())
miss = PoseSequence(np.where(np.arange(n)[:, None, None] < 45, np.nan, pose.xy).astype(np.float32), pose.conf,
                    np.arange(n) >= 45, FPS, W, H)
ym = poseswitch.yolo_clip(miss, *spans[0])
check("undetected frames are zero with validity 0, never NaN", np.isfinite(ym).all() and (ym[:5, :, 2] == 0).all()
      and (ym[5:9, :, 2] == 1).all())
check("a span past the end of the pose is zero there, not an index error",
      (poseswitch.yolo_clip(pose, n - 3, n + 3)[3:7] == 0).all())

print("alignment")
al = poseswitch.align(pose, clips)
check("aligned spans: best offset 0, sharp", al["best"] == 0 and al["sharp"], str({k: al[k] for k in ("best", "ratio")}))
shifted = [poseswitch.HeldOutClip("V2", c.index, c.start - 4, c.end - 4, 1, c.alphapose) for c in clips]
al4 = poseswitch.align(pose, shifted)
check("spans numbered 4 frames early: best offset +4", al4["best"] == 4 and al4["sharp"], str(al4["best"]))
still = np.broadcast_to(POSE + (900.0, 600.0), (n, N_JOINTS, 2)).astype(np.float32)
flat = PoseSequence(still, pose.conf, pose.detected, FPS, W, H)
other = [poseswitch.HeldOutClip("V2", c.index, c.start, c.end, 1, corpus_clip(xy + 300, c.start, c.end)) for c in clips]
alf = poseswitch.align(flat, other)
check("no frame matches better than any other: not sharp, so nothing would be scored",
      not alf["sharp"], str({k: alf[k] for k in ("best", "ratio")}))
check("an all-undetected pose: no best offset, not sharp",
      not poseswitch.align(PoseSequence(np.full_like(pose.xy, np.nan), pose.conf, np.zeros(n, bool), FPS, W, H),
                           clips)["sharp"])

rng = np.random.default_rng(1)
slow = np.broadcast_to(POSE + (900.0, 600.0), (n, N_JOINTS, 2)).copy()
slow[:, :, 0] += (np.arange(n) * 3.0)[:, None]          # 3 px a frame: neighbours look alike
slow_pose = PoseSequence(slow.astype(np.float32), pose.conf, pose.detected, FPS, W, H)
jit = [poseswitch.HeldOutClip("V2", c.index, c.start, c.end, 1,
                              corpus_clip(slow + rng.normal(0, 3, slow.shape), c.start, c.end)) for c in clips]
als = poseswitch.align(slow_pose, jit)
check("smooth motion plus two-model jitter: +-1 is nearly as good as 0, yet the offset is sharp against +-3",
      als["best"] == 0 and als["sharp"] and als["curve"][1] < 1.5 * als["curve"][0],
      str({k: als[k] for k in ("best", "ratio")}))

print("identity")
check("the same person is the same person", poseswitch.same_person(clips[0], y) is True)
far_pose = PoseSequence((xy + (500, 0)).astype(np.float32), pose.conf, pose.detected, FPS, W, H)
check("a person 500 px away is someone else", poseswitch.same_person(clips[0], poseswitch.yolo_clip(far_pose, *spans[0])) is False)
check("no shared joint: unknown, not 'same'",
      poseswitch.same_person(clips[0], np.zeros((25, N_JOINTS, 3), np.float32)) is None)

print("held-out clip selection")
import pandas as pd  # noqa: E402

tmp = Path(tempfile.mkdtemp())
(tmp / "Skeleton_data").mkdir()
(tmp / "Annotation_files").mkdir()
arr = np.zeros((4, 25, N_JOINTS, 2), np.float32)
for i, (s, e) in enumerate([(10, 18), (30, 38), (50, 58), (70, 78)]):
    arr[i, :9] = xy[s:e + 1] / (W, H)
np.save(tmp / "Skeleton_data" / "V2.npy", arr)
pd.DataFrame({"start": [10, 30, 50, 70], "end": [18, 40, 58, 78],
              "class": ["jab", "cross", None, "lead hook"]}).to_excel(tmp / "Annotation_files" / "V2.xlsx", index=False)
got, mism = poseswitch.held_out_clips(tmp, "V2")
check("a span whose length is not the clip's frame count is left out and counted; unlabelled rows skipped",
      [c.index for c in got] == [0, 3] and mism == 1, f"{[c.index for c in got]} {mism}")
check("the clip keeps its annotated frames and the corpus validity channel",
      got[0].start == 10 and got[0].end == 18 and got[0].alphapose.shape == (25, N_JOINTS, 3))
np.save(tmp / "Skeleton_data" / "V2.npy", arr[:3])
try:
    poseswitch.held_out_clips(tmp, "V2")
    check("skeleton rows that do not pair with annotation rows are refused", False)
except ValueError:
    check("skeleton rows that do not pair with annotation rows are refused", True)

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
