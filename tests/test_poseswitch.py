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
sys.path.insert(0, str(ROOT / "cli"))

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

print("eval_pose_switch.aggregate: which punches each comparison is read on")
import json  # noqa: E402
import subprocess  # noqa: E402

import eval_pose_switch as eps  # noqa: E402


def fake(pred: int) -> np.ndarray:
    """A clip the fake classifier below reads as `pred`."""
    return np.full((25, N_JOINTS, 3), float(pred), np.float32)


def read_fake(clips):
    return np.array([int(c[0, 0, 0]) for c in clips], dtype=int)


def row(label, a, y_all, y, seen, seen_all, video="V5", cut=True):
    return {"video": video, "label": label, "A": fake(a), "Y_all": fake(y_all), "Y": fake(y),
            "Y_fix": fake(y_all) if cut else None, "Y_ext": fake(y_all) if cut else None,
            "Y_clip": fake(y_all) if cut else None, "iou": 0.5 if cut else None,
            "seen": seen, "seen_all": seen_all, "same": True}


# 6 punches both rules see (A right, Y_all right, Y wrong); 3 only the pipeline's rule sees,
# where Y_all is wrong; 1 YOLO missed. The old filter read A - Y_all as 0 on 6; the pipeline's
# rule reads it as 1/3 on 9.
rows_ = ([row(n % 6, n % 6, n % 6, (n + 1) % 6, True, True, "V5" if n < 4 else "V9") for n in range(6)]
         + [row(n, n, (n + 1) % 6, 0, False, True, "V9") for n in range(3)]
         + [row(0, 0, 0, 0, False, False)])
rep = eps.aggregate(rows_, read_fake, ["V5", "V9"])
s, c = rep["all_seen"], rep["legacy_common"]
check("A vs Y_all is read on every punch the pipeline's rule sees, not only those the old rule sees",
      s["n"] == 9 and s["A"]["n"] == 9 and s["A_minus_Y_all"]["n"] == 9
      and abs(s["Y_all"]["accuracy"] - 6 / 9) < 1e-12 and abs(s["A_minus_Y_all"]["accuracy"]["point"] - 3 / 9) < 1e-12,
      json.dumps({k: s[k] for k in ("n",)} | {"Y_all": s["Y_all"].get("accuracy")}))
check("the legacy rule is not scored on all_seen at all (it would be read on punches it blanks)",
      "Y" not in s and "A_minus_Y" not in s)
check("the legacy comparison is on the named common subset, with A and Y_all re-read on it",
      c["n"] == 6 and c["Y"]["accuracy"] == 0.0 and c["Y_all"]["accuracy"] == 1.0 and c["A"]["n"] == 6
      and c["A_minus_Y"]["n"] == 6 and abs(c["A_minus_Y_all"]["accuracy"]["point"]) < 1e-12)
check("both n's and the blanked/missed counts are reported, and named",
      rep["n_clips"] == 10 and rep["n_yolo_missed"] == 1 and rep["n_legacy_blanked"] == 3
      and set(rep["subsets"]) == {"all_seen", "legacy_common"})
check("per video and same-person blocks carry both subsets with their own n's",
      rep["by_video"]["V9"]["all_seen"]["n"] == 5 and rep["by_video"]["V9"]["legacy_common"]["n"] == 2
      and rep["by_video"]["V5"]["all_seen"]["n"] == 4 and rep["same_person"]["all_seen"]["n"] == 9
      and rep["same_person"]["legacy_common"]["n"] == 6)
check("the cut readings start from all_seen too", rep["window"]["n"] == 9 and rep["window"]["Y_all"]["n"] == 9)
rows_cut = [dict(r, Y_ext=None) if r["seen_all"] and not r["seen"] else r for r in rows_]
check("a punch with any cut missing leaves the cut readings and is counted",
      eps.aggregate(rows_cut, read_fake, ["V5", "V9"])["window"]["n_uncuttable"] == 3)
empty = eps.aggregate([row(0, 0, 0, 0, False, False)], read_fake, ["V5"])
empty.update({"scored_videos": ["V5"], "held_out_videos": ["V2", "V5", "V9"], "excluded_videos": {}})
try:
    lines = eps.summary_lines(empty)
    json.dumps(empty, allow_nan=False, default=float)
    check("nothing seen: every reading n 0, printed as n/a, still JSON", empty["all_seen"]["n"] == 0
          and any("n/a" in ln for ln in lines))
except Exception as e:  # noqa: BLE001
    check("nothing seen: every reading n 0, printed as n/a, still JSON", False, f"{type(e).__name__}: {e}")

print("eval_pose_switch.py end to end: alignment passes, no punch can be cut")
import pandas as pd  # noqa: E402, F811
import torch  # noqa: E402

from boxingvi import classify  # noqa: E402
from boxingvi.data import CLASSES, NATIVE_FPS, VIDEOS, load_corpus, split_pipeline, to_arrays  # noqa: E402
from boxingvi.stgcnpp import STGCNPP  # noqa: E402

e2e = Path(tempfile.mkdtemp())
data_ = e2e / "data"
(data_ / "Skeleton_data").mkdir(parents=True)
(data_ / "Annotation_files").mkdir()


def write_video(v, xy_, spans_):
    arr_ = np.zeros((len(spans_), 25, N_JOINTS, 2), np.float32)
    for i, (s_, e_) in enumerate(spans_):
        arr_[i, :e_ - s_ + 1] = xy_[s_:e_ + 1] / (W, H)
    np.save(data_ / "Skeleton_data" / f"{v}.npy", arr_)
    pd.DataFrame({"start": [s_ for s_, _ in spans_], "end": [e_ for _, e_ in spans_],
                  "class": [CLASSES[i % len(CLASSES)] for i in range(len(spans_))]}).to_excel(
        data_ / "Annotation_files" / f"{v}.xlsx", index=False)


# V5: a rigid body drifting on a whole-pixel random walk, so every frame differs (alignment is
# sharp at 0), while the wrists are never detected: no wrist speed, so no punch has a cut.
n5, rng5 = 900, np.random.default_rng(7)
xy5 = (POSE + (900.0, 600.0))[None] + np.round(rng5.normal(0, 6, (n5, 2)).cumsum(axis=0))[:, None, :]
conf5 = np.full((n5, N_JOINTS), 0.9, np.float32)
conf5[:, [9, 10]] = 0.0
spans5 = [(40 + 40 * i, 48 + 40 * i) for i in range(20)]
body_joints = [j for j in range(N_JOINTS) if j not in (9, 10)]
for s_, e_ in spans5[:4]:   # detected, but every joint under the old 0.3 rule
    conf5[s_:e_ + 1, body_joints] = 0.2
write_video("V5", xy5, spans5)
for k, v in enumerate(v_ for v_ in VIDEOS if v_ not in ("V5", "V6")):
    write_video(v, video_pose(300, seed=10 + k), [(20 + 30 * i, 28 + 30 * i) for i in range(8)])
poses_ = e2e / "poses"
poses_.mkdir()
PoseSequence(xy5.astype(np.float32), conf5, np.ones(n5, bool), NATIVE_FPS["V5"], W, H).save(poses_ / "V5.pose.npz")

GEOM = classify.WindowGeometry(duration_s=0.375, lead_in_s=0.125, fps=24.0, torso_px=200.0,
                               hip_x=0.5, hip_y=0.55, variant="relative", n_clips=100)
torch.manual_seed(0)
ck = e2e / "clf.pt"
torch.save(STGCNPP().state_dict(), ck)
mpath = e2e / "clf.json"
classify.write_manifest(mpath, ck, GEOM, 50, {})
m_, man_ = classify.load_classifier(mpath)
_, val_ = split_pipeline(load_corpus(data_))
xv_, yv_ = to_arrays(val_)
g1 = eps.scores(yv_, eps.predict(m_, man_, list(xv_)))   # gate 1 is not under test: let it pass
classify.write_manifest(mpath, ck, GEOM, 50, {"note": "random init, test only",
                                             "held_out": {"accuracy": g1["accuracy"], "macro_f1": g1["macro_f1"]}})
ext_ = e2e / "extent.json"
ext_.write_text(json.dumps({"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8, "variant": "relative",
                            "beats_fixed_window": True, "gated_cut": classify.GATED_CUT}), encoding="utf-8")
out_ = e2e / "results" / "pose_switch.json"
run = subprocess.run([sys.executable, str(ROOT / "cli" / "eval_pose_switch.py"), "--data", str(data_), "--poses", str(poses_),
                      "--classifier", str(mpath), "--extent", str(ext_), "--videos", "V5", "--out", str(out_)],
                     cwd=e2e, capture_output=True, text=True, timeout=600)
tail = (run.stdout + run.stderr)[-1500:]
check("the run exits 0 and writes its results file (was: KeyError 'accuracy' before writing)",
      run.returncode == 0 and out_.is_file(), tail)
res = json.loads(out_.read_text(encoding="utf-8")) if out_.is_file() else {}
al5 = (res.get("videos") or {}).get("V5", {}).get("alignment", {})
check("alignment really passed: V5 scored at offset 0", res.get("scored_videos") == ["V5"]
      and al5.get("best") == 0 and al5.get("sharp") is True, json.dumps(al5.get("best")))
w5 = res.get("window") or {}
check("no punch could be cut: the cut readings are empty and every punch is counted uncuttable",
      w5.get("n") == 0 and w5.get("n_uncuttable") == 20 and w5.get("Y_all") == {"n": 0}
      and (res.get("videos") or {}).get("V5", {}).get("clip_cut", "missing") is None, json.dumps(w5)[:300])
check("the run says so instead of crashing", "no punch had every cut" in run.stdout, tail)
check("through the CLI too: A vs Y_all on all 20 punches, the legacy rule on the 16 it sees",
      (res.get("all_seen") or {}).get("n") == 20 and (res.get("legacy_common") or {}).get("n") == 16
      and res.get("n_legacy_blanked") == 4 and res.get("n_yolo_missed") == 0,
      json.dumps({k: res.get(k) for k in ("n_legacy_blanked", "n_yolo_missed")}))

print("eval_pose_switch.y_clip: a training-cut fallback reads cut_window's own frames")
from boxingvi.data import Track  # noqa: E402

# The fallback cut, exactly as classify.clip_cut builds it for a clip with too few punches.
span_ = (GEOM.n_frames - 1) / GEOM.fps
few_ = classify.ClipCut(span_, min(1.0, GEOM.lead_in_s / span_), 0, 0, False, "training")
bad_ = []
for f_ in (25.0, 29.97, 59.94):
    n_ = int(16 * f_)
    xy_ = np.tile(np.linspace(400.0, 600.0, N_JOINTS)[None, :, None], (n_, 1, 2))
    xy_[:, :, 1] += 300.0
    xy_[:, 0, 0] += 0.001 * np.arange(n_)          # each frame's index coded into the nose's x
    tr_ = Track(np.arange(n_), xy_, f_, conf=np.ones((n_, N_JOINTS)))
    for p_ in range(int(0.5 * f_), int(15 * f_), 7):
        got_ = eps.y_clip(tr_, p_, few_, GEOM, 100.0)
        ref_ = classify.cut_window(tr_, p_, GEOM, 100.0).keypoints
        if got_.shape != ref_.shape or not np.array_equal(got_, ref_, equal_nan=True):
            bad_.append((f_, p_, got_.shape, ref_.shape))
check("Y_clip under a training-cut fallback is cut_window's window, frame for frame, at 25, 29.97 and "
      "59.94 fps (clip_window + cut_span read 8 of its 9 frames there)", not bad_, str(bad_[:3]))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
