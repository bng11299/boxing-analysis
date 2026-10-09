"""Tests for the classify step: windows around detected punches, the classifier's
manifest and checkpoint, the hand check, and classify_punches.py end to end.

Synthetic fixtures only, and a randomly initialised classifier: these prove the
plumbing and the refusals. No number here says anything about boxing.

Run:  python tests/test_classify.py
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from boxingvi import classify, pose_yolo, punchfind  # noqa: E402
from boxingvi.data import CLASSES, N_JOINTS, SOURCE_SIZE, Clip, Track  # noqa: E402
from boxingvi.stgcnpp import STGCNPP, prepare_input  # noqa: E402

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


# A standing body, torso 100 px, hips centred at (hx, hy): every joint placed, so
# nothing sits at (0, 0).
POSE = np.array([
    [0, -160], [-8, -168], [8, -168], [-16, -164], [16, -164],      # head
    [-50, -100], [50, -100], [-60, -50], [60, -50], [-55, -10], [55, -10],  # shoulders, elbows, wrists
    [-30, 0], [30, 0], [-32, 90], [32, 90], [-34, 180], [34, 180],   # hips, knees, ankles
], dtype=np.float64)


def body(n, fps, hx=540.0, hy=1100.0, scale=1.0, punches=(), hand=0, ids=None, conf=None):
    t = np.arange(n) / fps
    xy = np.broadcast_to(POSE * scale + (hx, hy), (n, N_JOINTS, 2)).copy()
    for p in punches:
        ph = np.clip((t - p) / 0.25, 0, 1)
        xy[:, 9 + hand, 0] += (-1 if hand == 0 else 1) * 120 * scale * np.sin(np.pi * ph)
    f = np.arange(n) if ids is None else np.asarray(ids)
    return Track(frame_ids=f.astype(np.int64), xy=xy, fps=fps, conf=conf)


GEOM = classify.WindowGeometry(duration_s=0.375, lead_in_s=0.125, fps=24.0, torso_px=200.0,
                               hip_x=0.5, hip_y=0.55, variant="relative", n_clips=100)

print("window geometry")
check("9 frames per window at 24 fps for 0.375 s", GEOM.n_frames == 9)
for k, v in (("duration_s", float("nan")), ("fps", 0.0), ("torso_px", 1.0), ("lead_in_s", 0.5),
             ("hip_x", 1.2), ("variant", "sideways"), ("n_clips", True), ("duration_s", True),
             ("duration_s", 0.01)):
    raises(f"geometry refuses {k}={v!r}", lambda k=k, v=v: replace(GEOM, **{k: v}))

print("training geometry, from synthetic clips")
w, h = SOURCE_SIZE


def corpus_clip(n_real, fps_src, peak_at, hip=(960, 600), torso=200.0):
    tr = body(n_real, fps_src, hx=hip[0], hy=hip[1], scale=torso / 100, punches=[peak_at - 0.125], hand=1)
    kp = np.zeros((25, N_JOINTS, 3), np.float32)
    kp[:n_real, :, 0] = tr.xy[:, :, 0] / w
    kp[:n_real, :, 1] = tr.xy[:, :, 1] / h
    kp[:n_real, :, 2] = 1
    return kp


clips = [Clip(corpus_clip(9, 24.0, 4 / 24), 1, "", "V1") for _ in range(20)]
g = classify.training_geometry(clips, "relative")
check("duration is the real frames over the native rate (9 / 24 s), padding excluded",
      abs(g.duration_s - 9 / 24) < 1e-12, str(g))
check("torso and hip centre recovered in 1080p pixels and frame fractions (float32 corpus)",
      abs(g.torso_px - 200) < 1e-3 and abs(g.hip_x - 0.5) < 1e-6 and abs(g.hip_y - 600 / 1080) < 1e-6, str(g))
later = classify.training_geometry([Clip(corpus_clip(9, 24.0, 6 / 24), 1, "", "V1") for _ in range(20)], "relative")
check("the lead-in follows the wrist-speed peak: a punch 2 frames later moves it 2 frames later",
      abs((later.lead_in_s - g.lead_in_s) - 2 / 24) < 1e-9, f"{g.lead_in_s:.4f} -> {later.lead_in_s:.4f}")
check("the lead-in sits inside the clip", 0 < g.lead_in_s < g.duration_s)
check("fps is the training clips' native rate", g.fps == 24.0)
raises("a clip from a video with no recorded frame rate is refused",
       lambda: classify.training_geometry([Clip(corpus_clip(9, 24.0, 0.2), 1, "", "V99")] * 20, "relative"))
raises("too few measurable clips is refused, not a geometry from 3 clips",
       lambda: classify.training_geometry(clips[:3], "relative"))
dropped = corpus_clip(9, 24.0, 4 / 24)
dropped[:, 10] = 0   # the punching wrist dropped in every frame, as AlphaPose marks it
g2 = classify.training_geometry(clips + [Clip(dropped, 1, "", "V1")] * 5, "relative")
check("a dropped joint at (0, 0) is never differenced into a fake peak (medians unmoved)",
      g2.lead_in_s == g.lead_in_s and g2.duration_s == g.duration_s)

print("cutting windows")
fps = 59.946
tr = body(1200, fps, punches=[5.0], hand=0)
peak = int(round(5.125 * fps))
win = classify.cut_window(tr, peak, GEOM, punchfind.body_scale(tr))
check("a window has the training length", win.keypoints is not None and win.keypoints.shape == (9, N_JOINTS, 3))
k = win.keypoints
hip_px = ((k[:, 11, :2] + k[:, 12, :2]) / 2 * np.array([w, h]))
check("hips land at the training hip centre", np.allclose(hip_px, (0.5 * w, 0.55 * h), atol=1e-3), str(hip_px[0]))
sh = (k[:, 5, :2] + k[:, 6, :2]) / 2 * np.array([w, h])
check("the torso is scaled to the training length in 1080p pixels",
      np.allclose(np.linalg.norm(sh - hip_px, axis=1), 200.0, atol=1e-3))
check("every joint is observed when the pose is whole", (k[..., 2] == 1).all())
# Which video frames were sampled: encode the frame id in a joint, then read it back.
coded = Track(tr.frame_ids, tr.xy.copy(), fps)
coded.xy[:, 0, 0] = tr.xy[:, 0, 0] + 0.001 * tr.frame_ids
cw = classify.cut_window(coded, peak, GEOM, 100.0).keypoints
got = np.round((cw[:, 0, 0] * w - 0.5 * w) / 2 / 0.001)   # scale 2 (torso 100 -> 200 px)
start = peak / fps - GEOM.lead_in_s
want = np.round((start + np.arange(9) / 24.0) * fps)
check("frames are sampled at the training rate (24 fps) from 60 fps footage, starting a lead-in before the peak",
      np.array_equal(got, want), f"{got} vs {want}")
check("the window spans 0.375 s of the clip, not 9 consecutive 60 fps frames", want[-1] - want[0] > 8 * 2)

gap_ids = np.r_[np.arange(0, peak - 3), np.arange(peak + 4, 1200)]
gapped = body(len(gap_ids), fps, punches=[5.0], ids=gap_ids)
gw = classify.cut_window(gapped, peak - 4, GEOM, 100.0)
# Anchored 4 frames earlier, so the sampled frames are `want - 4`; those inside the hole have no pose.
missing = [j for j, f in enumerate(want - 4) if peak - 3 <= f < peak + 4]
check("a sample time with no pose frame is a zero frame with validity 0, as the corpus pads",
      gw.keypoints is not None and all((gw.keypoints[j] == 0).all() for j in missing) and len(missing) > 0,
      f"missing {missing}")
check("... and the window says how many frames it saw", gw.observed == 9 - len(missing))
c = np.ones((1200, N_JOINTS)); c[:, 15] = 0.1
lw = classify.cut_window(body(1200, fps, conf=c), peak, GEOM, 100.0).keypoints
check("a joint below the confidence floor is zeroed with validity 0", (lw[:, 15] == 0).all() and (lw[:, 14, 2] == 1).all())
far = classify.cut_window(tr, 5000, GEOM, 100.0)
check("a window past the end of the pose is unclassifiable, with a reason",
      far.keypoints is None and "pose in only 0" in far.reason, far.reason)
thin_ids = np.r_[np.arange(0, peak - 5), np.arange(peak + 12, 1200)]
thin = classify.cut_window(body(len(thin_ids), fps, ids=thin_ids), peak, GEOM, 100.0)
check("a window with pose in under half its frames is unclassifiable (the resampler would invent it)",
      thin.keypoints is None and thin.observed < 5, f"{thin.observed} {thin.reason}")
nh = np.ones((1200, N_JOINTS)); nh[:, 11] = 0.0
nohip = classify.cut_window(body(1200, fps, conf=nh), peak, GEOM, 100.0)
check("no frame with both hips -> unclassifiable, not centred on a guess", nohip.keypoints is None and "hips" in nohip.reason)
wild = Track(tr.frame_ids, tr.xy.copy(), fps)
wild.xy[peak, 9] = (90000, 90000)  # one wrist flung across the room
ww = classify.cut_window(wild, peak, GEOM, 100.0)
check("a joint flung far outside the frame -> unclassifiable, not a crash in the batch",
      ww.keypoints is None and "outside" in ww.reason)
raises("a fractional peak frame is refused", lambda: classify.cut_window(tr, 300.5, GEOM, 100.0))
raises("a boolean peak frame is refused", lambda: classify.cut_window(tr, True, GEOM, 100.0))
raises("a NaN torso is refused", lambda: classify.cut_window(tr, peak, GEOM, float("nan")))
v6like = Track(tr.frame_ids, tr.xy, fps, conf=None)
check("a track without confidence (V6) counts every present joint",
      (classify.cut_window(v6like, peak, GEOM, 100.0).keypoints[..., 2] == 1).all())
near = body(1200, fps, scale=2.5, hx=300, hy=900, punches=[5.0], hand=0)
kn = classify.cut_window(near, peak, GEOM, punchfind.body_scale(near)).keypoints
check("a nearer, off-centre boxer maps to the same skeleton (position and scale removed)",
      np.allclose(kn, k, atol=1e-5))

print("manifest and checkpoint")
tmp = Path(tempfile.mkdtemp())
torch.manual_seed(0)
model = STGCNPP()
ck = tmp / "clf.pt"
torch.save(model.state_dict(), ck)
mpath = tmp / "clf.json"
classify.write_manifest(mpath, ck, GEOM, 50, {"note": "random init, test only"})
loaded, man = classify.load_classifier(mpath)
check("a manifest round-trips, and the geometry comes back as a WindowGeometry",
      man["geometry"] == GEOM and man["t_out"] == 50)
probs = classify.predict_proba(loaded, [k, kn], 50)
check("probabilities: one row per window, each summing to 1",
      probs.shape == (2, len(CLASSES)) and np.allclose(probs.sum(axis=1), 1))
with torch.no_grad():
    direct = torch.softmax(model.eval()(prepare_input(np.stack([k, kn]), 50)), 1).numpy()
check("the loaded model predicts exactly as the saved one", np.allclose(probs, direct))
check("predict_proba of no windows is an empty array", classify.predict_proba(loaded, [], 50).shape == (0, len(CLASSES)))


def manifest_with(**changes):
    m = json.loads(mpath.read_text(encoding="utf-8"))
    for key, val in changes.items():
        if val is None:
            m.pop(key, None)
        else:
            m[key] = val
    bad = tmp / "bad.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    return lambda: classify.load_classifier(bad)


raises("manifest: another version is refused", manifest_with(manifest_version=2))
raises("manifest: a different class list is refused", manifest_with(classes=list(reversed(CLASSES))))
raises("manifest: a checkpoint path outside its folder is refused", manifest_with(checkpoint="../clf.pt"))
raises("manifest: a checkpoint in a subfolder is refused", manifest_with(checkpoint="sub/clf.pt"))
raises("manifest: an absolute checkpoint path is refused", manifest_with(checkpoint=str(ck)))
raises("manifest: a short sha256 is refused", manifest_with(sha256="abc"))
raises("manifest: '..' as the checkpoint name is refused as a name, before any file lookup",
       manifest_with(checkpoint=".."))
raises("manifest: an extra geometry field is refused",
       manifest_with(geometry={**asdict(GEOM), "fudge": 1}))
raises("manifest: a boolean t_out is refused", manifest_with(t_out=True))
raises("manifest: a missing size is refused", manifest_with(size=None))
(tmp / "nan.json").write_text(mpath.read_text(encoding="utf-8").replace('"t_out": 50', '"t_out": NaN'), encoding="utf-8")
raises("manifest: NaN in the JSON is refused", lambda: classify.load_classifier(tmp / "nan.json"))
raises("checkpoint: a size that disagrees with the manifest is refused", manifest_with(size=12345))
raises("checkpoint: a sha256 that disagrees with the manifest is refused", manifest_with(sha256="0" * 64))
raises("checkpoint: a missing file is refused", manifest_with(checkpoint="gone.pt"), FileNotFoundError)


MARKER = tmp / "pickle-code-ran"


class Boom:
    def __reduce__(self):   # unpickling this runs code: it writes MARKER
        return (exec, (f"open({str(MARKER)!r}, 'w').write('x')",))


evil = tmp / "evil.pt"
evil.write_bytes(pickle.dumps({"backbone.x": Boom()}))
classify.write_manifest(tmp / "evil.json", evil, GEOM, 50, {})
raises("checkpoint: a pickle carrying code is refused, even with a matching hash",
       lambda: classify.load_classifier(tmp / "evil.json"))
check("... and its code never ran (the safe loader refused it, not a later check)", not MARKER.exists())
torch.save([torch.zeros(1)], tmp / "list.pt")
classify.write_manifest(tmp / "list.json", tmp / "list.pt", GEOM, 50, {})
raises("checkpoint: tensors that are not a state_dict are refused", lambda: classify.load_classifier(tmp / "list.json"))
sd = model.state_dict(); sd.pop("cls_head.fc_cls.bias")
torch.save(sd, tmp / "part.pt")
classify.write_manifest(tmp / "part.json", tmp / "part.pt", GEOM, 50, {})
raises("checkpoint: a state_dict missing a tensor is refused (strict load)",
       lambda: classify.load_classifier(tmp / "part.json"), RuntimeError)

print("hands")
check("orthodox: left leads", classify.hand_side("left", "orthodox") == "lead"
      and classify.hand_side("right", "orthodox") == "rear")
check("southpaw: right leads", classify.hand_side("right", "southpaw") == "lead")
check("jab, lead hook and lead uppercut are lead; cross, rear hook and rear uppercut are rear",
      [classify.class_side(c) for c in CLASSES] == ["rear", "lead", "lead", "lead", "rear", "rear"])
raises("an unknown stance is refused", lambda: classify.hand_side("left", "switch"))
raises("an unknown hand is refused", lambda: classify.hand_side("both", "orthodox"))
raises("an unknown punch type is refused", lambda: classify.class_side("haymaker"))
ag = classify.hand_agreement(["lead", "lead", "rear", "rear"], ["lead", "rear", "rear", "rear"])
check("agreement and its chance level (independent readings, same proportions)",
      ag["agree"] == 0.75 and abs(ag["chance"] - (0.5 * 0.25 + 0.5 * 0.75)) < 1e-12, str(ag))
check("no punches: agreement is None, not 0%", classify.hand_agreement([], [])["agree"] is None)
raises("readings of different lengths are refused", lambda: classify.hand_agreement(["lead"], []))

print("classify_punches CLI")
import classify_punches  # noqa: E402
import find_punches  # noqa: E402

n, fps = 1200, 59.946
lt = body(n, fps, punches=[3.0, 9.0], hand=0)
rt = body(n, fps, punches=[14.0], hand=1)
xy = lt.xy.copy(); xy[:, 10] = rt.xy[:, 10]
cf = np.full((n, N_JOINTS), 0.9, np.float32)
pose = pose_yolo.PoseSequence(xy.astype(np.float32), cf, np.ones(n, bool), fps, 1080, 1920, "IMG_7777.mov", "test")
pose.save(tmp / "IMG_7777.pose.npz")
rule = tmp / "rule.json"
rule.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": 1.0, "min_gap_s": 0.4}}), encoding="utf-8")
find_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o")])
punches = tmp / "o" / "IMG_7777.punches.csv"
args = ["--pose", str(tmp / "IMG_7777.pose.npz"), "--punches", str(punches), "--classifier", str(mpath),
        "--out-dir", str(tmp / "o")]
rc = classify_punches.main(args + ["--stance", "orthodox"])
rows = list(csv.DictReader((tmp / "o" / "IMG_7777.classified.csv").open(encoding="utf-8")))
check("end to end: one row per finder punch, every finder column kept",
      rc == 0 and len(rows) == 3 and rows[0]["frame"] == next(csv.DictReader(punches.open(encoding="utf-8")))["frame"])
check("each row has a type from the class list and a confidence in (0, 1]",
      all(r["punch_type"] in CLASSES and 0 < float(r["confidence"]) <= 1 for r in rows), str(rows))
check("orthodox: left-hand punches are lead, right-hand punches rear",
      [r["lead_rear"] for r in rows] == ["lead", "lead", "rear"])
check("hands_agree says whether the type's hand is the finder's",
      all((r["hands_agree"] == "yes") == (r["type_hand"] == r["lead_rear"]) for r in rows))
classify_punches.main(args + ["--stance", "southpaw"])
rows_s = list(csv.DictReader((tmp / "o" / "IMG_7777.classified.csv").open(encoding="utf-8")))
check("southpaw flips lead and rear, and nothing else",
      [r["lead_rear"] for r in rows_s] == ["rear", "rear", "lead"]
      and [r["punch_type"] for r in rows_s] == [r["punch_type"] for r in rows])
raises("no stance -> refused (the tool does not guess one)",
       lambda: classify_punches.main(args), SystemExit)

# A punch near the end whose window runs past the pose: kept, marked, not guessed.
edge = punches.read_text(encoding="utf-8") + f"{n - 1},{(n - 1) / fps:.3f},left,9.000,no\n"
(tmp / "edge.csv").write_text(edge, encoding="utf-8")
part = pose_yolo.PoseSequence(xy.astype(np.float32), cf, np.r_[np.ones(n - 3, bool), np.zeros(3, bool)],
                              fps, 1080, 1920, "IMG_7777.mov", "test")
xy_p = part.xy.copy(); xy_p[n - 3:] = np.nan
part = pose_yolo.PoseSequence(xy_p, cf, part.detected, fps, 1080, 1920, "IMG_7777.mov", "test")
part.save(tmp / "IMG_8888.pose.npz")
classify_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--punches", str(tmp / "edge.csv"),
                       "--classifier", str(mpath), "--stance", "orthodox", "--out-dir", str(tmp / "o")])
rows_e = list(csv.DictReader((tmp / "o" / "IMG_8888.classified.csv").open(encoding="utf-8")))
check("a punch whose window is mostly past the pose is kept as unclassified, with the reason",
      rows_e[-1]["punch_type"] == "unclassified" and rows_e[-1]["confidence"] == "" and rows_e[-1]["note"] != "",
      str(rows_e[-1]))
check("... and the others are still classified", all(r["punch_type"] in CLASSES for r in rows_e[:-1]))

lab = tmp / "IMG_7777.labels.json"
spans = [{"start": int(s * fps), "end": int((s + 0.25) * fps), "label": l}
         for s, l in ((3.0, "jab"), (9.0, "unclear"), (14.0, "cross"))]
lab.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": spans}), encoding="utf-8")
classify_punches.main(args + ["--stance", "orthodox", "--labels", str(lab)])
sc = json.loads((tmp / "o" / "IMG_7777.classify_score.json").read_text(encoding="utf-8"))
check("scored against labels: 'unclear' punches are not scored for type",
      sc["matched_detections"]["n"] == 2 and sc["every_labelled_punch"]["n"] == 2, str(sc))
check("the score names the classifier it scored, by hash", sc["classifier_sha256"] == man["sha256"])


def punches_with(text):
    p = tmp / "p.csv"
    p.write_text(text, encoding="utf-8")
    return lambda: classify_punches.read_punches(p, n, fps)


hdr = ",".join(classify_punches.PUNCH_HEADER) + "\n"
raises("punches: another header is refused", punches_with("frame,time,hand\n1,0.0,left\n"))
raises("punches: a frame past the clip is refused", punches_with(hdr + f"{n},{n / fps:.3f},left,1.0,no\n"))
raises("punches: a negative frame is refused", punches_with(hdr + "-1,0.000,left,1.0,no\n"))
raises("punches: a fractional frame is refused", punches_with(hdr + "1.5,0.025,left,1.0,no\n"))
raises("punches: a time that does not match the frame (a CSV from another clip) is refused",
       punches_with(hdr + "600,5.000,left,1.0,no\n"))
raises("punches: a NaN time is refused", punches_with(hdr + "600,nan,left,1.0,no\n"))
raises("punches: an unknown hand is refused", punches_with(hdr + f"600,{600 / fps:.3f},both,1.0,no\n"))
raises("punches: a short row is refused", punches_with(hdr + "600,10.009\n"))
raises("punches: an unknown interpolation flag is refused", punches_with(hdr + f"600,{600 / fps:.3f},left,1.0,maybe\n"))

print("run_baseline saves the model it scored")
from run_baseline import run_stgcnpp  # noqa: E402

rng = np.random.default_rng(0)
xs = np.zeros((24, 25, N_JOINTS, 3), np.float32)
xs[:, :9, :, :2] = rng.uniform(0.3, 0.7, (24, 9, N_JOINTS, 2)); xs[:, :9, :, 2] = 1
ys = np.arange(24) % len(CLASSES)
r = run_stgcnpp(xs[:18], ys[:18], xs[18:], ys[18:], epochs=1, seed=0, save_to=tmp / "saved.pt")
reloaded = STGCNPP(); reloaded.load_state_dict(torch.load(tmp / "saved.pt", weights_only=True)); reloaded.eval()
with torch.no_grad():
    again = reloaded(prepare_input(xs[18:], 50)).argmax(1).numpy()
check("the saved state_dict reproduces the predictions run_stgcnpp scored", np.array_equal(again, r["_pred"]))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
