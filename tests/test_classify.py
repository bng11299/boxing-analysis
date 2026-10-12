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
for k, v in (("duration_s", 2.0), ("duration_s", float("nan")), ("fps", 0.0), ("torso_px", 1.0), ("lead_in_s", 0.5),
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
# Worked by hand, not by the code's formula: a 3/24 s lead-in is 7.493 frames at
# 59.946 fps, and each 1/24 s step is 2.498 frames; nearest frames of -7.493 + 2.498 k.
want = peak + np.array([-7, -5, -2, 0, 2, 5, 7, 10, 12])
check("frames are sampled at the training rate (24 fps) from 60 fps footage, starting a lead-in before the peak",
      np.array_equal(got, want), f"{got} vs {want}")
check("the window spans 0.375 s of the clip, not 9 consecutive 60 fps frames", want[-1] - want[0] > 8 * 2)
check("the peak frame itself is one of the samples", peak in got)

print("round trip: a window cut at the finder's peak is the corpus clip")
f24 = 24.0
starts = [2.0 + 3 * i for i in range(12)]
cont = body(int(40 * f24), f24, hx=960, hy=600, punches=starts, hand=1)   # inside a 1080p frame
sig24 = punchfind.wrist_signal(cont, "relative")
pk, _ = punchfind.detect(cont, sig24, 1.0, 0.4)
check("control: the finder finds each planted punch once", len(pk) == len(starts), str(pk))
LEAD = 3   # each corpus clip starts 3 frames before the finder's peak and runs 9 frames
rt_clips = []
for q in pk:
    kp = np.zeros((25, N_JOINTS, 3), np.float32)
    kp[:9, :, 0] = cont.xy[q - LEAD:q - LEAD + 9, :, 0] / w
    kp[:9, :, 1] = cont.xy[q - LEAD:q - LEAD + 9, :, 1] / h
    kp[:9, :, 2] = 1
    rt_clips.append(Clip(kp, 1, "", "V1"))
g_rt = classify.training_geometry(rt_clips, "relative")
check("training_geometry puts the peak where the clips have it: 3 frames in, at 24 fps",
      abs(g_rt.lead_in_s - LEAD / 24) < 1e-12 and g_rt.n_frames == 9, str(g_rt))
cont_coded = Track(cont.frame_ids, cont.xy.copy(), f24)
cont_coded.xy[:, 0, 0] += 0.001 * cont.frame_ids
ok_rt = True
for q in pk:
    kq = classify.cut_window(cont_coded, int(q), g_rt, punchfind.body_scale(cont)).keypoints
    s_rt = g_rt.torso_px / 100.0
    fr = np.round(((kq[:, 0, 0] - g_rt.hip_x) * w) / s_rt / 0.001)
    ok_rt &= np.array_equal(fr, np.arange(q - LEAD, q - LEAD + 9))
check("cut_window at each finder peak returns exactly the corpus clip's 9 frames", bool(ok_rt))
half = replace(GEOM, lead_in_s=3.5 / 24)
offs = []
for q in (480, 481):   # an even and an odd peak frame, at exactly 24 fps
    kq = classify.cut_window(cont_coded, q, half, 100.0).keypoints
    offs.append(np.round(((kq[:, 0, 0] - half.hip_x) * w) / 2 / 0.001) - q)
check("a half-frame lead-in samples the same offsets whatever the peak frame's parity (ties go up)",
      np.array_equal(offs[0], offs[1]) and np.array_equal(offs[0], np.arange(-3, 6)), str(offs))

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
raises("write_manifest: extra fields may not replace the pinned hash",
       lambda: classify.write_manifest(tmp / "x.json", ck, GEOM, 50, {"sha256": "0" * 64}))
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
big = json.loads(mpath.read_text(encoding="utf-8")); big["pad"] = "x" * 70_000
(tmp / "big.json").write_text(json.dumps(big), encoding="utf-8")
raises("manifest: an oversized file is refused before it is parsed", lambda: classify.load_classifier(tmp / "big.json"))
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

print("punch extent: the width of the wrist-speed peak")
ids60 = np.arange(400)
flat = body(400, 60.0)


def tri(centre, half_width, frames=ids60, height=4.0):
    """A triangular speed peak on the left hand: `height` at `centre`, zero `half_width` frames away."""
    sg = np.zeros((len(frames), 2))
    sg[:, 0] = np.clip(1 - np.abs(frames - centre) / half_width, 0, None) * height
    return sg


w7, cut7 = punchfind.peak_width(flat, tri(100, 7), 100, 0)
check("a triangle that falls to zero 7 frames out is 7 frames wide at half height (3.5 each side, interpolated)",
      abs(w7 - 7 / 60) < 1e-9 and not cut7, f"{w7 * 60:.3f} frames")
w14, _ = punchfind.peak_width(flat, tri(100, 14), 100, 0)
check("a peak twice as wide measures twice as wide", abs(w14 - 2 * w7) < 1e-9)
check("the width does not depend on the peak's height", abs(punchfind.peak_width(flat, tri(100, 7, height=9.0), 100, 0)[0] - w7) < 1e-9)
_, cut_edge = punchfind.peak_width(flat, tri(2, 7), 2, 0)
check("a peak still above half height at the first frame of the pose is censored", cut_edge)
gap_ids = np.r_[np.arange(0, 103), np.arange(140, 437)]
gapped = body(len(gap_ids), 60.0, ids=gap_ids)
_, cut_gap = punchfind.peak_width(gapped, tri(100, 7, frames=gap_ids), 100, 0)
check("... and so is one that runs into a gap in the pose (never measured across it)", cut_gap)
raises("width: a hand other than 0 or 1 is refused", lambda: punchfind.peak_width(flat, tri(100, 7), 100, 2))
raises("width: a level of 1.0 is refused", lambda: punchfind.peak_width(flat, tri(100, 7), 100, 0, 1.0))
raises("width: a peak frame that is not in the track is refused", lambda: punchfind.peak_width(gapped, tri(100, 7, frames=gap_ids), 120, 0))
raises("width: the other hand, which has no speed there, is refused (not its peak)",
       lambda: punchfind.peak_width(flat, tri(100, 7), 100, 1))
raises("width: a signal of the wrong shape is refused", lambda: punchfind.peak_width(flat, tri(100, 7)[:50], 100, 0))

print("punch extent: start and end")
check("1 width before the peak and 2 after: 7 frames wide -> frames 93 to 114",
      punchfind.extent(flat, tri(100, 7), 100, 0, 1.0, 2.0) == (93, 114))
a14, b14 = punchfind.extent(flat, tri(100, 14), 100, 0, 1.0, 2.0)
check("a slower punch (twice the width) gets twice the extent: the window follows the tempo",
      (100 - a14, b14 - 100) == (14, 28))
check("an extent never collapses to one frame", punchfind.extent(flat, tri(100, 1), 100, 0, 0.0, 0.01)[1] > 100)
raises("extent: negative widths are refused", lambda: punchfind.extent(flat, tri(100, 7), 100, 0, -1.0, 2.0))
raises("extent: zero widths both sides are refused", lambda: punchfind.extent(flat, tri(100, 7), 100, 0, 0.0, 0.0))
check("span IoU: (0-9) and (5-14) share 5 of 15 frames", abs(punchfind.span_iou((0, 9), (5, 14)) - 5 / 15) < 1e-12)
check("span IoU: identical 1, disjoint 0", punchfind.span_iou((3, 8), (3, 8)) == 1 and punchfind.span_iou((0, 4), (9, 12)) == 0)
raises("span IoU: a span that ends before it starts is refused", lambda: punchfind.span_iou((5, 4), (0, 9)))

print("cutting a punch by its extent")
coded60 = body(400, 60.0)
coded60.xy[:, 0, 0] += 0.001 * coded60.frame_ids


def frames_of(win):
    return np.round(((win.keypoints[:, 0, 0] - GEOM.hip_x) * w) / 2 / 0.001)   # scale 2: torso 100 -> 200 px


cs = classify.cut_span(coded60, 100, 124, GEOM, 100.0)
# Worked by hand: 24 frames at 60 fps is 0.4 s, 9.6 steps of 1/24 s, so 10 samples 2.5 frames apart from 100;
# x.5 goes up: 100 103 105 108 110 113 115 118 120 123.
check("a 60 fps span is resampled at the training rate from its first frame to its last",
      np.array_equal(frames_of(cs), [100, 103, 105, 108, 110, 113, 115, 118, 120, 123]), str(frames_of(cs)))
c24 = body(400, 24.0, hx=960, hy=600)
c24.xy[:, 0, 0] += 0.001 * c24.frame_ids
k24 = classify.cut_span(c24, 50, 61, GEOM, 100.0)
check("at the training rate a span is its own frames, as the corpus cuts a labelled span",
      np.array_equal(frames_of(k24), np.arange(50, 62)))
check("a span longer than the corpus's 25 frames keeps its first 25 (the corpus trims the tail too)",
      np.array_equal(frames_of(classify.cut_span(c24, 50, 99, GEOM, 100.0)), np.arange(50, 75)))
one = classify.cut_span(coded60, 100, 101, GEOM, 100.0)
check("a punch under two training frames long is unclassifiable, with the reason",
      one.keypoints is None and "shorter than two frames" in one.reason)
raises("a span that ends before it starts is refused", lambda: classify.cut_span(coded60, 120, 100, GEOM, 100.0))
raises("a fractional span frame is refused", lambda: classify.cut_span(coded60, 100.5, 120, GEOM, 100.0))
off = classify.cut_span(coded60, 380, 430, GEOM, 100.0)
check("a span running past the end of the pose is unclassifiable when under half of it has pose",
      off.keypoints is None and "pose in only" in off.reason, off.reason)

print("extent ratios from training clips")
whole = []
for q in pk:   # the round-trip track: clips from 5 frames before each finder peak, 14 frames long
    kp = np.zeros((25, N_JOINTS, 3), np.float32)
    kp[:14, :, 0] = cont.xy[q - 5:q + 9, :, 0] / w
    kp[:14, :, 1] = cont.xy[q - 5:q + 9, :, 1] / h
    kp[:14, :, 2] = 1
    whole.append(Clip(kp, 1, "", "V1"))
er = classify.training_extent_ratios(whole, "relative", 0.5)
one_tr = classify._clip_track(whole[0].keypoints, 24.0, "V1")
one_sig = punchfind.wrist_signal(one_tr, "relative")
pk0 = int(np.argmax(one_sig.max(axis=1)))
w0, cut0 = punchfind.peak_width(one_tr, one_sig, pk0, int(np.argmax(one_sig[pk0])), 0.5)
check("control: these clips hold a whole speed peak", not cut0 and er["n_clips"] == len(whole) and er["n_censored"] == 0,
      str(er))
check("lead and tail are the clip's frames before and after its peak, in peak-widths",
      abs(er["lead_widths"] - (pk0 / 24) / w0) < 1e-9 and abs(er["tail_widths"] - ((13 - pk0) / 24) / w0) < 1e-9, str(er))
moving = body(14, 24.0, hx=960, hy=600).xy      # the right wrist already at full speed in the clip's first frame,
moving[:5, 10, 0] += np.linspace(0, 200, 5)     # then still: its speed peak was cut off by the start of the clip
moving[5:, 10, 0] += 200
kp_cut = np.zeros((25, N_JOINTS, 3), np.float32)
kp_cut[:14, :, 0], kp_cut[:14, :, 1], kp_cut[:14, :, 2] = moving[..., 0] / w, moving[..., 1] / h, 1
cut_clips = [Clip(kp_cut, 1, "", "V1") for _ in whole]
mixed = classify.training_extent_ratios(whole + cut_clips, "relative", 0.5)
check("clips whose peak is cut by the clip's first frame are left out and counted, not measured",
      mixed["n_censored"] == len(cut_clips) and mixed["n_clips"] == len(whole)
      and mixed["lead_widths"] == er["lead_widths"], str(mixed))
raises("too few whole clips is refused", lambda: classify.training_extent_ratios(whole[:3], "relative", 0.5))
raises("an unknown variant is refused", lambda: classify.training_extent_ratios(whole, "sideways", 0.5))

print("the extent rule file")
extent_file = tmp / "punch_extent.json"


def extent_json(**kw):
    d = {"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8, "variant": "relative", "beats_fixed_window": True, **kw}
    extent_file.write_text(json.dumps({k: v for k, v in d.items() if v is not None}), encoding="utf-8")
    return lambda: classify.read_extent(extent_file, "relative")


check("a good rule file is read", extent_json()() == {"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8})
raises("a rule that did not beat the fixed window is refused", extent_json(beats_fixed_window=False))
raises("a rule with no verdict is refused (truthy is not True)", extent_json(beats_fixed_window="yes"))
raises("a rule measured with another wrist-speed variant is refused", extent_json(variant="radial"))
raises("a width level of 1.0 is refused", extent_json(frac=1.0))
raises("a missing lead is refused", extent_json(lead_widths=None))
raises("a boolean tail is refused", extent_json(tail_widths=True))
raises("an absurd tail is refused", extent_json(tail_widths=500.0))
extent_file.write_text('{"frac": NaN}', encoding="utf-8")
raises("NaN in the rule file is refused", lambda: classify.read_extent(extent_file, "relative"))
extent_json()

print("windows of different lengths")
def frame_reader(x):
    """A stand-in classifier whose answer is a known function of EVERY frame it is given:
    logit 0 is 100 x the mean nose-x over the resampled frames. Nothing random to be insensitive."""
    out = torch.zeros(len(x), len(CLASSES))
    out[:, 0] = 100 * x[:, 0, :, 0, 0].mean(dim=1)
    return out


def alone(win):   # the stand-in on this window's own frames, with no batching or padding code in the way
    return torch.softmax(frame_reader(prepare_input(win[None].astype(np.float32), 50)), 1).numpy()[0]


short_w = cs.keypoints                               # 10 frames
long_w = short_w[np.r_[0:10, 0:4]].copy()            # 14 frames: the same punch, then four more
long_w[10:, 0, 0] += 0.2                             # ... in which the nose is somewhere else
both = classify.predict_proba(frame_reader, [short_w, long_w], 50)
check("control: a punch's later frames change the answer (so cutting them off would show)",
      not np.allclose(alone(long_w), alone(long_w[:10]), atol=1e-3))
check("a 10-frame and a 14-frame punch classify together exactly as each does read whole and alone",
      np.allclose(both[0], alone(short_w), atol=1e-6) and np.allclose(both[1], alone(long_w), atol=1e-6))
raises("a window longer than the corpus's 25 frames is refused",
       lambda: classify.predict_proba(loaded, [np.ones((26, N_JOINTS, 3), np.float32)], 50))

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
args = ["--rule", str(rule), "--extent", str(extent_file), "--pose", str(tmp / "IMG_7777.pose.npz"),
        "--punches", str(punches), "--classifier", str(mpath), "--out-dir", str(tmp / "o")]
rc = classify_punches.main(args + ["--stance", "orthodox"])
rows = list(csv.DictReader((tmp / "o" / "IMG_7777.classified.csv").open(encoding="utf-8")))
check("end to end: one row per finder punch, every finder column kept",
      rc == 0 and len(rows) == 3 and rows[0]["frame"] == next(csv.DictReader(punches.open(encoding="utf-8")))["frame"])
check("each row has a type from the class list and a confidence in (0, 1]",
      all(r["punch_type"] in CLASSES and 0 < float(r["confidence"]) <= 1 for r in rows), str(rows))
check("each punch records the frames it was read from, around its peak",
      all(int(r["punch_start_frame"]) <= int(r["frame"]) <= int(r["punch_end_frame"]) for r in rows), str(rows[0]))
check("... and they are a tempo-scaled extent, not one fixed length for every clip",
      all(2 <= int(r["punch_end_frame"]) - int(r["punch_start_frame"]) <= 60 for r in rows))
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
(tmp / "IMG_8888.punches.csv").write_text(edge, encoding="utf-8")
part = pose_yolo.PoseSequence(xy.astype(np.float32), cf, np.r_[np.ones(n - 3, bool), np.zeros(3, bool)],
                              fps, 1080, 1920, "IMG_7777.mov", "test")
xy_p = part.xy.copy(); xy_p[n - 3:] = np.nan
part = pose_yolo.PoseSequence(xy_p, cf, part.detected, fps, 1080, 1920, "IMG_7777.mov", "test")
part.save(tmp / "IMG_8888.pose.npz")
classify_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--punches", str(tmp / "IMG_8888.punches.csv"),
                       "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file), "--stance", "orthodox",
                       "--out-dir", str(tmp / "o")])
raises("a punches CSV named for another clip is refused",
       lambda: classify_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--punches", str(punches),
                                      "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file), "--stance", "orthodox",
                                      "--out-dir", str(tmp / "o")]))
radial = tmp / "radial.json"
radial.write_text(json.dumps({"chosen": {"variant": "radial", "threshold": 1.0, "min_gap_s": 0.4}}), encoding="utf-8")
raises("a finder rule with another wrist-speed variant than the window's is refused",
       lambda: classify_punches.main(args + ["--rule", str(radial), "--stance", "orthodox"]))
extent_json(beats_fixed_window=False)
raises("an extent rule that did not beat the fixed window is refused by the CLI",
       lambda: classify_punches.main(args + ["--stance", "orthodox"]))
extent_json()
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
check("control: no matched punch is unclassifiable on the whole clip", sc.get("matched_unclassifiable") == 0)
lab8 = tmp / "IMG_8888.labels.json"
lab8.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
    {"start": int(3.0 * fps), "end": int(3.25 * fps), "label": "jab"},
    {"start": n - 4, "end": n - 1, "label": "jab"}]}), encoding="utf-8")   # the punch at the end of the pose
classify_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--punches", str(tmp / "IMG_8888.punches.csv"),
                       "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file), "--stance", "orthodox",
                       "--out-dir", str(tmp / "o"), "--labels", str(lab8)])
sc8 = json.loads((tmp / "o" / "IMG_8888.classify_score.json").read_text(encoding="utf-8"))
check("a matched punch that could not be classified is counted, not silently dropped from the score",
      sc8["matched_unclassifiable"] == 1 and sc8["matched_detections"]["n"] == 1, str(sc8))


def punches_with(text):
    p = tmp / "p.csv"
    p.write_text(text, encoding="utf-8")
    return lambda: classify_punches.read_punches(p, n, fps)


hdr = ",".join(classify_punches.PUNCH_HEADER) + "\n"
raises("punches: another header is refused", punches_with("frame,time,hand\n1,0.0,left\n"))
raises("punches: a frame past the clip is refused", punches_with(hdr + f"{n},{n / fps:.3f},left,1.0,no\n"))
raises("punches: a negative frame is refused", punches_with(hdr + "-1,0.000,left,1.0,no\n"))
raises("punches: a fractional frame is refused", punches_with(hdr + "1.5,0.025,left,1.0,no\n"))
raises("punches: a non-ASCII digit is refused cleanly", punches_with(hdr + "²,0.033,left,1.0,no\n"))
raises("punches: a time that does not match the frame (a CSV from another clip) is refused",
       punches_with(hdr + "600,5.000,left,1.0,no\n"))
raises("punches: a NaN time is refused", punches_with(hdr + "600,nan,left,1.0,no\n"))
raises("punches: an unknown hand is refused", punches_with(hdr + f"600,{600 / fps:.3f},both,1.0,no\n"))
raises("punches: a short row is refused", punches_with(hdr + "600,10.009\n"))
raises("punches: an unknown interpolation flag is refused", punches_with(hdr + f"600,{600 / fps:.3f},left,1.0,maybe\n"))

print("train_classifier reproduces only a reference that records how it was trained")
import train_classifier  # noqa: E402
from boxingvi.stgcnpp import PRETRAINED  # noqa: E402

good_run = {"model": "st-gcn++ pretrained", "accuracy": 0.8, "macro_f1": 0.7, "t_out": 50,
            "pretrained_from": {"sha256": PRETRAINED[train_classifier.SPEC]["sha256"]}}


def reference(**top):
    p = tmp / "ref.json"
    p.write_text(json.dumps({"seed": 0, "balanced": False, "include_v6": False, "epochs": 30,
                             "results": [good_run], **top}), encoding="utf-8")
    return p


check("the epochs come from the reference's stgcnpp_epochs, not its epochs",
      train_classifier.reference_metrics(reference(stgcnpp_epochs=20), 0, 50)["epochs"] == 20)
raises("a reference without stgcnpp_epochs is refused (the 09-16 file: 30 for the other arms, 20 for ST-GCN++)",
       lambda: train_classifier.reference_metrics(reference(), 0, 50))
raises("another seed is refused", lambda: train_classifier.reference_metrics(reference(stgcnpp_epochs=20, seed=1), 0, 50))
raises("another pretrained checkpoint is refused", lambda: train_classifier.reference_metrics(
    reference(stgcnpp_epochs=20, results=[{**good_run, "pretrained_from": {"sha256": "0" * 64}}]), 0, 50))

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
