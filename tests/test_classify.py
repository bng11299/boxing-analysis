"""Tests for the classify step: windows around detected punches, the classifier's
manifest and checkpoint, the hand check, and classify_punches.py end to end.

Synthetic fixtures only, and a randomly initialised classifier: these prove the
plumbing and the refusals. No number here says anything about boxing.

Run:  python tests/test_classify.py
"""

from __future__ import annotations

import csv
import hashlib
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
from boxingvi.data import CLASSES, N_JOINTS, SOURCE_SIZE, Clip, Span, Track  # noqa: E402
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


def body(n, fps, hx=540.0, hy=1100.0, scale=1.0, punches=(), hand=0, ids=None, conf=None, dur=0.25):
    t = np.arange(n) / fps
    xy = np.broadcast_to(POSE * scale + (hx, hy), (n, N_JOINTS, 2)).copy()
    for p in punches:
        ph = np.clip((t - p) / dur, 0, 1)
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
check("an unsure joint (confidence 0.1) is KEPT: the classifier was trained on whole skeletons",
      (lw[:, 15, 2] == 1).all() and (lw[:, 15, :2] != 0).all())
lw3 = classify.cut_window(body(1200, fps, conf=c), peak, GEOM, 100.0, min_conf=0.3).keypoints
check("control: asked to, the old 0.3 rule still blanks that joint", (lw3[:, 15] == 0).all() and (lw3[:, 14, 2] == 1).all())
c0 = np.ones((1200, N_JOINTS)); c0[:, 15] = 0.0
lw0 = classify.cut_window(body(1200, fps, conf=c0), peak, GEOM, 100.0).keypoints
check("a joint the pose model did not detect at all (confidence 0) is blanked, validity 0",
      (lw0[:, 15] == 0).all() and (lw0[:, 14, 2] == 1).all())
check("the pipeline's default is to keep every detected joint", classify.CLASSIFY_MIN_CONF == 0.0)
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
e7 = punchfind.extent(flat, tri(100, 7), 100, 0, 1.0, 2.0)
check("1 width before the peak and 2 after: 7 frames wide -> frames 93 to 114, nothing flagged",
      (e7.start, e7.end, e7.censored, e7.capped) == (93, 114, False, False), str(e7))
a14, b14 = punchfind.extent(flat, tri(100, 14), 100, 0, 1.0, 2.0)[:2]
check("a slower punch (twice the width) gets twice the extent: the window follows the tempo",
      (100 - a14, b14 - 100) == (14, 28))
check("an extent never collapses to one frame", punchfind.extent(flat, tri(100, 1), 100, 0, 0.0, 0.01)[1] > 100)
raises("extent: negative widths are refused", lambda: punchfind.extent(flat, tri(100, 7), 100, 0, -1.0, 2.0))
raises("extent: zero widths both sides are refused", lambda: punchfind.extent(flat, tri(100, 7), 100, 0, 0.0, 0.0))
check("span IoU: (0-9) and (5-14) share 5 of 15 frames", abs(punchfind.span_iou((0, 9), (5, 14)) - 5 / 15) < 1e-12)
check("span IoU: identical 1, disjoint 0", punchfind.span_iou((3, 8), (3, 8)) == 1 and punchfind.span_iou((0, 4), (9, 12)) == 0)
raises("span IoU: a span that ends before it starts is refused", lambda: punchfind.span_iou((5, 4), (0, 9)))

print("punch extent: a punch too long to show, and a wrist that drops out")
wide = punchfind.extent(flat, tri(200, 60), 200, 0, 1.33, 1.83, max_s=1.0)   # 1.0 s wide: 3.16 s of extent
check("an extent longer than the cap is shortened to it and flagged",
      wide.capped and (wide.end - wide.start) / 60.0 <= 1.0 + 1e-9, str(wide))
check("... on both sides in proportion, so the peak stays where the rule puts it (1.33 : 1.83, 42% in)",
      abs((200 - wide.start) / (wide.end - wide.start) - 1.33 / 3.16) < 0.02, str(wide))
uncapped = punchfind.extent(flat, tri(200, 60), 200, 0, 1.33, 1.83)
check("control: without a cap the same punch runs 3.16 s", abs((uncapped.end - uncapped.start) / 60 - 3.16) < 0.05
      and not uncapped.capped)
shown = classify.cut_span(flat, *classify.punch_extent(flat, tri(200, 60), 200, 0,
                                                       {"lead_widths": 1.33, "tail_widths": 1.83, "frac": 0.5}, GEOM)[:2],
                          GEOM, 100.0)
check("the pipeline's extent always fits the corpus's 25 frames, so nothing is trimmed off one end",
      shown.keypoints is not None and len(shown.keypoints) == 25)
drop = np.ones((400, N_JOINTS)); drop[103:111, 9] = 0.05      # the left wrist lost for 8 frames, 3 after the peak
lost = body(400, 60.0, conf=drop)
w_lost, cut_lost = punchfind.peak_width(lost, tri(100, 7), 100, 0)
check("a wrist that drops out beside the peak censors the width (the signal stopped, not the punch)", cut_lost)
check("... and the other side is still measured: 3.5 frames left, 2.5 right to the dropout",
      abs(w_lost * 60 - 6.0) < 1e-9, f"{w_lost * 60:.3f}")
check("... and the extent carries the flag", punchfind.extent(lost, tri(100, 7), 100, 0, 1.0, 2.0).censored)
flick = np.ones((400, N_JOINTS)); flick[103:105, 9] = 0.05    # a 2-frame flicker: bridged, as the signal bridges it
check("a 2-frame flicker of the wrist is bridged, not treated as the edge",
      not punchfind.peak_width(body(400, 60.0, conf=flick), tri(100, 7), 100, 0)[1])
raises("width: a peak on a frame where the hand has no trustworthy signal is refused",
       lambda: punchfind.peak_width(lost, tri(106, 7), 106, 0))
raises("extent: a non-positive cap is refused", lambda: punchfind.extent(flat, tri(100, 7), 100, 0, 1.0, 2.0, max_s=0.0))

print("punch extent: the same punch at two frame rates")
secs = {}
for f_ in (24.0, 60.0):
    bt = body(int(8 * f_), f_, punches=[3.0], hand=0, dur=0.4)
    sg = punchfind.wrist_signal(bt, "relative")
    pk_ = int(bt.frame_ids[np.argmax(sg[:, 0])])
    e_ = punchfind.extent(bt, sg, pk_, 0, 1.33, 1.83)
    secs[f_] = (e_.end - e_.start) / f_
check("its extent in seconds agrees within 20% at 24 and 60 fps (the width is measured on the sampled curve)",
      abs(secs[24.0] / secs[60.0] - 1) < 0.20, str(secs))

print("one window per clip: the clip's median punch length")
RULE = {"lead_widths": 1.0, "tail_widths": 2.0, "frac": 0.5}
ids800 = np.arange(800)
flat800 = body(800, 60.0)
FIVE = (100, 230, 360, 490, 620)


def clip_sig(centres_halfwidths, frames=ids800):
    sg = np.zeros((len(frames), 2))
    for c_, hw in centres_halfwidths:
        sg[:, 0] = np.maximum(sg[:, 0], np.clip(1 - np.abs(frames - c_) / hw, 0, None) * 4.0)
    return sg


def lost(*centres):
    """The left wrist below confidence for 8 frames, starting 3 after each of these peaks."""
    cf_ = np.ones((800, N_JOINTS))
    for c_ in centres:
        cf_[c_ + 3:c_ + 11, 9] = 0.05
    return body(800, 60.0, conf=cf_)


peaks5 = [(c_, 0) for c_ in FIVE]
# Five punches 7, 7, 7, 14 and 14 frames wide at 60 fps: own extents (1 + 2) x width = 21, 21, 21, 42, 42 frames.
five = clip_sig(zip(FIVE, (7, 7, 7, 14, 14)))
cc = classify.clip_cut(flat800, five, peaks5, RULE, GEOM)
check("the clip's length is the median of its punches' own extents: 21 frames, 0.35 s, a third before the peak",
      abs(cc.length_s - 21 / 60) < 1e-9 and abs(cc.lead_share - 1 / 3) < 1e-12 and cc.n_used == 5
      and cc.n_left_out == 0 and not cc.capped and cc.source == "clip", str(cc))
check("every punch is read over that length around its own peak: 7 frames before, 14 after",
      [classify.clip_window(pf, cc, 60.0) for pf in FIVE[:2] + FIVE[3:4]] == [(93, 114), (223, 244), (483, 504)])
check("... including a wide one, whose own extent would have been twice as long",
      classify.punch_extent(flat800, five, 490, 0, RULE, GEOM)[:2] == (476, 518))
slow = classify.clip_cut(flat800, clip_sig(zip(FIVE, (14,) * 5)), peaks5, RULE, GEOM)
check("a clip of slower punches gets a longer window: tempo is still followed, per clip", abs(slow.length_s - 42 / 60) < 1e-9)
six = clip_sig(zip(FIVE + (720,), (7, 7, 7, 14, 14, 14)))
c6 = classify.clip_cut(flat800, six, peaks5 + [(720, 0)], RULE, GEOM)
check("an even number of punches: the median is the mean of the middle two, (21 + 42) / 2 = 31.5 frames",
      abs(c6.length_s - 31.5 / 60) < 1e-9, str(c6))
# One punch (the 14-wide one at 360) loses its wrist after the peak: its extent is unknown.
cl = classify.clip_cut(lost(360), clip_sig(zip(FIVE, (7, 7, 14, 7, 7))), peaks5, RULE, GEOM)
check("a punch whose speed peak is cut off by a wrist dropout is left out of the median, and counted",
      cl.n_used == 4 and cl.n_left_out == 1 and abs(cl.length_s - 21 / 60) < 1e-9, str(cl))
# Three of five cut off: each of those reads 3.5 frames left + 2.5 to the dropout = 6 wide, extent 18 frames.
la = classify.clip_cut(lost(100, 230, 360), clip_sig(zip(FIVE, (7,) * 5)), peaks5, RULE, GEOM)
check("when the whole punches are a minority, all are used: median of 18, 18, 18, 21, 21 frames is 18",
      la.n_used == 5 and la.n_left_out == 0 and abs(la.length_s - 18 / 60) < 1e-9, str(la))
tie = classify.clip_cut(lost(100, 230, 360), clip_sig(zip(FIVE + (720,), (7,) * 6)), peaks5 + [(720, 0)], RULE, GEOM)
check("a tie (3 whole, 3 cut off) also uses all: median of 18, 18, 18, 21, 21, 21 is 19.5 frames",
      tie.n_used == 6 and abs(tie.length_s - 19.5 / 60) < 1e-9, str(tie))
long_c = classify.clip_cut(flat800, clip_sig(zip(FIVE, (60,) * 5)), peaks5, RULE, GEOM)
check("a clip whose punches run longer than the classifier can be shown is capped at 25 training frames, and says so",
      long_c.capped and abs(long_c.length_s - 1.0) < 1e-9, str(long_c))
ws, we = classify.clip_window(360, long_c, 60.0)
check("... and the capped window still holds the peak, a third in, and fits the corpus's 25 frames",
      ws < 360 < we and (we - ws) / 60 <= 1.0 + 1e-9 and abs((360 - ws) / (we - ws) - 1 / 3) < 0.02
      and len(classify.cut_span(flat800, ws, we, GEOM, 100.0).keypoints) == 25)
few = classify.clip_cut(flat800, five, peaks5[:4], RULE, GEOM)
# length_s is first frame to last: the training window's 9 frames are 8 steps of 1/24 s, not GEOM.duration_s
# (9 / 24 s, which counts samples); its peak is 3 of those 8 steps in. Which frames this reads is checked below.
check("under 5 punches there is no tempo to read: the training clips' median window (9 frames: 8/24 s from first "
      "to last, the peak 3/8 in), and it says so",
      few.source == "training" and abs(few.length_s - 8 / 24) < 1e-12 and abs(few.lead_share - 3 / 8) < 1e-12
      and few.n_used == 0 and classify.MIN_CLIP_PUNCHES == 5, str(few))
raises("a geometry whose peak would fall after the window's last frame is refused (lead-in = duration, 9/24 s)",
       lambda: replace(GEOM, lead_in_s=GEOM.duration_s))
late = classify.clip_cut(flat800, five, peaks5[:4], RULE, replace(GEOM, lead_in_s=8 / 24))
check("... a peak ON the last frame is allowed, and the fallback window ends there",
      late.lead_share == 1.0
      and classify.read_clip_window(body(400, 24.0), 50, late, replace(GEOM, lead_in_s=8 / 24), 100.0)[1] == (42, 50),
      str(late))
check("no punch, no cut", classify.clip_cut(flat800, five, [], RULE, GEOM) is None)
raises("a peak whose hand has no speed there is refused, not skipped (the list was not made from this pose)",
       lambda: classify.clip_cut(flat800, five, peaks5[:4] + [(620, 1)], RULE, GEOM))
raises("a fractional peak frame is refused by the window", lambda: classify.clip_window(60.5, cc, 60.0))
raises("a non-positive frame rate is refused by the window", lambda: classify.clip_window(60, cc, 0.0))
check("a window near the start of a clip may begin before frame 0 (the cutter marks those frames empty)",
      classify.clip_window(3, cc, 60.0) == (-4, 17))

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


def coded(f_, n_=400):
    t_ = body(n_, f_, hx=960, hy=600)
    t_.xy[:, 0, 0] += 0.001 * t_.frame_ids
    return t_


# Codex on PR #14: the under-5 fallback read 47-56 (10 frames) at a 24 fps peak at 50, where cut_window reads
# 47-55. Compared here by the frames actually read (each frame's index is coded into the nose's x).
w50, s50 = classify.read_clip_window(c24, 50, few, GEOM, 100.0)
check("the under-5 fallback reads cut_window's frames: 47-55 at a 24 fps peak at 50, not 47-56, and says so",
      np.array_equal(frames_of(w50), np.arange(47, 56)) and s50 == (47, 55)
      and np.array_equal(frames_of(classify.cut_window(c24, 50, GEOM, 100.0)), np.arange(47, 56)),
      f"{frames_of(w50)} {s50}")
# Phone footage is not at 24 fps. Through clip_window + cut_span a correct 8-step span reads 8 frames at 25,
# 29.97 and 59.94 fps (its start snaps to a frame; only whole steps fit), so the rate matters to this check.
bad_rates = []
for f_ in (23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0):
    tf_ = coded(f_, int(16 * f_))
    for p_ in range(int(0.5 * f_), int(15 * f_), 7):
        w_, s_ = classify.read_clip_window(tf_, p_, few, GEOM, 100.0)
        got, ref = frames_of(w_), frames_of(classify.cut_window(tf_, p_, GEOM, 100.0))
        if not (len(got) == GEOM.n_frames and np.array_equal(got, ref) and s_ == (got[0], got[-1])
                and s_[0] <= p_ <= s_[1]):
            bad_rates.append((f_, p_, got, ref, s_))
check("... and at 23.976, 24, 25, 29.97, 30, 59.94 and 60 fps, at every peak tried: cut_window's 9 frames, "
      "frame for frame, with the peak inside and the span its first and last", not bad_rates, str(bad_rates[:2]))
via_span = classify.cut_span(c24, *classify.clip_window(50, few, 24.0), GEOM, 100.0)
check("control: the same 24 fps fallback read through clip_window + cut_span holds 9 frames (the length is right; "
      "the phone-rate check above is what needs cut_window)", len(frames_of(via_span)) == 9)
w_cc, s_cc = classify.read_clip_window(coded60, 200, cc, GEOM, 100.0)
check("a clip's own cut is read exactly as gated: clip_window, then cut_span",
      s_cc == classify.clip_window(200, cc, 60.0)
      and np.array_equal(w_cc.keypoints, classify.cut_span(coded60, *s_cc, GEOM, 100.0).keypoints))
raises("a cut from an unknown source is refused, not read as one or the other",
       lambda: classify.read_clip_window(coded60, 200, cc._replace(source="per punch"), GEOM, 100.0))
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
check("control: the clip's own speed peak is where it was cut to be, 5 frames in", pk0 == 5)
longer = []
for q in pk:   # the same punches with 2 more frames before the peak: 7 before, 8 after
    kp = np.zeros((25, N_JOINTS, 3), np.float32)
    kp[:16, :, 0] = cont.xy[q - 7:q + 9, :, 0] / w
    kp[:16, :, 1] = cont.xy[q - 7:q + 9, :, 1] / h
    kp[:16, :, 2] = 1
    longer.append(Clip(kp, 1, "", "V1"))
er2 = classify.training_extent_ratios(longer, "relative", 0.5)
# Worked from the clips, not from the code: the peak's width is the same punch's in both sets, so the ratios
# must scale with the frames planted before and after it: lead 7/5, tail unchanged.
check("lead is the time before the peak in peak-widths: 7 frames before against 5 gives 7/5 the lead, the same tail",
      abs(er2["lead_widths"] / er["lead_widths"] - 7 / 5) < 1e-6 and abs(er2["tail_widths"] - er["tail_widths"]) < 1e-9,
      f"{er} {er2}")
check("the report of what was kept: every clip measurable, none skipped, durations 14/24 s",
      er["n_skipped"] == 0 and abs(er["duration_s_median_kept"] - 14 / 24) < 1e-9
      and er["censored_by_class"]["jab"] == {"censored": 0, "of": len(whole)}, str(er))
moving = body(14, 24.0, hx=960, hy=600).xy      # the right wrist already at full speed in the clip's first frame,
moving[:5, 10, 0] += np.linspace(0, 200, 5)     # then still: its speed peak was cut off by the start of the clip
moving[5:, 10, 0] += 200
kp_cut = np.zeros((25, N_JOINTS, 3), np.float32)
kp_cut[:14, :, 0], kp_cut[:14, :, 1], kp_cut[:14, :, 2] = moving[..., 0] / w, moving[..., 1] / h, 1
cut_clips = [Clip(kp_cut, 1, "", "V1") for _ in whole]
mixed = classify.training_extent_ratios(whole + cut_clips, "relative", 0.5)
check("clips whose peak is cut by the clip's first frame are left out and counted, not measured",
      mixed["n_censored"] == len(cut_clips) and mixed["n_clips"] == len(whole)
      and mixed["lead_widths"] == er["lead_widths"]
      and mixed["censored_by_class"]["jab"] == {"censored": len(cut_clips), "of": 2 * len(whole)}, str(mixed))
raises("too few whole clips is refused", lambda: classify.training_extent_ratios(whole[:3], "relative", 0.5))
raises("an unknown variant is refused", lambda: classify.training_extent_ratios(whole, "sideways", 0.5))

print("the extent rule file")
extent_file = tmp / "punch_extent.json"


def extent_json(**kw):
    d = {"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8, "variant": "relative", "beats_fixed_window": True,
         "gated_cut": classify.GATED_CUT, **kw}
    extent_file.write_text(json.dumps({k: v for k, v in d.items() if v is not None}), encoding="utf-8")
    return lambda: classify.read_extent(extent_file, "relative")


check("a good rule file is read", extent_json()() == {"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8})
raises("a rule that did not beat the fixed window is refused", extent_json(beats_fixed_window=False))
raises("a rule file from before the per-clip cut (the flag, but no gated cut) is refused", extent_json(gated_cut=None))
raises("a rule file that gated another cut is refused", extent_json(gated_cut="per punch, at labelled peaks"))
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
# Left hand: a punch at 3 s and a SLOW one at 9 s (twice as long). Right hand: a punch at 14 s.
xy = body(n, fps, punches=[3.0], hand=0).xy.copy()
xy[:, 9] += body(n, fps, punches=[9.0], hand=0, dur=0.5).xy[:, 9] - body(n, fps).xy[:, 9]
xy[:, 10] = body(n, fps, punches=[14.0], hand=1).xy[:, 10]
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
length = [int(r["punch_end_frame"]) - int(r["punch_start_frame"]) for r in rows]
check("three punches are too few to read a tempo: each is read over the training window, 8/24 s first to last "
      "(20 frames at 60 fps)", all(abs(k_ - 8 / 24 * fps) <= 1 for k_ in length), str(length))
check("orthodox: left-hand punches are lead, right-hand punches rear",
      [r["lead_rear"] for r in rows] == ["lead", "lead", "rear"])
check("hands_agree says whether the type's hand is the finder's",
      all((r["hands_agree"] == "yes") == (r["type_hand"] == r["lead_rear"]) for r in rows))
prov = json.loads((tmp / "o" / "IMG_7777.classified.json").read_text(encoding="utf-8"))
check("the sidecar names the classifier, the extent rule and the stance, and hashes the CSV beside it",
      prov["classifier_sha256"] == man["sha256"] and prov["stance"] == "orthodox" and prov["pose_frames"] == n
      and prov["extent"] == {"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8}
      and prov["csv_sha256"] == hashlib.sha256((tmp / "o" / "IMG_7777.classified.csv").read_bytes()).hexdigest()
      and prov["cut"]["per"] == "clip" and prov["cut"]["source"] == "training" and prov["cut"]["n_used"] == 0
      and abs(prov["cut"]["length_s"] - 8 / 24) < 1e-12, str(prov))
classify_punches.main(args + ["--stance", "southpaw"])
rows_s = list(csv.DictReader((tmp / "o" / "IMG_7777.classified.csv").open(encoding="utf-8")))
check("southpaw flips lead and rear, and nothing else",
      [r["lead_rear"] for r in rows_s] == ["rear", "rear", "lead"]
      and [r["punch_type"] for r in rows_s] == [r["punch_type"] for r in rows])
raises("no stance -> refused (the tool does not guess one)",
       lambda: classify_punches.main(args), SystemExit)

# The same clip with the hips unsure around the first punch: that punch cannot be centred, the others can.
cf8 = cf.copy(); cf8[120:300, 11] = 0.0   # the left hip not detected at all there
pose_yolo.PoseSequence(xy.astype(np.float32), cf8, np.ones(n, bool), fps, 1080, 1920, "IMG_8888.mov", "test").save(
    tmp / "IMG_8888.pose.npz")
find_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o")])
args8 = ["--pose", str(tmp / "IMG_8888.pose.npz"), "--punches", str(tmp / "o" / "IMG_8888.punches.csv"),
         "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file), "--stance", "orthodox",
         "--out-dir", str(tmp / "o")]
classify_punches.main(args8)
rows_e = list(csv.DictReader((tmp / "o" / "IMG_8888.classified.csv").open(encoding="utf-8")))
check("a punch with no hips in view is kept as unclassified, with the reason, and its frames",
      rows_e[0]["punch_type"] == "unclassified" and rows_e[0]["confidence"] == "" and "hips" in rows_e[0]["note"]
      and rows_e[0]["punch_start_frame"] != "", str(rows_e[0]))
check("... and the others are still classified", all(r["punch_type"] in CLASSES for r in rows_e[1:]) and len(rows_e) == 3)
raises("a punches CSV named for another clip is refused",
       lambda: classify_punches.main(args8[:2] + ["--punches", str(punches)] + args8[4:]))
stray = punches.read_text(encoding="utf-8") + f"600,{600 / fps:.3f},left,9.000,no\n"
(tmp / "o" / "IMG_8888.punches.csv").write_text(stray, encoding="utf-8")
raises("a CSV row that is not a speed peak of that hand in this pose is refused, not classified",
       lambda: classify_punches.main(args8))
find_punches.main(["--pose", str(tmp / "IMG_8888.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o")])
# A row above the threshold but on the rising slope, not at the peak: the per-hand finder could never report it.
p8 = tmp / "o" / "IMG_8888.punches.csv"
good8 = p8.read_text(encoding="utf-8")
r8 = list(csv.DictReader(p8.open(encoding="utf-8")))[1]
tr8 = punchfind.bridge(pose_yolo.PoseSequence.load(tmp / "IMG_8888.pose.npz").to_track(usable_only=False), 3)
sg8 = punchfind.wrist_signal(tr8, "relative")
h8, k8 = ("left", "right").index(r8["hand"]), int(np.flatnonzero(tr8.frame_ids == int(r8["frame"]))[0])
slope = [k for k in range(k8 - 10, k8) if sg8[k, h8] >= 1.0 and sg8[k + 1, h8] > sg8[k, h8]]
check("control: the second punch's speed is above the threshold and still rising before its peak", bool(slope))
f8 = int(tr8.frame_ids[slope[-1]])
lines8 = good8.splitlines()
lines8[2] = f"{f8},{f8 / fps:.3f},{r8['hand']},{r8['peak_speed_bl_per_s']},{r8['peak_on_interpolated_frame']}"
p8.write_text("\n".join(lines8) + "\n", encoding="utf-8")
raises("a CSV row above the threshold but on the rising slope (not a local peak of that hand) is refused",
       lambda: classify_punches.main(args8))
p8.write_text(good8, encoding="utf-8")
check("control: the untouched CSV is accepted", classify_punches.main(args8) == 0)
radial = tmp / "radial.json"
radial.write_text(json.dumps({"chosen": {"variant": "radial", "threshold": 1.0, "min_gap_s": 0.4}}), encoding="utf-8")
raises("a finder rule with another wrist-speed variant than the window's is refused",
       lambda: classify_punches.main(args + ["--rule", str(radial), "--stance", "orthodox"]))
extent_json(beats_fixed_window=False)
raises("an extent rule that did not beat the fixed window is refused by the CLI",
       lambda: classify_punches.main(args + ["--stance", "orthodox"]))
extent_json()

# Six punches: five quick ones and a slow one (twice as long). Enough to read the clip's tempo.
xy6 = body(n, fps, punches=[2.0, 5.0, 8.0], hand=0).xy.copy()
xy6[:, 9] += body(n, fps, punches=[11.0], hand=0, dur=0.5).xy[:, 9] - body(n, fps).xy[:, 9]
xy6[:, 10] = body(n, fps, punches=[14.0, 17.0], hand=1).xy[:, 10]


def run6(name, conf6):
    pose_yolo.PoseSequence(xy6.astype(np.float32), conf6, np.ones(n, bool), fps, 1080, 1920, name + ".mov", "test").save(
        tmp / f"{name}.pose.npz")
    find_punches.main(["--pose", str(tmp / f"{name}.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o")])
    classify_punches.main(["--pose", str(tmp / f"{name}.pose.npz"), "--punches", str(tmp / "o" / f"{name}.punches.csv"),
                           "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file),
                           "--stance", "orthodox", "--out-dir", str(tmp / "o")])
    return (list(csv.DictReader((tmp / "o" / f"{name}.classified.csv").open(encoding="utf-8"))),
            json.loads((tmp / "o" / f"{name}.classified.json").read_text(encoding="utf-8"))["cut"])


rows_6, cut6 = run6("IMG_6000", cf)
len6 = [int(r["punch_end_frame"]) - int(r["punch_start_frame"]) for r in rows_6]
check("control: the finder finds all six planted punches", len(rows_6) == 6, str(len(rows_6)))
check("six punches: every one is read over the same length, the slow one included",
      max(len6) - min(len6) <= 1 and cut6["source"] == "clip" and cut6["n_used"] == 6 and cut6["n_left_out"] == 0,
      f"{len6} {cut6}")
tr6 = punchfind.bridge(pose_yolo.PoseSequence.load(tmp / "IMG_6000.pose.npz").to_track(usable_only=False), 3)
sg6 = punchfind.wrist_signal(tr6, "relative")
own6 = sorted(punchfind.extent(tr6, sg6, int(r["frame"]), 0 if r["hand"] == "left" else 1, 1.3, 1.8).end
              - punchfind.extent(tr6, sg6, int(r["frame"]), 0 if r["hand"] == "left" else 1, 1.3, 1.8).start for r in rows_6)
check("... and that length is the quick punches' (the median), not the slow one's own, which is over 1.6 times longer",
      abs(len6[0] - (own6[2] + own6[3]) / 2) <= 1.5 and own6[5] > 1.6 * own6[2], f"{len6} vs own {own6}")
# The slow punch's wrist drops out right after its peak: its own extent is unknown, so it is left out of the
# clip's median, and it is still read over the clip's length like every other punch.
cf9 = cf.copy(); cf9[int(11.0 * fps) + 8:int(11.0 * fps) + 40, 9] = 0.05
rows_9, cut9 = run6("IMG_9999", cf9)
check("a punch whose wrist drops out beside the peak is left out of the clip's median, and the sidecar counts it",
      cut9["n_left_out"] == 1 and cut9["n_used"] == 5 and len(rows_9) == 6, f"{cut9} {len(rows_9)}")
check("... and every punch is still named, over the clip's one length, with no note",
      all(r["punch_type"] in CLASSES and r["note"] == "" for r in rows_9)
      and max(int(r["punch_end_frame"]) - int(r["punch_start_frame"]) for r in rows_9)
      - min(int(r["punch_end_frame"]) - int(r["punch_start_frame"]) for r in rows_9) <= 1,
      str([(r["frame"], r["punch_type"], r["note"]) for r in rows_9]))

# A clip in which the finder found nothing: an empty classified CSV and a sidecar that says there was no cut.
empty_dir = tmp / "empty"
pose_yolo.PoseSequence(body(n, fps).xy.astype(np.float32), cf, np.ones(n, bool), fps, 1080, 1920, "IMG_0000.mov",
                       "test").save(tmp / "IMG_0000.pose.npz")
(tmp / "IMG_0000.punches.csv").write_text(",".join(classify_punches.PUNCH_HEADER) + "\n", encoding="utf-8")
args0 = ["--pose", str(tmp / "IMG_0000.pose.npz"), "--punches", str(tmp / "IMG_0000.punches.csv"),
         "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file), "--stance", "orthodox",
         "--out-dir", str(empty_dir)]
rc0 = classify_punches.main(args0)
check("no punches: a header-only CSV and a sidecar whose cut is null",
      rc0 == 0 and (empty_dir / "IMG_0000.classified.csv").read_text(encoding="utf-8").strip()
      == ",".join(classify_punches.OUT_HEADER)
      and json.loads((empty_dir / "IMG_0000.classified.json").read_text(encoding="utf-8"))["cut"] is None)
lab0 = tmp / "IMG_0000.labels.json"
lab0.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [{"start": 100, "end": 120, "label": "jab"}]}),
                encoding="utf-8")
fresh = tmp / "empty2"
raises("no punches and labels to score: refused before anything is written",
       lambda: classify_punches.main(args0[:-1] + [str(fresh), "--labels", str(lab0)]))
check("... nothing was written", not fresh.exists() or not any(fresh.iterdir()))

# The other hand drifting slowly while the left one punches: a CSV that names the WRONG hand at that frame
# points at a hand with some speed there, but not its peak. It must be refused, not measured.
xy_d = xy.copy()
xy_d[150:260, 10, 1] += np.arange(110) * 0.8
pose_yolo.PoseSequence(xy_d.astype(np.float32), cf, np.ones(n, bool), fps, 1080, 1920, "IMG_5555.mov", "test").save(
    tmp / "IMG_5555.pose.npz")
find_punches.main(["--pose", str(tmp / "IMG_5555.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "o")])
p5 = tmp / "o" / "IMG_5555.punches.csv"
lines5 = p5.read_text(encoding="utf-8").splitlines()
check("control: the finder's first punch is the left hand's", lines5[1].split(",")[2] == "left", lines5[1])
p5.write_text("\n".join([lines5[0], lines5[1].replace(",left,", ",right,")] + lines5[2:]) + "\n", encoding="utf-8")
raises("a CSV naming the other hand at a punch's frame is refused (it has speed there, but it is not its peak)",
       lambda: classify_punches.main(["--pose", str(tmp / "IMG_5555.pose.npz"), "--punches", str(p5),
                                      "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file),
                                      "--stance", "orthodox", "--out-dir", str(tmp / "o5")]))
check("... and nothing was written for it", not (tmp / "o5" / "IMG_5555.classified.csv").exists())

# Rows the finder could not have written (Codex on PR #14). Each CSV below is IMG_7777's own, changed in one way.
tr7 = punchfind.bridge(pose_yolo.PoseSequence.load(tmp / "IMG_7777.pose.npz").to_track(usable_only=False),
                       int(round(punchfind.BRIDGE_S * fps)))
sig7 = punchfind.wrist_signal(tr7, "relative")
best7 = sig7.max(axis=1)
lines7 = punches.read_text(encoding="utf-8").splitlines()
f1, hand1 = int(lines7[1].split(",")[0]), lines7[1].split(",")[2]
(tmp / "bad").mkdir(exist_ok=True)


def classify_csv(body_lines, rule_path=rule, out="obad"):
    p = tmp / "bad" / "IMG_7777.punches.csv"
    p.write_text("\n".join([lines7[0]] + body_lines) + "\n", encoding="utf-8")
    return lambda: classify_punches.main(["--pose", str(tmp / "IMG_7777.pose.npz"), "--punches", str(p),
                                          "--classifier", str(mpath), "--rule", str(rule_path),
                                          "--extent", str(extent_file), "--stance", "orthodox",
                                          "--out-dir", str(tmp / out)])


def rule_at(threshold):
    p = tmp / "bad" / "rule_t.json"
    p.write_text(json.dumps({"chosen": {"variant": "relative", "threshold": float(threshold), "min_gap_s": 0.4}}),
                 encoding="utf-8")
    return p


def row7(fr, hand):
    return f"{fr},{fr / fps:.3f},{hand},1.000,no"


def accepted(fn):   # a control that must run: a refusal is a FAIL with its reason, not a crash of the suite
    try:
        return fn() == 0
    except ValueError as e:
        print(f"        refused: {e}")
        return False


check("control: the finder's own CSV, rewritten by the helper, is accepted",
      accepted(classify_csv(lines7[1:], out="ogood")) and len(lines7) == 4)
raises("five copies of one punch are refused (they would pass the five-punch gate and set a fake tempo)",
       classify_csv([lines7[1]] * 5))
raises("... and so are two", classify_csv([lines7[1], lines7[1]] + lines7[2:]))
check("... and nothing was written for them", not (tmp / "obad").exists())
check("control: the frames beside the first peak are the same hand's, above the threshold, and slower",
      bool(tr7.frame_ids[f1] == f1 and best7[f1 - 1] >= 1.0 and best7[f1 + 1] >= 1.0
           and best7[f1 - 1] < best7[f1] > best7[f1 + 1]
           and int(np.argmax(sig7[f1 - 1])) == int(np.argmax(sig7[f1 + 1])) == (0 if hand1 == "left" else 1)),
      f"{best7[f1 - 1:f1 + 2]}")
raises("a row one frame up the rising slope of a punch is refused (not a peak)",
       classify_csv([row7(f1 - 1, hand1)] + lines7[2:]))
raises("a row one frame down the falling slope is refused", classify_csv([row7(f1 + 1, hand1)] + lines7[2:]))
slowest = min(float(best7[int(l_.split(",")[0])]) for l_ in lines7[1:])
check("control: a rule whose threshold is exactly the slowest punch's peak speed accepts the CSV (>=, as find_peaks)",
      accepted(classify_csv(lines7[1:], rule_path=rule_at(slowest), out="ogood")))
raises("a row below the rule's threshold is refused (the CSV was not made with this rule)",
       classify_csv(lines7[1:], rule_path=rule_at(np.nextafter(slowest, np.inf))))
# A rule at threshold 0 makes ">= threshold" and "no faster neighbour" true on any still stretch: a row there
# must still be refused, because that hand has no speed at all (find_peaks needs a strict peak).
h7 = 0 if hand1 == "left" else 1
flat = [f_ for f_ in range(5, len(sig7) - 5) if not sig7[f_ - 1:f_ + 2, h7].any()]
check("control: there is a still stretch of that hand, and at threshold 0 the finder's own CSV is accepted",
      bool(flat) and accepted(classify_csv(lines7[1:], rule_path=rule_at(0.0), out="ogood")), str(len(flat)))


def refusal(fn) -> str:
    try:
        fn()
    except ValueError as e:
        return str(e)
    return ""


still_msg = refusal(classify_csv([row7(int(tr7.frame_ids[flat[0]]), hand1)] + lines7[2:], rule_path=rule_at(0.0)))
check("at threshold 0, a row on a still frame of that hand is refused by the peak check (no speed there, so no peak), "
      "not later", f"is not a {hand1}-hand speed peak" in still_msg, still_msg)
check("... and nothing was written for any of them", not (tmp / "obad").exists())

# Labels equal to the frames each punch was read from: the extent's overlap with them must be exactly 1.
lab = tmp / "IMG_7777.labels.json"
spans = [{"start": max(int(r["punch_start_frame"]), 0), "end": int(r["punch_end_frame"]), "label": l}
         for r, l in zip(rows, ("jab", "unclear", "cross"))]
lab.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": spans}), encoding="utf-8")
classify_punches.main(args + ["--stance", "orthodox", "--labels", str(lab)])
sc = json.loads((tmp / "o" / "IMG_7777.classify_score.json").read_text(encoding="utf-8"))
check("scored against labels: 'unclear' punches are not scored for type, in any of the three readings",
      sc["matched_detections"]["n"] == 2 and sc["labelled_punch_by_clip_window_at_its_peak"]["n"] == 2
      and sc["labelled_punch_from_labelled_frames"]["n"] == 2, str(sc))
check("the old ambiguous key is gone: the two perfect-finder readings have their own names",
      "every_labelled_punch" not in sc)
check("the window's overlap with labels that equal it is 1, over all 3 matched punches",
      sc["window_iou_with_labels"] == {"n": 3, "mean": 1.0}, str(sc["window_iou_with_labels"]))
check("the score records the cut it scored", sc["cut"]["per"] == "clip" and sc["cut"]["source"] == "training")
check("the score names the classifier it scored, by hash", sc["classifier_sha256"] == man["sha256"])
check("control: no matched punch is unclassifiable on the whole clip", sc.get("matched_unclassifiable") == 0)
half = [{"start": s_["start"], "end": (s_["start"] + s_["end"]) // 2, "label": s_["label"]} for s_ in spans]
lab.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": half}), encoding="utf-8")
classify_punches.main(args + ["--stance", "orthodox", "--labels", str(lab)])
half_iou = json.loads((tmp / "o" / "IMG_7777.classify_score.json").read_text(encoding="utf-8"))["window_iou_with_labels"]["mean"]
check("... and with labels covering only the first half of each, about a half", 0.45 < half_iou < 0.56, str(half_iou))
lab8 = tmp / "IMG_8888.labels.json"
lab8.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
    {"start": int(3.0 * fps), "end": int(3.25 * fps), "label": "jab"},
    {"start": int(14.0 * fps), "end": int(14.25 * fps), "label": "cross"}]}), encoding="utf-8")
classify_punches.main(args8 + ["--labels", str(lab8)])
sc8 = json.loads((tmp / "o" / "IMG_8888.classify_score.json").read_text(encoding="utf-8"))
check("a matched punch that could not be classified is counted, not silently dropped from the score",
      sc8["matched_unclassifiable"] == 1 and sc8["matched_detections"]["n"] == 1, str(sc8))
check("... and the perfect-finder readings count theirs too",
      sc8["labelled_punch_by_clip_window_unclassifiable"] == 1 and sc8["labelled_punch_from_labelled_frames_unclassifiable"] == 1,
      str(sc8))
# A labelled "punch" where neither wrist moves (19 s: the body is still): there is no speed peak to place the
# clip's window at, so that reading cannot name it; its labelled frames still can be read.
still = [{"start": int(s_ * fps), "end": int((s_ + 0.25) * fps), "label": l_} for s_, l_ in ((3.0, "jab"), (19.0, "cross"))]
lab.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": still}), encoding="utf-8")
classify_punches.main(args + ["--stance", "orthodox", "--labels", str(lab)])
sc_still = json.loads((tmp / "o" / "IMG_7777.classify_score.json").read_text(encoding="utf-8"))
check("a labelled punch with no wrist speed in it is not scored 'at its peak' (it has none), and is counted",
      sc_still["labelled_punch_by_clip_window_unclassifiable"] == 1
      and sc_still["labelled_punch_by_clip_window_at_its_peak"]["n"] == 1
      and sc_still["labelled_punch_from_labelled_frames"]["n"] == 2, str(sc_still))

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
raises("punches: a repeated (frame, hand) row is refused",
       punches_with(hdr + f"600,{600 / fps:.3f},left,1.0,no\n700,{700 / fps:.3f},left,1.0,no\n600,{600 / fps:.3f},left,1.0,no\n"))
check("punches: control: one frame with both hands is two rows, not a repeat (main() checks which hand peaks)",
      len(punches_with(hdr + f"600,{600 / fps:.3f},left,1.0,no\n600,{600 / fps:.3f},right,1.0,no\n")()) == 2)

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

print("per-hand finder rows, point labels and framing marks")
# A jab and a cross 0.25 s apart, then a lone jab: the per-hand finder reports both hands of the combination.
xyc = body(n, fps, punches=[3.0, 8.0], hand=0).xy.copy()
xyc[:, 10] = body(n, fps, punches=[3.25], hand=1).xy[:, 10]
pose_yolo.PoseSequence(xyc.astype(np.float32), cf, np.ones(n, bool), fps, 1080, 1920, "IMG_5050.mov", "test").save(
    tmp / "IMG_5050.pose.npz")
find_punches.main(["--pose", str(tmp / "IMG_5050.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "ph"),
                   "--per-hand"])
ph_rows = list(csv.DictReader((tmp / "ph" / "IMG_5050.punches.csv").open(encoding="utf-8")))
check("control: the per-hand finder reports the combination's two hands", [r["hand"] for r in ph_rows][:2] == ["left", "right"],
      str([(r["frame"], r["hand"]) for r in ph_rows]))
args_ph = ["--pose", str(tmp / "IMG_5050.pose.npz"), "--punches", str(tmp / "ph" / "IMG_5050.punches.csv"),
           "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file), "--stance", "orthodox",
           "--out-dir", str(tmp / "ph")]
check("per-hand rows are accepted: each is a peak of its own hand above the rule's threshold",
      classify_punches.main(args_ph) == 0
      and len(list(csv.DictReader((tmp / "ph" / "IMG_5050.classified.csv").open(encoding="utf-8")))) == len(ph_rows))
# Point labels: a click inside each punch; the lone jab marked 'unclear' (framing).
lab50 = tmp / "IMG_5050.labels.json"
clicks = [(int(r["frame"]) + 5, l_) for r, l_ in zip(ph_rows, ("jab", "cross", "unclear"))]
lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [{"start": f_, "end": f_, "label": l_} for f_, l_ in clicks]}),
                 encoding="utf-8")
classify_punches.main(args_ph + ["--labels", str(lab50), "--unclear", "not-a-punch"])
s50 = json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))
check("point labels: the finder found 2 of the 2 punches, and 2 of its 3 detections were real",
      s50["finder"] == {"per_hand": True, "labelled_punches": 2, "detections": 3, "matched": 2, "recall": 1.0,
                        "precision": 2 / 3}
      and s50["point_labels"] == 2 and s50["unclear_means"] == "not-a-punch", str(s50["finder"]))
check("the framing mark that drew a detection is reported, with what the classifier called it",
      s50["framing"]["marks"] == 1 and s50["framing"]["flagged_as_punches_at_least"] == 1
      and s50["framing"]["flagged_as_punches_at_most"] == 1
      and sum(s50["framing"]["lower_bound_named_as"].values()) == 1
      and set(s50["framing"]["lower_bound_named_as"]) <= set(CLASSES) | {"unclassified"}, str(s50["framing"]))
check("point labels have no start and end, so the 'from labelled frames' reading is empty, and no length is claimed",
      s50["labelled_punch_from_labelled_frames"]["n"] == 0 and s50["labelled_length_s_median"] is None
      and s50["window_iou_with_labels"]["n"] == 0, str(s50))
check("... while the perfect-finder reading uses the speed peak within 0.2 s of each click",
      s50["labelled_punch_by_clip_window_at_its_peak"]["n"] == 2)
# A click on a still frame just before the lone jab, within 0.2 s of its speed peak: the point rule finds the
# peak; reading the click as a one-frame span would find no wrist speed there.
tr50 = punchfind.bridge(pose_yolo.PoseSequence.load(tmp / "IMG_5050.pose.npz").to_track(usable_only=False), 3)
sg50 = punchfind.wrist_signal(tr50, "relative")
jab_peak = int(ph_rows[2]["frame"])
still = [f_ for f_ in range(jab_peak - 11, jab_peak + 12) if sg50[f_].max() == 0]   # the wrist at rest after the punch
check("control: there is a still frame within 0.2 s of the jab's speed peak", bool(still))
still_click = min(still, key=lambda f_: abs(f_ - jab_peak))
lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
    {"start": still_click, "end": still_click, "label": "jab"}]}), encoding="utf-8")
classify_punches.main(args_ph + ["--labels", str(lab50)])
s_still = json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))
check("a click on a still frame within 0.2 s of a punch's peak is read at that peak, not at the click",
      s_still["labelled_punch_by_clip_window_at_its_peak"]["n"] == 1
      and s_still["labelled_punch_by_clip_window_unclassifiable"] == 0, f"click {still_click}, peak {jab_peak}: {s_still}")
lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [{"start": f_, "end": f_, "label": l_} for f_, l_ in clicks]}),
                 encoding="utf-8")
classify_punches.main(args_ph + ["--labels", str(lab50)])
s50p = json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))
check("with 'unclear' as a punch (the default), all 3 are punches and there is no framing report",
      s50p["finder"]["labelled_punches"] == 3 and s50p["finder"]["matched"] == 3 and "framing" not in s50p, str(s50p["finder"]))
# Mixed labels: a drag 15 frames long, a click, and a framing click. The length is the drag's alone.
f0 = int(ph_rows[0]["frame"])
lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
    {"start": f0 - 5, "end": f0 + 10, "label": "jab"}, {"start": clicks[1][0], "end": clicks[1][0], "label": "cross"},
    {"start": clicks[2][0], "end": clicks[2][0], "label": "unclear"}]}), encoding="utf-8")
classify_punches.main(args_ph + ["--labels", str(lab50), "--unclear", "not-a-punch"])
smix = json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))
check("mixed labels: the labelled length is the drags' median (15 frames), the clicks are not zero-length punches",
      smix["labelled_length_s_median"] is not None and abs(smix["labelled_length_s_median"] - 15 / fps) < 1e-9
      and smix["point_labels"] == 1, str(smix["labelled_length_s_median"]))
# Every label a framing mark: nothing to score as a punch, and no crash.
lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
    {"start": f_, "end": f_ + 10, "label": "unclear"} for f_, _ in clicks]}), encoding="utf-8")
check("every label a framing mark: scored without a crash",
      classify_punches.main(args_ph + ["--labels", str(lab50), "--unclear", "not-a-punch"]) == 0)
sall = json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))
check("... with no punch to find, no length claimed, and all 3 marks reported",
      sall["finder"]["labelled_punches"] == 0 and sall["finder"]["recall"] is None
      and sall["labelled_length_s_median"] is None and sall["framing"]["marks"] == 3, str(sall))
# A framing click beside the lone jab's click: the jab takes the one detection there, so the mark is 0 to 1.
f2 = int(ph_rows[2]["frame"])
lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
    {"start": f2 + 2, "end": f2 + 2, "label": "jab"}, {"start": f2 + 4, "end": f2 + 4, "label": "unclear"}]}),
    encoding="utf-8")
classify_punches.main(args_ph + ["--labels", str(lab50), "--unclear", "not-a-punch"])
snear = json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))["framing"]
check("a framing mark whose only detection a punch took is reported as 0 to 1, with nothing named",
      snear["flagged_as_punches_at_least"] == 0 and snear["flagged_as_punches_at_most"] == 1
      and snear["lower_bound_named_as"] == {}, str(snear))
# The jab-cross: both detections are within reach of a click between them, the framing click 8 frames after the
# cross reaches only the cross. The greedy gives the punch the jab, but a punch matching that takes the cross is
# just as large and leaves the mark nothing: 0 to 1, not 1. A second framing click on a still stretch draws nothing.
d0, d1 = int(ph_rows[0]["frame"]), int(ph_rows[1]["frame"])
pc, fc, still_f = (d0 + d1) // 2, d1 + 8, int(12 * fps)
tol50 = int(round(find_punches.TOL_S * fps))
check("control: both detections reach the punch click, only the cross reaches the framing click, the greedy gives the "
      "punch the jab, and nothing is near the still click",
      abs(d0 - pc) <= 12 and abs(d1 - pc) <= 12 and abs(d1 - fc) <= 12 and abs(d0 - fc) > 12
      and punchfind.match([d0, d1], punchfind.for_matching([Span(pc, pc, 0)], tol50, 12), tol50) == [(0, 0)]
      and all(abs(int(r["frame"]) - still_f) > 12 for r in ph_rows), f"{d0} {d1} {pc} {fc}")
lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
    {"start": pc, "end": pc, "label": "jab"}, {"start": fc, "end": fc, "label": "unclear"},
    {"start": still_f, "end": still_f, "label": "unclear"}]}), encoding="utf-8")
classify_punches.main(args_ph + ["--labels", str(lab50), "--unclear", "not-a-punch"])
sx = json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))["framing"]
check("framing bounds hold whichever maximum punch matching is chosen: 0 to 1 of 2, nothing named as certain",
      sx == {"marks": 2, "flagged_as_punches_at_least": 0, "flagged_as_punches_at_most": 1,
             "lower_bound_named_as": {}}, str(sx))

print("per-hand rows where the OTHER hand is faster, and the read sites")
# Both hands punch together, the right at half the left's reach: each hand peaks at the same frames, the left
# faster. The per-hand finder reports the right hand's own peaks too; each is a peak of the right hand, so it
# is accepted although the left is faster there. (A faster-hand rule would refuse every right-hand row.)
xys = body(n, fps, punches=[3.0, 8.0, 13.0], hand=0).xy.copy()
base_r = body(n, fps).xy[:, 10]
xys[:, 10] = base_r + 0.5 * (body(n, fps, punches=[3.0, 8.0, 13.0], hand=1).xy[:, 10] - base_r)
pose_yolo.PoseSequence(xys.astype(np.float32), cf, np.ones(n, bool), fps, 1080, 1920, "IMG_5151.mov", "test").save(
    tmp / "IMG_5151.pose.npz")
find_punches.main(["--pose", str(tmp / "IMG_5151.pose.npz"), "--rule", str(rule), "--out-dir", str(tmp / "sl"),
                   "--per-hand"])
sl_rows = list(csv.DictReader((tmp / "sl" / "IMG_5151.punches.csv").open(encoding="utf-8")))
tr51 = punchfind.bridge(pose_yolo.PoseSequence.load(tmp / "IMG_5151.pose.npz").to_track(usable_only=False), 3)
sg51 = punchfind.wrist_signal(tr51, "relative")
slower = [r for r in sl_rows if r["hand"] == "right"
          and sg51[int(np.flatnonzero(tr51.frame_ids == int(r["frame"]))[0])].argmax() == 0]
check("control: the per-hand finder reports right-hand rows at frames where the left hand is faster",
      len(slower) >= 3, str([(r["frame"], r["hand"]) for r in sl_rows]))
args_sl = ["--pose", str(tmp / "IMG_5151.pose.npz"), "--punches", str(tmp / "sl" / "IMG_5151.punches.csv"),
           "--classifier", str(mpath), "--rule", str(rule), "--extent", str(extent_file), "--stance", "orthodox",
           "--out-dir", str(tmp / "sl")]
check("a per-hand row is accepted when the other hand is faster at its frame (judged on its own hand)",
      accepted(lambda: classify_punches.main(args_sl))
      and len(list(csv.DictReader((tmp / "sl" / "IMG_5151.classified.csv").open(encoding="utf-8")))) == len(sl_rows))

# Both pipeline read sites go through classify.read_clip_window: on a clip with fewer than 5 punches the
# cut is the training window, which only read_clip_window reads with cut_window's own frames (#14).
calls, real_read = [], classify.read_clip_window


def spy(track_, peak, cut_, *rest, **kw):
    calls.append((int(peak), cut_.source))
    return real_read(track_, peak, cut_, *rest, **kw)


classify.read_clip_window = spy
try:
    classify_punches.main(args_ph)
    no_labels = list(calls)
    calls.clear()
    lab50.write_text(json.dumps({"n_frames": n, "fps": fps, "spans": [
        {"start": f_, "end": f_, "label": l_} for f_, l_ in clicks]}), encoding="utf-8")
    classify_punches.main(args_ph + ["--labels", str(lab50)])
    with_labels = list(calls)
finally:
    classify.read_clip_window = real_read
check("control: this clip has fewer than 5 punches, so its cut is the training window",
      len(ph_rows) < classify.MIN_CLIP_PUNCHES and {s_ for _, s_ in no_labels} == {"training"}, str(no_labels))
check("the pipeline reads every finder row through read_clip_window, at the row's own frame",
      sorted(f_ for f_, _ in no_labels) == sorted(int(r["frame"]) for r in ph_rows), str(no_labels))
check("... and the perfect-finder reading reads each named labelled punch through it too (3 rows + 2 named)",
      len(with_labels) == len(ph_rows) + 2 and {s_ for _, s_ in with_labels} == {"training"}, str(with_labels))

print("which finder made the punch list: recorded from the sidecar find_punches.py writes")
prov50 = json.loads((tmp / "ph" / "IMG_5050.classified.json").read_text(encoding="utf-8"))
check("a per-hand list is recorded as per-hand, in the provenance and the score",
      prov50["finder_per_hand"] is True and sx is not None
      and json.loads((tmp / "ph" / "IMG_5050.classify_score.json").read_text(encoding="utf-8"))["finder"]["per_hand"] is True)
classify_punches.main(args8)
check("a pooled list is recorded as pooled",
      json.loads((tmp / "o" / "IMG_8888.classified.json").read_text(encoding="utf-8"))["finder_per_hand"] is False)
side50 = tmp / "ph" / "IMG_5050.punches.json"
good_side = side50.read_text(encoding="utf-8")


def per_hand_with(sidecar_text):
    if sidecar_text is None:
        side50.unlink()
    else:
        side50.write_text(sidecar_text, encoding="utf-8")
    classify_punches.main(args_ph)
    side50.write_text(good_side, encoding="utf-8")
    return json.loads((tmp / "ph" / "IMG_5050.classified.json").read_text(encoding="utf-8"))["finder_per_hand"]


gs = json.loads(good_side)
check("no sidecar: unknown (null), not a guess", per_hand_with(None) is None)
check("a sidecar hashed to other bytes than the CSV holds: unknown",
      per_hand_with(json.dumps(gs | {"csv_sha256": "0" * 64})) is None)
check("a sidecar for another clip: unknown", per_hand_with(json.dumps(gs | {"clip": "IMG_8888"})) is None)
check("a sidecar whose per_hand is not a boolean: unknown", per_hand_with(json.dumps(gs | {"per_hand": "yes"})) is None)
check("a sidecar for another rule than this run's: unknown",
      per_hand_with(json.dumps(gs | {"rule": gs["rule"] | {"threshold": gs["rule"]["threshold"] + 0.5}})) is None)
check("a sidecar that is not JSON: unknown, not a crash", per_hand_with("{not json") is None)
deep = "[" * 30000 + "]" * 30000
check("control: the deeply nested sidecar is under the size cap", len(deep.encode()) < 64_000)
raises("control: ... and json itself cannot parse it", lambda: json.loads(deep), RecursionError)
check("a deeply nested sidecar (RecursionError in json): unknown, not a crash", per_hand_with(deep) is None)
check("control: the untouched sidecar is read", per_hand_with(good_side) is True)

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
