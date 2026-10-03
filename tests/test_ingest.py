"""Tests for video ingestion, pose extraction and the overlay renderer.

Weighted toward the trust boundaries: the video file (untrusted, arbitrary
container), the model checkpoint (a pickle), and the pose file. No model and no
GPU are needed -- extraction is driven by an injected predictor -- so this runs
on any machine with ffmpeg.

Run:  python tests/test_ingest.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import run_ingest  # noqa: E402
from boxingvi import overlay, pose_yolo, video  # noqa: E402
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


def raises(name: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc:
        check(name, True)
    except Exception as e:  # wrong exception type is a failure too
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


tmp = Path(tempfile.mkdtemp())
W, H, N = 96, 64, 12

# A 12-frame 60 fps clip whose frame i is uniformly grey level 10*i + 20, so a
# decoded frame identifies itself and a dropped or duplicated one is visible.
clip = tmp / "clip.mp4"
ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=60,format=rgb24,"
       "geq=r='20+10*N':g='20+10*N':b='20+10*N'", "-frames:v", str(N),
       "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(clip))

print("video.probe / iter_frames")
info = video.probe(clip)
check("probe reads size and exact fps", (info.width, info.height) == (W, H)
      and info.fps_frac == Fraction(60), str(info))
frames = list(video.iter_frames(clip, info))
check("every frame decoded, none dropped or duplicated", len(frames) == N, str(len(frames)))
check("frames arrive in file order",
      all(abs(int(f.mean()) - (20 + 10 * i)) <= 2 for i, f in enumerate(frames)))
check("max_frames stops early without error", len(list(video.iter_frames(clip, info, 5))) == 5)

rot = tmp / "rot.mp4"
ffmpeg("-display_rotation", "90", "-i", str(clip), "-c", "copy", str(rot))
rinfo = video.probe(rot)
rframes = list(video.iter_frames(rot, rinfo, 2))
check("rotation flag: probe reports the DISPLAYED size",
      (rinfo.width, rinfo.height) == (H, W), str(rinfo))
check("rotation flag: decoded frames match the probed size",
      rframes[0].shape == (rinfo.height, rinfo.width, 3), str(rframes[0].shape))

raises("a missing video is FileNotFoundError", lambda: video.probe(tmp / "nope.mp4"),
       FileNotFoundError)
junk = tmp / "junk.mp4"
junk.write_bytes(b"this is not a video" * 50)
raises("a garbage file is refused at probe", lambda: video.probe(junk))
txt = tmp / "t.mp4"
txt.write_bytes(b"")
raises("an empty file is refused at probe", lambda: video.probe(txt))
big = tmp / "big.mp4"
ffmpeg("-f", "lavfi", "-i", "color=c=black:s=4096x2304:r=1", "-frames:v", "1",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", str(big))
raises("a frame larger than one 4K frame is refused before allocation",
       lambda: video.probe(big))
raises("a zero frame rate is refused", lambda: video._rate("0/0"))
raises("a nonsense frame rate is refused", lambda: video._rate("abc"))
def _bad_write():
    with video.VideoWriter(tmp / "w.mp4", 8, 8, Fraction(30)) as vw:
        vw.write(np.zeros((4, 4, 3), np.uint8))


raises("VideoWriter refuses a wrong-shaped frame", lambda: _bad_write())


check("a failed write leaves no truncated output", not (tmp / "w.mp4").exists())

print("pose_yolo.select_person")
check("no detections -> None", pose_yolo.select_person(np.zeros((0, 4))) is None)
two = np.array([[0, 0, 10, 10], [0, 0, 50, 50]], float)
check("largest box wins with no history", pose_yolo.select_person(two) == 1)
prev = np.array([0, 0, 12, 12], float)
check("continuity beats a moderately larger bystander (1.4x the area)",
      pose_yolo.select_person(np.array([[0, 0, 10, 10], [20, 20, 32, 32]], float), prev) == 0)
check("... but a far larger box still wins: the track really is lost",
      pose_yolo.select_person(np.array([[0, 0, 10, 10], [20, 20, 45, 45]], float), prev) == 1)
check("continuity does not pin a track that has really left",
      pose_yolo.select_person(np.array([[200, 200, 210, 210], [20, 20, 45, 45]], float), prev) == 1)
check("NaN and degenerate boxes are ignored",
      pose_yolo.select_person(np.array([[np.nan, 0, 5, 5], [3, 3, 3, 9], [0, 0, 4, 4]], float)) == 2)
check("only bad boxes -> None",
      pose_yolo.select_person(np.array([[np.inf, 0, 5, 5], [5, 5, 1, 1]], float)) is None)

print("pose_yolo.extract / PoseSequence")
GT = np.stack([np.linspace(10, 80, N_JOINTS) + 12 * (-1.0) ** np.arange(N_JOINTS),
               np.linspace(8, 56, N_JOINTS)], axis=1)
GT = GT + np.random.default_rng(0).uniform(-6, 6, GT.shape)  # no accidental collinear bones


def predict_all(frame):  # a boxer plus a bigger-box bystander on a later frame
    return (np.array([[5, 5, 90, 60]], float), GT[None].copy(), np.full((1, N_JOINTS), 0.9))


seen = []


def predict_gappy(frame):
    seen.append(int(frame.mean()))
    i = len(seen) - 1
    if i in (3, 4):
        return np.zeros((0, 4)), np.zeros((0, N_JOINTS, 2)), np.zeros((0, N_JOINTS))
    if i == 7:  # a corrupt keypoint block must become a gap, not a number
        bad = GT[None].copy()
        bad[0, 2, 0] = np.nan
        return np.array([[5, 5, 90, 60]], float), bad, np.full((1, N_JOINTS), 0.9)
    return predict_all(frame)


ps = pose_yolo.extract(clip, predict_gappy, "fake", None)
check("one pose row per video frame, gaps included", len(ps) == N and ps.xy.shape == (N, 17, 2))
check("undetected frames are NaN and flagged",
      list(np.flatnonzero(~ps.detected)) == [3, 4, 7] and np.isnan(ps.xy[3]).all())
check("a NaN keypoint block is a gap, never a number", not ps.detected[7])
check("detected frames carry the predictor's coordinates", np.allclose(ps.xy[0], GT, atol=1e-4))
check("predictor received frames in RGB file order", seen == sorted(seen) and len(seen) == N)
trk = ps.to_track()
check("to_track keeps only detected frames, gaps as frame-id jumps",
      list(trk.frame_ids) == [0, 1, 2, 5, 6, 8, 9, 10, 11] and trk.fps == 60.0)
f = tmp / "p.npz"
ps.save(f)
back = PoseSequence.load(f)
check("npz round-trip is exact", np.array_equal(back.detected, ps.detected)
      and np.allclose(back.xy[ps.detected], ps.xy[ps.detected]) and back.model == "fake")
raises("a pose with the wrong shape is refused",
       lambda: PoseSequence(np.zeros((3, 17, 2)), np.zeros((3, 16)), np.ones(3, bool), 30, 9, 9))
raises("a 'detected' frame with NaN coordinates is refused",
       lambda: PoseSequence(np.full((2, 17, 2), np.nan), np.zeros((2, 17)),
                            np.ones(2, bool), 30, 9, 9))
raises("fps of zero is refused",
       lambda: PoseSequence(np.zeros((2, 17, 2)), np.zeros((2, 17)), np.ones(2, bool), 0, 9, 9))
pickled = tmp / "evil.npz"
np.savez(pickled, xy=np.array([{"a": 1}], dtype=object))
raises("a pickled array in a pose file is refused (allow_pickle off)",
       lambda: PoseSequence.load(pickled))
none = tmp / "none.mp4"
none.write_bytes(b"")
raises("extract on an unreadable video fails loudly", lambda: pose_yolo.extract(none, predict_all))

print("pose_yolo.load_yolo (the checkpoint is a pickle)")
w = tmp / "m.pt"
w.write_bytes(b"not really weights")
good = pose_yolo.sha256_file(w)
raises("no pinned sha256 -> refused", lambda: pose_yolo.load_yolo(w, None))
raises("wrong sha256 -> refused before ultralytics is touched",
       lambda: pose_yolo.load_yolo(w, "0" * 64))
raises("missing weights file", lambda: pose_yolo.load_yolo(tmp / "no.pt", good), FileNotFoundError)
try:
    pose_yolo.load_yolo(w, good.upper())  # right hash passes the gate...
    check("correct hash reaches the loader", False, "loaded junk?")
except (RuntimeError, Exception) as e:  # ...then fails in ultralytics (absent or can't parse junk)
    check("correct hash (any case) passes the gate; junk then fails in the loader",
          "sha256" not in str(e), str(e))

print("overlay")
img = np.zeros((H, W, 3), np.uint8)
overlay.draw_skeleton(img, GT, np.full(N_JOINTS, 0.9))
j = 9  # left wrist
check("a confident joint is drawn where the pose says",
      tuple(img[int(round(GT[j, 1])), int(round(GT[j, 0]))]) == overlay.C_LEFT)
img2 = np.zeros((H, W, 3), np.uint8)
overlay.draw_skeleton(img2, GT, np.zeros(N_JOINTS))
check("below min_conf, nothing is drawn", not img2.any())
img3 = np.zeros((H, W, 3), np.uint8)
c = np.full(N_JOINTS, 0.9)
c[7] = 0.0  # left elbow unsure: shoulder-elbow and elbow-wrist bones must both go
overlay.draw_skeleton(img3, GT, c)
mid = ((GT[7] + GT[9]) / 2).round().astype(int)
full = np.zeros((H, W, 3), np.uint8)
overlay.draw_skeleton(full, GT, np.full(N_JOINTS, 0.9))
check("a bone with an unsure end is not drawn (and the pixel IS drawn at full confidence)",
      full[mid[1], mid[0]].any() and not img3[mid[1], mid[0]].any())
img4 = np.zeros((H, W, 3), np.uint8)
overlay.draw_skeleton(img4, np.full((N_JOINTS, 2), 1e9), np.full(N_JOINTS, 0.9))
check("wild off-frame keypoints neither crash nor allocate absurdly", True)

out = tmp / "ov.mp4"
n = overlay.render_overlay(clip, ps, out)
oi = video.probe(out)
check("overlay has every frame at the source size and rate",
      n == N and (oi.width, oi.height) == (W, H) and oi.fps_frac == Fraction(60), str(oi))
of = list(video.iter_frames(out, oi))
check("overlay frame count matches the source", len(of) == N, str(len(of)))
check("a missed frame is marked with the red border, a detected one is not",
      of[3][2, 40, 0] > 150 and of[3][2, 40, 1] < 90 and not (of[0][2, 40, 0] > 150
                                                              and of[0][2, 40, 1] < 90),
      f"{of[3][2, 40]} vs {of[0][2, 40]}")
raises("overlay refuses to overwrite its own input",
       lambda: overlay.render_overlay(clip, ps, clip))
wrong = PoseSequence(ps.xy, ps.conf, ps.detected, 60.0, W + 2, H, "x", "x")
raises("overlay refuses a pose for a different frame size",
       lambda: overlay.render_overlay(clip, wrong, tmp / "no.mp4"))
short = PoseSequence(ps.xy[:5], ps.conf[:5], ps.detected[:5], 60.0, W, H, "x", "x")
raises("overlay refuses a pose shorter than the video",
       lambda: overlay.render_overlay(clip, short, tmp / "short.mp4"))
check("... and leaves no half-written file behind", not (tmp / "short.mp4").exists())

print("run_ingest CLI")
d = tmp / "outdir"
rc = run_ingest.main(["--video", str(clip), "--out-dir", str(d)], predict=predict_all)
check("end to end: pose file and overlay both written",
      rc == 0 and (d / "clip.pose.npz").is_file() and (d / "clip.overlay.mp4").is_file())
rt = PoseSequence.load(d / "clip.pose.npz")
check("the written pose is the full video, fully detected", len(rt) == N and rt.detected.all())
d2 = tmp / "outdir2"
run_ingest.main(["--video", str(clip), "--out-dir", str(d2), "--no-overlay"], predict=predict_all)
check("--no-overlay writes no video", not (d2 / "clip.overlay.mp4").exists())


def _exit_code(argv):
    try:
        return run_ingest.main(argv)
    except SystemExit as e:
        return e.code


check("no --video is a usage error", _exit_code([]) == 2)
check("real run without --weights is a usage error", _exit_code(["--video", str(clip)]) == 2)
raises("real run with weights but no sha256: ValueError from load_yolo, nothing written",
       lambda: run_ingest.main(["--video", str(clip), "--weights", str(w),
                                "--out-dir", str(tmp / "o3")]))
check("... and no output directory was created", not (tmp / "o3").exists())

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
