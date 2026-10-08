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

print("video timing (VFR is measured, not inferred)")
check("a constant-rate clip is measured and not flagged",
      info.timing is not None and info.timing.n_frames == N and info.timing.n_irregular == 0
      and not info.vfr, str(info.timing))
# The case the old metadata test missed: nominal 60, one frame dropped in 600, so
# the average is 59.9 -- within the old 0.5 fps tolerance of the nominal rate.
drop = tmp / "drop.mp4"
ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=60", "-frames:v", "600",
       "-vf", "select='not(eq(n\\,300))'", "-fps_mode", "passthrough",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", str(drop))
dinfo = video.probe(drop)
check("control: the dropped-frame clip's average rate is within 0.5 fps of 60 "
      "(the old check would pass it)", abs(dinfo.fps - 60) < 0.5, str(dinfo.fps))
check("a dropped frame at 60 fps is flagged VFR", dinfo.vfr, str(dinfo.timing))
check("... as exactly one irregular interval, two frames long",
      dinfo.timing.n_irregular == 1 and abs(dinfo.timing.max_gap - 2.0) < 0.05, str(dinfo.timing))
# Millisecond timestamps make 60 fps intervals alternate 16/17 ms (V6 does the same
# at 41/42 ms). That is rounding, not a variable rate, and must not be flagged.
jit = tmp / "jitter.mp4"
ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=60", "-frames:v", "120",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-video_track_timescale", "1000", str(jit))
jinfo = video.probe(jit)
check("control: the jitter clip really has unequal intervals",
      jinfo.timing is not None and jinfo.timing.min_gap < 0.99, str(jinfo.timing))
check("millisecond rounding jitter is not flagged VFR", not jinfo.vfr, str(jinfo.timing))
# A phone dropping 30 -> 24 fps in low light: every interval is only 0.25x off the
# median, so the per-interval check alone passes it (found in review, iteration 2).
sw = tmp / "switch.mp4"
ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=30", "-frames:v", "240",
       # Without a 1/600 time base in the filter AND the encoder, 1/24 s rounds to 1/15.
       "-vf", "settb=1/600,setpts='if(lt(N\\,120)\\,N/30\\,4+(N-120)/24)/TB'",
       "-fps_mode", "passthrough", "-enc_time_base:v", "1/600", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-video_track_timescale", "600", str(sw))
swinfo = video.probe(sw)
check("control: a 30 -> 24 fps switch has no irregular interval",
      swinfo.timing is not None and swinfo.timing.n_irregular == 0, str(swinfo.timing))
check("a 30 -> 24 fps switch is flagged VFR, by drift", swinfo.vfr
      and swinfo.timing.max_drift > video.MAX_DRIFT, str(swinfo.timing))
# A stream-copy cut at a non-keyframe keeps the earlier frames as discard packets.
gop = tmp / "gop.mp4"
ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=60", "-frames:v", "300", "-g", "120",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", str(gop))
cut = tmp / "cut.mp4"
ffmpeg("-ss", "1.5", "-i", str(gop), "-c", "copy", str(cut))
packets = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(cut)],
                         capture_output=True, text=True).stdout.strip()
cinfo = video.probe(cut)
decoded = sum(1 for _ in video.iter_frames(cut, cinfo))
check("control: the cut clip holds more packets than it decodes to",
      int(packets) > decoded, f"{packets} packets, {decoded} decoded")
check("discard packets are excluded: timing counts the frames actually decoded",
      cinfo.timing is not None and cinfo.timing.n_frames == decoded and not cinfo.vfr,
      f"{cinfo.timing} vs {decoded} decoded")


def timing_from(stdout: bytes, rc: int = 0):
    real = video.subprocess.run
    video.subprocess.run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, rc, stdout, b"")
    try:
        return video.frame_timing(clip)
    finally:
        video.subprocess.run = real


check("control: well-formed timestamps parse", timing_from(b"0.0\n0.5\n1.0\n") is not None)
check("missing timestamps (N/A) -> unmeasurable, not a guess",
      timing_from(b"0.0\nN/A\n1.0\n") is None)
check("duplicated timestamps -> unmeasurable", timing_from(b"1.0\n1.0\n1.0\n") is None)
check("non-text ffprobe output -> unmeasurable", timing_from(b"\xff\xfe\x00junk") is None)
check("ffprobe failing -> unmeasurable", timing_from(b"0.0\n0.5\n1.0\n", rc=1) is None)
check("infinite timestamps -> unmeasurable", timing_from(b"0.0\ninf\n1.0\n") is None)
skew = timing_from("\n".join(f"{round(i * 0.0417, 3):.3f}" for i in range(2000)).encode())
check("control: constant skew puts the median (42 ms) off the true rate (41.7 ms)",
      abs(skew.median_dt - 0.042) < 1e-9, str(skew))
check("constant skew is regular: drift is measured at the mean rate, not the median",
      skew.regular and skew.max_drift < 0.1, str(skew))
check("packets flagged D are left out of the timing",
      timing_from(b"-1.0,KD_\n0.0,K__\n0.5,___\n1.0,___\n").n_frames == 3)
real_cap = video.MAX_TIMING_PACKETS
video.MAX_TIMING_PACKETS = 3
try:
    check("more packets than the cap -> refused unparsed, so unmeasurable",
          timing_from(b"0.0\n0.5\n1.0\n1.5\n") is None
          and timing_from(b"0.0\n0.5\n1.0\n") is not None)
finally:
    video.MAX_TIMING_PACKETS = real_cap
real_ft = video.frame_timing
video.frame_timing = lambda p: None
try:
    check("probe treats unmeasurable timing as VFR (deny by default)", video.probe(clip).vfr)
finally:
    video.frame_timing = real_ft
off = video.Timing(N, 1 / 60, 0, 1.0, 1.0, 0.0, 60 * 1.01)  # regular, but 1% off the container
video.frame_timing = lambda p: off
try:
    check("a regular clip whose measured rate is 1% off the container's average is VFR",
          off.regular and video.probe(clip).vfr)
finally:
    video.frame_timing = real_ft

rot = tmp / "rot.mp4"
ffmpeg("-display_rotation", "90", "-i", str(clip), "-c", "copy", str(rot))
rinfo = video.probe(rot)
rframes = list(video.iter_frames(rot, rinfo, 2))
check("rotation flag: probe reports the DISPLAYED size",
      (rinfo.width, rinfo.height) == (H, W), str(rinfo))
check("rotation flag: decoded frames match the probed size",
      rframes[0].shape == (rinfo.height, rinfo.width, 3), str(rframes[0].shape))

# Two video streams, the larger one marked default: ffmpeg's own choice. probe
# measures v:0 (64x48 red), so the decoder must read v:0 too -- before the fix it
# decoded the 320x240 blue stream as 250 chopped "frames" of a 10-frame file.
two = tmp / "two.mp4"
ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x48:r=30", "-f", "lavfi",
       "-i", "color=c=blue:s=320x240:r=30", "-map", "0", "-map", "1",
       "-frames:v:0", "10", "-frames:v:1", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-disposition:v:0", "0", "-disposition:v:1", "default", str(two))
tinfo = video.probe(two)
tframes = list(video.iter_frames(two, tinfo))
check("two video streams: probe measures v:0", (tinfo.width, tinfo.height) == (64, 48), str(tinfo))
check("two video streams: decoder reads the SAME stream (frame count)",
      len(tframes) == 10, str(len(tframes)))
check("two video streams: decoder reads the SAME stream (content is red, not blue)",
      all(f[..., 0].mean() > 200 and f[..., 2].mean() < 50 for f in tframes))


def fake_ffmpeg(payload_frames: float, exit_code: int):
    """Swap ffmpeg for a process that writes `payload_frames` WxH frames and exits."""
    real = subprocess.Popen
    script = (f"import sys; sys.stdout.buffer.write(bytes({int(payload_frames * W * H * 3)}));"
              f" sys.stdout.flush(); sys.exit({exit_code})")
    return real, lambda cmd, **kw: real([sys.executable, "-c", script], **kw)


def decode_with(payload_frames, exit_code, max_frames=None):
    real, fake = fake_ffmpeg(payload_frames, exit_code)
    video.subprocess.Popen = fake
    try:
        return len(list(video.iter_frames(clip, info, max_frames)))
    finally:
        video.subprocess.Popen = real


# The exit code was never read: ffmpeg is still alive at stdout EOF (20/20 runs),
# so the old poll() test always killed it and the failure check was dead code.
check("control: whole frames and exit 0 decode cleanly", decode_with(3, 0) == 3)
raises("ffmpeg exiting non-zero after its frames is an error, not a short video",
       lambda: decode_with(3, 1), RuntimeError)
raises("a partial trailing frame is an error, not silently dropped",
       lambda: decode_with(2.5, 0), RuntimeError)
check("max_frames reached before a failing exit: no error (we stopped it)",
      decode_with(3, 1, max_frames=2) == 2)
gen = video.iter_frames(clip, info)
next(gen)
try:
    gen.close()  # caller abandons the stream mid-file
    check("closing the generator early kills ffmpeg without raising", True)
except Exception as e:
    check("closing the generator early kills ffmpeg without raising", False, repr(e))

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
check("to_track carries joint confidence, row for row",
      trk.conf is not None and trk.conf.shape == (9, N_JOINTS) and np.allclose(trk.conf, 0.9))
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
raises("confidence above 1 is refused",
       lambda: PoseSequence(np.zeros((2, 17, 2)), np.full((2, 17), 2.0), np.ones(2, bool), 30, 9, 9))
raises("NaN confidence is refused",
       lambda: PoseSequence(np.zeros((2, 17, 2)), np.full((2, 17), np.nan), np.ones(2, bool), 30, 9, 9))
badconf = tmp / "badconf.npz"
np.savez(badconf, xy=ps.xy, conf=np.full_like(ps.conf, -1.0), detected=ps.detected, fps=60.0,
         width=W, height=H, source="x", model="x")
raises("a pose file with out-of-range confidence is refused on load",
       lambda: PoseSequence.load(badconf))
from boxingvi.data import Track  # noqa: E402
raises("a Track with confidence of the wrong shape is refused",
       lambda: Track(np.arange(3), np.zeros((3, N_JOINTS, 2)), 30.0, conf=np.ones((3, 5))))
raises("a Track with NaN confidence is refused",
       lambda: Track(np.arange(3), np.zeros((3, N_JOINTS, 2)), 30.0,
                     conf=np.full((3, N_JOINTS), np.nan)))
check("a Track with no confidence (V6) is still accepted",
      Track(np.arange(3), np.zeros((3, N_JOINTS, 2)), 30.0).conf is None)
pickled = tmp / "evil.npz"
np.savez(pickled, xy=np.array([{"a": 1}], dtype=object))
raises("a pickled array in a pose file is refused (allow_pickle off)",
       lambda: PoseSequence.load(pickled))
import dataclasses  # noqa: E402
fp_ps = video.fingerprint(clip)
bound_ps = dataclasses.replace(ps, video_fp=fp_ps)
bound_ps.save(tmp / "bound.pose.npz")
check("a PoseSequence's video_fp round-trips; one written without it loads as ''",
      PoseSequence.load(tmp / "bound.pose.npz").video_fp == fp_ps and PoseSequence.load(f).video_fp == "")
raises("a malformed fingerprint on a PoseSequence is refused", lambda: dataclasses.replace(ps, video_fp="xyz"))
try:
    overlay.render_overlay(clip, dataclasses.replace(ps, video_fp="0" * 64), tmp / "fpno.mp4")
    fp_msg = ""
except ValueError as e:
    fp_msg = str(e)
check("the overlay refuses a pose whose fingerprint is another recording's", "fingerprint" in fp_msg, fp_msg)
check("... and draws one whose fingerprint matches", overlay.render_overlay(clip, bound_ps, tmp / "fpok.mp4") == N)
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

print("usable vs detected")
cu = np.full((N, N_JOINTS), 0.9)
cu[1, :] = 0.05                    # detected, but a blurred lens: nothing solid
# The count bar, tested with shoulders (5, 6) and wrists (9, 10) kept solid so the
# required-joint rule cannot be what decides these two frames.
cu[2, [0, 1, 2, 3, 4, 7, 8, 11, 12, 13]] = 0.05   # 7 solid joints: just under the bar
cu[5, [0, 1, 2, 3, 4, 7, 8, 11, 12]] = 0.05       # 8 solid joints: exactly at the bar
cu[6, 9] = 0.05                    # 16 solid joints, but the LEFT WRIST is a guess
cu[8, 6] = 0.05                    # 16 solid joints, but the RIGHT SHOULDER is a guess
pu = PoseSequence(np.where(ps.detected[:, None, None], ps.xy, 0.0), cu, ps.detected, 60.0, W, H)
u = pu.usable()
check("a detected frame with no solid joints is NOT usable", pu.detected[1] and not u[1])
check("7 solid joints is not usable; 8 is", not u[2] and u[5])
check("16 solid joints with an unsure wrist is NOT usable", pu.detected[6] and not u[6])
check("16 solid joints with an unsure shoulder is NOT usable", pu.detected[8] and not u[8])
check("... and both are usable when no joint is required (the old, count-only rule)",
      pu.usable(required=())[6] and pu.usable(required=())[8])
check("an undetected frame is never usable", not u[3] and not u[4] and not u[7])
check("usable is a subset of detected", not (u & ~pu.detected).any())
check("to_track drops unusable frames by default; usable_only=False keeps every detected one",
      {1, 2, 6, 8}.isdisjoint(pu.to_track().frame_ids)
      and {1, 2, 6, 8} <= set(pu.to_track(usable_only=False).frame_ids))
check("to_track honours the run's min_conf (0.04 admits the 0.05 frames)",
      {1, 2, 6, 8} <= set(pu.to_track(min_conf=0.04).frame_ids))

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
long_ = PoseSequence(np.concatenate([ps.xy, ps.xy[:5]]), np.concatenate([ps.conf, ps.conf[:5]]),
                     np.concatenate([ps.detected, ps.detected[:5]]), 60.0, W, H, "x", "x")
raises("overlay refuses a pose LONGER than the video (it belongs to another video)",
       lambda: overlay.render_overlay(clip, long_, tmp / "long.mp4"))
check("... and leaves no half-written file behind", not (tmp / "long.mp4").exists())
check("a max_frames prefix of a longer pose is allowed (the caller asked for it)",
      overlay.render_overlay(clip, long_, tmp / "pre.mp4", max_frames=5) == 5)
raises("max_frames past the end of both does not excuse a longer pose",
       lambda: overlay.render_overlay(clip, long_, tmp / "past.mp4", max_frames=10 * N))
check("max_frames past the end of both, equal lengths: every frame written",
      overlay.render_overlay(clip, ps, tmp / "past2.mp4", max_frames=10 * N) == N)
slow = PoseSequence(ps.xy, ps.conf, ps.detected, 30.0, W, H, "x", "x")
raises("overlay refuses a pose at a different frame rate",
       lambda: overlay.render_overlay(clip, slow, tmp / "slow.mp4"))

ovb = tmp / "ovb.mp4"
overlay.render_overlay(clip, pu, ovb)
bf = list(video.iter_frames(ovb, video.probe(ovb)))
red = lambda f: f[2, 40, 0] > 150 and f[2, 40, 1] < 90
check("overlay borders every unusable frame (no joints, too few, unsure wrist or shoulder)",
      red(bf[1]) and red(bf[2]) and red(bf[6]) and red(bf[8]) and not red(bf[5])
      and not red(bf[0]))
sx, sy = np.round(GT[5]).astype(int)  # left shoulder: solid on frame 6, well inside the border
check("an unusable frame still shows what WAS found (skeleton drawn under the border)",
      bf[6][sy, sx, 2] > 180 and frames[6][sy, sx, 2] < 120,
      f"overlay {bf[6][sy, sx]} vs source {frames[6][sy, sx]}")

print("run_ingest CLI")
d = tmp / "outdir"
rc = run_ingest.main(["--video", str(clip), "--out-dir", str(d)], predict=predict_all)
check("end to end: pose file and overlay both written",
      rc == 0 and (d / "clip.pose.npz").is_file() and (d / "clip.overlay.mp4").is_file())
rt = PoseSequence.load(d / "clip.pose.npz")
check("the written pose is the full video, fully detected", len(rt) == N and rt.detected.all())
check("... and carries the fingerprint of the video it came from", rt.video_fp == video.fingerprint(clip))
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
