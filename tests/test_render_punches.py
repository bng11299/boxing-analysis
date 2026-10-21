"""Tests for render_punches.py: punch labels drawn on the skeleton overlay.

Synthetic frames and a synthetic 30-frame clip only: planted punches whose
labelled frames are worked out by hand below. Weighted toward the trust boundary
(the classified CSV, its sidecar, the pose and the video, all files sitting on
disk between tools): most checks are refusals.

Run:  python tests/test_render_punches.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import render_punches as rp  # noqa: E402
from boxingvi import overlay, report, video  # noqa: E402
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
W, H, N, FPS = 160, 120, 30, 60.0
CLIP = "TESTCLIP"
WHY = "pose in only 1 of the window's 9 frames"

# Orthodox: left leads. Four punches; spans as classify_punches.py writes them.
#  A  jab, left/lead, peak 0, span -5..3     -> starts before the clip: frames 0..3
#  B  unclassified, right/rear, peak 12, span 10..14             -> frames 10..14
#  C  lead hook thrown by the RIGHT (rear) hand, peak 13, span 12..16: hands disagree
#  D  cross, right/rear, peak 29 (the last frame), span 26..34 -> ends after: 26..29
# Labelled frames: 0-3, 10-16, 26-29. Nothing on 4-9, 17-25.
LABELLED = set(range(0, 4)) | set(range(10, 17)) | set(range(26, 30))
HEADER = report.CLASSIFIED_HEADER


def csv_row(frame, hand, lead_rear, ptype, conf, type_hand, agree, a, b, observed=9, note=""):
    return [str(frame), f"{frame / FPS:.3f}", hand, "5.000", "no", lead_rear, ptype, conf, type_hand,
            agree, str(a), str(b), str(observed), note]


ROWS = [csv_row(0, "left", "lead", "jab", "0.950", "lead", "yes", -5, 3),
        csv_row(12, "right", "rear", "unclassified", "", "", "", 10, 14, 1, WHY),
        csv_row(13, "right", "rear", "lead_hook", "0.700", "lead", "no", 12, 16),
        csv_row(29, "right", "rear", "cross", "0.800", "rear", "yes", 26, 34)]


def write_case(d: Path, rows=ROWS, clip=CLIP, side: "dict | None" = None) -> Path:
    """<clip>.classified.csv + its sidecar in `d`; `side` overrides sidecar fields."""
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{clip}.classified.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    meta = {"clip": clip, "csv_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "classifier_sha256": "0" * 64, "stance": "orthodox", "pose_frames": N, "fps": FPS}
    meta.update(side or {})
    p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return p


def make_pose(n=N, source=f"{CLIP}.mp4", wrist_conf=0.9) -> PoseSequence:
    """A standing figure in the bottom half of the frame, clear of the label block."""
    xy = np.zeros((n, N_JOINTS, 2))
    xy[:, :, 0] = np.linspace(50, 110, N_JOINTS)
    xy[:, :, 1] = np.linspace(75, 112, N_JOINTS)
    xy[:, 9], xy[:, 10] = (35, 95), (125, 95)   # wrists far enough apart that each ring is its own
    conf = np.full((n, N_JOINTS), 0.9)
    conf[:, 9:11] = wrist_conf
    return PoseSequence(xy, conf, np.ones(n, bool), FPS, W, H, source=source, model="test")


def _not_utf8() -> Path:
    """A hash-matched CSV whose bytes are not UTF-8."""
    p = write_case(tmp / "latin1")
    p.write_bytes(p.read_bytes().replace(b"jab", b"j\xe9b"))
    side = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    side["csv_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    p.with_suffix(".json").write_text(json.dumps(side), encoding="utf-8")
    return p


def rows_of(p: Path, n=N):
    return report.read_classified(p, n / FPS, fps=FPS)


SCALE, LX, LY, MAXC = rp.layout(W, H)
LINE_H = overlay.GLYPH_H * SCALE


def label_area(f: np.ndarray) -> np.ndarray:
    """The first label line (and its backing), across the frame's width."""
    return f[LY - SCALE:LY + LINE_H + SCALE, :]


# ---------------------------------------------------------------- drawing ---
print("text and ring drawing (overlay.py)")
img = np.zeros((40, 200, 3), np.uint8)
overlay.draw_text(img, "jab 0.95", 2, 2, 2, (255, 255, 255))
check("draw_text marks pixels", img.any())
check("lower case draws as upper case", np.array_equal(
    img, (lambda f: (overlay.draw_text(f, "JAB 0.95", 2, 2, 2, (255, 255, 255)), f)[1])(np.zeros_like(img))))
ys_, xs_ = np.nonzero(img.any(axis=2))
check("text_size is exactly the drawn extent",
      (xs_.max() + 1 - 2, ys_.max() + 1 - 2) == overlay.text_size("JAB 0.95", 2), str(overlay.text_size("JAB 0.95", 2)))
raises("a character with no glyph is refused, not dropped",
       lambda: overlay.draw_text(np.zeros((20, 50, 3), np.uint8), "JAB!", 0, 0, 1, (255, 0, 0)))
edge = np.zeros((10, 10, 3), np.uint8)
overlay.draw_text(edge, "WWWW", 5, 5, 3, (255, 0, 0))  # mostly off the frame: clipped, no IndexError
check("text running off the frame is clipped", edge[5:, 5:].any() and not edge[:5].any())
ring = np.zeros((60, 60, 3), np.uint8)
overlay.draw_ring(ring, (30, 30), 15, (0, 255, 0), 1)
check("ring draws its outline, not its centre", ring[30, 15:46].any() and not ring[27:34, 27:34].any())

print("label text")
rs = rows_of(write_case(tmp / "txt"))
check("named punch: type in words, hand, confidence as written",
      rp.punch_text(rs[0]) == ("JAB - LEAD HAND - 0.950", rp.C_NAMED), str(rp.punch_text(rs[0])))
check("unnamed punch: grey, with its reason",
      rp.punch_text(rs[1]) == (f"UNCLASSIFIED - REAR HAND ({WHY})", rp.C_UNNAMED), str(rp.punch_text(rs[1])))
check("hands disagreeing is said, not hidden",
      rp.punch_text(rs[2])[0] == "LEAD HOOK - REAR HAND - 0.700 (TYPE IS A LEAD PUNCH)")
check("wrap keeps every word, within the width",
      " ".join(rp._wrap("ONE TWO THREE FOUR FIVE", 9)) == "ONE TWO THREE FOUR FIVE"
      and all(len(x) <= 9 for x in rp._wrap("ONE TWO THREE FOUR FIVE", 9)))
check("a word wider than the frame is cut, not lost",
      "".join(rp._wrap("ABCDEFGHIJKLMNOPQRST", 8)) == "ABCDEFGHIJKLMNOPQRST")

print("PunchLabels on synthetic frames")
pose = make_pose()
labels = rp.PunchLabels(rs, pose)
rng = np.random.default_rng(0)
base = rng.integers(40, 216, (H, W, 3), dtype=np.uint8)    # textured, so a darkened box shows
changed = set()
for i in range(N):
    f = base.copy()
    labels(f, i)
    if not np.array_equal(label_area(f), label_area(base)):
        changed.add(i)
check("label area changes on exactly the punch frames", changed == LABELLED,
      f"extra {sorted(changed - LABELLED)}, missing {sorted(LABELLED - changed)}")
check("a punch whose span starts before the clip is drawn on frame 0", 0 in changed)
check("a punch whose span ends after the clip is drawn on the last frame", N - 1 in changed)


def expected(lines) -> np.ndarray:
    """A black frame with exactly these (text, colour) lines in the label block."""
    f = np.zeros((H, W, 3), np.uint8)
    y = LY
    for text, color in lines:
        overlay.draw_text(f, text, LX, y, SCALE, color)
        y += LINE_H + 3 * SCALE
    return f


def drawn(lab, i) -> np.ndarray:
    f = np.zeros((H, W, 3), np.uint8)   # black: the darkened backing stays black, only text shows
    lab(f, i)
    return f


no_rings = rp.PunchLabels(rs, make_pose(wrist_conf=0.1))
want12 = ([(x, rp.C_UNNAMED) for x in rp._wrap(rp.punch_text(rs[1])[0], MAXC)]
          + [(x, rp.C_NAMED) for x in rp._wrap(rp.punch_text(rs[2])[0], MAXC)])
check("overlapping punches: both labels, every line, in peak order (frame 12)",
      np.array_equal(drawn(no_rings, 12), expected(want12)) and len(want12) >= 4, str(want12))
check("frame 0 reads exactly the jab's label",
      np.array_equal(drawn(no_rings, 0), expected([("JAB - LEAD HAND - 0.950", rp.C_NAMED)])))
quiet = base.copy()
labels(quiet, 6)
check("nothing at all is drawn between punches", np.array_equal(quiet, base))


def ring_pixels(f, i, hand):
    """Pixels on the wrist ring's circle that differ from the base frame."""
    cx, cy = pose.xy[i, rp.WRIST[hand]]
    t = np.linspace(0, 2 * np.pi, 64)
    xs = np.clip(np.rint(cx + labels.radius * np.cos(t)).astype(int), 0, W - 1)
    ys = np.clip(np.rint(cy + labels.radius * np.sin(t)).astype(int), 0, H - 1)
    return int((f[ys, xs] != base[ys, xs]).any(axis=1).sum())


check("the punching wrist is ringed (jab, left wrist, frame 2)",
      ring_pixels((lambda f: (labels(f, 2), f)[1])(base.copy()), 2, "left") > 32)
check("the other wrist is not (frame 2, right wrist)",
      ring_pixels((lambda f: (labels(f, 2), f)[1])(base.copy()), 2, "right") < 8)
blind = rp.PunchLabels(rs, make_pose(wrist_conf=0.1))
fb = base.copy()
blind(fb, 2)
check("no ring on a wrist the model is unsure of (label still drawn)",
      ring_pixels(fb, 2, "left") < 8 and not np.array_equal(label_area(fb), label_area(base)))
nospan = rows_of(write_case(tmp / "nospan", rows=[csv_row(7, "right", "rear", "unclassified", "", "", "",
                                                         "", "", 0, WHY)]))
ns = rp.PunchLabels(nospan, pose)
check("an unclassified punch with no span is labelled on its peak frame only",
      [i for i in range(N) if ns.active(i)] == [7])
huge = rows_of(write_case(tmp / "huge", rows=[csv_row(5, "left", "lead", "jab", "0.900", "lead", "yes",
                                                      -10 ** 30, 10 ** 30)]))
hl = rp.PunchLabels(huge, pose)
check("a span far past both ends of the clip: no overflow, every frame labelled",
      all(hl.active(i) == [0] for i in range(N)))
crowd = rp.PunchLabels([dict(rs[0], span=(0, 5)) for _ in range(rp.MAX_LABELS + 2)], make_pose(wrist_conf=0.1))
check("more overlapping punches than fit: capped, the rest counted",
      np.array_equal(drawn(crowd, 1), expected([("JAB - LEAD HAND - 0.950", rp.C_NAMED)] * rp.MAX_LABELS
                                               + [("+2 MORE", rp.C_UNNAMED)])))
raises("a label the font cannot draw is refused before any frame",
       lambda: rp.PunchLabels([dict(rs[1], note="pose lost!")], pose))

# --------------------------------------------------------------- end to end ---
print("render_punch_overlay end to end")
vid = tmp / f"{CLIP}.mp4"
ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=60", "-frames:v", str(N),
       "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(vid))
pose_npz = tmp / f"{CLIP}.pose.npz"
pose.save(pose_npz)
good = write_case(tmp / "good")
out = tmp / f"{CLIP}.punches.overlay.mp4"
n = rp.render_punch_overlay(vid, pose_npz, good, out)
check("one output frame per video frame", n == N, str(n))
streams = json.loads(subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name", "-of", "json", str(out)],
    capture_output=True, check=True).stdout)["streams"]
check("H.264, and no audio stream",
      [(s["codec_type"], s["codec_name"]) for s in streams] == [("video", "h264")], str(streams))
info = video.probe(out)
check("constant frame rate, at the video's rate", not info.vfr and info.fps_frac == 60, f"{info}")
plain = tmp / f"{CLIP}.overlay.mp4"
overlay.render_overlay(vid, pose, plain)
got = list(video.iter_frames(out))
ref = list(video.iter_frames(plain))
diff = [float(np.abs(label_area(g).astype(int) - label_area(r).astype(int)).mean()) for g, r in zip(got, ref)]
seen = {i for i, d in enumerate(diff) if d > 3.0}
check("decoded video: label area differs from the plain overlay on exactly the punch frames",
      seen == LABELLED and len(got) == N, f"diff by frame {[round(d, 1) for d in diff]}")


def read_number(f: np.ndarray, digits: int = 2) -> str:
    """Read the burned-in number back off a decoded frame, bit by bit: each of a
    digit's 3x5 bits is a 3x3 block at scale 3, on if its mean yellowness
    ((R+G)/2 - B: ~200 for the digits, ~10-50 for grey) is over 120. Lossy 4:2:0
    output bleeds colour at the digits' edges, which a block mean absorbs."""
    g = f.astype(int)
    yellow = (g[..., 0] + g[..., 1]) / 2 - g[..., 2]
    out = ""
    for k in range(digits):
        s_, y0 = overlay.NUMBER_SCALE, overlay.NUMBER_Y
        x0 = overlay.NUMBER_X + k * 4 * s_
        bits = "".join("1" if yellow[y0 + s_ * r:y0 + s_ * (r + 1), x0 + s_ * c:x0 + s_ * (c + 1)].mean() > 120
                       else "0" for r in range(5) for c in range(3))
        out += next((d for d, b in overlay._DIGITS.items() if b == bits), "" if "1" not in bits else "?")
    return out


def shows_number(f: np.ndarray, i: int) -> bool:
    return read_number(f) == str(i)


check("the burned-in frame number is each frame's own index, labels or not",
      all(shows_number(g, i) for i, g in enumerate(got)),
      str([i for i, g in enumerate(got) if not shows_number(g, i)]))
check("... and the check can fail: frame 12 does not read as 17", not shows_number(got[12], 17))
check("an in-memory PoseSequence works the same",
      rp.render_punch_overlay(vid, pose, good, tmp / "mem.mp4") == N)
n_pre = rp.render_punch_overlay(vid, pose_npz, good, tmp / "pre.mp4", max_frames=5)
check("max_frames renders a prefix", n_pre == 5)

print("refusals")
r_out = tmp / "refused.mp4"
bad = write_case(tmp / "edited")
bad.write_bytes(bad.read_bytes().replace(b"0.950", b"0.990"))
raises("a CSV edited since classify_punches.py wrote it",
       lambda: rp.render_punch_overlay(vid, pose_npz, bad, r_out))
raises("a sidecar for another clip",
       lambda: rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "other", side={"clip": "IMG_0001"}),
                                       r_out))
raises("a sidecar recording another pose length",
       lambda: rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "len", side={"pose_frames": N + 1}), r_out))
raises("a sidecar recording another frame rate",
       lambda: rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "fps", side={"fps": 30.0}), r_out))
raises("a sidecar with no frame rate",
       lambda: rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "nofps", side={"fps": None}), r_out))
# Sidecar frames and fps match this pose and the hash matches, but every time is frame / 30: the CSV
# came from a 30 fps clip. Only the row check (report.read_classified's fps=) can see it.
slow = [r_[:1] + [f"{int(r_[0]) / 30:.3f}"] + r_[2:] for r_ in ROWS]
try:
    rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "slow", rows=slow), r_out)
    why = "accepted"
except ValueError as e:
    why = str(e)
check("a hash-matched CSV whose times are frame / 30 is refused against this pose", "is not frame" in why, why)
gp = write_case(tmp / "bytes")
check("read_classified takes the caller's bytes and the fps together",
      report.read_classified(gp, N / FPS, gp.read_bytes(), fps=FPS) == rows_of(gp))
nos = write_case(tmp / "noside")
nos.with_suffix(".json").unlink()
raises("no sidecar beside the CSV", lambda: rp.render_punch_overlay(vid, pose_npz, nos, r_out))
junk = write_case(tmp / "junk")
junk.with_suffix(".json").write_text("not json", encoding="utf-8")
raises("a sidecar that is not JSON", lambda: rp.render_punch_overlay(vid, pose_npz, junk, r_out))
past = write_case(tmp / "past", rows=ROWS[:3] + [csv_row(N, "right", "rear", "cross", "0.800", "rear", "yes",
                                                         N - 2, N + 3)])
raises("a punch peaking past the pose's last frame (sidecar hash matching)",
       lambda: rp.render_punch_overlay(vid, pose_npz, past, r_out))
other_pose = tmp / "IMG_0001.pose.npz"
shutil.copy(pose_npz, other_pose)
raises("a pose file for another clip", lambda: rp.render_punch_overlay(vid, other_pose, good, r_out))
other_vid = tmp / "IMG_0001.mp4"
shutil.copy(vid, other_vid)
raises("a video for another clip", lambda: rp.render_punch_overlay(other_vid, pose, good, r_out))
vid2 = tmp / "v2" / f"{CLIP}.mov"
vid2.parent.mkdir()
shutil.copy(vid, vid2)
raises("a pose extracted from a different file", lambda: rp.render_punch_overlay(vid2, pose, good, r_out))
raises("output onto the video", lambda: rp.render_punch_overlay(vid, pose_npz, good, vid))
raises("output onto the CSV", lambda: rp.render_punch_overlay(vid, pose_npz, good, good))
raises("output onto the pose", lambda: rp.render_punch_overlay(vid, pose_npz, good, pose_npz))
long_pose = make_pose(n=N + 5)
raises("a pose longer than the video (render_overlay's own check)",
       lambda: rp.render_punch_overlay(vid, long_pose, write_case(tmp / "long", side={"pose_frames": N + 5}), r_out))

# Crafted inputs that pass the hash check (the sidecar is unsigned: whoever can write
# the CSV can rewrite it) must still come out as ValueError, the caller's "refused".
raises("a sidecar fps too big for a float",
       lambda: rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "bigfps", side={"fps": 10 ** 400}), r_out))
raises("a sidecar pose length that is not an integer",
       lambda: rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "fl", side={"pose_frames": float(N)}), r_out))
deep = write_case(tmp / "deep")
deep.with_suffix(".json").write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
raises("a sidecar nested deep enough to exhaust the parser",
       lambda: rp.render_punch_overlay(vid, pose_npz, deep, r_out))
raises("a CSV field longer than the csv module allows",
       lambda: rp.render_punch_overlay(vid, pose_npz, write_case(tmp / "wide", rows=[ROWS[0][:-1] + ["x" * 200_000]]),
                                       r_out))
raises("a CSV that is not UTF-8 (hash matching)", lambda: rp.render_punch_overlay(vid, pose_npz, _not_utf8(), r_out))
keyless = tmp / "k" / f"{CLIP}.pose.npz"
keyless.parent.mkdir()
np.savez(keyless, xy=np.zeros((N, N_JOINTS, 2)))
raises("a pose file missing its arrays", lambda: rp.render_punch_overlay(vid, keyless, good, r_out))
notzip = tmp / "z" / f"{CLIP}.pose.npz"
notzip.parent.mkdir()
notzip.write_bytes(b"PK\x03\x04 not really a zip")
raises("a pose file that is not an npz", lambda: rp.render_punch_overlay(vid, notzip, good, r_out))
bomb = tmp / "bomb" / f"{CLIP}.pose.npz"   # ~300 bytes claiming a 10^12-frame pose
bomb.parent.mkdir()
with zipfile.ZipFile(bomb, "w") as zf:
    for key, shape in (("xy", (10 ** 12, N_JOINTS, 2)), ("conf", (10 ** 12, N_JOINTS)), ("detected", (10 ** 12,))):
        buf = io.BytesIO()
        np.lib.format.write_array_header_1_0(buf, {"descr": "<f8", "fortran_order": False, "shape": shape})
        zf.writestr(f"{key}.npy", buf.getvalue())
raises("a tiny pose file claiming an enormous array", lambda: rp.render_punch_overlay(vid, bomb, good, r_out))
real_probe = rp.videomod.probe


def _hang(path):
    raise subprocess.TimeoutExpired(["ffprobe"], 30)


rp.videomod.probe = _hang
try:
    raises("ffprobe hanging on the video", lambda: rp.render_punch_overlay(vid, pose_npz, good, r_out))
finally:
    rp.videomod.probe = real_probe
raises("a missing video", lambda: rp.render_punch_overlay(tmp / "gone" / f"{CLIP}.mp4", pose, good, r_out))
nocsv = write_case(tmp / "nocsv")
nocsv.unlink()
raises("a missing CSV (its sidecar present)", lambda: rp.render_punch_overlay(vid, pose_npz, nocsv, r_out))
anon = make_pose(source="")
raises("a pose that does not record its video (fail closed)",
       lambda: rp.render_punch_overlay(vid, anon, good, r_out))
link = tmp / "link.mp4"
os.link(vid, link)
raises("output onto a hard link to the video", lambda: rp.render_punch_overlay(vid, pose_npz, good, link))
check("... and the video survived it", video.probe(vid).width == W and link.stat().st_size == vid.stat().st_size)
check("a refusal leaves no output file", not r_out.exists())
check("inputs untouched by the refusals", vid.is_file() and pose_npz.is_file() and good.is_file()
      and report.verify_sidecar(good, good.with_suffix(".json"), CLIP)["clip"] == CLIP)

print("render_punches.py command line")
d = tmp / "cli"
rc = rp.main(["--video", str(vid), "--pose", str(pose_npz), "--classified", str(good), "--out-dir", str(d)])
check("CLI writes <clip>.punches.overlay.mp4", rc == 0 and (d / f"{CLIP}.punches.overlay.mp4").is_file())

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
