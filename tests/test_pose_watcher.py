"""Tests for pose_watcher.py (laptop) and wait_for_pose.py (desktop).

Synthetic only: generated videos, a predictor injected in place of YOLO (as in
tests/test_analyse_clip.py), fake clocks, and folders in a temp directory standing in
for the two Syncthing folders. Weighted toward refusals: a clip still arriving, a clip
that changes, a failed pose, a pose from another recording, participant footage, a
staging folder inside a synced one, a second watcher. One run goes through the whole
chain (wait_for_pose -> analyse_clip --skip-ingest) to show the handover works.

Run:  python tests/test_pose_watcher.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pose_watcher  # noqa: E402
import wait_for_pose  # noqa: E402
import analyse_clip  # noqa: E402
from boxingvi import classify  # noqa: E402
from boxingvi.data import CLASSES, N_JOINTS  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from boxingvi.stgcnpp import STGCNPP  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def quiet(fn, *args, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = fn(*args, **kw)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


tmp = Path(tempfile.mkdtemp())
W, H, FPS, SECONDS = 160, 120, 60, 20
N = FPS * SECONDS


def make_video(path: Path, colour: str = "gray", n: int = N) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s={W}x{H}:r={FPS}",
                    "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


POSE = np.array([
    [0, -150], [-8, -158], [8, -158], [-16, -154], [16, -154],
    [-50, -100], [50, -100], [-60, -50], [60, -50], [-55, -10], [55, -10],
    [-30, 0], [30, 0], [-32, 90], [32, 90], [-34, 180], [34, 180]], dtype=np.float64)
t = np.arange(N) / FPS
XY = np.broadcast_to(POSE * 0.25 + (80, 60), (N, N_JOINTS, 2)).copy()
for hand, times in ((0, (2, 5, 8, 11, 14)), (1, (3.5, 9.5, 16))):
    for p in times:
        XY[:, 9 + hand, 0] += (-1 if hand == 0 else 1) * 30 * np.sin(np.pi * np.clip((t - p) / 0.25, 0, 1))
calls = {"n": 0}


def make_predictor(hook=None):
    i = {"k": 0}

    def predict(frame):
        calls["n"] += 1
        if hook is not None:
            hook(i["k"])
        kxy = XY[i["k"] % len(XY)]
        i["k"] += 1
        box = np.array([[kxy[:, 0].min(), kxy[:, 1].min(), kxy[:, 0].max(), kxy[:, 1].max()]])
        return box, kxy[None], np.full((1, N_JOINTS), 0.9)
    return predict


weights = tmp / "fake-pose.pt"            # never loaded: a predictor is injected; its name is the model
weights.write_bytes(b"not a checkpoint")
SHA = "0" * 64


class Clock:
    def __init__(self, t0=1e10):
        self.t = t0

    def __call__(self):
        return self.t


def setup(name: str):
    base = tmp / name
    src, ingest, staging = base / "data" / "own", base / "results" / "ingest", base / "results" / ".pose-staging"
    for d in (src, ingest, staging):
        d.mkdir(parents=True)
    return src, ingest, staging


def watcher(src, ingest, staging, settle=0.0, predict="new", clock=None):
    return pose_watcher.Watcher(src, ingest, staging, weights, SHA, "cpu", settle,
                                make_predictor() if predict == "new" else predict, None, clock or Clock())


base_video = make_video(tmp / "base.mp4")
other_video = make_video(tmp / "other.mp4", colour="white")


def twice(w):
    """Two passes: a clip is ready only when the second sees what the first saw."""
    r1, o1, _ = quiet(w.poll)
    r2, o2, _ = quiet(w.poll)
    return r1 + r2, o1 + o2, ""


def place(src: Path, name: str, video: Path = base_video) -> Path:
    p = src / name
    p.write_bytes(video.read_bytes())
    return p


print("== pose_watcher: the happy path ==")
src, ingest, staging = setup("happy")
clip = place(src, "IMG_9001.mp4")
w = watcher(src, ingest, staging)
rc, out, _ = quiet(w.poll)
check("the FIRST look at a clip never reads it, however old its mtime (Syncthing copies the mtime)",
      rc == 0 and not any(ingest.iterdir()) and calls["n"] == 0, out)
rc, out, _ = quiet(w.poll)
npz = ingest / "IMG_9001.pose.npz"
check("a settled clip gets its pose in the synced folder", rc == 1 and npz.is_file(), out)
pose = PoseSequence.load(npz)
check("the pose is this video's, whole (check_pose passes)",
      quiet(analyse_clip.check_pose, pose, analyse_clip.probe(clip), clip, weights.name)[0] is None)
check("the pose has one row per frame", len(pose) == N, str(len(pose)))
check("nothing is left in the staging folder", list(staging.iterdir()) == [], str(list(staging.iterdir())))
check("nothing but the pose was written to the synced folder", sorted(p.name for p in ingest.iterdir())
      == ["IMG_9001.pose.npz"])
n0 = calls["n"]
rc, _, _ = quiet(w.poll)
check("a second pass leaves a clip that has its pose alone", rc == 0 and calls["n"] == n0)

print("== pose_watcher: not until it has finished arriving ==")
src, ingest, staging = setup("arriving")
clip = place(src, "IMG_9002.mp4")
tmpfile = src / "~syncthing~IMG_9002.mp4.tmp"
tmpfile.write_bytes(b"partial")
w = watcher(src, ingest, staging)
rc, out, _ = quiet(w.poll)
check("a Syncthing temp file (Windows name) holds the clip back", rc == 0 and not any(ingest.iterdir())
      and "waiting" in out, out)
tmpfile.unlink()
(src / ".syncthing.IMG_9002.mp4.tmp").write_bytes(b"partial")
rc, _, _ = quiet(w.poll)
check("... and so does the Linux/macOS temp name", rc == 0 and not any(ingest.iterdir()))
(src / ".syncthing.IMG_9002.mp4.tmp").unlink()
rc, _, _ = twice(w)
check("once the temp file is gone the clip is processed", rc == 1 and (ingest / "IMG_9002.pose.npz").is_file())

src, ingest, staging = setup("settle")
clip = place(src, "IMG_9003.mp4")
clock = Clock(clip.stat().st_mtime + 10)
w = watcher(src, ingest, staging, settle=60, clock=clock)
rc, _, _ = twice(w)
check("a clip modified 10 s ago waits out a 60 s settle, even when two looks agree", rc == 0 and not any(ingest.iterdir()))
with open(clip, "ab") as f:
    f.write(b"\0" * 10)
clock.t += 120
rc, _, _ = quiet(w.poll)
check("a clip whose size changed since the last pass is not ready, however old", rc == 0
      and not any(ingest.iterdir()))
clip.write_bytes(base_video.read_bytes())
os.utime(clip, (clock.t - 1000, clock.t - 1000))
rc, _, _ = twice(w)
check("unchanged across two passes and settled: processed", rc == 1 and (ingest / "IMG_9003.pose.npz").is_file())

print("== pose_watcher: failures are recorded, not retried, and retried on request ==")
src, ingest, staging = setup("fail")
clip = place(src, "IMG_9004.mp4")


def boom(frame):
    raise RuntimeError("CUDA out of memory")


w = watcher(src, ingest, staging, predict=boom)
rc, out, _ = twice(w)
failed = ingest / "IMG_9004.pose.failed"
check("a failed pose writes X.pose.failed and no pose", rc == 0 and failed.is_file()
      and not (ingest / "IMG_9004.pose.npz").exists(), out)
check("the failure says why, in one line", "CUDA out of memory" in failed.read_text(encoding="utf-8")
      and failed.read_text(encoding="utf-8").count("\n") == 1)
check("the staging folder is cleaned after a failure", list(staging.iterdir()) == [], str(list(staging.iterdir())))
n0 = calls["n"]
w.predict = make_predictor()
rc, _, _ = quiet(w.poll)
check("a clip with X.pose.failed is not retried", rc == 0 and calls["n"] == n0)
failed.unlink()
rc, _, _ = twice(w)
check("deleting X.pose.failed retries it", rc == 1 and (ingest / "IMG_9004.pose.npz").is_file())

src, ingest, staging = setup("changed")
clip = place(src, "IMG_9005.mp4")


def touch_mid_run(k):
    if k == 100:
        with open(clip, "ab") as f:
            f.write(b"\0" * 16)


w = watcher(src, ingest, staging, predict=make_predictor(touch_mid_run))
rc, out, _ = twice(w)
check("a clip that changes while pose runs publishes nothing and is not marked failed",
      rc == 0 and not any(ingest.iterdir()) and "changed while pose ran" in out, out)

src, ingest, staging = setup("short")
clip = place(src, "IMG_9006.mp4")


def short_pose(real_extract=pose_watcher.run_ingest.pose_yolo.extract):
    def extract(video, predict, name, max_frames, info):
        return real_extract(video, predict, name, 50, info)   # a pose cut short
    return extract


saved = pose_watcher.run_ingest.pose_yolo.extract
pose_watcher.run_ingest.pose_yolo.extract = short_pose()
try:
    w = watcher(src, ingest, staging)
    rc, _, _ = twice(w)
finally:
    pose_watcher.run_ingest.pose_yolo.extract = saved
check("a pose that is not the whole video is refused (failed, not published)",
      rc == 0 and not (ingest / "IMG_9006.pose.npz").exists() and (ingest / "IMG_9006.pose.failed").is_file()
      and "frames" in (ingest / "IMG_9006.pose.failed").read_text(encoding="utf-8"))

print("== pose_watcher: what counts as a clip ==")
src, ingest, staging = setup("names")
for n in ("IMG_1.mov", "IMG_2.MP4", "-bad.mov", "a b.mov", "notes.txt", "DUP.mov", "DUP.mp4", ".hidden.mov"):
    (src / n).write_bytes(b"x")
(src / "sub").mkdir()
(src / "sub" / "IMG_3.mov").write_bytes(b"x")
got = sorted(p.name for p in pose_watcher.candidates(src))
check("only top-level videos with accepted names; duplicate stems skipped", got == ["IMG_1.mov", "IMG_2.MP4"], str(got))
try:
    os.symlink(src / "IMG_1.mov", src / "LINK.mov")
    linked = True
except (OSError, NotImplementedError):
    linked = False
if linked:
    check("a link in data/own is not a clip", "LINK.mov" not in [p.name for p in pose_watcher.candidates(src)])
else:
    print("  (skip) cannot create a symbolic link here without privilege")
check("a missing --src gives no clips, not a crash", pose_watcher.candidates(tmp / "absent") == [])

print("== pose_watcher: folders, deny by default ==")
root = tmp / "participant"
(root / "own").mkdir(parents=True)


def refused(fn, *args, **kw):
    try:
        fn(*args, **kw)
        return False
    except ValueError:
        return True


check("a --src under a participant folder is refused",
      refused(pose_watcher.check_folders, root / "own", tmp / "i", tmp / "s", roots=(root,)))
check("... also when reached through '..'",
      refused(pose_watcher.check_folders, tmp / "x" / ".." / "participant" / "own", tmp / "i", tmp / "s", roots=(root,)))
check("a staging folder inside --ingest-dir is refused",
      refused(pose_watcher.check_folders, tmp / "src", tmp / "ing", tmp / "ing" / "stage", roots=()))
check("a staging folder inside --src is refused",
      refused(pose_watcher.check_folders, tmp / "src", tmp / "ing", tmp / "src" / "stage", roots=()))
check("--src and --ingest-dir the same folder is refused",
      refused(pose_watcher.check_folders, tmp / "same", tmp / "same", tmp / "st", roots=()))
check("separate folders are accepted", not refused(pose_watcher.check_folders, tmp / "src", tmp / "ing", tmp / "st",
                                                   roots=(root,)))
src, ingest, staging = setup("cli")
rc, _, err = quiet(pose_watcher.main, ["--src", str(src), "--ingest-dir", str(ingest), "--staging",
                                       str(ingest / "st"), "--once"], predict=make_predictor())
check("main refuses a staging folder inside the synced folder, writing nothing", rc == 2
      and not (ingest / "st").exists(), err)
rc, _, err = quiet(pose_watcher.main, ["--src", str(src), "--ingest-dir", str(ingest), "--staging", str(staging),
                                       "--once", "--interval", "0"], predict=make_predictor())
check("an interval under 1 s is a usage error", rc == 2, err)
rc, _, err = quiet(pose_watcher.main, ["--src", str(src), "--ingest-dir", str(ingest), "--staging", str(staging),
                                       "--once", "--weights", str(tmp / "absent.pt")])
check("missing weights are refused before any pass", rc == 2 and "weights" in err, err)
place(src, "IMG_9010.mp4")
os.utime(src / "IMG_9010.mp4", (1, 1))
rc, out, _ = quiet(pose_watcher.main, ["--src", str(src), "--ingest-dir", str(ingest), "--staging", str(staging),
                                       "--once", "--settle", "5", "--weights", str(weights), "--weights-sha256", SHA,
                                       "--device", "cpu", "--interval", "1"], predict=make_predictor(), sleep=lambda s: None)
check("main --once makes the pose and exits 0", rc == 0 and (ingest / "IMG_9010.pose.npz").is_file(), out)

with pose_watcher.SingleInstance(staging / "watcher.lock"):
    rc, out, _ = quiet(pose_watcher.main, ["--src", str(src), "--ingest-dir", str(ingest), "--staging", str(staging),
                                           "--once", "--weights", str(weights), "--device", "cpu"],
                       predict=make_predictor())
check("a second watcher exits 1 while one holds the lock", rc == 1 and "already running" in out, out)
with pose_watcher.SingleInstance(staging / "watcher.lock"):
    pass
check("the lock is free again once its holder ends", True)

print("== wait_for_pose ==")
src, ingest, staging = setup("wait")
clip = place(src, "IMG_9020.mp4")
twice(watcher(src, ingest, staging))
out_dir = tmp / "wait" / "results" / "analyse"
seen = {}


def fake_analyse(args):
    seen["args"] = args
    return 0


def W(extra, analyse=fake_analyse, clock=None):
    c = clock or Clock(0)

    def sleep(s):
        c.t += s
    return quiet(wait_for_pose.main, ["--video", str(clip), "--stance", "orthodox", "--ingest-dir", str(ingest),
                                      "--out-dir", str(out_dir), "--weights", str(weights), "--poll", "5"] + extra,
                 sleep=sleep, clock=c, analyse=analyse)


rc, out, err = W(["--rounds", "0-20"])
a = seen.get("args", [])
check("a pose that has arrived is checked, copied and handed to analyse_clip --skip-ingest",
      rc == 0 and "--skip-ingest" in a and a[a.index("--stance") + 1] == "orthodox"
      and a[a.index("--rounds") + 1] == "0-20", out + err)
check("the copy in --out-dir is byte-identical to the synced pose",
      (out_dir / "IMG_9020.pose.npz").read_bytes() == (ingest / "IMG_9020.pose.npz").read_bytes())
check("no temporary copy is left in --out-dir", sorted(p.name for p in out_dir.iterdir()) == ["IMG_9020.pose.npz"])

seen.clear()
(ingest / "IMG_9020.pose.failed").write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")
rc, _, err = W([])
check("X.pose.failed: exit 1, its reason printed, nothing run", rc == 1 and "CUDA out of memory" in err
      and "args" not in seen, err)
(ingest / "IMG_9020.pose.failed").unlink()

src2, ingest2, _ = setup("wait2")
clip = place(src2, "IMG_9021.mp4")
out_dir = tmp / "wait2" / "out"
rc, out, err = W(["--ingest-dir", str(ingest2), "--timeout", "30"])
check("no pose by --timeout: exit 1, nothing run", rc == 1 and "args" not in seen and "after 30" in err, err)
check("... and nothing written to --out-dir", not out_dir.exists())

(ingest2 / "IMG_9021.pose.npz").write_bytes((ingest / "IMG_9020.pose.npz").read_bytes())
(ingest2 / "~syncthing~IMG_9021.pose.npz.tmp").write_bytes(b"partial")
rc, _, _ = W(["--ingest-dir", str(ingest2), "--timeout", "30"])
check("a pose Syncthing is still writing is waited for, not read", rc == 1 and "args" not in seen)
(ingest2 / "~syncthing~IMG_9021.pose.npz.tmp").unlink()
rc, _, err = W(["--ingest-dir", str(ingest2)])
check("a pose made from ANOTHER clip, renamed onto this one, is refused before anything runs",
      rc == 1 and "not this video's pose" in err and "args" not in seen and not out_dir.exists(), err)

clip = place(src2, "IMG_9022.mp4", other_video)
twice(watcher(src2, ingest2, tmp / "wait2" / "results" / ".pose-staging"))
(ingest2 / "IMG_9022.pose.npz").replace(ingest2 / "IMG_9023.pose.npz")
clip = place(src2, "IMG_9023.mp4")
rc, _, err = W(["--ingest-dir", str(ingest2)])
check("a pose of a different recording with this clip's name is refused (source / fingerprint)",
      rc == 1 and "args" not in seen, err)

proot = tmp / "pp"
proot.mkdir()
pvideo = place(proot, "IMG_9030.mp4")
saved_roots = pose_watcher.PARTICIPANT_ROOTS
pose_watcher.PARTICIPANT_ROOTS = (proot,)
try:
    rc, _, err = quiet(wait_for_pose.main, ["--video", str(pvideo), "--stance", "orthodox", "--timeout", "0"],
                       analyse=fake_analyse)
finally:
    pose_watcher.PARTICIPANT_ROOTS = saved_roots
check("participant footage is refused", rc == 2 and "participant" in err and "args" not in seen, err)
bad = place(src2, "a b.mp4")
rc, _, _ = quiet(wait_for_pose.main, ["--video", str(bad), "--stance", "orthodox"], analyse=fake_analyse)
check("a clip name analyse_clip would refuse is refused up front", rc == 2)
rc, _, _ = quiet(wait_for_pose.main, ["--video", str(clip)], analyse=fake_analyse)
check("no --stance is a usage error: it is never guessed", rc == 2)

print("== the whole chain: wait_for_pose -> analyse_clip --skip-ingest ==")
torch.manual_seed(0)
ck = tmp / "clf.pt"
torch.save(STGCNPP().state_dict(), ck)
C = np.array([[90, 5, 0, 0, 5, 0], [2, 98, 0, 0, 0, 0], [0, 0, 80, 20, 0, 0],
              [0, 10, 40, 50, 0, 0], [30, 0, 0, 0, 20, 0], [10, 0, 0, 0, 0, 90]])
F1 = {c: 2 * int(C[i, i]) / int(C[i].sum() + C[:, i].sum()) for i, c in enumerate(CLASSES)}
manifest = tmp / "clf.json"
classify.write_manifest(manifest, ck, classify.WindowGeometry(0.375, 0.125, 24.0, 200.0, 0.5, 0.55, "relative", 100),
                        50, {"held_out": {"per_class_f1": F1, "confusion": C.tolist()}})
reference = tmp / "ref.json"
reference.write_text(json.dumps({"val_videos": ["V2", "V5", "V9"], "results": [
    {"model": "st-gcn++ pretrained", "confusion": C.tolist(), "per_class_f1": F1}]}), encoding="utf-8")
rule = tmp / "rule.json"
rule.write_text(json.dumps({"video": "V6", "chosen": {"variant": "relative", "threshold": 1.0, "min_gap_s": 0.4,
                "test": {"recall": 0.741, "precision": 0.465, "count_ratio": 1.596, "n_labelled": 406}}}),
                encoding="utf-8")
extent = tmp / "extent.json"
extent.write_text(json.dumps({"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8, "variant": "relative",
                              "beats_fixed_window": True, "gated_cut": classify.GATED_CUT}), encoding="utf-8")
FILES = ["--classifier", str(manifest), "--reference", str(reference), "--rule", str(rule), "--extent", str(extent)]
clip = tmp / "wait" / "data" / "own" / "IMG_9020.mp4"
out_dir = tmp / "chain"
n0 = calls["n"]
rc, out, err = W([], analyse=lambda args: analyse_clip.main(args + FILES))
check("the chain runs end to end from the laptop's pose", rc == 0, (out + err)[-800:])
check("... reusing the pose: the pose model is not run on the desktop", calls["n"] == n0)
for suffix in ("punches.csv", "classified.csv", "report.xlsx", "summary.txt"):
    check(f"... and publishes IMG_9020.{suffix}", (out_dir / f"IMG_9020.{suffix}").is_file())
summary = (out_dir / "IMG_9020.summary.txt").read_text(encoding="utf-8") if (out_dir / "IMG_9020.summary.txt").is_file() else ""
check("the summary counts the planted punches' footage (0:20)", "0:20 of footage" in summary, summary[:300])

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
