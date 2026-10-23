"""Tests for analyse_clip.py: the four stages chained into one command.

Synthetic only: a generated video, a predictor injected in place of YOLO (as in
tests/test_ingest.py, so no GPU or ultralytics), a randomly initialised classifier
and hand-written rule files. No number here says anything about boxing. Weighted
toward what the chain adds: a missing stance, a stale or mismatched pose, a stage
that fails or writes nothing, provenance that does not check out, and stale outputs
from an earlier run.

Run:  python tests/test_analyse_clip.py
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from openpyxl import load_workbook

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import analyse_clip  # noqa: E402
import classify_punches  # noqa: E402
import find_punches  # noqa: E402
import report_punches  # noqa: E402
from boxingvi import classify, video  # noqa: E402
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


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


tmp = Path(tempfile.mkdtemp())
W, H, FPS, SECONDS = 160, 120, 60, 20
N = FPS * SECONDS


def make_video(path: Path, n: int = N, fps: int = FPS, size: str = f"{W}x{H}") -> Path:
    ffmpeg("-f", "lavfi", "-i", f"color=c=gray:s={size}:r={fps}", "-frames:v", str(n),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return path


clip = make_video(tmp / "IMG_9001.mp4")

# A standing body with planted punches: the left wrist at 2, 5, 8, 11 and 14 s, the right at 3.5, 9.5 and 16 s.
# Coordinates in pixels of the 160x120 frame; the classifier sees them only after its own normalisation.
POSE = np.array([
    [0, -150], [-8, -158], [8, -158], [-16, -154], [16, -154],
    [-50, -100], [50, -100], [-60, -50], [60, -50], [-55, -10], [55, -10],
    [-30, 0], [30, 0], [-32, 90], [32, 90], [-34, 180], [34, 180]], dtype=np.float64)


def body_xy(n: int, fps: float) -> np.ndarray:
    t = np.arange(n) / fps
    xy = np.broadcast_to(POSE * 0.25 + (80, 60), (n, N_JOINTS, 2)).copy()
    for hand, times in ((0, (2, 5, 8, 11, 14)), (1, (3.5, 9.5, 16))):
        for p in times:
            ph = np.clip((t - p) / 0.25, 0, 1)
            xy[:, 9 + hand, 0] += (-1 if hand == 0 else 1) * 30 * np.sin(np.pi * ph)
    return xy


XY = body_xy(N, FPS)
calls = {"n": 0}


def make_predictor():
    """Frame i gets XY[i]: extract() calls the predictor once per decoded frame, in order."""
    i = {"k": 0}

    def predict(frame):
        calls["n"] += 1
        kxy = XY[i["k"] % len(XY)]
        i["k"] += 1
        box = np.array([[kxy[:, 0].min(), kxy[:, 1].min(), kxy[:, 0].max(), kxy[:, 1].max()]])
        return box, kxy[None], np.full((1, N_JOINTS), 0.9)
    return predict


# The model files the later stages read. Random weights: the plumbing is under test, not the types.
torch.manual_seed(0)
ck = tmp / "clf.pt"
torch.save(STGCNPP().state_dict(), ck)
C = np.array([[90, 5, 0, 0, 5, 0], [2, 98, 0, 0, 0, 0], [0, 0, 80, 20, 0, 0],
              [0, 10, 40, 50, 0, 0], [30, 0, 0, 0, 20, 0], [10, 0, 0, 0, 0, 90]])
F1 = {c: 2 * int(C[i, i]) / int(C[i].sum() + C[:, i].sum()) for i, c in enumerate(CLASSES)}
GEOM = classify.WindowGeometry(0.375, 0.125, 24.0, 200.0, 0.5, 0.55, "relative", 100)
manifest = tmp / "clf.json"
classify.write_manifest(manifest, ck, GEOM, 50, {"held_out": {"per_class_f1": F1, "confusion": C.tolist()}})
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
weights = tmp / "fake-pose.pt"   # never loaded: a predictor is injected; its name is the pose's model
weights.write_bytes(b"not a checkpoint")
FILES = ["--classifier", str(manifest), "--reference", str(reference), "--rule", str(rule),
         "--extent", str(extent), "--weights", str(weights)]
NAMES = ("pose.npz", "punches.csv", "classified.csv", "classified.json", "report.xlsx", "summary.txt",
         "overlay.mp4")


def run(argv, predict="new"):
    """(exit code, stdout, stderr); argparse refusals give their exit code."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = analyse_clip.main(FILES + argv, predict=make_predictor() if predict == "new" else predict)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


def args(out_dir, *extra, video_=clip, stance="orthodox"):
    a = ["--video", str(video_), "--out-dir", str(out_dir), *extra]
    return a + (["--stance", stance] if stance else [])


def cells(p: Path) -> list:
    """A workbook's sheets and cell values: openpyxl stamps each file with the time it was
    written, so two identical reports differ in bytes."""
    wb = load_workbook(p)
    return [(ws.title, list(ws.iter_rows(values_only=True))) for ws in wb]


def state(d: Path) -> dict:
    """Every file in a folder, recursively, with its hash (a workbook: its cells). 'Unchanged'
    and 'the same outputs' are comparisons of this."""
    return {str(p.relative_to(d)): (cells(p) if p.suffix == ".xlsx" else sha(p))
            for p in sorted(Path(d).rglob("*")) if p.is_file()}


@contextlib.contextmanager
def patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


print("end to end")
out = tmp / "out"
rc, so, se = run(args(out))
check("exit 0", rc == 0, se[-2000:])
check("every output is in --out-dir, named for the clip", all((out / f"IMG_9001.{n}").is_file() for n in NAMES),
      str(sorted(p.name for p in out.iterdir())))
check("nothing else is left in --out-dir (the staging folder is gone)",
      sorted(p.name for p in out.iterdir()) == sorted(f"IMG_9001.{n}" for n in NAMES))
rows = list(csv.DictReader((out / "IMG_9001.classified.csv").open(encoding="utf-8")))
check("the planted punches were found and classified (8 planted)", len(rows) == 8
      and sorted(r["hand"] for r in rows).count("left") == 5, f"{len(rows)} rows")
check("orthodox: the left hand is the lead", all((r["hand"] == "left") == (r["lead_rear"] == "lead") for r in rows))
pose = PoseSequence.load(out / "IMG_9001.pose.npz")
check("the pose is the whole video, at its rate", len(pose) == N and abs(pose.fps - FPS) < 1e-9
      and pose.source == "IMG_9001.mp4" and pose.model == "fake-pose.pt")
ov = video.probe(out / "IMG_9001.overlay.mp4")
check("the overlay has every frame", analyse_clip.video_frames(ov) == N)
summary = (out / "IMG_9001.summary.txt").read_text(encoding="utf-8").strip()
closing = so[so.rindex("==== IMG_9001: done"):]
check("the closing block names every output's path", all(str(out / f"IMG_9001.{n}") in closing for n in NAMES))
check("the closing block ends with the summary text, as written", closing.rstrip().endswith(summary) and summary)
check("the closing block's hash for the classified CSV is the file's",
      sha(out / "IMG_9001.classified.csv")[:12] in closing)
check("the predictor ran once per frame (ingest ran, once)", calls["n"] == N, str(calls["n"]))

print("the same outputs, stage by stage")
sb = tmp / "stagewise"
find_punches.main(["--pose", str(out / "IMG_9001.pose.npz"), "--rule", str(rule), "--out-dir", str(sb)])
classify_punches.main(["--pose", str(out / "IMG_9001.pose.npz"), "--punches", str(sb / "IMG_9001.punches.csv"),
                       "--classifier", str(manifest), "--stance", "orthodox", "--rule", str(rule),
                       "--extent", str(extent), "--out-dir", str(sb)])
report_punches.main(["--pose", str(out / "IMG_9001.pose.npz"), "--classified", str(sb / "IMG_9001.classified.csv"),
                     "--classifier", str(manifest), "--reference", str(reference), "--rule", str(rule),
                     "--out-dir", str(sb)])
for n in ("punches.csv", "classified.csv", "classified.json", "summary.txt"):
    check(f"{n}: byte-identical to running the stages by hand", sha(sb / f"IMG_9001.{n}") == sha(out / f"IMG_9001.{n}"))
check("report.xlsx: the same cells as running the stages by hand",
      cells(sb / "IMG_9001.report.xlsx") == cells(out / "IMG_9001.report.xlsx"))

print("--skip-ingest")
calls["n"] = 0
first = state(out)
rc, so, se = run(args(out, "--skip-ingest"))
check("a matching pose is reused: exit 0, and the pose model never ran", rc == 0 and calls["n"] == 0, se[-2000:])
diff = sorted(k for k in first if state(out).get(k) != first[k])
check("the same outputs as the full run (the pose file untouched)", diff == [], str(diff))
check("the closing block says the pose was reused", "reused: --skip-ingest" in so)


n_skip = [0]


def skip_with(pose_: PoseSequence, why: str, video_=clip, stem="IMG_9001"):
    n_skip[0] += 1
    d = tmp / f"skip_{n_skip[0]}"
    d.mkdir()
    pose_.save(d / f"{stem}.pose.npz")
    before = state(d)
    calls["n"] = 0
    rc_, _, se_ = run(args(d, "--skip-ingest", video_=video_))
    check(f"refused at the ingest stage, nothing later ran: {why}",
          rc_ == 1 and calls["n"] == 0 and "stopped at ingest" in se_, f"rc {rc_}: {se_[-500:]}")
    check(f"... and the folder is unchanged ({why})", state(d) == before)
    return se_


def pose_like(**kw):
    p = dict(xy=pose.xy, conf=pose.conf, detected=pose.detected, fps=pose.fps, width=pose.width,
             height=pose.height, source=pose.source, model=pose.model)
    p.update(kw)
    return PoseSequence(**p)


se = skip_with(pose_like(xy=pose.xy[:-1], conf=pose.conf[:-1], detected=pose.detected[:-1]),
               "a pose one frame short of the video")
check("... and the message gives both frame counts", f"{N - 1} frames" in se and str(N) in se, se)
skip_with(pose_like(fps=30.0), "a pose at another frame rate")
skip_with(pose_like(width=W * 2), "a pose for another frame size")
skip_with(pose_like(source="IMG_9002.mp4"), "a pose extracted from another video")
skip_with(pose_like(model="yolo11n-pose.pt"), "a pose from another pose model")
longer = make_video(tmp / "IMG_9003.mp4", n=N + 60)
skip_with(pose_like(source="IMG_9003.mp4"), "a pose from a shorter cut of the same file name",
          video_=longer, stem="IMG_9003")
d = tmp / "nopose"
rc, _, se = run(args(d, "--skip-ingest"))
check("--skip-ingest with no pose file is refused, naming the file", rc == 1 and "IMG_9001.pose.npz" in se, se)
check("... and creates nothing", not d.exists())
d = tmp / "badpose"
d.mkdir()
(d / "IMG_9001.pose.npz").write_bytes(b"PK not really")
rc, _, se = run(args(d, "--skip-ingest"))
check("a corrupt pose file stops the chain at the ingest stage", rc == 1 and "ingest" in se, se)

print("--skip-ingest: a reused pose's fingerprint is checked, with or without the overlay")
for extra_ in ((), ("--no-overlay",)):
    tag_ = "no overlay" if extra_ else "with overlay"
    d = tmp / ("fp_noov" if extra_ else "fp_ov")
    shutil.copytree(out, d)                                   # a previous run's good outputs
    pose_like(video_fp="0" * 64).save(d / "IMG_9001.pose.npz")  # same name and metadata, another recording
    before_ = state(d)
    calls["n"] = 0
    rc, _, se = run(args(d, "--skip-ingest", *extra_))
    check(f"a reused pose whose fingerprint is another recording's is refused at ingest ({tag_})",
          rc == 1 and calls["n"] == 0 and "stopped at ingest" in se and "fingerprint" in se, se[-600:])
    check(f"... and the previous outputs are kept as they were ({tag_})", state(d) == before_)
d = tmp / "fp_match"
d.mkdir()
pose_like(video_fp=video.fingerprint(clip)).save(d / "IMG_9001.pose.npz")
rc, _, se = run(args(d, "--skip-ingest", "--no-overlay"))
check("a reused pose whose fingerprint matches this video is accepted", rc == 0, se[-600:])

print("the overlay stage draws the classified punches")
seen_ov = []
real_ov = analyse_clip.render_punches.render_punch_overlay


def spy_ov(video_, pose_, classified_, out_, **kw):
    seen_ov.append((Path(classified_).name, kw.get("min_conf"), kw.get("info") is not None))
    return real_ov(video_, pose_, classified_, out_, **kw)


d = tmp / "ovspy"
d.mkdir()
shutil.copy(out / "IMG_9001.pose.npz", d / "IMG_9001.pose.npz")
with patched(analyse_clip.render_punches, "render_punch_overlay", spy_ov):
    rc, _, se = run(args(d, "--skip-ingest"))
check("the overlay is render_punches.render_punch_overlay, given this run's classified CSV, "
      "the overlay confidence and the probed video",
      rc == 0 and seen_ov == [("IMG_9001.classified.csv", analyse_clip.OVERLAY_MIN_CONF, True)],
      f"{seen_ov} {se[-400:]}")

print("the stance is never guessed")
d = tmp / "nostance"
calls["n"] = 0
rc, _, se = run(args(d, stance=None))
check("no --stance: a usage error", rc == 2 and "--stance" in se, se)
check("... before anything runs or is written", calls["n"] == 0 and not d.exists())
rc, _, se = run(args(d, stance="switch"))
check("an unknown stance is a usage error", rc == 2 and not d.exists())

print("arguments")
for bad, why in ((["--device", "cuda"], "a device without its number"),
                 (["--device", "cpu --weights x"], "a device carrying more text"),
                 (["--rounds", "0-500"], "rounds past the end of the clip")):
    calls["n"] = 0
    rc, _, se = run(args(tmp / "argbad", *bad))
    check(f"refused before ingest: {why}", rc == 2 and calls["n"] == 0 and not (tmp / "argbad").exists(), se)
dotted = make_video(tmp / "my.clip.mp4", n=30)
rc, _, se = run(args(tmp / "dotted", video_=dotted))
check("a clip name with a dot is refused (its outputs would be named 'my.*')", rc == 2 and "my.clip.mp4" in se, se)
dashed = make_video(tmp / "-h.mp4", n=30)
rc, _, se = run(args(tmp / "dashed", video_=dashed))
check("a clip name starting with - is refused (a stage would read it as an option)", rc == 2, se)
rc, _, se = run(args(tmp / "badsha", "--weights-sha256", "abc"))
check("a weights hash that is not 64 hex digits is a usage error", rc == 2 and "64 hex" in se, se)
(tmp / "afile").write_text("x", encoding="utf-8")
rc, _, se = run(args(tmp / "afile"))
check("an --out-dir that is a file: exit 1 with a message, not a traceback", rc == 1 and "cannot write" in se, se)
rc, _, se = run(args(tmp / "novideo", video_=tmp / "IMG_0000.mp4"))
check("a missing video is a usage error", rc == 2 and not (tmp / "novideo").exists(), se)
rc, _, se = run(args(tmp / "noweights") + ["--weights", str(tmp / "absent.pt")], predict=None)
check("a real run with no weights file is refused before anything is written",
      rc == 1 and "absent.pt" in se and not (tmp / "noweights").exists(), se)
rc, _, se = run(args(tmp / "nomanifest") + ["--classifier", str(tmp / "absent.json")])
check("a missing classifier is caught by the preflight, before the pose model runs",
      rc == 1 and "preflight" in se and not (tmp / "nomanifest").exists(), se)
bad_extent = tmp / "extent_bad.json"
bad_extent.write_text(json.dumps({"frac": 0.5, "lead_widths": 1.3, "tail_widths": 1.8, "variant": "relative",
                                  "beats_fixed_window": False, "gated_cut": classify.GATED_CUT}), encoding="utf-8")
calls["n"] = 0
rc, _, se = run(args(tmp / "badextent") + ["--extent", str(bad_extent)])
check("an extent rule that failed its gate is caught by the preflight", rc == 1 and calls["n"] == 0
      and "preflight" in se, se)
old_manifest = tmp / "clf_old.json"   # the shape of every manifest written before 2026-10-21: no matrix
classify.write_manifest(old_manifest, ck, GEOM, 50, {"held_out": {"per_class_f1": F1}})
calls["n"] = 0
rc, _, se = run(args(tmp / "oldmanifest") + ["--classifier", str(old_manifest)])
check("a manifest with no pinned held-out matrix is caught by the preflight, before the pose model runs",
      rc == 1 and calls["n"] == 0 and "re-run train_classifier.py" in se and not (tmp / "oldmanifest").exists(), se)
transposed = tmp / "ref_t.json"
transposed.write_text(json.dumps({"results": [{"model": "st-gcn++ pretrained", "confusion": C.T.tolist(),
                                               "per_class_f1": F1}]}), encoding="utf-8")
calls["n"] = 0
rc, _, se = run(args(tmp / "reft") + ["--reference", str(transposed)])
check("a transposed reference matrix is caught by the preflight", rc == 1 and calls["n"] == 0
      and "rows and columns swapped" in se, se)

print("a stage that fails stops the chain; out-dir keeps the previous run")


def failing_run(module, name, replacement, why, expect_in_msg, later=()):
    before = state(out)
    seen = []
    spies = [(m, n, lambda *a_, m=m, n=n, **k: seen.append(n) or 0) for m, n in later]
    with contextlib.ExitStack() as st:
        st.enter_context(patched(module, name, replacement))
        for m, n, spy in spies:
            st.enter_context(patched(m, n, spy))
        rc_, so_, se_ = run(args(out, "--skip-ingest"))
    check(f"{why}: exit 1, the message names the stage", rc_ == 1 and expect_in_msg in se_, se_[-800:])
    check(f"{why}: no later stage ran", seen == [], str(seen))
    check(f"{why}: out-dir unchanged, no staging folder left", state(out) == before
          and not any(p.name.startswith(".") for p in out.iterdir()))
    return se_


def boom(argv):
    raise ValueError("the classifier refused this window")


se = failing_run(classify_punches, "main", boom, "classify raises", "classify_punches: the classifier refused",
                 later=[(report_punches, "main")])
failing_run(find_punches, "main", lambda argv: 3, "find_punches exits non-zero", "find_punches: exited with code 3",
            later=[(classify_punches, "main"), (report_punches, "main")])
failing_run(find_punches, "main", lambda argv: None, "find_punches returns no exit code",
            "find_punches: exited with code None", later=[(classify_punches, "main")])


def bad_args(argv):
    raise SystemExit(2)


failing_run(report_punches, "main", bad_args, "report's argparse refuses", "report_punches: stopped with exit code 2",
            later=[(analyse_clip, "render_overlay")])


def crash(*a_, **k):
    raise TypeError("a bug")


failing_run(analyse_clip, "render_overlay", crash, "the overlay crashes", "overlay: TypeError: a bug")


def top(d: Path) -> dict:
    return {k: v for k, v in state(d).items() if not k.startswith(".")}


before = state(out)
with patched(classify_punches, "main", boom):
    rc, _, se = run(args(out))   # a full run: the pose model runs
kept = [p for p in out.iterdir() if p.name.startswith(".")]
check("a full run that fails after ingest: exit 1, the earlier run's outputs (its pose too) untouched",
      rc == 1 and top(out) == before, se[-500:])
check("... and this run's checked pose is kept, alone, where the message says",
      len(kept) == 1 and [p.name for p in kept[0].iterdir()] == ["IMG_9001.pose.npz"]
      and str(kept[0] / "IMG_9001.pose.npz") in se and len(PoseSequence.load(kept[0] / "IMG_9001.pose.npz")) == N,
      se[-500:])
for k in kept:
    shutil.rmtree(k)

print("no stage reads a stale file from an earlier run")
# out/ holds a complete earlier run. A finder that 'succeeds' without writing must not hand
# classify that earlier punches.csv.
real_classify = classify_punches.main
failing_run(find_punches, "main", lambda argv: 0, "find_punches writes nothing",
            "find_punches: finished but wrote no IMG_9001.punches.csv", later=[(classify_punches, "main")])


def classify_then_edit_csv(argv):
    real_classify(argv)
    p = Path(argv[argv.index("--out-dir") + 1]) / "IMG_9001.classified.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")   # one byte more
    return 0


failing_run(classify_punches, "main", classify_then_edit_csv, "a classified CSV changed after classify hashed it",
            "has changed since classify_punches.py wrote it", later=[(report_punches, "main")])


def classify_as(stance):
    def run_(argv):
        return real_classify([f"--stance={stance}" if x.startswith("--stance") else x for x in argv])
    return run_


failing_run(classify_punches, "main", classify_as("southpaw"), "a sidecar recording another stance",
            "classified as 'southpaw', not orthodox", later=[(report_punches, "main")])


def classify_with_other_sidecar(argv):
    real_classify(argv)
    p = Path(argv[argv.index("--out-dir") + 1]) / "IMG_9001.classified.json"
    side = json.loads(p.read_text(encoding="utf-8"))
    side["classifier_sha256"] = "0" * 64
    p.write_text(json.dumps(side), encoding="utf-8")
    return 0


failing_run(classify_punches, "main", classify_with_other_sidecar, "a sidecar naming another classifier",
            "named by another classifier", later=[(report_punches, "main")])


def classify_on_other_pose(argv):
    real_classify(argv)
    p = Path(argv[argv.index("--out-dir") + 1]) / "IMG_9001.classified.json"
    side = json.loads(p.read_text(encoding="utf-8"))
    side["pose_frames"] = N + 1
    p.write_text(json.dumps(side), encoding="utf-8")
    return 0


failing_run(classify_punches, "main", classify_on_other_pose, "a sidecar recording a pose of another length",
            "was classified on another pose", later=[(report_punches, "main")])


def sidecar_edit(**kw):
    """A classify stage that runs for real, then has its sidecar rewritten (the sidecar is unsigned)."""
    def run_(argv):
        real_classify(argv)
        p = Path(argv[argv.index("--out-dir") + 1]) / "IMG_9001.classified.json"
        p.write_text(json.dumps({**json.loads(p.read_text(encoding="utf-8")), **kw}), encoding="utf-8")
        return 0
    return run_


failing_run(classify_punches, "main", sidecar_edit(fps=30.0), "a sidecar recording only another frame rate",
            "was classified on another pose: these punches were classified on a pose at 30.0 fps",
            later=[(report_punches, "main")])
failing_run(classify_punches, "main", sidecar_edit(fps=10 ** 400), "a sidecar fps too big for a float",
            "was classified on another pose", later=[(report_punches, "main")])
failing_run(classify_punches, "main", sidecar_edit(rule={"variant": "relative", "threshold": 2.0, "min_gap_s": 0.4}),
            "a sidecar recording another finder rule", "report_punches: these punches were found with another rule")
real_find = find_punches.main


def find_then_swap_pose(argv):
    real_find(argv)
    p = Path(argv[argv.index("--pose") + 1])
    pose_like(model="swapped.pt").save(p)
    return 0


before = state(out)
with patched(find_punches, "main", find_then_swap_pose):
    rc, _, se = run(args(out, "--skip-ingest"))
check("a reused pose replaced while the chain read it is caught before publishing",
      rc == 1 and "changed while the chain was reading it" in se, se[-500:])
pose.save(out / "IMG_9001.pose.npz")   # restore the reused pose for the cases below
check("... and every other output is the earlier run's", {k: v for k, v in state(out).items() if "pose" not in k}
      == {k: v for k, v in before.items() if "pose" not in k})

print("the overlay is drawn through one function, after classify")
seen_overlay = []


def spy_overlay(video_, pose_, classified_csv, out_, info):
    seen_overlay.append((Path(video_).name, len(pose_), list(csv.reader(open(classified_csv, encoding="utf-8")))[0],
                         Path(out_).name))
    Path(out_).write_bytes(b"spy")
    return 7


with patched(analyse_clip, "render_overlay", spy_overlay):
    rc, so, _ = run(args(out, "--skip-ingest"))
check("called once, with the video, the pose, this run's classified CSV and the overlay's name",
      rc == 0 and seen_overlay == [("IMG_9001.mp4", N, classify_punches.OUT_HEADER, "IMG_9001.overlay.mp4")],
      str(seen_overlay))
check("what it wrote is the published overlay", (out / "IMG_9001.overlay.mp4").read_bytes() == b"spy")

print("--no-overlay")
rc, so, _ = run(args(out, "--skip-ingest", "--no-overlay"))
check("exit 0, and the closing block says no overlay was drawn", rc == 0 and "not drawn (--no-overlay)" in so)
check("an earlier run's overlay is not left beside this run's outputs", not (out / "IMG_9001.overlay.mp4").exists())

print("publishing")
rc, _, _ = run(args(out, "--skip-ingest"))
before = state(out)
if os.name == "nt":   # Windows refuses to rename a file another program holds open, as Excel does
    with open(out / "IMG_9001.report.xlsx", "rb"):
        rc, _, se = run(args(out, "--skip-ingest"))
    check("a report held open elsewhere: exit 1, named, and out-dir exactly as it was",
          rc == 1 and "could not move IMG_9001.report.xlsx" in se and "was not changed" in se
          and state(out) == before and not any(p.name.startswith(".") for p in out.iterdir()), se[-500:])
else:
    print("  SKIP  a report held open elsewhere (only Windows refuses to rename an open file)")
real_replace = os.replace


def replace_failing(on, also_undo=False, exc=PermissionError(13, "Access is denied")):
    def rep_(src, dst):
        if Path(src).name == on and Path(src).parent.name.startswith(".IMG_9001."):   # moving this run's file in
            raise exc
        if also_undo and Path(src).parent.name == "previous":   # ... and putting the old set back
            raise PermissionError(13, "Access is denied")
        return real_replace(src, dst)
    return rep_


with patched(os, "replace", replace_failing("IMG_9001.summary.txt")):
    rc, _, se = run(args(out, "--skip-ingest"))
check("a move failing half way through publishing is undone: out-dir exactly as it was",
      rc == 1 and "could not move IMG_9001.summary.txt" in se and state(out) == before
      and not any(p.name.startswith(".") for p in out.iterdir()), se[-500:])
with patched(os, "replace", replace_failing("IMG_9001.summary.txt", also_undo=True)):
    rc, _, se = run(args(out, "--skip-ingest"))
kept = [p for p in out.iterdir() if p.name.startswith(".")]
check("if undoing fails too, it says out-dir is mixed and keeps the staging folder that holds the rest",
      rc == 1 and "part of this run and part of the previous one" in se and len(kept) == 1 and str(kept[0]) in se,
      se[-500:])
for k in kept:
    shutil.rmtree(k)
rc, _, _ = run(args(out, "--skip-ingest"))
check("and the next run publishes a whole set again", rc == 0 and all((out / f"IMG_9001.{n}").is_file()
                                                                      for n in NAMES))
before = state(out)
interrupted = False
with patched(os, "replace", replace_failing("IMG_9001.summary.txt", exc=KeyboardInterrupt())):
    try:
        run(args(out, "--skip-ingest"))
    except KeyboardInterrupt:
        interrupted = True
check("Ctrl+C half way through publishing: the moves are undone, then the interrupt goes on",
      interrupted and state(out) == before and not any(p.name.startswith(".") for p in out.iterdir()))
(out / "IMG_9001.report.xlsx").rename(tmp / "report.keep")
(out / "IMG_9001.report.xlsx").mkdir()
(out / "IMG_9001.report.xlsx" / "mine.txt").write_text("not an output", encoding="utf-8")
before = state(out)
rc, _, se = run(args(out, "--skip-ingest"))
check("a folder with an output's name is refused, not moved and deleted: out-dir exactly as it was",
      rc == 1 and "is not a file" in se and state(out) == before
      and not any(p.name.startswith(".") for p in out.iterdir()), se[-500:])
shutil.rmtree(out / "IMG_9001.report.xlsx")
(tmp / "report.keep").rename(out / "IMG_9001.report.xlsx")
real_unlink = Path.unlink


def unlink_failing(self, *a_, **k):
    if self.name == "IMG_9001.punches.csv":
        raise PermissionError(13, "Access is denied")
    return real_unlink(self, *a_, **k)


before = top(out)
with patched(classify_punches, "main", boom), patched(Path, "unlink", unlink_failing):
    rc, _, se = run(args(out))   # a full run failing after ingest, whose tidy-up then fails
kept = [p for p in out.iterdir() if p.name.startswith(".")]
check("if tidying the kept pose's folder fails, the pose is still kept and the run ends with a message",
      rc == 1 and len(kept) == 1 and (kept[0] / "IMG_9001.pose.npz").is_file() and "could not tidy" in se
      and top(out) == before, se[-500:])
for k in kept:
    shutil.rmtree(k)

print("the clip name contract with the stages")
check("every stage names outputs from the text before the first dot, so the clip name has none",
      analyse_clip._CLIP_NAME.fullmatch("IMG_4406") and not analyse_clip._CLIP_NAME.fullmatch("a.b")
      and not analyse_clip._CLIP_NAME.fullmatch("..") and not analyse_clip._CLIP_NAME.fullmatch("a/b"))
check("the pinned pose-model hash is the one docs/MACHINE-SYNC.md records",
      analyse_clip.PINNED_WEIGHTS_SHA256 in (ROOT / "docs" / "MACHINE-SYNC.md").read_text(encoding="utf-8"))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
