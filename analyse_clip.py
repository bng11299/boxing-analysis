"""One clip, one command: video in, every Phase 1 output out.

  python analyse_clip.py --video data/own/IMG_4406.mov --stance orthodox --device cuda:0
  python analyse_clip.py --video data/own/IMG_4406.mov --stance orthodox --skip-ingest
  ... --rounds 0-180,240-420                      # round boundaries in seconds
  ... --out-dir results/analyse                   # the default

Runs the four stages in order, each through its own main(): run_ingest.py (video ->
pose, YOLO11-pose against the pinned weights hash), find_punches.py, classify_punches.py
(--stance is required: the tool never guesses one) and report_punches.py, then draws the
overlay. For a clip X, every output lands in --out-dir as X.pose.npz, X.punches.csv,
X.classified.csv, X.classified.json, X.report.xlsx, X.summary.txt and X.overlay.mp4.

The stages run in a fresh staging folder inside --out-dir, so a stage can only ever
read what an earlier stage wrote in THIS run. A stage that fails stops the chain and
--out-dir is left exactly as it was (a pose the model has just extracted is kept in the
staging folder, and the message says where). Only when every stage has passed, and
classify's provenance hashes check out, does this run's set replace the previous one;
the replacement is all renames, undone if any of them fails, so --out-dir never holds
two runs mixed.

--skip-ingest reuses --out-dir/X.pose.npz instead of running the pose model, but only
if its source file name, frame count, frame rate, frame size and pose model match. The
model is matched by its weights' file NAME (the pose file records no hash), and the
frame count is the decoder's: a video whose frame count cannot be read is refused.

Outputs contain your footage or a skeleton of it: results/ is gitignored. Never commit
them, and never put participant footage in the vault or the public demo.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import classify_punches  # noqa: E402
import find_punches  # noqa: E402
import report_punches  # noqa: E402
import run_ingest  # noqa: E402
from boxingvi import classify, overlay, report  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from boxingvi.video import VideoInfo, probe  # noqa: E402

DEFAULT_OUT = Path("results/analyse")
DEFAULT_WEIGHTS = Path("weights/yolo11m-pose.pt")
# docs/MACHINE-SYNC.md section 4: equal to GitHub's digest for the Ultralytics v8.4.0 asset.
PINNED_WEIGHTS_SHA256 = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"
DEFAULT_CLASSIFIER = Path("weights/classifier/stgcnpp-boxingvi-seed0.json")
OVERLAY_MIN_CONF = 0.3   # run_ingest.py's default: the joints the overlay draws as solid
# Each stage names its outputs <clip>.<suffix> from the text before the first dot of a file
# name, so a clip name with a dot (my.clip.mp4) would scatter its outputs across two names;
# a leading - would read as an option.
_CLIP_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,99}")
_DEVICE = re.compile(r"cpu|cuda:[0-9]{1,2}")
_HEX64 = re.compile(r"[0-9a-fA-F]{64}")
OUTPUTS = (("pose.npz", "pose"), ("punches.csv", "punches"), ("classified.csv", "classified"),
           ("classified.json", "provenance"), ("report.xlsx", "report"), ("summary.txt", "summary"),
           ("overlay.mp4", "overlay"))


class StageFailed(Exception):
    pass


class PublishFailed(StageFailed):
    """`state` says what --out-dir holds afterwards; `keep`, that the staging folder holds
    files that are nowhere else."""

    def __init__(self, msg: str, state: str, keep: bool):
        super().__init__(msg)
        self.state, self.keep = state, keep


def render_overlay(video: Path, pose: PoseSequence, classified_csv: Path, out: Path, info: VideoInfo) -> int:
    """The one place the overlay is drawn; returns the frames written and raises ValueError on
    a refusal. It runs last, so the classified punches exist: feat/punch-overlay swaps its
    punch-labelled overlay in here. Until then it is the skeleton overlay run_ingest.py draws."""
    return overlay.render_overlay(video, pose, out, OVERLAY_MIN_CONF, None, info)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _device(text: str) -> str:
    if not _DEVICE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r}: use cpu or cuda:N")
    return text


def _sha256_arg(text: str) -> str:
    if not _HEX64.fullmatch(text):
        raise argparse.ArgumentTypeError("a sha256 is 64 hex digits")
    return text.lower()


def _abs(path: Path) -> str:
    """Paths go to a stage absolute, so none can start with - and read as an option."""
    return str(Path(path).resolve())


def video_frames(info: VideoInfo) -> int:
    """Frames the decoder will output: counted from the packets (discarded ones excluded),
    else the container's claim. The pose has one row per decoded frame, so the two are
    equal for a whole pose of this video; a wrong container claim fails closed."""
    n = info.timing.n_frames if info.timing is not None else info.nb_frames
    if not n:
        raise ValueError("the video's frame count cannot be read, so a pose cannot be checked against it")
    return n


def check_pose(pose: PoseSequence, info: VideoInfo, video: Path, model: str) -> None:
    """The pose is this video's, whole: same source file, length, rate, size and pose model."""
    name = Path(video).resolve().name
    if pose.source != name:
        raise ValueError(f"the pose was extracted from {pose.source!r}, not {name!r}")
    n = video_frames(info)
    if len(pose) != n:
        raise ValueError(f"the pose has {len(pose)} frames, the video {n}: not the same video, or cut short")
    if abs(pose.fps - info.fps) > 1e-6:
        raise ValueError(f"the pose is at {pose.fps:.6f} fps, the video at {info.fps:.6f}")
    if (pose.width, pose.height) != (info.width, info.height):
        raise ValueError(f"the pose is for {pose.width}x{pose.height}, the video is {info.width}x{info.height}")
    if pose.model != model:
        raise ValueError(f"the pose came from {pose.model!r}, not {model!r} (pass --weights for another model)")


def load_pose(path: Path, info: VideoInfo, video: Path, model: str) -> PoseSequence:
    pose = PoseSequence.load(path)
    check_pose(pose, info, video, model)
    return pose


def unchanged(path: Path, sha: str) -> None:
    if _sha256(path) != sha:
        raise ValueError(f"{path.name} changed while the chain was reading it")


def run_stage(name: str, fn, *args, is_main: bool = False, **kwargs):
    """Run one stage and return what it returns; any failure becomes StageFailed naming the
    stage. A stage's main() must return 0: anything else, None included, is a failure."""
    print(f"\n== {name} ==", flush=True)
    try:
        rc = fn(*args, **kwargs)
    except SystemExit as e:   # a stage's argparse refusing what it was handed
        raise StageFailed(f"{name}: stopped with exit code {e.code}") from None
    except StageFailed:
        raise
    except Exception as e:
        kind = "" if isinstance(e, (ValueError, FileNotFoundError)) else f"{type(e).__name__}: "
        raise StageFailed(f"{name}: {kind}{e}") from e
    if is_main and rc != 0:
        raise StageFailed(f"{name}: exited with code {rc!r}")
    return rc


def _expect(path: Path, stage: str) -> Path:
    if not path.is_file():
        raise StageFailed(f"{stage}: finished but wrote no {path.name}")
    return path


def preflight(a) -> None:
    """Every file the later stages need, read and validated before the pose model runs."""
    find_punches.load_rule(a.rule)
    man = classify.read_manifest(a.classifier)
    classify.read_extent(a.extent, man["geometry"].variant)
    report.benchmark_reliability(a.reference, man)
    report.finder_reliability(a.rule)


def check_classified(csv_path: Path, a, clip: str, pose: PoseSequence) -> None:
    """classify_punches.py's sidecar hashes: the CSV is the one it wrote, by this classifier,
    for this stance and this pose."""
    man = report.verify_provenance(csv_path, csv_path.with_suffix(".json"), a.classifier, clip)
    side = man["_sidecar"]
    if side.get("stance") != a.stance:
        raise ValueError(f"{csv_path.name} was classified as {side.get('stance')!r}, not {a.stance}")
    if side.get("pose_frames") != len(pose) or side.get("fps") != pose.fps:
        raise ValueError(f"{csv_path.name} was classified on another pose "
                         f"({side.get('pose_frames')} frames at {side.get('fps')} fps)")


def publish(staging: Path, out_dir: Path, clip: str, produced: "list[str]", keep: "set[str]") -> None:
    """Replace the previous run's outputs with this run's: move the old set aside into the
    staging folder, then move the new set in. Every step is a rename, so a file another
    program holds open (a report open in Excel) stops it, and then every move made so far
    is undone: --out-dir is never left holding two runs mixed."""
    aside = staging / "previous"
    aside.mkdir()
    names = [f"{clip}.{suffix}" for suffix, _ in OUTPUTS if suffix not in keep]
    moves = ([(out_dir / n, aside / n) for n in names if (out_dir / n).exists() or (out_dir / n).is_symlink()]
             + [(staging / f"{clip}.{suffix}", out_dir / f"{clip}.{suffix}") for suffix in produced])
    done = []
    for src, dst in moves:
        try:
            os.replace(src, dst)
        except OSError as e:
            why = f"could not move {src.name} ({e.strerror or e}): is it open in another program?"
            try:
                for s_, d_ in reversed(done):
                    os.replace(d_, s_)
            except OSError:
                raise PublishFailed(f"publish: {why} Undoing the moves failed too",
                                    f"it holds part of this run and part of the previous one; the rest of both "
                                    f"is in {staging}, which is kept", keep=True) from None
            raise PublishFailed(f"publish: {why} Close it and run again", "it was not changed", keep=False) from None
        done.append((src, dst))


def main(argv=None, predict=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--stance", choices=sorted(classify.STANCES), required=True,
                    help="the boxer's stance; required, never guessed")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-ingest", action="store_true",
                    help="reuse <out-dir>/<clip>.pose.npz if it matches the video")
    ap.add_argument("--device", type=_device, default="cpu", help="cpu or cuda:N")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--weights-sha256", type=_sha256_arg, default=PINNED_WEIGHTS_SHA256)
    ap.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    ap.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--extent", type=Path, default=classify_punches.DEFAULT_EXTENT)
    ap.add_argument("--reference", type=Path, default=report_punches.DEFAULT_REFERENCE)
    ap.add_argument("--rounds", help="round boundaries in seconds, e.g. 0-180,240-420")
    ap.add_argument("--no-overlay", action="store_true", help="skip the overlay video")
    a = ap.parse_args(argv)

    clip = a.video.stem
    if not _CLIP_NAME.fullmatch(clip):
        ap.error(f"{a.video.name}: name the video with letters, digits, _ and - only (not starting with -), "
                 "then one extension (every output is named <clip>.<what>)")
    try:
        info = probe(a.video)
        duration = video_frames(info) / info.fps
        report.parse_rounds(a.rounds, duration)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        ap.error(str(e))

    out_dir = a.out_dir
    pose_path = out_dir / f"{clip}.pose.npz"
    if a.skip_ingest and not pose_path.is_file():
        print(f"analyse_clip: --skip-ingest, but there is no {pose_path}", file=sys.stderr)
        return 1
    if not a.skip_ingest and predict is None and not a.weights.is_file():
        print(f"analyse_clip: no pose-model weights at {a.weights} (pass --weights, or --skip-ingest)",
              file=sys.stderr)
        return 1
    try:
        run_stage("preflight", preflight, a)
    except StageFailed as e:
        print(f"analyse_clip: {e}\nNothing was run or written.", file=sys.stderr)
        return 1

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{clip}.", dir=out_dir))
    except OSError as e:
        print(f"analyse_clip: cannot write in {out_dir}: {e}", file=sys.stderr)
        return 1
    print(f"{clip}: working in {staging}; outputs move to {out_dir} when every stage has passed")
    staged = lambda suffix: staging / f"{clip}.{suffix}"  # noqa: E731
    produced, keep_staging, fresh_pose = [], False, False
    try:
        if a.skip_ingest:
            pose_sha = run_stage("ingest (skipped: hashing the existing pose)", _sha256, pose_path)
            pose = run_stage("ingest (skipped: checking the existing pose)", load_pose, pose_path, info, a.video,
                             a.weights.name)
        else:
            run_stage("ingest", run_ingest.main,
                      ["--video", _abs(a.video), "--weights", _abs(a.weights), f"--weights-sha256={a.weights_sha256}",
                       f"--device={a.device}", "--out-dir", _abs(staging), "--no-overlay"],
                      predict=predict, is_main=True)
            pose_path = _expect(staged("pose.npz"), "ingest")
            pose = run_stage("ingest check", load_pose, pose_path, info, a.video, a.weights.name)
            fresh_pose = True
            produced.append("pose.npz")

        run_stage("find_punches", find_punches.main,
                  ["--pose", _abs(pose_path), "--rule", _abs(a.rule), "--out-dir", _abs(staging)], is_main=True)
        _expect(staged("punches.csv"), "find_punches")
        produced.append("punches.csv")

        run_stage("classify_punches", classify_punches.main,
                  ["--pose", _abs(pose_path), "--punches", _abs(staged("punches.csv")),
                   "--classifier", _abs(a.classifier), f"--stance={a.stance}", "--rule", _abs(a.rule),
                   "--extent", _abs(a.extent), "--out-dir", _abs(staging)], is_main=True)
        classified = _expect(staged("classified.csv"), "classify_punches")
        _expect(staged("classified.json"), "classify_punches")
        run_stage("classify provenance", check_classified, classified, a, clip, pose)
        produced += ["classified.csv", "classified.json"]

        run_stage("report_punches", report_punches.main,
                  ["--pose", _abs(pose_path), "--classified", _abs(classified), "--classifier", _abs(a.classifier),
                   "--reference", _abs(a.reference), "--rule", _abs(a.rule), "--out-dir", _abs(staging)]
                  + ([f"--rounds={a.rounds}"] if a.rounds else []), is_main=True)
        _expect(staged("report.xlsx"), "report_punches")
        _expect(staged("summary.txt"), "report_punches")
        produced += ["report.xlsx", "summary.txt"]

        if not a.no_overlay:
            n = run_stage("overlay", render_overlay, a.video, pose, classified, staged("overlay.mp4"), info)
            _expect(staged("overlay.mp4"), "overlay")
            print(f"overlay: {n} frames")
            produced.append("overlay.mp4")

        if a.skip_ingest:
            run_stage("pose unchanged", unchanged, pose_path, pose_sha)
        # Everything published was made from this run's pose; the rest of the old set is moved out.
        run_stage("publish", publish, staging, out_dir, clip, produced, {"pose.npz"} if a.skip_ingest else set())
    except StageFailed as e:
        state = getattr(e, "state", "it was not changed")
        print(f"\nanalyse_clip: stopped at {e}\nNo later stage ran. {out_dir}: {state}.", file=sys.stderr)
        keep_staging = getattr(e, "keep", False)
        if fresh_pose and not keep_staging:
            # The pose model's output passed its check and may have taken hours: keep it, alone.
            for p in staging.iterdir():
                if p.name != f"{clip}.pose.npz":
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
            keep_staging = True
            print(f"This run's pose is kept: {staged('pose.npz')}. To reuse it, move it into {out_dir} "
                  "and run again with --skip-ingest.", file=sys.stderr)
        return 1
    finally:
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)

    summary = (out_dir / f"{clip}.summary.txt").read_text(encoding="utf-8").strip()
    lines = [f"\n==== {clip}: done ({a.stance}) ===="]
    for suffix, label in OUTPUTS:
        p = out_dir / f"{clip}.{suffix}"
        if suffix == "overlay.mp4" and a.no_overlay:
            lines.append(f"  {label:<11} not drawn (--no-overlay)")
            continue
        note = "  (reused: --skip-ingest)" if suffix == "pose.npz" and a.skip_ingest else ""
        lines.append(f"  {label:<11} {p}  sha256 {_sha256(p)[:12]}{note}")
    lines += ["", summary]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
