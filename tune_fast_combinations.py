"""Tune the two fast-combination arms on TRAINING videos only, and pick ONE.

    python tune_fast_combinations.py --data data --poses results/ingest

Follows docs/gates/fast-combinations.md, committed before this was run (its commit is
recorded). Refuses to run until the athlete-disjoint split (#38, O55) is on main, and
never reads a held-out video: the tuning videos are every BoxingVI source NOT in
data.HELD_OUT_VIDEOS. Each needs YOLO11-pose of its source at the native frame rate
(results/ingest/<V>.pose.npz) and must pass the alignment gate; a video that fails is
excluded and listed. V6 labels only hooks and uppercuts, so it counts towards the
recall checks and not towards F1.

Writes results/fast_combinations_tune.json (the whole grid, the choice within each
arm, the chosen arm, the re-tuned pooled control) and results/fast_combinations_rule.json
(the chosen arm as a find_punches.py rule). Nothing here is scored on held-out footage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import eval_pipeline_v5v9 as e2e  # noqa: E402
import find_punches  # noqa: E402
from boxingvi import data as bdata  # noqa: E402
from boxingvi import fastcombo as fc  # noqa: E402
from boxingvi import poseswitch, punchfind  # noqa: E402
from boxingvi.data import NATIVE_FPS, Span  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402

GATE = Path("docs/gates/fast-combinations.md")
DEFAULT_TUNE = Path("results/fast_combinations_tune.json")
DEFAULT_RULE_OUT = Path("results/fast_combinations_rule.json")
DEFAULT_HELDOUT = Path("results/fast_combinations_heldout.json")
# Created by eval_fast_combinations.py, atomically, before it scores the arm. Fixed under the repo,
# never an option, so no choice of output path lets a second held-out run or a retune through.
STARTED = ROOT / "results" / "fast_combinations_heldout.started"


def split_ready() -> None:
    """The O55 split must be merged: it is what makes the held-out list athlete-disjoint."""
    if not hasattr(bdata, "PERSON_OF_VIDEO"):
        raise SystemExit("REFUSED: the athlete-disjoint split (#38, O55) is not on this branch "
                         "(data.PERSON_OF_VIDEO is missing). Tuning waits for it: rebase onto main after it merges.")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def gate_commit() -> str:
    """The commit that last changed the gate, which must be committed and unmodified."""
    if _git("status", "--porcelain", "--", str(GATE)):
        raise SystemExit(f"REFUSED: {GATE} has uncommitted changes; the gate is fixed before any number")
    c = _git("log", "-1", "--format=%H", "--", str(GATE))
    if not c:
        raise SystemExit(f"REFUSED: {GATE} is not committed")
    return c


def refuse_if_started() -> None:
    if STARTED.exists():
        raise SystemExit(f"REFUSED: {STARTED} exists: the held-out run has started (or finished). Tuning again, "
                         "or a second held-out run, would choose with the held-out result known")


def video_order(v: str) -> int:
    return int(v[1:])


def training_videos() -> "list[str]":
    held = set(bdata.HELD_OUT_VIDEOS)
    out = sorted((v for v in NATIVE_FPS if v not in held), key=video_order)
    if held & set(out):
        raise AssertionError("a held-out video is in the tuning list")
    return out


def load_video(data: Path, poses: Path, v: str) -> "tuple[fc.Video, dict]":
    """One video's pose, ground truth and alignment, checked as eval_pipeline_v5v9 checks
    V5 and V9 (its gate 2 and the alignment), with the O61 pose pairing for V2 and V3."""
    pose_path = poses / f"{fc.POSE_FILE.get(v, v)}.pose.npz"
    pose = PoseSequence.load(pose_path)
    if abs(pose.fps - NATIVE_FPS[v]) > 1e-3:
        raise SystemExit(f"{v}: pose {pose_path.name} is {pose.fps:.3f} fps, the source is {NATIVE_FPS[v]:.3f}")
    if v == "V6":
        spans, clips, bad = v6_ground_truth(data)
    else:
        clips, mismatched = poseswitch.held_out_clips(data, v)
        bad = e2e.unusable_rows(data, v, {c.index for c in clips})
        if len(bad) != mismatched:
            raise SystemExit(f"{v}: the left-out labelled rows do not reconcile")
        spans = [Span(c.start, c.end, c.label) for c in clips]
    if not spans or max(s.end for s in spans) >= len(pose) or any(e >= len(pose) for _, _, e in bad):
        raise SystemExit(f"{v}: the spans run past the pose's {len(pose)} frames")
    al = poseswitch.align(pose, clips)
    track = punchfind.bridge(pose.to_track(usable_only=False), int(round(punchfind.BRIDGE_S * pose.fps)))
    video = fc.Video(name=v, fps=pose.fps, n_frames=len(pose), track=track,
                     signal=punchfind.wrist_signal(track, fc.VARIANT),
                     spans=spans, unusable=[(s, e) for _, s, e in bad])
    return video, {"pose_file": pose_path.name, "best_offset": al["best"], "sharpness": al["ratio"],
                   "aligned": bool(al["sharp"] and al["best"] == 0), "labelled": len(spans),
                   "alignment_windows": len(clips), "unusable_rows": len(bad)}


def v6_ground_truth(data: Path) -> "tuple[list[Span], list, list]":
    """V6 ships ONE continuous skeleton in video frame numbers (load_v6_track), not clips, and
    its labels as spans in the same numbering (load_v6_spans, which refuses an unusable row, so
    none is left out). For the alignment gate, each span's first frames (up to the corpus's 25)
    become a comparison window in poseswitch's form, so V6 passes or fails the same test as every
    other video. A window with a frame the skeleton lacks is not used for alignment; its span
    still counts as ground truth."""
    track = bdata.load_v6_track(data / "Skeleton_data")
    spans = bdata.load_v6_spans(data / "Annotation_files")
    row = {int(f): i for i, f in enumerate(track.frame_ids)}
    w, h = bdata.SOURCE_SIZE
    clips = []
    for k, s in enumerate(spans):
        frames = list(range(s.start, min(s.end, s.start + bdata.MAX_FRAMES - 1) + 1))
        if len(frames) < 2 or any(f not in row for f in frames):
            continue
        r = [row[f] for f in frames]
        kp = np.zeros((len(frames), bdata.N_JOINTS, 3), np.float32)
        kp[..., 0], kp[..., 1], kp[..., 2] = track.xy[r, :, 0] / w, track.xy[r, :, 1] / h, 1.0
        clips.append(poseswitch.HeldOutClip("V6", k, frames[0], frames[-1], s.label, bdata.pad_or_trim(kp)))
    if not clips:
        raise SystemExit("V6: no labelled span lies on the skeleton; nothing to align")
    return spans, clips, []


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def baseline_rule(path: Path) -> dict:
    r = find_punches.load_rule(path)
    if ("finder" in r or r["variant"] != fc.VARIANT or r["min_gap_s"] != fc.MIN_GAP_S
            or not math.isfinite(r["threshold"]) or abs(r["threshold"] - fc.BASELINE_THRESHOLD) > 1e-9):
        raise SystemExit(f"{path} is not the pooled V6 rule the gate compares against")
    return r


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--out", type=Path, default=DEFAULT_TUNE)
    ap.add_argument("--rule-out", type=Path, default=DEFAULT_RULE_OUT)
    a = ap.parse_args(argv)

    split_ready()
    refuse_if_started()
    commit = gate_commit()
    base = baseline_rule(a.baseline)
    videos = training_videos()
    missing = [v for v in videos if not (a.poses / f"{fc.POSE_FILE.get(v, v)}.pose.npz").is_file()]
    if missing:
        raise SystemExit("REFUSED: no pose for tuning video(s) " + ", ".join(missing)
                         + f" in {a.poses}: run run_ingest.py on the GPU laptop first (docs/commands/ingest.md)")

    loaded, checks, excluded = {}, {}, []
    for v in videos:
        video, chk = load_video(a.data, a.poses, v)
        checks[v] = chk
        if chk["aligned"]:
            loaded[v] = video
        else:
            excluded.append(v)
            print(f"{v}: EXCLUDED, fails the alignment gate (best offset {chk['best_offset']}, "
                  f"sharpness {chk['sharpness']})")
    f1_videos = [v for v in loaded if v not in fc.F1_EXCLUDED]
    if not f1_videos:
        raise SystemExit("REFUSED: no tuning video with every punch type labelled passed the alignment gate")
    print(f"tuning on {', '.join(loaded)} (F1 on {', '.join(f1_videos)}); excluded {excluded or 'none'}")

    base_c = {v: fc.counts(vd, fc.detections(vd, base)) for v, vd in loaded.items()}
    base_all, base_f1 = fc.total(base_c), fc.total(base_c, f1_videos)
    masks = {v: {} for v in loaded}
    points = []
    for arm, param, thr in fc.grid():
        r = fc.rule(thr, fc.finder_of(arm, param))
        c = {v: fc.counts(vd, fc.detections(vd, r, masks[v])) for v, vd in loaded.items()}
        t_all, t_f1 = fc.total(c), fc.total(c, f1_videos)
        ok, why = fc.admissible(t_all, base_all)
        points.append({"arm": arm, "param": param, "threshold": thr, "f1": fc.f1(t_f1),
                       "precision_lower_bound": fc.precision(t_f1), "recall": fc.recall(t_f1),
                       "recall_by_group_all_videos": {g: fc.group_recall(t_all, g) for g in fc.GROUPS},
                       "admissible": ok, "not_admissible_because": why})

    best = {arm: fc.choose([p for p in points if p["arm"] == arm]) for arm in fc.ARMS}
    control = fc.choose([{**p, "admissible": True} for p in points if p["arm"] == "control"])
    arm = fc.choose_arm(best)
    chosen = best[arm]
    r = fc.rule(chosen["threshold"], fc.finder_of(arm, chosen["param"]))
    print(f"baseline (V6 rule, threshold {base['threshold']:.2f}): training F1 {_f(fc.f1(base_f1))}")
    for k in fc.ARMS:
        b = best[k]
        print(f"{k}: {fc.PARAM[k]} {b['param']}, threshold {b['threshold']}: training F1 {_f(b['f1'])}"
              + ("" if b["admissible"] else "  (NO admissible point; highest F1 taken)"))
    print(f"control (pooled, threshold re-tuned {control['threshold']}): training F1 {_f(control['f1'])}")
    print(f"CHOSEN: {arm}")

    # Checked again just before anything is written: a held-out run that started while this ran must
    # not have its rule file replaced under it (it also reads only a frozen copy).
    refuse_if_started()
    a.rule_out.parent.mkdir(parents=True, exist_ok=True)
    a.rule_out.write_text(json.dumps({
        "chosen": {**{k: r[k] for k in ("variant", "threshold", "min_gap_s")}, "finder": r["finder"],
                   "train": {k: chosen[k] for k in ("f1", "precision_lower_bound", "recall", "admissible")}},
        "tuned_on": list(loaded), "gate": str(GATE), "gate_commit": commit,
        "note": "tuned on training videos only; no held-out score yet (eval_fast_combinations.py)"},
        indent=1, allow_nan=False), encoding="utf-8")
    report = {
        "gate": str(GATE), "gate_commit": commit, "head": _git("rev-parse", "HEAD"),
        "held_out_videos_at_tuning": list(bdata.HELD_OUT_VIDEOS),
        "baseline_rule": base, "baseline_rule_sha256": sha256(a.baseline),
        "videos": checks, "tuned_on": list(loaded), "f1_videos": f1_videos, "excluded": excluded,
        "baseline_training": {"f1_videos": fc.summary(base_f1), "all_videos": fc.summary(base_all),
                              "per_video": {v: fc.summary(c.sum(axis=0)) for v, c in base_c.items()}},
        "best_per_arm": best, "control": control, "chosen_arm": arm, "chosen_rule": r,
        "rule_file": str(a.rule_out), "rule_file_sha256": sha256(a.rule_out),
        "grid": points}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
    print(f"-> {a.out}, {a.rule_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
