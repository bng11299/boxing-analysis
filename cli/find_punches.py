"""Find punches in one clip with the wrist-speed rule, per hand, dropouts bridged.

  python cli/find_punches.py --pose results/ingest/IMG_4406.pose.npz
  python cli/find_punches.py --pose results/ingest/IMG_4406.pose.npz --labels data/own/IMG_4406.labels.json

Writes results/punches/<clip>.punches.csv: one row per detected punch (video
frame, seconds, hand, peak wrist speed in body lengths per second). Hands are
the BOXER's left and right (COCO), not lead and rear: that needs the stance.

The rule (variant, threshold, minimum gap) is the one tuned on V6 and recorded
in results/punch_finder_v6.json. V6 is AlphaPose keypoints at 24 fps of hooks
and uppercuts by one athlete, so on your footage this is a first look, not a
tuned detector. With --labels it is scored against them, next to the chance
floor of its own detections shifted in time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import lsimport, punchfind  # noqa: E402
from boxingvi.data import CLASSES, Span  # noqa: E402
from boxingvi.lsimport import UNCLEAR_LABEL  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402

TOL_S = 0.1                     # the matching tolerance used on V6
MIN_SHIFT_S = 10.0
DEFAULT_RULE = Path("results/punch_finder_v6.json")
DEFAULT_OUT = Path("results/punches")


def load_rule(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"no tuned rule at {path}: run cli/run_punch_finder.py on V6 first")
    c = json.loads(path.read_text(encoding="utf-8"))["chosen"]
    if c["variant"] not in punchfind.VARIANTS:
        raise ValueError(f"{path}: unknown rule variant {c['variant']!r}")
    rule = {"variant": c["variant"], "threshold": float(c["threshold"]), "min_gap_s": float(c["min_gap_s"])}
    # An optional `finder` block names an arm of docs/gates/fast-combinations.md. Without one the
    # rule is the pooled rule and this dict is exactly what it always was (sidecars compare it).
    if "finder" in c:
        try:
            rule["finder"] = punchfind.parse_finder(c["finder"], rule["min_gap_s"])
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
    return rule


def _frame(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"labels: {what} is not a whole frame number: {v!r}")
    return v


def load_labels(path: Path, n_frames: int, fps: float) -> "list[Span]":
    """Ground truth written by import_labels.py, checked again here: the file sits on
    disk between the two, and a hand-edited or stale one must not be scored as truth."""
    lab = json.loads(path.read_text(encoding="utf-8"), parse_constant=lsimport._no_constant)
    fps_l = lab.get("fps")
    if (lab.get("n_frames") != n_frames or isinstance(fps_l, bool) or not isinstance(fps_l, (int, float))
            or abs(fps_l - fps) > 1e-3):
        raise ValueError(f"{path.name} is for a {lab.get('n_frames')}-frame clip at {fps_l} fps; "
                         f"this pose has {n_frames} frames at {fps:.3f}: not the same video")
    names = {lsimport.UNCLEAR: UNCLEAR_LABEL} | {c: i for i, c in enumerate(CLASSES)}
    spans = []
    for s in lab.get("spans") or []:
        if not isinstance(s, dict) or s.get("label") not in names:
            raise ValueError(f"labels: unknown label in {s!r}")
        a, b = _frame(s.get("start"), "start"), _frame(s.get("end"), "end")
        if not 0 <= a <= b < n_frames:
            raise ValueError(f"labels: {s['label']} at {a}-{b} is not inside the clip (0-{n_frames - 1})")
        spans.append(Span(a, b, names[s["label"]]))
    if not spans:
        raise ValueError(f"{path.name} has no labelled punches to score against")
    return spans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-hand", action="store_true",
                    help="each hand's own peaks, the minimum gap applied within a hand (default: the pooled rule)")
    ap.add_argument("--unclear", choices=("punch", "not-a-punch"), default="punch",
                    help="what an 'unclear' label means: a punch nobody could name (the labelling guide), or a "
                         "motion that is not a punch, such as framing or a feint")
    a = ap.parse_args(argv)

    rule = load_rule(a.rule)
    if a.per_hand and "finder" in rule:
        raise ValueError(f"--per-hand cannot be combined with a rule that names its own finder "
                         f"({rule['finder']['mode']}, in {a.rule.name})")
    per_hand_out = a.per_hand or punchfind.rule_per_hand(rule)
    pose = PoseSequence.load(a.pose)
    clip = a.pose.name.split(".")[0]
    raw = pose.to_track(usable_only=False)            # every detected frame, with confidence
    track = punchfind.bridge(raw, int(round(punchfind.BRIDGE_S * pose.fps)))
    sig = punchfind.wrist_signal(track, rule["variant"])
    if "finder" in rule:
        det, hands = punchfind.detect_rule(track, sig, rule)
    else:
        det, hands = punchfind.detect(track, sig, rule["threshold"], rule["min_gap_s"], per_hand=a.per_hand)
    row = {int(f): i for i, f in enumerate(track.frame_ids)}
    observed = set(int(f) for f in raw.frame_ids)
    valid = punchfind.hand_valid(track)

    # One denominator for all three: every frame of the clip.
    whole = (0, len(pose) - 1)
    old = float(pose.usable().sum()) / len(pose)
    per_hand = punchfind.signal_coverage(raw, bridge_s=0.0, span=whole)
    bridged = punchfind.signal_coverage(track, span=whole)
    print(f"{clip}: {len(pose)} frames @ {pose.fps:.3f} fps; rule {rule['variant']} > "
          f"{rule['threshold']:.2f} body lengths/s, min gap {rule['min_gap_s']} s"
          + (f", finder {rule['finder']}" if "finder" in rule else " (tuned on V6)"))
    print(f"  share of all {len(pose)} frames with a speed signal: both-wrists frame rule {old:.1%} | "
          f"per hand L {per_hand['left']:.1%} R {per_hand['right']:.1%} | "
          f"per hand + dropouts bridged L {bridged['left']:.1%} R {bridged['right']:.1%}")
    print(f"  {len(det)} punches found: left {int((hands == 0).sum())}, right {int((hands == 1).sum())}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{clip}.punches.csv"
    n_interp = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "hand", "peak_speed_bl_per_s", "peak_on_interpolated_frame"])
        for fr, h in zip(det, hands):
            i = row[int(fr)]
            interp = int(fr) not in observed or not valid[i, h]   # bridged frame, or a filled hole in this hand
            n_interp += interp
            w.writerow([int(fr), f"{fr / pose.fps:.3f}", "left" if h == 0 else "right",
                        f"{sig[i, h]:.3f}", "yes" if interp else "no"])
    # Which finder made the list, beside it rather than in it: the CSV's header is read downstream.
    (a.out_dir / f"{clip}.punches.json").write_text(json.dumps({
        "clip": clip, "csv_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "rule": rule,
        "per_hand": per_hand_out}, indent=1, allow_nan=False), encoding="utf-8")
    print(f"  -> {out} (+ .json: the rule, pooled or per hand)"
          + (f"  ({n_interp} peak(s) on an interpolated frame, flagged)" if n_interp else ""))

    if a.labels:
        labelled = load_labels(a.labels, len(pose), pose.fps)
        tol = int(round(TOL_S * pose.fps))
        point_tol = int(round(punchfind.POINT_TOL_S * pose.fps))
        # Framing marks, when 'unclear' means not a punch, are scored apart: a detection on one is a false positive.
        not_punch = [s_ for s_ in labelled if a.unclear == "not-a-punch" and s_.label == UNCLEAR_LABEL]
        spans = punchfind.for_matching([s_ for s_ in labelled if s_ not in not_punch], tol, point_tol)
        n_points = sum(s_.start == s_.end for s_ in labelled if s_ not in not_punch)   # among the punches
        s = punchfind.score(det, spans, tol, whole)
        if not spans:
            # Every label a framing mark: there is no punch to find, so recall, F1 and the count
            # ratio are undefined, not 0 (score() writes NaN, which JSON cannot hold). Precision
            # stays where there are detections (none is a labelled punch); with none it is 0/0.
            s = s | {"recall": None, "f1": None, "count_ratio": None,
                     "precision": s["precision"] if s["n_detected"] else None}
        result = {"clip": clip, "rule": rule, "per_hand": per_hand_out, "tolerance_frames": tol,
                  "point_labels": n_points, "point_tolerance_frames": point_tol, "unclear_means": a.unclear,
                  "score": s}
        if not_punch:
            # Bounds that hold whichever punch matching is chosen (punchfind.framing_bounds).
            fb = punchfind.framing_bounds(det, [s_ for s_ in labelled if s_ not in not_punch], not_punch,
                                          tol, point_tol)
            result["framing"] = {"marks": fb["marks"], "flagged_as_punches_at_least": fb["at_least"],
                                 "flagged_as_punches_at_most": fb["at_most"],
                                 "lower_bound_frames": [int(det[i]) for i in fb["lower_set"]]}
        min_shift = int(round(MIN_SHIFT_S * pose.fps))
        if not spans:
            floor = "no chance floor: no punches labelled"
        elif len(pose) >= 2 * min_shift + 2:
            fl = punchfind.shift_floor(det, spans, tol, whole, min_shift)
            result["shift_floor"] = fl
            floor = f"chance floor F1 {fl['f1_mean']:.3f} (97.5th pct {fl['f1_p97_5']:.3f})"
        else:  # too few distinct shifts for a percentile to mean anything
            floor = f"no chance floor: the clip is under {2 * MIN_SHIFT_S:.0f} s"
        within = (f"+-{tol} frames" + (f"; the {n_points} point labels +-{point_tol}" if n_points else ""))
        if spans:
            print(f"  vs {s['n_labelled']} labelled punches ({within}): precision {s['precision']:.3f} "
                  f"recall {s['recall']:.3f} F1 {s['f1']:.3f}; {floor}")
        else:
            print(f"  no punches labelled: none of the {s['n_detected']} detections can be a labelled punch; {floor}")
        if "framing" in result:
            fr = result["framing"]
            lo_, hi_ = fr["flagged_as_punches_at_least"], fr["flagged_as_punches_at_most"]
            print(f"  {lo_ if lo_ == hi_ else f'{lo_} to {hi_}'} of {fr['marks']} framing marks were flagged as "
                  "punches (detections that matched no punch count as false positives above)")
        (a.out_dir / f"{clip}.score.json").write_text(json.dumps(result, indent=1, allow_nan=False),
                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
