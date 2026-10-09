"""Name each punch find_punches.py found: punch type, lead or rear, and confidence.

  python classify_punches.py --pose results/ingest/IMG_4406.pose.npz \\
      --punches results/punches/IMG_4406.punches.csv \\
      --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox
  ... --labels data/own/IMG_4406.labels.json     # also score against your labels

Writes results/punches/<clip>.classified.csv: the finder's rows, plus the hand as
lead/rear (from --stance, which is required: the tool does not guess a stance),
the punch type, the classifier's confidence (its softmax probability), and
whether the type names the same hand the finder did. A row the classifier cannot
see well enough is kept and marked unclassified, with the reason; it is never
given a guessed type.

The classifier was trained on AlphaPose skeletons from YouTube footage at 24-30
fps. On YOLO11-pose skeletons of phone footage its accuracy is UNMEASURED until
the pose-switch measurement exists (checklist W7). The hand-agreement line below
is a consistency check between two readings of the same skeleton, not accuracy.
With --labels, accuracy is measured: on the finder's matched punches, and on
every labelled punch as if the finder had been perfect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import classify, punchfind  # noqa: E402
from boxingvi.data import CLASSES  # noqa: E402
from boxingvi.lsimport import UNCLEAR_LABEL  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from find_punches import DEFAULT_RULE, TOL_S, load_labels, load_rule  # noqa: E402

PUNCH_HEADER = ["frame", "time_s", "hand", "peak_speed_bl_per_s", "peak_on_interpolated_frame"]
OUT_HEADER = PUNCH_HEADER + ["lead_rear", "punch_type", "confidence", "type_hand", "hands_agree",
                             "window_frames_observed", "note"]
DEFAULT_OUT = Path("results/punches")
MAX_ROWS = 100_000
_DIGITS = re.compile(r"[0-9]+")


def read_punches(path: Path, n_frames: int, fps: float) -> "list[dict]":
    """find_punches.py's CSV, checked again: it sits on disk between the two tools.
    Every frame must lie in this clip and every time must equal frame / this clip's
    fps. That catches a CSV from a clip at another frame rate or a longer clip; it
    cannot catch one from another clip of the same rate and length, which is why
    main() also requires the file names to name the same clip."""
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header != PUNCH_HEADER:
            raise ValueError(f"{path.name}: not a find_punches.py CSV (header {header!r})")
        rows = []
        for i, row in enumerate(r, start=2):
            if len(rows) >= MAX_ROWS:
                raise ValueError(f"{path.name}: more than {MAX_ROWS} rows")
            if len(row) != len(PUNCH_HEADER):
                raise ValueError(f"{path.name} line {i}: {len(row)} fields, expected {len(PUNCH_HEADER)}")
            d = dict(zip(PUNCH_HEADER, row))
            if not _DIGITS.fullmatch(d["frame"]) or not 0 <= int(d["frame"]) < n_frames:
                raise ValueError(f"{path.name} line {i}: frame {d['frame']!r} is not a frame of this clip")
            try:
                t = float(d["time_s"])
            except ValueError:
                raise ValueError(f"{path.name} line {i}: time {d['time_s']!r} is not a number") from None
            if not math.isfinite(t) or abs(t - int(d["frame"]) / fps) > 2e-3:
                raise ValueError(f"{path.name} line {i}: time {d['time_s']} does not match frame "
                                 f"{d['frame']} at {fps:.3f} fps: this CSV is not from this clip")
            if d["hand"] not in ("left", "right"):
                raise ValueError(f"{path.name} line {i}: hand {d['hand']!r}")
            if d["peak_on_interpolated_frame"] not in ("yes", "no"):
                raise ValueError(f"{path.name} line {i}: interpolated flag {d['peak_on_interpolated_frame']!r}")
            d["frame"] = int(d["frame"])
            rows.append(d)
    return rows


def span_peak(track, signal: np.ndarray, start: int, end: int) -> int:
    """The frame of the faster wrist's peak inside a labelled span: where the finder
    would have put this punch had it found it. The span's middle if no frame of
    the span has a signal."""
    lo, hi = np.searchsorted(track.frame_ids, [start, end + 1])
    if hi > lo:
        best = signal[lo:hi].max(axis=1)
        if best.max() > 0:
            return int(track.frame_ids[lo + int(np.argmax(best))])
    return (start + end) // 2


def type_scores(y_true: "list[int]", y_pred: "list[int]") -> dict:
    """Accuracy, macro F1 over the classes present in the labels, and the
    always-guess-the-commonest-labelled-type floor."""
    from sklearn.metrics import f1_score

    if not y_true:
        return {"n": 0}
    yt, yp = np.array(y_true), np.array(y_pred)
    present = sorted(set(y_true))
    major = np.bincount(yt).argmax()
    return {"n": len(yt), "accuracy": float((yt == yp).mean()),
            "macro_f1_present_classes": float(f1_score(yt, yp, labels=present, average="macro",
                                                       zero_division=0)),
            "majority_floor_accuracy": float((yt == major).mean()),
            "side_accuracy": float(np.mean([classify.class_side(CLASSES[a]) == classify.class_side(CLASSES[b])
                                            for a, b in zip(yt, yp)])),
            "classes_present": [CLASSES[c] for c in present]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--punches", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True, help="the manifest .json next to the checkpoint")
    ap.add_argument("--stance", choices=sorted(classify.STANCES), required=True)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--rule", type=Path, default=DEFAULT_RULE,
                    help="the rule find_punches.py ran with; its variant must be the window's")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    pose = PoseSequence.load(a.pose)
    clip = a.pose.name.split(".")[0]
    if a.punches.name.split(".")[0] != clip:
        raise ValueError(f"{a.punches.name} does not name the clip {clip}: classify the CSV "
                         "find_punches.py wrote for this pose")
    rows = read_punches(a.punches, len(pose), pose.fps)
    model, man = classify.load_classifier(a.classifier)
    geom: classify.WindowGeometry = man["geometry"]
    # The window's lead-in was measured with one wrist-speed variant; a peak found
    # with the other one is a different moment of the punch.
    if load_rule(a.rule)["variant"] != geom.variant:
        raise ValueError(f"the punch finder's rule uses another wrist-speed variant than the "
                         f"classifier's window ({geom.variant}); re-run one of them to match")

    # The same track the finder ran on, so every peak frame it reported exists here.
    raw = pose.to_track(usable_only=False)
    track = punchfind.bridge(raw, int(round(punchfind.BRIDGE_S * pose.fps)))
    torso = punchfind.body_scale(track)

    wins = [classify.cut_window(track, d["frame"], geom, torso) for d in rows]
    ok = [i for i, w in enumerate(wins) if w.keypoints is not None]
    probs = classify.predict_proba(model, [wins[i].keypoints for i in ok], man["t_out"])
    pred = {i: p for i, p in zip(ok, probs)}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{clip}.classified.csv"
    type_sides, finder_sides, counts = [], [], {c: 0 for c in CLASSES}
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(OUT_HEADER)
        for i, d in enumerate(rows):
            side = classify.hand_side(d["hand"], a.stance)
            base = [d[k] for k in PUNCH_HEADER] + [side]
            if i not in pred:
                w.writerow(base + ["unclassified", "", "", "", wins[i].observed, wins[i].reason])
                continue
            k = int(np.argmax(pred[i]))
            t_side = classify.class_side(CLASSES[k])
            type_sides.append(t_side)
            finder_sides.append(side)
            counts[CLASSES[k]] += 1
            w.writerow(base + [CLASSES[k], f"{pred[i][k]:.3f}", t_side,
                               "yes" if t_side == side else "no", wins[i].observed, ""])

    agree = classify.hand_agreement(type_sides, finder_sides)
    print(f"{clip}: {len(rows)} punches from the finder, {len(ok)} classified, "
          f"{len(rows) - len(ok)} unclassified (window: {geom.n_frames} frames at {geom.fps:g} fps)")
    print("  " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    if agree["n"]:
        print(f"  type and finder name the same hand on {agree['agree']:.1%} of {agree['n']} "
              f"(chance for these proportions {agree['chance']:.1%}); a consistency check, not accuracy")
    print(f"  -> {out}")

    if a.labels:
        spans = load_labels(a.labels, len(pose), pose.fps)
        named = [(j, s) for j, s in enumerate(spans) if s.label != UNCLEAR_LABEL]
        tol = int(round(TOL_S * pose.fps))
        pairs = punchfind.match([d["frame"] for d in rows], spans, tol)
        yt, yp, matched_unclassified = [], [], 0
        for si, di in pairs:
            if spans[si].label == UNCLEAR_LABEL:
                continue
            if di not in pred:
                matched_unclassified += 1
                continue
            yt.append(spans[si].label)
            yp.append(int(np.argmax(pred[di])))
        sig = punchfind.wrist_signal(track, geom.variant)
        pw = [classify.cut_window(track, span_peak(track, sig, s.start, s.end), geom, torso) for _, s in named]
        keep = [k for k, w_ in enumerate(pw) if w_.keypoints is not None]
        pp = classify.predict_proba(model, [pw[k].keypoints for k in keep], man["t_out"])
        result = {"clip": clip, "stance": a.stance, "tolerance_frames": tol,
                  "classifier_sha256": man["sha256"],
                  "matched_detections": type_scores(yt, yp),
                  "matched_unclassifiable": matched_unclassified,
                  "every_labelled_punch": type_scores([named[k][1].label for k in keep],
                                                      [int(np.argmax(p)) for p in pp]),
                  "labelled_unclassifiable": len(named) - len(keep)}
        m, e = result["matched_detections"], result["every_labelled_punch"]
        for name, s in (("matched detections", m), ("every labelled punch", e)):
            if s["n"]:
                print(f"  vs labels, {name}: type accuracy {s['accuracy']:.3f} on {s['n']} "
                      f"(commonest-type floor {s['majority_floor_accuracy']:.3f}), "
                      f"hand right {s['side_accuracy']:.3f}")
        (a.out_dir / f"{clip}.classify_score.json").write_text(
            json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
