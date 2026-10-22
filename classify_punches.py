"""Name each punch find_punches.py found: punch type, lead or rear, and confidence.

  python classify_punches.py --pose results/ingest/IMG_4406.pose.npz \\
      --punches results/punches/IMG_4406.punches.csv \\
      --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --stance orthodox
  ... --labels data/own/IMG_4406.labels.json     # also score against your labels

Writes results/punches/<clip>.classified.csv: the finder's rows, plus the hand as
lead/rear (from --stance, which is required: the tool does not guess a stance),
the punch type, the classifier's confidence (its softmax probability), whether
the type names the same hand the finder did, and the frames the punch was read
from. A row the classifier cannot see well enough is kept and marked
unclassified, with the reason; it is never given a guessed type. Beside it,
<clip>.classified.json records which classifier, rule, stance and cut produced it.

How a punch is cut (vault ADR-0014, amended 2026-10-13): every punch in a clip is
read over ONE length, the median of the punches' own extents in that clip, placed
around each punch's wrist-speed peak. Each extent comes from the width of the
punch's speed peak (punchfind.extent, results/punch_extent.json from
run_punch_extent.py). A clip with fewer than five punches has no tempo to read
and gets the training clips' median window. Sizing each punch from its own width
was worse on normal-tempo footage (V5 + V9: 0.774 against 0.826) and no better on
slow footage (V6: 0.608 against 0.631, a tie); the old fixed 0.375 s window
scored 0.806 and 0.500. The cut was chosen looking at those numbers, so they are
not a clean test of it: labelled footage it has not seen is.

The classifier was trained on AlphaPose skeletons from YouTube footage at 24-30
fps. On the benchmark's own held-out punches YOLO11-pose costs it nothing
measurable (0.896 against 0.889), but on phone footage its accuracy is UNMEASURED
until a clip is labelled. The hand-agreement line below is a consistency check
between two readings of the same skeleton, not accuracy. With --labels, accuracy
is measured three ways: on the finder's matched punches (the pipeline as it
runs); on every labelled punch cut by the clip's window at its own wrist-speed
peak (a perfect finder, the pipeline's cut); and on every labelled punch read
from its labelled frames (a perfect finder AND perfect start and end).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import classify, lsimport, punchfind  # noqa: E402
from boxingvi.data import CLASSES  # noqa: E402
from boxingvi.lsimport import UNCLEAR_LABEL  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from find_punches import DEFAULT_RULE, TOL_S, load_labels, load_rule  # noqa: E402

PUNCH_HEADER = ["frame", "time_s", "hand", "peak_speed_bl_per_s", "peak_on_interpolated_frame"]
OUT_HEADER = PUNCH_HEADER + ["lead_rear", "punch_type", "confidence", "type_hand", "hands_agree",
                             "punch_start_frame", "punch_end_frame", "window_frames_observed", "note"]
DEFAULT_OUT = Path("results/punches")
DEFAULT_EXTENT = Path("results/punch_extent.json")
MAX_ROWS = 100_000
_DIGITS = re.compile(r"[0-9]+")


def finder_per_hand(punches: Path, clip: str, rule: dict) -> "bool | None":
    """Whether find_punches.py made this CSV with --per-hand, from the <clip>.punches.json
    it wrote beside it. None (unknown) when that file is missing, unreadable, for another
    clip or another rule than this run's, or describes other bytes than the CSV now holds:
    it is recorded, never trusted for anything else, so a stale or hand-made one must not
    lend the CSV a provenance."""
    side = punches.with_suffix(".json")
    try:
        if not side.is_file() or side.stat().st_size > 64_000:
            return None
        d = json.loads(side.read_text(encoding="utf-8"), parse_constant=lsimport._no_constant)
    except (OSError, ValueError, RecursionError):   # RecursionError: deeply nested JSON, as lsimport
        return None
    if (not isinstance(d, dict) or d.get("clip") != clip or d.get("rule") != rule
            or not isinstance(d.get("per_hand"), bool)
            or d.get("csv_sha256") != hashlib.sha256(punches.read_bytes()).hexdigest()):
        return None
    return d["per_hand"]


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
    ap.add_argument("--extent", type=Path, default=DEFAULT_EXTENT,
                    help="the punch-extent rule from run_punch_extent.py")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--unclear", choices=("punch", "not-a-punch"), default="punch",
                    help="with --labels: what an 'unclear' label means: a punch nobody could name (the labelling "
                         "guide), or a motion that is not a punch, such as framing or a feint")
    a = ap.parse_args(argv)

    pose = PoseSequence.load(a.pose)
    clip = a.pose.name.split(".")[0]
    if a.punches.name.split(".")[0] != clip:
        raise ValueError(f"{a.punches.name} does not name the clip {clip}: classify the CSV "
                         "find_punches.py wrote for this pose")
    rows = read_punches(a.punches, len(pose), pose.fps)
    if a.labels and not rows:   # before anything is written: a stale score file must not outlive this run
        raise ValueError(f"{a.punches.name} holds no punch, so there is nothing to score against labels")
    model, man = classify.load_classifier(a.classifier)
    geom: classify.WindowGeometry = man["geometry"]
    # The window's lead-in was measured with one wrist-speed variant; a peak found
    # with the other one is a different moment of the punch.
    rule = load_rule(a.rule)
    if rule["variant"] != geom.variant:
        raise ValueError(f"the punch finder's rule uses another wrist-speed variant than the "
                         f"classifier's window ({geom.variant}); re-run one of them to match")
    ex = classify.read_extent(a.extent, geom.variant)
    per_hand = finder_per_hand(a.punches, clip, rule)

    # The same track the finder ran on, so every peak frame it reported exists here.
    raw = pose.to_track(usable_only=False)
    track = punchfind.bridge(raw, int(round(punchfind.BRIDGE_S * pose.fps)))
    torso = punchfind.body_scale(track)
    sig = punchfind.wrist_signal(track, geom.variant)

    row_of = {int(f): i for i, f in enumerate(track.frame_ids)}
    peaks = []
    for d in rows:
        hand = 0 if d["hand"] == "left" else 1
        i = row_of.get(d["frame"])
        # The finder reported a speed peak of this hand at this frame, above the rule's threshold:
        # if this pose has no frame there, or that hand is not at a peak above it there, the CSV
        # was not made from this pose with this rule. (The other hand may be faster: the per-hand
        # finder reports each hand's own peaks.)
        h_ = sig[:, hand]
        if (i is None or not h_[i] >= rule["threshold"]
                or (i > 0 and h_[i - 1] > h_[i]) or (i + 1 < len(h_) and h_[i + 1] > h_[i])):
            raise ValueError(f"{a.punches.name}: frame {d['frame']} is not a {d['hand']}-hand speed peak in "
                             f"this pose; it was not made from {a.pose.name} with this rule")
        peaks.append((d["frame"], hand))
    # One length for the whole clip, from its punches' own extents; each punch read around its own peak.
    cut = classify.clip_cut(track, sig, peaks, ex, geom)
    spans_read = [classify.clip_window(p, cut, pose.fps) for p, _ in peaks]
    wins = [classify.cut_span(track, s_, e_, geom, torso) for s_, e_ in spans_read]
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
            span = list(spans_read[i])
            if i not in pred:
                w.writerow(base + ["unclassified", "", "", ""] + span + [wins[i].observed, wins[i].reason])
                continue
            k = int(np.argmax(pred[i]))
            t_side = classify.class_side(CLASSES[k])
            type_sides.append(t_side)
            finder_sides.append(side)
            counts[CLASSES[k]] += 1
            w.writerow(base + [CLASSES[k], f"{pred[i][k]:.3f}", t_side,
                               "yes" if t_side == side else "no"] + span + [wins[i].observed, ""])

    agree = classify.hand_agreement(type_sides, finder_sides)
    print(f"{clip}: {len(rows)} punches from the finder, {len(ok)} classified, {len(rows) - len(ok)} unclassified")
    if cut is not None and cut.source == "clip":
        print(f"  every punch read over {cut.length_s:.2f} s, the clip's median punch length (from {cut.n_used} "
              f"punches; {cut.n_left_out} left out, their speed peak cut off by a wrist dropout"
              + ("; capped at what the classifier can be shown)" if cut.capped else ")"))
    elif cut is not None:
        print(f"  under {classify.MIN_CLIP_PUNCHES} punches: too few to read this clip's tempo, so every punch is "
              f"read over {cut.length_s:.2f} s, the training clips' median")
    print("  " + ", ".join(f"{c} {n}" for c, n in counts.items()))
    if agree["n"]:
        print(f"  type and finder name the same hand on {agree['agree']:.1%} of {agree['n']} "
              f"(chance for these proportions {agree['chance']:.1%}); a consistency check, not accuracy")
    # Provenance for whatever reads the CSV next (report_punches.py): which model named
    # these punches, from which finder rule, extent rule, stance and cut, and the CSV's own hash.
    (a.out_dir / f"{clip}.classified.json").write_text(json.dumps({
        "clip": clip, "csv_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "classifier_sha256": man["sha256"], "stance": a.stance, "rule": rule, "finder_per_hand": per_hand,
        "extent": ex,
        "cut": None if cut is None else {"per": "clip", **cut._asdict()},
        "pose_frames": len(pose), "fps": pose.fps}, indent=1, allow_nan=False), encoding="utf-8")
    print(f"  -> {out} (+ .json provenance)")

    if a.labels:
        labelled = load_labels(a.labels, len(pose), pose.fps)
        tol = int(round(TOL_S * pose.fps))
        point_tol = int(round(punchfind.POINT_TOL_S * pose.fps))
        # A single-frame label is a point inside a punch (punchfind.POINT_TOL_S); an 'unclear'
        # label is either an unnamed punch or, with --unclear not-a-punch, framing or a feint.
        framing = [x for x in labelled if a.unclear == "not-a-punch" and x.label == UNCLEAR_LABEL]
        spans = [x for x in labelled if x not in framing]
        n_points = sum(x.start == x.end for x in spans)       # point labels among the punches
        drags = [x for x in spans if x.start != x.end]
        named = [(j, x) for j, x in enumerate(spans) if x.label != UNCLEAR_LABEL]
        pairs = punchfind.match([d["frame"] for d in rows], punchfind.for_matching(spans, tol, point_tol), tol)
        yt, yp, matched_unclassified = [], [], 0
        for si, di in pairs:
            if spans[si].label == UNCLEAR_LABEL:
                continue
            if di not in pred:
                matched_unclassified += 1
                continue
            yt.append(spans[si].label)
            yp.append(int(np.argmax(pred[di])))
        # Framing marks that drew a detection: bounds that hold whichever punch matching is chosen
        # (punchfind.framing_bounds). Named: the detections the lower bound's matching counted, each
        # out of every punch's reach and on a framing mark (others like them may exist).
        fb = (punchfind.framing_bounds([d["frame"] for d in rows], spans, framing, tol, point_tol)
              if framing else None)

        def scored(windows) -> "tuple[dict, int]":
            keep = [k for k, w_ in enumerate(windows) if w_.keypoints is not None]
            pp = classify.predict_proba(model, [windows[k].keypoints for k in keep], man["t_out"])
            return (type_scores([named[k][1].label for k in keep], [int(np.argmax(p)) for p in pp]),
                    len(named) - len(keep))

        # Every labelled punch, two ways. From its labelled frames: perfect finder AND perfect start
        # and end (not possible for a point label: it has no start or end). By the clip's window at
        # the wrist-speed peak inside the punch (for a point label, within POINT_TOL_S of it):
        # a perfect finder, the pipeline's cut.
        by_label, by_window = [], []
        for _, x in named:
            if x.start == x.end:
                by_label.append(classify.Window(None, 0, "a point label has no start and end to read"))
                lo_, hi_ = x.start - point_tol, x.end + point_tol
            else:
                by_label.append(classify.cut_span(track, x.start, x.end, geom, torso))
                lo_, hi_ = x.start, x.end
            pk = span_peak(track, sig, lo_, hi_)
            i = row_of.get(pk)
            if i is None or not sig[i].max() > 0:   # span_peak fell back to the middle: there is no peak
                by_window.append(classify.Window(None, 0, "no wrist speed inside the labelled punch"))
            else:
                by_window.append(classify.cut_span(track, *classify.clip_window(pk, cut, pose.fps), geom, torso))
        ious = [punchfind.span_iou(spans_read[di], (spans[si].start, spans[si].end))
                for si, di in pairs if spans[si].start != spans[si].end]
        (label_scores, label_unc), (window_scores, window_unc) = scored(by_label), scored(by_window)
        found = len(pairs)
        result = {"clip": clip, "stance": a.stance, "tolerance_frames": tol, "point_labels": n_points,
                  "point_tolerance_frames": point_tol, "unclear_means": a.unclear,
                  "classifier_sha256": man["sha256"], "cut": {"per": "clip", **cut._asdict()},
                  "finder": {"per_hand": per_hand, "labelled_punches": len(spans), "detections": len(rows),
                             "matched": found,
                             "recall": found / len(spans) if spans else None,
                             "precision": found / len(rows) if rows else None},
                  "matched_detections": type_scores(yt, yp),
                  "matched_unclassifiable": matched_unclassified,
                  "labelled_punch_by_clip_window_at_its_peak": window_scores,
                  "labelled_punch_by_clip_window_unclassifiable": window_unc,
                  "labelled_punch_from_labelled_frames": label_scores,
                  "labelled_punch_from_labelled_frames_unclassifiable": label_unc,
                  "window_iou_with_labels": {"n": len(ious), "mean": float(np.mean(ious)) if ious else None},
                  "labelled_length_s_median": (float(np.median([(x.end - x.start) / pose.fps for x in drags]))
                                               if drags else None)}
        if fb:
            types = [CLASSES[int(np.argmax(pred[k]))] if k in pred else "unclassified" for k in fb["lower_set"]]
            result["framing"] = {"marks": fb["marks"], "flagged_as_punches_at_least": fb["at_least"],
                                 "flagged_as_punches_at_most": fb["at_most"],
                                 "lower_bound_named_as": {t: types.count(t) for t in sorted(set(types))}}
        print(f"  vs labels: the finder found {found} of {len(spans)} labelled punches, and {found} of its "
              f"{len(rows)} detections were real" + (f" ({n_points} point labels, matched within +-{point_tol} frames)"
                                                     if n_points else ""))
        for name, sc in (("matched detections", result["matched_detections"]),
                         ("every labelled punch, cut by the clip's window at its peak", window_scores),
                         ("every labelled punch, from its labelled frames", label_scores)):
            if sc["n"]:
                print(f"  vs labels, {name}: type accuracy {sc['accuracy']:.3f} on {sc['n']} "
                      f"(commonest-type floor {sc['majority_floor_accuracy']:.3f}), "
                      f"hand right {sc['side_accuracy']:.3f}")
        if fb:
            fr = result["framing"]
            lo_, hi_ = fr["flagged_as_punches_at_least"], fr["flagged_as_punches_at_most"]
            print(f"  framing marks: {lo_ if lo_ == hi_ else f'{lo_} to {hi_}'} of {fr['marks']} were flagged as "
                  "punches; the detections the lower bound counted were named "
                  + (", ".join(f"{t} {n}" for t, n in fr["lower_bound_named_as"].items()) or "nothing"))
        if result["labelled_length_s_median"] is not None:
            print(f"  the clip's window is {cut.length_s:.2f} s; your labelled punches are a median "
                  f"{result['labelled_length_s_median']:.2f} s")
        (a.out_dir / f"{clip}.classify_score.json").write_text(
            json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
