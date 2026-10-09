"""Does cutting windows out of continuous pose cost the classifier accuracy? Measured on V6.

    python eval_classify_v6.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json

V6 is the only BoxingVI video that ships as continuous pose with labelled punches,
and the classifier never trained on it (V6 is excluded from the corpus). So it can
separate the cost of the classify step's windowing from everything else. Three
readings of the same labelled punches, by the same saved classifier:

  A  corpus clips      each span cut exactly as the corpus loader cuts V6
                       (data._load_v6): the span's frames, normalised by 1920x1080
  B  span windows      classify.cut_window anchored at the wrist-speed peak inside
                       the span: the pipeline's window with a perfect finder
  C  finder windows    classify.cut_window at the finder's own detections that
                       match a span: the pipeline as it runs, on matched punches

A - B is the cost of the window (fixed length, peak-anchored, resampled, scaled
and centred) given perfect timing; B - C adds the finder's timing. Paired
bootstrap intervals on the punches all three can read.

Limits, stated in the results: V6 has only hooks and uppercuts (no jabs or
crosses), one athlete, and the same channel as training video V1, so its numbers
may be optimistic about a new athlete (ISSUES-LOG O55). Its skeletons are
AlphaPose, as in training: nothing here measures the YOLO11-pose switch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import classify, punchfind  # noqa: E402
from boxingvi.data import (CLASSES, SOURCE_SIZE, _with_validity_channel, load_v6_spans,  # noqa: E402
                           load_v6_track, pad_or_trim)
from boxingvi.stats import bootstrap_gap_ci  # noqa: E402
from classify_punches import span_peak, type_scores  # noqa: E402

TOL_S = 0.1   # the finder's matching tolerance on V6
DEFAULT_RULE = Path("results/punch_finder_v6.json")
DEFAULT_OUT = Path("results/classify_v6.json")


def corpus_clip(track, s) -> "np.ndarray | None":
    """The span exactly as data._load_v6 builds a V6 clip."""
    lo, hi = np.searchsorted(track.frame_ids, [s.start, s.end + 1])
    if hi - lo < 2:
        return None
    return pad_or_trim(_with_validity_channel((track.xy[lo:hi] / np.array(SOURCE_SIZE)).astype(np.float32)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder holding Skeleton_data/ and Annotation_files/")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    model, man = classify.load_classifier(a.classifier)
    geom: classify.WindowGeometry = man["geometry"]
    rule = json.loads(a.rule.read_text(encoding="utf-8"))["chosen"]
    if rule["variant"] != geom.variant:
        raise SystemExit(f"the rule's variant {rule['variant']} is not the one the window was measured with")

    track = load_v6_track(a.data / "Skeleton_data")
    spans = load_v6_spans(a.data / "Annotation_files")
    track = punchfind.bridge(track, int(round(punchfind.BRIDGE_S * track.fps)))
    torso = punchfind.body_scale(track)
    sig = punchfind.wrist_signal(track, geom.variant)
    det, _ = punchfind.detect(track, sig, rule["threshold"], rule["min_gap_s"])
    pairs = dict(punchfind.match(det, spans, int(round(TOL_S * track.fps))))

    readings = {"A": {}, "B": {}, "C": {}}
    for j, s in enumerate(spans):
        c = corpus_clip(track, s)
        if c is not None:
            readings["A"][j] = c
        w = classify.cut_window(track, span_peak(track, sig, s.start, s.end), geom, torso)
        if w.keypoints is not None:
            readings["B"][j] = w.keypoints
        if j in pairs:
            w = classify.cut_window(track, int(det[pairs[j]]), geom, torso)
            if w.keypoints is not None:
                readings["C"][j] = w.keypoints

    preds = {}
    for k, r in readings.items():
        idx = sorted(r)
        # The corpus clips are 25 frames padded; windows are geom.n_frames. Each batch is one length.
        p = classify.predict_proba(model, [r[i] for i in idx], man["t_out"])
        preds[k] = {i: int(np.argmax(q)) for i, q in zip(idx, p)}

    y = {j: s.label for j, s in enumerate(spans)}
    result = {"video": "V6", "n_labelled": len(spans), "classifier_sha256": man["sha256"],
              "geometry": man["geometry"].__dict__, "finder_matched": len(pairs),
              "readings": {}, "paired": {}, "limits": [
                  "V6 has hooks and uppercuts only: no jabs or crosses",
                  "one athlete, same channel as training video V1 (O55): may be optimistic",
                  "AlphaPose skeletons as in training: says nothing about the YOLO11-pose switch",
                  "the finder's rule was tuned on V6's first half; C covers both halves"]}
    for k, p in preds.items():
        idx = sorted(p)
        s = type_scores([y[i] for i in idx], [p[i] for i in idx])
        s["predicted_counts"] = {c: int(sum(p[i] == n for i in idx)) for n, c in enumerate(CLASSES)}
        result["readings"][k] = s
        print(f"{k}: {s['n']} punches, accuracy {s['accuracy']:.3f}, macro F1 (present) "
              f"{s['macro_f1_present_classes']:.3f}, hand right {s['side_accuracy']:.3f}, "
              f"commonest-type floor {s['majority_floor_accuracy']:.3f}")
    for a_, b_ in (("A", "B"), ("B", "C"), ("A", "C")):
        common = sorted(set(preds[a_]) & set(preds[b_]))
        yt = np.array([y[i] for i in common])
        pa, pb = np.array([preds[a_][i] for i in common]), np.array([preds[b_][i] for i in common])
        ci = bootstrap_gap_ci(yt, pa, pb, metric="accuracy")
        result["paired"][f"{a_}_minus_{b_}"] = {"n": len(common), **ci}
        print(f"{a_} - {b_} accuracy on the {len(common)} both can read: {ci['point']:+.3f} "
              f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
