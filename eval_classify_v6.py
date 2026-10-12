"""What does cutting punches out of continuous pose cost the classifier? Measured on V6.

    python eval_classify_v6.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json \\
        --extent results/punch_extent.json

V6 is the only BoxingVI video that ships as continuous pose with labelled punches,
and the classifier never trained on it (V6 is excluded from the corpus). So it can
separate the cost of cutting the punch out from everything else. Five readings of
the same labelled punches, by the same saved classifier:

  A  corpus clips        each span cut exactly as the corpus loader cuts V6
                         (data._load_v6): the span's frames, normalised by 1920x1080
  B  fixed, at the peak  classify.cut_window at the wrist-speed peak inside the span:
                         the fixed-length window with a perfect finder (the baseline)
  C  fixed, at finder    cut_window at the finder's own detections that match a span
  D  extent, at the peak punchfind.extent + classify.cut_span at the span's peak:
                         the pipeline's cut with a perfect finder
  E  extent, at finder   the same at the finder's matched detections: the pipeline
                         as it runs, on the punches it found
  F  long fixed, at peak a CONTROL: one fixed window for every punch, as long as the
                         median extent and split before and after the peak as the
                         extent rule splits it. D - F is what following each punch's
                         own tempo buys beyond simply using a longer window

A - D is what the pipeline's cut still costs; D - B is what the extent bought over
the fixed window; D - E adds the finder's timing. Reported on all of V6 and on its
second half alone, because the extent rule's width level and the choice of rule
were made looking at the first half (run_punch_extent.py); the second half is the
clean number. Paired bootstrap intervals on the punches both readings can read.

Limits, stated in the results: V6 has only hooks and uppercuts (no jabs or
crosses), one athlete, and the same channel as training video V1, so its numbers
may be optimistic about a new athlete (ISSUES-LOG O55). Its skeletons are
AlphaPose, as in training: nothing here measures the YOLO11-pose switch.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
DEFAULT_EXTENT = Path("results/punch_extent.json")
DEFAULT_OUT = Path("results/classify_v6.json")
PAIRS = (("A", "B"), ("A", "D"), ("D", "B"), ("D", "F"), ("D", "E"), ("E", "C"))


def corpus_clip(track, s) -> "np.ndarray | None":
    """The span exactly as data._load_v6 builds a V6 clip."""
    lo, hi = np.searchsorted(track.frame_ids, [s.start, s.end + 1])
    if hi - lo < 2:
        return None
    return pad_or_trim(_with_validity_channel((track.xy[lo:hi] / np.array(SOURCE_SIZE)).astype(np.float32)))


def block(preds: dict, y: dict, keep) -> dict:
    """Scores and paired gaps over the spans in `keep`."""
    out = {"readings": {}, "paired": {}}
    for k, p in preds.items():
        idx = sorted(set(p) & keep)
        if not idx:
            out["readings"][k] = {"n": 0}
            continue
        s = type_scores([y[i] for i in idx], [p[i] for i in idx])
        s["predicted_counts"] = {c: int(sum(p[i] == n for i in idx)) for n, c in enumerate(CLASSES)}
        out["readings"][k] = s
    for a_, b_ in PAIRS:
        common = sorted(set(preds[a_]) & set(preds[b_]) & keep)
        if len(common) < 2:
            out["paired"][f"{a_}_minus_{b_}"] = {"n": len(common)}
            continue
        yt = np.array([y[i] for i in common])
        pa, pb = np.array([preds[a_][i] for i in common]), np.array([preds[b_][i] for i in common])
        out["paired"][f"{a_}_minus_{b_}"] = {"n": len(common), **bootstrap_gap_ci(yt, pa, pb, metric="accuracy")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder holding Skeleton_data/ and Annotation_files/")
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    ap.add_argument("--extent", type=Path, default=DEFAULT_EXTENT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    model, man = classify.load_classifier(a.classifier)
    geom: classify.WindowGeometry = man["geometry"]
    r = json.loads(a.rule.read_text(encoding="utf-8"))
    rule = r["chosen"]
    if rule["variant"] != geom.variant:
        raise SystemExit(f"the rule's variant {rule['variant']} is not the one the window was measured with")
    ex = classify.read_extent(a.extent, geom.variant)

    raw = load_v6_track(a.data / "Skeleton_data")   # reading A is cut from this, as the loader cuts
    spans = load_v6_spans(a.data / "Annotation_files")
    track = punchfind.bridge(raw, int(round(punchfind.BRIDGE_S * raw.fps)))
    torso = punchfind.body_scale(track)
    sig = punchfind.wrist_signal(track, geom.variant)
    row = {int(f): i for i, f in enumerate(track.frame_ids)}
    det, dh = punchfind.detect(track, sig, rule["threshold"], rule["min_gap_s"])
    pairs = dict(punchfind.match(det, spans, int(round(TOL_S * track.fps))))

    flags = {"censored": 0, "capped": 0, "n": 0}
    lengths = []

    def extent_window(peak: int, hand: int, count: bool = False):
        e = classify.punch_extent(track, sig, peak, hand, ex, geom)
        if count:
            flags["n"] += 1
            flags["censored"] += e.censored
            flags["capped"] += e.capped
            lengths.append((e.end - e.start) / track.fps)
        return classify.cut_span(track, e.start, e.end, geom, torso).keypoints

    # The control window: the median extent's length for every punch, split as the rule splits it.
    peaks = {}
    for j, s in enumerate(spans):
        p = span_peak(track, sig, s.start, s.end)
        if p in row and sig[row[p]].max() > 0:
            peaks[j] = p
            extent_window(p, int(np.argmax(sig[row[p]])), count=True)
    med = float(np.median(lengths))
    share = ex["lead_widths"] / (ex["lead_widths"] + ex["tail_widths"])

    def control_window(peak: int):
        a_ = int(np.floor(peak - share * med * track.fps + 0.5))
        return classify.cut_span(track, a_, int(np.floor(peak + (1 - share) * med * track.fps + 0.5)),
                                 geom, torso).keypoints

    readings = {k: {} for k in "ABCDEF"}
    for j, s in enumerate(spans):
        c = corpus_clip(raw, s)
        if c is not None:
            readings["A"][j] = c
        p = span_peak(track, sig, s.start, s.end)
        w = classify.cut_window(track, p, geom, torso).keypoints
        if w is not None:
            readings["B"][j] = w
        if j in peaks:
            w = extent_window(p, int(np.argmax(sig[row[p]])))
            if w is not None:
                readings["D"][j] = w
            w = control_window(p)
            if w is not None:
                readings["F"][j] = w
        if j in pairs:
            w = classify.cut_window(track, int(det[pairs[j]]), geom, torso).keypoints
            if w is not None:
                readings["C"][j] = w
            w = extent_window(int(det[pairs[j]]), int(dh[pairs[j]]))
            if w is not None:
                readings["E"][j] = w

    preds = {}
    for k, rd in readings.items():
        idx = sorted(rd)
        p = classify.predict_proba(model, [rd[i] for i in idx], man["t_out"])
        preds[k] = {i: int(np.argmax(q)) for i, q in zip(idx, p)}

    y = {j: s.label for j, s in enumerate(spans)}
    lo, hi = r["test_window"]
    second = {j for j, s in enumerate(spans) if lo <= s.start and s.end <= hi}
    result = {"video": "V6", "n_labelled": len(spans), "classifier_sha256": man["sha256"],
              "geometry": asdict(geom), "extent": ex, "finder_matched": len(pairs),
              "extents_at_span_peaks": {**flags, "median_length_s": med, "control_window_s": med},
              "all": block(preds, y, set(y)), "second_half": {"window": [lo, hi], **block(preds, y, second)},
              "limits": [
                  "V6 has hooks and uppercuts only: no jabs or crosses",
                  "one athlete, same channel as training video V1 (O55): may be optimistic",
                  "AlphaPose skeletons as in training: says nothing about the YOLO11-pose switch",
                  "the finder's rule was tuned on V6's first half, and the extent rule's width level and form "
                  "were chosen looking at it: the second half is the clean number"]}
    for name in ("all", "second_half"):
        b = result[name]
        print(f"V6 {name.replace('_', ' ')}:")
        for k, s in b["readings"].items():
            if s["n"]:
                print(f"  {k}: {s['n']:3d} punches, accuracy {s['accuracy']:.3f}, macro F1 (present) "
                      f"{s['macro_f1_present_classes']:.3f}, hand right {s['side_accuracy']:.3f}, "
                      f"floor {s['majority_floor_accuracy']:.3f}")
        for k, g in b["paired"].items():
            if g.get("n", 0) >= 2:
                print(f"  {k.replace('_minus_', ' - ')} accuracy on {g['n']}: {g['point']:+.3f} "
                      f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
