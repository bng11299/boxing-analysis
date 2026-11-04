"""What does cutting punches out of continuous pose cost the classifier? Measured on V6.

    python eval_classify_v6.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json \\
        --extent results/punch_extent.json

V6 is the only BoxingVI video that ships as continuous pose with labelled punches,
and the classifier never trained on it (V6 is excluded from the corpus). So it can
separate the cost of cutting the punch out from everything else. Readings of the
same labelled punches, by the same saved classifier:

  A  corpus clips         each span cut exactly as the corpus loader cuts V6
                          (data._load_v6): the span's frames, normalised by 1920x1080
  B  old window, at peak  classify.cut_window (fixed 0.375 s) at the wrist-speed peak
                          inside the span: the first design, with a perfect finder
  C  old window, at finder  the same at the finder's own detections that match a span
  D  per punch, at peak   each punch cut by ITS OWN extent (punchfind.extent): the
                          design ADR-0014 first chose, with a perfect finder
  E  per punch, at finder the same at the finder's matched detections
  F  per clip, at peak    THE PIPELINE'S CUT since 2026-10-13 (classify.clip_cut +
                          clip_window): one length for the whole video, the median of
                          the punches' own extents, around each punch's peak. Here
                          the median is over the LABELLED punches: a perfect finder
  G  per clip, at finder  the pipeline as it runs: the median is over EVERY punch the
                          finder detected (V6's unlabelled jabs and crosses included),
                          and the window sits at the finder's matched detections

F - B and G - C are what the per-clip window buys over the old one; D - F and
E - G what sizing each punch on its own adds (on V6, nothing measurable either
way); A - F what cutting still costs;
F - G the finder's timing and its unlabelled detections. Which detection a labelled
punch is read at in C, E and G is decided by punchfind.match_nearest (`finder_pairs`):
among the matchings with the most hits, the one with the least total distance from
each detection to its span's middle.

Reported on all of V6 and on its second half alone, because the extent rule's
width level and the choice of rule were made looking at the first half
(run_punch_extent.py); the second half is the clean number. Paired bootstrap intervals on the punches both readings read.

Limits, stated in the results: V6 has only hooks and uppercuts (no jabs or
crosses), one athlete, and the same person as training video V1 (ISSUES-LOG O55): not
athlete-disjoint, so its numbers may be optimistic about a new athlete. Its skeletons are
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
PAIRS = (("A", "B"), ("A", "F"), ("F", "B"), ("D", "F"), ("D", "B"), ("G", "C"), ("E", "G"), ("F", "G"))


def corpus_clip(track, s) -> "np.ndarray | None":
    """The span exactly as data._load_v6 builds a V6 clip."""
    lo, hi = np.searchsorted(track.frame_ids, [s.start, s.end + 1])
    if hi - lo < 2:
        return None
    return pad_or_trim(_with_validity_channel((track.xy[lo:hi] / np.array(SOURCE_SIZE)).astype(np.float32)))


def finder_pairs(det, spans, fps: float) -> "dict[int, int]":
    """Labelled span -> the finder's detection that readings C, E and G read it at.

    Nearest attribution (punchfind.match_nearest), not punchfind.match's greedy: 212 of
    V6's 685 spans overlap the next one, and the greedy hands a combination's one
    detection to whichever span ends first. The count is the same either way."""
    return dict(punchfind.match_nearest(det, spans, int(round(TOL_S * fps))))


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
    pairs = finder_pairs(det, spans, track.fps)

    def extent_window(peak: int, hand: int):
        e = classify.punch_extent(track, sig, peak, hand, ex, geom)
        return classify.cut_span(track, e.start, e.end, geom, torso).keypoints

    def clip_window(peak: int, c: classify.ClipCut):
        return classify.read_clip_window(track, peak, c, geom, torso)[0].keypoints

    # The labelled punches' peaks (a perfect finder), and every punch the finder detected (the pipeline).
    peaks = {}
    for j, s in enumerate(spans):
        p = span_peak(track, sig, s.start, s.end)
        if p in row and sig[row[p]].max() > 0:
            peaks[j] = (p, int(np.argmax(sig[row[p]])))
    cut_labelled = classify.clip_cut(track, sig, list(peaks.values()), ex, geom)
    cut_finder = classify.clip_cut(track, sig, [(int(f), int(h)) for f, h in zip(det, dh)], ex, geom)

    readings = {k: {} for k in "ABCDEFG"}
    for j, s in enumerate(spans):
        c = corpus_clip(raw, s)
        if c is not None:
            readings["A"][j] = c
        p = span_peak(track, sig, s.start, s.end)
        w = classify.cut_window(track, p, geom, torso).keypoints
        if w is not None:
            readings["B"][j] = w
        if j in peaks:
            w = extent_window(*peaks[j])
            if w is not None:
                readings["D"][j] = w
            w = clip_window(p, cut_labelled)
            if w is not None:
                readings["F"][j] = w
        if j in pairs:
            d_ = int(det[pairs[j]])
            for k, w in (("C", classify.cut_window(track, d_, geom, torso).keypoints),
                         ("E", extent_window(d_, int(dh[pairs[j]]))), ("G", clip_window(d_, cut_finder))):
                if w is not None:
                    readings[k][j] = w

    preds = {}
    for k, rd in readings.items():
        idx = sorted(rd)
        p = classify.predict_proba(model, [rd[i] for i in idx], man["t_out"])
        preds[k] = {i: int(np.argmax(q)) for i, q in zip(idx, p)}

    y = {j: s.label for j, s in enumerate(spans)}
    lo, hi = r["test_window"]
    second = {j for j, s in enumerate(spans) if lo <= s.start and s.end <= hi}
    result = {"video": "V6", "n_labelled": len(spans), "classifier_sha256": man["sha256"],
              "geometry": asdict(geom), "extent": ex, "finder_matched": len(pairs), "finder_attribution": "nearest",
              "clip_cut_from_labelled_punches": cut_labelled._asdict(),
              "clip_cut_from_finder_detections": {**cut_finder._asdict(), "n_detections": int(len(det))},
              "all": block(preds, y, set(y)), "second_half": {"window": [lo, hi], **block(preds, y, second)},
              "limits": [
                  "the per-clip cut (F, G) was CHOSEN looking at this file's second-half numbers and at V5/V9 "
                  "(ADR-0014, 2026-10-13): they are no longer a clean test of it",
                  "V6 has hooks and uppercuts only: no jabs or crosses",
                  "one athlete, the same person as training video V1 (O55): not athlete-disjoint, may be optimistic",
                  "AlphaPose skeletons as in training: says nothing about the YOLO11-pose switch",
                  "the finder's rule was tuned on V6's first half, and the extent rule's width level and form "
                  "were chosen looking at it: the second half is the clean number"]}
    print(f"per-clip window: {cut_labelled.length_s:.3f} s from the {cut_labelled.n_used} labelled punches; "
          f"{cut_finder.length_s:.3f} s from the finder's {cut_finder.n_used} detections "
          f"({cut_finder.n_left_out} left out)")
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
