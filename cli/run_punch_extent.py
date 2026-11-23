"""Derive the punch-extent rule from the training clips, and test it on V6.

    python cli/run_punch_extent.py --data data --classifier weights/classifier/stgcnpp-boxingvi-seed0.json

The rule (punchfind.extent): a punch runs `lead_widths` peak-widths before its
wrist-speed peak to `tail_widths` after, where the width is how long the punching
wrist holds at least half its peak speed. The two ratios are medians over the
TRAINING clips (classify.training_extent_ratios). V6 did not set them; it did
decide the half-height level and the form of the rule (see below).

V6 is then the test: overlap (IoU) with each labelled span, for two cuts built on
the rule, against the old fixed window (the training clips' median length and
peak position, from the classifier's manifest):
  per punch   each punch's own extent
  per clip    THE PIPELINE'S CUT (ADR-0014 as amended 2026-10-13): one length for
              the whole video, the median of the punches' own extents, around each
              punch's peak (classify.clip_cut / clip_window)
Both are measured with a perfect finder (at the wrist-speed peak inside each
labelled span). THE GATE is measured as the pipeline runs: the per-clip length is
the median over every punch the FINDER detected in V6, and the windows sit at its
detections that match a labelled span in V6's second half. The result file says
whether that beat the old window in overlap with the labels, with a paired
bootstrap interval, and names the cut it gated; classify_punches.py refuses a
rule file that gated another cut, or whose cut did not win.

What V6's first half WAS used for, so the second half is the clean number: the
width level (0.5, from 0.5 / 0.6 / 0.7) and the choice of this rule over a
threshold walk were made looking at the first half only.

Limits: V6 is one athlete -- the same person as training video V1 (O55) -- hooks and uppercuts only, AlphaPose at 24 fps, and its
labelled spans run nearly twice as long as the training clips of the same types
(0.67 s against 0.3-0.4 s), so an extent built from the training clips is
expected to come up short of them. The extent of a jab or a cross is checked on
V2/V5/V9 by eval_pose_switch.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import classify, punchfind  # noqa: E402
from boxingvi.data import PIPELINE_HELD_OUT_VIDEOS, load_corpus, load_v6_spans, load_v6_track, split_pipeline  # noqa: E402
from classify_punches import span_peak  # noqa: E402

DEFAULT_RULE = Path("results/punch_finder_v6.json")
DEFAULT_OUT = Path("results/punch_extent.json")
TOL_S = 0.1


def fixed_span(peak: int, fps: float, geom: classify.WindowGeometry) -> "tuple[int, int]":
    """The frames the fixed window covers, as an inclusive span."""
    t0 = peak / fps - geom.lead_in_s
    return (int(np.floor(t0 * fps + 0.5)),
            int(np.floor((t0 + (geom.n_frames - 1) / geom.fps) * fps + 0.5)))


def describe(pred: "list[tuple[int, int]]", spans) -> dict:
    iou = np.array([punchfind.span_iou(p, (s.start, s.end)) for p, s in zip(pred, spans)])
    ds = np.array([p[0] - s.start for p, s in zip(pred, spans)])
    de = np.array([p[1] - s.end for p, s in zip(pred, spans)])
    return {"n": len(spans), "iou_mean": float(iou.mean()), "iou_median": float(np.median(iou)),
            "start_error_frames_median": float(np.median(ds)), "end_error_frames_median": float(np.median(de)),
            "start_abs_error_frames_mean": float(np.abs(ds).mean()),
            "end_abs_error_frames_mean": float(np.abs(de).mean()),
            "length_frames_median": float(np.median([p[1] - p[0] + 1 for p in pred])),
            "labelled_length_frames_median": float(np.median([s.end - s.start + 1 for s in spans]))}


def paired_gain(a, b, spans, n_boot: int = 2000, seed: int = 0) -> dict:
    d = np.array([punchfind.span_iou(x, (s.start, s.end)) - punchfind.span_iou(y, (s.start, s.end))
                  for x, y, s in zip(a, b, spans)])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(d), "point": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": n_boot}


def matched_inside(det, spans, fps: float, window: "tuple[int, int]") -> "list[tuple[int, int]]":
    """(span, detection) pairs for the spans inside `window`, each read at the detection
    the matching attributes to it. Nearest attribution (punchfind.match_nearest), as
    eval_classify_v6.py's readings: the greedy punchfind.match hands an overlapping
    combination's one detection to whichever span ends first, and the overlap is then
    measured against the wrong span. classify.GATED_CUT names this attribution, so a rule
    file gated under the greedy one is refused."""
    return [(si, di) for si, di in punchfind.match_nearest(det, spans, int(round(TOL_S * fps)))
            if window[0] <= spans[si].start and spans[si].end <= window[1]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True, help="manifest: its geometry is the fixed-window baseline")
    ap.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    man = classify.read_manifest(a.classifier)
    geom: classify.WindowGeometry = man["geometry"]
    r = json.loads(a.rule.read_text(encoding="utf-8"))
    rule = r["chosen"]
    if rule["variant"] != geom.variant:
        raise SystemExit("the finder's rule and the classifier's window use different wrist-speed variants")

    train, _ = split_pipeline(load_corpus(a.data))   # the classifier checkpoint's training side
    ratios = classify.training_extent_ratios(train, geom.variant, punchfind.WIDTH_FRAC)
    print(f"from {ratios['n_clips']} training clips with a whole speed peak ({ratios['n_censored']} cut by the "
          f"clip's edge, left out): a punch runs {ratios['lead_widths']:.2f} peak-widths before its peak "
          f"and {ratios['tail_widths']:.2f} after")
    print(f"  the clips kept are a median {ratios['duration_s_median_kept']:.3f} s long, all measurable clips "
          f"{ratios['duration_s_median_all']:.3f} s; left out by type: " + ", ".join(
              f"{c} {v['censored']}/{v['of']}" for c, v in ratios["censored_by_class"].items()))

    track = load_v6_track(a.data / "Skeleton_data")
    track = punchfind.bridge(track, int(round(punchfind.BRIDGE_S * track.fps)))
    spans = load_v6_spans(a.data / "Annotation_files")
    sig = punchfind.wrist_signal(track, geom.variant)
    row = {int(f): i for i, f in enumerate(track.frame_ids)}
    peaks = [span_peak(track, sig, s.start, s.end) for s in spans]
    usable = [p in row and sig[row[p]].max() > 0 for p in peaks]   # a span over a gap in the pose has no peak
    hands = [int(np.argmax(sig[row[p]])) if u else 0 for p, u in zip(peaks, usable)]

    def ext(p, h):
        e = classify.punch_extent(track, sig, p, h, ratios, geom)
        return e.start, e.end

    clip = classify.clip_cut(track, sig, [(p, h) for p, h, u in zip(peaks, hands, usable) if u], ratios, geom)

    out = {"rule": "extent = [peak - lead_widths * width, peak + tail_widths * width]; width = time the punching "
                   "wrist holds >= frac of its peak speed",
           **ratios, "ratios_from": "BoxingVI training side; held out " + "/".join(PIPELINE_HELD_OUT_VIDEOS),
           "tested_on": "V6", "fps": track.fps, "v6": {}, "v6_clip_cut": clip._asdict(),
           "pipeline_cut": "per clip: every punch read over the median of the clip's own punch extents"}
    for name, w in (("first_half", tuple(r["tune_window"])), ("second_half", tuple(r["test_window"]))):
        idx = [j for j, s in enumerate(spans) if w[0] <= s.start and s.end <= w[1] and usable[j]]
        sp = [spans[j] for j in idx]
        e, f = [ext(peaks[j], hands[j]) for j in idx], [fixed_span(peaks[j], track.fps, geom) for j in idx]
        c = [classify.clip_window(peaks[j], clip, track.fps) for j in idx]
        out["v6"][name] = {"window": list(w), "extent": describe(e, sp), "fixed_window": describe(f, sp),
                           "clip_window": describe(c, sp), "extent_minus_fixed_iou": paired_gain(e, f, sp),
                           "clip_minus_fixed_iou": paired_gain(c, f, sp),
                           "clip_minus_extent_iou": paired_gain(c, e, sp)}
    # As the pipeline runs: anchored at the finder's own detections, second half only.
    det, dh = punchfind.detect(track, sig, rule["threshold"], rule["min_gap_s"])
    w = tuple(r["test_window"])
    # Sorted by span index (match_nearest's order); paired_gain resamples by position, so its interval also
    # moved with the change from the greedy's span-end order, not only with the attribution.
    m = matched_inside(det, spans, track.fps, w)
    if m:
        sp = [spans[si] for si, _ in m]
        e = [ext(int(det[di]), int(dh[di])) for _, di in m]
        f = [fixed_span(int(det[di]), track.fps, geom) for _, di in m]
        # The pipeline's cut as it runs: one length from EVERY detection, labelled or not.
        clip_f = classify.clip_cut(track, sig, [(int(a_), int(b_)) for a_, b_ in zip(det, dh)], ratios, geom)
        c = [classify.clip_window(int(det[di]), clip_f, track.fps) for _, di in m]
        out["v6_clip_cut_from_finder_detections"] = {**clip_f._asdict(), "n_detections": int(len(det))}
        out["v6"]["second_half_at_finder_detections"] = {
            "extent": describe(e, sp), "fixed_window": describe(f, sp), "clip_window": describe(c, sp),
            "extent_minus_fixed_iou": paired_gain(e, f, sp), "clip_minus_fixed_iou": paired_gain(c, f, sp),
            "clip_minus_extent_iou": paired_gain(c, e, sp)}
    gate = out["v6"].get("second_half_at_finder_detections", {}).get("clip_minus_fixed_iou")
    out["gated_cut"] = classify.GATED_CUT
    out["beats_fixed_window"] = bool(gate is not None and gate["ci_low"] > 0)
    out["first_half_was_used_for"] = "choosing the width level (0.5 of 0.5/0.6/0.7) and this rule over a threshold walk"
    out["limits"] = ["the per-clip cut was chosen looking at V6's second half and at V5/V9 (ADR-0014, 2026-10-13): "
                     "this gate is no longer an independent test of it",
                     "V6: one athlete, the same person as training video V1 (O55: not athlete-disjoint), hooks and "
                     "uppercuts only, AlphaPose at 24 fps",
                     "V6's labelled spans run nearly twice as long as training clips of the same types",
                     "the extent of a jab or cross has not been checked against a label here"]

    for name in ("first_half", "second_half", "second_half_at_finder_detections"):
        t = out["v6"].get(name)
        if not t:
            continue
        g = t["extent_minus_fixed_iou"]
        print(f"V6 {name.replace('_', ' ')} ({t['extent']['n']} punches): per-punch extent IoU "
              f"{t['extent']['iou_mean']:.3f} (median {t['extent']['length_frames_median']:.0f} frames; labels "
              f"{t['extent']['labelled_length_frames_median']:.0f}) vs old window "
              f"{t['fixed_window']['iou_mean']:.3f}; gain {g['point']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]")
        if "clip_window" in t:
            g = t["clip_minus_fixed_iou"]
            print(f"   per-clip window IoU {t['clip_window']['iou_mean']:.3f} "
                  f"({t['clip_window']['length_frames_median']:.0f} frames); over the old window {g['point']:+.3f} "
                  f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]")
    print("GATE - the pipeline's cut as it runs (per clip, from the finder's detections) beats the old window "
          "on the second half: " + ("YES" if out["beats_fixed_window"] else "NO"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
