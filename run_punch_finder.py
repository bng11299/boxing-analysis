"""The wrist-speed punch finder on V6, behind an alignment gate.

    python run_punch_finder.py --data data [--video data/rgb/V6.mp4]

1. ALIGNMENT GATE. Before anything is scored: labels must line up with wrist
   motion (shift scan peaks within +-3 frames of 0), and, with --video, the
   skeleton must line up with the source video (cross-correlation peaks at
   lag 0). A failure exits non-zero and writes nothing.
2. TUNE on the first half of V6's labelled range: rule variant, threshold and
   minimum gap, chosen by F1.
3. SCORE once on the second half, with a block-bootstrap interval and a
   random-placement floor at the same detection count.

The halves are cut at the point nearest the middle that lies inside no span,
so no punch straddles them. The unlabelled tail after the last span is never
scored. V6 is one athlete and has only hooks and uppercuts; see data.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import align, punchfind  # noqa: E402
from boxingvi.data import load_v6_spans, load_v6_track  # noqa: E402

TOL_S = 0.1          # fixed before any result was seen: labels are good to ~2 frames
BLOCK_S = 30.0       # bootstrap block length
MIN_SHIFT_S = 10.0   # chance floor: never shift detections by less than this
MIN_GAPS_S = (0.15, 0.25, 0.4)
QUANTILES = np.arange(0.50, 0.995, 0.01)


def cut_point(track, spans, lo: int, hi: int) -> int:
    """The frame nearest the middle of [lo, hi] that is inside no span."""
    covered = np.zeros(hi - lo + 1, dtype=bool)
    for s in spans:
        covered[max(s.start, lo) - lo:min(s.end, hi) - lo + 1] = True
    free = np.flatnonzero(~covered) + lo
    if len(free) == 0:
        raise ValueError("every frame is inside a span; there is no clean cut")
    return int(free[np.abs(free - (lo + hi) // 2).argmin()])


def alignment_gate(track, spans, video: "Path | None") -> tuple[dict, list[str]]:
    speed = punchfind.wrist_signal(track, "relative").max(axis=1)
    shifts = [-48, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 48]
    scan = align.label_shift_scan(track, spans, speed, shifts)
    peri = align.peri_event_peak(track, spans, speed)
    report = {"label_shift_ratio": scan, "peri_event": peri}
    failures = []
    best = max(scan, key=scan.get)
    if abs(best) > 3:
        failures.append(f"labels: speed ratio peaks at shift {best}, not within +-3")
    if not scan[0] > max(scan[-48], scan[48]):
        failures.append("labels: a 48-frame shift fits as well as none")
    median_len = int(np.median([s.end - s.start for s in spans]))
    if not 0 <= peri["peak_lag"] <= median_len:
        failures.append(f"labels: speed peaks {peri['peak_lag']} frames from the "
                        f"labelled start, outside a typical span (0..{median_len})")
    if video is not None:
        energy = align.video_motion_energy(video)
        xc = align.skeleton_video_xcorr(track, energy)
        report["video"] = {"decoded_frames": int(len(energy)),
                           "last_skeleton_frame": int(track.frame_ids[-1]),
                           "xcorr_by_lag": xc}
        if max(xc, key=xc.get) != 0:
            failures.append("video: skeleton-video correlation does not peak at lag 0")
        if track.frame_ids[-1] >= len(energy):
            failures.append("video: skeleton frame ids run past the end of the video")
    return report, failures


def tune(track, spans, window, tol) -> dict:
    lo_row = np.searchsorted(track.frame_ids, window[0])
    hi_row = np.searchsorted(track.frame_ids, window[1], side="right")
    best = None
    for variant in punchfind.VARIANTS:
        sig = punchfind.wrist_signal(track, variant)
        thresholds = np.quantile(sig[lo_row:hi_row].max(axis=1), QUANTILES)
        for gap in MIN_GAPS_S:
            for q, th in zip(QUANTILES, thresholds):
                det, _ = punchfind.detect(track, sig, float(th), gap)
                s = punchfind.score(det, spans, tol, window)
                cand = {"variant": variant, "threshold": float(th),
                        "threshold_quantile": round(float(q), 2),
                        "min_gap_s": gap, "tune": s}
                # Per-variant best too: the plan names the radial rule, so its
                # own tuned test score is reported next to the chosen one.
                if best is None or s["f1"] > best["overall"]["tune"]["f1"]:
                    best = (best or {}) | {"overall": cand}
                if variant not in best or s["f1"] > best[variant]["tune"]["f1"]:
                    best[variant] = cand
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True,
                    help="folder holding Skeleton_data/ and Annotation_files/")
    ap.add_argument("--video", type=Path, default=None,
                    help="V6 source video, for the skeleton-video check")
    ap.add_argument("--out", type=Path, default=Path("results/punch_finder_v6.json"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    track = load_v6_track(a.data / "Skeleton_data")
    spans = load_v6_spans(a.data / "Annotation_files")
    fps = track.fps
    tol = int(round(TOL_S * fps))

    report, failures = alignment_gate(track, spans, a.video)
    print(json.dumps(report, indent=2))
    if failures:
        print("\nALIGNMENT GATE FAILED -- nothing scored:", *failures, sep="\n  ")
        return 2
    print("\nalignment gate passed")

    lo, hi = min(s.start for s in spans), max(s.end for s in spans)
    cut = cut_point(track, spans, lo, hi)
    tune_w, test_w = (lo, cut - 1), (cut, hi)

    best = tune(track, spans, tune_w, tol)
    results, curve = {}, []
    for key in ("overall", *punchfind.VARIANTS):
        c = best[key]
        sig = punchfind.wrist_signal(track, c["variant"])
        det, hands = punchfind.detect(track, sig, c["threshold"], c["min_gap_s"])
        test = punchfind.score(det, spans, tol, test_w)
        entry = c | {"test": test}
        if key == "overall":
            # Conditional on the tuned rule: the variance of tuning itself is
            # not in this interval.
            entry["test_ci_conditional_on_tuned_rule"] = punchfind.block_bootstrap(
                det, spans, tol, test_w, int(round(BLOCK_S * fps)), a.bootstrap, a.seed)
            entry["shift_floor"] = punchfind.shift_floor(
                det, spans, tol, test_w, int(round(MIN_SHIFT_S * fps)), seed=a.seed)
            entry["uniform_floor"] = punchfind.random_floor(
                test["n_detected"], spans, tol, test_w, track.frame_ids, seed=a.seed)
            # Diagnostic only, never used for selection -- and kept OUT of the
            # chosen entry so nobody later picks a threshold from it.
            for th in np.quantile(sig.max(axis=1), QUANTILES[::5]):
                d, _ = punchfind.detect(track, sig, float(th), c["min_gap_s"])
                curve.append({"threshold": float(th)} | punchfind.score(d, spans, tol, test_w))
        results[key] = entry

    o = results["overall"]
    ci = o["test_ci_conditional_on_tuned_rule"]
    beats = ci["f1"][0] > o["shift_floor"]["f1_p97_5"]
    payload = {
        "video": "V6", "fps": fps, "tolerance_frames": tol, "tolerance_s": TOL_S,
        "n_labelled_spans": len(spans), "labelled_range": [lo, hi],
        "tune_window": list(tune_w), "test_window": list(test_w),
        "alignment": report,
        "chosen": results["overall"],
        "best_radial": results["radial"], "best_relative": results["relative"],
        "beats_shift_floor": bool(beats),
        "grid": {"min_gap_s": list(MIN_GAPS_S),
                 "threshold_quantiles": [round(float(QUANTILES[0]), 2),
                                         round(float(QUANTILES[-1]), 2)]},
        "test_threshold_curve_diagnostic_not_for_selection": curve,
        "caveats": [
            "The grid was fixed before any result was seen and was not widened "
            "afterwards, although the chosen min_gap_s sits at its upper edge.",
            "The alignment gate reads labels from both halves, as pass/fail only; "
            "it selects nothing.",
            "V6 is one athlete and contains only hooks and uppercuts: nothing here "
            "measures jabs or crosses.",
            "Precision is a lower bound, and a loose one: V6 labels only hooks and "
            "uppercuts, but the video shows straight punches too. Checked by eye "
            "2026-09-27: of 6 sampled false positives more than 24 frames from any "
            "label, 5 were jabs, crosses or combinations (the video's own captions "
            "name them); 1 was a conditioning movement. Recall is the trustworthy number.",
            "Tuned and tested on two halves of the same video; this is not an "
            "estimate for other people or other footage.",
        ],
        "seed": a.seed,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    t = o["test"]
    print(f"\nchosen on tune half: {o['variant']} rule, threshold "
          f"{o['threshold']:.2f} body-lengths/s (q{o['threshold_quantile']}), "
          f"min gap {o['min_gap_s']}s")
    print(f"test half ({t['n_labelled']} punches): precision {t['precision']:.3f} "
          f"recall {t['recall']:.3f} F1 {t['f1']:.3f}  "
          f"95% CI {ci['f1'][0]:.3f}-{ci['f1'][1]:.3f} (conditional on the tuned rule)")
    print(f"chance floor, own detections time-shifted: F1 "
          f"{o['shift_floor']['f1_mean']:.3f} (97.5th pct "
          f"{o['shift_floor']['f1_p97_5']:.3f}) -> beats floor: {beats}; "
          f"uniform placement {o['uniform_floor']['f1_mean']:.3f}")
    for v in punchfind.VARIANTS:
        r = results[v]
        print(f"  best {v:8s}: tune F1 {r['tune']['f1']:.3f}, test F1 {r['test']['f1']:.3f}")
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
