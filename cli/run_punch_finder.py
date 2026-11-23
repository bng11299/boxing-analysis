"""The wrist-speed punch finder on V6, behind an alignment gate.

    python cli/run_punch_finder.py --data data [--video data/rgb/V6.mp4]

1. ALIGNMENT GATE. Before anything is scored: labels must line up with wrist
   motion (every integer shift in +-3 s scanned; the best must be within the
   scoring tolerance and clearly above shifts a second away), and the skeleton
   must line up with the source video (correlation peaks at lag 0 over +-48
   lags, with r >= 0.30 and 3x every lag beyond +-6). Each check tests strength as
   well as position. A failure exits non-zero and writes nothing.
2. TUNE on the first half of V6's labelled range: rule variant, threshold and
   minimum gap, chosen by F1.
3. SCORE once on the second half, with a block-bootstrap interval and two
   chance floors at the same detection count: the detector's own detections
   time-shifted (primary) and uniform placement (secondary).

The halves are cut at the point nearest the middle that is at least the
scoring tolerance away from every span, so no punch straddles them and no
detection can be matched across the cut. The unlabelled tail after the last span is never
scored. V6 is one athlete and has only hooks and uppercuts; see data.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import align, punchfind  # noqa: E402
from boxingvi.data import load_v6_spans, load_v6_track  # noqa: E402

TOL_S = 0.1          # fixed before any result was seen: labels are good to ~2 frames
BLOCK_S = 30.0       # bootstrap block length
MIN_SHIFT_S = 10.0   # chance floor: never shift detections by less than this
MIN_GAPS_S = (0.15, 0.25, 0.4)
QUANTILES = np.arange(0.50, 0.995, 0.01)
DEFAULT_OUT = Path("results/punch_finder_v6.json")


def cut_point(track, spans, lo: int, hi: int, guard: int = 0) -> int:
    """The frame nearest the middle of [lo, hi] that is at least `guard` frames
    from every span, so a detection within the matching tolerance of the cut
    cannot be scored against a punch on the other side of it."""
    covered = np.zeros(hi - lo + 1, dtype=bool)
    for s in spans:
        a, b = max(s.start - guard, lo), min(s.end + guard, hi)
        if a > b:  # the span (with its guard) lies wholly outside [lo, hi]
            continue
        covered[a - lo:b - lo + 1] = True
    free = np.flatnonzero(~covered) + lo
    if len(free) == 0:
        raise ValueError("every frame is inside a span; there is no clean cut")
    return int(free[np.abs(free - (lo + hi) // 2).argmin()])


# Gate thresholds. A gate that checks only WHERE a peak is passes noise: before
# these existed, 27 of 388 deliberately wrong skeleton offsets (7%) passed the
# video check on V6, because among 13 lags of pure noise one of them is lag 0
# about 1 time in 13. Each check below therefore also demands STRENGTH. The
# values sit well below what aligned V6 shows (video r 0.55 against at most
# ~0.06 beyond +-6 lags; label ratio 1.58 against ~1.0 a second away; peri
# peak 2.0x its median) and well above what misaligned data shows (r <= 0.11).
XCORR_MAX_LAG = 48      # frames either side scanned against the video
XCORR_MIN_R = 0.30      # correlation required at lag 0
XCORR_MIN_MARGIN = 3.0  # r(0) must be this many times every |r(lag)| far from 0
XCORR_NEAR_LAG = 6      # lags within a quarter second are "near": a smooth
#                         motion correlates with itself there, aligned or not,
#                         so the margin is taken only against lags beyond it.
#                         Lag 0 must still beat the near lags to be the peak.
SHIFT_SCAN_S = 3.0      # labels scanned over every integer shift in +-3 s
SHIFT_MIN_RATIO = 1.25  # inside/outside speed ratio required at the best shift
SHIFT_MIN_MARGIN = 1.25  # ... and over the median ratio a second or more away
PERI_MIN_PROMINENCE = 1.5


def alignment_gate(track, spans, video: "Path | None", tol: int) -> tuple[dict, list[str]]:
    """Every check must find the alignment AND find it clearly. `tol` is the
    scoring tolerance: labels off by more than it would be scored wrongly."""
    speed = punchfind.wrist_signal(track, "relative").max(axis=1)
    reach = int(round(SHIFT_SCAN_S * track.fps))
    scan = align.label_shift_scan(track, spans, speed, range(-reach, reach + 1))
    peri = align.peri_event_peak(track, spans, speed)
    report = {"label_shift_ratio": scan, "peri_event": peri}
    failures = []
    best = max(scan, key=scan.get)
    # The scan's peak can sit a frame off-centre on aligned labels, so labels
    # off by tol + 1 could read as tol. Allow one frame less than the scoring
    # tolerance, so anything beyond it is caught.
    limit = max(tol - 1, 0)
    if abs(best) > limit:
        failures.append(f"labels: speed ratio peaks at shift {best}, beyond +-{limit} "
                        f"frames (scoring tolerance {tol})")
    far = [v for k, v in scan.items() if abs(k) >= track.fps]
    if not scan[best] >= SHIFT_MIN_RATIO:
        failures.append(f"labels: best inside/outside speed ratio {scan[best]:.2f} "
                        f"is below {SHIFT_MIN_RATIO}; the labels barely track motion")
    if not scan[best] >= SHIFT_MIN_MARGIN * float(np.median(far)):
        failures.append("labels: the best shift fits little better than shifts a "
                        "second or more away")
    median_len = int(np.median([s.end - s.start for s in spans]))
    if not 0 <= peri["peak_lag"] <= median_len:
        failures.append(f"labels: speed peaks {peri['peak_lag']} frames from the "
                        f"labelled start, outside a typical span (0..{median_len})")
    if not peri["peak_over_median"] >= PERI_MIN_PROMINENCE:
        failures.append(f"labels: the averaged speed peak is only "
                        f"{peri['peak_over_median']:.2f}x its median")
    if video is None:
        report["video"] = "SKIPPED"
        return report, failures
    energy = align.video_motion_energy(video)
    xc = align.skeleton_video_xcorr(track, energy,
                                    range(-XCORR_MAX_LAG, XCORR_MAX_LAG + 1))
    report["video"] = {"decoded_frames": int(len(energy)),
                       "last_skeleton_frame": int(track.frame_ids[-1]),
                       "xcorr_by_lag": xc}
    failures += video_failures(xc)
    if track.frame_ids[-1] >= len(energy):
        failures.append("video: skeleton frame ids run past the end of the video")
    return report, failures


def video_failures(xc: dict) -> list[str]:
    """The skeleton-video check on a lag -> r map: a clear peak at lag 0."""
    out = []
    if max(xc, key=xc.get) != 0:
        out.append("video: skeleton-video correlation does not peak at lag 0")
    if not xc[0] >= XCORR_MIN_R:
        out.append(f"video: correlation at lag 0 is {xc[0]:.3f}, below {XCORR_MIN_R}")
    far = [abs(v) for k, v in xc.items() if abs(k) > XCORR_NEAR_LAG]
    if not far:
        out.append(f"video: no lag beyond +-{XCORR_NEAR_LAG} was scanned")
    elif not xc[0] >= XCORR_MIN_MARGIN * max(far):
        out.append(f"video: lag 0 ({xc[0]:.3f}) is not {XCORR_MIN_MARGIN}x every "
                   f"lag beyond +-{XCORR_NEAR_LAG} (max {max(far):.3f})")
    return out


def json_safe(x):
    """Non-finite floats as strings: bare Infinity/NaN is not JSON, and strict
    parsers (jq, JSON.parse) reject the whole file over one of them."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


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
                    help="V6 source video, for the skeleton-video check (required "
                         "unless --no-video-check)")
    ap.add_argument("--no-video-check", action="store_true",
                    help="score without the skeleton-video check; the output is "
                         "stamped SKIPPED and may not use the default --out")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    # The headline must never come from a run that skipped half the gate.
    if a.video is None and not a.no_video_check:
        ap.error("--video is required; pass --no-video-check to score without it")
    if a.video is not None and a.no_video_check:
        ap.error("--video and --no-video-check contradict each other")
    if a.no_video_check and a.out.resolve() == DEFAULT_OUT.resolve():
        ap.error("a run without the video check may not write the default results "
                 "file; pass a different --out")

    track = load_v6_track(a.data / "Skeleton_data")
    spans = load_v6_spans(a.data / "Annotation_files")
    if (max(s.end for s in spans) > track.frame_ids[-1]
            or min(s.start for s in spans) < track.frame_ids[0]):
        print(f"labels span frames {spans[0].start}..{max(s.end for s in spans)}, "
              f"outside the pose ({track.frame_ids[0]}..{track.frame_ids[-1]})")
        return 2
    fps = track.fps
    tol = int(round(TOL_S * fps))

    report, failures = alignment_gate(track, spans, a.video, tol)
    print(json.dumps(json_safe(report), indent=2))
    if failures:
        print("\nALIGNMENT GATE FAILED -- nothing scored:", *failures, sep="\n  ")
        return 2
    print("\nalignment gate passed")

    lo, hi = min(s.start for s in spans), max(s.end for s in spans)
    cut = cut_point(track, spans, lo, hi, guard=tol)
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
        "video_check": "SKIPPED" if a.video is None else "passed",
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
    a.out.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False),
                     encoding="utf-8")

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
