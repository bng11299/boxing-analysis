"""Finding punches in continuous pose: the wrist-speed rule, and how it is scored.

The rule is the plan's baseline, the thing any learned punch finder must beat:
a punch is a burst of wrist speed relative to its own shoulder. Everything is
in SECONDS and BODY LENGTHS, never frames or pixels, so the same thresholds
mean the same thing at 24 fps and at 60 fps, and for a near or a far camera
(ISSUES-LOG O54: the corpus is not one frame rate).

Two variants, because the plan's wording ("away from the shoulder") describes
straight punches, and V6 -- the only continuous labelled footage -- contains
only hooks and uppercuts, which barely extend the arm:

    radial    rate of change of the wrist-shoulder distance, extension only
    relative  speed of the wrist in the shoulder's frame, any direction

Which one is used is chosen on a tuning portion of the footage, never on the
frames that are scored.
"""

from __future__ import annotations

import numpy as np

from .data import Span, Track

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12
VARIANTS = ("radial", "relative")


def body_scale(track: Track) -> float:
    """Median shoulder-midpoint to hip-midpoint distance, in pixels."""
    sh = (track.xy[:, L_SHOULDER] + track.xy[:, R_SHOULDER]) / 2
    hp = (track.xy[:, L_HIP] + track.xy[:, R_HIP]) / 2
    scale = float(np.median(np.linalg.norm(sh - hp, axis=1)))
    if not scale > 1.0:
        raise ValueError(f"torso length {scale:.2f}px: no usable body in this track")
    return scale


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average along axis 0, edge-normalised (no zero bias)."""
    # A segment shorter than the window gets the widest odd window that fits;
    # np.convolve(mode="same") returns max(len, w) values and would overflow.
    w = min(w, len(x) - (len(x) + 1) % 2)
    if w <= 1 or len(x) < 2:
        return x
    k = np.ones(w)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, x)
    den = np.convolve(np.ones(len(x)), k, mode="same")
    return num / den.reshape((-1,) + (1,) * (x.ndim - 1))


def wrist_signal(track: Track, variant: str = "relative",
                 smooth_s: float = 0.1) -> np.ndarray:
    """(F, 2) wrist speed per hand (left, right), in body lengths per second.

    Computed segment by segment: a gap in the pose is never differenced
    across, because a jump over missing frames would read as a punch. A
    one-frame segment has no velocity and scores zero.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    scale = body_scale(track)
    rel = np.stack([track.xy[:, L_WRIST] - track.xy[:, L_SHOULDER],
                    track.xy[:, R_WRIST] - track.xy[:, R_SHOULDER]], axis=1) / scale
    w = max(1, int(round(smooth_s * track.fps)))
    w += (w + 1) % 2  # odd, so the window is centred
    out = np.zeros((len(rel), 2))
    for lo, hi in track.segments():
        if hi - lo < 2:
            continue
        seg = _smooth(rel[lo:hi], w)
        if variant == "relative":
            vel = np.gradient(seg, axis=0) * track.fps
            out[lo:hi] = np.linalg.norm(vel, axis=-1)
        else:
            out[lo:hi] = np.clip(np.gradient(np.linalg.norm(seg, axis=-1), axis=0)
                                 * track.fps, 0, None)
    return out


def detect(track: Track, signal: np.ndarray, threshold: float,
           min_gap_s: float) -> "tuple[np.ndarray, np.ndarray]":
    """Peaks of the faster hand's speed above `threshold`, at least `min_gap_s`
    apart. Returns (video frame ids, hand: 0 left / 1 right)."""
    from scipy.signal import find_peaks

    if signal.shape != (len(track.frame_ids), 2):
        raise ValueError(f"signal shape {signal.shape} does not match the track")
    if not (np.isfinite(threshold) and min_gap_s >= 0):
        raise ValueError("threshold must be finite and min_gap_s non-negative")
    dist = max(1, int(round(min_gap_s * track.fps)))
    best = signal.max(axis=1)
    frames, hands = [], []
    for lo, hi in track.segments():
        peaks, _ = find_peaks(best[lo:hi], height=threshold, distance=dist)
        frames.append(track.frame_ids[lo + peaks])
        hands.append(signal[lo + peaks].argmax(axis=1))
    return np.concatenate(frames), np.concatenate(hands)


def match(det_frames, spans: "list[Span]", tol: int) -> "list[tuple[int, int]]":
    """Maximum one-to-one matching of detections to labelled spans.

    A detection can match a span if it falls within [start - tol, end + tol].
    Spans are taken in order of END, each given the earliest free detection
    that fits -- the greedy that is provably optimal for points-to-intervals.
    Optimality matters here: 212 of V6's 685 spans overlap the next one, and a
    naive first-come assignment undercounts combinations.

    Returns (span index, detection index) pairs.
    """
    if tol < 0:
        raise ValueError("tol must be >= 0")
    det = np.asarray(det_frames)
    order = np.argsort(det, kind="stable")
    sorted_det = det[order]
    used = np.zeros(len(det), dtype=bool)
    pairs = []
    for si in sorted(range(len(spans)), key=lambda i: (spans[i].end, spans[i].start)):
        sp = spans[si]
        j = int(np.searchsorted(sorted_det, sp.start - tol, side="left"))
        while j < len(sorted_det) and sorted_det[j] <= sp.end + tol:
            if not used[j]:
                used[j] = True
                pairs.append((si, int(order[j])))
                break
            j += 1
    return pairs


def _in(frames, window) -> np.ndarray:
    lo, hi = window
    f = np.asarray(frames)
    return (f >= lo) & (f <= hi)


def score(det_frames, spans: "list[Span]", tol: int, window: "tuple[int, int]") -> dict:
    """Precision, recall and F1 inside an inclusive frame window.

    A span belongs to the window by its start, a detection by its frame; both
    outside it are ignored. Precision is a LOWER bound wherever the labels are
    incomplete -- an unlabelled real punch counts against the detector.
    """
    det = np.asarray(det_frames)[_in(det_frames, window)]
    sp = [s for s in spans if window[0] <= s.start <= window[1]]
    tp = len(match(det, sp, tol))
    p = tp / len(det) if len(det) else 0.0
    r = tp / len(sp) if sp else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "n_detected": int(len(det)), "n_labelled": len(sp),
            "precision": p, "recall": r, "f1": f1,
            "count_ratio": len(det) / len(sp) if sp else float("nan")}


def shift_floor(det_frames, spans, tol: int, window, min_shift: int,
                n_rep: int = 200, seed: int = 0) -> dict:
    """The chance floor: the detector's OWN detections, circularly shifted in
    time by a random offset of at least `min_shift` frames, then scored.

    This keeps everything about the detector except its timing -- the count,
    the minimum spacing, the clustering into combinations -- so what remains is
    what that pattern of guesses scores by luck. It is the primary floor. The
    uniform one below lets guesses land a frame apart, wastes hits on a single
    span, and so sits lower than luck really does (0.318 vs ~0.353 on V6).
    """
    det = np.asarray(det_frames)[_in(det_frames, window)]
    length = window[1] - window[0] + 1
    if not 0 < min_shift < length // 2:
        raise ValueError(f"min_shift must be in (0, {length // 2}), got {min_shift}")
    rng = np.random.default_rng(seed)
    f1s = np.empty(n_rep)
    for i in range(n_rep):
        k = int(rng.integers(min_shift, length - min_shift + 1))
        shifted = window[0] + (det - window[0] + k) % length
        f1s[i] = score(shifted, spans, tol, window)["f1"]
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return {"n_detections": int(len(det)), "f1_mean": float(f1s.mean()),
            "f1_p2_5": float(lo), "f1_p97_5": float(hi), "n_rep": n_rep,
            "min_shift_frames": int(min_shift)}


def random_floor(n_det: int, spans, tol: int, window, valid_frames,
                 n_rep: int = 200, seed: int = 0) -> dict:
    """F1 of `n_det` detections placed uniformly at random on frames that have
    pose inside the window. Secondary to `shift_floor`: it ignores the
    detector's spacing, so it understates what luck scores."""
    pool = np.asarray(valid_frames)[_in(valid_frames, window)]
    if len(pool) == 0 or n_det < 0:
        raise ValueError("no valid frames in the window, or negative count")
    rng = np.random.default_rng(seed)
    f1s = np.array([score(rng.choice(pool, size=min(n_det, len(pool)), replace=False),
                          spans, tol, window)["f1"] for _ in range(n_rep)])
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return {"n_detections": int(n_det), "f1_mean": float(f1s.mean()),
            "f1_p2_5": float(lo), "f1_p97_5": float(hi), "n_rep": n_rep}


def block_bootstrap(det_frames, spans, tol: int, window, block: int,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile CIs for precision, recall and F1, resampling BLOCKS of time.

    Punches within seconds of each other are not independent -- combinations,
    fatigue, one camera angle -- so frames are grouped into `block`-frame
    chunks and whole chunks are resampled. The matching is done once, globally,
    and each hit is credited to the block its span starts in.
    """
    if block < 1:
        raise ValueError("block must be >= 1 frame")
    det = np.asarray(det_frames)[_in(det_frames, window)]
    sp = [s for s in spans if window[0] <= s.start <= window[1]]
    pairs = match(det, sp, tol)
    n_blocks = (window[1] - window[0]) // block + 1
    bid = lambda f: (np.asarray(f) - window[0]) // block  # noqa: E731
    n_det = np.bincount(bid(det), minlength=n_blocks)
    n_gt = np.bincount(bid([s.start for s in sp]), minlength=n_blocks)
    tp = np.bincount(bid([sp[i].start for i, _ in pairs]) if pairs else np.zeros(0, int),
                     minlength=n_blocks)
    rng = np.random.default_rng(seed)
    stats = np.full((n_boot, 3), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        t, d, g = tp[idx].sum(), n_det[idx].sum(), n_gt[idx].sum()
        # A draw with no detections, or no labels, has no precision or recall
        # -- it is undefined, not zero. Scoring it 0 drags the interval down
        # whenever the footage has stretches with no punches.
        if d == 0 or g == 0:
            continue
        p, r = t / d, t / g
        stats[b] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    kept = stats[~np.isnan(stats[:, 0])]
    if len(kept) < n_boot // 2:
        raise ValueError(f"only {len(kept)} of {n_boot} resamples were defined; "
                         "blocks are too short for this density of punches")
    lo, hi = np.percentile(kept, [2.5, 97.5], axis=0)
    return {name: [float(lo[i]), float(hi[i])]
            for i, name in enumerate(("precision", "recall", "f1"))} | {
            "block_frames": block, "n_blocks": int(n_blocks), "n_boot": n_boot,
            "n_undefined_dropped": int(n_boot - len(kept))}
