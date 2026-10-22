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
HANDS = ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST))

# With joint confidence (YOLO11-pose), each hand is judged on its OWN joints: a far
# wrist hidden behind the body (side-on and three-quarter views) must not erase
# the near hand's punches, which a whole-frame "both wrists" rule did -- on
# IMG_4395 it threw away 13 points of frames. Without confidence (BoxingVI's V6)
# every frame counts for both hands, exactly as before.
MIN_CONF = 0.3        # a joint below this is a guess (pose_yolo.USABLE_MIN_CONF)
BODY_MIN_JOINTS = 8   # ... and a frame with fewer solid joints is no body at all
# Dropouts this short are interpolated instead of splitting the signal: 3 frames
# at 60 fps (the one- and two-frame wrist flicker seen on IMG_4395), 1 at 24, so
# V6's gaps (3 and 48 frames) stay gaps and its results are unchanged.
BRIDGE_S = 0.05


def hand_valid(track: Track, min_conf: float = MIN_CONF) -> np.ndarray:
    """(F, 2) bool: is each hand's shoulder and wrist trustworthy in each frame."""
    n = len(track.frame_ids)
    if track.conf is None:
        return np.ones((n, 2), dtype=bool)
    body = (track.conf >= min_conf).sum(axis=1) >= BODY_MIN_JOINTS
    return np.stack([body & (track.conf[:, s] >= min_conf) & (track.conf[:, w] >= min_conf)
                     for s, w in HANDS], axis=1)


def body_scale(track: Track, min_conf: float = MIN_CONF) -> float:
    """Median shoulder-midpoint to hip-midpoint distance, in pixels, over frames
    whose shoulders and hips are trustworthy (all frames when there is no confidence)."""
    rows = slice(None)
    if track.conf is not None:
        rows = (track.conf[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] >= min_conf).all(axis=1)
        if not rows.any():
            raise ValueError("no frame has confident shoulders and hips: no usable body in this track")
    sh = (track.xy[rows, L_SHOULDER] + track.xy[rows, R_SHOULDER]) / 2
    hp = (track.xy[rows, L_HIP] + track.xy[rows, R_HIP]) / 2
    scale = float(np.median(np.linalg.norm(sh - hp, axis=1)))
    if not scale > 1.0:
        raise ValueError(f"torso length {scale:.2f}px: no usable body in this track")
    return scale


def bridge(track: Track, max_gap: int) -> Track:
    """Fill frame-id gaps of at most `max_gap` missing frames by linear
    interpolation. An inserted frame gets the LOWER confidence of its two
    neighbours, so a hand counts there only if it counts on both sides. Longer
    gaps are left alone: across them nothing is ever differenced."""
    if max_gap < 0:
        raise ValueError("max_gap must be >= 0")
    ids = track.frame_ids
    gaps = np.flatnonzero((np.diff(ids) > 1) & (np.diff(ids) <= max_gap + 1))
    if max_gap == 0 or len(gaps) == 0:
        return track
    new = np.concatenate([np.arange(ids[i] + 1, ids[i + 1]) for i in gaps])
    all_ids = np.union1d(ids, new)
    flat = track.xy.reshape(len(ids), -1)
    xy = np.stack([np.interp(all_ids, ids, flat[:, k]) for k in range(flat.shape[1])],
                  axis=1).reshape(len(all_ids), *track.xy.shape[1:])
    conf = None
    if track.conf is not None:
        right = np.searchsorted(ids, all_ids)          # first original id >= this one
        left = np.clip(right - 1, 0, len(ids) - 1)
        right = np.clip(right, 0, len(ids) - 1)
        orig = np.isin(all_ids, ids)
        conf = np.where(orig[:, None], track.conf[right],
                        np.minimum(track.conf[left], track.conf[right]))
    return Track(frame_ids=all_ids.astype(np.int64), xy=xy, fps=track.fps,
                 source=track.source, conf=conf)


def _fill_holes(valid: np.ndarray, max_gap: int) -> "tuple[np.ndarray, np.ndarray]":
    """Interior runs of False no longer than `max_gap`, with True on both sides,
    become True. Returns (new mask, which entries were filled)."""
    out = valid.copy()
    if max_gap <= 0 or valid.all() or not valid.any():
        return out, np.zeros_like(valid)
    d = np.diff(np.r_[1, valid.astype(int), 1])
    starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
    for a, b in zip(starts, ends):
        if 0 < a and b < len(valid) and b - a <= max_gap:
            out[a:b] = True
    return out, out & ~valid


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


def wrist_signal(track: Track, variant: str = "relative", smooth_s: float = 0.1,
                 min_conf: float = MIN_CONF, bridge_s: float = BRIDGE_S) -> np.ndarray:
    """(F, 2) wrist speed per hand (left, right), in body lengths per second.

    Each hand is computed on its own runs of consecutive, trustworthy frames
    (`hand_valid`): a gap in the pose, or in that hand, is never differenced
    across, because a jump over missing frames would read as a punch. Holes in
    a hand no longer than `bridge_s` inside a run are interpolated first. A hand
    with no trustworthy run scores zero. Without confidence (V6) every frame is
    valid and this is exactly the per-segment computation it always was.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    scale = body_scale(track, min_conf)
    w = max(1, int(round(smooth_s * track.fps)))
    w += (w + 1) % 2  # odd, so the window is centred
    max_gap = int(round(bridge_s * track.fps))
    out = np.zeros((len(track.frame_ids), 2))
    for h, (sh, wr) in enumerate(HANDS):
        rel = (track.xy[:, wr] - track.xy[:, sh]) / scale
        for lo, hi, ok, filled in _hand_runs(track, h, min_conf, max_gap):
            r = rel[lo:hi].copy()
            if filled.any():  # straight line across a short hole in this hand
                idx = np.arange(hi - lo)
                for c in range(2):
                    r[filled, c] = np.interp(idx[filled], idx[~filled & ok], r[~filled & ok, c])
            for a, b in _runs(ok):
                seg = _smooth(r[a:b], w)
                if variant == "relative":
                    out[lo + a:lo + b, h] = np.linalg.norm(np.gradient(seg, axis=0) * track.fps, axis=-1)
                else:
                    out[lo + a:lo + b, h] = np.clip(np.gradient(np.linalg.norm(seg, axis=-1), axis=0)
                                                    * track.fps, 0, None)
    return out


def _runs(ok: np.ndarray) -> "list[tuple[int, int]]":
    """[a, b) runs of True at least 2 long: the only places a speed exists."""
    d = np.diff(np.r_[0, ok.astype(int), 0])
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= 2]


def _hand_runs(track: Track, h: int, min_conf: float, max_gap: int):
    """(lo, hi, ok, filled) per frame-id segment for hand h: the ONE definition of
    where a hand has a signal, shared by wrist_signal and signal_coverage."""
    valid = hand_valid(track, min_conf)[:, h]
    for lo, hi in track.segments():
        ok, filled = _fill_holes(valid[lo:hi], max_gap)
        yield lo, hi, ok, filled


def signal_coverage(track: Track, min_conf: float = MIN_CONF, bridge_s: float = BRIDGE_S,
                    span: "tuple[int, int] | None" = None) -> dict:
    """Share of the frames in `span` (default: first to last pose frame) at which
    each hand has a speed, by exactly the rule wrist_signal uses. Frames outside
    the track (never detected) count as not covered, so different tracks of the
    same clip are compared on one denominator."""
    lo_f, hi_f = span if span is not None else (int(track.frame_ids[0]), int(track.frame_ids[-1]))
    total = hi_f - lo_f + 1
    max_gap = int(round(bridge_s * track.fps))
    got = np.zeros(2)
    for h in range(2):
        for lo, hi, ok, _ in _hand_runs(track, h, min_conf, max_gap):
            for a, b in _runs(ok):
                f = track.frame_ids[lo + a:lo + b]
                got[h] += int(((f >= lo_f) & (f <= hi_f)).sum())
    return {"frames": total, "left": got[0] / total, "right": got[1] / total}


def signal_frames(track: Track, min_conf: float = MIN_CONF, bridge_s: float = BRIDGE_S) -> np.ndarray:
    """(F, 2) bool: does each hand have a speed in each of the track's frames, by exactly the
    rule `wrist_signal` and `signal_coverage` use (same runs, same bridging). The union over
    the two hands (`.any(axis=1)`) is the exposure of a detector that takes the faster hand."""
    max_gap = int(round(bridge_s * track.fps))
    out = np.zeros((len(track.frame_ids), 2), dtype=bool)
    for h in range(2):
        for lo, hi, ok, _ in _hand_runs(track, h, min_conf, max_gap):
            for a, b in _runs(ok):
                out[lo + a:lo + b, h] = True
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
    chunks and whole chunks are resampled. The matching is done once, globally.

    A hit is credited TWICE: to its detection's block for precision, and to its
    span's block for recall. With one shared credit, a hit whose detection and
    span fall in different blocks could be drawn without its detection, and
    precision could exceed 1.

    Draws with no detections or no labels are dropped as undefined, so the
    interval is conditional on draws containing at least one punch.
    """
    if block < 1:
        raise ValueError("block must be >= 1 frame")
    det = np.asarray(det_frames)[_in(det_frames, window)]
    sp = [s for s in spans if window[0] <= s.start <= window[1]]
    pairs = match(det, sp, tol)
    n_blocks = (window[1] - window[0]) // block + 1
    bid = lambda f: (np.asarray(f, dtype=np.int64) - window[0]) // block  # noqa: E731
    n_det = np.bincount(bid(det), minlength=n_blocks)
    n_gt = np.bincount(bid([s.start for s in sp]), minlength=n_blocks)
    tp_det = np.bincount(bid([det[j] for _, j in pairs]), minlength=n_blocks)
    tp_span = np.bincount(bid([sp[i].start for i, _ in pairs]), minlength=n_blocks)
    rng = np.random.default_rng(seed)
    stats = np.full((n_boot, 3), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        d, g = n_det[idx].sum(), n_gt[idx].sum()
        # A draw with no detections, or no labels, has no precision or recall
        # -- it is undefined, not zero. Scoring it 0 drags the interval down
        # whenever the footage has stretches with no punches.
        if d == 0 or g == 0:
            continue
        p, r = tp_det[idx].sum() / d, tp_span[idx].sum() / g
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
