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

from typing import NamedTuple

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


SMOOTH_S = 0.1       # wrist_signal's centred moving-average window, in seconds


def smooth_window(fps: float, smooth_s: float = SMOOTH_S) -> int:
    """The moving-average window of `wrist_signal` in frames: round(smooth_s x fps), at least 1, made odd so it is
    centred. The ONE place it is computed (the signal and the skip mask both call it)."""
    w = max(1, int(round(smooth_s * fps)))
    return w + (w + 1) % 2


def wrist_signal(track: Track, variant: str = "relative", smooth_s: float = SMOOTH_S,
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
    out = np.zeros((len(track.frame_ids), 2))
    for h, a, b, seg in _smoothed_runs(track, smooth_s, min_conf, bridge_s):
        if variant == "relative":
            out[a:b, h] = np.linalg.norm(np.gradient(seg, axis=0) * track.fps, axis=-1)
        else:
            out[a:b, h] = np.clip(np.gradient(np.linalg.norm(seg, axis=-1), axis=0) * track.fps, 0, None)
    return out


def _smoothed_runs(track: Track, smooth_s: float, min_conf: float, bridge_s: float):
    """(hand, first row, end row, smoothed wrist-minus-shoulder in body lengths) for every run
    of consecutive trustworthy frames of each hand, holes up to `bridge_s` filled by a straight
    line first. The ONE place the positions behind wrist_signal and extension_mask are made."""
    scale = body_scale(track, min_conf)
    w = smooth_window(track.fps, smooth_s)  # odd, so the window is centred
    max_gap = int(round(bridge_s * track.fps))
    for h, (sh, wr) in enumerate(HANDS):
        rel = (track.xy[:, wr] - track.xy[:, sh]) / scale
        for lo, hi, ok, filled in _hand_runs(track, h, min_conf, max_gap):
            r = rel[lo:hi].copy()
            if filled.any():  # straight line across a short hole in this hand
                idx = np.arange(hi - lo)
                for c in range(2):
                    r[filled, c] = np.interp(idx[filled], idx[~filled & ok], r[~filled & ok, c])
            for a, b in _runs(ok):
                yield h, lo + a, lo + b, _smooth(r[a:b], w)


def extension_mask(track: Track, window_s: float, smooth_s: float = SMOOTH_S,
                   min_conf: float = MIN_CONF, bridge_s: float = BRIDGE_S) -> np.ndarray:
    """(F, 2) bool: is each hand's wrist moving AWAY from its shoulder at each frame.

    A punch extends the arm; the other hand's retraction during it shortens. With d the
    shoulder-to-wrist distance (the same smoothed, hole-filled positions as wrist_signal)
    and k = max(1, round(window_s x fps / 2)), frame i is True iff frames i-k and i+k lie
    in the same run of trustworthy frames as i and d[i+k] - d[i-k] > 0. Over a window,
    not one frame difference, so pose jitter at the turn of a punch does not decide it.

    Fails closed: a frame whose window leaves its run (a missing joint, a pose gap, the
    run's edge), or where d is not finite, is False. Missing data never passes."""
    if isinstance(window_s, bool) or not (np.isfinite(window_s) and window_s > 0):
        raise ValueError(f"window_s must be a positive number of seconds, got {window_s!r}")
    k = max(1, int(round(window_s * track.fps / 2)))
    out = np.zeros((len(track.frame_ids), 2), dtype=bool)
    for h, a, b, seg in _smoothed_runs(track, smooth_s, min_conf, bridge_s):
        if b - a <= 2 * k:
            continue
        d = np.linalg.norm(seg, axis=-1)
        rise = d[2 * k:] - d[:-2 * k]
        out[a + k:b - k, h] = np.isfinite(rise) & (rise > 0)
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
           min_gap_s: float, *, eligible: "np.ndarray | None" = None,
           per_hand: bool = False, cross_gap_s: "float | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """Peaks of wrist speed above `threshold`. Returns (video frame ids, hand: 0
    left / 1 right), in (frame, hand) order.

    Pooled (the V6-tuned rule, default): peaks of the FASTER hand's speed, at least
    `min_gap_s` apart whichever hand threw them. A jab-cross faster than that is
    one detection.
    Per hand (`per_hand=True`): each hand's own peaks, at least `min_gap_s` apart
    within that hand, so alternating combinations are seen as separate punches. On
    Brendan's own labelled clips, 40% and 32% of punches follow the previous one
    within 0.4 s, but only 11% and 0% follow one by the same hand (2026-10-19). The
    threshold and the gap are the pooled rule's, unchanged.

    `eligible` ((F, 2) bool, default None = every frame): a frame whose hand is not
    eligible can be a neighbour in the peak test (its signal value is compared as it
    is) but is never a CANDIDATE, and is excluded BEFORE the minimum-gap suppression,
    so it cannot suppress a real peak near it and then be discarded itself. In pooled
    mode a candidate's hand is the faster one there; per hand, it is that hand. None
    is the original computation, untouched. Both options are keyword-only: they used
    to share the fifth positional slot on two branches.

    `cross_gap_s` (pooled only; default None = `min_gap_s`): the minimum gap between two
    peaks of DIFFERENT hands, `min_gap_s` still applying between peaks of the same hand
    (arm 2 of docs/gates/fast-combinations.md). Suppression is find_peaks's: taller peaks
    first, a shorter one dropped if it is within the gap that applies to the pair. Equal to
    `min_gap_s`, the result is the pooled rule's exactly.
    """
    if signal.shape != (len(track.frame_ids), 2):
        raise ValueError(f"signal shape {signal.shape} does not match the track")
    if not (np.isfinite(threshold) and min_gap_s >= 0):
        raise ValueError("threshold must be finite and min_gap_s non-negative")
    if eligible is not None and (eligible.shape != signal.shape or eligible.dtype != bool):
        raise ValueError("eligible must be a bool array shaped like the signal")
    if cross_gap_s is not None:
        if per_hand:
            raise ValueError("cross_gap_s is a pooled-rule option; per hand, hands never suppress each other")
        if isinstance(cross_gap_s, bool) or not (np.isfinite(cross_gap_s) and 0 <= cross_gap_s <= min_gap_s):
            raise ValueError(f"cross_gap_s must lie in [0, min_gap_s], got {cross_gap_s!r}")
    dist = max(1, int(round(min_gap_s * track.fps)))
    frames, hands = [], []
    for lo, hi in track.segments():
        if per_hand:
            for h in (0, 1):
                peaks = _peaks(signal[lo:hi, h], threshold, dist,
                               None if eligible is None else eligible[lo:hi, h])
                frames.append(track.frame_ids[lo + peaks])
                hands.append(np.full(len(peaks), h, dtype=np.int64))
        else:
            best = signal[lo:hi].max(axis=1)
            ok = None
            if eligible is not None:
                ok = eligible[lo:hi][np.arange(hi - lo), signal[lo:hi].argmax(axis=1)]
            if cross_gap_s is None:
                peaks = _peaks(best, threshold, dist, ok)
            else:
                cross = max(1, int(round(cross_gap_s * track.fps)))
                peaks = _peaks_two_gaps(best, signal[lo:hi].argmax(axis=1), threshold, dist, cross, ok)
            frames.append(track.frame_ids[lo + peaks])
            hands.append(signal[lo + peaks].argmax(axis=1))
    f, h = np.concatenate(frames), np.concatenate(hands)
    order = np.lexsort((h, f))
    return f[order], h[order]


def _peaks(x: np.ndarray, threshold: float, dist: int, ok: "np.ndarray | None") -> np.ndarray:
    """find_peaks with a minimum distance, over the candidates `ok` allows (None: all).
    Local maxima are found on the true values; the ineligible ones are dropped; the
    minimum-gap suppression then runs on the survivors alone."""
    from scipy.signal import find_peaks

    if ok is None:
        return find_peaks(x, height=threshold, distance=dist)[0]
    cand, _ = find_peaks(x, height=threshold)
    cand = cand[ok[cand]]
    spikes = np.zeros(len(x))
    spikes[cand] = x[cand]
    return find_peaks(spikes, height=threshold, distance=dist)[0]


def _peaks_two_gaps(x: np.ndarray, hand: np.ndarray, threshold: float, same: int, cross: int,
                    ok: "np.ndarray | None") -> np.ndarray:
    """`_peaks` with a minimum gap that depends on the pair: `same` frames between candidates
    of one hand, `cross` between candidates of different hands. The candidates and the order
    of suppression are find_peaks's own (its distance step, scipy _select_by_peak_distance:
    taller first, by np.argsort of the heights; a pair exactly the gap apart both survive),
    so with same == cross this returns what `_peaks` returns."""
    from scipy.signal import find_peaks

    cand, _ = find_peaks(x, height=threshold)
    if ok is not None:
        cand = cand[ok[cand]]
    keep = np.ones(len(cand), dtype=bool)
    reach = max(same, cross)
    for j in np.argsort(x[cand])[::-1]:
        if not keep[j]:
            continue
        for step in (-1, 1):
            k = j + step
            while 0 <= k < len(cand) and abs(int(cand[k]) - int(cand[j])) < reach:
                if abs(int(cand[k]) - int(cand[j])) < (same if hand[cand[k]] == hand[cand[j]] else cross):
                    keep[k] = False
                k += step
    return cand[keep]


# ------------------------------------------------- a rule's finder, by name ---

# How a rule file's `chosen.finder` block selects the arm (docs/gates/fast-combinations.md).
# A rule with no such block is the pooled rule, exactly as before.
FINDER_MODES = ("per_hand_extension", "cross_hand_gap")


def parse_finder(block, min_gap_s: float) -> dict:
    """A rule file's `finder` block, validated: it is read from disk, so every field is
    checked for type and range and anything unexpected is refused."""
    if not isinstance(block, dict) or not isinstance(block.get("mode"), str):
        raise ValueError("finder: expected an object with a 'mode'")
    mode = block["mode"]
    if mode not in FINDER_MODES:
        raise ValueError(f"finder: unknown mode {mode!r}; expected one of {FINDER_MODES}")
    key = "extension_window_s" if mode == "per_hand_extension" else "cross_gap_s"
    if set(block) != {"mode", key}:
        raise ValueError(f"finder: mode {mode} takes exactly 'mode' and '{key}', got {sorted(block)}")
    v = block[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise ValueError(f"finder: {key} must be a finite number, got {v!r}")
    hi = 1.0 if mode == "per_hand_extension" else min_gap_s
    if not 0 < v <= hi:
        raise ValueError(f"finder: {key} must lie in (0, {hi}], got {v}")
    return {"mode": mode, key: float(v)}


def detect_rule(track: Track, signal: np.ndarray, rule: dict) -> "tuple[np.ndarray, np.ndarray]":
    """Detections by a loaded rule (find_punches.load_rule): the pooled rule, or the arm its
    `finder` block names. The one place a rule is turned into a call to `detect`."""
    f = rule.get("finder")
    thr, gap = rule["threshold"], rule["min_gap_s"]
    if f is None:
        return detect(track, signal, thr, gap)
    if f["mode"] == "per_hand_extension":
        return detect(track, signal, thr, gap, per_hand=True,
                      eligible=extension_mask(track, f["extension_window_s"]))
    if f["mode"] == "cross_hand_gap":
        return detect(track, signal, thr, gap, cross_gap_s=f["cross_gap_s"])
    raise ValueError(f"unknown finder mode {f['mode']!r}")


def rule_per_hand(rule: dict) -> bool:
    """Whether a rule's detections are per-hand peaks (each a peak of its own hand only) or
    pooled ones (each also a peak of the faster hand's speed): what classify_punches checks."""
    f = rule.get("finder")
    return f is not None and f["mode"] == "per_hand_extension"


# A label that is a single frame marks one instant somewhere inside a punch (a click in
# the labelling tool, not a drag from start to end). The finder reports the wrist-speed
# peak, which can sit anywhere in the same punch, so a point label matches a detection
# within half a punch. Half of 0.4 s: the training clips' median punch is 0.375 s, and the
# per-clip windows on Brendan's clips are 0.39 and 0.41 s. Chosen 2026-10-19 after the
# scores at 0.1 s and 0.2 s had been seen on those clips; both are in docs/quality-loop.md.
POINT_TOL_S = 0.2


def for_matching(spans: "list[Span]", tol: int, point_tol: int) -> "list[Span]":
    """Spans as `match` should see them: a single-frame span (a point label) is widened
    so that, with `match`'s own `tol` on top, it matches within `point_tol` frames.
    Start is kept at frame 0 or later, so a point near the clip's start still counts as
    inside a window that starts there. The widened span decides ELIGIBILITY only: its
    middle moves when the start is clipped at 0, so attribution uses `centers`."""
    if tol < 0 or point_tol < 0:
        raise ValueError("tolerances must be >= 0")
    extra = max(point_tol - tol, 0)
    return [Span(max(s.start - extra, 0), s.end + extra, s.label) if s.start == s.end else s for s in spans]


def centers(spans: "list[Span]") -> np.ndarray:
    """Each label's own middle (a point label's instant), for `match_nearest`."""
    return np.array([(s.start + s.end) / 2 for s in spans], dtype=np.float64)


def match(det_frames, spans: "list[Span]", tol: int) -> "list[tuple[int, int]]":
    """Maximum one-to-one matching of detections to labelled spans.

    A detection can match a span if it falls within [start - tol, end + tol].
    Spans are taken in order of END, each given the earliest free detection
    that fits -- the greedy that is provably optimal for points-to-intervals.
    Optimality matters here: 212 of V6's 685 spans overlap the next one, and a
    naive first-come assignment undercounts combinations. It is optimal in COUNT
    only: where several maximum matchings exist, WHICH detections it takes is a
    tie-break, not a fact about the footage, so nothing should be read off the
    detections it leaves over (see framing_bounds).

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


def framing_bounds(det_frames, punch_spans: "list[Span]", framing_spans: "list[Span]",
                   tol: int, point_tol: int) -> dict:
    """How many framing marks (motions labelled as NOT punches) drew a detection, as
    bounds that hold whichever matching of punches is chosen (maximum or not).

    Punches are matched first, and several equally large matchings can exist; which
    detections they leave free decides how many framing marks are hit. So:

      at_least  framing marks matched (one-to-one, maximally) to detections that lie
                in NO punch span's reach. No punch matching can take those, so this
                count is reached under every one of them.
      at_most   framing marks matched to ANY detection: no matching can leave more.

    Neither is claimed to be reached: the lower bound ignores detections a punch
    COULD take but need not, so it can sit below the true minimum. Where the two
    agree, the count is exact. Both spans lists are the labels as written; point
    labels are widened here (`for_matching`). `lower_set` is the indices into
    `det_frames` of the detections ONE maximum framing matching used for the lower
    bound: each is out of every punch's reach and on a framing mark, but where a mark
    has several such detections, only one is listed.
    """
    det = np.asarray(det_frames)
    punch = for_matching(punch_spans, tol, point_tol)
    framing = for_matching(framing_spans, tol, point_tol)
    free = np.flatnonzero([not any(s.start - tol <= d <= s.end + tol for s in punch) for d in det])
    lower = match(det[free], framing, tol)
    return {"marks": len(framing_spans), "at_least": len(lower), "at_most": len(match(det, framing, tol)),
            "lower_set": sorted(int(free[di]) for _, di in lower)}


def match_nearest(det_frames, spans: "list[Span]", tol: int,
                  centers: "np.ndarray | None" = None) -> "list[tuple[int, int]]":
    """A maximum one-to-one matching of detections to spans, as `match`, that among
    all maximum matchings has the least total distance from each detection to its
    span's MIDDLE (ties broken by the solver, deterministically).

    Use it wherever a detection's span decides what it is scored against (type, hand,
    overlap); `match` is enough for counts. `match` takes spans in order of end and
    hands each the earliest free detection that fits, which maximises the count but
    not who gets what. In a fast jab-cross the finder (0.4 s minimum gap) reports one
    peak, usually the cross; that peak also fits inside the jab's span plus tolerance,
    and the jab ends first, so `match` credits the jab with the cross's detection. On
    V5 147 of 356 matched detections sat off their span's own wrist-speed peak, 115 of
    them on another labelled punch's: a window correctly read as a cross was scored
    against 'jab'. Counts (recall, precision) are the same under both, and that is
    checked on every call. Solved as an assignment: an infeasible pair costs more than
    every feasible pair together, so the count is maximised first and the distance second.

    `centers` (one per span, default each span's middle) is what distance is measured
    to; eligibility is always [start - tol, end + tol]. Pass the labels' own middles
    when the spans were widened for matching (`for_matching`), so a widened point
    label is still attributed by where it was clicked.

    Returns (span index, detection index) pairs, sorted by span.
    """
    if tol < 0:
        raise ValueError("tol must be >= 0")
    det = np.asarray(det_frames, dtype=np.int64)
    if not len(det) or not spans:
        return []
    from scipy.optimize import linear_sum_assignment

    st = np.array([s.start for s in spans], dtype=np.int64)
    en = np.array([s.end for s in spans], dtype=np.int64)
    ok = (det[None, :] >= st[:, None] - tol) & (det[None, :] <= en[:, None] + tol)
    if centers is None:
        mid = (st + en) / 2
    else:
        mid = np.asarray(centers, dtype=np.float64)
        if mid.shape != (len(spans),) or not np.isfinite(mid).all():
            raise ValueError("centers must be one finite number per span")
    dist = np.abs(det[None, :] - mid[:, None])
    big = float(dist[ok].sum()) + 1.0 if ok.any() else 1.0
    ri, ci = linear_sum_assignment(np.where(ok, dist, big))
    pairs = [(int(s), int(d)) for s, d in zip(ri, ci) if ok[s, d]]
    if len(pairs) != len(match(det, spans, tol)):   # both are maximum matchings
        raise AssertionError("nearest matching is not a maximum matching")
    return sorted(pairs)


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


# --------------------------------------------------------------------------
# Where a punch starts and ends
# --------------------------------------------------------------------------
#
# The classifier was trained on labelled spans, start to end. A fixed window
# around the peak assumes every punch takes the training clips' median time; V6's
# slow instructional hooks and uppercuts take nearly twice as long, and the fixed
# window cost 21 points of accuracy there (results/classify_v6.json, vault
# ADR-0014). Tempo is read off the curve the finder already uses: the WIDTH of the
# punching wrist's speed peak. The punch then runs a fixed number of widths before
# and after the peak. Those two numbers are medians over the training clips
# (classify.training_extent_ratios). What V6 contributed: the level the width is
# measured at (half height) and the choice of this rule were made looking at V6's
# first half, so only its second half is a clean test.
#
# Tried first and dropped (2026-10-11, V6 first half): walking out from the peak
# while speed holds above a fraction of it. Its typical error was a frame, but its
# tails were not: 8 to 42 frames against labels of 11 to 23, merging combinations
# or stopping at the dip at full extension (accuracy 0.538 vs 0.584 for the width).

WIDTH_FRAC = 0.5     # the peak's width is measured at this fraction of its height


def _hand_run_at(track: Track, hand: int, i: int, min_conf: float, bridge_s: float) -> "tuple[int, int] | None":
    """[lo, hi) rows of the run of frames around row i where `hand` has a speed at
    all: consecutive frame ids AND that hand's own joints trustworthy (short holes
    bridged), by the rule wrist_signal uses. None if row i is not in such a run."""
    valid = hand_valid(track, min_conf)[:, hand]
    lo, hi = next((a, b) for a, b in track.segments() if a <= i < b)
    ok, _ = _fill_holes(valid[lo:hi], int(round(bridge_s * track.fps)))
    for a, b in _runs(ok):
        if a <= i - lo < b:
            return lo + a, lo + b
    return None


def peak_width(track: Track, signal: np.ndarray, peak_frame: int, hand: int, frac: float = WIDTH_FRAC,
               min_conf: float = MIN_CONF, bridge_s: float = BRIDGE_S) -> "tuple[float, bool]":
    """(seconds the punching hand's speed stays at or above `frac` of its value at
    `peak_frame`, whether that was cut short by the edge of the hand's signal).

    Each side walks out from the peak while the speed holds, and the crossing is
    interpolated between the last frame above and the first below, so the width is
    not quantised to whole frames (a jab's peak is two or three frames wide at 24
    fps). The walk stays inside the run of frames where this hand HAS a speed: a
    gap in the pose, or the wrist dropping below confidence, is where the signal
    stops, not where the punch does. A side that reaches that edge still above the
    level is CENSORED: its true width is unknown, half a frame is added, and the
    flag is set.

    Measured on the sampled, smoothed signal, so the same punch reads some 5 to 12%
    wider at 24 fps than at 60: a known approximation, small beside the tempo
    differences the width is there to follow.
    """
    if signal.shape != (len(track.frame_ids), 2):
        raise ValueError(f"signal shape {signal.shape} does not match the track")
    if hand not in (0, 1):
        raise ValueError(f"hand must be 0 or 1, got {hand!r}")
    if not (np.isfinite(frac) and 0 < frac < 1):
        raise ValueError("frac must be in (0, 1)")
    i = int(np.searchsorted(track.frame_ids, peak_frame))
    if i >= len(track.frame_ids) or track.frame_ids[i] != peak_frame:
        raise ValueError(f"peak frame {peak_frame} is not a frame of this track")
    s = signal[:, hand]
    run = _hand_run_at(track, hand, i, min_conf, bridge_s)
    thr = frac * s[i]
    if run is None or not thr > 0:
        raise ValueError(f"hand {hand} has no speed at frame {peak_frame}: not its peak")
    lo, hi = run

    def side(step: int) -> "tuple[float, bool]":
        j = i
        while lo <= j + step < hi and s[j + step] >= thr:
            j += step
        k = j + step
        if lo <= k < hi:                      # s[k] < thr <= s[j]: interpolate the crossing
            return abs(j - i) + (s[j] - thr) / (s[j] - s[k]), False
        return abs(j - i) + 0.5, True

    (left, cl), (right, cr) = side(-1), side(1)
    return float((left + right) / track.fps), bool(cl or cr)


class Extent(NamedTuple):
    """A punch's inclusive video frames, and what is known to be wrong with them."""

    start: int
    end: int
    censored: bool   # the speed peak ran into the edge of the hand's signal: the true extent may be longer
    capped: bool     # it was longer than `max_s` and was shortened around the peak


def extent(track: Track, signal: np.ndarray, peak_frame: int, hand: int, lead_widths: float,
           tail_widths: float, frac: float = WIDTH_FRAC, max_s: "float | None" = None) -> Extent:
    """The punch whose wrist-speed peak is at `peak_frame`: `lead_widths` peak-widths
    before it to `tail_widths` after. Frames outside the pose are allowed here; the
    window cutter marks them empty.

    An extent longer than `max_s` is shortened on BOTH sides in the same
    proportion, so the peak stays where the classifier expects it. (Cutting only
    the tail, or only the head, would hand the classifier a window that might not
    contain the punch at all.)
    """
    if not (np.isfinite(lead_widths) and np.isfinite(tail_widths) and lead_widths >= 0 and tail_widths >= 0
            and lead_widths + tail_widths > 0):
        raise ValueError("lead and tail must be non-negative widths, not both zero")
    if max_s is not None and not (np.isfinite(max_s) and max_s > 0):
        raise ValueError("max_s must be positive")
    w, censored = peak_width(track, signal, peak_frame, hand, frac)
    lead, tail = lead_widths * w, tail_widths * w
    capped = max_s is not None and lead + tail > max_s
    if capped:
        k = max_s / (lead + tail)
        lead, tail = lead * k, tail * k
    start = int(np.floor(peak_frame - lead * track.fps + 0.5))
    end = int(np.floor(peak_frame + tail * track.fps + 0.5))
    if capped:   # rounding both ends outward can add a frame; never exceed the cap
        while (end - start) / track.fps > max_s and end > peak_frame:
            end -= 1
    return Extent(start, max(end, start + 1), bool(censored), bool(capped))   # plain bools: they are counted and written to JSON


def span_iou(a: "tuple[int, int]", b: "tuple[int, int]") -> float:
    """Intersection over union of two inclusive frame spans."""
    if a[1] < a[0] or b[1] < b[0]:
        raise ValueError("a span ends before it starts")
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return max(inter, 0) / union
