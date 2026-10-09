"""Did the referee label every punch? Per boxer, by the wrist-speed rule.

The Olympic Boxing paper has a licensed referee judging 312,774 frames. Whether that
means EVERY punch was labelled is a claim, not a fact. A test that can fail: take each
linked boxer's own per-frame pose, run the per-hand wrist-speed rule (punchfind, the
rule tuned on V6 and saved in results/punch_finder_v6.json), and ask of every burst
whether a labelled punch plausibly accounts for it.

"Accounts for it" is a ONE-TO-ONE MATCH with attribution, not temporal proximity:
  * every labelled event is OWNED by boxer k when the punch was credited
    (olympic.attribute) to a track id that is boxer k's identity somewhere in the
    event's frames, otherwise it is UNATTRIBUTED (ambiguous, unreached, or credited
    to a track that is not one of the two selected boxer identities: no owner among them);
  * each boxer's bursts are matched first to that boxer's OWN labels, each label to at
    most one burst, within TOL_S of the label's frames, as an assignment that is OPTIMAL
    WITHIN THIS PHASE (most bursts explained, then least total frame gap; a greedy
    nearest-first pass can spend a broad label on one burst and strand the next); a label
    thrown by the other boxer never explains this boxer's burst;
  * the bursts still unmatched, of both boxers together, are then matched in the same way,
    one-to-one, to the UNATTRIBUTED labels (each used once across both boxers), again
    optimal within the phase;
  * the composition of the two phases is NOT necessarily the global maximum of explained
    bursts: an owned label may take a burst that an unattributed label could also have
    explained, stranding another burst only the owned label could reach (tolerance 5, bursts
    at 100 and 120, owned label [99,120], unattributed label [95,95]: the owned phase takes
    100 and 120 stays unexplained; the reverse split explains both). That is the documented
    policy (ownership first) and a test pins it;
  * exact ties go to the burst that sorts first by (frame, boxer), which is A on a
    simultaneous tie, so ties favour A;
  * what is left is "unexplained": a burst no label accounts for.

An unexplained burst is a CANDIDATE for a missed label, nothing more. The rule misses
real punches and fires on non-punches (on V6's tuning half its precision was 0.46: a
feint, a guard reset, a slip or a wave to the corner also moves a wrist fast), so the
unexplained count is neither an upper nor a lower bound on unlabelled punches until a
human has classified a random sample of them. This module draws that sample (seeded,
simple random over ALL unexplained bursts, every burst with inclusion probability n/N)
and judges none of it. The hand of the matched label is recorded (`hand_agrees`) as
information only: the left/right convention holds in 78% of credited punches, so it is
not required.

No signal crosses a fragment join. A boxer's rows are split into SEGMENTS: a new one starts
wherever the track id changes or a frame is missing. The bridging, smoothing, differencing
and peak finding all run inside one segment, so a pose jump at an id handover (two
different detections of the same body, a torso apart) is never read as wrist speed.

EXPOSURE. A burst is only seen where the linking identified the boxer, in a segment long
enough for the rule, with a confident wrist. Those frames (bridged ones included) are the
boxer's exposure, an explicit mask (see `find_bursts`), and it is the ONE denominator of every
rate of the bursts and of the own labels that touch it: those are all counted against it, and a
burst outside it is an error. Identified minutes (all rows the linking names) are descriptive
only; whole-video label density is another population and is never compared with these rates.

Caveat carried into the report: the rule's threshold is V6's (AlphaPose, 24 fps, hooks and
uppercuts of one athlete).

CREDITED FRAGMENTS (ADR-0012 Option 2, `find_bursts_credited`): the same test with NO linking and no
boxer identity. The population is every track id that is credited with at least `min_punches`
labelled punches; each such id (a FRAGMENT) is one person for its whole length, because ByteTrack
never gives one id to two people by design (an id switch inside a fragment is the tracker's residual
error and is not detected here). A label is OWNED by the fragment it is credited to; matching, exposure
and the sample rules are the ones above, with the fragment in the place of the boxer.

LIMITS, stated once and carried into every output: this sees ONLY credited fragments. Boxer time in a
fragment with no credited punch is outside it, and so are punches too close to call (attribution
ambiguity): restricting to credited fragments loses missed punches outside them, so coverage can only
lose. In the other direction, fragment-level detections can DUPLICATE one physical punch when two
overlapping ids follow the same person. So neither a bout-level upper nor a lower bound on missed labels
follows without de-duplication; `possible_duplicates` counts the unexplained bursts that have a likely
twin (see `find_bursts_credited`). A clean result is weak evidence of completeness; a dirty one (many
sampled bursts classified as real unlabelled punches) shows labels are missing inside the credited
fragments, though the count may include duplicates. A credited fragment can also be a non-boxer wrongly
credited (a referee, a spectator): the fragments with exactly one credited punch are the least certain,
and are counted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import olympic, punchfind
from .data import Track
from .linking import Linking
from .pose_yolo import PoseTable

TOL_S = 0.1          # the matching tolerance used on V6 and by find_punches.py
SAMPLE_N = 40
SAMPLE_SEED = 20261007
HAND_NAMES = ("left", "right")
MATCHES = ("own", "unattributed", "none")
# A segment shorter than the rule's smoothing window (0.1 s: 5 frames at 50 fps) cannot be
# smoothed or differenced meaningfully; it is skipped and counted, not guessed at.
MIN_SEGMENT_S = 0.1


@dataclass(frozen=True)
class Burst:
    video: str
    frame: int
    time_s: float
    boxer: str          # "A" / "B"
    hand: str           # the boxer's own left / right (COCO)
    peak_speed: float   # body lengths per second
    match: str = "none"           # "own" | "unattributed" | "none": see the module docstring
    hand_agrees: "bool | None" = None   # matched label's hand == this burst's hand; None if unmatched. Information only.
    track_id: "int | None" = None       # credited mode only: the ByteTrack id (the fragment) the burst is in
    box: "tuple | None" = None          # credited mode only: that fragment's box (centre x, centre y, height) at the burst frame, pixels
    possible_duplicate: bool = False    # credited mode only: unexplained, with an unexplained twin in another fragment (see DUP_IOU)

    def __post_init__(self) -> None:
        if self.match not in MATCHES:
            raise ValueError(f"match must be one of {MATCHES}, got {self.match!r}")

    @property
    def explained(self) -> bool:
        return self.match != "none"


def boxer_rows(table: PoseTable, linking: Linking, k: int) -> np.ndarray:
    """Row indices of boxer k, in frame order. A row belongs to the boxer only if the linking
    NAMES a track id in that frame (membership >= 0: -1 means "nobody") AND that id is the
    row's. A negative id in a table must never match the -1 sentinel."""
    if linking.n_frames != table.n_frames or linking.source != table.source:
        raise ValueError("the linking was made from a different pose table than this one")
    want = linking.membership[table.frame, k]
    rows = np.flatnonzero((want >= 0) & (want == table.track_id))
    return rows[np.argsort(table.frame[rows], kind="stable")]


def _track_of(table: PoseTable, rows: np.ndarray, name: str) -> Track:
    return Track(frame_ids=table.frame[rows].astype(np.int64), xy=table.xy[rows].astype(np.float64),
                 fps=table.fps, source=name, conf=table.conf[rows].astype(np.float64))


def boxer_track(table: PoseTable, linking: Linking, k: int) -> "Track | None":
    """Boxer k's rows in frame order, as one Track with confidence; None under two frames.
    The fragments are JOINED here (the ids are not kept), so nothing may be differenced
    across it: use `boxer_segments` for any signal."""
    rows = boxer_rows(table, linking, k)
    if len(rows) < 2:
        return None
    return _track_of(table, rows, f"{table.source}#boxer{'AB'[k]}")


def _split_rows(table: PoseTable, rows: np.ndarray, max_gap: int) -> "list[np.ndarray]":
    """`rows` (frame order) cut into segments wherever the track id changes or the frames are more
    than `max_gap` missing frames apart. The one place a segment is defined."""
    if len(rows) == 0:
        return []
    tid, fr = table.track_id[rows], table.frame[rows]
    cut = np.flatnonzero((np.diff(tid) != 0) | (np.diff(fr) > max_gap + 1)) + 1
    return [rows[a:b] for a, b in zip(np.r_[0, cut], np.r_[cut, len(rows)])]


def boxer_segments(table: PoseTable, linking: Linking, k: int, max_gap: int = 0) -> "list[np.ndarray]":
    """Boxer k's rows split into SEGMENTS (arrays of row indices, frame order): a new segment
    starts wherever the track id changes (a handover: two detections of possibly different
    poses, never differenced) or the frames of ONE id are more than `max_gap` missing frames
    apart. A shorter hole inside one id is the same person and is left for `punchfind.bridge`,
    as on the footage the rule was tuned on."""
    if isinstance(max_gap, bool) or not isinstance(max_gap, (int, np.integer)) or max_gap < 0:
        raise ValueError("max_gap must be a whole number >= 0")
    return _split_rows(table, boxer_rows(table, linking, k), max_gap)


def event_owner(attr, linking: Linking) -> "int | None":
    """0 / 1 when the punch was credited to a track id that is that boxer's identity in at
    least one frame of the event's span; None (unattributed) otherwise."""
    if attr.track_id is None:
        return None
    e = attr.event
    if e.first >= linking.n_frames:
        return None
    seg = linking.membership[e.first:min(e.last, linking.n_frames - 1) + 1]
    for k in (0, 1):
        if (seg[:, k] == attr.track_id).any():
            return k
    return None


def span_gap(frame: int, event) -> int:
    """Frames between `frame` and the labelled punch's [first, last]; 0 inside it."""
    return max(event.first - int(frame), int(frame) - event.last, 0)


def _match(bursts: "list[tuple]", labels: "list[int]", events, tol: int, taken: set) -> "dict[int, int]":
    """One-to-one matching of bursts to the labels offered in THIS call that MAXIMISES the number
    of matched pairs, then minimises their total span gap (an optimal assignment,
    `scipy.optimize.linear_sum_assignment`), so within the call a broad label is never spent on a
    burst that a narrow one could have explained when that leaves another burst unexplained.
    `find_bursts` calls it once per phase (own labels per boxer, then unattributed labels), so the
    result is optimal within each phase and not necessarily the global maximum (see the module
    docstring). `bursts` is [(key, frame)], `labels` indexes into `events`;
    labels already in `taken` are unavailable and the used ones are added to it. Only pairs within
    `tol` frames are feasible. Returns {burst key: label index}.

    Exact ties are resolved by ORDER, not by the solver's whim: bursts are sorted by (frame, key)
    and labels by (first frame, last frame, index) before the cost matrix is built, and the cost
    carries two integer tie-breaks below the gap (matched bursts with a lower sorted position
    first, then labels with a lower position), so of two simultaneous bursts `key=("A", ..)`
    sorts before `("B", ..)` and takes the single label. The tie-breaks are scaled to stay
    strictly below one frame of gap, so they never trade a match or a frame of gap for order.
    Beyond them (a different pair of bursts with the same position sum) the choice is the
    solver's, which is deterministic for the same input."""
    avail = sorted((li for li in labels if li not in taken),
                   key=lambda li: (events[li].first, events[li].last, li))
    pool = sorted(bursts, key=lambda kf: (kf[1], kf[0]))
    if not pool or not avail:
        return {}
    nb, nl = len(pool), len(avail)
    r = min(nb, nl)
    gap = np.array([[span_gap(f, events[li]) for li in avail] for _k, f in pool], dtype=np.int64)
    feasible = gap <= tol
    if not feasible.any():
        return {}
    k1 = r * nl + 1                              # > any sum of label positions in a matching
    k2 = r * nb * k1 + k1                        # > any sum of (burst position * k1 + label position)
    cost = (gap * k2 + np.arange(nb, dtype=np.int64)[:, None] * k1 + np.arange(nl, dtype=np.int64)[None, :])
    big = int(cost[feasible].max()) * r + 1      # > the cost of ANY assignment made of feasible pairs
    cost = np.where(feasible, cost, big)
    rows, cols = linear_sum_assignment(cost)
    got: "dict[int, int]" = {}
    for i, j in zip(rows.tolist(), cols.tolist()):
        if feasible[i, j]:
            got[pool[i][0]] = avail[j]
            taken.add(avail[j])
    return got


def check_skips(skips, n_frames: int) -> "np.ndarray | None":
    """`skips` (None, or the sorted frame ids `skips.find_skips` returns) as an int64 array, refused
    unless it is a 1-D whole-number array, strictly increasing, with every id in 1..n_frames-1
    (frame 0 has no previous frame and cannot be a skip)."""
    if skips is None:
        return None
    a = np.asarray(skips)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError("skips must be a 1-D array of whole frame numbers")
    a = a.astype(np.int64)
    if a.size and (a.min() < 1 or a.max() > n_frames - 1):
        raise ValueError(f"skips must lie in 1..{n_frames - 1}")
    if a.size > 1 and not (np.diff(a) > 0).all():
        raise ValueError("skips must be sorted and distinct")
    return a


def skip_mask_half(fps: float, smooth_s: float = punchfind.SMOOTH_S) -> int:
    """How many frames on each side of a skip f the wrist-speed signal can be contaminated by the one-frame
    jump between f-1 and f, derived from the rule itself.

    `wrist_signal` smooths the position with a centred moving average of `w` = `punchfind.smooth_window`
    frames (half-width h = w // 2: 2 at 50 fps and 0.1 s), then takes `np.gradient`, which at frame j reads
    the smoothed values at j-1 and j+1. A step between f-1 and f enters the smoothed value at i exactly
    when i-h <= f-1 and i+h >= f, that is i in [f-h, f+h-1]; the gradient at j is contaminated when j-1 or
    j+1 is in that range, that is j in [f-h-1, f+h]. The mask is the symmetric h + 1 either side:
    [f-(h+1), f+(h+1)], a superset of the contaminated range by one frame on the right. At 50 fps this is
    3, but it is computed, so it follows both `smooth_s` and the frame rate."""
    return punchfind.smooth_window(fps, smooth_s) // 2 + 1


def skip_mask(track: Track, orig_frames: np.ndarray, skips: np.ndarray, half: int, bridge: int) -> np.ndarray:
    """(F, 2) bool over a BRIDGED track's frames: True where the wrist speed of that hand cannot be trusted
    because a camera skip is within reach, so the signal there is set to 0 and the frame leaves the exposure.

    For each skip f: the frames f-half .. f+half (see `skip_mask_half`). Two things make a jump reach further than
    that, and both are covered: (1) f lies in a hole of the table that `punchfind.bridge` interpolated across
    (`orig_frames` are the rows before bridging, p the last before f and q the first from f on): the jump becomes
    a ramp over the whole hole, so the mask runs from p+1-half to q-1+half; (2) the zone meets a hole in ONE
    hand's trustworthy frames (at most `bridge` frames, with trustworthy frames on both sides) that
    `wrist_signal` interpolates across: the zone grows to cover that hole plus `half` beyond it, for that
    hand only. Growth repeats until nothing more is reached."""
    n = len(track.frame_ids)
    m = np.zeros((n, 2), dtype=bool)
    if skips is None or len(skips) == 0 or n == 0:
        return m
    fr = track.frame_ids
    off = int(fr[0])
    valid = punchfind.hand_valid(track)
    holes = []                                           # per hand: [(first, last)] frame ids of interpolated holes
    for h in range(2):
        _, filled = punchfind._fill_holes(valid[:, h], bridge)
        d = np.diff(np.r_[0, filled.astype(int), 0])
        holes.append([(int(fr[a]), int(fr[b - 1])) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))])
    for f in skips.tolist():
        if f < fr[0] - half or f > fr[-1] + half:
            continue
        lo, hi = f - half, f + half
        j = int(np.searchsorted(orig_frames, f))
        if 0 < j < len(orig_frames):
            lo, hi = min(lo, int(orig_frames[j - 1]) + 1 - half), max(hi, int(orig_frames[j]) - 1 + half)
        for h in range(2):
            a, b = lo, hi
            while True:
                a0, b0 = a, b
                for ra, rb in holes[h]:
                    if ra <= b and rb >= a:
                        a, b = min(a, ra - half), max(b, rb + half)
                if (a, b) == (a0, b0):
                    break
            m[max(a - off, 0):max(min(b - off, n - 1) + 1, 0), h] = True
    return m


def _scan_segments(table: PoseTable, segs: "list[np.ndarray]", rule: dict, who: str, bridge: int,
                   min_rows: int, skips: "np.ndarray | None" = None, smooth_s: float = punchfind.SMOOTH_S) -> dict:
    """The rule on each segment (bridge, wrist_signal, detect) and the EXPOSURE it ran on, for one
    person: the single implementation behind `find_bursts` (a linked boxer) and
    `find_bursts_credited` (a credited fragment). Returns {"mask", "raw" [(frame, hand index, peak
    speed)], "segments", "short_segments", "bodyless_segments", "valid" (frames with a valid
    signal: left, right)}. Every burst is asserted to lie in the mask (RuntimeError otherwise).

    `skips` (sorted frame ids where the camera dropped a frame, `skips.find_skips`; None = ignore them) are
    MASKED, not cut: the segments, the body scale and the smoothing are exactly what they are without skips.
    After `wrist_signal`, every frame within `skip_mask_half` of a skip (`skip_mask`) gets signal 0. The
    frames within ONE MORE of a skip leave the exposure (per hand) and cannot be a burst, so they are in
    neither the numerator nor the denominator of any rate: the extra frame is the peak test's own reach.
    `detect` calls frame j a peak by comparing it with j-1 and j+1; next to a zeroed frame the first
    unmasked frame would be a "peak" for being higher than a zero, and a punch whose true peak is under
    the mask would come back as a burst at the mask's edge (measured: it doubled the burst rate there).
    That frame keeps its true signal, so the next one is compared with it, not with a zero. With None the
    result is exactly what it was without the argument."""
    skips = check_skips(skips, table.n_frames)
    half = skip_mask_half(table.fps, smooth_s)
    out = {"mask": np.zeros(table.n_frames, dtype=bool), "raw": [], "segments": 0, "short_segments": 0,
           "bodyless_segments": 0, "valid": np.zeros(2)}
    mask = out["mask"]
    for seg in segs:
        out["segments"] += 1
        if len(seg) < min_rows:
            out["short_segments"] += 1
            continue
        track = punchfind.bridge(_track_of(table, seg, f"{table.source}#{who}"), bridge)
        try:
            sig = punchfind.wrist_signal(track, rule["variant"], smooth_s)
        except ValueError:                           # no confident body anywhere in this segment
            out["bodyless_segments"] += 1
            continue
        have = punchfind.signal_frames(track)
        gone = None
        if skips is not None and len(skips):
            masked = skip_mask(track, table.frame[seg], skips, half, bridge)
            sig[masked] = 0.0
            gone = skip_mask(track, table.frame[seg], skips, half + 1, bridge)   # + the peak test's own reach
            have = have & ~gone
        out["valid"] += have.sum(axis=0)
        mask[track.frame_ids[have.any(axis=1)]] = True
        frames, hands = punchfind.detect(track, sig, rule["threshold"], rule["min_gap_s"])
        row = {int(f): i for i, f in enumerate(track.frame_ids)}
        for f, h in zip(frames, hands):
            if gone is not None and gone[row[int(f)], int(h)]:
                continue
            out["raw"].append((int(f), int(h), float(sig[row[int(f)], int(h)])))
    for f, _h, _p in out["raw"]:
        if not mask[f]:
            raise RuntimeError(f"{who}: a burst at frame {f} lies outside the exposure it was "
                               "counted against; numerator and denominator would differ")
    return out


def _attr_key(a) -> tuple:
    return (a.track_id, a.wrist, a.dist, a.runner_up, a.ambiguous, a.hand_clear)


def check_attrs_shape(events, attrs) -> None:
    """One attribution per event, in order, for these very events."""
    if len(attrs) != len(events) or any(a.event is not e for a, e in zip(attrs, events)):
        raise ValueError("attrs must be olympic.attribute(events, table): one attribution per event, in order")


def check_attrs_bound(table: PoseTable, events, attrs) -> None:
    """The attributions must be the ones THIS table produces: recompute `olympic.attribute(events, table)` (default
    parameters) and compare the credited track id, the wrist, both distances and the two flags that ownership and the
    hand test depend on, event by event. Track ids are only labels: attributions made from another tracking of the
    same video can carry the same id set with the people swapped, and nothing else would notice."""
    want = olympic.attribute(events, table)
    for i, (a, w) in enumerate(zip(attrs, want)):
        if _attr_key(a) != _attr_key(w):
            raise ValueError(f"event {i}: the attribution (track {a.track_id}, wrist {a.wrist}) is not what this pose "
                             f"table gives (track {w.track_id}, wrist {w.wrist}): attributions made from another table "
                             "(or with other parameters)?")


def find_bursts(table: PoseTable, linking: Linking, events, attrs, rule: dict, video: str = "",
                tol_s: float = TOL_S, skips: "np.ndarray | None" = None) -> dict:
    """Bursts per boxer, matched one-to-one to labels (see the module docstring), the EXPOSURE
    they were looked for in, and the labels each boxer owns.

    EXPOSURE (the denominator of the burst rates and of the exposure-qualified own-label rate): per boxer, a mask over the video's frames
    that is True on every frame of every PROCESSED segment (not skipped as too short, not
    bodyless) where the bridged track has a valid signal for at least one hand
    (`punchfind.signal_frames(track).any(axis=1)`). Policy: frames that `punchfind.bridge`
    interpolated across a short hole inside one id ARE in the exposure, because the detector
    runs on the bridged track and may place a peak on one; a frame of a skipped segment is NOT,
    although the boxer was identified there. Every burst frame is checked to lie in its
    boxer's exposure (RuntimeError otherwise), so numerator and denominator are the same
    observation by construction. `identified_min` (rows the linking names) is descriptive only.

    An owned label is "in exposure" when its span [first, last] intersects the owner's
    exposure; an unattributed label when it intersects either boxer's. The masks are returned
    under "exposure" as {"A": bool array, "B": bool array}.

    `attrs` is `olympic.attribute(events, table)`: one attribution per event, same order.
    `skips`: frames where the camera dropped a frame; the frames within reach of one are masked (`_scan_segments`),
    so the exposure shrinks by them. None = as before."""
    if not tol_s >= 0:
        raise ValueError("tol_s must be non-negative")
    check_attrs_shape(events, attrs)
    check_attrs_bound(table, events, attrs)
    tol = int(round(tol_s * table.fps))
    bridge = int(round(punchfind.BRIDGE_S * table.fps))
    min_rows = max(2, int(np.ceil(MIN_SEGMENT_S * table.fps - 1e-9)))
    owner = [event_owner(a, linking) for a in attrs]

    raw: "list[list[tuple]]" = [[], []]            # per boxer: (frame, hand index, peak speed)
    boxers, masks = {}, {}
    for k in (0, 1):
        rows = boxer_rows(table, linking, k)
        info = {"identified_frames": int(len(rows)), "identified_min": len(rows) / table.fps / 60,
                "exposure_frames": 0, "exposure_min": 0.0, "signal_min_left": 0.0,
                "signal_min_right": 0.0, "segments": 0, "short_segments": 0, "bodyless_segments": 0,
                "bursts": 0, "unexplained": 0, "bursts_by_hand": {"left": 0, "right": 0},
                "unexplained_by_hand": {"left": 0, "right": 0}, "own_labels": sum(o == k for o in owner),
                "own_labels_in_exposure": 0, "own_labels_outside_exposure": 0,
                "labels_matched_own": 0, "note": ""}
        boxers["AB"[k]] = info
        scan = _scan_segments(table, boxer_segments(table, linking, k, max_gap=bridge), rule,
                              f"boxer{'AB'[k]}", bridge, min_rows, skips)
        mask = scan["mask"]
        masks["AB"[k]] = mask
        raw[k] = scan["raw"]
        for key in ("segments", "short_segments", "bodyless_segments"):
            info[key] = scan[key]
        info["exposure_frames"] = int(mask.sum())
        info["exposure_min"] = int(mask.sum()) / table.fps / 60
        info["signal_min_left"], info["signal_min_right"] = (scan["valid"] / table.fps / 60).tolist()
        if len(rows) < 2:
            info["note"] = "fewer than two identified frames"

    def in_exposure(e, which) -> bool:
        lo, hi = e.first, min(e.last, table.n_frames - 1)
        return lo <= hi and any(masks[n][lo:hi + 1].any() for n in which)

    for i, o in enumerate(owner):
        if o is not None:
            key = "own_labels_in_exposure" if in_exposure(events[i], "AB"[o]) else "own_labels_outside_exposure"
            boxers["AB"[o]][key] += 1

    # Matching. Keys are (boxer, index in raw[boxer]).
    taken: set = set()
    matched: "dict[tuple, int]" = {}
    how: "dict[tuple, str]" = {}
    for k in (0, 1):
        own = [i for i, o in enumerate(owner) if o == k]
        got = _match([((k, j), b[0]) for j, b in enumerate(raw[k])], own, events, tol, taken)
        matched.update(got)
        how.update({key: "own" for key in got})
    left = [((k, j), b[0]) for k in (0, 1) for j, b in enumerate(raw[k]) if (k, j) not in matched]
    got = _match(left, [i for i, o in enumerate(owner) if o is None], events, tol, taken)
    matched.update(got)
    how.update({key: "unattributed" for key in got})

    bursts = []
    for k in (0, 1):
        info = boxers["AB"[k]]
        for j, (f, h, peak) in enumerate(raw[k]):
            li = matched.get((k, j))
            bursts.append(Burst(video, f, float(f) / table.fps, "AB"[k], HAND_NAMES[h], peak,
                                how.get((k, j), "none"),
                                None if li is None else bool(events[li].hand == HAND_NAMES[h])))
            info["bursts"] += 1
            info["bursts_by_hand"][HAND_NAMES[h]] += 1
            if li is None:
                info["unexplained"] += 1
                info["unexplained_by_hand"][HAND_NAMES[h]] += 1
            elif how[(k, j)] == "own":
                info["labels_matched_own"] += 1
    labels = {"total": len(events), "owned_a": sum(o == 0 for o in owner), "owned_b": sum(o == 1 for o in owner),
              "unattributed": sum(o is None for o in owner),
              "unattributed_in_exposure": sum(o is None and in_exposure(events[i], "AB") for i, o in enumerate(owner)),
              "matched_own": sum(v == "own" for v in how.values()),
              "matched_unattributed": sum(v == "unattributed" for v in how.values())}
    return {"bursts": bursts, "boxers": boxers, "labels": labels, "exposure": masks}


def random_sample(unexplained: "dict[str, list[Burst]]", n: int = SAMPLE_N, seed: int = SAMPLE_SEED
                  ) -> "tuple[list[Burst], int]":
    """A seeded SIMPLE random sample, without replacement, of `n` bursts (all of them when
    there are fewer) from ALL the unexplained bursts of ALL videos pooled, so every burst has
    the same inclusion probability len(sample)/N and an unweighted human-positive fraction
    estimates the population fraction. Returns (sample sorted by video, frame, boxer, hand;
    N, the population size). The result does not depend on dict order."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("sample size must be a whole number >= 1")
    pop = sorted((b for v in sorted(unexplained) for b in unexplained[v]),
                 key=lambda b: (b.video, b.frame, b.boxer, b.hand))
    big_n = len(pop)
    take = sorted(random.Random(seed).sample(range(big_n), min(n, big_n)))
    return [pop[i] for i in take], big_n


BIAS_CREDITED = ("Only fragments (track ids) credited with a labelled punch are tested. Missed punches outside them (boxer "
                 "time in fragments with no credited punch, punches too close to call) are never seen, so coverage can only "
                 "lose; but fragment-level detections can DUPLICATE one physical punch when two overlapping ids follow the "
                 "same person. So neither a bout-level upper nor a lower bound follows without de-duplication "
                 "(possible_duplicates counts the unexplained bursts that have a likely twin). A clean result is weak "
                 "evidence of completeness; a dirty one (many sampled bursts classified as real unlabelled punches) shows "
                 "labels are missing inside the credited fragments, though the count may include duplicates. A credited "
                 "fragment can be a non-boxer wrongly credited; an id switch inside a fragment is the tracker's residual "
                 "error.")
DUP_IOU = 0.3
# DUP_IOU: two ids that follow ONE person give near-identical boxes (IoU well above 0.5), while two fighters in a clinch
# overlap only partly. 0.3 is a deliberately generous SCREENING level: it flags a "possible" twin, it does not prove one,
# and it was fixed from the review brief, not tuned on the Olympic numbers. The time window is the matching tolerance.


def box_at(table: PoseTable, rows: np.ndarray, frame: int) -> "tuple[float, float, float]":
    """(centre x, centre y, height) in pixels of the box of ONE track id at `frame`: its own row's
    box, or, on a frame the bridge interpolated across (no row), the linear interpolation between the
    rows on either side. `rows` are that id's row indices in frame order; `frame` must lie inside
    their span (a burst always does)."""
    b = box_xyxy_at(table, rows, frame)
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]


def box_xyxy_at(table: PoseTable, rows: np.ndarray, frame: int) -> "tuple[float, float, float, float]":
    """(x1, y1, x2, y2) of ONE track id at `frame`, interpolated as `box_at` does."""
    fr = table.frame[rows].astype(np.float64)
    if not (fr[0] <= frame <= fr[-1]):
        raise ValueError(f"frame {frame} is outside this fragment's span {int(fr[0])}-{int(fr[-1])}")
    return tuple(float(np.interp(frame, fr, table.box[rows, c].astype(np.float64))) for c in range(4))


def iou(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes; 0 when they do not touch or either is empty."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def find_bursts_credited(table: PoseTable, events, attrs, rule: dict, video: str = "",
                         tol_s: float = TOL_S, min_punches: int = 1, skips: "np.ndarray | None" = None) -> dict:
    """The completeness test on CREDITED FRAGMENTS, with no linking (module docstring: ADR-0012
    Option 2 and its bias). Population: every track id credited (`attrs` from `olympic.attribute`,
    one per event, same order) with at least `min_punches` labelled punches.

    Per fragment: its rows are split into segments exactly as `boxer_segments` does (the id never
    changes inside a fragment, so only a hole longer than the bridge splits it), and the same
    `_scan_segments` runs the rule, builds the EXPOSURE (processed segments, bridged frames included,
    at least one hand valid) and asserts every burst lies in it.

    Ownership: a label is OWNED by the fragment it is credited to. Phase 1 matches each fragment's
    bursts to its OWN labels; phase 2 matches every burst still unmatched, of all fragments together,
    to the UNATTRIBUTED labels (credited to no one: ambiguous or unreached), each label once. Both
    use `_match` (optimal within the phase, ties to the burst that sorts first by (frame, key), here
    the lower track id). A label credited to a track id that is NOT in the population (fewer than
    `min_punches` credited punches) belongs to a person this test does not follow; it explains
    nothing (counted as `credited_to_excluded`), exactly as another fragment's label does not.
    Two fragments on screen together are two people: their bursts are kept apart. The same person
    under two overlapping ids would be counted twice (the tracker's error); `possible_duplicates` is
    the number of unexplained bursts that have such a likely twin (unexplained, another fragment,
    within the matching tolerance, boxes overlapping at IoU >= DUP_IOU), `possible_duplicate_pairs`
    the number of such pairs. They are reported, never subtracted. `attrs` must be exactly
    `olympic.attribute(events, table)` for THIS table (verified: ValueError otherwise).

    `skips`: frames where the camera dropped a frame (`skips.find_skips`); the frames within reach of one are masked
    (`_scan_segments`): no signal there, not in the exposure. None = as before.

    Returns {"bursts": [Burst] (track_id and box set), "fragments": {track id: info}, "labels": {...},
    "exposure": {track id: bool mask}, "min_punches": int, "bias": str}."""
    if not tol_s >= 0:
        raise ValueError("tol_s must be non-negative")
    if isinstance(min_punches, bool) or not isinstance(min_punches, (int, np.integer)) or min_punches < 1:
        raise ValueError("min_punches must be a whole number >= 1")
    check_attrs_shape(events, attrs)
    present = set(np.unique(table.track_id).tolist())
    credited: "dict[int, int]" = {}
    for a in attrs:
        if a.track_id is not None:
            if a.track_id not in present:
                raise ValueError(f"a punch is credited to track id {a.track_id}, which is not in this pose table "
                                 "(attributions from another video?)")
            credited[int(a.track_id)] = credited.get(int(a.track_id), 0) + 1
    check_attrs_bound(table, events, attrs)
    tids = sorted(t for t, n in credited.items() if n >= int(min_punches))
    index = {t: i for i, t in enumerate(tids)}
    tol = int(round(tol_s * table.fps))
    bridge = int(round(punchfind.BRIDGE_S * table.fps))
    min_rows = max(2, int(np.ceil(MIN_SEGMENT_S * table.fps - 1e-9)))
    owner = [index.get(int(a.track_id)) if a.track_id is not None else None for a in attrs]
    excluded = [a.track_id is not None and int(a.track_id) not in index for a in attrs]

    frags, masks, raw, rowsof = {}, {}, [], []
    for t in tids:
        rows = np.flatnonzero(table.track_id == t)
        rows = rows[np.argsort(table.frame[rows], kind="stable")]
        rowsof.append(rows)
        scan = _scan_segments(table, _split_rows(table, rows, bridge), rule, f"fragment{t}", bridge, min_rows, skips)
        mask = scan["mask"]
        masks[t] = mask
        raw.append(scan["raw"])
        frags[t] = {"track_id": t, "first_frame": int(table.frame[rows[0]]), "last_frame": int(table.frame[rows[-1]]),
                    "rows": int(len(rows)), "credited_punches": credited[t],
                    "exposure_frames": int(mask.sum()), "exposure_min": int(mask.sum()) / table.fps / 60,
                    "signal_min_left": float(scan["valid"][0] / table.fps / 60),
                    "signal_min_right": float(scan["valid"][1] / table.fps / 60),
                    "segments": scan["segments"], "short_segments": scan["short_segments"],
                    "bodyless_segments": scan["bodyless_segments"], "bursts": 0, "unexplained": 0,
                    "bursts_by_hand": {"left": 0, "right": 0}, "unexplained_by_hand": {"left": 0, "right": 0},
                    "own_labels_in_exposure": 0, "own_labels_outside_exposure": 0, "labels_matched_own": 0}

    def in_exposure(e, which) -> bool:
        lo, hi = e.first, min(e.last, table.n_frames - 1)
        return lo <= hi and any(masks[t][lo:hi + 1].any() for t in which)

    for i, o in enumerate(owner):
        if o is not None:
            key = "own_labels_in_exposure" if in_exposure(events[i], (tids[o],)) else "own_labels_outside_exposure"
            frags[tids[o]][key] += 1

    # Matching. Keys are (fragment index, index in raw[fragment]).
    taken: set = set()
    matched: "dict[tuple, int]" = {}
    how: "dict[tuple, str]" = {}
    for fi in range(len(tids)):
        own = [i for i, o in enumerate(owner) if o == fi]
        got = _match([((fi, j), b[0]) for j, b in enumerate(raw[fi])], own, events, tol, taken)
        matched.update(got)
        how.update({key: "own" for key in got})
    left = [((fi, j), b[0]) for fi in range(len(tids)) for j, b in enumerate(raw[fi]) if (fi, j) not in matched]
    unattr = [i for i, a in enumerate(attrs) if a.track_id is None]
    got = _match(left, unattr, events, tol, taken)
    matched.update(got)
    how.update({key: "unattributed" for key in got})

    # Possible duplicates: an UNEXPLAINED burst with an unexplained burst of ANOTHER credited fragment within the matching
    # tolerance whose box overlaps it (IoU >= DUP_IOU) at the earlier of the two frames (both fragments must be in their
    # exposure there). Two ids that follow one person would each report the same punch. Symmetric by construction.
    unex = sorted((raw[fi][j][0], fi, j) for fi in range(len(tids)) for j in range(len(raw[fi])) if (fi, j) not in matched)
    twin: set = set()
    pairs = 0
    for a_i, (fa_, fia, ja) in enumerate(unex):
        for b_i in range(a_i + 1, len(unex)):
            fb_, fib, jb = unex[b_i]
            if fb_ - fa_ > tol:
                break
            if fib == fia or not masks[tids[fib]][fa_]:
                continue
            if iou(box_xyxy_at(table, rowsof[fia], fa_), box_xyxy_at(table, rowsof[fib], fa_)) >= DUP_IOU:
                twin.update({(fia, ja), (fib, jb)})
                pairs += 1
    bursts = []
    for fi, t in enumerate(tids):
        info = frags[t]
        info["possible_duplicates"] = sum((fi, j) in twin for j in range(len(raw[fi])))
        for j, (f, h, peak) in enumerate(raw[fi]):
            li = matched.get((fi, j))
            bursts.append(Burst(video, f, float(f) / table.fps, str(t), HAND_NAMES[h], peak, how.get((fi, j), "none"),
                                None if li is None else bool(events[li].hand == HAND_NAMES[h]),
                                t, box_at(table, rowsof[fi], f), (fi, j) in twin))
            info["bursts"] += 1
            info["bursts_by_hand"][HAND_NAMES[h]] += 1
            if li is None:
                info["unexplained"] += 1
                info["unexplained_by_hand"][HAND_NAMES[h]] += 1
            elif how[(fi, j)] == "own":
                info["labels_matched_own"] += 1
    labels = {"total": len(events), "owned": sum(o is not None for o in owner),
              "owned_in_exposure": sum(f["own_labels_in_exposure"] for f in frags.values()),
              "owned_outside_exposure": sum(f["own_labels_outside_exposure"] for f in frags.values()),
              "credited_to_excluded": sum(excluded), "unattributed": len(unattr),
              "unattributed_in_exposure": sum(in_exposure(events[i], tids) for i in unattr),
              "matched_own": sum(v == "own" for v in how.values()),
              "matched_unattributed": sum(v == "unattributed" for v in how.values())}
    return {"bursts": bursts, "fragments": frags, "labels": labels, "exposure": masks,
            "possible_duplicates": len(twin), "possible_duplicate_pairs": pairs,
            "min_punches": int(min_punches), "bias": BIAS_CREDITED}


NEAR_SKIP = 2     # a burst is "near a skip" when a skip lies within this many frames of it


def near_skip_stats(bursts: "list[Burst]", exposure: "dict[object, np.ndarray]", skips, n_frames: int,
                    within: int = NEAR_SKIP) -> dict:
    """How the bursts sit relative to the camera's dropped frames, with the BASE RATE they must be read against.

    A frame is NEAR a skip when a skip lies within `within` frames of it (either side). Returns counts
    (add them across videos, divide once at the end): `skips`, `exposure_frames` and `exposure_near_skip`
    (summed over the masks in `exposure`, one per person, as the burst rates' denominator is), and for the
    unexplained and the explained bursts their number and how many are near a skip. If skips make no
    difference to where bursts are, `unexplained_near_skip / unexplained` is about
    `exposure_near_skip / exposure_frames`; a skip-made burst pushes the first above the second."""
    if isinstance(within, bool) or not isinstance(within, (int, np.integer)) or within < 0:
        raise ValueError("within must be a whole number >= 0")
    sk = check_skips(skips, n_frames)
    near = np.zeros(n_frames, dtype=bool)
    for f in sk:
        near[max(0, int(f) - within):int(f) + within + 1] = True
    un = [b for b in bursts if not b.explained]
    ex = [b for b in bursts if b.explained]
    for b in bursts:
        if not 0 <= b.frame < n_frames:
            raise ValueError(f"a burst at frame {b.frame} is outside the {n_frames}-frame video")
    return {"skips": int(len(sk)), "within_frames": int(within),
            "exposure_frames": int(sum(int(m.sum()) for m in exposure.values())),
            "exposure_near_skip": int(sum(int((m & near).sum()) for m in exposure.values())),
            "unexplained": len(un), "unexplained_near_skip": sum(bool(near[b.frame]) for b in un),
            "explained": len(ex), "explained_near_skip": sum(bool(near[b.frame]) for b in ex)}


def mss(seconds: float) -> str:
    """Seconds as m:ss.s, rounded to a tenth FIRST so 59.96 s is 1:00.0 and never 0:60.0."""
    if not (isinstance(seconds, (int, float, np.floating)) and np.isfinite(seconds) and seconds >= 0):
        raise ValueError("a time must be a finite number of seconds >= 0")
    tenths = int(round(float(seconds) * 10))
    return f"{tenths // 600}:{(tenths % 600) / 10:04.1f}"
