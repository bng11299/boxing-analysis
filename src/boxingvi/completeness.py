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
    to someone who is not an identified boxer);
  * each boxer's bursts are matched first to that boxer's OWN labels, each label to at
    most one burst, nearest first, within TOL_S of the label's frames; a label thrown
    by the other boxer never explains this boxer's burst;
  * the bursts still unmatched, of both boxers together, are then matched one-to-one to
    the UNATTRIBUTED labels (each used once across both boxers);
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

Caveats carried into the report: the rule's threshold is V6's (AlphaPose, 24 fps, hooks
and uppercuts of one athlete), and a burst is only seen where the linking identified the
boxer, in a segment long enough for the rule, with a confident wrist.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from . import punchfind
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


def boxer_segments(table: PoseTable, linking: Linking, k: int) -> "list[np.ndarray]":
    """Boxer k's rows split into SEGMENTS (arrays of row indices, frame order): a new segment
    starts wherever the track id changes or the frame number jumps by more than one."""
    rows = boxer_rows(table, linking, k)
    if len(rows) == 0:
        return []
    tid, fr = table.track_id[rows], table.frame[rows]
    cut = np.flatnonzero((np.diff(tid) != 0) | (np.diff(fr) != 1)) + 1
    return [rows[a:b] for a, b in zip(np.r_[0, cut], np.r_[cut, len(rows)])]


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
    """Greedy one-to-one matching, nearest first. `bursts` is [(key, frame)], `labels` indexes
    into `events`; labels already in `taken` are unavailable and the used ones are added to it.
    Returns {burst key: label index}. Ties are broken by frame, then key, then label index."""
    pairs = []
    for key, f in bursts:
        for li in labels:
            if li not in taken:
                g = span_gap(f, events[li])
                if g <= tol:
                    pairs.append((g, f, key, li))
    pairs.sort()
    got: "dict[int, int]" = {}
    for _g, _f, key, li in pairs:
        if key not in got and li not in taken:
            got[key] = li
            taken.add(li)
    return got


def find_bursts(table: PoseTable, linking: Linking, events, attrs, rule: dict, video: str = "",
                tol_s: float = TOL_S) -> dict:
    """Bursts per boxer, matched one-to-one to labels (see the module docstring), the minutes
    they were looked for in, and the labels each boxer owns.

    `attrs` is `olympic.attribute(events, table)`: one attribution per event, same order."""
    if not tol_s >= 0:
        raise ValueError("tol_s must be non-negative")
    if len(attrs) != len(events) or any(a.event is not e for a, e in zip(attrs, events)):
        raise ValueError("attrs must be olympic.attribute(events, table): one attribution per event, in order")
    tol = int(round(tol_s * table.fps))
    bridge = int(round(punchfind.BRIDGE_S * table.fps))
    min_rows = max(2, int(np.ceil(MIN_SEGMENT_S * table.fps - 1e-9)))
    owner = [event_owner(a, linking) for a in attrs]

    raw: "list[list[tuple]]" = [[], []]            # per boxer: (frame, hand index, peak speed)
    boxers = {}
    for k in (0, 1):
        rows = boxer_rows(table, linking, k)
        info = {"identified_min": len(rows) / table.fps / 60, "signal_min": 0.0, "signal_min_left": 0.0,
                "signal_min_right": 0.0, "segments": 0, "short_segments": 0, "bodyless_segments": 0,
                "bursts": 0, "unexplained": 0, "bursts_by_hand": {"left": 0, "right": 0},
                "unexplained_by_hand": {"left": 0, "right": 0}, "own_labels": sum(o == k for o in owner),
                "labels_matched_own": 0, "note": ""}
        boxers["AB"[k]] = info
        valid = np.zeros(2)                          # frames with a valid signal: left, right
        either = 0
        for seg in boxer_segments(table, linking, k):
            info["segments"] += 1
            if len(seg) < min_rows:
                info["short_segments"] += 1
                continue
            track = punchfind.bridge(_track_of(table, seg, f"{table.source}#boxer{'AB'[k]}"), bridge)
            try:
                sig = punchfind.wrist_signal(track, rule["variant"])
            except ValueError:                       # no confident body anywhere in this segment
                info["bodyless_segments"] += 1
                continue
            have = punchfind.signal_frames(track)
            valid += have.sum(axis=0)
            either += int(have.any(axis=1).sum())
            frames, hands = punchfind.detect(track, sig, rule["threshold"], rule["min_gap_s"])
            row = {int(f): i for i, f in enumerate(track.frame_ids)}
            for f, h in zip(frames, hands):
                raw[k].append((int(f), int(h), float(sig[row[int(f)], int(h)])))
        info["signal_min"] = either / table.fps / 60
        info["signal_min_left"], info["signal_min_right"] = (valid / table.fps / 60).tolist()
        if len(rows) < 2:
            info["note"] = "fewer than two identified frames"

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
              "matched_own": sum(v == "own" for v in how.values()),
              "matched_unattributed": sum(v == "unattributed" for v in how.values())}
    return {"bursts": bursts, "boxers": boxers, "labels": labels}


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
