"""Did the referee label every punch? Per boxer, by the wrist-speed rule.

The Olympic Boxing paper has a licensed referee judging 312,774 frames. Whether that
means EVERY punch was labelled is a claim, not a fact. A test that can fail: take each
linked boxer's own per-frame pose, run the per-hand wrist-speed rule (punchfind, the
rule tuned on V6 and saved in results/punch_finder_v6.json), and keep every burst more
than 0.1 s from ANY labelled punch of that video. Those are "unexplained bursts".

Unexplained is not "missed": a feint, a guard reset, a slip, a wave to the corner or
a stumble also moves a wrist fast. The count is an upper bound on unlabelled punches
and only becomes a number about the labels once a human has classified a random
sample of the bursts. This module draws that sample (fixed seed, stratified over
videos) and judges none of it.

Caveats carried into the report: the rule's threshold is V6's (AlphaPose, 24 fps,
hooks and uppercuts of one athlete), and a burst is only seen where the linking
identified the boxer and the wrist was confident.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from . import punchfind
from .data import Track
from .linking import Linking
from .pose_yolo import PoseTable

TOL_S = 0.1          # "near a labelled punch": the matching tolerance used on V6 and by find_punches.py
SAMPLE_N = 40
SAMPLE_SEED = 20261007
HAND_NAMES = ("left", "right")


@dataclass(frozen=True)
class Burst:
    video: str
    frame: int
    time_s: float
    boxer: str          # "A" / "B"
    hand: str           # the boxer's own left / right (COCO)
    peak_speed: float   # body lengths per second
    explained: bool     # within TOL_S of a labelled punch


def boxer_track(table: PoseTable, linking: Linking, k: int) -> "Track | None":
    """Boxer k's rows in frame order, as a Track with confidence; None under two frames.
    A frame belongs to the boxer only if the linking names a track id there AND that id
    has a row in that frame."""
    if linking.n_frames != table.n_frames or linking.source != table.source:
        raise ValueError("the linking was made from a different pose table than this one")
    want = linking.membership[table.frame, k]
    rows = np.flatnonzero(want == table.track_id)
    if len(rows) < 2:
        return None
    rows = rows[np.argsort(table.frame[rows], kind="stable")]
    return Track(frame_ids=table.frame[rows].astype(np.int64), xy=table.xy[rows].astype(np.float64),
                 fps=table.fps, source=f"{table.source}#boxer{'AB'[k]}", conf=table.conf[rows].astype(np.float64))


def _near_label(frames: np.ndarray, events, tol: int) -> np.ndarray:
    """Is each frame within `tol` frames of ANY labelled punch's [first, last]?"""
    if not events:
        return np.zeros(len(frames), dtype=bool)
    first = np.array([e.first for e in events])
    last = np.array([e.last for e in events])
    f = np.asarray(frames)[:, None]
    gap = np.maximum(first[None, :] - f, f - last[None, :])      # <= 0 inside the punch
    return (gap <= tol).any(axis=1)


def find_bursts(table: PoseTable, linking: Linking, events, rule: dict, video: str = "",
                tol_s: float = TOL_S) -> dict:
    """Bursts per boxer, explained or not, and the minutes they were looked for in."""
    if not tol_s >= 0:
        raise ValueError("tol_s must be non-negative")
    tol = int(round(tol_s * table.fps))
    bridge = int(round(punchfind.BRIDGE_S * table.fps))
    out = {"bursts": [], "boxers": {}}
    for k in (0, 1):
        name = "AB"[k]
        track = boxer_track(table, linking, k)
        info = {"identified_min": 0.0, "signal_min": 0.0, "bursts": 0, "unexplained": 0, "note": ""}
        out["boxers"][name] = info
        if track is None:
            info["note"] = "fewer than two identified frames"
            continue
        info["identified_min"] = len(track.frame_ids) / table.fps / 60
        try:
            track = punchfind.bridge(track, bridge)
            sig = punchfind.wrist_signal(track, rule["variant"])
        except ValueError as e:     # no confident body anywhere in this boxer's track
            info["note"] = str(e)
            continue
        cov = punchfind.signal_coverage(track, span=(0, table.n_frames - 1))
        info["signal_min"] = (cov["left"] + cov["right"]) / 2 * table.n_frames / table.fps / 60
        frames, hands = punchfind.detect(track, sig, rule["threshold"], rule["min_gap_s"])
        row = {int(f): i for i, f in enumerate(track.frame_ids)}
        near = _near_label(frames, events, tol)
        for f, h, ok in zip(frames, hands, near):
            out["bursts"].append(Burst(video, int(f), float(f) / table.fps, name, HAND_NAMES[int(h)],
                                       float(sig[row[int(f)], int(h)]), bool(ok)))
        info["bursts"] = int(len(frames))
        info["unexplained"] = int((~near).sum())
    # Context for reading the unexplained count: how many labelled punches the rule also sees,
    # in the frames where a boxer is identified (the rule's recall here, a lower bound).
    bf = np.sort(np.array([b.frame for b in out["bursts"]], dtype=np.int64))
    out["events_with_burst"] = int(sum(
        np.searchsorted(bf, e.last + tol, side="right") > np.searchsorted(bf, e.first - tol, side="left")
        for e in events))
    return out


def stratified_sample(unexplained: "dict[str, list[Burst]]", n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> "list[Burst]":
    """`n` bursts (or all of them, if fewer), spread over the videos: one from every video
    that has any (when there is room), the rest in proportion to how many each has
    (largest remainder), drawn without replacement with a seeded generator. The result
    does not depend on dict order, so the same data and seed give the same sample."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("sample size must be a whole number >= 1")
    pools = {v: sorted(b, key=lambda x: (x.frame, x.boxer, x.hand)) for v, b in sorted(unexplained.items()) if b}
    total = sum(len(p) for p in pools.values())
    if total <= n:
        return [b for v in sorted(pools) for b in pools[v]]
    quota = {v: (1 if len(pools) <= n else 0) for v in pools}
    left = n - sum(quota.values())
    room = {v: len(p) - quota[v] for v, p in pools.items()}
    weight = sum(room.values())
    exact = {v: left * room[v] / weight for v in pools}
    for v in pools:
        quota[v] += min(int(exact[v]), room[v])
    left = n - sum(quota.values())
    while left > 0:       # largest remainder first; a video with nothing left to give is skipped
        for v in sorted(pools, key=lambda v: (-(exact[v] - int(exact[v])), v)):
            if left > 0 and quota[v] < len(pools[v]):
                quota[v] += 1
                left -= 1
    rng = random.Random(seed)
    out = []
    for v in sorted(pools):
        out += rng.sample(pools[v], quota[v])
    return sorted(out, key=lambda b: (b.video, b.frame, b.boxer, b.hand))
