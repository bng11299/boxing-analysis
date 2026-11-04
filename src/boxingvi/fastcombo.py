"""Scoring, tuning and the paired comparison for the fast-combination arms.

Everything here follows docs/gates/fast-combinations.md, written and committed before
any tuning or held-out number: the grids, the type groups, the admissibility rule, the
tie-breaks, the arm choice and the block bootstrap. The scripts
(tune_fast_combinations.py, eval_fast_combinations.py) only load videos and call this.

A detection set is scored the way the end-to-end evaluation scores it (RESULTS row 5a):
nearest matching within 0.1 s, detections on an unusable labelled row ignored, the whole
video. Counts are kept per 30 s block so that a block bootstrap can resample them, and
so that two finders on the same video are compared block for block (paired).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import punchfind
from .data import CLASSES, N_CLASSES, Span, Track

THRESHOLDS = tuple(round(2.0 + 0.1 * i, 1) for i in range(41))      # 2.0 .. 6.0 body lengths/s
WINDOWS_S = (0.05, 0.10, 0.15, 0.20, 0.30)                          # arm 1: extension window
CROSS_GAPS_S = (0.10, 0.12, 0.15, 0.20, 0.25, 0.30)                 # arm 2: gap between hands
VARIANT, MIN_GAP_S = "relative", 0.4                                 # fixed, the V6 rule's
BASELINE_THRESHOLD = 3.6238817996457016   # results/punch_finder_v6.json, the registered baseline (3.62)
TOL_S, BLOCK_S = 0.1, 30.0
N_BOOT, SEED, ALPHA = 2000, 0, 0.05
MARGIN = 0.05            # no type group's recall may fall by more than this
EPS = 1e-12              # a fall of exactly MARGIN is allowed, whatever the float rounding
GROUPS = {"jab": ("jab",), "cross": ("cross",), "hooks": ("lead_hook", "rear_hook"),
          "uppercuts": ("lead_uppercut", "rear_uppercut")}
GUARDED = ("cross", "hooks", "uppercuts")    # the groups the margin protects; jab must rise
F1_EXCLUDED = ("V6",)    # V6 labels only hooks and uppercuts: recall checks only, not F1
ARMS = ("per_hand_extension", "cross_hand_gap")
# The release's V2 and V3 video links are swapped (vault ISSUES-LOG O61): V2's labels sit in
# the video linked as V3 and the other way round. Each pairing still passes the alignment gate.
POSE_FILE = {"V2": "V3", "V3": "V2"}
PARAM = {"per_hand_extension": "extension_window_s", "cross_hand_gap": "cross_gap_s"}

# columns of a count row
DET, LAB, HIT_D, HIT_S = 0, 1, 2, 3
LAB_T, HIT_T = 4, 4 + N_CLASSES
N_COLS = 4 + 2 * N_CLASSES


@dataclass
class Video:
    """One scored video: the track and signal the finder runs on, and its ground truth."""
    name: str
    fps: float
    n_frames: int
    track: Track
    signal: np.ndarray
    spans: "list[Span]"
    unusable: "list[tuple[int, int]]"

    @property
    def tol(self) -> int:
        return int(round(TOL_S * self.fps))

    @property
    def block(self) -> int:
        return int(round(BLOCK_S * self.fps))

    @property
    def n_blocks(self) -> int:
        return (self.n_frames - 1) // self.block + 1


def rule(threshold: float, finder: "dict | None") -> dict:
    """A rule as find_punches.load_rule returns it."""
    r = {"variant": VARIANT, "threshold": float(threshold), "min_gap_s": MIN_GAP_S}
    if finder is not None:
        r["finder"] = punchfind.parse_finder(finder, MIN_GAP_S)
    return r


def _distance_to(frames: np.ndarray, intervals: "list[tuple[int, int]]") -> np.ndarray:
    if not intervals:
        return np.full(len(frames), np.iinfo(np.int64).max)
    iv = np.asarray(intervals, dtype=np.int64)
    d = np.maximum(np.maximum(iv[None, :, 0] - frames[:, None], frames[:, None] - iv[None, :, 1]), 0)
    return d.min(axis=1)


def counts(v: Video, det_frames) -> np.ndarray:
    """(n_blocks, N_COLS) int counts of one detection set on one video: detections scored
    (by their block), labels (by the span's start block), hits by the detection's block, hits by
    the span's block, then labels and hits per type (by the span's block). A detection on an
    unusable labelled row that matched nothing is neither a hit nor a false alarm."""
    det = np.asarray(det_frames, dtype=np.int64)
    if det.ndim != 1 or (len(det) and (det.min() < 0 or det.max() >= v.n_frames)):
        raise ValueError(f"{v.name}: detections must be frames of the video")
    pairs = punchfind.match_nearest(det, v.spans, v.tol)
    hit = np.zeros(len(det), bool)
    hit[[j for _, j in pairs]] = True
    scored = det[~(~hit & (_distance_to(det, v.unusable) <= v.tol))]
    c = np.zeros((v.n_blocks, N_COLS), np.int64)
    np.add.at(c[:, DET], scored // v.block, 1)
    for s in v.spans:
        c[s.start // v.block, LAB] += 1
        c[s.start // v.block, LAB_T + s.label] += 1
    for si, dj in pairs:
        s = v.spans[si]
        c[det[dj] // v.block, HIT_D] += 1
        c[s.start // v.block, HIT_S] += 1
        c[s.start // v.block, HIT_T + s.label] += 1
    return c


def _ratio(a, b) -> "float | None":
    return float(a / b) if b else None


def precision(t: np.ndarray) -> "float | None":
    return _ratio(t[HIT_D], t[DET])


def recall(t: np.ndarray) -> "float | None":
    return _ratio(t[HIT_S], t[LAB])


def f1(t: np.ndarray) -> "float | None":
    p, r = precision(t), recall(t)
    if p is None or r is None:
        return None
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def group_recall(t: np.ndarray, group: str) -> "float | None":
    idx = [CLASSES.index(c) for c in GROUPS[group]]
    return _ratio(sum(t[HIT_T + i] for i in idx), sum(t[LAB_T + i] for i in idx))


def summary(t: np.ndarray) -> dict:
    """Point metrics of a summed count row."""
    return {"detections_scored": int(t[DET]), "labelled": int(t[LAB]), "hits": int(t[HIT_S]),
            "precision_lower_bound": precision(t), "recall": recall(t), "f1": f1(t),
            "recall_by_group": {g: group_recall(t, g) for g in GROUPS},
            "recall_by_type": {c: {"labelled": int(t[LAB_T + i]), "found": int(t[HIT_T + i]),
                                   "recall": _ratio(t[HIT_T + i], t[LAB_T + i])} for i, c in enumerate(CLASSES)}}


def total(per_video: "dict[str, np.ndarray]", videos=None) -> np.ndarray:
    names = per_video if videos is None else videos
    return sum((per_video[v].sum(axis=0) for v in names), np.zeros(N_COLS, np.int64))


# ------------------------------------------------------------------ tuning ---

def grid():
    """Every (arm, parameter, threshold) the gate fixes, in a fixed order, plus the control
    (the pooled rule, threshold re-tuned; finder None)."""
    for thr in THRESHOLDS:
        yield "control", None, thr
    for w in WINDOWS_S:
        for thr in THRESHOLDS:
            yield "per_hand_extension", w, thr
    for g in CROSS_GAPS_S:
        for thr in THRESHOLDS:
            yield "cross_hand_gap", g, thr


def finder_of(arm: str, param) -> "dict | None":
    return None if arm == "control" else {"mode": arm, PARAM[arm]: float(param)}


def detections(v: Video, r: dict, masks: "dict[float, np.ndarray] | None" = None) -> np.ndarray:
    """Detection frames of rule r on v. `masks` caches extension masks by window."""
    f = r.get("finder")
    if f is not None and f["mode"] == "per_hand_extension" and masks is not None:
        w = f["extension_window_s"]
        if w not in masks:
            masks[w] = punchfind.extension_mask(v.track, w)
        frames, _ = punchfind.detect(v.track, v.signal, r["threshold"], r["min_gap_s"], per_hand=True,
                                     eligible=masks[w])
        return frames
    return punchfind.detect_rule(v.track, v.signal, r)[0]


def admissible(t_all: np.ndarray, base_all: np.ndarray) -> "tuple[bool, list[str]]":
    """The gate's training admissibility: jab recall above the baseline's, and no guarded
    group more than MARGIN below it. Returns (ok, the reasons it is not)."""
    why = []
    j, jb = group_recall(t_all, "jab"), group_recall(base_all, "jab")
    if j is None or jb is None or not j > jb:
        why.append(f"jab recall {j} not above the baseline's {jb}")
    for g in GUARDED:
        a, b = group_recall(t_all, g), group_recall(base_all, g)
        if a is None or b is None:
            why.append(f"{g}: no labelled punches")
        elif a < b - MARGIN - EPS:
            why.append(f"{g} recall {a:.3f} more than {MARGIN} below the baseline's {b:.3f}")
    return not why, why


def choose(points: "list[dict]") -> "dict | None":
    """The gate's choice within one arm. Each point: {threshold, param, f1, admissible}.
    Highest F1 among admissible points; ties to the higher threshold, then the smaller
    parameter. With no admissible point, the same order over all points (flagged)."""
    if not points:
        return None
    pool = [p for p in points if p["admissible"]] or points

    def key(p):
        return (-1.0 if p["f1"] is None else p["f1"], p["threshold"], -(p["param"] or 0.0))
    return max(pool, key=key)


def choose_arm(best: "dict[str, dict]") -> str:
    """ONE arm: the higher training F1 of the two chosen points; an exact tie to arm 2."""
    a1, a2 = best["per_hand_extension"], best["cross_hand_gap"]
    f1a = -1.0 if a1["f1"] is None else a1["f1"]
    f2a = -1.0 if a2["f1"] is None else a2["f1"]
    return "per_hand_extension" if f1a > f2a else "cross_hand_gap"


# --------------------------------------------------------- paired bootstrap ---

def paired_ci(arm: "dict[str, np.ndarray]", base: "dict[str, np.ndarray]", stat,
              n_boot: int = N_BOOT, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Percentile interval of stat(arm totals) - stat(base totals), resampling whole blocks
    WITH replacement inside each video, the SAME blocks for both finders (paired). Draws
    where either statistic is undefined are dropped and counted."""
    if set(arm) != set(base) or any(arm[v].shape != base[v].shape for v in arm):
        raise ValueError("the two finders must be counted on the same videos and blocks")

    def diff(ta, tb):
        a, b = stat(ta), stat(tb)
        return None if a is None or b is None else a - b

    point = diff(total(arm), total(base))
    if point is None:
        raise ValueError("the statistic is undefined on the data itself")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ta, tb = np.zeros(N_COLS, np.int64), np.zeros(N_COLS, np.int64)
        for v in sorted(arm):
            idx = rng.integers(0, len(arm[v]), size=len(arm[v]))
            ta += arm[v][idx].sum(axis=0)
            tb += base[v][idx].sum(axis=0)
        d = diff(ta, tb)
        if d is not None:
            vals.append(d)
    if len(vals) < n_boot // 2:
        raise ValueError(f"only {len(vals)} of {n_boot} resamples were defined")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": n_boot, "seed": seed,
            "n_undefined_dropped": n_boot - len(vals), "blocks": {v: int(len(arm[v])) for v in sorted(arm)}}


def verdict(arm_t: np.ndarray, base_t: np.ndarray, f1_ci: dict) -> dict:
    """The gate's three held-out pass conditions, each reported, and the AND of them."""
    jab_a, jab_b = group_recall(arm_t, "jab"), group_recall(base_t, "jab")
    c1 = f1_ci["point"] > 0 and f1_ci["ci_low"] > 0
    c2 = jab_a is not None and jab_b is not None and jab_a > jab_b
    guards = {}
    for g in GUARDED:
        a, b = group_recall(arm_t, g), group_recall(base_t, g)
        guards[g] = {"arm": a, "baseline": b, "ok": a is not None and b is not None and a >= b - MARGIN - EPS}
    c3 = all(x["ok"] for x in guards.values())
    return {"f1_up_interval_excludes_zero": c1, "jab_recall_up": c2, "no_group_down_more_than_margin": c3,
            "margin": MARGIN, "guarded_groups": guards, "pass": bool(c1 and c2 and c3)}
