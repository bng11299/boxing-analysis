"""Scoring glove identity against Brendan's left/right labels, exactly as docs/gates/glove-identity.md says.

Labels are a CSV he fills in on the BLURRED copy: one row every LABEL_EVERY_S seconds, `L` (boxer A is on
the left), `R` (on the right) or `?` (not clear). The template is written by `write_template` from the pose
table and carries the video's fingerprint, so a label file cannot be scored against another video.

Every method -- the glove rule and both baselines -- gives each of the two boxer rows of a moment an internal
name (glove cluster, linker chain, track id) or None. Internal names mean nothing until the method's first
decided moment (the ANCHOR) maps them to A and B; the anchor is then left out of that method's scores.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .headgear import wilson
from .linking import Region, link

LABEL_EVERY_S = 5
OVERLAP_FRAC = 0.25
ROUND_S = 180.0
LABEL_FORMAT = "glove-identity-labels-1"
MAX_LABEL_BYTES = 1024 * 1024
MAX_MINUTES = 600.0
MAX_A_CHARS = 80
MAX_DURATION_S = 6 * 3600.0
_FP = re.compile(r"[0-9a-f]{64}")
_HEADER_KEYS = ("format", "video_fp", "every_s", "A", "minutes")

# gate bars (docs/gates/glove-identity.md, PASS)
BAR_ACCURACY = 0.90
BAR_WILSON_LO = 0.80
BAR_MIN_DECIDED = 30
BAR_MAX_UNSURE = 0.30
BAR_MAX_SWAPS_PER_ROUND = 1.0
BAR_FAILCASE_MAX_WRONG = 0.10


def score_constants() -> dict:
    g = globals()
    return {k: g[k] for k in ("LABEL_EVERY_S", "OVERLAP_FRAC", "ROUND_S", "BAR_ACCURACY", "BAR_WILSON_LO",
                              "BAR_MIN_DECIDED", "BAR_MAX_UNSURE", "BAR_MAX_SWAPS_PER_ROUND",
                              "BAR_FAILCASE_MAX_WRONG")}


# ---- labels -------------------------------------------------------------------------------
def label_times(table) -> "list[int]":
    """Seconds 0, 5, 10, ... whose frame round(t x fps) lies inside the video. A video longer than
    MAX_DURATION_S (an absurd fps in a crafted table) is refused before the grid is built."""
    if not (math.isfinite(table.fps) and table.fps > 0) or table.n_frames / table.fps > MAX_DURATION_S:
        raise ValueError(f"the pose table claims more than {MAX_DURATION_S:.0f} s of video: not a sparring video")
    out, t = [], 0
    while int(round(t * table.fps)) < table.n_frames:
        out.append(t)
        t += LABEL_EVERY_S
    return out


def write_template(table, path) -> None:
    if not _FP.fullmatch(table.video_fp or ""):
        raise ValueError("the pose table carries no video fingerprint: labels could not be bound to the video")
    lines = [f"# format: {LABEL_FORMAT}", f"# video_fp: {table.video_fp}", f"# every_s: {LABEL_EVERY_S}",
             "# A: ", "# minutes: ",
             "# Fill A with the words that say who A is (e.g. 'the boxer in red gloves'); minutes = time spent.",
             "# a_side: L = A is on the left, R = A is on the right, ? = not clear. Every row needs one.",
             "t_s,a_side"] + [f"{t}," for t in label_times(table)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Labels:
    a_words: str
    minutes: float
    items: tuple            # ((t_s, side), ...), side in {"L", "R", "?"}


def parse_labels(path, table) -> Labels:
    """Strict: size capped, exactly the template's header keys and time grid, every row answered."""
    p = Path(path)

    def bad(msg):
        raise ValueError(f"{p.name}: {msg}")
    try:
        size = p.stat().st_size
    except OSError as e:
        raise ValueError(f"{p.name}: cannot be read ({e.strerror})") from e
    if size > MAX_LABEL_BYTES:
        bad(f"more than {MAX_LABEL_BYTES} bytes")
    try:
        text = p.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        bad("not UTF-8 text")
    head, body = {}, []
    for line in text.splitlines():
        if line.startswith("#"):
            m = re.fullmatch(r"#\s*([A-Za-z_]+)\s*:(.*)", line)
            if m and m.group(1) in _HEADER_KEYS:
                if m.group(1) in head:
                    bad(f"header '{m.group(1)}' appears twice")
                head[m.group(1)] = m.group(2).strip()
            continue
        if line.strip():
            body.append(line)
    missing = [k for k in _HEADER_KEYS if k not in head]
    if missing:
        bad(f"missing header {missing}")
    if head["format"] != LABEL_FORMAT:
        bad("unknown label format")
    if not _FP.fullmatch(head["video_fp"]):
        bad("video_fp must be a sha256 hex digest")
    if head["video_fp"] != table.video_fp:
        bad("labels were written for a different video than this pose table")
    if head["every_s"] != str(LABEL_EVERY_S):
        bad(f"every_s must be {LABEL_EVERY_S} (the gate's grid)")
    a = head["A"]
    if not a or len(a) > MAX_A_CHARS or not a.isprintable():
        bad(f"A must say who A is, in 1-{MAX_A_CHARS} printable characters")
    try:
        minutes = float(head["minutes"])
    except ValueError:
        bad("minutes must be a number")
    if not (math.isfinite(minutes) and 0 < minutes <= MAX_MINUTES):
        bad(f"minutes must be in (0, {MAX_MINUTES}]")
    try:
        rows = list(csv.reader(io.StringIO("\n".join(body))))
    except csv.Error:
        bad("not a readable CSV (a field too long, or broken quoting)")
    if not rows or [c.strip() for c in rows[0]] != ["t_s", "a_side"]:
        bad("the column header must be t_s,a_side")
    want = label_times(table)
    if len(rows) - 1 != len(want):
        bad(f"expected {len(want)} label rows, found {len(rows) - 1}")
    items = []
    for i, (row, t) in enumerate(zip(rows[1:], want), start=1):
        if len(row) != 2:
            bad(f"row {i}: needs exactly two fields")
        ts, side = row[0].strip(), row[1].strip().upper()
        if ts != str(t):
            bad(f"row {i}: time {ts!r} is not the template's {t}")
        if side not in ("L", "R", "?"):
            bad(f"row {i}: a_side must be L, R or ? (got {side!r})")
        items.append((t, side))
    return Labels(a, minutes, tuple(items))


# ---- moments -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Moment:
    t: int
    frame: int
    status: str                 # scored / unclear / not_tracked / overlapping
    left: int = -1              # table row of the person on the left
    right: int = -1
    truth_left: str = ""        # "A" or "B" when scored


def moments(table, labels: Labels) -> "list[Moment]":
    out = []
    area = (table.box[:, 2] - table.box[:, 0]) * (table.box[:, 3] - table.box[:, 1])
    cx = (table.box[:, 0] + table.box[:, 2]) / 2
    wid = table.box[:, 2] - table.box[:, 0]
    for t, side in labels.items:
        f = int(round(t * table.fps))
        if side == "?":
            out.append(Moment(t, f, "unclear"))
            continue
        idx = np.flatnonzero(table.frame == f)
        if len(idx) < 2:
            out.append(Moment(t, f, "not_tracked"))
            continue
        two = idx[np.argsort(-area[idx], kind="stable")[:2]]
        if abs(cx[two[0]] - cx[two[1]]) < OVERLAP_FRAC * float(wid[two].mean()):
            out.append(Moment(t, f, "overlapping"))
            continue
        lft, rgt = (two[0], two[1]) if cx[two[0]] < cx[two[1]] else (two[1], two[0])
        out.append(Moment(t, f, "scored", int(lft), int(rgt), "A" if side == "L" else "B"))
    return out


# ---- methods -------------------------------------------------------------------------------
def glove_names(identity):
    lab = identity.label
    return lambda row: (None if lab[row] < 0 else int(lab[row]))


def bytetrack_names(table):
    return lambda row: int(table.track_id[row])


def linker_names(table):
    """B2: the pose-only linker, every fragment credited one punch, the ring = the whole frame (sparring has
    no punch labels; this adaptation is part of the baseline)."""
    credited = {int(t): 1 for t in np.unique(table.track_id)}
    lk = link(table, credited, Region(0.0, 0.0, float(table.width), float(table.height)))

    def name(row):
        f, t = int(table.frame[row]), int(table.track_id[row])
        for k in (0, 1):
            if lk.membership[f, k] == t:
                return k
        return None
    return name


def score_method(ms: "list[Moment]", name) -> dict:
    """The gate's metrics for one method over one video's moments."""
    mapping, anchor_t = None, None
    seq = []                                   # (t, correct) for decided moments after the anchor
    scored = 0
    for m in ms:
        if m.status != "scored":
            continue
        nl, nr = name(m.left), name(m.right)
        decided_names = nl is not None and nr is not None and nl != nr
        if mapping is None:
            if decided_names:
                tr = "B" if m.truth_left == "A" else "A"
                mapping, anchor_t = {nl: m.truth_left, nr: tr}, m.t
            continue
        scored += 1
        if not decided_names or nl not in mapping or nr not in mapping or mapping[nl] == mapping[nr]:
            continue
        seq.append((m.t, mapping[nl] == m.truth_left))
    if mapping is None:              # never decided: no free label was used, every scored moment is unsure
        scored = sum(1 for m in ms if m.status == "scored")
    decided = len(seq)
    correct = sum(1 for _, c in seq if c)
    swaps = sum(1 for i in range(1, decided) if seq[i][1] != seq[i - 1][1])
    times = [m.t for m in ms if m.status == "scored" and anchor_t is not None and m.t > anchor_t]
    span = (times[-1] - times[0]) if len(times) > 1 else 0
    return {"anchor_t": anchor_t, "scored": scored, "decided": decided, "correct": correct,
            "wrong": decided - correct,
            "accuracy": (correct / decided) if decided else None, "wilson95": wilson(correct, decided),
            "unsure_share": (1 - decided / scored) if scored else None,
            "correct_share": (correct / scored) if scored else None,
            "swaps": swaps, "span_s": span,
            "swaps_per_round": (swaps / (span / ROUND_S)) if span > 0 else None}


def status_counts(ms) -> dict:
    out = {k: 0 for k in ("scored", "unclear", "not_tracked", "overlapping")}
    for m in ms:
        out[m.status] += 1
    return out


def gate_verdict(test: dict, fail_case: dict) -> dict:
    """The four PASS bars from two score results (each: {"methods": {"glove", "bytetrack", "linker"}, ...})."""
    g, fc = test["methods"]["glove"], fail_case["methods"]["glove"]
    baselines_ran = all("error" not in test["methods"][b] for b in ("bytetrack", "linker"))
    base = max((test["methods"][b]["correct_share"] or 0.0) for b in ("bytetrack", "linker"))
    # the share bars compare whole counts (a share computed in floats misses its bar at exact equality:
    # 1 - 35/50 = 0.30000000000000004); the bars are tenths, so x/y <= k/10 is 10x <= ky
    dec, cor, sc = g["decided"], g["correct"], g["scored"]
    w = wilson(cor, dec)
    b1 = dec >= BAR_MIN_DECIDED and 10 * cor >= round(10 * BAR_ACCURACY) * dec and w is not None \
        and w[0] >= BAR_WILSON_LO
    b2 = sc > 0 and 10 * (sc - dec) <= round(10 * BAR_MAX_UNSURE) * sc
    spr = g["swaps_per_round"]
    b3 = (baselines_ran and spr is not None and spr <= BAR_MAX_SWAPS_PER_ROUND and sc > 0
          and cor / sc > base)
    fc_scored = fc["scored"]
    b4 = fc_scored > 0 and 10 * fc["wrong"] <= round(10 * BAR_FAILCASE_MAX_WRONG) * fc_scored
    return {"1_accuracy": b1, "2_unsure": b2, "3_better_than_tracking": b3, "4_fails_safe": b4,
            "baseline_best_correct_share": base, "PASS": bool(b1 and b2 and b3 and b4)}
