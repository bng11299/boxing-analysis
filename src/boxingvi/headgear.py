"""Corner colour from the HEADGEAR, read while two people are on screen together (kit colour v3).

Why a v3. ADR-0013's torso rule (v2) scored chance (52.2% opposite on 25 held-out videos). Brendan's eye
check on 2026-10-22 (development video GH159681) found why: both boxers of that bout wear RED vests, and the
corner colour is on the headgear and gloves. The same check found that a ByteTrack id is not always one
person (id 1102 hops to the referee and to the other boxer), so a colour summarised over a whole fragment
mixes people. docs/gates/appearance-identity.md is the gate, committed before this design.

Designed on the four DEVELOPMENT videos only (olympic_colour.DEV_VIDEOS) and frozen before any held-out
video was read; the dev numbers that chose it are in docs/quality-loop.md (`feat/appearance-identity`).

The rule, in order:
  1. Rows: every pose-table row on a frame divisible by SAMPLE_EVERY (10 Hz at 50 fps).
  2. Head size s: the ear-to-ear distance when both ears are confident, else 2.5 x the eye-to-eye distance,
     else 0.45 x the torso (shoulder-mid to hip-mid); a head under MIN_HEAD_PX is not read.
  3. The CROWN disc: centre = face centre (mean of the confident nose / eye / ear keypoints) moved
     CROWN_UP x s along the neck axis (shoulder-mid -> face centre; straight up in the image without both
     shoulders), radius CROWN_RADIUS x s. That is the top of the headgear, above the face: the face itself
     is skin, which overlaps red in hue. A crown of fewer than MIN_REGION_PX pixels in the frame is skipped.
  4. Colour per pixel in HSV (exactly the bins the design was scored on):
        RED   hue in [345, 360) + [0, 30) degrees, saturation >= 0.70, value >= 0.20
        BLUE  hue in [195, 285) degrees,           saturation >= 0.50, value >= 0.08
  5. A decision over a SET of rows (ADR-0013 v2's constants, unchanged): a kit row has enough red + blue
     pixels; the median kit index (B - R) / (B + R) over kit rows decides blue (>= 0.5) or red (<= -0.5),
     and a well-seen set with almost no kit is `none`; anything else is `unsure` (`appearance.decide`).
  6. The self-check reads each fragment of a co-present pair over the frames the two SHARE only, not its
     whole length, so a later hop of the id to someone else does not leak into the reading.

Dev result that chose it (numbers only, docs/quality-loop.md): crown read on the co-present window 98 / 108
decided pairs opposite (kam4 97 / 106); the same crown over whole fragments 89 / 124; the v2 face-centred
head disc over whole fragments 85 / 123; gloves 82-86 / 141-145. On kam2 the blue kit is washed out (blue-hue
pixels sit at saturation < 0.3) and the rule rarely decides there: kam2 is expected to be "not established".

PIXELS NEVER LEAVE MEMORY: frames are decoded one at a time and only three counts per row survive
(red, blue, region pixels). Nothing here writes a crop, an image or a pixel array (minors, academic licence).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .appearance import DECISIONS, MIN_REGION_PX, decide, hsv, is_kit_row
from .punchfind import L_HIP, L_SHOULDER, MIN_CONF, R_HIP, R_SHOULDER
from .video import iter_frames

SAMPLE_EVERY = 5            # 10 Hz at 50 fps
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
FACE = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
HEAD_FROM_EARS = 1.0
HEAD_FROM_EYES = 2.5
HEAD_FROM_TORSO = 0.45
MIN_HEAD_PX = 4.0
CROWN_UP = 0.5
CROWN_RADIUS = 0.45

RED_HUE_LO, RED_HUE_HI = 345.0, 30.0       # red is [345, 360) + [0, 30)
RED_SAT_MIN, RED_VAL_MIN = 0.70, 0.20
BLUE_HUE_LO, BLUE_HUE_HI = 195.0, 285.0    # blue is [195, 285)
BLUE_SAT_MIN, BLUE_VAL_MIN = 0.50, 0.08

CO_PRESENT_S = 0.5          # ADR-0013: on screen together this long = two different people
FORMAT = "boxingvi-headgear-1"
MAX_FILE_BYTES = 256 * 1024 * 1024
_FP = re.compile(r"[0-9a-f]{64}")


# ---- geometry ----------------------------------------------------------------------
def head_size(xy, conf, min_conf: float = MIN_CONF) -> float:
    """Head size in pixels (ears, else eyes, else torso), NaN when none is usable or it is under MIN_HEAD_PX."""
    xy = np.asarray(xy, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)

    def ok(*j):
        return all(conf[k] >= min_conf for k in j)
    if ok(L_EAR, R_EAR):
        s = HEAD_FROM_EARS * float(np.linalg.norm(xy[L_EAR] - xy[R_EAR]))
    elif ok(L_EYE, R_EYE):
        s = HEAD_FROM_EYES * float(np.linalg.norm(xy[L_EYE] - xy[R_EYE]))
    elif ok(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        sh = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
        hp = (xy[L_HIP] + xy[R_HIP]) / 2
        s = HEAD_FROM_TORSO * float(np.linalg.norm(sh - hp))
    else:
        return float("nan")
    return s if s >= MIN_HEAD_PX else float("nan")


def crown_disc(xy, conf, min_conf: float = MIN_CONF):
    """(cx, cy, r) of the crown disc, or None without a confident face keypoint or a usable head size."""
    xy = np.asarray(xy, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    face = [j for j in FACE if conf[j] >= min_conf]
    s = head_size(xy, conf, min_conf)
    if not face or s != s:
        return None
    fc = xy[face].mean(axis=0)
    up = np.array([0.0, -1.0])
    if conf[L_SHOULDER] >= min_conf and conf[R_SHOULDER] >= min_conf:
        d = fc - (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
        n = float(np.linalg.norm(d))
        if n > 1.0:
            up = d / n
    c = fc + CROWN_UP * s * up
    return float(c[0]), float(c[1]), CROWN_RADIUS * s


def kit_masks(rgb):
    """(red, blue) boolean arrays for (n, 3) uint8 RGB pixels, v3 bands."""
    h, s, v = hsv(rgb)
    red = (s >= RED_SAT_MIN) & (v >= RED_VAL_MIN) & ((h >= RED_HUE_LO) | (h < RED_HUE_HI))
    blue = (s >= BLUE_SAT_MIN) & (v >= BLUE_VAL_MIN) & (h >= BLUE_HUE_LO) & (h < BLUE_HUE_HI)
    return red, blue


def crown_counts(frame: np.ndarray, xy, conf, min_conf: float = MIN_CONF, min_px: int = MIN_REGION_PX):
    """(red px, blue px, crown px) of one person-row, or None when the crown is unreadable or under `min_px`
    pixels inside the frame. Only the counts leave this function."""
    disc = crown_disc(xy, conf, min_conf)
    if disc is None:
        return None
    cx, cy, r = disc
    H, W = frame.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(W - 1, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(H - 1, int(cy + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    X, Y = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float64), np.arange(y0, y1 + 1, dtype=np.float64))
    m = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    n = int(m.sum())
    if n < min_px:
        return None
    red, blue = kit_masks(frame[y0:y1 + 1, x0:x1 + 1][m])
    return int(red.sum()), int(blue.sum()), n


# ---- the counts of one video ---------------------------------------------------------
@dataclass(frozen=True)
class HeadgearCounts:
    """Crown counts per read row of ONE video's PoseTable, bound to it by source name, frame count, row
    count and video fingerprint. `rows` are indices into the table, strictly increasing."""

    rows: np.ndarray
    red: np.ndarray
    blue: np.ndarray
    n: np.ndarray
    source: str
    n_frames: int
    n_table_rows: int
    video_fp: str = ""

    def check_table(self, table) -> None:
        """Refuse a table these counts were not computed from."""
        if self.source != table.source:
            raise ValueError(f"the headgear counts are for {self.source!r}, the pose table for {table.source!r}")
        if self.n_frames != table.n_frames:
            raise ValueError(f"the headgear counts cover {self.n_frames} frames, the pose table {table.n_frames}")
        if self.n_table_rows != len(table.frame):
            raise ValueError(f"the headgear counts index {self.n_table_rows} table rows, the pose table has "
                             f"{len(table.frame)}: they were computed from another table")
        if self.video_fp != table.video_fp:
            raise ValueError("the headgear counts and the pose table carry different video fingerprints")

    def save(self, path) -> None:
        np.savez_compressed(path, format=np.array(FORMAT), source=np.array(self.source),
                            video_fp=np.array(self.video_fp),
                            meta=np.array([self.n_frames, self.n_table_rows], dtype=np.int64),
                            rows=self.rows.astype(np.int64), red=self.red.astype(np.int64),
                            blue=self.blue.astype(np.int64), n=self.n.astype(np.int64))

    @classmethod
    def load(cls, path) -> "HeadgearCounts":
        """Load a counts file. It is an INPUT file: size capped, no pickles, exactly the known arrays with the
        known dtypes and shapes, and counts that are internally possible (0 <= red, blue; red + blue <= n;
        n >= MIN_REGION_PX; rows strictly increasing inside the table)."""
        p = Path(path)
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{p.name}: more than {MAX_FILE_BYTES} bytes")

        def bad(msg):
            raise ValueError(f"{p.name}: {msg}")
        try:
            z = np.load(p, allow_pickle=False)
        except (ValueError, OSError) as e:
            raise ValueError(f"{p.name}: not a readable npz ({e})") from e
        with z:
            keys = {"format", "source", "video_fp", "meta", "rows", "red", "blue", "n"}
            if set(z.files) != keys:
                bad("not a headgear counts file (wrong arrays)")
            for k in ("format", "source", "video_fp"):
                if z[k].dtype.kind != "U" or z[k].shape != ():
                    bad(f"{k} must be one text value")
            if str(z["format"]) != FORMAT:
                bad(f"unknown format {str(z['format'])!r}")
            fp = str(z["video_fp"])
            if fp and not _FP.fullmatch(fp):
                bad("video_fp must be empty or a 64-character lowercase sha256 hex digest")
            meta = z["meta"]
            if meta.dtype != np.int64 or meta.shape != (2,) or meta[0] < 1 or meta[1] < 0:
                bad("meta must be two whole numbers (frames >= 1, table rows >= 0)")
            arrs = {}
            for k in ("rows", "red", "blue", "n"):
                a = z[k]
                if a.dtype != np.int64 or a.ndim != 1:
                    bad(f"{k} must be a 1-D int64 array")
                arrs[k] = a
            m = len(arrs["rows"])
            if any(len(arrs[k]) != m for k in ("red", "blue", "n")):
                bad("rows, red, blue and n differ in length")
            rows, red, blue, n = arrs["rows"], arrs["red"], arrs["blue"], arrs["n"]
            if m and (rows[0] < 0 or rows[-1] >= meta[1] or (np.diff(rows) <= 0).any()):
                bad("rows must be strictly increasing table row indices")
            if (red < 0).any() or (blue < 0).any() or (red + blue > n).any() or (n < MIN_REGION_PX).any():
                bad("counts are impossible (negative, more kit than pixels, or a region under the minimum)")
            return cls(rows.copy(), red.copy(), blue.copy(), n.copy(), str(z["source"]), int(meta[0]),
                       int(meta[1]), fp)


def compute_headgear(video, table, info=None, every: int = SAMPLE_EVERY, min_conf: float = MIN_CONF,
                     frames=None) -> HeadgearCounts:
    """Decode `video` once and count crown pixels for every table row on a frame divisible by `every`.
    `frames` replaces the decoder with any iterator of (H, W, 3) uint8 frames (tests)."""
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError("every must be a whole number >= 1")
    pick = np.flatnonzero(table.frame % every == 0)
    by_frame: "dict[int, list]" = {}
    for r in pick.tolist():
        by_frame.setdefault(int(table.frame[r]), []).append(r)
    last = max(by_frame) if by_frame else -1
    if frames is None:
        frames = iter_frames(video, info, max_frames=last + 1) if last >= 0 else iter(())
    got: "dict[int, tuple]" = {}
    seen = 0
    for i, frame in enumerate(frames):
        if i > last:
            break
        seen = i + 1
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 \
                or frame.shape[:2] != (table.height, table.width):
            raise ValueError(f"frame {i} is {frame.dtype} {frame.shape}, the table is {table.height}x{table.width}")
        for r in by_frame.get(i, ()):
            c = crown_counts(frame, table.xy[r], table.conf[r], min_conf)
            if c is not None:
                got[r] = c
    if seen < last + 1:
        raise ValueError(f"the video ended after {seen} frames, the pose table has rows up to frame {last}: "
                         "it was extracted from a different video")
    rows = np.array(sorted(got), dtype=np.int64)
    cnt = np.array([got[r] for r in rows.tolist()], dtype=np.int64).reshape(-1, 3)
    return HeadgearCounts(rows, cnt[:, 0], cnt[:, 1], cnt[:, 2], table.source, table.n_frames,
                          len(table.frame), table.video_fp)


# ---- decisions -----------------------------------------------------------------------
def decide_rows(red, blue, n) -> str:
    """red / blue / none / unsure over a set of read rows (v2's constants, appearance.decide)."""
    idx, share = [], []
    for r, b, k in zip(np.asarray(red).tolist(), np.asarray(blue).tolist(), np.asarray(n).tolist()):
        share.append((r + b) / k)
        if is_kit_row(r, b, k):
            idx.append((b - r) / (b + r))
    index = float(np.median(idx)) if idx else 0.0
    kit = float(np.median(share)) if share else 0.0
    return decide(index, kit, len(share), len(idx))


def headgear_check(table, attrs, counts: HeadgearCounts) -> dict:
    """The non-circular self-check (ADR-0013's pairs), each fragment read over the shared frames only.

    Two fragments that both carry a labelled punch credited to them and share at least CO_PRESENT_S of frames
    are two different people, so the corner colours should come out opposite. Counts only;
    `opposite_share` is None when no pair is decided (no measurement, not 0%)."""
    counts.check_table(table)
    credited = sorted({int(a.track_id) for a in attrs if a.track_id is not None})
    frames = {t: np.unique(table.frame[table.track_id == t]) for t in credited}
    need = max(1, int(math.ceil(CO_PRESENT_S * table.fps - 1e-9)))
    r_tid = table.track_id[counts.rows]
    r_frame = table.frame[counts.rows]
    pairs = opposite = same = 0
    by = {d: 0 for d in DECISIONS}
    for i, a in enumerate(credited):
        for b in credited[i + 1:]:
            common = np.intersect1d(frames[a], frames[b], assume_unique=True)
            if len(common) < need:
                continue
            pairs += 1
            dec = []
            for t in (a, b):
                m = (r_tid == t) & np.isin(r_frame, common)
                dec.append(decide_rows(counts.red[m], counts.blue[m], counts.n[m]))
            for d in dec:
                by[d] += 1
            if dec[0] in ("red", "blue") and dec[1] in ("red", "blue"):
                if dec[0] == dec[1]:
                    same += 1
                else:
                    opposite += 1
    decided = opposite + same
    return {"credited_fragments": len(credited), "copresent_pairs": pairs, "decided_pairs": decided,
            "opposite": opposite, "same": same, "window_decisions": by,
            "opposite_share": (opposite / decided) if decided else None}


def pool(rows) -> dict:
    """Pool per-video `headgear_check` dicts from their integer counts, divided once."""
    rows = list(rows)
    out = {"videos": len(rows)}
    for k in ("credited_fragments", "copresent_pairs", "decided_pairs", "opposite", "same"):
        out[k] = sum(r[k] for r in rows)
    out["window_decisions"] = {d: sum(r["window_decisions"][d] for r in rows) for d in DECISIONS}
    d = out["decided_pairs"]
    out["opposite_share"] = (out["opposite"] / d) if d else None
    out["decided_share"] = (d / out["copresent_pairs"]) if out["copresent_pairs"] else None
    out["wilson95"] = wilson(out["opposite"], d)
    return out


# ---- intervals -----------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval [lo, hi] for k successes of n, or None when n == 0."""
    if isinstance(k, bool) or isinstance(n, bool) or not isinstance(k, int) or not isinstance(n, int) \
            or n < 0 or not 0 <= k <= n:
        raise ValueError("wilson needs whole numbers 0 <= k <= n")
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def cluster_bootstrap(per_video, draws: int = 10_000, seed: int = 20261022, level: float = 0.95):
    """Percentile interval of the pooled opposite share, resampling VIDEOS with replacement.

    `per_video` is a list of (opposite, decided) integer pairs. A draw with no decided pair has no share and
    is dropped; how many were dropped is returned. Returns {"lo", "hi", "draws", "dropped"} or None when no
    video has a decided pair."""
    pv = [(int(o), int(d)) for o, d in per_video]
    if any(d < 0 or not 0 <= o <= d for o, d in pv):
        raise ValueError("each video needs 0 <= opposite <= decided")
    if not pv or sum(d for _, d in pv) == 0:
        return None
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1 or not 0 < level < 1:
        raise ValueError("draws must be a whole number >= 1 and level in (0, 1)")
    o = np.array([x for x, _ in pv], dtype=np.int64)
    d = np.array([y for _, y in pv], dtype=np.int64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(pv), size=(draws, len(pv)))
    so, sd = o[pick].sum(axis=1), d[pick].sum(axis=1)
    ok = sd > 0
    share = so[ok] / sd[ok]
    a = (1 - level) / 2
    return {"lo": float(np.quantile(share, a)), "hi": float(np.quantile(share, 1 - a)),
            "draws": int(draws), "dropped": int((~ok).sum())}
