"""Glove colour: which of two sparring boxers a tracked person is (glove identity v1).

Why: in sparring where both boxers wear black, pose and tracking alone swap the two when they cross
(linking.py's stated residual), and the clothes carry no colour. If their GLOVES differ in colour, the
glove is the one part of the picture that says who is who. docs/gates/glove-identity.md is the gate,
committed before this file. Pixels are read for identity only, never for punch classification.

The rule, in order (every constant below was written from the geometry of a glove and a forearm and from
synthetic fixtures, BEFORE any study frame was read; DEV-only changes are recorded in docs/quality-loop.md):
  1. Rows: every pose-table row on a frame divisible by `sample_every(fps)` (about SAMPLE_HZ per second).
  2. Glove disc per hand, only where the WRIST is confident (>= WRIST_MIN_CONF, stricter than the 0.3 the
     finder uses) and the elbow gives the forearm (>= ELBOW_MIN_CONF): centre = wrist + GLOVE_AHEAD x
     (wrist - elbow), i.e. a little beyond the wrist where the glove sits; radius = GLOVE_RADIUS x forearm.
     A forearm under MIN_FOREARM_PX or a disc wider than MAX_RADIUS_FRAC of the frame is not read.
  3. A read is CONTESTED when another person's confident wrist is within CONTEST_FACTOR x radius of the
     disc centre: two gloves together (a clinch, a parry, a glove on the other's glove) mix colours.
     Contested reads are counted and never used.
  4. Each disc pixel goes to exactly one of 16 classes in HSV: DARK (value < DARK_V; black clothes and
     shadow), SKIN (hue < SKIN_HUE_MAX with NEUTRAL_S <= saturation < SKIN_SAT_MAX), GREY / WHITE (saturation
     < NEUTRAL_S, split at WHITE_V), or one of 12 HUE bins of 30 degrees (bin 0 centred on red). Skin is
     counted and excluded from the descriptor; black is NOT excluded, because black gloves are a colour too.
  5. Row descriptor: the class shares (15 classes, skin out) over the row's uncontested reads, both hands
     pooled; a row needs MIN_GLOVE_PX non-skin pixels.
  6. Per VIDEO, unsupervised: two clusters of the descriptors (2-means in the Hellinger embedding, a
     deterministic farthest-point start). No label and no colour name is used. The video is SEPARABLE only if
     the centroids are at least SEP_MIN apart (Hellinger), each cluster holds MIN_CLUSTER_SHARE of the rows,
     and -- the internal check -- two boxers on screen at the same time fall into DIFFERENT clusters in at
     least COPRESENT_SPLIT_MIN of at least MIN_COPRESENT_FRAMES such frames. Otherwise every row is unsure.
  7. A row's raw decision is its nearer cluster when the margin (d_far - d_near) / (d_far + d_near) is at
     least MARGIN_MIN, else unsure (motion blur, odd light, half a glove).
  8. Every table row (sampled or not) is then named by the raw decisions of the SAME track id within
     +-SMOOTH_S: at least SMOOTH_MIN_VOTES votes and a SMOOTH_MAJORITY majority, else unsure. Two rows of
     one frame given the same name are both made unsure (they cannot be the same boxer).

What it cannot do (reported, not hidden): two gloves of the same colour (the gate's FAIL-CASE; the
right output is "not separable"); tan or brown gloves overlap skin and are read as skin; coloured light moves
every hue; a ByteTrack id that hops to the other boxer inside SMOOTH_S carries a wrong vote with it.

PIXELS NEVER LEAVE MEMORY: frames are decoded one at a time and only 16 class counts per glove read survive.
Nothing here writes a crop, an image or a pixel array.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .appearance import hsv
from .npyread import read_header
from .punchfind import MIN_CONF
from .video import iter_frames

L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
HANDS = ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW))   # hand 0 = left, 1 = right (the person's, COCO order)

SAMPLE_HZ = 10.0
WRIST_MIN_CONF = 0.5
ELBOW_MIN_CONF = MIN_CONF
GLOVE_AHEAD = 0.25
GLOVE_RADIUS = 0.30
MIN_FOREARM_PX = 6.0
MAX_RADIUS_FRAC = 0.25
MIN_DISC_PX = 20
CONTEST_FACTOR = 2.0

DARK_V = 0.25
NEUTRAL_S = 0.20
WHITE_V = 0.75
SKIN_HUE_MAX = 50.0
SKIN_SAT_MAX = 0.60
HUE_BINS = 12
N_CLASSES = HUE_BINS + 4          # 12 hues, dark, grey, white, skin
DESC = HUE_BINS + 3               # the descriptor: everything but skin
CLASS_NAMES = tuple(f"hue{i * 30:03d}" for i in range(HUE_BINS)) + ("dark", "grey", "white", "skin")

MIN_GLOVE_PX = 30
MIN_FIT_ROWS = 20
SEP_MIN = 0.30
MIN_CLUSTER_SHARE = 0.15
MARGIN_MIN = 0.20
COPRESENT_SPLIT_MIN = 0.70
MIN_COPRESENT_FRAMES = 10
KMEANS_ITERS = 50
SMOOTH_S = 1.0
SMOOTH_MIN_VOTES = 3
SMOOTH_MAJORITY = 0.75

VERDICTS = ("separable", "not_separable", "too_few_rows", "too_few_copresent")
FORMAT = "boxingvi-gloves-1"
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_REGION_PX = 3840 * 2160
_KEYS = {"format", "source", "video_fp", "table_sha256", "meta", "rows", "hand", "bins", "n", "contested"}
_FP = re.compile(r"[0-9a-f]{64}")


def rule_constants() -> dict:
    """Every constant the decision depends on. Its digest is what `glove_identity.py freeze` records."""
    names = ("SAMPLE_HZ", "WRIST_MIN_CONF", "ELBOW_MIN_CONF", "GLOVE_AHEAD", "GLOVE_RADIUS", "MIN_FOREARM_PX",
             "MAX_RADIUS_FRAC", "MIN_DISC_PX", "CONTEST_FACTOR", "DARK_V", "NEUTRAL_S", "WHITE_V",
             "SKIN_HUE_MAX", "SKIN_SAT_MAX", "HUE_BINS", "MIN_GLOVE_PX", "MIN_FIT_ROWS", "SEP_MIN",
             "MIN_CLUSTER_SHARE", "MARGIN_MIN", "COPRESENT_SPLIT_MIN", "MIN_COPRESENT_FRAMES", "KMEANS_ITERS",
             "SMOOTH_S", "SMOOTH_MIN_VOTES", "SMOOTH_MAJORITY", "FORMAT")
    g = globals()
    return {k: g[k] for k in names}


def sample_every(fps: float) -> int:
    if not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
        raise ValueError("fps must be a finite number > 0")
    return max(1, int(round(fps / SAMPLE_HZ)))


# ---- geometry and pixels ----------------------------------------------------------------
def glove_disc(xy, conf, hand: int, width: int, height: int):
    """(cx, cy, r) of one hand's glove disc, or None when the wrist or elbow is not confident enough or the
    forearm is too short or implausibly long."""
    w, e = HANDS[hand]
    if conf[w] < WRIST_MIN_CONF or conf[e] < ELBOW_MIN_CONF:
        return None
    d = np.asarray(xy[w], dtype=np.float64) - np.asarray(xy[e], dtype=np.float64)
    fl = float(np.hypot(d[0], d[1]))
    if not fl >= MIN_FOREARM_PX:
        return None
    r = GLOVE_RADIUS * fl
    if r > MAX_RADIUS_FRAC * min(width, height):
        return None
    c = np.asarray(xy[w], dtype=np.float64) + GLOVE_AHEAD * d
    return float(c[0]), float(c[1]), r


def pixel_classes(rgb) -> np.ndarray:
    """(N_CLASSES,) int64 counts of (n, 3) uint8 RGB pixels; every pixel lands in exactly one class."""
    h, s, v = hsv(rgb)
    dark = v < DARK_V
    skin = ~dark & (h < SKIN_HUE_MAX) & (s >= NEUTRAL_S) & (s < SKIN_SAT_MAX)
    neutral = ~dark & ~skin & (s < NEUTRAL_S)
    chroma = ~dark & ~skin & ~neutral
    hb = np.minimum((((h + 15.0) % 360.0) // 30.0).astype(np.int64), HUE_BINS - 1)
    out = np.zeros(N_CLASSES, dtype=np.int64)
    out[:HUE_BINS] = np.bincount(hb[chroma], minlength=HUE_BINS)
    out[HUE_BINS] = int(dark.sum())
    out[HUE_BINS + 1] = int((neutral & (v < WHITE_V)).sum())
    out[HUE_BINS + 2] = int((neutral & (v >= WHITE_V)).sum())
    out[HUE_BINS + 3] = int(skin.sum())
    return out


def disc_classes(frame: np.ndarray, disc):
    """(class counts, disc pixels inside the frame), or None under MIN_DISC_PX pixels. Counts only."""
    cx, cy, r = disc
    H, W = frame.shape[:2]
    x0, x1 = max(0, int(math.floor(cx - r))), min(W - 1, int(math.ceil(cx + r)))
    y0, y1 = max(0, int(math.floor(cy - r))), min(H - 1, int(math.ceil(cy + r)))
    if x1 < x0 or y1 < y0:
        return None
    X, Y = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float64), np.arange(y0, y1 + 1, dtype=np.float64))
    m = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    n = int(m.sum())
    if n < MIN_DISC_PX:
        return None
    return pixel_classes(frame[y0:y1 + 1, x0:x1 + 1][m]), n


def frame_reads(frame: np.ndarray, table, rows_here, read_rows):
    """Glove reads of the rows in `read_rows` (a subset of `rows_here`, all rows of this frame):
    a list of (row, hand, counts, n, contested)."""
    H, W = frame.shape[:2]
    wrists = {}
    for r in rows_here:
        wrists[r] = [table.xy[r][w] for w, _ in HANDS if table.conf[r][w] >= WRIST_MIN_CONF]
    out = []
    for r in read_rows:
        for hand in (0, 1):
            disc = glove_disc(table.xy[r], table.conf[r], hand, W, H)
            if disc is None:
                continue
            got = disc_classes(frame, disc)
            if got is None:
                continue
            cx, cy, rad = disc
            contested = any(math.hypot(p[0] - cx, p[1] - cy) <= CONTEST_FACTOR * rad
                            for o in rows_here if o != r for p in wrists[o])
            out.append((r, hand, got[0], got[1], contested))
    return out


# ---- the counts of one video ------------------------------------------------------------
def table_digest(table, every: int) -> str:
    """sha256 of the table's CONTENT and the sampling step (counts are row-indexed)."""
    h = hashlib.sha256(f"{FORMAT}|every={int(every)}|".encode())
    for a, dt in ((table.frame, np.int64), (table.track_id, np.int64), (table.box, np.float64),
                  (table.xy, np.float64), (table.conf, np.float64)):
        h.update(np.ascontiguousarray(a, dtype=dt).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class GloveCounts:
    """Class counts per glove read of ONE video's PoseTable, bound to it by source, frame count, row count,
    video fingerprint and a content digest. (rows, hand) pairs are strictly increasing."""

    rows: np.ndarray
    hand: np.ndarray
    bins: np.ndarray        # (m, N_CLASSES) int64
    n: np.ndarray
    contested: np.ndarray   # 0 / 1
    source: str
    n_frames: int
    n_table_rows: int
    every: int
    table_sha256: str
    video_fp: str = ""

    def check_table(self, table) -> None:
        if self.source != table.source:
            raise ValueError("the glove counts and the pose table name different sources")
        if self.n_frames != table.n_frames:
            raise ValueError(f"the glove counts cover {self.n_frames} frames, the pose table {table.n_frames}")
        if self.n_table_rows != len(table.frame):
            raise ValueError("the glove counts were computed from a pose table with a different row count")
        if self.video_fp != table.video_fp:
            raise ValueError("the glove counts and the pose table carry different video fingerprints")
        if self.table_sha256 != table_digest(table, self.every):
            raise ValueError("the glove counts were computed from a pose table with different content: recompute")

    def save(self, path) -> None:
        np.savez_compressed(path, format=np.array(FORMAT), source=np.array(self.source),
                            video_fp=np.array(self.video_fp), table_sha256=np.array(self.table_sha256),
                            meta=np.array([self.n_frames, self.n_table_rows, self.every], dtype=np.int64),
                            rows=self.rows.astype(np.int64), hand=self.hand.astype(np.int64),
                            bins=self.bins.astype(np.int64), n=self.n.astype(np.int64),
                            contested=self.contested.astype(np.int64))

    @classmethod
    def load(cls, path) -> "GloveCounts":
        """Load a counts file: an INPUT file. Size capped; the zip directory and each member's NPY header are
        read before numpy allocates anything; no pickles; exactly the known arrays, dtypes and shapes; and
        counts that are possible (every disc pixel in exactly one class, so each row of bins sums to n)."""
        p = Path(path)

        def bad(msg):
            raise ValueError(f"{p.name}: {msg}")
        try:
            size = p.stat().st_size
        except OSError as e:
            raise ValueError(f"{p.name}: cannot be read ({e.strerror})") from e
        if size > MAX_FILE_BYTES:
            bad(f"more than {MAX_FILE_BYTES} bytes")
        try:
            with zipfile.ZipFile(p) as zf:
                info = zf.infolist()
                if len(info) != len(_KEYS) or {i.filename for i in info} != {k + ".npy" for k in _KEYS}:
                    bad("not a glove counts file (wrong arrays)")
                if sum(i.file_size for i in info) > MAX_UNCOMPRESSED_BYTES:
                    bad(f"declares more than {MAX_UNCOMPRESSED_BYTES} bytes uncompressed")
                for i in info:
                    with zf.open(i) as f:
                        shape, _, dt = read_header(f, i.filename)
                        if dt.hasobject or dt.kind not in "iU":
                            bad(f"{i.filename}: dtype {dt} is not allowed")
                        need = math.prod(shape) * dt.itemsize
                        if need != i.file_size - f.tell():
                            bad(f"{i.filename}: its header and its size disagree")
        except (zipfile.BadZipFile, OSError, EOFError) as e:
            raise ValueError(f"{p.name}: not a readable npz ({type(e).__name__})") from e
        except ValueError as e:
            if str(e).startswith(p.name):
                raise
            raise ValueError(f"{p.name}: not a readable npz ({e})") from e
        try:
            z = np.load(p, allow_pickle=False)
        except (ValueError, OSError) as e:
            raise ValueError(f"{p.name}: not a readable npz") from e
        with z:
            for k in ("format", "source", "video_fp", "table_sha256"):
                if z[k].dtype.kind != "U" or z[k].shape != ():
                    bad(f"{k} must be one text value")
            if str(z["format"]) != FORMAT:
                bad("unknown format")
            fp = str(z["video_fp"])
            if fp and not _FP.fullmatch(fp):
                bad("video_fp must be empty or a sha256 hex digest")
            if not _FP.fullmatch(str(z["table_sha256"])):
                bad("table_sha256 must be a sha256 hex digest")
            meta = z["meta"]
            if meta.dtype != np.int64 or meta.shape != (3,) or meta[0] < 1 or meta[1] < 0 or meta[2] < 1:
                bad("meta must be three whole numbers (frames >= 1, table rows >= 0, every >= 1)")
            arrs = {}
            for k in ("rows", "hand", "n", "contested"):
                a = z[k]
                if a.dtype != np.int64 or a.ndim != 1:
                    bad(f"{k} must be a 1-D int64 array")
                arrs[k] = a
            bins = z["bins"]
            m = len(arrs["rows"])
            if bins.dtype != np.int64 or bins.shape != (m, N_CLASSES):
                bad(f"bins must be an int64 array of shape ({m}, {N_CLASSES})")
            if any(len(arrs[k]) != m for k in ("hand", "n", "contested")):
                bad("rows, hand, n and contested differ in length")
            rows, hand, n, con = arrs["rows"], arrs["hand"], arrs["n"], arrs["contested"]
            if ((hand != 0) & (hand != 1)).any() or ((con != 0) & (con != 1)).any():
                bad("hand and contested must be 0 or 1")
            if m and (rows[0] < 0 or rows[-1] >= meta[1]):
                bad("rows must be table row indices")
            key = rows * 2 + hand
            if m and (np.diff(key) <= 0).any():
                bad("(row, hand) must be strictly increasing")
            if (n < MIN_DISC_PX).any() or (n > MAX_REGION_PX).any() or (bins < 0).any() or (bins > MAX_REGION_PX).any():
                bad("counts are impossible (negative, or a disc outside its size bounds)")
            if (bins.sum(axis=1) != n).any():
                bad("counts are impossible (a read's classes do not add up to its pixels)")
            return cls(rows.copy(), hand.copy(), bins.copy(), n.copy(), con.copy(), str(z["source"]),
                       int(meta[0]), int(meta[1]), int(meta[2]), str(z["table_sha256"]), fp)


def compute_gloves(video, table, info=None, frames=None) -> GloveCounts:
    """Decode `video` once and read both gloves of every table row on a sampled frame. `frames` replaces
    the decoder with any iterator of (H, W, 3) uint8 frames (tests)."""
    every = sample_every(table.fps)
    by_frame: "dict[int, list]" = {}
    for r in range(len(table.frame)):
        by_frame.setdefault(int(table.frame[r]), []).append(r)
    pick = sorted(f for f in by_frame if f % every == 0)
    last = pick[-1] if pick else -1
    if frames is None:
        frames = iter_frames(video, info, max_frames=last + 1) if last >= 0 else iter(())
    reads = []
    seen = 0
    want = set(pick)
    for i, frame in enumerate(frames):
        if i > last:
            break
        seen = i + 1
        if i not in want:
            continue
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 \
                or frame.shape[:2] != (table.height, table.width):
            raise ValueError(f"frame {i} is {frame.dtype} {frame.shape}, the table is {table.height}x{table.width}")
        rows = by_frame[i]
        reads += frame_reads(frame, table, rows, rows)
    if seen < last + 1:
        raise ValueError(f"the video ended after {seen} frames, the pose table has rows up to frame {last}: "
                         "it was extracted from a different video")
    reads.sort(key=lambda x: (x[0], x[1]))
    m = len(reads)
    return GloveCounts(np.array([x[0] for x in reads], dtype=np.int64), np.array([x[1] for x in reads], dtype=np.int64),
                       np.array([x[2] for x in reads], dtype=np.int64).reshape(m, N_CLASSES),
                       np.array([x[3] for x in reads], dtype=np.int64),
                       np.array([int(x[4]) for x in reads], dtype=np.int64),
                       table.source, table.n_frames, len(table.frame), every, table_digest(table, every),
                       table.video_fp)


# ---- the per-video model ------------------------------------------------------------------
def row_descriptors(counts: GloveCounts):
    """(rows, P, readable): table rows with at least MIN_GLOVE_PX uncontested non-skin glove pixels, their
    class shares (DESC columns) and that pixel count."""
    use = counts.contested == 0
    rows = counts.rows[use]
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, DESC)), np.zeros(0, dtype=np.int64)
    u, inv = np.unique(rows, return_inverse=True)
    agg = np.zeros((len(u), DESC), dtype=np.int64)
    np.add.at(agg, inv, counts.bins[use][:, :DESC])
    readable = agg.sum(axis=1)
    keep = readable >= MIN_GLOVE_PX
    return u[keep], agg[keep] / readable[keep][:, None], readable[keep]


def hellinger(S, c) -> np.ndarray:
    """Hellinger distance of sqrt-share rows S (n, k) to a sqrt-share vector c (k,): in [0, 1]."""
    return np.sqrt(np.maximum(0.0, ((S - c) ** 2).sum(axis=1) / 2.0))


def two_means(P: np.ndarray):
    """Deterministic 2-means in the Hellinger embedding: (centroids (2, k) of sqrt shares, assignment).
    Start: the row farthest from the mean, then the row farthest from it. Clusters are then ordered so that
    cluster 0's centroid is lexicographically larger (a fixed, meaningless naming)."""
    S = np.sqrt(P)
    i0 = int(np.argmax(hellinger(S, S.mean(axis=0))))
    i1 = int(np.argmax(hellinger(S, S[i0])))
    C = np.stack([S[i0], S[i1]])
    a = None
    for _ in range(KMEANS_ITERS):
        new = np.argmin(np.stack([hellinger(S, C[0]), hellinger(S, C[1])], axis=1), axis=1)
        if a is not None and (new == a).all():
            break
        a = new
        if (a == 0).all() or (a == 1).all():   # one cluster: identical rows; the separation test refuses it
            break
        C = np.stack([S[a == 0].mean(axis=0), S[a == 1].mean(axis=0)])
    if tuple(C[1]) > tuple(C[0]):
        C = C[::-1].copy()
        a = 1 - a
    return C, a


@dataclass(frozen=True)
class GloveIdentity:
    """Per table row: 0 or 1 (which glove cluster; the naming means nothing on its own) or -1 (unsure)."""

    verdict: str
    label: np.ndarray
    diagnostics: dict


def identify(table, counts: GloveCounts) -> GloveIdentity:
    counts.check_table(table)
    n_rows = len(table.frame)
    label = np.full(n_rows, -1, dtype=np.int64)
    rows, P, readable = row_descriptors(counts)
    diag = {"reads": int(len(counts.rows)), "contested_reads": int(counts.contested.sum()),
            "readable_rows": int(len(rows)),
            "skin_share": (float(counts.bins[:, HUE_BINS + 3].sum() / counts.n.sum()) if len(counts.n) else None)}
    if len(rows) < MIN_FIT_ROWS:
        return GloveIdentity("too_few_rows", label, diag)
    C, a = two_means(P)
    S = np.sqrt(P)
    d0, d1 = hellinger(S, C[0]), hellinger(S, C[1])
    near = np.where(d0 <= d1, 0, 1)
    dn, df = np.minimum(d0, d1), np.maximum(d0, d1)
    margin = np.where(dn + df > 0, (df - dn) / np.where(dn + df > 0, dn + df, 1.0), 0.0)
    raw = np.where(margin >= MARGIN_MIN, near, -1)
    sep = float(np.sqrt(((C[0] - C[1]) ** 2).sum() / 2.0))
    shares = [float((a == k).mean()) for k in (0, 1)]
    diag.update({"separation": sep, "cluster_shares": shares, "raw_decided_share": float((raw >= 0).mean()),
                 "centroid_shares": [dict(zip(CLASS_NAMES[:DESC], np.round(C[k] ** 2, 4).tolist())) for k in (0, 1)]})
    # internal check: two people on screen together should fall into different clusters
    frame_of = table.frame[rows]
    area = (table.box[rows, 2] - table.box[rows, 0]) * (table.box[rows, 3] - table.box[rows, 1])
    pairs = split = 0
    for f in np.unique(frame_of):
        idx = np.flatnonzero(frame_of == f)
        if len(idx) < 2:
            continue
        top = idx[np.argsort(-area[idx], kind="stable")[:2]]
        if (raw[top] < 0).any():
            continue
        pairs += 1
        split += int(raw[top[0]] != raw[top[1]])
    diag.update({"copresent_pairs": pairs, "copresent_split": (split / pairs) if pairs else None})
    if sep < SEP_MIN or min(shares) < MIN_CLUSTER_SHARE:
        return GloveIdentity("not_separable", label, diag)
    if pairs < MIN_COPRESENT_FRAMES:
        return GloveIdentity("too_few_copresent", label, diag)
    if split / pairs < COPRESENT_SPLIT_MIN:
        return GloveIdentity("not_separable", label, diag)
    # smoothing within a track id, then the one-name-per-frame rule
    half = SMOOTH_S * table.fps
    dec = raw >= 0
    for t in np.unique(table.track_id):
        mine = np.flatnonzero(table.track_id == t)
        vote = dec & (table.track_id[rows] == t)
        vf = table.frame[rows][vote].astype(np.float64)
        vl = raw[vote]
        order = np.argsort(vf, kind="stable")
        vf, vl = vf[order], vl[order]
        ones = np.concatenate([[0], np.cumsum(vl == 1)])
        for r in mine.tolist():
            f = float(table.frame[r])
            lo, hi = np.searchsorted(vf, f - half, "left"), np.searchsorted(vf, f + half, "right")
            n = int(hi - lo)
            if n < SMOOTH_MIN_VOTES:
                continue
            k1 = int(ones[hi] - ones[lo])
            if k1 >= SMOOTH_MAJORITY * n:
                label[r] = 1
            elif n - k1 >= SMOOTH_MAJORITY * n:
                label[r] = 0
    for f in np.unique(table.frame):
        idx = np.flatnonzero((table.frame == f) & (label >= 0))
        for k in (0, 1):
            same = idx[label[idx] == k]
            if len(same) > 1:
                label[same] = -1
    diag["named_rows_share"] = float((label >= 0).mean()) if n_rows else None
    return GloveIdentity("separable", label, diag)


def rule_digest() -> str:
    return hashlib.sha256(json.dumps(rule_constants(), sort_keys=True).encode()).hexdigest()
