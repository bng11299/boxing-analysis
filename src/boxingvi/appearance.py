"""Kit colour per fragment: which corner a track id belongs to (red or blue), or that it is nobody.

VERSION 2 (2026-10-08). Version 1 (kit = saturated red / blue pixels in torso, head and gloves) failed
its non-circular self-check (29.1% opposite): skin and warm-lit kit overlap in hue, and the blue kit
on this footage is DARK navy. Version 2 was designed on four DEVELOPMENT videos only (GH078416,
GH128416, GH069681, GH159681; a split fixed by video before the design, see olympic_colour.py) and is
judged once on the other 25 (the TEST videos). The reason for each change is given below.

Why: pose alone cannot say WHICH boxer a fragment is (the linker swaps two boxers who cross while
their ids are lost) or whether a still, well-seen person is a boxer at all (a spectator can stand in
for a missing fragment). Olympic boxers wear their corner's colour. This module turns that into a
per-fragment decision, from NUMBERS only.

PIXELS NEVER LEAVE MEMORY. Frames are decoded one at a time, one region of each is classified into
red / blue / neither, and only counts survive. Nothing here writes a crop, a thumbnail, a debug image
or a pixel array, and `Colours` holds counts and medians, never pixels (the footage shows minors and
is under an academic licence; see the project rules).

The test, in order:
  1. Rows. The PoseTable is sampled every SAMPLE_EVERY frames (10 Hz at 50 fps), plus each
     fragment's first and last row with a confident torso, so even a fragment of a few frames
     has its ends read.
  2. Region per row: the TORSO only (quadrilateral shoulders + hips, shrunk 20% towards its
     centroid because keypoints are a few pixels off). Head and glove discs are still built by
     `region_masks` but are NOT read (DECISION_REGIONS): on the development videos they did not
     help, see "what changed and why" below.
  3. Colour per pixel in HSV:
        RED   hue in [340, 360) + [0, 25) degrees, saturation >= 0.70, value >= 0.20
        BLUE  hue in [200, 290) degrees,           saturation >= 0.50, value >= 0.08
  4. Per row: the count of red pixels R and blue pixels B among the N torso pixels. A KIT ROW has
     R + B >= max(ROW_KIT_PX, ROW_KIT_MIN x N); its kit index is (B - R) / (B + R): +1 all blue,
     -1 all red.
  5. Per fragment (rows = rows with a readable torso; kit_rows = its kit rows): the MEDIAN kit index
     over the kit rows (robust to a clinch frame in which the other boxer's pixels fill the box) and
     the MEDIAN kit share (R + B) / N over all rows.
        blue    rows >= MIN_ROWS, kit_rows >= MIN_ROWS, kit_rows >= KIT_ROW_FRAC x rows, and
                median index >= INDEX_MIN
        red     the same with median index <= -INDEX_MIN
        none    rows >= NONE_ROWS and median kit share < NONE_MAX: well seen, wearing neither
                colour (a referee in white, most of the crowd)
        unsure  everything else: too few rows, both colours (|index| < INDEX_MIN), too little kit.

What changed from version 1, and why (development videos only; numbers in docs/quality-loop.md):
  * Skin band excluded. Bare skin on the dev videos sits at hue 15-40 degrees with saturation up
    to about 0.6; v1's floor of 0.45 read it as red. The red kit is orange-red (hue 10-20) on the
    kam2 camera and hue 350-5 on kam4, both at saturation 0.7-1.0. RED_SAT_MIN = 0.70 sits in the
    gap: fragments that are not red kit have 0.000-0.009 of their torso at hue 5-25 and saturation
    >= 0.7, red-kit fragments 0.3-0.7. Decisions barely move for floors 0.6-0.8 and hue ceilings
    15-35 degrees (red fragments 104-111 of 192 credited on dev).
  * The blue kit is DARK. On both cameras the blue boxer's vest is navy at value 0.08-0.2 (hue
    220-240, saturation ~0.9); v1's value floor of 0.20 threw it away. BLUE_VAL_MIN = 0.08 recovers
    it (credited fragments called blue on dev: 27 with a 0.20 floor, 54 with 0.08); below 0.08 a
    camera's hue is noise. Black clothing is under 0.08 and stays out. Red has no dark band
    (dark-red pixels are rare), so red keeps 0.20.
  * Relative, not absolute. A fixed share floor (v1: 10%) misses a navy vest that is 7% of the
    torso. The decision is the index (B - R) / (B + R) over kit pixels only; |index| >= 0.5 means
    one colour at least three times the other (v1's ratio), with a minimum number of kit rows.
  * Torso only. The head disc includes background (ring corner pads and ropes are red and blue) and
    the gloves are mostly arm. On dev, torso-only had the most decided fragments and no region
    combination improved the self-check beyond noise (about 100-150 decided pairs each: torso
    50%, head 64%, torso + head pooled 53%, torso + head + gloves 50%).
  * No per-video adaptive threshold: the index distribution on dev is bimodal at about -1 and +1
    with few fragments between, and the decisions are flat for thresholds 0.2-0.5.

What it cannot do (reported, not hidden): a spectator, a coach or a judge in a red or blue top is
decided red or blue like a boxer (the punch requirement in the linker is what keeps them out of an
identity); strong coloured lighting moves every hue; a clinch fills the box with the other boxer's
kit for a stretch (the median over rows absorbs it); where both boxers of a bout wear the same vest
colour the torso cannot tell them apart (seen on dev, GH159681 from about 90 s on); and the colours
were checked by the non-circular self-check (`colour_check`), not by eye.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .punchfind import L_HIP, L_SHOULDER, MIN_CONF, R_HIP, R_SHOULDER
from .video import iter_frames

# ---- rows --------------------------------------------------------------------------
SAMPLE_EVERY = 5          # 10 Hz at 50 fps

# ---- regions -----------------------------------------------------------------------
TORSO_SHRINK = 0.8        # the torso quadrilateral keeps 80% of its size around its centroid
DISC_RADIUS_TORSO = 0.35  # head and glove discs, in torso lengths
MIN_REGION_PX = 30        # a smaller region is a few pixels of noise: skipped
FACE = (0, 1, 2, 3, 4)    # COCO-17: nose, eyes, ears
WRISTS = (9, 10)
TORSO_PER_BOX_HEIGHT = 0.3   # the linker's fallback when the four torso joints are not all confident
MIN_TORSO_PX = 2.0           # a torso shorter than this (pixels) has no usable discs
ALL_REGIONS = ("torso", "head", "glove_l", "glove_r")

# ---- colour ------------------------------------------------------------------------
RED_HUE_MIN = 340.0       # degrees: red is [340, 360) + [0, 25)
RED_HUE_MAX = 25.0        # exclusive
RED_SAT_MIN = 0.70
RED_VAL_MIN = 0.20
BLUE_HUE_MIN = 200.0      # degrees: blue is [200, 290)
BLUE_HUE_MAX = 290.0      # exclusive
BLUE_SAT_MIN = 0.50
BLUE_VAL_MIN = 0.08       # dark navy vests live at value 0.08-0.2

# ---- decision ----------------------------------------------------------------------
DECISION_REGIONS = ("torso",)   # regions the decision reads; ALL_REGIONS is the version-1 choice
ROW_KIT_MIN = 0.05        # a row is a kit row when R + B is at least this share of its pixels ...
ROW_KIT_PX = 20           # ... and at least this many pixels
INDEX_MIN = 0.5           # |median index| >= 0.5: one colour at least 3x the other
MIN_ROWS = 3
KIT_ROW_FRAC = 0.5        # at least half of the rows must be kit rows
NONE_ROWS = 5
NONE_MAX = 0.02           # median kit share under 2% over >= 5 rows: not a boxer

DECISIONS = ("red", "blue", "none", "unsure")
FORMAT = "boxingvi-colours-2"
MAX_FILE_BYTES = 64 * 1024 * 1024
_FP = re.compile(r"[0-9a-f]{64}")


# ---- geometry ----------------------------------------------------------------------
def _torso_length(xy: np.ndarray, conf: np.ndarray, box, min_conf: float) -> float:
    """Shoulder-mid to hip-mid in pixels when all four joints are confident, else the box height x 0.3
    (the linker's fallback). NaN when neither is usable."""
    idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if (conf[idx] >= min_conf).all():
        sh = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2
        hp = (xy[L_HIP] + xy[R_HIP]) / 2
        n = float(np.linalg.norm(sh - hp))
        if n >= MIN_TORSO_PX:
            return n
    h = float(box[3]) - float(box[1])
    return h * TORSO_PER_BOX_HEIGHT if h * TORSO_PER_BOX_HEIGHT >= MIN_TORSO_PX else float("nan")


def shrink_quad(quad: np.ndarray, keep: float = TORSO_SHRINK) -> np.ndarray:
    """The quadrilateral scaled by `keep` about its centroid (the mean of its vertices)."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    return c + keep * (q - c)


def _in_quad(X: np.ndarray, Y: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for pixel coordinates (X, Y), any vertex order."""
    inside = np.zeros(X.shape, dtype=bool)
    n = len(quad)
    for k in range(n):
        x0, y0 = quad[k]
        x1, y1 = quad[(k + 1) % n]
        if y0 == y1:
            continue
        cross = ((Y < y0) != (Y < y1))
        xi = x0 + (Y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cross & (X < xi)
    return inside


def region_masks(xy: np.ndarray, conf: np.ndarray, box, width: int, height: int,
                 min_conf: float = MIN_CONF, min_px: int = MIN_REGION_PX, use=ALL_REGIONS):
    """The pooled region of one person-row, as a window of the frame.

    Returns (x0, y0, mask, parts) or None when no region has `min_px` pixels in the frame. `mask`
    is the boolean UNION of the kept regions over the window [y0, y0 + h) x [x0, x0 + w); `parts`
    maps a region name (torso, head, glove_l, glove_r) to its pixel count BEFORE the union (only
    kept regions). A pixel (ix, iy) is inside a region when its integer coordinates are; pixels
    outside the frame are ignored. `use` names the regions to build at all (the decision reads
    DECISION_REGIONS); a region not in `use` is neither built nor counted."""
    xy = np.asarray(xy, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    shapes = []                                    # (name, polygon-or-disc, bounding box x0, y0, x1, y1)
    tl = _torso_length(xy, conf, box, min_conf)
    idx = [L_SHOULDER, R_SHOULDER, R_HIP, L_HIP]      # a proper quadrilateral: same-side joints are neighbours
    if "torso" in use and (conf[idx] >= min_conf).all():
        q = shrink_quad(xy[idx])
        shapes.append(("torso", ("quad", q), q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()))
    if tl == tl:                                   # not NaN
        r = DISC_RADIUS_TORSO * tl
        face = [j for j in FACE if conf[j] >= min_conf]
        if "head" in use and face:
            c = xy[face].mean(axis=0)
            shapes.append(("head", ("disc", c, r), c[0] - r, c[1] - r, c[0] + r, c[1] + r))
        for name, j in (("glove_l", WRISTS[0]), ("glove_r", WRISTS[1])):
            if name in use and conf[j] >= min_conf:
                c = xy[j]
                shapes.append((name, ("disc", c, r), c[0] - r, c[1] - r, c[0] + r, c[1] + r))
    if not shapes:
        return None
    bx0 = max(0, int(math.floor(min(s[2] for s in shapes))))
    by0 = max(0, int(math.floor(min(s[3] for s in shapes))))
    bx1 = min(width - 1, int(math.ceil(max(s[4] for s in shapes))))
    by1 = min(height - 1, int(math.ceil(max(s[5] for s in shapes))))
    if bx1 < bx0 or by1 < by0:
        return None
    X, Y = np.meshgrid(np.arange(bx0, bx1 + 1, dtype=np.float64), np.arange(by0, by1 + 1, dtype=np.float64))
    union = np.zeros(X.shape, dtype=bool)
    parts = {}
    for name, geo, *_ in shapes:
        if geo[0] == "quad":
            m = _in_quad(X, Y, geo[1])
        else:
            c, r = geo[1], geo[2]
            m = (X - c[0]) ** 2 + (Y - c[1]) ** 2 <= r * r
        n = int(m.sum())
        if n >= min_px:
            union |= m
            parts[name] = n
    if not parts:
        return None
    return bx0, by0, union, parts


# ---- colour ------------------------------------------------------------------------
def hsv(rgb: np.ndarray):
    """(hue in degrees [0, 360), saturation, value) of (n, 3) uint8 RGB, as float64 arrays."""
    p = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    mx, mn = p.max(axis=1), p.min(axis=1)
    d = mx - mn
    safe = np.where(d > 0, d, 1.0)
    h = np.zeros_like(mx)
    h = np.where(mx == r, ((g - b) / safe) % 6.0, h)
    h = np.where((mx == g) & (mx != r), (b - r) / safe + 2.0, h)
    h = np.where((mx == b) & (mx != r) & (mx != g), (r - g) / safe + 4.0, h)
    h = np.where(d > 0, 60.0 * h, 0.0) % 360.0
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    return h, s, mx


def kit_masks(rgb: np.ndarray, red_sat: float = RED_SAT_MIN, blue_sat: float = BLUE_SAT_MIN,
              red_val: float = RED_VAL_MIN, blue_val: float = BLUE_VAL_MIN):
    """(red, blue) boolean arrays for (n, 3) uint8 RGB pixels."""
    h, s, v = hsv(rgb)
    red = (s >= red_sat) & (v >= red_val) & ((h >= RED_HUE_MIN) | (h < RED_HUE_MAX))
    blue = (s >= blue_sat) & (v >= blue_val) & (h >= BLUE_HUE_MIN) & (h < BLUE_HUE_MAX)
    return red, blue


def row_counts(frame: np.ndarray, xy, conf, box, min_conf: float = MIN_CONF, use=DECISION_REGIONS,
               red_sat: float = RED_SAT_MIN, blue_sat: float = BLUE_SAT_MIN):
    """(red pixels, blue pixels, region pixels) of one person-row in one frame, or None when no
    region could be read. Only the counts leave this function."""
    h, w = frame.shape[:2]
    reg = region_masks(xy, conf, box, w, h, min_conf, use=use)
    if reg is None:
        return None
    x0, y0, mask, _ = reg
    crop = frame[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    px = crop[mask]
    if len(px) == 0:
        return None
    red, blue = kit_masks(px, red_sat, blue_sat)
    return int(red.sum()), int(blue.sum()), len(px)


def is_kit_row(red: int, blue: int, n: int, row_kit_min: float = ROW_KIT_MIN, row_kit_px: int = ROW_KIT_PX) -> bool:
    """A row with enough red + blue pixels, both in absolute number and as a share of the region."""
    k = red + blue
    return n > 0 and k >= row_kit_px and k >= row_kit_min * n


def decide(index: float, kit: float, rows: int, kit_rows: int, index_min: float = INDEX_MIN,
           min_rows: int = MIN_ROWS, kit_row_frac: float = KIT_ROW_FRAC, none_rows: int = NONE_ROWS,
           none_max: float = NONE_MAX) -> str:
    """The fragment decision from its median kit index, median kit share and row counts."""
    if rows >= min_rows and kit_rows >= min_rows and kit_rows >= kit_row_frac * rows:
        if index >= index_min:
            return "blue"
        if index <= -index_min:
            return "red"
        return "unsure"
    if rows >= none_rows and kit < none_max:
        return "none"
    return "unsure"


# ---- the colours of one video ------------------------------------------------------
@dataclass(frozen=True)
class FragmentColour:
    decision: str
    index: float      # median kit index (B - R) / (B + R) over the kit rows: -1 red ... +1 blue; 0.0 without kit rows
    kit: float        # median over all rows of (R + B) / region pixels
    rows: int         # rows that had a readable region
    kit_rows: int     # of those, rows with enough red + blue pixels


@dataclass(frozen=True)
class Colours:
    """Kit colour per track id of ONE video, bound to its PoseTable (source name, frame count, video
    fingerprint when the table has one). `by_id` maps track id -> FragmentColour."""

    by_id: dict
    source: str
    n_frames: int
    video_fp: str = ""

    def decision(self, tid: int) -> str:
        return self.by_id[int(tid)].decision

    def counts(self) -> dict:
        return {d: sum(c.decision == d for c in self.by_id.values()) for d in DECISIONS}

    def check_table(self, table) -> None:
        """Refuse a table these colours were not computed from, or one whose ids differ."""
        if self.source != table.source:
            raise ValueError(f"the colours are for {self.source!r}, the pose table for {table.source!r}")
        if self.n_frames != table.n_frames:
            raise ValueError(f"the colours cover {self.n_frames} frames, the pose table {table.n_frames}")
        if self.video_fp != table.video_fp:
            raise ValueError("the colours and the pose table carry different video fingerprints")
        ids = set(int(t) for t in table.ids())
        if set(self.by_id) != ids:
            raise ValueError(f"the colours name {len(self.by_id)} track ids, the pose table has {len(ids)}: "
                             "they were computed from another table")

    def save(self, path) -> None:
        doc = {"format": FORMAT, "source": self.source, "n_frames": self.n_frames, "video_fp": self.video_fp,
               "fragments": {str(t): {"decision": c.decision, "index": c.index, "kit": c.kit, "rows": c.rows,
                                      "kit_rows": c.kit_rows}
                             for t, c in sorted(self.by_id.items())}}
        Path(path).write_text(json.dumps(doc, indent=1, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Colours":
        """Load a colours file. It is an INPUT file, so everything is checked: JSON without NaN or
        Infinity, exactly the known keys, whole-number counts, an index in [-1, 1], a kit share in
        [0, 1], a known decision that is the one the numbers imply (a file cannot claim a colour its
        own medians do not show)."""
        p = Path(path)
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{p.name}: more than {MAX_FILE_BYTES} bytes")

        def _no_const(s):
            raise ValueError(f"{p.name}: {s} is not a number")
        try:
            doc = json.loads(p.read_text(encoding="utf-8"), parse_constant=_no_const)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p.name}: not valid JSON ({e.msg})") from e
        return cls._from_doc(doc, p.name)

    @classmethod
    def _from_doc(cls, doc, name: str = "colours") -> "Colours":
        def bad(msg):
            raise ValueError(f"{name}: {msg}")

        if not isinstance(doc, dict) or set(doc) != {"format", "source", "n_frames", "video_fp", "fragments"}:
            bad("not a colours file (wrong keys)")
        if doc["format"] != FORMAT:
            bad(f"unknown format {doc['format']!r}")
        if not isinstance(doc["source"], str):
            bad("source must be text")
        nf = doc["n_frames"]
        if isinstance(nf, bool) or not isinstance(nf, int) or nf < 1:
            bad("n_frames must be a whole number >= 1")
        fp = doc["video_fp"]
        if not isinstance(fp, str) or (fp and not _FP.fullmatch(fp)):
            bad("video_fp must be empty or a 64-character lowercase sha256 hex digest")
        if not isinstance(doc["fragments"], dict):
            bad("fragments must be an object")
        out = {}
        for k, v in doc["fragments"].items():
            if not isinstance(k, str) or not k.isascii() or not k.isdigit() or (len(k) > 1 and k[0] == "0"):
                bad(f"track id {k!r} is not a whole number")
            if not isinstance(v, dict) or set(v) != {"decision", "index", "kit", "rows", "kit_rows"}:
                bad(f"fragment {k}: wrong keys")
            if v["decision"] not in DECISIONS:
                bad(f"fragment {k}: unknown decision {v['decision']!r}")
            for key, lo in (("index", -1.0), ("kit", 0.0)):
                x = v[key]
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= 1.0:
                    bad(f"fragment {k}: {key} must be a finite number in [{lo:g}, 1]")
            for key in ("rows", "kit_rows"):
                x = v[key]
                if isinstance(x, bool) or not isinstance(x, int) or x < 0:
                    bad(f"fragment {k}: {key} must be a whole number >= 0")
            if v["kit_rows"] > v["rows"]:
                bad(f"fragment {k}: kit_rows {v['kit_rows']} exceeds rows {v['rows']}")
            index, kit = float(v["index"]), float(v["kit"])
            if v["kit_rows"] == 0 and index != 0.0:
                bad(f"fragment {k}: an index without kit rows")
            if v["decision"] != decide(index, kit, v["rows"], v["kit_rows"]):
                bad(f"fragment {k}: decision {v['decision']!r} is not what its numbers imply "
                    f"({decide(index, kit, v['rows'], v['kit_rows'])!r})")
            out[int(k)] = FragmentColour(v["decision"], index, kit, v["rows"], v["kit_rows"])
        return cls(out, doc["source"], nf, fp)


# ---- extraction --------------------------------------------------------------------
def select_rows(table, every: int = SAMPLE_EVERY, min_conf: float = MIN_CONF) -> np.ndarray:
    """Row indices of the table to read: frames divisible by `every`, plus each fragment's first and
    last row with a confident torso (all four joints)."""
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError("every must be a whole number >= 1")
    pick = (table.frame % every) == 0
    torso = (table.conf[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] >= min_conf).all(axis=1)
    order = np.lexsort((table.frame, table.track_id))
    tids = table.track_id[order]
    if len(order):
        cut = np.flatnonzero(np.diff(tids)) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(order)]):
            ok = order[s:e][torso[order[s:e]]]
            if len(ok):
                pick[ok[0]] = True
                pick[ok[-1]] = True
    return np.flatnonzero(pick)


def fragment_colours(table, rows_read: dict) -> "dict":
    """FragmentColour for EVERY track id of the table from {row index: (red px, blue px, region px)}. A
    fragment with no readable row has rows = 0 and is 'unsure'."""
    by_tid: "dict[int, list]" = {int(t): [] for t in table.ids()}
    for r, rbn in rows_read.items():
        by_tid[int(table.track_id[r])].append(rbn)
    out = {}
    for tid, lst in by_tid.items():
        idxs, shares = [], []
        for red, blue, n in lst:
            shares.append((red + blue) / n)
            if is_kit_row(red, blue, n):
                idxs.append((blue - red) / (blue + red))
        index = float(np.median(idxs)) if idxs else 0.0
        kit = float(np.median(shares)) if shares else 0.0
        out[tid] = FragmentColour(decide(index, kit, len(lst), len(idxs)), index, kit, len(lst), len(idxs))
    return out


def compute_colours(video, table, info=None, every: int = SAMPLE_EVERY, min_conf: float = MIN_CONF,
                    frames=None) -> Colours:
    """Decode `video` once and read the colour of every fragment of `table`. `frames` replaces the
    decoder with any iterator of (H, W, 3) uint8 frames (tests). Only numbers are kept."""
    rows = select_rows(table, every, min_conf)
    by_frame: "dict[int, list]" = {}
    for r in rows.tolist():
        by_frame.setdefault(int(table.frame[r]), []).append(r)
    last = max(by_frame) if by_frame else -1
    read = {}
    if frames is None:
        frames = iter_frames(video, info, max_frames=last + 1) if last >= 0 else iter(())
    seen = 0
    for i, frame in enumerate(frames):
        if i > last:
            break
        seen = i + 1
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 \
                or frame.shape[:2] != (table.height, table.width):
            raise ValueError(f"frame {i} is {frame.dtype} {frame.shape}, the table is "
                             f"{table.height}x{table.width}")
        for r in by_frame.get(i, ()):
            got = row_counts(frame, table.xy[r], table.conf[r], table.box[r], min_conf)
            if got is not None:
                read[r] = got
    if seen < last + 1:
        raise ValueError(f"the video ended after {seen} frames, the pose table has rows up to frame {last}: "
                         "it was extracted from a different video")
    return Colours(fragment_colours(table, read), table.source, table.n_frames, table.video_fp)


# ---- the non-circular self-check ----------------------------------------------------
CO_PRESENT_S = 0.5      # two fragments on screen together at least this long are different people


def colour_check(table, attrs, colours: Colours) -> dict:
    """Do two boxers on screen together get OPPOSITE colours? The labels never say red or blue, so
    this cannot be passed by construction.

    Fragments credited with a labelled punch are paired when they are on screen at the same time for
    at least CO_PRESENT_S. Two people on screen together are different people (a tracker that
    doubles one person's id is the rare exception), so a working colour test gives them opposite
    colours; chance would be about half among decided pairs. Counts only; `opposite_share` and
    `same_share` are None when no pair is decided (no measurement, not 0%)."""
    colours.check_table(table)
    credited = sorted({int(a.track_id) for a in attrs if a.track_id is not None})
    frames = {t: np.unique(table.frame[table.track_id == t]) for t in credited}
    need = max(1, int(math.ceil(CO_PRESENT_S * table.fps - 1e-9)))
    pairs = opposite = same = 0
    for i, a in enumerate(credited):
        for b in credited[i + 1:]:
            if len(np.intersect1d(frames[a], frames[b], assume_unique=True)) < need:
                continue
            pairs += 1
            da, db = colours.decision(a), colours.decision(b)
            if da in ("red", "blue") and db in ("red", "blue"):
                if da == db:
                    same += 1
                else:
                    opposite += 1
    decided = opposite + same
    by = {d: sum(colours.decision(t) == d for t in credited) for d in DECISIONS}
    return {"credited_fragments": len(credited), "credited_by_decision": by,
            "copresent_pairs": pairs, "decided_pairs": decided, "opposite": opposite, "same": same,
            "opposite_share": (opposite / decided) if decided else None,
            "same_share": (same / decided) if decided else None}


def pool_checks(rows) -> dict:
    """Pool per-video `colour_check` dicts from their integer counts, divided once."""
    rows = list(rows)
    out = {"videos": len(rows)}
    for k in ("credited_fragments", "copresent_pairs", "decided_pairs", "opposite", "same"):
        out[k] = sum(r[k] for r in rows)
    out["credited_by_decision"] = {d: sum(r["credited_by_decision"][d] for r in rows) for d in DECISIONS}
    out["opposite_share"] = (out["opposite"] / out["decided_pairs"]) if out["decided_pairs"] else None
    out["same_share"] = (out["same"] / out["decided_pairs"]) if out["decided_pairs"] else None
    return out
