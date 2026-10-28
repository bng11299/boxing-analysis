"""The ring floor: which tracked people stand INSIDE the ring, from a polygon clicked per video.

Why the floor and the feet, not a pixel mask (development videos only; docs/gates/ring-floor-filter.md):
the Olympic cameras sit at a ring corner at about head height, so the crowd behind the far ropes occupies
the same pixels as a boxer standing at those ropes. On the two kam4 development videos, NO region that
keeps every credited boxer could black out a single other person (0.0% of non-credited person-frames
lie mostly outside even the union of credited boxers' boxes). Depth separates them, and in one image
the cue to depth is where a person's feet are: a spectator stands behind the ring, a boxer on its floor.

What this module does, and nothing more:
  * `load_floors` / `check_floor`: the polygon file is a trust boundary (clicked by a person, but
    read back from disk), so every field is checked and anything unexpected is refused.
  * `signed_distance`, `feet`, `ids_outside`, `without_ids`: a track id is dropped when fewer than
    half its rows have the feet point (bottom centre of the box) inside the floor grown by `margin_px`.
    Whole ids only: a kept id is never cut into pieces, so no new gaps appear in anyone's track.
  * `sample_frames`, `background`, `phase_shift`, `camera_motion`: is the camera still during a
    video (one polygon is then valid for all of it)? Label-free: pixels only, no pose, no labels.
    The background image exists for the click tool; it stays in the gitignored results/ folder.

The referee stands on the floor and is never removed by this rule.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boxingvi.boxers import replace_bytes
from boxingvi.pose_yolo import MAX_POSE_TEXT, PoseTable
from boxingvi.video import MAX_PIXELS, PROBE_TIMEOUT_S, VideoInfo, _tool, probe

FORMAT = "boxing-comps ring floor v1"
MAX_FILE_BYTES = 1 << 20          # a 29-video file is about 10 KB
MAX_VIDEOS = 64
MIN_POINTS, MAX_POINTS = 3, 32
MIN_AREA_SHARE = 0.01             # a floor under 1% of the frame is a slip of the mouse, not a ring
MAX_SIDE = 16384
MAX_CLICK_S = 24 * 3600
MIN_INSIDE_SHARE = 0.5            # an id is kept when at least half its rows stand inside
VIDEO_ID = re.compile(r"[A-Za-z0-9]{1,32}")
FINGERPRINT = re.compile(r"[0-9a-f]{64}")
ENTRY_KEYS = frozenset({"points", "width", "height", "video_fp", "click_s"})

# Camera stillness. Pixels only, never labels.
STILL_SAMPLES = 24                # frames sampled evenly across the video
STILL_QUANTILE = 90               # one bad frame (a body filling the view) must not condemn a video
STILL_REF_SAMPLES = 3             # the camera's starting position: the median of the first 3 samples
SHIFT_SCALE = 2                   # phase correlation on a half-size grey image
SAMPLE_TIMEOUT_S = 120


@dataclass(frozen=True)
class Floor:
    video: str
    points: np.ndarray            # (k, 2) pixels, read-only
    width: int
    height: int
    video_fp: str
    click_s: "float | None"       # seconds the person took to click it; None when not timed

    def to_json(self) -> dict:
        return {"points": [[float(x), float(y)] for x, y in self.points], "width": self.width,
                "height": self.height, "video_fp": self.video_fp, "click_s": self.click_s}


# ---------------------------------------------------------------- polygon checks

def _number(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} is not a number")
    try:
        f = float(v)
    except OverflowError:                  # a JSON integer like 10**400 (Astra, review 1)
        raise ValueError(f"{what} is not finite") from None
    if not math.isfinite(f):
        raise ValueError(f"{what} is not finite")
    return f


def _whole(v, what: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"{what} must be a whole number in [{lo}, {hi}]")
    return v


def _orient(a, b, c) -> int:
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0 if v == 0 else (1 if v > 0 else -1)


def _on_segment(a, b, c) -> bool:
    """c lies on segment ab, given the three are collinear."""
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def _segments_meet(p1, p2, q1, q2) -> bool:
    o1, o2, o3, o4 = _orient(p1, p2, q1), _orient(p1, p2, q2), _orient(q1, q2, p1), _orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(p1, p2, q1)) or (o2 == 0 and _on_segment(p1, p2, q2))
            or (o3 == 0 and _on_segment(q1, q2, p1)) or (o4 == 0 and _on_segment(q1, q2, p2)))


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def check_floor(video: str, points, width, height, video_fp, click_s=None) -> Floor:
    """A validated Floor, or ValueError naming what is wrong. Used by the reader AND by the click
    tool before it saves, so a polygon that would be refused later is never written."""
    if not isinstance(video, str) or not VIDEO_ID.fullmatch(video):
        raise ValueError(f"{video!r} is not a GoPro id (letters and digits only)")
    width = _whole(width, f"{video}: width", 1, MAX_SIDE)
    height = _whole(height, f"{video}: height", 1, MAX_SIDE)
    if width * height > MAX_PIXELS:
        raise ValueError(f"{video}: frame {width}x{height} is larger than any video this project reads")
    if not isinstance(video_fp, str) or not FINGERPRINT.fullmatch(video_fp):
        raise ValueError(f"{video}: video_fp must be a 64-character lowercase sha256 hex")
    if click_s is not None:
        click_s = _number(click_s, f"{video}: click_s")
        if not 0 <= click_s <= MAX_CLICK_S:
            raise ValueError(f"{video}: click_s must lie in [0, {MAX_CLICK_S}]")
    if not isinstance(points, (list, tuple)) or not MIN_POINTS <= len(points) <= MAX_POINTS:
        raise ValueError(f"{video}: a floor needs {MIN_POINTS} to {MAX_POINTS} points")
    pts = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"{video}: point {i} is not an [x, y] pair")
        x, y = _number(p[0], f"{video}: point {i} x"), _number(p[1], f"{video}: point {i} y")
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"{video}: point {i} ({x:g}, {y:g}) lies outside the {width}x{height} frame")
        pts.append((x, y))
    k = len(pts)
    for i in range(k):
        if pts[i] == pts[(i + 1) % k]:
            raise ValueError(f"{video}: points {i} and {(i + 1) % k} are the same point")
    for i in range(k):
        for j in range(i + 1, k):
            if j == i + 1 or (i == 0 and j == k - 1):
                continue           # neighbouring edges share a corner by construction
            if _segments_meet(pts[i], pts[(i + 1) % k], pts[j], pts[(j + 1) % k]):
                raise ValueError(f"{video}: the outline crosses itself (edges {i} and {j})")
    arr = np.asarray(pts, np.float64)
    if polygon_area(arr) < MIN_AREA_SHARE * width * height:
        raise ValueError(f"{video}: the floor covers under {MIN_AREA_SHARE:.0%} of the frame")
    arr.setflags(write=False)
    return Floor(video, arr, width, height, video_fp, click_s)


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"key {k!r} appears twice")
        out[k] = v
    return out


def _refuse_constant(name):
    raise ValueError(f"{name} is not a number this file may hold")


def load_floors(path) -> "dict[str, Floor]":
    """Read and check a floor file. OSError if it cannot be opened; ValueError for anything wrong
    inside it (size, encoding, JSON, duplicate keys, NaN/Infinity, unknown or missing fields, a
    bad polygon). Nothing is guessed or repaired."""
    p = Path(path)
    with open(p, "rb") as fh:
        raw = fh.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"{p.name}: larger than {MAX_FILE_BYTES} bytes")
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
                         parse_constant=_refuse_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as e:
        raise ValueError(f"{p.name}: not readable UTF-8 JSON ({type(e).__name__})") from None
    except ValueError as e:
        raise ValueError(f"{p.name}: {e}") from None
    if not isinstance(doc, dict) or set(doc) != {"format", "videos"}:
        raise ValueError(f"{p.name}: expected exactly the keys 'format' and 'videos'")
    if doc["format"] != FORMAT:
        raise ValueError(f"{p.name}: format is not {FORMAT!r}")
    videos = doc["videos"]
    if not isinstance(videos, dict) or len(videos) > MAX_VIDEOS:
        raise ValueError(f"{p.name}: 'videos' must be an object of at most {MAX_VIDEOS} entries")
    out = {}
    for vid, e in videos.items():
        if not isinstance(e, dict) or set(e) != ENTRY_KEYS:
            raise ValueError(f"{p.name}: {vid!r} must have exactly the keys {sorted(ENTRY_KEYS)}")
        try:
            out[vid] = check_floor(vid, e["points"], e["width"], e["height"], e["video_fp"], e["click_s"])
        except ValueError as err:
            raise ValueError(f"{p.name}: {err}") from None
    return out


def dump_floors(floors: "dict[str, Floor]", path) -> None:
    """Write the file atomically, never through a link planted at `path`; checks every entry first."""
    for vid, f in floors.items():
        if vid != f.video:
            raise ValueError(f"entry {vid!r} holds the floor of {f.video!r}")
        check_floor(f.video, f.points.tolist(), f.width, f.height, f.video_fp, f.click_s)
    doc = {"format": FORMAT, "videos": {v: floors[v].to_json() for v in sorted(floors)}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    replace_bytes(path, json.dumps(doc, indent=1, allow_nan=False).encode("utf-8"))


# ---------------------------------------------------------------- the filter

def signed_distance(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance in pixels from each point to the polygon's outline: positive inside, negative
    outside, 0 on the outline (even-odd rule)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    poly = np.asarray(poly, np.float64)
    if len(xy) == 0:
        return np.zeros(0)
    a, b = poly, np.roll(poly, -1, axis=0)                     # edges a -> b
    px, py = xy[:, :1], xy[:, 1:]
    ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    dist = np.hypot(px - (ax + t * dx), py - (ay + t * dy)).min(axis=1)
    crosses = ((ay > py) != (by > py)) & (px < ax + (py - ay) * dx / np.where(dy == 0, np.inf, dy))
    inside = crosses.sum(axis=1) % 2 == 1
    return np.where(inside, dist, -dist)


def feet(table: PoseTable) -> np.ndarray:
    """(N, 2): bottom centre of each row's box, held inside the frame."""
    b = table.box.astype(np.float64)
    x = np.clip((b[:, 0] + b[:, 2]) / 2, 0, table.width)
    y = np.clip(b[:, 3], 0, table.height)
    return np.stack([x, y], axis=1)


def _check_pair(table: PoseTable, floor: Floor) -> None:
    if (table.width, table.height) != (floor.width, floor.height):
        raise ValueError(f"{floor.video}: floor drawn on {floor.width}x{floor.height}, "
                         f"table is {table.width}x{table.height}")
    if table.video_fp and table.video_fp != floor.video_fp:
        raise ValueError(f"{floor.video}: the floor was clicked on a different recording than the table's")


def ids_outside(table: PoseTable, floor: Floor, margin_px: float,
                min_inside_share: float = MIN_INSIDE_SHARE) -> "frozenset[int]":
    """Track ids with fewer than `min_inside_share` of their rows standing on the floor grown by
    `margin_px` (negative shrinks it). An id with exactly that share is kept."""
    margin_px = _number(margin_px, "margin_px")
    if not 0 < min_inside_share <= 1:
        raise ValueError("min_inside_share must lie in (0, 1]")
    _check_pair(table, floor)
    if len(table.frame) == 0:
        return frozenset()
    inside = signed_distance(feet(table), floor.points) >= -margin_px
    ids, inv = np.unique(table.track_id, return_inverse=True)
    share = np.bincount(inv, weights=inside) / np.bincount(inv)
    return frozenset(int(i) for i in ids[share < min_inside_share])


def without_ids(table: PoseTable, drop, note: str = "") -> PoseTable:
    """A copy of the table without the given ids; everything else, the fingerprint included, unchanged.
    `note` is appended to `model` so a written table says it was filtered."""
    model = table.model + (f" {note}" if note else "")
    if len(model) > MAX_POSE_TEXT:      # the strict loader would refuse the written table (Astra, review 2)
        raise ValueError(f"model string would exceed {MAX_POSE_TEXT} characters with the filter's note")
    keep = ~np.isin(table.track_id, np.fromiter(drop, np.int64, len(drop)))
    return PoseTable(table.frame[keep], table.track_id[keep], table.box[keep], table.xy[keep], table.conf[keep],
                     table.n_frames, table.fps, table.width, table.height, source=table.source,
                     model=model, video_fp=table.video_fp)


# ---------------------------------------------------------------- camera stillness (pixels, never labels)

def sample_frames(video, info: "VideoInfo | None" = None, n: int = STILL_SAMPLES) -> "list[np.ndarray]":
    """`n` RGB frames at evenly spaced times (k + 0.5) / n of the duration, one ffmpeg seek each."""
    video = Path(video).resolve()
    info = info or probe(video)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1000:
        raise ValueError("n must be a whole number in [1, 1000]")
    if info.width * info.height > MAX_PIXELS:
        raise ValueError(f"{video.name}: frame too large")
    if info.timing is None:
        raise ValueError(f"{video.name}: no readable timestamps, so no duration to sample")
    dur = info.timing.n_frames / info.fps
    size = info.width * info.height * 3
    out = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        cmd = [_tool("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-map", "0:v:0",
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        done = subprocess.run(cmd, capture_output=True, timeout=SAMPLE_TIMEOUT_S)
        if done.returncode != 0 or len(done.stdout) != size:
            raise RuntimeError(f"{video.name}: could not decode a frame at {t:.1f} s")
        out.append(np.frombuffer(done.stdout, np.uint8).reshape(info.height, info.width, 3).copy())
    return out


def background(frames: "list[np.ndarray]") -> np.ndarray:
    """Per-pixel median: people who move are gone, the ring and the static crowd stay."""
    if not frames:
        raise ValueError("no frames")
    return np.median(np.stack(frames), axis=0).round().astype(np.uint8)


def _grey_small(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    s = SHIFT_SCALE
    h, w = (g.shape[0] // s) * s, (g.shape[1] // s) * s
    return g[:h, :w].reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def phase_shift(a: np.ndarray, b: np.ndarray) -> "tuple[float, float]":
    """Whole-pixel translation (dx, dy) between two same-size grey images, by phase correlation with
    a Hann window. Its sign is not used: only the size of the move is."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("two grey images of one size")
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-12)
    corr = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = iy - a.shape[0] if iy > a.shape[0] // 2 else iy
    dx = ix - a.shape[1] if ix > a.shape[1] // 2 else ix
    return float(dx), float(dy)


def camera_motion(frames: "list[np.ndarray]") -> dict:
    """How far each sampled frame sits from where the camera STARTED (the median of the first
    STILL_REF_SAMPLES samples), in full-size pixels: the STILL_QUANTILE-th percentile (the number the
    stillness rule reads) and the maximum (reported).

    Not against the whole-video median: a camera that pans partway through blends both positions into
    that median, and every frame then correlates best at zero shift (GH208416 pans ~145 px at 62% of
    the video and read 0 px that way; OpenCV's phaseCorrelate agreed, with a peak response of 0.08)."""
    if len(frames) < STILL_REF_SAMPLES:
        raise ValueError(f"need at least {STILL_REF_SAMPLES} frames")
    ref = _grey_small(background(frames[:STILL_REF_SAMPLES]))
    shifts = [math.hypot(*phase_shift(ref, _grey_small(f))) * SHIFT_SCALE for f in frames]
    return {"motion_px": float(np.percentile(shifts, STILL_QUANTILE)), "max_px": float(max(shifts)),
            "samples": len(shifts)}
