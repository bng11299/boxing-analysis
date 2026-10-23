"""Video in, video out, through ffmpeg -- no OpenCV, matching `align.py`.

Every subprocess takes an argument LIST (never a shell string), frames are
decoded with passthrough timing so the i-th frame is the i-th frame the file
holds, and sizes are capped before anything is allocated: the file is untrusted
input, and a header claiming 60000x60000 must not become a 10 GB buffer.

Rotation is handled here, not by the caller: phone footage is routinely stored
landscape with a rotation flag. ffmpeg auto-rotates on decode, so the frames we
read are the displayed orientation, and `probe` must report the displayed size
or every frame-size read below is misaligned (a silent, plausible-looking skew).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

MAX_PIXELS = 3840 * 2160  # one 4K frame; anything bigger is refused, not resized
PROBE_TIMEOUT_S = 30
TIMING_TIMEOUT_S = 300   # a packet scan reads no pixels: 46,705 packets take ~2 s
# ~9 h at 60 fps; beyond this the scan is refused, not parsed. This bounds the parse,
# not the capture: ffprobe's output is buffered whole first, so a crafted file with
# tens of millions of packets costs memory until the timeout. Accepted -- the input
# is our own footage, and such a file would itself be enormous.
MAX_TIMING_PACKETS = 2_000_000
MAX_RATE_MISMATCH = 0.001  # measured mean rate vs the container's average rate
IRREGULAR_DT = 0.5       # an interval this many median intervals off is a drop or a duplicate
MAX_DRIFT = 1.0          # ... and no frame may sit more than one interval off constant-rate time


@dataclass(frozen=True)
class Timing:
    """Measured frame timing, from the packet timestamps of the probed stream.

    Two checks, because each misses what the other catches. Per interval: a
    dropped frame (2x) or a duplicate (0x) is far outside IRREGULAR_DT, while
    rounding jitter (V6 alternates 41 and 42 ms) is far inside it. Cumulative: a
    rate change (a phone dropping 30 -> 24 fps in low light) moves each interval
    only 0.25x, but frames drift seconds away from i / fps, which is what every
    per-second feature assumes. Drift is measured against the clip's own MEAN
    rate (the line through its first and last frame), not the median: V6's median
    is 42 ms against a true 41.7 ms, and i x median drifts ~325 frames over V6.

    Not detectable here: a slow-motion export already retimed onto a regular
    timeline (its timestamps are regular; only the motion is slowed).
    """
    n_frames: int       # frames the decoder will output (packets flagged discard excluded)
    median_dt: float    # seconds
    n_irregular: int    # intervals more than IRREGULAR_DT median intervals off the median
    max_gap: float      # longest interval, in median intervals (2.0 = one frame missing)
    min_gap: float      # shortest, likewise (0.0 = a duplicated timestamp)
    max_drift: float    # worst |t_i - constant-rate t_i|, in median intervals
    mean_fps: float     # (n - 1) / (last pts - first pts): the rate drift is measured at

    @property
    def regular(self) -> bool:
        return self.n_irregular == 0 and self.max_drift <= MAX_DRIFT


@dataclass(frozen=True)
class VideoInfo:
    width: int          # displayed size, after rotation
    height: int
    fps_frac: Fraction  # average rate, kept exact so 59.94 stays 60000/1001
    nb_frames: "int | None"  # container's claim; None when absent. A hint only.
    rotation: int       # degrees, as stored; already applied by the decoder
    vfr: bool           # frame i is NOT at time i / fps: irregular intervals, or unmeasurable
    timing: "Timing | None" = None  # None when the timestamps could not be read

    @property
    def fps(self) -> float:
        return float(self.fps_frac)


FINGERPRINT_EDGE = 4 * 1024 * 1024   # bytes read from each end of the file


def fingerprint(path) -> str:
    """Content identity of a video file: sha256 hex of (file size as 8 bytes big-endian,
    the first 4 MiB, the last 4 MiB). Cheap enough for a 2 GB recording (8 MiB read), and
    a different recording differs in size or in the first/last 4 MiB (container header and
    the final packets) in practice; it is NOT a hash of every byte, so it detects the wrong
    file, not a deliberately edited middle. A file under 8 MiB is hashed from both ends,
    which then overlap: still deterministic. Pose tables store it (`PoseTable.video_fp`) so
    a table renamed onto another recording of the same size, rate and length is refused."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(size.to_bytes(8, "big"))
    with p.open("rb") as f:
        h.update(f.read(FINGERPRINT_EDGE))
        f.seek(max(0, size - FINGERPRINT_EDGE))
        h.update(f.read(FINGERPRINT_EDGE))
    return h.hexdigest()


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"{name} is not on PATH")
    return exe


def _rate(text: str) -> Fraction:
    try:
        num, _, den = str(text).partition("/")
        r = Fraction(int(num), int(den or 1))
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"unusable frame rate {text!r}") from e
    if r <= 0:
        raise ValueError(f"frame rate must be positive, got {text!r}")
    return r


def probe(path) -> VideoInfo:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no video at {path}")
    cmd = [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames"
           ":stream_tags=rotate:stream_side_data=rotation",
           "-of", "json", str(path)]
    done = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT_S)
    if done.returncode != 0:
        raise ValueError(f"ffprobe could not read {path.name}")
    try:
        streams = json.loads(done.stdout)["streams"]
        s = streams[0]
        w, h = int(s["width"]), int(s["height"])
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise ValueError(f"{path.name} has no readable video stream") from e
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if not rot:
        rot = int(round(float((s.get("tags") or {}).get("rotate", 0) or 0)))
    rot %= 360
    if rot in (90, 270):
        w, h = h, w
    if w <= 0 or h <= 0 or w * h > MAX_PIXELS:
        raise ValueError(f"{path.name}: frame size {w}x{h} is outside 1..{MAX_PIXELS} pixels")
    avg = _rate(s.get("avg_frame_rate", "0/0")).limit_denominator(1001)  # 23.976..., not 23.97603269
    nb = s.get("nb_frames")
    nb = int(nb) if str(nb).isdigit() else None
    # VFR is MEASURED, not inferred from metadata. The old test (average vs nominal
    # rate differ by > 0.5 fps) passed the first phone clip: nominal 60, average
    # 59.946. Every timing feature assumes frame i sits at i / fps, so anything we
    # cannot confirm counts as variable (deny by default).
    timing = frame_timing(path)
    # Drift is measured against the clip's measured mean rate, but every consumer
    # times frames with `avg` (the container's figure), so the two must agree too.
    off_rate = timing is not None and abs(timing.mean_fps / float(avg) - 1) > MAX_RATE_MISMATCH
    return VideoInfo(w, h, avg, nb, rot, timing is None or not timing.regular or off_rate, timing)


def frame_timing(path) -> "Timing | None":
    """Timing statistics of stream v:0, from packet timestamps (no decoding).

    Returns None when the timestamps are missing or unreadable. Packets arrive in
    decode order, so they are sorted by presentation time first. Packets flagged
    D (discard) are dropped: a stream-copy cut at a non-keyframe keeps the frames
    before the cut, with negative pts, for decoding only -- one 600-packet cut
    decoded to 510 frames.
    """
    path = Path(path).resolve()
    cmd = [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=TIMING_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    if done.returncode != 0:
        return None
    try:
        lines = done.stdout.decode("ascii").split()
        if len(lines) > MAX_TIMING_PACKETS:
            return None
        rows = [ln.split(",") for ln in lines]
        pts = np.sort(np.array([float(r[0]) for r in rows if len(r) < 2 or "D" not in r[1]]))
    except (ValueError, UnicodeDecodeError):  # "N/A", or not ffprobe's output at all
        return None
    if len(pts) < 3 or not np.all(np.isfinite(pts)):
        return None
    dt = np.diff(pts)
    med = float(np.median(dt))
    if not med > 0:
        return None
    off = np.abs(dt - med) / med
    line = np.linspace(pts[0], pts[-1], len(pts))  # constant rate at the clip's mean
    return Timing(len(pts), med, int((off > IRREGULAR_DT).sum()),
                  float(dt.max() / med), float(dt.min() / med),
                  float(np.abs(pts - line).max() / med),
                  float((len(pts) - 1) / (pts[-1] - pts[0])))


def iter_frames(path, info: "VideoInfo | None" = None, max_frames: "int | None" = None):
    """Yield (H, W, 3) uint8 RGB frames, in file order, none duplicated or dropped."""
    path = Path(path).resolve()
    info = info or probe(path)
    w, h = info.width, info.height
    frame_bytes = w * h * 3
    # -map 0:v:0 decodes the stream `probe` measured. Without it ffmpeg picks its
    # own "best" stream, and a file whose default stream is a different size
    # yields chopped-up frames of the wrong stream with no error.
    cmd = [_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n, at_eof, tail = 0, False, b""
    # stderr is discarded rather than piped: an unread pipe fills on a corrupt
    # file and deadlocks ffmpeg while we wait on stdout (same reasoning as align.py).
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        try:
            while max_frames is None or n < max_frames:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    at_eof, tail = True, buf
                    break
                n += 1
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        finally:
            # Kill only when WE stopped (max_frames, or the caller closed us). At EOF
            # ffmpeg is still shutting down, and killing it there discards its exit code.
            if not at_eof:
                proc.kill()
            try:
                proc.wait(timeout=PROBE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError(f"ffmpeg did not exit after the last frame of {path.name}")
    if at_eof and proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name} (exit {proc.returncode})")
    if tail:  # a partial frame: decoded size is not the probed size, so every frame is suspect
        raise RuntimeError(f"{path.name}: decoder output is not a whole number of "
                           f"{w}x{h} frames ({len(tail)} stray bytes after frame {n})")


_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def frame_change(path, info: "VideoInfo | None" = None, step: int = 4) -> np.ndarray:
    """One number per frame: the mean absolute difference between the grey image of frame f and of
    frame f-1 (NaN for frame 0), on every `step`-th pixel in each direction. The video is decoded
    once, by `iter_frames`, so entry f is the f-th frame the file holds.

    Why: a camera that drops captured frames still writes regular timestamps (the packet timing
    `probe` checks is clean), but what is on screen jumps forward in one frame. That jump is the
    only trace, and it is this number that spikes (see `skips.find_skips`). It is ONE aggregate
    number per frame: no pixel, crop or thumbnail leaves this function."""
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a whole number >= 1")
    out: "list[float]" = []
    prev = None
    for frame in iter_frames(path, info):
        grey = frame[::step, ::step].astype(np.float32) @ _LUMA
        out.append(float("nan") if prev is None else float(np.abs(grey - prev).mean()))
        prev = grey
    return np.array(out, dtype=np.float64)


class VideoWriter:
    """H.264 mp4 writer fed raw RGB frames. Use as a context manager.

    The input is a raw pipe, so nothing of the source file (audio, tags, GPS,
    creation time) can reach the output. `private=True` also strips what the muxer
    adds on its own (the Lavf encoder tag, creation time) and maps no non-video
    stream. What remains is x264's own settings string in the bitstream ("x264 -
    core ... options: ..."): encoder settings, no data about the footage."""

    PRIVATE_ARGS = ("-map_metadata", "-1", "-map_chapters", "-1", "-an", "-sn", "-dn",
                    "-fflags", "+bitexact", "-flags:v", "+bitexact")

    def __init__(self, path, width: int, height: int, fps: Fraction, private: bool = False):
        self.path = Path(path).resolve()
        if width * height > MAX_PIXELS or width <= 0 or height <= 0:
            raise ValueError(f"bad frame size {width}x{height}")
        self.size = (width, height)
        cmd = [_tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}",
               "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               *(self.PRIVATE_ARGS if private else ()), str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> None:
        w, h = self.size
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 {(h, w, 3)}, got {frame.dtype} {frame.shape}")
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path.name} (exit {rc})")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is not None:  # don't leave a truncated file masquerading as output
            self._proc.kill()
            self._proc.wait()
            self.path.unlink(missing_ok=True)
        else:
            self.close()
