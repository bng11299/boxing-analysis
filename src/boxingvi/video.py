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

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

MAX_PIXELS = 3840 * 2160  # one 4K frame; anything bigger is refused, not resized
PROBE_TIMEOUT_S = 30


@dataclass(frozen=True)
class VideoInfo:
    width: int          # displayed size, after rotation
    height: int
    fps_frac: Fraction  # average rate, kept exact so 59.94 stays 60000/1001
    nb_frames: "int | None"  # container's claim; None when absent. A hint only.
    rotation: int       # degrees, as stored; already applied by the decoder
    vfr: bool           # average and nominal rates disagree

    @property
    def fps(self) -> float:
        return float(self.fps_frac)


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
    nominal = _rate(s.get("r_frame_rate", s.get("avg_frame_rate", "0/0")))
    nb = s.get("nb_frames")
    nb = int(nb) if str(nb).isdigit() else None
    return VideoInfo(w, h, avg, nb, rot, abs(float(avg - nominal)) > 0.5)


def iter_frames(path, info: "VideoInfo | None" = None, max_frames: "int | None" = None):
    """Yield (H, W, 3) uint8 RGB frames, in file order, none duplicated or dropped."""
    path = Path(path).resolve()
    info = info or probe(path)
    w, h = info.width, info.height
    cmd = [_tool("ffmpeg"), "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    n, stopped_early = 0, False
    # stderr is discarded rather than piped: an unread pipe fills on a corrupt
    # file and deadlocks ffmpeg while we wait on stdout (same reasoning as align.py).
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        try:
            while max_frames is None or n < max_frames:
                buf = proc.stdout.read(w * h * 3)
                if len(buf) < w * h * 3:
                    break
                n += 1
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            else:
                stopped_early = True
        finally:
            if stopped_early or proc.poll() is None:
                proc.kill()
                stopped_early = True
            proc.wait()
    if not stopped_early and proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name} (exit {proc.returncode})")


class VideoWriter:
    """H.264 mp4 writer fed raw RGB frames. Use as a context manager."""

    def __init__(self, path, width: int, height: int, fps: Fraction):
        self.path = Path(path).resolve()
        if width * height > MAX_PIXELS or width <= 0 or height <= 0:
            raise ValueError(f"bad frame size {width}x{height}")
        self.size = (width, height)
        cmd = [_tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}",
               "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(self.path)]
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
