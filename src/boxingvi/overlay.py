"""Skeleton overlay renderer -- the thing you review pose quality with.

Pure numpy drawing (no OpenCV): a skeleton drawn on your own footage is the
fastest honest check that the pose model sees what you think it sees, so the
renderer is kept small enough to trust by reading it. Frames with no usable
pose (none found, or too few confident joints) get a red border instead of silently showing bare video, and every frame
carries its index so a bad stretch can be quoted to the pose code by number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import EDGES, N_JOINTS
from .pose_yolo import PoseSequence
from .video import VideoWriter, iter_frames, probe

LEFT = {1, 3, 5, 7, 9, 11, 13, 15}   # COCO-17 odd indices
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
C_LEFT, C_RIGHT, C_MID = (60, 200, 255), (255, 170, 40), (255, 255, 255)
C_MISS = (230, 40, 40)

_DIGITS = {  # 3x5 bitmap font, row-major
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111",
}


def _side(i: int) -> tuple:
    return C_LEFT if i in LEFT else C_RIGHT if i in RIGHT else C_MID


def _disk(r: int) -> np.ndarray:
    g = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(g, g)
    keep = dx * dx + dy * dy <= r * r
    return np.stack([dx[keep], dy[keep]], axis=1)


def draw_line(frame: np.ndarray, p0, p1, color, r: int = 2) -> None:
    h, w = frame.shape[:2]
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    # A wild keypoint (far outside the frame) must not allocate a huge line.
    p0, p1 = np.clip(p0, -4 * max(h, w), 5 * max(h, w)), np.clip(p1, -4 * max(h, w), 5 * max(h, w))
    n = int(max(abs(p1 - p0))) + 1
    pts = np.rint(np.linspace(p0, p1, n)).astype(int)
    off = _disk(r)
    xy = (pts[:, None, :] + off[None, :, :]).reshape(-1, 2)
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    frame[xy[ok, 1], xy[ok, 0]] = color


def draw_number(frame: np.ndarray, n: int, x: int = 6, y: int = 6, scale: int = 3) -> None:
    for k, ch in enumerate(str(max(int(n), 0))):
        bits = np.array([int(b) for b in _DIGITS[ch]]).reshape(5, 3)
        big = np.kron(bits, np.ones((scale, scale), int)).astype(bool)
        y1, x1 = y + big.shape[0], x + k * 4 * scale
        sub = frame[y:y1, x1:x1 + big.shape[1]]
        if sub.shape[:2] == big.shape:
            sub[big] = (255, 255, 0)


def draw_skeleton(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray,
                  min_conf: float = 0.3, r: int = 2) -> None:
    """Draw in place. An edge needs both ends above `min_conf`; a bone with a
    guessed end would assert something the model did not."""
    ok = (conf >= min_conf) & np.all(np.isfinite(xy), axis=1)
    for a, b in EDGES:
        if ok[a] and ok[b]:
            same_side = (a in LEFT and b in LEFT) or (a in RIGHT and b in RIGHT)
            draw_line(frame, xy[a], xy[b], _side(a) if same_side else C_MID, r)
    for j in range(N_JOINTS):
        if ok[j]:
            draw_line(frame, xy[j], xy[j], _side(j), r + 1)


def _border(frame: np.ndarray, color, t: int = 6) -> None:
    frame[:t], frame[-t:], frame[:, :t], frame[:, -t:] = color, color, color, color


def render_overlay(video, pose: PoseSequence, out, min_conf: float = 0.3,
                   max_frames: "int | None" = None) -> int:
    """Write `out` with the skeleton drawn on `video`; return frames written."""
    video, out = Path(video).resolve(), Path(out).resolve()
    if out == video:
        raise ValueError("refusing to overwrite the input video with its own overlay")
    info = probe(video)
    if (info.width, info.height) != (pose.width, pose.height):
        raise ValueError(f"pose is for {pose.width}x{pose.height}, video is "
                         f"{info.width}x{info.height}; not drawing a misaligned skeleton")
    good = pose.usable(min_conf)
    n = 0
    with VideoWriter(out, info.width, info.height, info.fps_frac) as vw:
        for i, frame in enumerate(iter_frames(video, info, max_frames)):
            if i >= len(pose):
                raise ValueError(f"video has more frames than the pose ({len(pose)})")
            if good[i]:
                draw_skeleton(frame, pose.xy[i], pose.conf[i], min_conf)
            else:  # undetected OR too few solid joints: never show bare video unmarked
                _border(frame, C_MISS)
            draw_number(frame, i)
            vw.write(frame)
            n += 1
    return n
