"""Skeleton overlay renderer -- the thing you review pose quality with.

Pure numpy drawing (no OpenCV): a skeleton drawn on your own footage is the
fastest honest check that the pose model sees what you think it sees, so the
renderer is kept small enough to trust by reading it. Frames with no usable
pose (none found, too few confident joints, or a wrist or shoulder unsure) get a
red border, with whatever WAS found still drawn so the reason is visible, and
every frame carries its index so a bad stretch can be quoted by number.

`render_overlay` takes an optional `annotate(frame, i)` hook, called after the
skeleton and before the frame number, so a caller can add to each frame (punch
labels, render_punches.py) while every alignment check here still applies.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from .data import EDGES, N_JOINTS
from .pose_yolo import PoseSequence
from .video import VideoInfo, VideoWriter, iter_frames, probe

LEFT = {1, 3, 5, 7, 9, 11, 13, 15}   # COCO-17 odd indices
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
C_LEFT, C_RIGHT, C_MID = (60, 200, 255), (255, 170, 40), (255, 255, 255)
C_MISS = (230, 40, 40)
NUMBER_X, NUMBER_Y, NUMBER_SCALE = 6, 6, 3   # where draw_number puts the frame index

_DIGITS = {  # 3x5 bitmap font, row-major
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111",
}


# 5x7 capitals for labels, row-major, one string of 5 bits per row. Only what the
# labels need: a character outside this set is refused, not skipped, so a label
# can never silently lose a letter.
_GLYPHS = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110", "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 10001 11001 10101 10011 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 10001 01010 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    " ": "00000 00000 00000 00000 00000 00000 00000", "(": "00010 00100 01000 01000 01000 00100 00010",
    ")": "01000 00100 00010 00010 00010 00100 01000", ".": "00000 00000 00000 00000 00000 01100 01100",
    "'": "01100 00100 01000 00000 00000 00000 00000", "-": "00000 00000 00000 11111 00000 00000 00000",
    ":": "00000 01100 01100 00000 01100 01100 00000", "/": "00000 00001 00010 00100 01000 10000 00000",
    "%": "11000 11001 00010 00100 01000 10011 00011", "+": "00000 00100 00100 11111 00100 00100 00000",
}
GLYPH_W, GLYPH_H = 5, 7   # plus one column of spacing per character


def missing_glyphs(text: str) -> "set[str]":
    """Characters of `text` that draw_text would refuse (empty when it can draw it all)."""
    return set(text.upper()) - set(_GLYPHS)


@lru_cache(maxsize=None)
def _glyph_pixels(ch: str, scale: int) -> "tuple[np.ndarray, np.ndarray]":
    bits = np.array([[c == "1" for c in row] for row in _GLYPHS[ch].split()])
    ys, xs = np.nonzero(np.kron(bits, np.ones((scale, scale), bool)))
    ys.flags.writeable = xs.flags.writeable = False   # cached and shared: never mutated
    return ys, xs


def text_size(text: str, scale: int) -> "tuple[int, int]":
    """(width, height) in pixels of `text` drawn at `scale`."""
    return max(len(text) * (GLYPH_W + 1) - 1, 0) * scale, GLYPH_H * scale


def draw_text(frame: np.ndarray, text: str, x: int, y: int, scale: int, color) -> None:
    """Draw capitals in place, clipped at the frame edge. Lower case is upper-cased;
    any character without a glyph raises ValueError."""
    h, w = frame.shape[:2]
    text = text.upper()
    if missing_glyphs(text):
        raise ValueError(f"no glyph for {sorted(missing_glyphs(text))}")
    for k, ch in enumerate(text):
        ys, xs = _glyph_pixels(ch, scale)
        ys, xs = ys + y, xs + x + k * (GLYPH_W + 1) * scale
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        frame[ys[ok], xs[ok]] = color


def draw_ring(frame: np.ndarray, center, radius: float, color, r: int = 2) -> None:
    """A circle outline of `radius` around `center`, drawn with `draw_line`."""
    t = np.linspace(0, 2 * np.pi, 49)
    pts = np.asarray(center, float) + radius * np.stack([np.cos(t), np.sin(t)], axis=1)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw_line(frame, p0, p1, color, r)


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


def draw_number(frame: np.ndarray, n: int, x: int = NUMBER_X, y: int = NUMBER_Y,
                scale: int = NUMBER_SCALE) -> None:
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
                   max_frames: "int | None" = None, info: "VideoInfo | None" = None,
                   annotate: "Callable[[np.ndarray, int], None] | None" = None) -> int:
    """Write `out` with the skeleton drawn on `video`; return frames written.
    `info` is `probe(video)`, passed in to spare a second packet scan.
    `annotate(frame, i)`, if given, draws on frame i in place after the skeleton
    and before the frame number, which stays on top."""
    video, out = Path(video).resolve(), Path(out).resolve()
    if out == video:
        raise ValueError("refusing to overwrite the input video with its own overlay")
    info = info or probe(video)
    if (info.width, info.height) != (pose.width, pose.height):
        raise ValueError(f"pose is for {pose.width}x{pose.height}, video is "
                         f"{info.width}x{info.height}; not drawing a misaligned skeleton")
    if abs(pose.fps - info.fps) > 1e-6:
        raise ValueError(f"pose is at {pose.fps:.3f} fps, video at {info.fps:.3f}; "
                         "it was extracted from a different video")
    good = pose.usable(min_conf)
    n = 0
    with VideoWriter(out, info.width, info.height, info.fps_frac) as vw:
        for i, frame in enumerate(iter_frames(video, info, max_frames)):
            if i >= len(pose):
                raise ValueError(f"video has more frames than the pose ({len(pose)})")
            if pose.detected[i]:  # draw what was found, even when it is not usable:
                draw_skeleton(frame, pose.xy[i], pose.conf[i], min_conf)  # shows WHY
            if not good[i]:  # undetected, too few solid joints, or a wrist/shoulder missing
                _border(frame, C_MISS)
            if annotate is not None:
                annotate(frame, i)
            draw_number(frame, i)
            vw.write(frame)
            n += 1
        # Inside the `with`, so a refusal deletes the half-written file. Stopping at
        # max_frames is a prefix the caller asked for; ending early otherwise means
        # the pose belongs to a longer video.
        if n < len(pose) and (max_frames is None or n < max_frames):
            raise ValueError(f"pose has {len(pose)} frames, video only {n}; "
                             "it was extracted from a different video")
    return n
