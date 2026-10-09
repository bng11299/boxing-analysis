"""Frames the camera dropped: where what is on screen jumps forward in one frame.

The Olympic GoPro files have perfectly regular packet timestamps (every interval 20 ms, `video.probe`
reports no variable frame rate), yet the camera drops CAPTURED frames: the picture jumps forward while
the timestamps stay regular. The wrist-speed rule reads that jump as a very fast punch. The only
trace is in the pixels, as one number per frame (`video.frame_change`: how much the picture changed
since the previous frame).

THE RULE: frame f is a SKIP when change[f] > SKIP_RATIO x the median of `change` over
[f - 10, f + 10] EXCLUDING f. The neighbours' median is the local normal: a camera pan or a crowd
surge raises it for the whole stretch and nothing in it is flagged; an isolated one-frame jump
stands out. Frame 0 has no previous frame (NaN) and is never a skip; NaN values never enter a median.

DECLARATION: the rule and SKIP_RATIO = 3.0 come from Brendan's observation (two sampled "bursts" he
called not-a-punch sat on such a jump) and a reviewer's inspection of one DEVELOPMENT video (GH069681).
They were fixed by reading that observation, not tuned to any completeness, precision or score, and
no other video was used to choose them. How well they transfer to the other 28 videos is a claim the
numbers in docs/quality-loop.md report, not one this module makes.

A `Skips` is bound to the video it was measured on (source file name, number of frames, content
fingerprint) and saved as `<id>.framechange.npz` beside the pose table; the loader is strict.
"""

from __future__ import annotations

import os
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pose_yolo import _float_array, _scalar, _text

SKIP_RATIO = 3.0
HALF_WINDOW = 10
STEP = 4                    # video.frame_change's pixel step when this file was written
MAX_FRAMES = 20_000_000     # ~55 h at 100 fps; a file claiming more is refused
_FP = re.compile(r"[0-9a-f]{64}")


def _check_params(ratio, half_window) -> None:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float, np.integer, np.floating)) \
            or not (np.isfinite(ratio) and ratio > 0):
        raise ValueError("ratio must be a finite number > 0")
    if isinstance(half_window, bool) or not isinstance(half_window, (int, np.integer)) or half_window < 1:
        raise ValueError("half_window must be a whole number >= 1")


def _check_change(change) -> np.ndarray:
    a = np.asarray(change)
    if a.ndim != 1 or a.dtype.kind != "f":
        raise ValueError("change must be a 1-D floating-point array")
    if len(a) > MAX_FRAMES:
        raise ValueError(f"change has {len(a)} entries: more than {MAX_FRAMES} frames is refused")
    if np.isinf(a).any() or (a < 0).any():
        raise ValueError("change must be >= 0 and finite (NaN marks 'no previous frame')")
    return a.astype(np.float64)


def find_skips(change, ratio: float = SKIP_RATIO, half_window: int = HALF_WINDOW) -> np.ndarray:
    """Sorted frame ids f with change[f] > ratio x median(change[f-half_window : f+half_window+1] without f).

    The window is clipped at the ends of the video (fewer neighbours, same rule); NaN entries are
    ignored. A frame with no finite neighbour (a one-frame video) is never a skip. Strictly greater:
    exactly `ratio` x the median is not a skip. On a perfectly still stretch the median is 0, so a
    non-zero change in it is a skip: there is no absolute floor, because none was observed to be
    needed and an unobserved one would be a tuned constant."""
    _check_params(ratio, half_window)
    c = _check_change(change)
    n = len(c)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    padded = np.concatenate([np.full(half_window, np.nan), c, np.full(half_window, np.nan)])
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1).copy()
    win[:, half_window] = np.nan                       # the frame itself is never its own reference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # an all-NaN window: no reference
        local = np.nanmedian(win, axis=1)
    hit = np.isfinite(c) & np.isfinite(local) & (c > ratio * local)
    return np.flatnonzero(hit).astype(np.int64)


@dataclass(frozen=True)
class Skips:
    """The per-frame change of ONE video and the skips read from it, bound to that video."""
    change: np.ndarray      # (n_frames,) float64, NaN at frame 0
    source: str             # the video's file name
    video_fp: str           # video.fingerprint of that file
    step: int = STEP
    ratio: float = SKIP_RATIO
    half_window: int = HALF_WINDOW

    def __post_init__(self) -> None:
        a = _check_change(self.change)
        if len(a) < 1:
            raise ValueError("a video has at least one frame")
        if not np.isnan(a[0]) or np.isnan(a[1:]).any():
            raise ValueError("change must be NaN at frame 0 and a number everywhere else")
        _check_params(self.ratio, self.half_window)
        if isinstance(self.step, bool) or not isinstance(self.step, (int, np.integer)) or self.step < 1:
            raise ValueError("step must be a whole number >= 1")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be the video's file name")
        if not isinstance(self.video_fp, str) or not _FP.fullmatch(self.video_fp):
            raise ValueError("video_fp must be a 64-character lowercase sha256 hex digest")

    @property
    def n_frames(self) -> int:
        return len(self.change)

    @property
    def frames(self) -> np.ndarray:
        """Sorted skip frame ids, by the rule and the parameters stored with the data."""
        return find_skips(self.change, self.ratio, self.half_window)

    def verify(self, table) -> None:
        """Refuse unless this is the framechange of the video `table` was extracted from: the same
        file name, the same number of frames, and the same content fingerprint when the table has
        one (an older table has none and is bound by name and length only)."""
        if self.source != table.source:
            raise ValueError(f"the frame changes are for {self.source!r}, the pose table for {table.source!r}")
        if self.n_frames != table.n_frames:
            raise ValueError(f"the frame changes cover {self.n_frames} frames, the pose table {table.n_frames}")
        if table.video_fp and self.video_fp != table.video_fp:
            raise ValueError("the frame changes and the pose table were made from different files "
                             "(content fingerprints differ)")

    def save(self, path) -> None:
        """Write atomically (a half-written file is never read as a complete one)."""
        path = Path(path)
        tmp = path.with_name(path.name + ".part")
        try:
            with tmp.open("wb") as f:
                np.savez_compressed(f, change=self.change, n_frames=self.n_frames, step=self.step,
                                    ratio=self.ratio, half_window=self.half_window,
                                    source=self.source, video_fp=self.video_fp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path) -> "Skips":
        """Strict: every field present with the right dtype and shape (read through the pose
        loader's helpers), `n_frames` agreeing with the array, no pickle. Anything else is a
        ValueError naming the field."""
        try:
            with np.load(path, allow_pickle=False) as z:
                change = _float_array(z, "change")
                n = _scalar(z, "n_frames", whole=True, lo=1)
                if change.ndim != 1 or len(change) != n:
                    raise ValueError(f"change must be a 1-D array of n_frames = {n} values")
                return cls(change=change.astype(np.float64), source=_text(z, "source"), video_fp=_text(z, "video_fp"),
                           step=_scalar(z, "step", whole=True, lo=1), ratio=_scalar(z, "ratio", whole=False),
                           half_window=_scalar(z, "half_window", whole=True, lo=1))
        except (zipfile.BadZipFile, EOFError) as e:      # a truncated or foreign file
            raise ValueError(f"{Path(path).name} is not a frame-change file ({type(e).__name__})") from e


def load_for(table, path) -> Skips:
    """Load `path` and verify it against the pose table; ValueError/FileNotFoundError otherwise."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no {path.name}: run olympic_framechange.py first")
    s = Skips.load(path)
    s.verify(table)
    return s
