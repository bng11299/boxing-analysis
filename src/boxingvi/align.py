"""Alignment checks between pose, labels and source video.

A label that is a few frames off, or a skeleton indexed from a different
origin than the video, produces plausible output everywhere downstream. These
checks turn "the frames line up" into a measured fact before anything is
scored on them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .data import Span, Track


def in_span_mask(track: Track, spans: "list[Span]", shift: int = 0) -> np.ndarray:
    """Per-row: is this frame inside any span, with every span moved by `shift`?"""
    starts = np.array([s.start for s in spans]) + shift
    ends = np.array([s.end for s in spans]) + shift
    f = track.frame_ids
    # A frame is inside some span iff more spans have started than ended by it.
    started = np.searchsorted(np.sort(starts), f, side="right")
    ended = np.searchsorted(np.sort(ends), f, side="left")
    return started > ended


def label_shift_scan(track: Track, spans, speed: np.ndarray, shifts) -> dict:
    """Mean speed inside labelled spans over mean speed outside, per shift.

    If the labels are aligned to the pose, the ratio peaks at shift 0 and
    falls away on both sides. Restricted to the labelled range, so the long
    unlabelled tail cannot dilute the "outside" mean.
    """
    if speed.shape != track.frame_ids.shape:
        raise ValueError("speed must be one value per track row")
    lo, hi = min(s.start for s in spans), max(s.end for s in spans)
    rng = (track.frame_ids >= lo) & (track.frame_ids <= hi)
    out = {}
    for k in shifts:
        m = in_span_mask(track, spans, int(k))
        inside, outside = speed[m & rng], speed[~m & rng]
        if len(inside) == 0 or len(outside) == 0 or outside.mean() == 0:
            raise ValueError(f"shift {k}: an empty side, the ratio is undefined")
        out[int(k)] = float(inside.mean() / outside.mean())
    return out


def peri_event_peak(track: Track, spans, speed: np.ndarray, half: int = 40) -> dict:
    """Lag, from each labelled start, at which the averaged speed profile peaks."""
    row = {int(f): i for i, f in enumerate(track.frame_ids)}
    lags = np.arange(-half, half + 1)
    prof = []
    for s in spans:
        if s.start not in row:
            continue
        idx = row[s.start] + lags
        # Only windows of consecutive frames -- a gap would splice two moments.
        if idx[0] < 0 or idx[-1] >= len(speed):
            continue
        if track.frame_ids[idx[-1]] - track.frame_ids[idx[0]] != 2 * half:
            continue
        prof.append(speed[idx])
    if not prof:
        raise ValueError("no span has a full window of consecutive pose around it")
    mean = np.mean(prof, axis=0)
    med = float(np.median(mean))
    # Median 0 (a body at rest for most of the window): a positive peak over it
    # is as prominent as a peak can be, and an all-zero profile has none.
    peak = float(mean.max())
    ratio = peak / med if med > 0 else (float("inf") if peak > 0 else 0.0)
    return {"peak_lag": int(lags[mean.argmax()]), "n_spans": len(prof),
            "peak_over_median": ratio}


def video_motion_energy(path: Path, size=(96, 54)) -> np.ndarray:
    """Mean absolute grey-level change per decoded frame (frame 0 scores 0).

    Decoded with passthrough timing, so the i-th value is the i-th frame the
    file contains: no frame is duplicated or dropped to hit a nominal rate.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no video at {path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    w, h = size
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
           "-vf", f"scale={w}:{h}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    # stderr is discarded, not piped: an unread pipe fills on a corrupt file's
    # per-packet errors and deadlocks ffmpeg while we wait on stdout. The
    # context manager kills the process if anything below raises.
    out, prev = [], None
    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL) as proc:
        while True:
            buf = proc.stdout.read(w * h)
            if len(buf) < w * h:
                break
            fr = np.frombuffer(buf, np.uint8).astype(np.float32)
            out.append(0.0 if prev is None else float(np.abs(fr - prev).mean()))
            prev = fr
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path.name} (exit {proc.returncode})")
    return np.asarray(out)


def skeleton_video_xcorr(track: Track, energy: np.ndarray, lags=range(-6, 7)) -> dict:
    """Pearson r between skeleton motion and video motion, with the skeleton's
    frame ids read as video frame index + lag. Peaks at 0 iff aligned."""
    motion = np.zeros(len(track.frame_ids))
    for lo, hi in track.segments():
        if hi - lo > 1:
            motion[lo + 1:hi] = np.linalg.norm(
                np.diff(track.xy[lo:hi], axis=0), axis=-1).sum(axis=1)
    out = {}
    for k in lags:
        idx = track.frame_ids + int(k)
        ok = (idx >= 0) & (idx < len(energy))
        if ok.sum() < 3:
            raise ValueError(f"lag {k}: too few overlapping frames")
        out[int(k)] = float(np.corrcoef(motion[ok], energy[idx[ok]])[0, 1])
    return out
