"""Local participant-data output and strict CFR checks for sparring tools."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

from .video import MAX_TIMING_PACKETS, TIMING_TIMEOUT_S, _tool, probe

REPO = Path(__file__).resolve().parents[2]


def external_path(path) -> Path:
    """Refuse this checkout and any other Git checkout, including linked paths."""
    raw = Path(os.path.abspath(Path(path).expanduser()))
    resolved = raw.resolve()
    for candidate in (raw, resolved):
        if candidate.is_relative_to(REPO):
            raise ValueError("output path is inside the git repo")
        if any((parent / ".git").exists() for parent in (candidate, *candidate.parents)):
            raise ValueError("output path is inside a git repo")
    return resolved


@contextmanager
def staged_output(path, force=False):
    """Publish a complete file; no-clobber is atomic, even if another writer wins."""
    out = external_path(path)
    if out.exists() and not force:
        raise ValueError("output exists; pass --force to replace it")
    if not out.parent.is_dir():
        raise ValueError("output directory does not exist")
    fd, name = tempfile.mkstemp(prefix=".sparring-", suffix=out.suffix, dir=out.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        # Recheck the parent before publication. Parent directories are trusted:
        # this does not defend against a privileged concurrent junction swap.
        if external_path(out) != out:
            raise ValueError("output path changed while writing")
        if force:
            os.replace(temporary, out)
        else:
            try:
                os.link(temporary, out)
            except FileExistsError:
                raise ValueError("output exists; pass --force to replace it") from None
            except OSError:  # FAT32 / exFAT / network shares have no hard links
                raise ValueError("this drive does not support hard links, so a safe no-overwrite "
                                 "write is impossible; write to an NTFS folder") from None
    finally:
        temporary.unlink(missing_ok=True)


def strict_cfr(path):
    """Check decoded presentation timestamps, allowing only one time-base tick.

    The general video probe tolerates camera jitter; a labelling copy cannot.
    Checking every timestamp also catches local VFR whose mean rate is 30.
    At least three frames and a sufficiently precise time base are required.
    """
    path = Path(path).resolve()
    info = probe(path)
    if info.timing is None or info.vfr:
        raise ValueError("VFR or unmeasurable frame timing; a constant-frame-rate video is required")
    cmd = [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=time_base:frame=best_effort_timestamp",
           "-of", "json", str(path)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=TIMING_TIMEOUT_S)
        if done.returncode:
            raise ValueError
        data = json.loads(done.stdout)
        tick = Fraction(data["streams"][0]["time_base"])
        frames = data["frames"]
        if not 3 <= len(frames) <= MAX_TIMING_PACKETS or len(frames) != info.timing.n_frames:
            raise ValueError
        pts = [frame["best_effort_timestamp"] for frame in frames]
        if any(type(p) is not int for p in pts):
            raise ValueError
        period = 1 / info.fps_frac / tick
        if tick <= 0 or period < 10:
            raise ValueError
        # Integer arithmetic avoids float drift over long recordings.
        num, den = period.numerator, period.denominator
        for i, timestamp in enumerate(pts):
            if abs((timestamp - pts[0]) * den - i * num) > den:
                raise ValueError
            if i and (timestamp <= pts[i - 1] or
                      abs((timestamp - pts[i - 1]) * den - num) > den):
                raise ValueError
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError,
            subprocess.TimeoutExpired):
        raise ValueError("VFR or unmeasurable frame timing; timestamps must be constant-rate") from None
    return info
