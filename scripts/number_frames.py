"""Burn zero-based decoded frame indices into a CFR H.264 copy, without audio.

python scripts/number_frames.py --video INPUT.mp4 --output OUTSIDE_REPO.mp4
Requires ffmpeg/ffprobe with libx264 (no fonts: digits are drawn in numpy).
Input must already be face-blurred. This script does not perform face blurring.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxingvi.sparring_io import external_path, staged_output, strict_cfr  # noqa: E402
from boxingvi.lsimport import FRAME_CHECKS  # noqa: E402
from boxingvi.overlay import draw_number  # noqa: E402
from boxingvi.video import VideoWriter, iter_frames  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    source = args.video.resolve()
    out = external_path(args.output)
    if out.suffix.lower() != ".mp4":
        raise ValueError("output must be an .mp4 file")
    if out == source or (out.exists() and source.exists() and out.samefile(source)):
        raise ValueError("output must not replace the input video")
    if out.exists() and not args.force:
        raise ValueError("output exists; pass --force to replace it")
    info = strict_cfr(source)
    # Refuse now what the importer would refuse after hours of labelling.
    if info.fps_frac != 30:
        raise ValueError("sparring-punch-labels.xml is set to 30 fps; video must be constant 30 fps")
    if info.timing.n_frames <= max(FRAME_CHECKS):
        raise ValueError(f"video too short for the frame check at frame {max(FRAME_CHECKS)}")
    with staged_output(out, args.force) as temporary:
        # Same 0-based index and bitmap digits as every other overlay in the repo
        # (overlay.draw_number), drawn in numpy: ffmpeg's drawtext needs a system
        # font config, which Windows ffmpeg builds do not ship. The raw-frame pipe
        # also means no audio or metadata from the source can reach the copy.
        # The frame-count check below (on the finished copy) also catches a short decode.
        with VideoWriter(temporary, info.width, info.height, info.fps_frac, private=True) as vw:
            for i, frame in enumerate(iter_frames(source, info)):
                draw_number(frame, i)
                vw.write(frame)
        copied = strict_cfr(temporary)
        if (copied.timing.n_frames != info.timing.n_frames or
                copied.fps_frac != info.fps_frac or
                (copied.width, copied.height) != (info.width, info.height)):
            raise ValueError("numbered copy changed the frame count, fps or dimensions")
    print(f"{info.timing.n_frames} frames; {info.fps:g} fps")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else "video processing failed"
        print(f"error: {message}", file=sys.stderr)
        sys.exit(1)
