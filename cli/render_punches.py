"""The footage with the skeleton AND each punch drawn on it -- the demo-page asset.

  python cli/render_punches.py --video data/own/IMG_4406.mov \\
      --pose results/ingest/IMG_4406.pose.npz \\
      --classified results/punches/IMG_4406.classified.csv

Writes <out-dir>/<clip>.punches.overlay.mp4 (default results/ingest): H.264 at the
video's constant frame rate, no audio. During each punch's frames
(punch_start_frame..punch_end_frame in the CSV) it draws the punch type, the hand
(lead/rear, from the wrist that moved) and the classifier's confidence, and rings
the punching wrist; a punch the classifier could not read is drawn in grey with its
reason. The skeleton, the red no-pose border and the burned-in frame number are
overlay.render_overlay's, unchanged.

Refuses unless the CSV is unchanged since classify_punches.py wrote it (the
<clip>.classified.json beside it records its sha256) and was made from this pose
(same length and frame rate), and the pose was extracted from this video.

The output shows your footage: it lands in a gitignored folder, never commit it,
and never render participant footage for the public demo.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import overlay, report, video as videomod  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from boxingvi.video import VideoInfo  # noqa: E402

DEFAULT_OUT = Path("results/ingest")
WRIST = {"left": 9, "right": 10}           # COCO-17
C_NAMED = (90, 235, 90)                    # a punch the classifier named
C_UNNAMED = (170, 170, 170)                # grey: found, but not named
MAX_LABELS = 4                             # overlapping punches drawn; the rest are counted
NUMBER_BOTTOM = overlay.NUMBER_Y + 5 * overlay.NUMBER_SCALE   # the frame index's last row
# What READING a missing, corrupt or crafted input file can raise besides ValueError.
# A caller (analyse_clip.py) treats ValueError as "refused", so at the load sites
# (and only there: elsewhere these would be bugs, not refusals) they are converted.
_INPUT_ERRORS = (OSError, KeyError, IndexError, TypeError, OverflowError, RecursionError, EOFError,
                 MemoryError, csv.Error, zipfile.BadZipFile, subprocess.SubprocessError)


def _load(path: Path, read):
    """`read()`, with an unreadable `path` reported as a ValueError naming it."""
    try:
        return read()
    except _INPUT_ERRORS as e:
        raise ValueError(f"cannot read {path.name} ({type(e).__name__})") from e


def layout(width: int, height: int) -> "tuple[int, int, int, int]":
    """(scale, x, y, max_chars) of the label block: under the burned-in frame
    number, glyphs ~1/120 of the short side tall (3 px per bit on 1080p)."""
    scale = max(1, min(width, height) // 360)
    x, y = 6 + 2 * scale, NUMBER_BOTTOM + 4 * scale
    return scale, x, y, max(8, (width - 2 * x) // ((overlay.GLYPH_W + 1) * scale))


def _wrap(text: str, max_chars: int) -> "list[str]":
    lines, cur = [], ""
    for word in text.split(" "):
        while len(word) > max_chars:            # a word wider than the frame is cut, not lost
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    return lines + ([cur] if cur else [])


def punch_text(row: dict) -> "tuple[str, tuple]":
    """The words drawn for one punch (a row of report.read_classified) and their colour."""
    hand = f"{row['lead_rear'].upper()} HAND"
    if row["punch_type"] == report.UNCLASSIFIED:
        return f"UNCLASSIFIED - {hand} ({row['note']})", C_UNNAMED
    text = f"{row['punch_type'].replace('_', ' ').upper()} - {hand} - {row['confidence']:.3f}"
    if row["hands_agree"] == "no":   # the type is thrown by the other hand than the one that moved
        text += f" (TYPE IS A {row['type_hand'].upper()} PUNCH)"
    return text, C_NAMED


class PunchLabels:
    """`annotate(frame, i)` for overlay.render_overlay: labels and a wrist ring on
    every frame inside a punch's span, nothing anywhere else."""

    def __init__(self, rows: "list[dict]", pose: PoseSequence, min_conf: float = 0.3):
        n = len(pose)
        self.rows, self.pose, self.min_conf = rows, pose, min_conf
        # A span can start before the clip (a punch thrown as recording began) or end
        # after it; start <= i <= end still holds on the first and last frames, so
        # they are labelled like any other. Bounded to one frame past either edge,
        # which changes no frame's labels but keeps a huge number in the CSV from
        # overflowing int64. A row without a span (allowed only for an unclassified
        # punch) shows on its peak frame.
        span = [r["span"] or (r["frame"], r["frame"]) for r in rows]
        self.start = np.array([max(a, -1) for a, _ in span], dtype=np.int64)
        self.end = np.array([min(b, n) for _, b in span], dtype=np.int64)
        self.scale, self.x, self.y, self.max_chars = layout(pose.width, pose.height)
        self.radius = max(6.0, 0.03 * min(pose.width, pose.height))
        # Every label's lines, built now: a character the font cannot draw is refused
        # before the first frame is encoded, not half way through the video.
        self.lines = []
        for r in rows:
            text, color = punch_text(r)
            if overlay.missing_glyphs(text):
                raise ValueError(f"cannot draw {sorted(overlay.missing_glyphs(text))} in the label {text!r}")
            self.lines.append([(ln, color) for ln in _wrap(text, self.max_chars)])

    def active(self, i: int) -> "list[int]":
        """Indices of the rows whose span holds frame i, in peak order."""
        return [int(k) for k in np.nonzero((self.start <= i) & (self.end >= i))[0]]

    def __call__(self, frame: np.ndarray, i: int) -> None:
        act = self.active(i)
        if not act:
            return
        lines = []
        for k in act[:MAX_LABELS]:
            lines += self.lines[k]
            self._ring(frame, i, self.rows[k]["hand"], self.lines[k][0][1])
        if len(act) > MAX_LABELS:
            lines.append((f"+{len(act) - MAX_LABELS} MORE", C_UNNAMED))
        s, y = self.scale, self.y
        for text, color in lines:
            w, h = overlay.text_size(text, s)
            box = frame[max(y - s, 0):y + h + s, max(self.x - s, 0):self.x + w + s]
            box //= 3                                   # darkened backing: legible on any footage
            overlay.draw_text(frame, text, self.x, y, s, color)
            y += h + 3 * s

    def _ring(self, frame: np.ndarray, i: int, hand: str, color) -> None:
        j = WRIST[hand]
        xy, c = self.pose.xy[i, j], self.pose.conf[i, j]
        # Only where the skeleton itself draws the wrist: a ring on a guessed wrist
        # would point at a place the model did not see.
        if self.pose.detected[i] and c >= self.min_conf and np.all(np.isfinite(xy)):
            overlay.draw_ring(frame, xy, self.radius, color, self.scale)


def _same_file(a: Path, b: Path) -> bool:
    """Same path, or (both existing) the same file under another name: a hard link
    or junction onto an input must not be overwritten either."""
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _checked_inputs(video: Path, pose, classified: Path, out: Path,
                    info: "VideoInfo | None") -> "tuple[PoseSequence, list[dict], VideoInfo]":
    """The pose, the CSV's rows and the video's probe, after every refusal
    render_punch_overlay promises."""
    clip = classified.name.split(".")[0]
    sidecar = classified.with_suffix(".json")
    inputs = [video, classified, sidecar]
    if not isinstance(pose, PoseSequence):
        if Path(pose).name.split(".")[0] != clip:
            raise ValueError(f"{Path(pose).name} does not name the clip {clip}")
        inputs.append(Path(pose))
        pose = _load(Path(pose), lambda: PoseSequence.load(pose))
    if video.name.split(".")[0] != clip:
        raise ValueError(f"{video.name} does not name the clip {clip}")
    if not video.is_file():
        raise ValueError(f"no video at {video.name}")
    # Fail closed: run_ingest.py always records the video's name; a pose without
    # one did not come from it, and nothing would tie it to this footage.
    if pose.source != video.name:
        raise ValueError(f"the pose was extracted from {pose.source or 'an unrecorded video'}, not {video.name}")
    if any(_same_file(out, p) for p in inputs):
        raise ValueError("refusing to overwrite an input with the overlay")
    info = info or _load(video, lambda: videomod.probe(video))   # a hanging ffprobe times out here

    if not sidecar.is_file():
        raise ValueError(f"no {sidecar.name} beside the CSV: re-run classify_punches.py, which writes it")
    data = _load(classified, classified.read_bytes)   # read once: the bytes hashed are the bytes parsed
    side = _load(sidecar, lambda: report.verify_sidecar(classified, sidecar, clip, data))
    # The report's own check (frames AND fps, exact, overflow-safe), so the overlay and the
    # report refuse the same sidecars; then every row's time is checked against frame / fps.
    report.check_pose(side, len(pose), pose.fps)
    rows = _load(classified, lambda: report.read_classified(classified, len(pose) / pose.fps, data, fps=pose.fps))
    for r in rows:
        if not 0 <= r["frame"] < len(pose):
            raise ValueError(f"{classified.name}: a punch peaks at frame {r['frame']}, outside this "
                             f"{len(pose)}-frame pose")
    return pose, rows, info


def render_punch_overlay(video, pose: "PoseSequence | Path | str", classified, out, *,
                         min_conf: float = 0.3, max_frames: "int | None" = None,
                         info: "VideoInfo | None" = None) -> int:
    """Write `out`: `video` with its skeleton and its classified punches drawn on it.
    Returns frames written. Every refusal is a ValueError: of the inputs, raised
    before anything is written; a video/pose length mismatch found while decoding,
    with the partial file removed. ffmpeg failing part way through (a video it can
    probe but not decode to the end, or the encoder failing, e.g. a full disk) is a
    RuntimeError, also with the partial file removed: the two cannot be told apart
    from here, and the second is not a refusal.

    `pose` is the clip's PoseSequence or its .pose.npz; `classified` is the
    <clip>.classified.csv, whose <clip>.classified.json must sit beside it.
    `info` is `probe(video)`, passed in to spare a second packet scan."""
    video, classified, out = Path(video), Path(classified), Path(out)
    pose, rows, info = _checked_inputs(video, pose, classified, out, info)
    labels = PunchLabels(rows, pose, min_conf)
    return overlay.render_overlay(video, pose, out, min_conf, max_frames, info, annotate=labels)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True, help="<clip>.pose.npz from run_ingest.py")
    ap.add_argument("--classified", type=Path, required=True, help="<clip>.classified.csv (+ .json beside it)")
    ap.add_argument("--min-conf", type=float, default=0.3, help="joint confidence to draw")
    ap.add_argument("--max-frames", type=int, help="render only this many frames from the start")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    clip = a.classified.name.split(".")[0]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{clip}.punches.overlay.mp4"
    n = render_punch_overlay(a.video, a.pose, a.classified, out, min_conf=a.min_conf, max_frames=a.max_frames)
    print(f"{n} frames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
