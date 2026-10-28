"""Brendan's left/right labels for the glove-identity gate (PR #31), filled in one moment at a time.

The labels CSV is written by `glove_identity.py template` (#31, `glovescore.write_template`) and read by
`glovescore.parse_labels`. This module changes exactly three things in it: the `# A:` value, the
`# minutes:` value and each row's a_side. Every other byte (header lines, comments, row order, line
endings, a BOM) is written back as it was read.

What it may read: the labels CSV, and the BLURRED video that CSV was made for (matched by fingerprint).
Nothing else. The glove method's output (`.npz`, `.log`) sits beside the labels and is never opened, so the
labels stay independent of what the method guessed.

Trust boundary: both files are checked before anything is shown. The video must lie under
`~/comps-participant-data/blurred/` after resolving links, must not be an unblurred original (`PXL_*`), must
carry no audio track (blurred copies carry none), must have regular frame timing (the gate scores frame
round(t x fps)), and must be the video the CSV names. It is held open from the check to the end of the
session, and its identity is re-checked before every decode, so it cannot be swapped once checked. The CSV
must lie under `~/comps-participant-data/`, follow the template's format exactly, and hold exactly the
gate's time grid for this video. `# A:` is generated from glove colours chosen off a fixed list, so no
name can be typed into it. Messages never repeat an input value or a path.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .video import MAX_PIXELS, fingerprint
from .video import probe as _video_probe

PARTICIPANT_ROOT = Path.home() / "comps-participant-data"
BLURRED_ROOT = PARTICIPANT_ROOT / "blurred"

LABEL_FORMAT = "glove-identity-labels-1"      # glovescore.LABEL_FORMAT
EVERY_S = 5                                   # glovescore.LABEL_EVERY_S
MAX_LABEL_BYTES = 1024 * 1024                 # glovescore.MAX_LABEL_BYTES
MAX_MINUTES = 600.0                           # glovescore.MAX_MINUTES
MAX_DURATION_S = 6 * 3600.0                   # glovescore.MAX_DURATION_S
MAX_ROWS = int(MAX_DURATION_S) // EVERY_S + 1
VALUES = ("L", "R", "?")
HEADER_KEYS = ("format", "video_fp", "every_s", "A", "minutes")

IDLE_CAP_S = 120.0       # a gap longer than this between two key presses counts as this long
CLIP_HALF_S = 1.0        # the clip runs from t - 1 s to t + 1 s
MAX_DISPLAY_W = 960
FFMPEG_TIMEOUT_S = 120
PROBE_TIMEOUT_S = 60

_HEADER = re.compile(r"(#\s*([A-Za-z_]+)\s*:)(.*)")
_ROW = re.compile(r"(0|[1-9][0-9]{0,5}),([LR?]?)")         # canonical times only: no 00, no 05
_FP = re.compile(r"[0-9a-f]{64}")
# str.splitlines (what parse_labels uses) also breaks on these; refusing them keeps both readers in step
_OTHER_LINE_BREAKS = "\x0b\x0c\x1c\x1d\x1e\x85  "

# `# A:` says who boxer A is, as "the boxer in <colour> gloves" or "the boxer in <c1> and <c2> gloves",
# built from this list. A free-text field cannot keep names out (Brown, Gray, White are surnames too);
# a fixed slot that only takes a glove colour can.
GLOVE_COLOURS = ("red", "blue", "black", "white", "green", "yellow", "orange", "pink", "purple", "grey",
                 "brown", "gold", "silver", "navy")
_A_TEXT = re.compile(r"the boxer in ([a-z]+)(?: and ([a-z]+))? gloves")


class Refused(ValueError):
    """An input fails a check; the message names the check, never the input."""


class SaveFailed(OSError):
    """The labels file could not be replaced (open in another program?); it is unchanged on disk."""


class FileChanged(SaveFailed):
    """The labels file on disk is no longer what this session read or last wrote (another copy of the tool,
    or an editor, saved it since). Nothing was written: an old copy must never overwrite newer answers."""


# ---- paths ----------------------------------------------------------------------------------------
def _norm(p) -> str:
    s = str(p)
    if s.startswith("\\\\?\\") and not s.upper().startswith("\\\\?\\UNC\\"):
        s = s[4:]
    return os.path.normcase(os.path.normpath(s))


def _under(p, root) -> bool:
    a, b = _norm(p), _norm(root)
    return a != b and a.startswith(b.rstrip("\\/") + os.sep)


def is_original_name(name: str) -> bool:
    """The camera's own files (`PXL_*.mp4`) are unblurred originals: never shown by this tool."""
    return name.upper().startswith("PXL_")


def check_video_path(video, blurred_root=BLURRED_ROOT) -> Path:
    p = Path(video)
    if is_original_name(p.name):
        raise Refused("that is an unblurred original (PXL_*): only blurred copies may be shown")
    try:
        r = p.resolve(strict=True)
        root = Path(blurred_root).resolve(strict=True)
    except OSError:
        raise Refused("the video (or the blurred-copies folder) does not exist") from None
    if is_original_name(r.name):
        raise Refused("that path leads to an unblurred original (PXL_*): only blurred copies may be shown")
    if not _under(r, root):
        raise Refused("the video is not under ~/comps-participant-data/blurred/: only blurred copies may be shown")
    if not r.is_file():
        raise Refused("the video is not a file")
    return r


def check_labels_path(labels, participant_root=PARTICIPANT_ROOT) -> Path:
    p = Path(labels)
    try:
        r = p.resolve(strict=True)
        root = Path(participant_root).resolve(strict=True)
    except OSError:
        raise Refused("the labels file (or ~/comps-participant-data/) does not exist") from None
    if not _under(r, root):
        raise Refused("the labels file is not under ~/comps-participant-data/: labels stay with the footage")
    if r.suffix.lower() != ".csv" or not r.is_file():
        raise Refused("the labels file must be a .csv file")
    return r


class VideoGuard:
    """The checked video, held open until the session ends. On Windows an open file can be neither replaced
    nor renamed, nor its folder renamed; everywhere, `check()` re-confirms before each decode that the path
    still resolves to the same file (device, file id, size, modification time) under the blurred root.
    `path_check(video, root)` is the path rule (default: blurred copies only); another tool may pass its own."""

    def __init__(self, video, blurred_root=BLURRED_ROOT, path_check=None):
        self.root = blurred_root
        self._path_check = path_check or check_video_path
        self.path = self._path_check(video, blurred_root)
        self._f = open(self.path, "rb")
        st = os.fstat(self._f.fileno())
        self._id = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)

    def check(self) -> Path:
        if self._f is None:
            raise Refused("the video was closed")
        try:
            again = self._path_check(self.path, self.root)
            st = os.stat(again)
        except (Refused, OSError):
            raise Refused("the video moved or changed after it was checked") from None
        if again != self.path or (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != self._id:
            raise Refused("the video moved or changed after it was checked")
        return self.path

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


# ---- the A description ----------------------------------------------------------------------------
def description_from_colours(text: str) -> str:
    """'red' -> 'the boxer in red gloves'; 'red and white' / 'red, white' / 'red/white' -> two colours.
    Only colours from GLOVE_COLOURS; anything else is refused, and the message never repeats it."""
    words = [w for w in re.split(r"[\s,/]+", (text or "").strip().lower()) if w and w != "and"]
    if not 1 <= len(words) <= 2:
        raise Refused("type one or two glove colours from the list")
    for n, w in enumerate(words, start=1):
        if w not in GLOVE_COLOURS:
            raise Refused(f"colour {n} is not on the list")
    if len(words) == 2 and words[0] == words[1]:
        raise Refused("two colours must differ")
    return "the boxer in " + " and ".join(words) + " gloves"


def check_description(text: str) -> str:
    """A `# A:` already in the file passes only if it is exactly what description_from_colours writes."""
    m = _A_TEXT.fullmatch(text or "")
    cols = [c for c in (m.groups() if m else ()) if c]
    if not cols or any(c not in GLOVE_COLOURS for c in cols) or (len(cols) == 2 and cols[0] == cols[1]):
        raise Refused("'# A:' is not of the form 'the boxer in <colour> gloves'")
    return text


# ---- the labels file ------------------------------------------------------------------------------
def replace_file(path: Path, data: bytes, on_disk, what: str, prefix: str = ".label_moments-") -> None:
    """Write `data` to `path` atomically: a temp file beside it, then a rename. A crash leaves the old file
    whole, and a hard link planted at the target is replaced, not written through. Refused (FileChanged) if
    the file on disk is not `on_disk` (what the caller read or last wrote), so a stale session never
    overwrites newer answers. `what` names the file in messages ("labels file")."""
    try:
        current = path.read_bytes()
    except OSError:
        raise SaveFailed(f"could not read the {what} back before saving") from None
    if current != on_disk:
        raise FileChanged(f"the {what} was changed by something else since this window opened it "
                          "(another copy of this tool?); nothing was written")
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=prefix, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise SaveFailed(f"could not save the {what} (is it open in Excel or another program?)") from None


@dataclass
class LabelFile:
    path: Path
    lines: list          # text lines, no line endings
    newline: str
    bom: bool
    trailing: bool       # the file ended with a newline
    head_idx: dict       # header key -> line index
    row_idx: list        # line index of each row
    times: list          # each row's t_s
    values: list         # each row's a_side: "", "L", "R" or "?"

    @classmethod
    def load(cls, path) -> "LabelFile":
        p = Path(path)
        try:
            if p.stat().st_size > MAX_LABEL_BYTES:
                raise Refused(f"the labels file is over {MAX_LABEL_BYTES} bytes: not a template")
            raw = p.read_bytes()
        except OSError:
            raise Refused("the labels file cannot be read") from None
        bom = raw.startswith(b"\xef\xbb\xbf")
        try:
            text = raw[3 if bom else 0:].decode("utf-8")
        except UnicodeDecodeError:
            raise Refused("the labels file is not UTF-8 text") from None
        if any(c in text for c in _OTHER_LINE_BREAKS):
            raise Refused("the labels file holds an unusual line break: not as the template wrote it")
        newline = "\r\n" if "\r\n" in text else "\n"
        if "\r" in text.replace("\r\n", "") or (newline == "\r\n" and "\n" in text.replace("\r\n", "")):
            raise Refused("the labels file mixes line endings: not as the template wrote it")
        lines = text.split(newline)
        trailing = lines[-1] == ""
        if trailing:
            lines.pop()
        head_idx, row_idx, times, values = {}, [], [], []
        seen_columns = False
        for i, line in enumerate(lines):
            if line.startswith("#"):
                m = _HEADER.fullmatch(line)
                if m and m.group(2) in HEADER_KEYS:
                    if m.group(2) in head_idx:
                        raise Refused(f"header '{m.group(2)}' appears twice")
                    head_idx[m.group(2)] = i
                continue
            if not line.strip():
                continue
            if not seen_columns:
                if line.strip() != "t_s,a_side":
                    raise Refused("the column header must be t_s,a_side")
                seen_columns = True
                continue
            m = _ROW.fullmatch(line)
            if not m:
                raise Refused(f"row {len(row_idx) + 1} is not 't,' or 't,L' / 't,R' / 't,?'")
            row_idx.append(i)
            times.append(int(m.group(1)))
            values.append(m.group(2))
        lf = cls(p, lines, newline, bom, trailing, head_idx, row_idx, times, values)
        lf.on_disk = raw            # what the file holds as far as this session knows
        missing = [k for k in HEADER_KEYS if k not in head_idx]
        if missing:
            raise Refused(f"missing header {missing}")
        if lf.head("format") != LABEL_FORMAT:
            raise Refused("unknown label format: this is not a glove-identity labels template")
        if lf.head("every_s") != str(EVERY_S):
            raise Refused(f"every_s must be {EVERY_S} (the gate's grid)")
        if not _FP.fullmatch(lf.head("video_fp")):
            raise Refused("video_fp must be a sha256 hex digest")
        if not times:
            raise Refused("the labels file has no rows")
        if len(times) > MAX_ROWS:
            raise Refused("the labels file has more rows than a 6 h video")
        for k, t in enumerate(times):
            if t != k * EVERY_S:
                raise Refused(f"row {k + 1} is off the {EVERY_S} s grid (rows must run 0, 5, 10, ... in order)")
        lf.prior_minutes()          # refuse an unreadable value now, not at the first save
        return lf

    def head(self, key: str) -> str:
        return _HEADER.fullmatch(self.lines[self.head_idx[key]]).group(3).strip()

    def _set_head(self, key: str, value: str) -> None:
        i = self.head_idx[key]
        self.lines[i] = _HEADER.fullmatch(self.lines[i]).group(1) + " " + value

    def prior_minutes(self) -> float:
        s = self.head("minutes")
        if not s:
            return 0.0
        try:
            m = float(s)
        except ValueError:
            raise Refused("'# minutes:' holds something that is not a number") from None
        if not (math.isfinite(m) and 0 <= m <= MAX_MINUTES):
            raise Refused(f"'# minutes:' must be between 0 and {MAX_MINUTES:.0f}")
        return m

    def set_value(self, k: int, value: str) -> None:
        if value not in VALUES:
            raise ValueError("a_side must be L, R or ?")
        self.values[k] = value
        self.lines[self.row_idx[k]] = f"{self.times[k]},{value}"

    def set_description(self, text: str) -> None:
        self._set_head("A", check_description(text))

    def set_minutes(self, minutes: float) -> None:
        if not (math.isfinite(minutes) and 0 < minutes <= MAX_MINUTES):
            raise ValueError("minutes out of range")
        self._set_head("minutes", f"{minutes:.2f}")

    def to_bytes(self) -> bytes:
        text = self.newline.join(self.lines) + (self.newline if self.trailing else "")
        return (b"\xef\xbb\xbf" if self.bom else b"") + text.encode("utf-8")

    def save(self) -> None:
        """Atomic: a temp file beside the target, then a rename. A crash leaves the old file whole, and a
        hard link planted at the target is replaced, not written through. Refused (FileChanged) if the file
        on disk is not what this session read or last wrote, so a stale session never overwrites newer
        answers."""
        data = self.to_bytes()
        replace_file(self.path, data, getattr(self, "on_disk", None), "labels file")
        self.on_disk = data

    def first_empty(self):
        return next((k for k, v in enumerate(self.values) if not v), None)

    def counts(self) -> dict:
        c = Counter(v or "empty" for v in self.values)
        return {k: c.get(k, 0) for k in ("L", "R", "?", "empty")}


def verify(lf: LabelFile) -> dict:
    """What the gate session needs, as counts only: rows per value, whether A passes, minutes."""
    try:
        check_description(lf.head("A"))
        a_ok = True
    except Refused:
        a_ok = False
    minutes = lf.prior_minutes()
    c = lf.counts()
    return {"rows": len(lf.values), **c, "A_ok": a_ok, "minutes": minutes,
            "complete": c["empty"] == 0 and a_ok and minutes > 0}


# ---- the video ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoMeta:
    width: int           # displayed size (after any rotation flag; ffmpeg applies it on decode)
    height: int
    fps: float
    n_frames: int        # frames the decoder outputs, measured from the packets
    duration: float      # n_frames / fps


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise Refused(f"{name} is not on PATH (install ffmpeg)")
    return exe


def probe(video: Path) -> VideoMeta:
    """Streams first (no audio, exactly one video stream), then the repo's own probe and frame-timing scan
    (`video.probe`): the gate scores frame round(t x fps), so the video must have regular frame timing."""
    cmd = [_tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file", "-show_entries",
           "stream=codec_type", "-of", "json", str(video)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Refused(f"ffprobe took over {PROBE_TIMEOUT_S} s to read the video") from None
    if done.returncode != 0:
        raise Refused("ffprobe could not read the video")
    try:
        kinds = [s.get("codec_type") for s in json.loads(done.stdout).get("streams", [])]
    except (ValueError, AttributeError, TypeError):
        raise Refused("ffprobe could not read the video") from None
    if "audio" in kinds:
        raise Refused("the video has an audio track; blurred copies carry none, so this is not one")
    if kinds.count("video") != 1:
        raise Refused("the video must have exactly one video stream")
    try:
        info = _video_probe(video)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired):
        raise Refused("the video has no readable size or frame rate") from None
    if info.vfr or info.timing is None:
        raise Refused("the video's frame timing is irregular: frame round(t x fps) would not be time t")
    fps, n = info.fps, int(info.timing.n_frames)
    if not (0 < info.width and 0 < info.height and info.width * info.height <= MAX_PIXELS):
        raise Refused("the video's frame size is out of range")
    if not (math.isfinite(fps) and 1 <= fps <= 240 and n > 0 and n / fps <= MAX_DURATION_S):
        raise Refused("the video's frame rate or length is out of range")
    return VideoMeta(info.width, info.height, fps, n, n / fps)


def label_grid(meta: VideoMeta) -> "list[int]":
    """glovescore.label_times: seconds 0, 5, 10, ... whose frame round(t x fps) lies inside the video."""
    out, t = [], 0
    while int(round(t * meta.fps)) < meta.n_frames:
        out.append(t)
        t += EVERY_S
    return out


def check_pair(lf: LabelFile, video: Path, meta: VideoMeta) -> None:
    """The CSV must be the one made for this blurred video, and hold exactly the gate's grid for it: no row
    past the end, none missing at the end."""
    if fingerprint(video) != lf.head("video_fp"):
        raise Refused("this labels file was made for a different video (fingerprints differ)")
    grid = label_grid(meta)
    if len(lf.times) > len(grid):
        k = len(grid)
        raise Refused(f"row {k + 1} (t = {lf.times[k]} s) is past the end of the video ({meta.duration:.1f} s)")
    if len(lf.times) < len(grid):
        raise Refused(f"rows are missing at the end: this video needs {len(grid)} rows, the file has "
                      f"{len(lf.times)}")


@dataclass(frozen=True)
class Clip:
    t: int
    width: int
    height: int
    fps: float
    start_f: int         # stored frame number of frames[0]
    frames: tuple        # raw RGB24 bytes, one per frame
    centre: int          # index of the frame at time t (stored frame round(t x fps))


def display_size(meta: VideoMeta, max_w: int = MAX_DISPLAY_W) -> "tuple[int, int]":
    w = min(meta.width, max_w)
    w -= w % 2
    h = max(2, int(round(meta.height * w / meta.width / 2)) * 2)
    return w, h


def decode_clip(guard: VideoGuard, meta: VideoMeta, t: int, half: float = CLIP_HALF_S) -> Clip:
    """Stored frames round((t - half) x fps) .. round((t + half) x fps), clamped to the video, scaled for
    display (decode_frames). A decode that stops before the red frame is refused rather than shifting the red
    frame onto another moment."""
    video = guard.check()
    start_f = max(0, int(round((t - half) * meta.fps)))
    t_f = int(round(t * meta.fps))
    end_f = min(meta.n_frames - 1, int(round((t + half) * meta.fps)))
    w, h = display_size(meta)
    frames = decode_frames(video, meta.fps, start_f, end_f - start_f + 1, w, h, f"the clip at t = {t} s",
                           min_frames=t_f - start_f + 1)
    return Clip(t, w, h, meta.fps, start_f, frames, t_f - start_f)


def decode_frames(video: Path, fps: float, start_f: int, n_want: int, w: int, h: int, what: str,
                  min_frames: int = 1) -> tuple:
    """Up to `n_want` stored frames from `start_f`, scaled to w x h, as raw RGB24 bytes, one per frame.
    -ss before -i seeks exactly (ffmpeg decodes from the keyframe and drops earlier frames); it is aimed half
    a frame early so a rounded timestamp never skips the first wanted frame, and -frames:v stops at the last.
    Passthrough timing keeps one output frame per stored frame. Fewer than `min_frames` is refused rather than
    shifting a frame onto another moment. `what` names the clip in messages."""
    buf, n = decode_raw(video, fps, start_f, n_want, w, h, what, min_frames)
    size = w * h * 3
    return tuple(buf[i * size:(i + 1) * size] for i in range(n))


def decode_raw(video: Path, fps: float, start_f: int, n_want: int, w: int, h: int, what: str,
               min_frames: int = 1) -> "tuple[bytes, int]":
    """decode_frames without splitting: (ffmpeg's whole RGB24 output, number of whole frames in it), so a
    caller can view it as one array instead of copying every frame."""
    cmd = [_tool("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error", "-protocol_whitelist", "file",
           "-ss", f"{max(0.0, (start_f - 0.5) / fps):.6f}", "-i", str(video),
           "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(n_want), "-fps_mode", "passthrough",
           "-vf", f"scale={w}:{h}:flags=bilinear", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Refused(f"ffmpeg took over {FFMPEG_TIMEOUT_S} s to decode {what}") from None
    size = w * h * 3
    n = min(len(done.stdout) // size, n_want)
    if done.returncode != 0 or n < min_frames:
        raise Refused(f"ffmpeg could not decode {what}")
    return done.stdout, n


# ---- one labelling session ------------------------------------------------------------------------
class Session:
    """Moves through the rows, saves after every answer, and times itself. Time counts from the start of
    the session; a gap of more than IDLE_CAP_S between two key presses counts as IDLE_CAP_S (a break is
    not labelling time). The total is added to the minutes already in the file, so resuming adds up."""

    def __init__(self, lf: LabelFile, clock=time.monotonic):
        self.lf = lf
        self.clock = clock
        self.prior = lf.prior_minutes()
        self.active_s = 0.0
        self._last = clock()
        first = lf.first_empty()
        self.i = 0 if first is None else first

    @property
    def n(self) -> int:
        return len(self.lf.values)

    def tick(self) -> None:
        now = self.clock()
        self.active_s += min(max(0.0, now - self._last), IDLE_CAP_S)
        self._last = now

    def minutes(self) -> float:
        return self.prior + self.active_s / 60.0

    def needs_description(self) -> bool:
        try:
            check_description(self.lf.head("A"))
            return False
        except Refused:
            return True

    def describe(self, colours: str) -> None:
        self.lf.set_description(description_from_colours(colours))   # refuses before anything is written
        self.save()

    def save(self) -> None:
        self.tick()
        m = round(self.minutes(), 2)
        if m > 0:
            self.lf.set_minutes(min(m, MAX_MINUTES))
        self.lf.save()

    def label(self, value: str) -> None:
        """Answer the current row, save, then move on: to the next row, or after the last one to the first
        row still empty. If the save fails the row stays current (the answer is kept in memory and written
        with the next successful save)."""
        self.lf.set_value(self.i, value)
        self.save()
        if self.i + 1 < self.n:
            self.i += 1
        else:
            first = self.lf.first_empty()
            if first is not None:
                self.i = first

    def back(self) -> None:
        self.tick()
        self.i = max(0, self.i - 1)

    def forward(self) -> None:
        self.tick()
        self.i = min(self.n - 1, self.i + 1)

    def to_first_empty(self) -> None:
        self.tick()
        first = self.lf.first_empty()
        if first is not None:
            self.i = first

    def done(self) -> bool:
        return self.lf.first_empty() is None and not self.needs_description()


KEYS = {
    "l": "L", "L": "L", "r": "R", "R": "R", "question": "?", "slash": "?",
    "BackSpace": "back", "Left": "back", "b": "back", "B": "back",
    "Right": "next", "n": "next", "N": "next",
    "space": "replay", "e": "first_empty", "E": "first_empty",
    "q": "quit", "Q": "quit", "Escape": "quit",
}


def action_for(keysym: str):
    return KEYS.get(keysym)
