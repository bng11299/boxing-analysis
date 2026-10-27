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
carry no audio track (blurred copies carry none), and must be the video the CSV names. The CSV must lie
under `~/comps-participant-data/`, follow the template's format exactly, and have no row past the video's end.
Messages never repeat an input value or a path.
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

PARTICIPANT_ROOT = Path.home() / "comps-participant-data"
BLURRED_ROOT = PARTICIPANT_ROOT / "blurred"

LABEL_FORMAT = "glove-identity-labels-1"      # glovescore.LABEL_FORMAT
EVERY_S = 5                                   # glovescore.LABEL_EVERY_S
MAX_LABEL_BYTES = 1024 * 1024                 # glovescore.MAX_LABEL_BYTES
MAX_A_CHARS = 80                              # glovescore.MAX_A_CHARS
MAX_MINUTES = 600.0                           # glovescore.MAX_MINUTES
MAX_ROWS = 6 * 3600 // EVERY_S + 1            # glovescore.MAX_DURATION_S on the 5 s grid
VALUES = ("L", "R", "?")
HEADER_KEYS = ("format", "video_fp", "every_s", "A", "minutes")

IDLE_CAP_S = 120.0       # a gap longer than this between two key presses counts as this long
CLIP_HALF_S = 1.0        # the clip runs from t - 1 s to t + 1 s
MAX_DISPLAY_W = 960
FFMPEG_TIMEOUT_S = 120

_HEADER = re.compile(r"(#\s*([A-Za-z_]+)\s*:)(.*)")
_ROW = re.compile(r"(\d{1,6}),([LR?]?)")
_FP = re.compile(r"[0-9a-f]{64}")

# `# A:` says who boxer A is. Deny by default: every word must be on these lists, and at least one must be
# a colour, so A is tied to how he looks, never to a name or to a position that changes.
COLOUR_WORDS = frozenset("""
    red blue black white green yellow orange pink purple grey gray brown gold golden silver navy maroon
    teal cyan beige tan light dark bright pale neon""".split())
PLAIN_WORDS = frozenset("""
    a an the boxer boxers fighter person gloves glove in with wearing wears left right on of and or who is
    has at side mostly both hand hands colour color coloured colored trim trimmed stripe stripes striped
    shirt top tank vest shorts trunks headgear head guard taller shorter tall short bigger smaller larger
    no without one other""".split())
_A_CHARS = re.compile(r"[A-Za-z ,.()/-]+")


class Refused(ValueError):
    """An input fails a check; the message names the check, never the input."""


class SaveFailed(OSError):
    """The labels file could not be replaced (open in another program?); it is unchanged on disk."""


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


# ---- the A description ----------------------------------------------------------------------------
def check_description(text: str) -> str:
    """Return the description stripped, or refuse it. Names are refused by construction: only listed
    words pass. The message says which check failed and which word position, never the word."""
    s = (text or "").strip()
    if not s:
        raise Refused("describe boxer A, e.g. 'the boxer in red gloves'")
    if len(s) > MAX_A_CHARS:
        raise Refused(f"keep it to {MAX_A_CHARS} characters")
    if not _A_CHARS.fullmatch(s):
        raise Refused("use letters, spaces and , . ( ) / - only (no digits, no apostrophes)")
    words = [w for w in re.split(r"[ ,.()/-]+", s.lower()) if w]
    for n, w in enumerate(words, start=1):
        if w not in COLOUR_WORDS and w not in PLAIN_WORDS:
            raise Refused(f"word {n} is not on the allowed list (colours and plain words only, never a name)")
    if not any(w in COLOUR_WORDS for w in words):
        raise Refused("name at least one colour (e.g. 'red gloves'), so A is tied to how he looks")
    return s


# ---- the labels file ------------------------------------------------------------------------------
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
        hard link planted at the target is replaced, not written through."""
        data = self.to_bytes()
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".label_moments-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise SaveFailed("could not save the labels file (is it open in Excel or another program?)") \
                from None

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
    width: int
    height: int
    fps: float
    duration: float


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise Refused(f"{name} is not on PATH (install ffmpeg)")
    return exe


def probe(video: Path) -> VideoMeta:
    cmd = [_tool("ffprobe"), "-v", "error", "-show_entries",
           "stream=codec_type,width,height,avg_frame_rate,duration:format=duration", "-of", "json", str(video)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise Refused("ffprobe took over 60 s to read the video") from None
    if done.returncode != 0:
        raise Refused("ffprobe could not read the video")
    try:
        doc = json.loads(done.stdout)
        streams = doc.get("streams", [])
        if any(s.get("codec_type") == "audio" for s in streams):
            raise Refused("the video has an audio track; blurred copies carry none, so this is not one")
        vs = [s for s in streams if s.get("codec_type") == "video"]
        if len(vs) != 1:
            raise Refused("the video must have exactly one video stream")
        s = vs[0]
        w, h = int(s["width"]), int(s["height"])
        num, _, den = str(s["avg_frame_rate"]).partition("/")
        fps = int(num) / int(den or 1)
        dur = float(s.get("duration") or doc.get("format", {}).get("duration"))
    except Refused:
        raise
    except (KeyError, ValueError, TypeError, ZeroDivisionError, AttributeError):
        raise Refused("the video has no readable size, frame rate or duration") from None
    if not (0 < w and 0 < h and w * h <= MAX_PIXELS):
        raise Refused("the video's frame size is out of range")
    if not (math.isfinite(fps) and 1 <= fps <= 240 and math.isfinite(dur) and dur > 0):
        raise Refused("the video's frame rate or duration is out of range")
    return VideoMeta(w, h, fps, dur)


def check_pair(lf: LabelFile, video: Path, meta: VideoMeta) -> None:
    """The CSV must be the one made for this blurred video, and no row may lie past its end."""
    if fingerprint(video) != lf.head("video_fp"):
        raise Refused("this labels file was made for a different video (fingerprints differ)")
    last_frame = int(round(meta.duration * meta.fps)) - 1
    for k, t in enumerate(lf.times):
        if int(round(t * meta.fps)) > last_frame:
            raise Refused(f"row {k + 1} (t = {t} s) is past the end of the video ({meta.duration:.1f} s)")


@dataclass(frozen=True)
class Clip:
    t: int
    width: int
    height: int
    fps: float
    frames: tuple        # raw RGB24 bytes, one per frame
    centre: int          # index of the frame at time t


def display_size(meta: VideoMeta, max_w: int = MAX_DISPLAY_W) -> "tuple[int, int]":
    w = min(meta.width, max_w)
    w -= w % 2
    h = max(2, int(round(meta.height * w / meta.width / 2)) * 2)
    return w, h


def decode_clip(video: Path, meta: VideoMeta, t: int, half: float = CLIP_HALF_S) -> Clip:
    """Frames from t - half to t + half (clamped to the video), scaled for display. -ss before -i seeks to
    the exact frame (ffmpeg decodes from the keyframe and drops what comes before); passthrough timing
    keeps one output frame per stored frame, so frame k of the clip is stored frame start_frame + k."""
    start_f = max(0, int(round((t - half) * meta.fps)))
    t_f = int(round(t * meta.fps))
    last_f = int(round(meta.duration * meta.fps)) - 1
    end_f = min(last_f, int(round((t + half) * meta.fps)))
    n_want = end_f - start_f + 1
    w, h = display_size(meta)
    cmd = [_tool("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error",
           "-ss", f"{start_f / meta.fps:.6f}", "-i", str(video),
           "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(n_want), "-fps_mode", "passthrough",
           "-vf", f"scale={w}:{h}:flags=bilinear", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Refused(f"ffmpeg took over {FFMPEG_TIMEOUT_S} s to decode the clip at t = {t} s") from None
    size = w * h * 3
    n = len(done.stdout) // size
    if done.returncode != 0 or n == 0:
        raise Refused(f"ffmpeg could not decode the clip at t = {t} s")
    frames = tuple(done.stdout[i * size:(i + 1) * size] for i in range(n))
    return Clip(t, w, h, meta.fps, frames, min(n - 1, t_f - start_f))


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

    def describe(self, text: str) -> None:
        self.lf.set_description(text)       # refuses a name before anything is written
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
