"""Loading BoxingVI skeletons, and a synthetic generator that mimics its format.

The real corpus is not redistributable, so the format below is transcribed from
the paper and the repository README rather than from the files themselves:

    Dataset/
      Annotation_files/   one .xlsx per source video; rows are
                          (start_frame, end_frame, punch_class)
      Skeleton_data/      AlphaPose output, COCO 17-keypoint format, 2D,
                          normalised by frame width/height, padded to 25 frames
      RGB_videos/         not needed for the keypoint baseline

    The paper and README describe a SUBJECT-DISJOINT official split -- S1-S15
    train (5,513 clips), S16-S20 validation (1,402 clips). VERIFIED 2026-09-02:
    that split is NOT reproducible from the release, which carries no subject
    identifier of any kind. `split_by_video` is the stand-in; see the
    real-corpus section at the foot of this file. The prohibition still binds:
    never fall back to a random clip-level split, which leaks by construction.

LAYOUT: verified 2026-09-02 by downloading the corpus. What arrived differs
from the description above in four ways that matter -- ten videos not twenty,
no subject ids, two channels not three, and one video (V6) shipped as raw
per-frame pose rather than clips. The header block above is the *documented*
format; the real-corpus section at the foot of this file records the *actual*
one, and is the only place that assumption now lives. `discover_layout` still
reports rather than guesses, and remains the first thing to run on new data.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .npyread import read_data, read_header

# COCO-17, the order AlphaPose emits.
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
N_JOINTS = len(KEYPOINTS)

# Bone list, used to build the ST-GCN adjacency.
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

CLASSES = [
    "cross", "jab", "lead_hook", "lead_uppercut", "rear_hook", "rear_uppercut",
]
N_CLASSES = len(CLASSES)

MAX_FRAMES = 25  # the corpus pads to this; see README

TRAIN_SUBJECTS = [f"S{i}" for i in range(1, 16)]
VAL_SUBJECTS = [f"S{i}" for i in range(16, 21)]


@dataclass
class Clip:
    """One punch. `keypoints` is (T, 17, 3) as x, y, confidence."""

    keypoints: np.ndarray
    label: int
    subject: str
    source: str = ""

    def __post_init__(self) -> None:
        if self.keypoints.ndim != 3 or self.keypoints.shape[1:] != (N_JOINTS, 3):
            raise ValueError(
                f"expected (T, {N_JOINTS}, 3), got {self.keypoints.shape}"
            )
        if self.keypoints.shape[0] == 0:
            raise ValueError("clip has zero frames")
        if not 0 <= self.label < N_CLASSES:
            raise ValueError(f"label {self.label} outside 0..{N_CLASSES - 1}")


def pad_or_trim(kp: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Match the corpus convention: zero-pad short clips, trim long ones.

    Padding with zeros rather than edge-repeat is deliberate -- a repeated final
    pose reads as "held still", which is a real posture and would be learned as
    signal. Zeros are distinguishable because confidence goes to zero too.
    """
    t = kp.shape[0]
    if t == max_frames:
        return kp
    if t > max_frames:
        return kp[:max_frames]
    pad = np.zeros((max_frames - t, N_JOINTS, 3), dtype=kp.dtype)
    return np.concatenate([kp, pad], axis=0)


def to_arrays(clips: list[Clip], max_frames: int = MAX_FRAMES):
    """Stack clips into (N, T, V, C) and (N,) label arrays."""
    if not clips:
        raise ValueError("no clips to stack")
    x = np.stack([pad_or_trim(c.keypoints, max_frames) for c in clips])
    y = np.array([c.label], dtype=np.int64) if len(clips) == 1 else np.array(
        [c.label for c in clips], dtype=np.int64
    )
    return x.astype(np.float32), y


def split_by_subject(clips: list[Clip]) -> tuple[list[Clip], list[Clip]]:
    """Apply the official subject-disjoint split.

    Any subject outside S1-S20 is an error rather than a silent drop: an
    unrecognised subject id means the loader misparsed something, and dropping
    it would shrink the corpus without saying so.
    """
    known = set(TRAIN_SUBJECTS) | set(VAL_SUBJECTS)
    unknown = sorted({c.subject for c in clips} - known)
    if unknown:
        if unknown == [""]:
            raise ValueError(
                "these clips carry no subject id, so the official "
                "S1-S15 / S16-S20 split cannot be applied -- the release does "
                "not ship subject identifiers (ISSUES-LOG O10). Use "
                "split_by_video() and report the split as video-disjoint."
            )
        raise ValueError(f"unrecognised subject ids: {unknown}")
    train = [c for c in clips if c.subject in set(TRAIN_SUBJECTS)]
    val = [c for c in clips if c.subject in set(VAL_SUBJECTS)]
    return train, val


# --------------------------------------------------------------------------
# Real-corpus loading
# --------------------------------------------------------------------------

def _read_alphapose_json(path: Path) -> np.ndarray:
    """AlphaPose writes a list of frames, each with a flat 51-value keypoint list."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    frames = raw if isinstance(raw, list) else raw.get("people", raw.get("frames", []))
    out = []
    for fr in frames:
        flat = fr.get("keypoints") if isinstance(fr, dict) else fr
        arr = np.asarray(flat, dtype=np.float32)
        if arr.size != N_JOINTS * 3:
            raise ValueError(
                f"{path.name}: expected {N_JOINTS * 3} values per frame, got {arr.size}"
            )
        out.append(arr.reshape(N_JOINTS, 3))
    if not out:
        raise ValueError(f"{path.name}: no frames")
    return np.stack(out)


# The two layouts the release ships, read from the headers of all ten files on
# 2026-10-20: nine (N, 25, 17, 2) clip arrays and V6's (46497, 1, 17, 3) per-frame
# pose, every one '<f8' in .npy format 1.0. Channels may be 2 or 3 in a clip array
# (`_with_validity_channel` takes either). Name -> (shape test, what it expects).
_NPY_LAYOUTS = {
    "clips": (lambda s: len(s) == 4 and s[1:3] == (MAX_FRAMES, N_JOINTS) and s[3] in (2, 3),
              f"(N, {MAX_FRAMES}, {N_JOINTS}, 2 or 3)"),
    "v6": (lambda s: len(s) == 4 and s[1:] == (1, N_JOINTS, 3), f"(F, 1, {N_JOINTS}, 3)"),
}


def load_corpus_npy(path: Path, layouts: "tuple[str, ...]") -> np.ndarray:
    """Load a corpus .npy, refusing it from its header before anything is allocated.

    The corpus is third-party data, so this is the trust boundary. `np.load`
    allocates whatever shape a header claims and only then finds the data missing:
    a 128-byte V6.npy claiming 2M frames cost 816 MB before a ValueError (measured
    2026-10-20). Here the header is read through a byte cap, then checked -- a plain
    float dtype of at most 8 bytes (never object, so nothing is ever unpickled),
    integer dims matching one of `layouts` (keys of `_NPY_LAYOUTS`), and promised
    bytes no more than the file holds after its header -- and only then is one
    buffer of exactly that size filled. A file that cannot be opened raises OSError;
    anything wrong inside it, ValueError.
    """
    p = Path(path)
    with open(p, "rb") as fh:
        shape, fortran, dtype = read_header(fh, p.name)
        if (dtype.kind != "f" or dtype.hasobject or dtype.fields is not None
                or dtype.subdtype is not None or dtype.itemsize > 8):
            raise ValueError(f"{p.name}: dtype {dtype.str!r}; corpus keypoints are "
                             "floats of at most 8 bytes")
        if (not all(type(d) is int and d >= 0 for d in shape)  # (True, ...) parses
                or not any(_NPY_LAYOUTS[k][0](shape) for k in layouts)):
            want = " or ".join(_NPY_LAYOUTS[k][1] for k in layouts)
            raise ValueError(f"{p.name}: expected {want}, got {shape}")
        need = dtype.itemsize * math.prod(shape)  # Python ints: cannot overflow
        room = os.fstat(fh.fileno()).st_size - fh.tell()
        if need > room:
            raise ValueError(f"{p.name}: header promises {need:,} bytes of data; "
                             f"the file holds {room:,} after its header")
        return read_data(fh, shape, fortran, dtype, p.name)


# A loose .npy under Skeleton_data is one of the two layouts the release ships.
SKELETON_READERS = {".json": _read_alphapose_json,
                    ".npy": lambda p: load_corpus_npy(p, ("clips", "v6"))}


def discover_layout(root: Path) -> dict:
    """Report what is actually on disk, instead of assuming.

    Returns counts by extension under Skeleton_data and the annotation files
    found. Call this first on the real corpus and read the output before
    trusting `load_corpus`.
    """
    root = Path(root)
    skel = root / "Skeleton_data"
    ann = root / "Annotation_files"
    if not skel.is_dir():
        raise FileNotFoundError(f"no Skeleton_data under {root}")
    exts: dict[str, int] = {}
    for p in skel.rglob("*"):
        if p.is_file():
            exts[p.suffix.lower()] = exts.get(p.suffix.lower(), 0) + 1
    return {
        "skeleton_files_by_ext": exts,
        "annotation_files": sorted(p.name for p in ann.glob("*.xlsx")) if ann.is_dir() else [],
        "unreadable_exts": sorted(set(exts) - set(SKELETON_READERS)),
    }


# --------------------------------------------------------------------------
# Real-corpus loading -- wired 2026-09-02 against the layout actually shipped
# --------------------------------------------------------------------------
#
# What the release contains, measured rather than assumed (see ISSUES-LOG O10):
#
#   * TEN videos, V1-V10 -- not the 20 the paper and README describe.
#   * NO subject identifiers anywhere. Not in filenames, not in any annotation
#     column, not in the arrays. The official S1-S15 / S16-S20 subject-disjoint
#     split is therefore NOT reproducible from the release. `split_by_subject`
#     is deliberately left to fail on real clips; use `split_by_video`.
#   * TWO channels per joint (x, y). The release ships no confidence value, so
#     the third channel this pipeline expects is synthesised below as an
#     explicit VALIDITY MASK -- 1.0 for an observed joint, 0.0 for one at
#     exactly (0, 0), which is how both dropped joints and the corpus's own
#     zero-padding are marked. It is a presence flag, not an AlphaPose score,
#     and must never be reported as one.
#   * Nine videos ship pre-segmented (N, 25, 17, 2) clip arrays whose first
#     axis aligns 1:1, in order, with the rows of the matching .xlsx. V6 is the
#     exception: (46497, 1, 17, 3) of raw per-frame pose, where channel 0 is a
#     video frame index (157, 158, ...) and channels 1-2 are x, y. V6 must be
#     re-segmented from its annotations, so it is OFF by default -- a headline
#     number should rest on the authors' segmentation, not on ours.
#   * Annotation schemas differ per file: header names vary, some are entirely
#     unnamed, some carry blank spacer columns. Only the class column is read
#     for the nine pre-segmented videos; their row order carries the alignment.

_LABEL_ALIASES = {
    "cross": "cross", "jab": "jab",
    "lead_hook": "lead_hook", "lead hook": "lead_hook",
    "lead_uppercut": "lead_uppercut", "lead uppercut": "lead_uppercut",
    "rear_hook": "rear_hook", "rear hook": "rear_hook",
    "rear_uppercut": "rear_uppercut", "rear uppercut": "rear_uppercut",
}

VIDEOS = tuple(f"V{i}" for i in range(1, 11))

# Held out for validation. Chosen ONCE, on label distribution alone and before
# any model was fit: 979 clips (20.5%, against the paper's 20.3% train/val
# ratio) with every one of the six classes represented at least 51 times. It is
# a video-disjoint split standing in for a subject-disjoint one, which is the
# closest approximation the release permits. It is NOT the official split and
# no number computed on it should be described as one.
HELD_OUT_VIDEOS = ("V2", "V5", "V9")


def _normalise_label(raw: object) -> "str | None":
    """Canonical class name, or None if the cell is not a punch label."""
    key = " ".join(str(raw).strip().lower().split())
    return _LABEL_ALIASES.get(key)


def _class_column(df):
    """The column holding punch names, found by content because headers vary."""
    for c in df.columns:
        hits = sum(_normalise_label(v) is not None for v in df[c])
        if hits > len(df) * 0.5:
            return c
    raise ValueError("no column in this annotation file holds punch labels")


def _numeric_columns_left_of(df, class_col) -> list:
    """The start/end frame columns: numeric, non-empty, left of the class column.

    Needed only for V6. Files interleave blank spacer columns between the two,
    so position alone is not enough -- emptiness has to be tested.
    """
    import pandas as pd
    out = []
    for c in df.columns:
        if c == class_col:
            break
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > len(df) * 0.5:
            out.append(c)
    if len(out) < 2:
        raise ValueError(f"expected two numeric frame columns before {class_col!r}")
    return out[:2]


def _with_validity_channel(kp: np.ndarray) -> np.ndarray:
    """Attach the third channel this pipeline expects to (T, 17, 2) input.

    The value is a VALIDITY MASK, not a confidence score: the release ships no
    confidence. A joint at exactly (0, 0) is absent -- that is how AlphaPose
    marks a dropped joint and how the corpus marks its zero-padding -- so the
    mask is 0.0 there and 1.0 elsewhere. This preserves exactly the property
    the rest of the code reads off channel 2 (padding drops out of `_valid`)
    without inventing per-joint certainty the data does not contain.
    """
    kp = np.asarray(kp, dtype=np.float32)
    if kp.ndim != 3 or kp.shape[1] != N_JOINTS:
        raise ValueError(f"expected (T, {N_JOINTS}, C), got {kp.shape}")
    if kp.shape[2] == 3:
        return kp
    if kp.shape[2] != 2:
        raise ValueError(f"expected 2 or 3 channels, got {kp.shape[2]}")
    present = (np.abs(kp).sum(axis=-1, keepdims=True) > 0).astype(np.float32)
    return np.concatenate([kp, present], axis=-1)


def _load_presegmented(video: str, skel: Path, ann: Path):
    """The nine videos the authors shipped already cut into 25-frame clips."""
    import pandas as pd

    arr = load_corpus_npy(skel / f"{video}.npy", ("clips",))
    df = pd.read_excel(ann / f"{video}.xlsx")

    # The alignment this loader depends on. If the counts ever diverge, the
    # row-to-clip correspondence is unknowable and guessing would silently
    # mislabel the whole video.
    if arr.shape[0] != len(df):
        raise ValueError(
            f"{video}: {arr.shape[0]} skeleton clips but {len(df)} annotation "
            "rows -- positional alignment cannot be assumed"
        )

    labels = [_normalise_label(v) for v in df[_class_column(df)]]
    clips, unlabelled, degenerate = [], 0, 0
    for i, name in enumerate(labels):
        if name is None:
            unlabelled += 1
            continue
        kp = _with_validity_channel(arr[i])
        if int((kp[..., 2].sum(axis=-1) > 0).sum()) < 2:
            degenerate += 1
            continue
        clips.append(Clip(keypoints=kp, label=CLASSES.index(name),
                          subject="", source=video))
    return clips, unlabelled, degenerate


def _load_v6(skel: Path, ann: Path):
    """V6 ships raw per-frame pose; rebuild clips from its annotation spans."""
    import pandas as pd

    arr = load_corpus_npy(skel / "V6.npy", ("v6",))
    frame_ids = arr[:, 0, 0, 0].astype(np.int64)  # channel 0 is a frame index
    xy = arr[:, 0, :, 1:3]                        # channels 1-2 are x, y
    index = {int(f): i for i, f in enumerate(frame_ids)}

    df = pd.read_excel(ann / "V6.xlsx")
    ccol = _class_column(df)
    scol, ecol = _numeric_columns_left_of(df, ccol)

    clips, unlabelled, degenerate = [], 0, 0
    for _, row in df.iterrows():
        name = _normalise_label(row[ccol])
        if name is None:
            unlabelled += 1
            continue
        s = pd.to_numeric(row[scol], errors="coerce")
        e = pd.to_numeric(row[ecol], errors="coerce")
        if not np.isfinite(s) or not np.isfinite(e) or e < s:
            unlabelled += 1
            continue
        rows = [index[f] for f in range(int(s), int(e) + 1) if f in index]
        if len(rows) < 2:
            degenerate += 1
            continue
        kp = pad_or_trim(_with_validity_channel(xy[rows]))
        clips.append(Clip(keypoints=kp, label=CLASSES.index(name),
                          subject="", source="V6"))
    return clips, unlabelled, degenerate


# --------------------------------------------------------------------------
# V6 as continuous footage -- for the punch finder, not the classifier
# --------------------------------------------------------------------------
#
# Measured 2026-09-27 against the downloaded source video (vault ISSUES-LOG
# O54), not assumed:
#
#   * V6.npy channel 0 is the 0-based index of the frame in the source video,
#     decoded at its NATIVE rate. Motion energy of the decoded video correlates
#     with skeleton motion sharply at lag 0 and ~5x lower at every other lag in
#     -6..+6 (exact values: results/punch_finder_v6.json). The 208 frames the array lacks against the video's 46,705 are
#     three gaps with no pose (0-156, 601-648, 46489-46491), not drift.
#   * V6.xlsx `start`/`end` are video frame ids in the same space, inclusive.
#     Read as skeleton ROW numbers instead, they would point 157 frames late
#     before the gap at 601-648 and 205 frames late after it, and the
#     wrist-speed peak around them vanishes.
#   * 804 rows, of which 685 carry a label; the other 119 are empty. Four
#     classes only -- lead/rear hook and lead/rear uppercut. V6 has no jabs or
#     crosses, so nothing measured on it says anything about straight punches.
#   * The labels end at frame 44,372; pose continues to 46,704. The unlabelled
#     tail must be excluded from any score, or every punch in it counts as a
#     false positive.

# Native frame rates from ffprobe on the downloaded sources (O54). The corpus
# is NOT 30 fps; anything time-based must be converted per video, from here.
NATIVE_FPS = {
    "V1": 24.0, "V2": 24000 / 1001, "V3": 24000 / 1001, "V4": 24000 / 1001,
    "V5": 30000 / 1001, "V6": 24000 / 1001, "V7": 30000 / 1001,
    "V8": 30000 / 1001, "V9": 24000 / 1001, "V10": 25.0,
}
SOURCE_SIZE = (1920, 1080)  # every source video, width x height


@dataclass
class Track:
    """Continuous single-person pose. `xy` is (F, 17, 2) in PIXELS.

    `frame_ids` are video frame indices, strictly increasing; a jump of more
    than one is a gap in the pose, and nothing may be differenced across it.
    `conf` is (F, 17) joint confidence in [0, 1] when the pose model gives one
    (YOLO11-pose), and None when the source has none (BoxingVI's V6).
    """

    frame_ids: np.ndarray
    xy: np.ndarray
    fps: float
    source: str = ""
    conf: "np.ndarray | None" = None

    def __post_init__(self) -> None:
        f = np.asarray(self.frame_ids)
        if f.ndim != 1 or len(f) < 2:
            raise ValueError("a track needs a 1-D frame-id array of length >= 2")
        if self.xy.shape != (len(f), N_JOINTS, 2):
            raise ValueError(f"expected xy ({len(f)}, {N_JOINTS}, 2), got {self.xy.shape}")
        if self.conf is not None and (
                self.conf.shape != (len(f), N_JOINTS)
                or not np.all((self.conf >= 0) & (self.conf <= 1))):  # NaN fails both
            raise ValueError(f"conf must be ({len(f)}, {N_JOINTS}) in [0, 1]")
        if not np.all(np.diff(f) > 0):
            raise ValueError("frame ids must be strictly increasing")
        if not np.all(np.isfinite(self.xy)):
            raise ValueError("track contains non-finite coordinates")
        if not (np.isfinite(self.fps) and self.fps > 0):
            raise ValueError(f"fps must be positive, got {self.fps!r}")

    def segments(self) -> "list[tuple[int, int]]":
        """[lo, hi) row ranges of consecutive frame ids."""
        breaks = np.flatnonzero(np.diff(self.frame_ids) != 1) + 1
        edges = np.r_[0, breaks, len(self.frame_ids)]
        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


@dataclass(frozen=True)
class Span:
    """One labelled punch: inclusive video frame ids."""

    start: int
    end: int
    label: int


def load_v6_track(skel: Path) -> Track:
    arr = load_corpus_npy(Path(skel) / "V6.npy", ("v6",))
    ids = arr[:, 0, :, 0]
    # Channel 0 is one frame id repeated over the 17 joints. If that ever stops
    # holding, it is not a frame id and the alignment above does not apply.
    if not np.all(ids == ids[:, :1]) or not np.all(ids[:, 0] == np.round(ids[:, 0])):
        raise ValueError("V6 channel 0 is not a per-frame integer frame id")
    if ids[0, 0] < 0:
        raise ValueError("V6 frame ids must be non-negative")
    xy = arr[:, 0, :, 1:3].astype(np.float64)
    # Elsewhere in this file (0, 0) marks a dropped joint. Read as a position it
    # would be a jump across the whole frame -- a fake punch. V6 has none today;
    # if a re-release does, it must be handled deliberately, not differenced.
    if np.any(np.all(xy == 0, axis=-1)):
        raise ValueError("V6 has joints at exactly (0, 0) -- dropped joints, "
                         "which the continuous loader does not handle")
    w, h = SOURCE_SIZE
    return Track(frame_ids=ids[:, 0].astype(np.int64),
                 xy=xy * np.array([w, h]), fps=NATIVE_FPS["V6"], source="V6")


def load_v6_spans(ann: Path) -> "list[Span]":
    """Labelled V6 punches, sorted by start. Empty rows are dropped, not errors;
    a row with a label but unusable frames IS an error, because silently losing
    it would shrink the ground truth and flatter recall."""
    import pandas as pd

    df = pd.read_excel(Path(ann) / "V6.xlsx")
    ccol = _class_column(df)
    scol, ecol = _numeric_columns_left_of(df, ccol)
    spans = []
    for i, row in df.iterrows():
        name = _normalise_label(row[ccol])
        s = pd.to_numeric(row[scol], errors="coerce")
        e = pd.to_numeric(row[ecol], errors="coerce")
        if name is None:
            if np.isfinite(s) or np.isfinite(e):
                raise ValueError(f"V6 row {i}: frames without a punch label")
            continue
        if (not (np.isfinite(s) and np.isfinite(e)) or s < 0 or e < s
                or s != int(s) or e != int(e)):
            raise ValueError(f"V6 row {i}: unusable span {s!r}..{e!r}")
        spans.append(Span(int(s), int(e), CLASSES.index(name)))
    if not spans:
        raise ValueError("V6: no labelled spans")
    return sorted(spans, key=lambda sp: (sp.start, sp.end))


def load_corpus(root: Path, include_v6: bool = False) -> "list[Clip]":
    """Load the released BoxingVI corpus into `Clip`s.

    `include_v6` re-segments V6 from its raw per-frame array. Off by default:
    that reconstruction is ours, not the authors', and the segment-boundary
    convention is unverified -- an off-by-one on the span would misalign every
    V6 clip while producing entirely plausible-looking output.
    """
    root = Path(root)
    skel, ann = root / "Skeleton_data", root / "Annotation_files"
    if not skel.is_dir() or not ann.is_dir():
        raise FileNotFoundError(
            f"need Skeleton_data/ and Annotation_files/ under {root}")

    clips: "list[Clip]" = []
    unlabelled = degenerate = 0
    for v in VIDEOS:
        if v == "V6":
            if not include_v6:
                continue
            c, u, d = _load_v6(skel, ann)
        else:
            c, u, d = _load_presegmented(v, skel, ann)
        clips.extend(c)
        unlabelled += u
        degenerate += d

    if not clips:
        raise ValueError(f"no usable clips found under {root}")
    n_videos = len(VIDEOS) if include_v6 else len(VIDEOS) - 1
    print(f"loaded {len(clips)} clips from {n_videos} videos "
          f"({unlabelled} rows without a punch label, {degenerate} clips with "
          "fewer than 2 observed frames -- both dropped)")
    return clips


def split_by_video(clips, held_out: tuple = HELD_OUT_VIDEOS):
    """Video-disjoint split, the stand-in for the unavailable official one.

    No clip from a held-out video appears in training. This is weaker than
    subject-disjoint -- if one athlete appears in two videos they straddle the
    fold -- but it is the strongest guarantee the release supports, and it is
    strictly better than a clip-level split, which leaks by construction.
    """
    present = {c.source for c in clips}
    missing = sorted(set(held_out) - present)
    if missing:
        raise ValueError(f"held-out videos absent from the corpus: {missing}")
    train = [c for c in clips if c.source not in set(held_out)]
    val = [c for c in clips if c.source in set(held_out)]
    if not train or not val:
        raise ValueError("split produced an empty side")
    return train, val


def _group_order(g: str) -> tuple:
    # V2 before V10: numeric where the id has a number, so the tie-break is the
    # order a person reading the corpus would expect, not string order.
    digits = "".join(ch for ch in g if ch.isdigit())
    return (int(digits) if digits else -1, g)


def group_folds(clips, k: int = 3, key=lambda c: c.source) -> list[tuple[str, ...]]:
    """Partition the groups (videos) into k folds for leave-groups-out CV.

    The assignment is LABEL-BLIND by construction: it reads only how many clips
    each group has. Groups go largest first to whichever fold currently holds
    the fewest clips, ties to the lowest fold. Nothing about labels or model
    scores can move a group between folds, so the folds cannot have been chosen
    to flatter a result -- the failure a hand-picked split invites.

    Why this exists: the single held-out split (HELD_OUT_VIDEOS) is also the
    set every design decision was checked against, so it is a validation set
    doubling as the test set. Cross-validating over every video puts each clip
    in a test fold exactly once, and most of those clips were never looked at.
    """
    sizes: dict[str, int] = {}
    for c in clips:
        g = key(c)
        if not g:
            raise ValueError("a clip has no group id; folds would silently "
                             "merge unrelated clips")
        sizes[g] = sizes.get(g, 0) + 1
    if not isinstance(k, int) or k < 2:
        raise ValueError(f"k must be an integer >= 2, got {k!r}")
    if k > len(sizes):
        raise ValueError(f"k={k} folds but only {len(sizes)} groups: "
                         "a fold would be empty")

    folds: list[list[str]] = [[] for _ in range(k)]
    load = [0] * k
    for g in sorted(sizes, key=lambda g: (-sizes[g], _group_order(g))):
        i = min(range(k), key=lambda j: (load[j], j))
        folds[i].append(g)
        load[i] += sizes[g]
    return [tuple(sorted(f, key=_group_order)) for f in folds]


def cv_splits(clips, k: int = 3, key=lambda c: c.source):
    """(held_out_groups, train, test) for each fold of `group_folds`.

    A group is never on both sides of a fold, and every clip lands in exactly
    one test side. A fold whose training side is missing a class is refused:
    that model could never predict the class, and its zero would be reported as
    a finding about the class rather than about the fold.
    """
    labels = {c.label for c in clips}
    out = []
    for held in group_folds(clips, k, key):
        held_set = set(held)
        train = [c for c in clips if key(c) not in held_set]
        test = [c for c in clips if key(c) in held_set]
        missing = sorted(labels - {c.label for c in train})
        if missing:
            raise ValueError(f"holding out {'/'.join(held)} leaves no training "
                             f"clips of class(es) {missing}")
        out.append((held, train, test))
    return out
