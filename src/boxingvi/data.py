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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


SKELETON_READERS = {".json": _read_alphapose_json, ".npy": lambda p: np.load(p)}


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

    arr = np.load(skel / f"{video}.npy", allow_pickle=False)
    df = pd.read_excel(ann / f"{video}.xlsx")

    if arr.ndim != 4 or arr.shape[1:3] != (MAX_FRAMES, N_JOINTS):
        raise ValueError(f"{video}: expected (N, {MAX_FRAMES}, {N_JOINTS}, C), "
                         f"got {arr.shape}")
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

    arr = np.load(skel / "V6.npy", allow_pickle=False)
    if arr.shape[1:] != (1, N_JOINTS, 3):
        raise ValueError(f"V6: expected (F, 1, {N_JOINTS}, 3), got {arr.shape}")
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
