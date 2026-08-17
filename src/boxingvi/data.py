"""Loading BoxingVI skeletons, and a synthetic generator that mimics its format.

The real corpus is not redistributable, so the format below is transcribed from
the paper and the repository README rather than from the files themselves:

    Dataset/
      Annotation_files/   one .xlsx per source video; rows are
                          (start_frame, end_frame, punch_class)
      Skeleton_data/      AlphaPose output, COCO 17-keypoint format, 2D,
                          normalised by frame width/height, padded to 25 frames
      RGB_videos/         not needed for the keypoint baseline

    Split is SUBJECT-DISJOINT and official: S1-S15 train (5,513 clips),
    S16-S20 validation (1,402 clips). Do not re-split; a random clip-level
    split leaks athletes across folds and inflates every number.

LAYOUT ASSUMPTION: the exact on-disk arrangement inside Skeleton_data could not
be verified (the Drive folder requires sign-in). `discover_layout` probes the
common AlphaPose arrangements and reports what it found rather than guessing
silently. If it raises, adjust `SKELETON_READERS` -- that is the only place the
assumption lives.
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
