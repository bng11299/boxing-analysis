"""How much does the classifier lose when its skeletons come from YOLO11-pose instead
of the AlphaPose it was trained on? Measured on the benchmark clips themselves.

The held-out videos (V2/V5/V9) ship as AlphaPose clips with labelled spans, and
their source videos are downloaded. Running YOLO11-pose on the same videos gives
a second skeleton for every held-out punch, over the same frames, with the same
label. The same saved classifier reads both; the difference is the cost of the
pose-model switch, with nothing else changed.

Before anything is scored, two things are checked rather than assumed:
* alignment: the spans' frame numbers must point at the same frames of the
  downloaded video that YOLO numbered. The AlphaPose and YOLO skeletons are
  compared at offsets around zero; the best offset must be clearly better than
  its neighbours (the V6 check, O54, done on skeletons instead of motion energy).
* identity: YOLO keeps one person per frame (the biggest, with continuity); a
  coaching video can show two. A clip whose YOLO skeleton is far from the
  AlphaPose one at the aligned frames is a different person. Such clips are
  reported and scored separately, never silently dropped, because dropping them
  would hide a cost the pipeline really pays on footage with other people in it.

Only clips whose span length equals their real frame count are used (all but
5 of 979 on the held-out side); the others cannot be placed in the video.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import (CLASSES, MAX_FRAMES, N_JOINTS, _class_column, _normalise_label, _numeric_columns_left_of,
                   _with_validity_channel, pad_or_trim)
from .pose_yolo import PoseSequence
from .punchfind import L_HIP, L_SHOULDER, MIN_CONF, R_HIP, R_SHOULDER

OFFSETS = tuple(range(-12, 13))
# The aligned offset must beat the best offset >= 3 frames away by this factor.
SHARPNESS = 1.5
# Same person: median joint distance under this many AlphaPose torso lengths.
IDENTITY_TORSO = 0.5


@dataclass(frozen=True)
class HeldOutClip:
    video: str
    index: int            # row in the video's annotation file
    start: int            # inclusive video frame ids, as annotated
    end: int
    label: int
    alphapose: np.ndarray  # (25, 17, 3) normalised, validity in channel 2, as the corpus loader builds it


def held_out_clips(data: Path, video: str) -> "tuple[list[HeldOutClip], int]":
    """The video's labelled clips whose span length equals their real frame
    count, and how many labelled clips were left out for not matching."""
    import pandas as pd

    arr = np.load(Path(data) / "Skeleton_data" / f"{video}.npy", allow_pickle=False)
    df = pd.read_excel(Path(data) / "Annotation_files" / f"{video}.xlsx")
    if arr.ndim != 4 or arr.shape[1:3] != (MAX_FRAMES, N_JOINTS) or arr.shape[0] != len(df):
        raise ValueError(f"{video}: skeleton {arr.shape} does not pair with {len(df)} annotation rows")
    cc = _class_column(df)
    sc, ec = _numeric_columns_left_of(df, cc)
    out, mismatched = [], 0
    for i in range(len(df)):
        name = _normalise_label(df[cc].iloc[i])
        if name is None:
            continue
        s, e = pd.to_numeric(df[sc].iloc[i], errors="coerce"), pd.to_numeric(df[ec].iloc[i], errors="coerce")
        kp = _with_validity_channel(arr[i])
        real = int((kp[..., 2] > 0).any(axis=1).sum())
        if not (np.isfinite(s) and np.isfinite(e) and s == int(s) and e == int(e) and s >= 0
                and int(e) - int(s) + 1 == real and real >= 2):
            mismatched += 1
            continue
        # The corpus pads AFTER the real frames; anything else breaks frame-for-frame pairing.
        if not (kp[:real, :, 2] > 0).any(axis=1).all():
            mismatched += 1
            continue
        out.append(HeldOutClip(video, i, int(s), int(e), CLASSES.index(name), kp))
    return out, mismatched


def yolo_clip(pose: PoseSequence, start: int, end: int, min_conf: "float | None" = MIN_CONF) -> np.ndarray:
    """YOLO11-pose over video frames start..end, in the corpus's shape and space:
    (25, 17, 3), x / width and y / height, validity in channel 2, zero padding.
    A joint is valid if its frame was detected and (with `min_conf`) it is at least
    that confident; `min_conf=None` keeps every detected joint, as AlphaPose does."""
    n = end - start + 1
    out = np.zeros((n, N_JOINTS, 3), dtype=np.float32)
    lo, hi = max(start, 0), min(end, len(pose) - 1)
    if hi >= lo:
        f = np.arange(lo, hi + 1)
        ok = pose.detected[f][:, None] & (pose.conf[f] >= (min_conf if min_conf is not None else 0.0))
        if min_conf is None:
            ok &= pose.conf[f] > 0
        xy = np.nan_to_num(pose.xy[f])
        rows = f - start
        out[rows, :, 0] = np.where(ok, xy[..., 0] / pose.width, 0)
        out[rows, :, 1] = np.where(ok, xy[..., 1] / pose.height, 0)
        out[rows, :, 2] = ok
    return pad_or_trim(out)


def _torso(kp: np.ndarray) -> float:
    m = kp[..., 2] > 0
    rows = m[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all(axis=1)
    if not rows.any():
        return float("nan")
    sh = (kp[rows, L_SHOULDER, :2] + kp[rows, R_SHOULDER, :2]) / 2
    hp = (kp[rows, L_HIP, :2] + kp[rows, R_HIP, :2]) / 2
    return float(np.median(np.linalg.norm((sh - hp) * [16 / 9, 1], axis=1)))


def joint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Median distance over joints valid in both, in frame heights (x scaled by
    16:9 so both axes are the same length). NaN if no joint is shared."""
    m = (a[..., 2] > 0) & (b[..., 2] > 0)
    if not m.any():
        return float("nan")
    d = (a[..., :2] - b[..., :2]) * [16 / 9, 1]
    return float(np.median(np.linalg.norm(d[m], axis=-1)))


def align(pose: PoseSequence, clips: "list[HeldOutClip]", offsets=OFFSETS) -> dict:
    """Median over clips of the AlphaPose-YOLO joint distance at each frame offset."""
    curve = {}
    for o in offsets:
        d = [joint_distance(c.alphapose, yolo_clip(pose, c.start + o, c.end + o, None)) for c in clips]
        d = np.array(d)
        curve[o] = float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")
    finite = {o: v for o, v in curve.items() if np.isfinite(v)}
    curve = {o: (v if np.isfinite(v) else None) for o, v in curve.items()}   # JSON-safe
    if not finite:
        return {"curve": curve, "best": None, "sharp": False, "ratio": None}
    best = min(finite, key=finite.get)
    far = [v for o, v in finite.items() if abs(o - best) >= 3]
    ratio = (min(far) / max(finite[best], 1e-9)) if far else None   # an exact match is very sharp, not undefined
    return {"curve": curve, "best": best, "ratio": ratio,
            "sharp": ratio is not None and ratio >= SHARPNESS}


def same_person(c: HeldOutClip, yolo: np.ndarray) -> "bool | None":
    """None when the AlphaPose skeleton has no torso or the two share no joint."""
    t, d = _torso(c.alphapose), joint_distance(c.alphapose, yolo)
    if not (np.isfinite(t) and t > 0 and np.isfinite(d)):
        return None
    return d / t < IDENTITY_TORSO
