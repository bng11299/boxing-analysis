"""Hand-engineered temporal features -- the interpretable arm.

Every feature here is nameable, which is the entire point: when this arm
misclassifies a rear hook as a cross, the responsible feature can be pointed at.
That property is what the accuracy gap against the learned arm is buying, and
what the proposal's interpretability-cost measurement prices.
"""

from __future__ import annotations

import numpy as np

from .data import N_JOINTS

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12


def _valid(kp: np.ndarray) -> np.ndarray:
    """Frames with any confidence -- padding is all zeros, so it drops out."""
    return kp[..., 2].sum(axis=-1) > 0


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b, in radians, per frame."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1) + 1e-8
    n2 = np.linalg.norm(v2, axis=-1) + 1e-8
    cos = np.clip((v1 * v2).sum(-1) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos)


def clip_features(kp: np.ndarray) -> dict[str, float]:
    """Features for one (T, 17, 3) clip. Returns a name -> value mapping."""
    if kp.ndim != 3 or kp.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"expected (T, {N_JOINTS}, 3), got {kp.shape}")
    m = _valid(kp)
    if m.sum() < 2:
        raise ValueError("clip has fewer than 2 valid frames")
    xy = kp[m][..., :2]
    t = xy.shape[0]

    # Torso scale normalises out camera distance; without it, every distance
    # feature also encodes how far away the athlete was standing.
    shoulder_mid = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    scale = np.linalg.norm(shoulder_mid - hip_mid, axis=-1).mean() + 1e-6

    f: dict[str, float] = {}
    for side, wr, el, sh in (("lead", L_WR, L_EL, L_SH), ("rear", R_WR, R_EL, R_SH)):
        reach = np.linalg.norm(xy[:, wr] - shoulder_mid, axis=-1) / scale
        vel = np.diff(xy[:, wr], axis=0) / scale
        speed = np.linalg.norm(vel, axis=-1)
        acc = np.diff(speed) if t > 2 else np.zeros(1)
        elbow = _angle(xy[:, sh], xy[:, el], xy[:, wr])

        f[f"{side}_reach_max"] = float(reach.max())
        f[f"{side}_reach_range"] = float(reach.max() - reach.min())
        # Where in the clip the punch peaks: straight punches peak later.
        f[f"{side}_peak_frac"] = float(reach.argmax() / max(t - 1, 1))
        f[f"{side}_speed_max"] = float(speed.max())
        f[f"{side}_speed_mean"] = float(speed.mean())
        f[f"{side}_accel_max"] = float(np.abs(acc).max())
        f[f"{side}_elbow_min"] = float(elbow.min())
        f[f"{side}_elbow_range"] = float(elbow.max() - elbow.min())
        # Horizontal vs vertical travel separates hooks from uppercuts.
        span = xy[:, wr].max(0) - xy[:, wr].min(0)
        f[f"{side}_dx"] = float(span[0] / scale)
        f[f"{side}_dy"] = float(span[1] / scale)
        f[f"{side}_dy_over_dx"] = float((span[1] + 1e-6) / (span[0] + 1e-6))

    # Which hand actually moved -- the lead/rear half of every class label.
    lead_d = float(np.linalg.norm(np.diff(xy[:, L_WR], axis=0), axis=-1).sum())
    rear_d = float(np.linalg.norm(np.diff(xy[:, R_WR], axis=0), axis=-1).sum())
    f["hand_dominance"] = (rear_d - lead_d) / (rear_d + lead_d + 1e-6)
    f["shoulder_rotation"] = float(
        np.ptp(np.arctan2(*(xy[:, R_SH] - xy[:, L_SH]).T[::-1]))
    )
    f["n_frames"] = float(t)

    # Fail at the boundary rather than downstream. Real AlphaPose output
    # contains dropped joints and zero-confidence frames, and a single NaN
    # propagates silently into the feature matrix, where sklearn reports it as
    # an unhelpful error about the estimator rather than about the clip. This
    # already caught a fixture bug once -- a negative base under a fractional
    # power -- which is exactly the kind of thing that reaches a result.
    bad = sorted(k for k, v in f.items() if not np.isfinite(v))
    if bad:
        raise ValueError(f"non-finite features: {bad}")
    return f


FEATURE_NAMES = sorted(clip_features(
    np.concatenate([np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (N_JOINTS, 1))[None].repeat(4, 0),
                    np.ones((4, N_JOINTS, 1), dtype=np.float32)], axis=2)
).keys())


def feature_matrix(clips_kp: list[np.ndarray]) -> np.ndarray:
    """(N, F) matrix in a stable column order."""
    rows = []
    for kp in clips_kp:
        d = clip_features(kp)
        rows.append([d[n] for n in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float32)
