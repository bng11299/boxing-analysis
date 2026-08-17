"""Synthetic punches, so the pipeline can be validated before the real corpus arrives.

This is a test fixture, not a simulator. It generates keypoint sequences with
crude but *class-dependent* kinematics, so a working model should separate them
and a broken one should not. Its only job is to make the difference between
"the pipeline runs" and "the pipeline learns" observable.

Nothing here should ever appear in a reported result. `run_baseline.py` refuses
to write synthetic numbers into the results file without an explicit flag.
"""

from __future__ import annotations

import numpy as np

from .data import CLASSES, MAX_FRAMES, N_JOINTS, Clip, TRAIN_SUBJECTS, VAL_SUBJECTS

# Rough normalised rest pose (x, y in [0,1], y down), COCO-17 order.
REST = np.array([
    [0.50, 0.12], [0.48, 0.10], [0.52, 0.10], [0.46, 0.11], [0.54, 0.11],
    [0.42, 0.25], [0.58, 0.25], [0.38, 0.38], [0.62, 0.38],
    [0.40, 0.48], [0.60, 0.48], [0.45, 0.52], [0.55, 0.52],
    [0.44, 0.72], [0.56, 0.72], [0.43, 0.92], [0.57, 0.92],
], dtype=np.float32)

# Per class: (which wrist, target offset from rest, elbow lift).
# Lead = left (index 9), rear = right (index 10) for an orthodox stance.
_SPEC = {
    "jab":           (9,  np.array([0.00, -0.34]), 0.10),
    "cross":         (10, np.array([-0.02, -0.36]), 0.12),
    "lead_hook":     (9,  np.array([0.22, -0.20]), 0.20),
    "rear_hook":     (10, np.array([-0.22, -0.20]), 0.20),
    "lead_uppercut": (9,  np.array([0.06, -0.30]), 0.26),
    "rear_uppercut": (10, np.array([-0.06, -0.30]), 0.26),
}


def _profile(kind: str, t: int) -> np.ndarray:
    """Extension over time, normalised to peak 1.0.

    Straight punches (jab, cross) peak late and snap back; hooks and uppercuts
    are rounder. A feint would be this curve truncated before its peak -- not
    generated here, but the shape is why the real question is answerable.
    """
    u = np.linspace(0.0, 1.0, t, dtype=np.float32)
    # Clip before the fractional power. float32 pi is marginally larger than
    # true pi, so sin(pi*1.0) lands at about -8.7e-8 rather than +0, and a
    # negative base raised to a non-integer exponent is NaN. Silent, and it
    # poisoned every downstream feature until the fixture was checked.
    s = np.clip(np.sin(np.pi * u), 0.0, None)
    return s ** 1.6 if kind in ("jab", "cross") else s


def make_clip(label: int, subject: str, rng: np.random.Generator,
              n_frames: int | None = None) -> Clip:
    name = CLASSES[label]
    wrist, offset, lift = _SPEC[name]
    t = int(n_frames or rng.integers(14, MAX_FRAMES + 1))

    kp = np.repeat(REST[None, :, :], t, axis=0).copy()
    # Per-subject body-size and stance jitter, so subjects are not identical.
    subj_seed = int(subject[1:])
    kp *= 1.0 + 0.04 * ((subj_seed % 7) - 3) / 3.0
    kp += rng.normal(0, 0.004, kp.shape).astype(np.float32)

    prof = _profile(name, t)
    elbow = wrist - 2  # 9->7, 10->8
    for i in range(t):
        kp[i, wrist] += offset * prof[i]
        kp[i, elbow] += offset * prof[i] * 0.45
        kp[i, elbow, 1] -= lift * prof[i]
        # Shoulders rotate into the punch.
        kp[i, 5 if wrist == 10 else 6, 0] += 0.03 * prof[i] * (1 if wrist == 10 else -1)

    kp += rng.normal(0, 0.006, kp.shape).astype(np.float32)
    conf = np.clip(rng.normal(0.85, 0.08, (t, N_JOINTS, 1)), 0.0, 1.0).astype(np.float32)
    return Clip(keypoints=np.concatenate([kp, conf], axis=2), label=label,
                subject=subject, source="synthetic")


def make_corpus(n_train: int = 900, n_val: int = 240, seed: int = 0) -> list[Clip]:
    """Class-balanced, subject-disjoint, shaped like the real split."""
    rng = np.random.default_rng(seed)
    clips: list[Clip] = []
    for subjects, n in ((TRAIN_SUBJECTS, n_train), (VAL_SUBJECTS, n_val)):
        for i in range(n):
            clips.append(make_clip(i % len(CLASSES),
                                   subjects[i % len(subjects)], rng))
    return clips
