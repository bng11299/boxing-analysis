"""Tests for the BoxingVI baseline pipeline.

Weighted toward the trust boundaries -- the places where data of unknown
quality enters the process. On this project that is the skeleton loader and the
feature extractor: AlphaPose output contains dropped joints, zero-confidence
frames and occasional garbage, and the corpus itself is third-party. The happy
path is cheap to test and rarely where the failures are.

Run:  python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from boxingvi import data, features, synthetic  # noqa: E402
from boxingvi.data import MAX_FRAMES, N_JOINTS, Clip  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc:
        check(name, True)
    except Exception as e:  # wrong exception type is still a failure
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "no exception raised")


def kp(t: int = 10, fill: float = 0.5) -> np.ndarray:
    a = np.full((t, N_JOINTS, 3), fill, dtype=np.float32)
    a[..., 2] = 1.0
    return a


print("--- 1. Clip validation rejects malformed input ---")
raises("wrong joint count", lambda: Clip(np.zeros((5, 12, 3), np.float32), 0, "S1"))
raises("wrong channel count", lambda: Clip(np.zeros((5, N_JOINTS, 2), np.float32), 0, "S1"))
raises("zero frames", lambda: Clip(np.zeros((0, N_JOINTS, 3), np.float32), 0, "S1"))
raises("label out of range", lambda: Clip(kp(), 99, "S1"))
raises("negative label", lambda: Clip(kp(), -1, "S1"))
check("well-formed clip accepted", Clip(kp(), 3, "S1").label == 3)

print("\n--- 2. Padding matches the corpus convention ---")
short = data.pad_or_trim(kp(9))
check("short clip padded to 25", short.shape[0] == MAX_FRAMES)
check("padding is zero, not edge-repeat", short[10].sum() == 0.0,
      "repeated final pose would be learned as a held stance")
check("padded confidence is zero", short[10, :, 2].sum() == 0.0)
check("long clip trimmed", data.pad_or_trim(kp(40)).shape[0] == MAX_FRAMES)
check("exact-length clip untouched",
      np.array_equal(data.pad_or_trim(kp(MAX_FRAMES)), kp(MAX_FRAMES)))

print("\n--- 3. The official split is subject-disjoint ---")
corpus = synthetic.make_corpus(n_train=120, n_val=30, seed=1)
tr, va = data.split_by_subject(corpus)
tr_s, va_s = {c.subject for c in tr}, {c.subject for c in va}
check("no subject appears in both halves", tr_s.isdisjoint(va_s),
      f"overlap: {tr_s & va_s}")
check("train subjects are S1-S15", tr_s <= set(data.TRAIN_SUBJECTS))
check("val subjects are S16-S20", va_s <= set(data.VAL_SUBJECTS))
check("no clips dropped", len(tr) + len(va) == len(corpus))
raises("unknown subject id is an error, not a silent drop",
       lambda: data.split_by_subject([Clip(kp(), 0, "S99")]))

print("\n--- 4. Feature extraction fails loudly on bad input ---")
raises("too few valid frames", lambda: features.clip_features(kp(1)))
raises("wrong shape", lambda: features.clip_features(np.zeros((5, 3), np.float32)))
nan_clip = kp(10)
nan_clip[3, 9, 0] = np.nan
raises("NaN in keypoints does not reach the feature matrix",
       lambda: features.clip_features(nan_clip))
inf_clip = kp(10)
inf_clip[2, 10, 1] = np.inf
raises("inf in keypoints does not reach the feature matrix",
       lambda: features.clip_features(inf_clip))

print("\n--- 5. Features are finite and ordered on real-shaped input ---")
c = synthetic.make_clip(0, "S1", np.random.default_rng(0))
f = features.clip_features(c.keypoints)
check("all features finite", all(np.isfinite(v) for v in f.values()))
check("feature names stable", sorted(f) == features.FEATURE_NAMES)
m = features.feature_matrix([x.keypoints for x in corpus[:20]])
check("matrix shape", m.shape == (20, len(features.FEATURE_NAMES)), str(m.shape))
check("matrix finite", bool(np.isfinite(m).all()))

print("\n--- 6. Features are scale-invariant (camera distance must not matter) ---")
base = synthetic.make_clip(2, "S3", np.random.default_rng(7))
scaled = base.keypoints.copy()
scaled[..., :2] *= 1.6  # same action, athlete further from / closer to camera
fa, fb = features.clip_features(base.keypoints), features.clip_features(scaled)
ratios = [abs(fa[k] - fb[k]) for k in fa
          if k.endswith(("_reach_max", "_dx", "_dy", "_speed_max"))]
check("distance features survive a 1.6x rescale", max(ratios) < 1e-3,
      f"max drift {max(ratios):.2e} -- torso normalisation is not working")

print("\n--- 7. Synthetic classes are actually separable (fixture sanity) ---")
prof_jab = synthetic._profile("jab", 25)
prof_hook = synthetic._profile("lead_hook", 25)
check("profiles finite", bool(np.isfinite(prof_jab).all() and np.isfinite(prof_hook).all()),
      "float32 pi makes sin(pi*1.0) negative; a fractional power then yields NaN")
check("profiles are non-negative", bool((prof_jab >= 0).all()))
check("straight and round profiles differ", float(np.abs(prof_jab - prof_hook).max()) > 0.05)

print("\n--- 8. Model shapes and the static baseline ---")
import torch  # noqa: E402
from boxingvi.stgcn import STGCN, StaticPoseBaseline, normalised_adjacency  # noqa: E402

adj = normalised_adjacency()
check("adjacency is square", adj.shape == (N_JOINTS, N_JOINTS))
check("adjacency is symmetric", bool(torch.allclose(adj, adj.T)))
check("adjacency has self-loops", bool((adj.diagonal() > 0).all()))
x = torch.randn(4, MAX_FRAMES, N_JOINTS, 3)
check("st-gcn output shape", STGCN()(x).shape == (4, data.N_CLASSES))
check("static baseline output shape", StaticPoseBaseline()(x).shape == (4, data.N_CLASSES))
check("st-gcn is small enough for CPU", STGCN().n_params() < 2_000_000,
      f"{STGCN().n_params():,} params")

print("\n--- 9. Layout discovery reports rather than guesses ---")
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    (root / "Skeleton_data").mkdir()
    (root / "Annotation_files").mkdir()
    (root / "Skeleton_data" / "a.json").write_text("[]", encoding="utf-8")
    (root / "Skeleton_data" / "b.pkl").write_bytes(b"\x00")
    (root / "Annotation_files" / "v1.xlsx").write_bytes(b"\x00")
    rep = data.discover_layout(root)
    check("counts json", rep["skeleton_files_by_ext"].get(".json") == 1)
    check("lists annotation files", rep["annotation_files"] == ["v1.xlsx"])
    check("flags formats it cannot read", rep["unreadable_exts"] == [".pkl"],
          str(rep["unreadable_exts"]))
    raises("missing Skeleton_data is an error",
           lambda: data.discover_layout(root / "nope"), FileNotFoundError)

print("\n--- 10. AlphaPose reader rejects malformed frames ---")
with tempfile.TemporaryDirectory() as d:
    good = Path(d) / "good.json"
    good.write_text(json.dumps([{"keypoints": [0.1] * (N_JOINTS * 3)}] * 3), encoding="utf-8")
    check("reads well-formed json", data._read_alphapose_json(good).shape == (3, N_JOINTS, 3))
    bad = Path(d) / "bad.json"
    bad.write_text(json.dumps([{"keypoints": [0.1] * 30}]), encoding="utf-8")
    raises("wrong keypoint count rejected", lambda: data._read_alphapose_json(bad))
    empty = Path(d) / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    raises("empty file rejected", lambda: data._read_alphapose_json(empty))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
