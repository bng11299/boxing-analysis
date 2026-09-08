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


print("\n--- 11. Validity channel: a mask, not an invented confidence ---")
two = np.zeros((4, N_JOINTS, 2), dtype=np.float32)
two[:2] = 0.4                                    # two observed frames
got = data._with_validity_channel(two)           # frames 2-3 are padding
check("adds a third channel", got.shape == (4, N_JOINTS, 3))
check("observed joints marked 1.0", bool((got[:2, :, 2] == 1.0).all()))
check("padding marked 0.0", bool((got[2:, :, 2] == 0.0).all()))
check("coordinates untouched", bool(np.allclose(got[..., :2], two)))
three = np.ones((4, N_JOINTS, 3), dtype=np.float32)
check("3-channel input passes through", bool(np.allclose(data._with_validity_channel(three), three)))
raises("wrong joint count rejected",
       lambda: data._with_validity_channel(np.zeros((4, 5, 2), dtype=np.float32)))
raises("4 channels rejected",
       lambda: data._with_validity_channel(np.zeros((4, N_JOINTS, 4), dtype=np.float32)))
# The property the mask exists for: features._valid must drop the padding.
check("features._valid drops padded frames", int(features._valid(got).sum()) == 2)

print("\n--- 12. Label normalisation across the corpus's inconsistent spellings ---")
check("canonical passes", data._normalise_label("Cross") == "cross")
check("lowercase variant", data._normalise_label("cross") == "cross")
check("space becomes underscore", data._normalise_label("Lead Hook") == "lead_hook")
check("mixed case variant", data._normalise_label("Lead hook") == "lead_hook")
check("trailing space tolerated", data._normalise_label("Rear uppercut ") == "rear_uppercut")
check("non-label is None", data._normalise_label("nan") is None)
check("empty is None", data._normalise_label("") is None)
# An unknown punch name must NOT quietly become a class.
check("unknown label is None, not a guess", data._normalise_label("Overhand Right") is None)

import pandas as pd  # noqa: E402
unnamed = pd.DataFrame([[10, 20, "Jab"], [30, 40, "Cross"], [50, 60, "Rear Hook"]])
check("class column found by content, not header",
      data._class_column(unnamed) == unnamed.columns[2])
raises("a frame with no punch column is an error",
       lambda: data._class_column(pd.DataFrame([[1, 2], [3, 4]])))

print("\n--- 13. Splits cannot leak, and refuse when they cannot be trusted ---")
def _clip(src, label=0):
    kp = np.zeros((MAX_FRAMES, N_JOINTS, 3), dtype=np.float32)
    kp[:, :, :2] = 0.5
    kp[:, :, 2] = 1.0
    return Clip(keypoints=kp, label=label, subject="", source=src)

corpus = [_clip(v, i % data.N_CLASSES) for i, v in enumerate(
    ["V1"] * 5 + ["V2"] * 3 + ["V5"] * 3 + ["V9"] * 2 + ["V3"] * 4)]
tr, va = data.split_by_video(corpus)
check("held-out videos go to validation",
      {c.source for c in va} == set(data.HELD_OUT_VIDEOS))
check("training excludes them", {c.source for c in tr} == {"V1", "V3"})
check("no video straddles the split",
      not ({c.source for c in tr} & {c.source for c in va}))
check("every clip is accounted for", len(tr) + len(va) == len(corpus))
raises("absent held-out video is an error",
       lambda: data.split_by_video(corpus, held_out=("V99",)))
# The guard that stops anyone applying the official split it cannot support.
raises("subject split refuses clips with no subject id",
       lambda: data.split_by_subject(corpus))
try:
    data.split_by_subject(corpus)
except ValueError as e:
    check("and the refusal explains why", "O10" in str(e) and "split_by_video" in str(e),
          str(e)[:60])

print("\n--- 14. The static-pose floor reads the middle OBSERVED frame ---")
padded = torch.zeros(1, MAX_FRAMES, N_JOINTS, 3)
padded[0, :9, :, :2] = 0.4
padded[0, :9, :, 2] = 1.0
check("padded clip picks the middle of the observed span",
      int(StaticPoseBaseline.middle_observed_frame(padded)[0]) == 4,
      "9 observed frames -> expected index 4, not MAX_FRAMES // 2")
full = torch.ones(1, MAX_FRAMES, N_JOINTS, 3)
check("fully observed clip is unchanged at T // 2",
      int(StaticPoseBaseline.middle_observed_frame(full)[0]) == MAX_FRAMES // 2)
blank = torch.zeros(1, MAX_FRAMES, N_JOINTS, 3)
check("all-padding clip falls back instead of crashing",
      int(StaticPoseBaseline.middle_observed_frame(blank)[0]) == MAX_FRAMES // 2)
mixed = torch.cat([padded, full, blank], dim=0)
check("batched selection stays per-clip",
      StaticPoseBaseline.middle_observed_frame(mixed).tolist() == [4, 12, 12])
check("floor still returns class logits",
      StaticPoseBaseline()(mixed).shape == (3, data.N_CLASSES))

print("\n--- 15. Real-corpus loader refuses misaligned input ---")
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    (root / "Skeleton_data").mkdir()
    (root / "Annotation_files").mkdir()
    arr = np.zeros((3, MAX_FRAMES, N_JOINTS, 2), dtype=np.float32)
    arr[:, :6, :, :] = 0.4
    np.save(root / "Skeleton_data" / "V1.npy", arr)
    pd.DataFrame([[1, 2, "Jab"], [3, 4, "Cross"], [5, 6, "Lead Hook"]]).to_excel(
        root / "Annotation_files" / "V1.xlsx", index=False)
    clips, unlabelled, degenerate = data._load_presegmented(
        "V1", root / "Skeleton_data", root / "Annotation_files")
    check("loads aligned clips", len(clips) == 3, f"got {len(clips)}")
    check("labels map to class indices",
          [c.label for c in clips] == [data.CLASSES.index(n)
                                       for n in ("jab", "cross", "lead_hook")])
    check("loaded clips carry their video, not a fake subject",
          all(c.source == "V1" and c.subject == "" for c in clips))
    check("validity channel was synthesised", clips[0].keypoints.shape[-1] == 3)

    # One fewer annotation row than clips: correspondence is unknowable.
    pd.DataFrame([[1, 2, "Jab"], [3, 4, "Cross"]]).to_excel(
        root / "Annotation_files" / "V1.xlsx", index=False)
    raises("count mismatch is an error, not a silent truncation",
           lambda: data._load_presegmented("V1", root / "Skeleton_data",
                                           root / "Annotation_files"))


print("\n--- 16. ST-GCN ignores padding entirely, as the feature arm does ---")
_obs = torch.zeros(1, MAX_FRAMES, N_JOINTS, 3)
_obs[0, :9, :, :2] = 0.4
_obs[0, :9, :, 2] = 1.0
_net = STGCN().eval()
_alt = _obs.clone()
_alt[0, 9:, :, :2] = 99.0            # garbage in the padded frames only
with torch.no_grad():
    _delta = float((_net(_obs) - _net(_alt)).abs().max())
check("padded content cannot reach the output", _delta < 1e-5,
      f"logit delta {_delta:.6f} -- padding is leaking into real frames")

# The pooled representation must divide by the observed length, not by 25.
_short = torch.zeros(2, MAX_FRAMES, N_JOINTS, 3)
_short[:, :5, :, :2] = 0.4
_short[:, :5, :, 2] = 1.0
_valid = (_short[..., 2].sum(-1) > 0).float()
_feat = torch.ones(2, 8, MAX_FRAMES, N_JOINTS)
_pooled = STGCN._masked_pool(_feat, _valid)
check("masked pool averages over observed frames only",
      bool(torch.allclose(_pooled, torch.ones_like(_pooled))),
      f"got {_pooled[0, 0].item():.4f}, expected 1.0 -- a plain mean would give 5/25 = 0.2")

_none = torch.zeros(1, MAX_FRAMES, N_JOINTS, 3)
with torch.no_grad():
    _out = _net(_none)
check("all-padding clip does not divide by zero",
      bool(torch.isfinite(_out).all()))

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
