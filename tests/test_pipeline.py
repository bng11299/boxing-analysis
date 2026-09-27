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


print("\n--- 17. Paired bootstrap: an interval that can actually say 'no' ---")
from boxingvi.stats import bootstrap_gap_ci, stratified_subsample  # noqa: E402

_rng = np.random.default_rng(0)
_y = _rng.integers(0, data.N_CLASSES, 400)
_perfect = _y.copy()
_noisy = np.where(_rng.random(400) < 0.5, _y, _rng.integers(0, data.N_CLASSES, 400))

# A gate whose pass condition is "excludes zero" is worthless until it has been
# shown to fire on a known positive AND stay silent on a known negative.
_signal = bootstrap_gap_ci(_y, _perfect, _noisy, n_boot=200)
check("real difference excludes zero", _signal["excludes_zero"],
      f"CI [{_signal['ci_low']:.3f}, {_signal['ci_high']:.3f}]")
check("point estimate has the right sign", _signal["point"] > 0)
check("interval brackets the point estimate",
      _signal["ci_low"] <= _signal["point"] <= _signal["ci_high"])

_none = bootstrap_gap_ci(_y, _noisy, _noisy.copy(), n_boot=200)
check("a model against itself does NOT exclude zero", not _none["excludes_zero"],
      "the interval would certify a difference that cannot exist")
check("and its point estimate is exactly zero", _none["point"] == 0.0)

_a = bootstrap_gap_ci(_y, _perfect, _noisy, n_boot=200, seed=7)
_b = bootstrap_gap_ci(_y, _perfect, _noisy, n_boot=200, seed=7)
check("same seed reproduces the interval exactly",
      (_a["ci_low"], _a["ci_high"]) == (_b["ci_low"], _b["ci_high"]))

_acc = bootstrap_gap_ci(_y, _perfect, _noisy, n_boot=200, metric="accuracy")
check("accuracy metric is selectable", _acc["metric"] == "accuracy")
check("and gives a different interval than macro F1",
      abs(_acc["point"] - _signal["point"]) > 1e-9)
check("macro_f1 is the default", bootstrap_gap_ci(
      _y, _perfect, _noisy, n_boot=50)["metric"] == "macro_f1")
raises("an unknown metric is rejected, not silently defaulted",
       lambda: bootstrap_gap_ci(_y, _perfect, _noisy, n_boot=50, metric="auc"))

raises("mismatched lengths are rejected",
       lambda: bootstrap_gap_ci(_y, _perfect, _noisy[:10]))

print("\n--- 18. Stratified subsample keeps the rare class alive ---")
# 300 of class 0, 4 of class 5: uniform sampling at 10% would routinely lose
# class 5 entirely, and the learning curve would then be measuring class
# coverage rather than training-set size.
_yc = np.array([0] * 300 + [1] * 60 + [5] * 4)
_idx = stratified_subsample(_yc, 0.1, seed=0)
_kept = set(np.unique(_yc[_idx]).tolist())
check("every class survives a 10% subsample", _kept == {0, 1, 5}, str(sorted(_kept)))
check("rare class keeps at least one clip", int((_yc[_idx] == 5).sum()) >= 1)
check("subsample is roughly the requested size",
      25 <= len(_idx) <= 50, f"got {len(_idx)} from 364 at frac 0.1")
check("indices are unique", len(set(_idx.tolist())) == len(_idx))
check("frac 1.0 keeps everything", len(stratified_subsample(_yc, 1.0, 0)) == len(_yc))
_s1 = stratified_subsample(_yc, 0.5, seed=3)
_s2 = stratified_subsample(_yc, 0.5, seed=3)
check("same seed gives the same subsample", _s1.tolist() == _s2.tolist())

print("\n--- 19. ST-GCN++ port: architecture pinned to PYSKL ---")
from boxingvi.stgcnpp import (STGCNPP, coco_spatial_adjacency,  # noqa: E402
                              load_pretrained, prepare_input)

_A = coco_spatial_adjacency()
check("adjacency is three 17x17 subsets", _A.shape == (3, N_JOINTS, N_JOINTS))
check("first subset is the identity", np.allclose(_A[0], np.eye(N_JOINTS)))
_cols = _A[1].sum(0)
check("inward subset is column-normalised",
      np.allclose(_cols[_cols > 0], 1.0), str(_cols))
_pp = STGCNPP()
check("parameter count matches PYSKL's ST-GCN++ (~1.39M)",
      1_370_000 <= _pp.n_params() <= 1_400_000, f"{_pp.n_params():,}")
check("backbone exposes 690 tensors, the PYSKL key count",
      sum(k.startswith("backbone.") for k in _pp.state_dict()) == 690)
_pp.eval()
check("ST-GCN++ output shape",
      _pp(torch.zeros(3, 1, 50, N_JOINTS, 3)).shape == (3, data.N_CLASSES))
raises("forward rejects a 4-D tensor",
       lambda: _pp(torch.zeros(3, 50, N_JOINTS, 3)))
raises("forward rejects the wrong joint count",
       lambda: _pp(torch.zeros(3, 1, 50, 25, 3)))

print("\n--- 20. ST-GCN++ input preparation ---")
_x = np.zeros((2, MAX_FRAMES, N_JOINTS, 3), np.float32)
_x[:, :5, :, 0] = 0.0          # x at the left edge
_x[:, :5, :, 1] = 1.0          # y at the bottom edge
_x[:, :5, :, 2] = 1.0          # five real frames, twenty of padding
_x[0, :5, 3, 2] = 0.0          # joint 3 never observed in clip 0
_t = prepare_input(_x, t_out=40)
check("shape is (N, persons=1, t_out, V, C)", tuple(_t.shape) == (2, 1, 40, N_JOINTS, 3))
check("x in [0,1] maps to 2x-1", float(_t[1, 0, :, 0, 0].max()) == -1.0)
check("y in [0,1] maps to 2y-1", float(_t[1, 0, :, 0, 1].min()) == 1.0)
check("no padding frame is ever sampled -- every output frame is observed",
      bool((_t[1, 0, :, :, 2] == 1).all()))
check("an unobserved joint is zeroed, as PYSKL zeroes low-score joints",
      bool((_t[0, 0, :, 3, :] == 0).all()))
check("the validity flag becomes the score channel",
      bool((_t[0, 0, :, 0, 2] == 1).all()))
raises("rejects the wrong joint count",
       lambda: prepare_input(np.zeros((1, 10, 12, 3), np.float32)))
raises("rejects the wrong channel count",
       lambda: prepare_input(np.zeros((1, 10, N_JOINTS, 2), np.float32)))
raises("rejects a 3-D array",
       lambda: prepare_input(np.zeros((10, N_JOINTS, 3), np.float32)))
_nan = _x.copy()
_nan[0, 0, 0, 0] = np.nan
raises("rejects NaN rather than propagating it", lambda: prepare_input(_nan))
raises("rejects a clip with no observed frame",
       lambda: prepare_input(np.zeros((1, 10, N_JOINTS, 3), np.float32)))
_px = _x.copy()
_px[:, :5, :, 0] = 640.0
raises("rejects pixel-space coordinates instead of feeding nonsense to the weights",
       lambda: prepare_input(_px))
raises("rejects a non-positive t_out", lambda: prepare_input(_x, t_out=0))
_edge = _x.copy()
_edge[:, :5, 0, 0] = -0.05     # a wrist a little past the left edge
_edge[:, :5, 1, 1] = 1.08      # an ankle a little below the frame
check("accepts joints slightly outside the frame -- estimators produce them",
      prepare_input(_edge).shape[0] == 2)

print("\n--- 21. Pretrained checkpoint is untrusted input ---")
import hashlib  # noqa: E402
_SENTINEL = Path(tempfile.gettempdir()) / "boxingvi_unsafe_unpickle_ran"


def _pwn(path):  # would only run if the checkpoint were unpickled unsafely
    Path(path).write_text("executed", encoding="utf-8")
    return {}


class _Evil:
    def __reduce__(self):
        return (_pwn, (str(_SENTINEL),))


with tempfile.TemporaryDirectory() as _d:
    _d = Path(_d)
    _src = STGCNPP(n_classes=60)
    for _p in _src.parameters():                # make the weights distinguishable
        torch.nn.init.normal_(_p, std=0.05)
    _good = _d / "good.pth"
    torch.save({"meta": {"note": "pyskl"}, "state_dict": _src.state_dict()}, _good)

    _dst = STGCNPP()
    _head_before = _dst.cls_head.fc_cls.weight.clone()
    _rep = load_pretrained(_dst, _good, allow_unpinned=True)
    check("a well-formed checkpoint loads", _rep["loaded_tensors"] == 690, str(_rep))
    check("backbone weights actually arrive",
          torch.equal(_dst.backbone.gcn[4].gcn.conv.weight,
                      _src.backbone.gcn[4].gcn.conv.weight))
    check("the source head is not copied -- it has a different label set",
          torch.equal(_dst.cls_head.fc_cls.weight, _head_before)
          and _dst.cls_head.fc_cls.out_features == data.N_CLASSES)
    check("the source class count is reported", _rep["source_classes"] == 60)
    check("a sha256 is reported for pinning", len(_rep["sha256"]) == 64)

    raises("no spec and no opt-out: an unpinned checkpoint is refused",
           lambda: load_pretrained(STGCNPP(), _good))
    raises("a spec with md5 but no sha256 is refused as unpinned",
           lambda: load_pretrained(STGCNPP(), _good,
                                   {"md5": hashlib.md5(_good.read_bytes()).hexdigest()}))
    raises("a missing file is an error",
           lambda: load_pretrained(STGCNPP(), _d / "nope.pth", allow_unpinned=True), FileNotFoundError)
    raises("a size mismatch is refused",
           lambda: load_pretrained(STGCNPP(), _good, {"size": 123}, allow_unpinned=True))
    raises("an md5 mismatch is refused",
           lambda: load_pretrained(STGCNPP(), _good, {"md5": "0" * 32}, allow_unpinned=True))
    raises("a sha256 mismatch is refused",
           lambda: load_pretrained(STGCNPP(), _good, {"sha256": "0" * 64}))

    _sd = _src.state_dict()
    _miss = {k: v for k, v in _sd.items() if k != "backbone.gcn.3.gcn.A"}
    torch.save({"state_dict": _miss}, _d / "missing.pth")
    raises("a missing backbone tensor is refused, not silently left random",
           lambda: load_pretrained(STGCNPP(), _d / "missing.pth", allow_unpinned=True))

    torch.save({"state_dict": {**_sd, "backbone.extra.weight": torch.zeros(1)}},
               _d / "extra.pth")
    raises("an unexpected backbone tensor is refused -- wrong architecture",
           lambda: load_pretrained(STGCNPP(), _d / "extra.pth", allow_unpinned=True))

    _shape = dict(_sd)
    _shape["backbone.data_bn.weight"] = torch.zeros(75)
    torch.save({"state_dict": _shape}, _d / "shape.pth")
    raises("a shape mismatch is refused (25-joint NTU weights, say)",
           lambda: load_pretrained(STGCNPP(), _d / "shape.pth", allow_unpinned=True))

    torch.save({"state_dict": {**_sd, "backbone.data_bn.weight": [1, 2, 3]}},
               _d / "notensor.pth")
    raises("a payload that is not tensors is refused -- full key set, one list value",
           lambda: load_pretrained(STGCNPP(), _d / "notensor.pth", allow_unpinned=True))

    if _SENTINEL.exists():
        _SENTINEL.unlink()
    torch.save({"state_dict": _Evil()}, _d / "evil.pth")
    raises("a checkpoint carrying arbitrary code is refused by the safe loader",
           lambda: load_pretrained(STGCNPP(), _d / "evil.pth", allow_unpinned=True))
    check("and the embedded code never ran", not _SENTINEL.exists())

    print("\n--- 22. The verified bytes are the loaded bytes ---")

    from boxingvi import stgcnpp as _pp  # noqa: E402
    _blob = _good.read_bytes()
    _pin = {"size": len(_blob), "md5": hashlib.md5(_blob).hexdigest(),
            "sha256": hashlib.sha256(_blob).hexdigest()}
    check("a checkpoint matching its pinned size, md5 and sha256 loads",
          load_pretrained(STGCNPP(), _good, _pin)["sha256"] == _pin["sha256"])

    # If the loader re-opened the file after hashing it, a swap in between
    # would load unverified bytes. Record what torch.load is actually handed.
    _seen = []
    _real_load = torch.load

    def _spy(f, *a, **k):
        _seen.append(f.getvalue() if hasattr(f, "getvalue") else f)
        return _real_load(f, *a, **k)

    _pp.torch.load = _spy
    try:
        load_pretrained(STGCNPP(), _good, _pin)
    finally:
        _pp.torch.load = _real_load
    check("torch.load is handed the in-memory bytes that were hashed, not the path",
          len(_seen) == 1 and isinstance(_seen[0], bytes)
          and hashlib.sha256(_seen[0]).hexdigest() == _pin["sha256"])
    _cap = _pp.MAX_CHECKPOINT_BYTES
    _reads = []
    _real_read = Path.read_bytes
    _pp.MAX_CHECKPOINT_BYTES = 10
    Path.read_bytes = lambda self: _reads.append(self) or _real_read(self)
    try:
        raises("a file over the size ceiling is refused",
               lambda: load_pretrained(STGCNPP(), _good, allow_unpinned=True))
    finally:
        _pp.MAX_CHECKPOINT_BYTES = _cap
        Path.read_bytes = _real_read
    check("... and refused before a single byte is read", _reads == [])

print("\n--- 23. run_baseline: seeds, intervals, and pinning ---")
sys.path.insert(0, str(Path(__file__).parent.parent))
import run_baseline  # noqa: E402

check("--seed is accepted and parsed",
      run_baseline.build_parser().parse_args(["--synthetic", "--seed", "2"]).seed == 2)
_models = ["random-forest", "st-gcn", "static-pose",
           "st-gcn++ pretrained", "st-gcn++ scratch"]
_cmp = {k: (hi, lo) for k, _l, hi, lo in
        run_baseline.interval_comparisons([{"model": m} for m in _models])}
_want = {
    "gap_st_gcn_minus_random_forest": ("st-gcn", "random-forest"),
    "gap_st_gcn_minus_static_pose": ("st-gcn", "static-pose"),
    "gap_stgcnpp_pretrained_minus_random_forest": ("st-gcn++ pretrained", "random-forest"),
    "gap_stgcnpp_pretrained_minus_scratch": ("st-gcn++ pretrained", "st-gcn++ scratch"),
    "gap_stgcnpp_scratch_minus_random_forest": ("st-gcn++ scratch", "random-forest"),
}
check("all five intervals exist, each with the named model on the higher side",
      {k: (hi["model"], lo["model"]) for k, (hi, lo) in _cmp.items()} == _want,
      str(_cmp))
_cmp_off = {k: (hi, lo) for k, _l, hi, lo in run_baseline.interval_comparisons(
    [{"model": m} for m in _models[:3]])}
check("without ST-GCN++ its intervals carry None and are skipped, not faked",
      _cmp_off["gap_stgcnpp_scratch_minus_random_forest"][0] is None)


def _exit_code(argv):
    import contextlib
    import io
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            run_baseline.main(argv)
    except SystemExit as e:
        return e.code, err.getvalue()
    return 0, err.getvalue()


with tempfile.TemporaryDirectory() as _d:
    _fake = Path(_d) / "w.pth"
    _fake.write_bytes(b"x")
    _code, _msg = _exit_code(["--synthetic", "--stgcnpp", "pretrained", "--pretrained",
                              str(_fake), "--pretrained-spec", "ntu60_xsub_hrnet_j"])
    check("a checkpoint spec with no pinned sha256 is refused at the CLI, for that reason",
          _code == 2 and "no pinned sha256" in _msg, _msg[-200:])

_sc = synthetic.make_corpus(n_train=36, n_val=12, seed=4)
_str, _sva = data.split_by_subject(_sc)
_xtr, _ytr = data.to_arrays(_str)
_xva, _yva = data.to_arrays(_sva)


def _rf(seed):
    return run_baseline.run_interpretable(
        [c.keypoints for c in _str], _ytr, [c.keypoints for c in _sva], _yva,
        seed)[0]["_pred"]


check("the random forest is deterministic for a fixed seed", _rf(3) == _rf(3))


def _pp_run(seed):
    r = run_baseline.run_stgcnpp(_xtr, _ytr, _xva, _yva, epochs=1, seed=seed, t_out=8)
    return r["_pred"], r["train_macro_f1"]


check("ST-GCN++ is deterministic for a fixed seed -- the reproduction check "
      "depends on it", _pp_run(5) == _pp_run(5))

print("\n--- 24. leave-videos-out cross-validation ---")
# The real per-video clip counts (V6 excluded), so the fold assignment the
# reported run will use is pinned here and any change to it is visible.
_SIZES = {"V1": 1865, "V2": 232, "V3": 811, "V4": 560, "V5": 595,
          "V7": 195, "V8": 199, "V9": 149, "V10": 151}


def _vclips(sizes, label_fn=lambda v, i: i % 6):
    return [Clip(kp(3), label_fn(v, i), "", source=v)
            for v, n in sizes.items() for i in range(n)]


_real_like = _vclips(_SIZES)
_f = data.group_folds(_real_like, 3)
check("the 3 folds on the real video sizes are the ones the run will use",
      _f == [("V1",), ("V2", "V3", "V8", "V10"), ("V4", "V5", "V7", "V9")], str(_f))
check("every video lands in exactly one fold",
      sorted(v for f in _f for v in f) == sorted(_SIZES))
_relabelled = _vclips(_SIZES, lambda v, i: (i * 7 + int(v[1:])) % 6)
check("folds are label-blind: relabelling every clip moves no video",
      data.group_folds(_relabelled, 3) == _f)
check("folds are deterministic", data.group_folds(list(reversed(_real_like)), 3) == _f)
raises("k=1 is refused", lambda: data.group_folds(_real_like, 1))
raises("more folds than videos is refused", lambda: data.group_folds(_real_like, 10))
raises("a clip with no group id is refused, not merged",
       lambda: data.group_folds(_vclips({"V1": 3}) + [Clip(kp(3), 0, "", "")], 2))

_splits = data.cv_splits(_real_like, 3)
check("no video is on both sides of any fold",
      all(not ({c.source for c in tr} & {c.source for c in te}) for _h, tr, te in _splits))
check("every clip is tested exactly once",
      sum(len(te) for _h, _tr, te in _splits) == len(_real_like)
      and len({id(c) for _h, _tr, te in _splits for c in te}) == len(_real_like))
# V1 holds the only clips of class 5 here: holding it out leaves a model that
# can never predict class 5, and its zero would be read as a finding.
_starved = _vclips({"V1": 12, "V2": 10, "V3": 10},
                   lambda v, i: 5 if v == "V1" else i % 5)
raises("a fold whose training side lacks a class is refused",
       lambda: data.cv_splits(_starved, 3))

with tempfile.TemporaryDirectory() as _d:
    _out = Path(_d) / "cv.json"
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        run_baseline.main(["--synthetic", "--allow-synthetic", "--cv", "3",
                           "--epochs", "1", "--bootstrap", "50",
                           "--out", str(_out)])
    _cv = json.loads(_out.read_text(encoding="utf-8"))
    _n = len(synthetic.make_corpus(seed=0))
    check("a CV run scores every clip once, pooled",
          _cv["n_val_clips"] == _n and _cv["n_train_clips"] is None,
          f"{_cv['n_val_clips']} vs {_n}")
    check("... and is stamped synthetic and says it is CV",
          _cv["synthetic"] is True and "cross-validation" in _cv["split"])
    _tg = [g for f in _cv["cv"]["folds"] for g in f["test_groups"]]
    check("... with the test folds partitioning the subjects",
          sorted(_tg) == sorted({c.subject for c in synthetic.make_corpus(seed=0)})
          and len(_tg) == len(set(_tg)))
    # If labels and predictions were concatenated in different orders the
    # pooled confusion would stop being the sum of the per-fold ones.
    _ok = True
    for _r in _cv["results"]:
        _sum = np.sum([f["models"][_r["model"]]["confusion"]
                       for f in _cv["cv"]["folds"]], axis=0)
        _ok &= bool(np.array_equal(_sum, np.array(_r["confusion"])))
    check("pooled predictions stay aligned with pooled labels, every model", _ok)
    _rf_conf = np.array(next(r for r in _cv["results"]
                             if r["model"] == "random-forest")["confusion"])
    check("... and that check is not vacuous: the forest predicts more than one class",
          int((_rf_conf.sum(axis=0) > 0).sum()) > 1)
    check("each fold keeps the forest's own feature importances",
          all("top_features" in f["models"]["random-forest"] for f in _cv["cv"]["folds"]))
    check("per-video gaps and a sign test are recorded for the headline",
          "sign_test_p" in _cv["cv"]["video_gaps"]
          ["gap_st_gcn_minus_random_forest"]["macro_f1"])
    check("per-fold gaps are recorded for the headline interval",
          len(_cv["cv"]["fold_gaps"]["gap_st_gcn_minus_random_forest"]
              ["macro_f1"]["gaps"]) == 3)
_code, _msg = _exit_code(["--synthetic", "--cv", "1"])
check("--cv 1 is refused at the CLI", _code == 2 and "--cv" in _msg, _msg[-200:])
_code, _msg = _exit_code(["--synthetic", "--cv", "25"])
check("more folds than subjects is a CLI error, not a traceback",
      _code == 2 and "--cv 25" in _msg, _msg[-200:])
_code, _msg = _exit_code(["--data", ".", "--cv", "3", "--include-v6"])
check("--cv with --include-v6 is refused before any data is read",
      _code == 2 and "V6" in _msg, _msg[-200:])
check("a CV run defaults to its own results file, never the headline one",
      run_baseline.build_parser().parse_args(["--synthetic"]).out is None)

# The leak that matters is at the point models are fitted, not in cv_splits:
# spy on what run_cv actually hands to run_arms.
_spy_calls = []


def _spy_arms(train, test, a, spec):
    _spy_calls.append(({c.source for c in train}, {c.source for c in test},
                       len(train), len(test)))
    y = np.array([c.label for c in test])
    return [{"model": "st-gcn++ pretrained", "_pred": y.tolist(),
             "macro_f1": 1.0, "accuracy": 1.0, "confusion": [[0]],
             "params": 7, "t_out": 50,
             "pretrained_from": {"sha256": "c6e0-fake"}}]


_real_arms = run_baseline.run_arms
run_baseline.run_arms = _spy_arms
try:
    import argparse
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        _pooled, _yall, _gall, _folds = run_baseline.run_cv(
            _real_like, argparse.Namespace(cv=3), None, lambda c: c.source)
finally:
    run_baseline.run_arms = _real_arms
check("run_arms is never given a video on both its train and test side",
      len(_spy_calls) == 3 and all(not (tr & te) for tr, te, _a, _b in _spy_calls))
check("... every fold trains on everything it does not test",
      all(n_tr + n_te == len(_real_like) for _a, _b, n_tr, n_te in _spy_calls))
check("pooled group ids line up with pooled labels",
      len(_gall) == len(_yall) == len(_real_like)
      and [g for g in dict.fromkeys(_gall.tolist())]
      == [v for f in _f for v in f])
check("the checkpoint's sha256 survives into every fold record and the pooled row",
      all(f["models"]["st-gcn++ pretrained"]["pretrained_from"]["sha256"] == "c6e0-fake"
          for f in _folds)
      and _pooled[0]["pretrained_from"]["sha256"] == "c6e0-fake"
      and _pooled[0]["t_out"] == 50)

from boxingvi.stats import per_group_gaps, sign_test_p  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

check("sign test: 9 of 9 agreeing gives p = 2/512", abs(sign_test_p(9, 0) - 2 / 512) < 1e-12)
check("sign test: a 1-1 split gives p = 1", sign_test_p(1, 1) == 1.0)
check("sign test: no untied groups gives p = 1, not a crash", sign_test_p(0, 0) == 1.0)
_pg = per_group_gaps([0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, 1], ["A", "A", "B", "B"],
                     lambda y, p: float(accuracy_score(y, p)))
check("per-video gaps: a zero gap is a tie, not a disagreement",
      _pg["ties"] == 1 and _pg["negative"] == 1 and _pg["positive"] == 0, str(_pg))
raises("per-video gaps refuse misaligned groups",
       lambda: per_group_gaps([0, 1], [0, 1], [0, 1], ["A"], accuracy_score))
check("per-fold macro F1 ignores a class the fold does not contain",
      run_baseline.present_class_macro_f1(np.array([0, 0, 1]), np.array([0, 4, 1]))
      == run_baseline.present_class_macro_f1(np.array([0, 0, 1]), np.array([0, 2, 1])))

with tempfile.TemporaryDirectory() as _d:
    _single = Path(_d) / "single.json"
    with contextlib.redirect_stdout(io.StringIO()):
        run_baseline.main(["--synthetic", "--allow-synthetic", "--epochs", "1",
                           "--bootstrap", "20", "--out", str(_single)])
    _sj = json.loads(_single.read_text(encoding="utf-8"))
    check("a single-split results file keeps exactly its old keys, in order",
          list(_sj) == ["synthetic", "seed", "epochs", "split", "balanced",
                        "n_train_clips", "n_val_clips", "train_videos", "val_videos",
                        "include_v6", "gap_st_gcn_minus_random_forest",
                        "gap_st_gcn_minus_random_forest_accuracy",
                        "gap_st_gcn_minus_static_pose",
                        "gap_st_gcn_minus_static_pose_accuracy", "stgcnpp",
                        "gap_stgcnpp_pretrained_minus_random_forest",
                        "gap_stgcnpp_pretrained_minus_scratch",
                        "gap_stgcnpp_scratch_minus_random_forest",
                        "stgcnpp_epochs", "environment", "intervals", "results"],
          str(list(_sj)))
check("... and the forest still reports its feature importances",
      "top_features" in next(r for r in _sj["results"] if r["model"] == "random-forest"))

print("\n--- V6 continuous track, alignment and the wrist-speed punch finder ---")
from boxingvi import align, punchfind  # noqa: E402
from boxingvi.data import Span, Track  # noqa: E402


def _track(n=240, fps=24.0, ids=None, punches=(), hand=1, dur_s=0.25, amp=1.0):
    """A still body (torso 100px) whose wrist moves out and back for each punch
    start in `punches` (seconds). Coordinates in pixels."""
    t = np.arange(n) / fps
    xy = np.zeros((n, N_JOINTS, 2))
    xy[:, 5] = (400, 300); xy[:, 6] = (500, 300)      # shoulders
    xy[:, 11] = (400, 400); xy[:, 12] = (500, 400)    # hips: torso = 100px
    xy[:, 9] = (400, 350); xy[:, 10] = (500, 350)     # wrists at rest
    for p in punches:
        phase = np.clip((t - p) / dur_s, 0, 1)
        xy[:, 9 + hand, 0] += amp * 100 * np.sin(np.pi * phase)
    f = np.arange(n) if ids is None else np.asarray(ids)
    return Track(frame_ids=f.astype(np.int64), xy=xy, fps=fps)


_t = _track()
raises("track refuses non-increasing frame ids",
       lambda: Track(np.array([0, 2, 1]), np.zeros((3, N_JOINTS, 2)), 24.0))
raises("track refuses NaN coordinates",
       lambda: Track(np.arange(3), np.full((3, N_JOINTS, 2), np.nan), 24.0))
raises("track refuses a non-positive frame rate",
       lambda: Track(np.arange(3), np.zeros((3, N_JOINTS, 2)), 0.0))
raises("track refuses xy that does not match the frame count",
       lambda: Track(np.arange(4), np.zeros((3, N_JOINTS, 2)), 24.0))
check("segments split exactly at gaps",
      Track(np.array([3, 4, 5, 9, 10, 20]), np.zeros((6, N_JOINTS, 2)), 24.0).segments()
      == [(0, 3), (3, 5), (5, 6)])
raises("body scale refuses a collapsed skeleton",
       lambda: punchfind.body_scale(Track(np.arange(3), np.zeros((3, N_JOINTS, 2)), 24.0)))

# A wrist that is simply in a different place on either side of a pose gap must
# not read as a punch: nothing may be differenced across the gap.
_g = _track(n=40, ids=np.r_[np.arange(20), np.arange(100, 120)])
_g.xy[20:, 10, 0] += 300
check("no speed is computed across a pose gap",
      punchfind.wrist_signal(_g, "relative").max() < 1e-9)

# The same motion must give the same speed in body lengths per second at 24
# and at 60 fps -- the reason the rule is in seconds (O54).
_s24 = punchfind.wrist_signal(_track(n=240, fps=24.0, punches=[4.0]), "relative").max()
_s60 = punchfind.wrist_signal(_track(n=600, fps=60.0, punches=[4.0]), "relative").max()
check("peak speed agrees within 15% at 24 and 60 fps", abs(_s24 / _s60 - 1) < 0.15,
      f"{_s24:.3f} vs {_s60:.3f}")
_s_far = punchfind.wrist_signal(_track(punches=[4.0], amp=0.5), "relative").max()
check("speed is in body lengths: half the reach, half the speed",
      abs(_s_far / _s24 - 0.5) < 0.02, f"{_s_far:.3f} vs {_s24:.3f}")
raises("unknown rule variant refused", lambda: punchfind.wrist_signal(_t, "sideways"))

_p = _track(n=480, punches=[2.0, 6.0, 12.0])
_sig = punchfind.wrist_signal(_p, "relative")
_det, _hand = punchfind.detect(_p, _sig, threshold=1.0, min_gap_s=0.4)
_gt = [Span(int(s * 24), int((s + 0.25) * 24), 0) for s in (2.0, 6.0, 12.0)]
check("three planted punches give exactly three detections",
      len(_det) == 3, str(_det))
check("... each inside its span, with the right hand",
      len(punchfind.match(_det, _gt, 0)) == 3 and set(_hand.tolist()) == {1})
check("a still body gives no detections",
      len(punchfind.detect(_t, punchfind.wrist_signal(_t), 0.5, 0.4)[0]) == 0)
raises("detect refuses a signal that does not match the track",
       lambda: punchfind.detect(_t, np.zeros((5, 2)), 1.0, 0.4))

# Matching is one-to-one and maximal.
check("two detections in one span count once",
      len(punchfind.match([5, 6], [Span(0, 10, 0)], 0)) == 1)
check("one detection in two overlapping spans counts once",
      len(punchfind.match([5], [Span(0, 10, 0), Span(4, 6, 0)], 0)) == 1)
check("matching is maximal where first-come would not be",
      len(punchfind.match([5, 9], [Span(0, 10, 0), Span(5, 6, 0)], 0)) == 2)
check("tolerance extends a span by exactly tol frames",
      len(punchfind.match([12], [Span(0, 10, 0)], 2)) == 1
      and len(punchfind.match([13], [Span(0, 10, 0)], 2)) == 0)
raises("negative tolerance refused", lambda: punchfind.match([1], [Span(0, 1, 0)], -1))
_sc = punchfind.score([5, 50, 500], [Span(0, 10, 0), Span(45, 55, 0), Span(400, 410, 0)],
                      0, (0, 100))
check("scoring ignores detections and spans outside the window",
      (_sc["n_detected"], _sc["n_labelled"], _sc["tp"]) == (2, 2, 2), str(_sc))
_fl = punchfind.random_floor(3, _gt, 0, (0, 479), _p.frame_ids, n_rep=50, seed=1)
check("random floor is deterministic and below a perfect score",
      _fl == punchfind.random_floor(3, _gt, 0, (0, 479), _p.frame_ids, n_rep=50, seed=1)
      and _fl["f1_mean"] < 1.0)
_bb = punchfind.block_bootstrap(_det, _gt, 0, (0, 479), block=48, n_boot=200)
check("block bootstrap of a perfect detector is [1, 1], empty draws dropped not zeroed",
      _bb["f1"] == [1.0, 1.0] and _bb["n_undefined_dropped"] > 0, str(_bb))
raises("block bootstrap refuses when most draws are undefined",
       lambda: punchfind.block_bootstrap([5], [Span(400, 410, 0)], 0, (0, 479),
                                         block=4, n_boot=200))

# Alignment checks must FIND an offset that is there, not just report zero.
_q = _track(n=2400, punches=list(np.arange(2.0, 95.0, 3.0)))
_speed = punchfind.wrist_signal(_q).max(axis=1)
_late = [Span(int(s * 24) + 12, int((s + 0.25) * 24) + 12, 0) for s in np.arange(2.0, 95.0, 3.0)]
_scan = align.label_shift_scan(_q, _late, _speed, range(-20, 21))
check("shift scan finds labels planted 12 frames late",
      max(_scan, key=_scan.get) in (-13, -12, -11), str(max(_scan, key=_scan.get)))
_energy = np.zeros(2500)
_mot = np.r_[0, np.linalg.norm(np.diff(_q.xy, axis=0), axis=-1).sum(1)]
_energy[3:3 + len(_mot)] = _mot                    # video runs 3 frames behind
_xc = align.skeleton_video_xcorr(_q, _energy, range(-6, 7))
check("video cross-correlation finds a planted 3-frame offset",
      max(_xc, key=_xc.get) == 3, str(max(_xc, key=_xc.get)))
raises("motion energy refuses a missing video",
       lambda: align.video_motion_energy(Path("no/such/video.mp4")), FileNotFoundError)

# The V6 loaders are the trust boundary for the continuous data.
with tempfile.TemporaryDirectory() as _d:
    import pandas as pd
    _ann = Path(_d)

    def _xlsx(rows):
        pd.DataFrame(rows, columns=["start", "end", "class"]).to_excel(
            _ann / "V6.xlsx", index=False)

    _xlsx([[10, 20, "Lead Hook"], [5, 8, "Rear Uppercut"], [None, None, None]])
    _sp = data.load_v6_spans(_ann)
    check("V6 spans: empty rows dropped, the rest sorted by start",
          [(s.start, s.label) for s in _sp]
          == [(5, data.CLASSES.index("rear_uppercut")), (10, data.CLASSES.index("lead_hook"))])
    _xlsx([[10, 20, "Lead Hook"], [30, 25, "Rear Hook"], [40, 50, "Jab"]])
    raises("V6 spans: a labelled row ending before it starts is an error",
           lambda: data.load_v6_spans(_ann))
    _xlsx([[10, 20, "Lead Hook"], [30, 40, None], [50, 60, "Rear Hook"]])
    raises("V6 spans: frames without a label are an error, not a silent drop",
           lambda: data.load_v6_spans(_ann))

    _skel = Path(_d)
    _arr = np.zeros((5, 1, N_JOINTS, 3))
    _arr[:, 0, :, 0] = np.arange(157, 162)[:, None]
    _arr[:, 0, :, 1:] = 0.5
    np.save(_skel / "V6.npy", _arr)
    _tr = data.load_v6_track(_skel)
    check("V6 track: channel 0 read as frame ids, x/y scaled to pixels",
          _tr.frame_ids.tolist() == [157, 158, 159, 160, 161]
          and np.allclose(_tr.xy[0, 0], [960, 540]) and abs(_tr.fps - 23.976) < 1e-3)
    _arr[2, 0, 4, 0] = 999
    np.save(_skel / "V6.npy", _arr)
    raises("V6 track: refuses a channel 0 that is not one id per frame",
           lambda: data.load_v6_track(_skel))

    _arr[2, 0, 4, 0] = 159
    _arr[3, 0, 9, 1:] = 0.0
    np.save(_skel / "V6.npy", _arr)
    raises("V6 track: refuses dropped joints at (0, 0) instead of differencing them",
           lambda: data.load_v6_track(_skel))
    np.save(_skel / "V6.npy", np.array([object()] * 3), allow_pickle=True)
    raises("V6 track: refuses a pickled (object) array",
           lambda: data.load_v6_track(_skel))

# Short pose segments at 60 fps: shorter than the smoothing window.
_short = Track(np.r_[0, 1, 10, 11, 12, np.arange(20, 60)],
               _track(n=45).xy, 60.0)
check("segments shorter than the smoothing window do not crash, and score finite",
      np.all(np.isfinite(punchfind.wrist_signal(_short))))

check("a detection exactly on the window's last frame counts; one past it does not",
      punchfind.score([100], [Span(95, 100, 0)], 2, (0, 100))["tp"] == 1
      and punchfind.score([101], [Span(95, 100, 0)], 2, (0, 100))["n_detected"] == 0)

# Shift floor: detections that hit every span only at the right time.
_sf = punchfind.shift_floor(_det, _gt, 0, (0, 479), min_shift=48, n_rep=100)
check("shift floor keeps the detector's count and scores far below it",
      _sf["n_detections"] == 3 and _sf["f1_p97_5"] < 0.7, str(_sf))
raises("shift floor refuses a minimum shift that leaves nothing to draw",
       lambda: punchfind.shift_floor(_det, _gt, 0, (0, 479), min_shift=300))

# The tune/test separation the headline depends on.
import run_punch_finder  # noqa: E402

_qs = [Span(int(s * 24), int((s + 0.25) * 24), 0) for s in np.arange(2.0, 95.0, 3.0)]
_cut = run_punch_finder.cut_point(_q, _qs, _qs[0].start, _qs[-1].end)
check("cut point lies inside no span",
      not any(s.start <= _cut <= s.end for s in _qs), str(_cut))
raises("cut point refuses when every frame is inside a span",
       lambda: run_punch_finder.cut_point(_q, [Span(0, 100, 0)], 0, 100))
_tw = (_qs[0].start, _cut - 1)
_tuned = run_punch_finder.tune(_q, _qs, _tw, 2)
_tuned_blind = run_punch_finder.tune(_q, [s for s in _qs if s.start < _cut], _tw, 2)
_tuned_moved = run_punch_finder.tune(
    _q, [s for s in _qs if s.start < _cut]
    + [Span(s.start + 7, s.end + 7, 0) for s in _qs if s.start >= _cut], _tw, 2)
check("tuning is blind to the test half: deleting or moving its labels changes nothing",
      _tuned == _tuned_blind == _tuned_moved)
check("... and tuning found the planted punches", _tuned["overall"]["tune"]["f1"] > 0.9,
      str(_tuned["overall"]["tune"]))
_, _fails = run_punch_finder.alignment_gate(_q, _qs, None)
check("alignment gate passes aligned labels", _fails == [], str(_fails))
_, _fails = run_punch_finder.alignment_gate(
    _q, [Span(s.start + 48, s.end + 48, 0) for s in _qs], None)
check("alignment gate fails labels 48 frames late (the row-index misreading)",
      len(_fails) > 0)

check("native frame rates cover every video, and none is 30 exactly",
      set(data.NATIVE_FPS) == set(data.VIDEOS) and 30.0 not in data.NATIVE_FPS.values())

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
