"""Run the BoxingVI baselines: interpretable arm, learned arm, and three floors.

    python run_baseline.py --synthetic            # validate the pipeline
    python run_baseline.py --data path/to/Dataset # the real corpus

Synthetic results are never written to the results file unless --allow-synthetic
is passed, and they are stamped as synthetic when they are. A number produced
from fixtures that reaches a report is the failure this guards against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi.data import CLASSES, N_CLASSES, split_by_subject, to_arrays  # noqa: E402
from boxingvi.features import FEATURE_NAMES, feature_matrix  # noqa: E402


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    return {
        # Underscored keys are stripped before the results file is written.
        # Predictions are kept in memory so a paired bootstrap can resample
        # both models on the same draw of validation clips.
        "_pred": np.asarray(y_pred).tolist(),
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class_f1": {
            CLASSES[i]: float(v) for i, v in enumerate(
                f1_score(y_true, y_pred, average=None,
                         labels=list(range(N_CLASSES)), zero_division=0))
        },
        "confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(N_CLASSES))).tolist(),
    }


def run_floors(ytr, yva, seed=0):
    rng = np.random.default_rng(seed)
    majority = np.full_like(yva, np.bincount(ytr, minlength=N_CLASSES).argmax())
    p = np.bincount(ytr, minlength=N_CLASSES) / len(ytr)
    weighted = rng.choice(N_CLASSES, size=len(yva), p=p)
    return [evaluate("majority-class", yva, majority),
            evaluate("weighted-random", yva, weighted)]


def run_interpretable(xtr_kp, ytr, xva_kp, yva, seed=0, balanced=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    ftr, fva = feature_matrix(xtr_kp), feature_matrix(xva_kp)
    out = []
    # Applied to BOTH arms or neither. Weighting only the arm that is losing
    # would not be a fairer comparison, just a differently unfair one.
    cw = "balanced" if balanced else None

    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1,
                                class_weight=cw)
    rf.fit(ftr, ytr)
    r = evaluate("random-forest", yva, rf.predict(fva))
    order = np.argsort(rf.feature_importances_)[::-1][:10]
    r["top_features"] = [
        {"feature": FEATURE_NAMES[i], "importance": float(rf.feature_importances_[i])}
        for i in order
    ]
    out.append(r)

    svm = make_pipeline(StandardScaler(),
                        SVC(kernel="rbf", random_state=seed, class_weight=cw))
    svm.fit(ftr, ytr)
    out.append(evaluate("svm-rbf", yva, svm.predict(fva)))
    return out


def run_learned(xtr, ytr, xva, yva, epochs=30, seed=0, static=False,
                balanced=False):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from boxingvi.stgcn import STGCN, StaticPoseBaseline

    torch.manual_seed(seed)
    model = StaticPoseBaseline() if static else STGCN()
    name = "static-pose" if static else "st-gcn"

    dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                    batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Same inverse-frequency scheme sklearn's class_weight="balanced" uses:
    # n / (k * count_k). Keeps the two arms comparable.
    if balanced:
        counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float64)
        w = len(ytr) / (N_CLASSES * np.clip(counts, 1, None))
        lossf = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32))
    else:
        lossf = torch.nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    {name} epoch {ep + 1:>3}/{epochs}  loss {tot / len(xtr):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xva)).argmax(1).numpy()
        tr_pred = model(torch.from_numpy(xtr)).argmax(1).numpy()
    r = evaluate(name, yva, pred)
    # Train-set score alongside the validation one, so the gap between them is
    # visible in the results file. Without it there is no way to tell an
    # undertrained model from an overfitted one, and the two call for opposite
    # remedies -- which is exactly the question a reader will ask of any arm
    # that loses a comparison.
    from sklearn.metrics import accuracy_score
    r["train_accuracy"] = float(accuracy_score(ytr, tr_pred))
    r["train_macro_f1"] = macro_f1(ytr, tr_pred)
    r["generalisation_gap"] = round(r["train_accuracy"] - r["accuracy"], 4)
    r["params"] = model.n_params() if hasattr(model, "n_params") else sum(
        p.numel() for p in model.parameters())
    r["train_seconds"] = round(time.time() - t0, 1)
    return r


def run_stgcnpp(xtr, ytr, xva, yva, epochs=30, seed=0, t_out=50,
                pretrained=None, spec=None, balanced=False):
    """ST-GCN++, from scratch or fine-tuned from a PYSKL checkpoint.

    Both variants get the same optimiser -- AdamW with the backbone at 1e-3 and
    the fresh head at 1e-2 -- so the only thing separating them is where the
    backbone weights started. That is the comparison that tests whether the
    random forest wins for lack of data.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from boxingvi.stgcnpp import STGCNPP, load_pretrained, prepare_input

    torch.manual_seed(seed)
    model = STGCNPP()
    name = "st-gcn++ pretrained" if pretrained else "st-gcn++ scratch"
    provenance = None
    if pretrained:
        provenance = load_pretrained(model, pretrained, spec)
        print(f"    loaded {provenance['loaded_tensors']} backbone tensors "
              f"(source head: {provenance['source_classes']} classes, "
              f"sha256 {provenance['sha256'][:12]}...)")

    ttr, tva = prepare_input(xtr, t_out), prepare_input(xva, t_out)
    dl = DataLoader(TensorDataset(ttr, torch.from_numpy(ytr)),
                    batch_size=64, shuffle=True)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-3},
        {"params": model.cls_head.parameters(), "lr": 1e-2},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if balanced:
        counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float64)
        w = len(ytr) / (N_CLASSES * np.clip(counts, 1, None))
        lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    else:
        lossf = torch.nn.CrossEntropyLoss()

    def predict(t):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(t[i:i + 256]).argmax(1)
                              for i in range(0, len(t), 256)]).numpy()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"    {name} epoch {ep + 1:>3}/{epochs}  loss {tot / len(ttr):.4f}"
                  f"  ({time.time() - t0:.0f}s)")

    r = evaluate(name, yva, predict(tva))
    from sklearn.metrics import accuracy_score
    tr_pred = predict(ttr)
    r["train_accuracy"] = float(accuracy_score(ytr, tr_pred))
    r["train_macro_f1"] = macro_f1(ytr, tr_pred)
    r["generalisation_gap"] = round(r["train_accuracy"] - r["accuracy"], 4)
    r["params"] = model.n_params()
    r["t_out"] = t_out
    r["train_seconds"] = round(time.time() - t0, 1)
    if provenance:
        r["pretrained_from"] = {"path": Path(pretrained).name, **provenance}
    return r


def interval_comparisons(results: list[dict]) -> list[tuple]:
    """Every (key, label, higher, lower) pair that gets a paired bootstrap CI.

    A pair whose models were not run carries None and is skipped by the caller,
    so a run without ST-GCN++ reproduces the earlier intervals unchanged. Each
    interval is seeded independently, so adding a pair here moves no other one.
    """
    def find(name):
        return next((r for r in results if r["model"] == name), None)

    rf, gcn, sp = find("random-forest"), find("st-gcn"), find("static-pose")
    ppp, pps = find("st-gcn++ pretrained"), find("st-gcn++ scratch")
    return [
        ("gap_st_gcn_minus_random_forest",
         "interpretability gap (st-gcn - random-forest)", gcn, rf),
        ("gap_st_gcn_minus_static_pose",
         "temporal margin     (st-gcn - static-pose)  ", gcn, sp),
        # The M2 question: does pretraining close the gap to the forest?
        ("gap_stgcnpp_pretrained_minus_random_forest",
         "pretrained gap  (st-gcn++ pre - random-forest)", ppp, rf),
        ("gap_stgcnpp_pretrained_minus_scratch",
         "pretraining effect (st-gcn++ pre - scratch)   ", ppp, pps),
        # Without this one, "the architecture alone beats the forest" is only a
        # point estimate, and pretraining and architecture cannot be separated.
        ("gap_stgcnpp_scratch_minus_random_forest",
         "architecture gap (st-gcn++ scratch - random-forest)", pps, rf),
    ]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="root containing Skeleton_data/")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="permit writing synthetic numbers to the results file")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/baseline.json"))
    ap.add_argument("--balanced", action="store_true",
                    help="inverse-frequency class weighting on every fitted "
                         "arm, interpretable and learned alike")
    ap.add_argument("--bootstrap", type=int, default=2000, metavar="N",
                    help="paired bootstrap resamples for the macro-F1 CIs "
                         "(st-gcn vs random-forest, and st-gcn vs the "
                         "static-pose floor); 0 to skip")
    ap.add_argument("--layout-only", action="store_true",
                    help="probe the on-disk layout and stop")
    ap.add_argument("--include-v6", action="store_true",
                    help="re-segment V6 from its raw per-frame array; off by "
                         "default because that segmentation is ours, not the "
                         "authors'")
    ap.add_argument("--stgcnpp", choices=["off", "scratch", "pretrained", "both"],
                    default="off",
                    help="add the ST-GCN++ arm: from scratch, fine-tuned from "
                         "--pretrained, or both. Off by default, so existing runs "
                         "reproduce unchanged")
    ap.add_argument("--pretrained", type=Path, metavar="PTH",
                    help="PYSKL ST-GCN++ checkpoint (see boxingvi.stgcnpp.PRETRAINED)")
    ap.add_argument("--pretrained-spec", default="ntu120_xsub_hrnet_j",
                    help="which PRETRAINED entry to verify the checkpoint against")
    ap.add_argument("--t-out", type=int, default=50,
                    help="frames each clip is resampled to for ST-GCN++")
    ap.add_argument("--stgcnpp-epochs", type=int, default=None,
                    help="epochs for the ST-GCN++ arm (default: --epochs)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)

    if not (a.data or a.synthetic):
        ap.error("pass --data or --synthetic")
    spec = None
    if a.stgcnpp in ("pretrained", "both"):
        from boxingvi.stgcnpp import PRETRAINED
        if not a.pretrained:
            ap.error("--stgcnpp pretrained/both needs --pretrained PTH")
        if a.pretrained_spec not in PRETRAINED:
            ap.error(f"unknown --pretrained-spec; choose from {sorted(PRETRAINED)}")
        spec = PRETRAINED[a.pretrained_spec]
        # Deny by default: the checkpoint is a pickle from the network, and md5
        # alone is not a pin. An entry with no sha256 has never been verified.
        if not spec.get("sha256"):
            ap.error(f"--pretrained-spec {a.pretrained_spec} has no pinned sha256; "
                     "pin it after a verified download before using it")

    if a.synthetic:
        from boxingvi.synthetic import make_corpus
        print("SYNTHETIC FIXTURES -- not a result about BoxingVI")
        clips = make_corpus(seed=a.seed)
        train, val = split_by_subject(clips)
        split_desc = "subject-disjoint S1-S15 / S16-S20 (synthetic fixtures)"
    else:
        from boxingvi.data import (HELD_OUT_VIDEOS, discover_layout,
                                   load_corpus, split_by_video)
        print(json.dumps(discover_layout(a.data), indent=2))
        if a.layout_only:
            raise SystemExit("--layout-only requested; stopping after the probe")
        clips = load_corpus(a.data, include_v6=a.include_v6)
        train, val = split_by_video(clips)
        # Stated in full, and carried into the results file, because the
        # difference between this and the paper's split is the single most
        # misreadable thing about any number produced here.
        split_desc = (
            "video-disjoint, held out " + "/".join(HELD_OUT_VIDEOS)
            + " -- NOT the paper's official subject-disjoint S1-S15 / S16-S20 "
              "split, which the release does not carry the subject ids to "
              "reproduce (ISSUES-LOG O10)"
        )
        print(f"split: {split_desc}")

    xtr, ytr = to_arrays(train)
    xva, yva = to_arrays(val)
    disjoint = "subject-disjoint" if a.synthetic else "video-disjoint"
    print(f"train {len(train)} clips / val {len(val)} clips  "
          f"({disjoint})  x={xtr.shape}")

    if a.balanced:
        print("class weighting: ON for every fitted arm "
              "(inverse frequency, n / (k * count_k))")
    results = run_floors(ytr, yva, a.seed)
    results += run_interpretable([c.keypoints for c in train], ytr,
                                 [c.keypoints for c in val], yva, a.seed,
                                 balanced=a.balanced)
    results.append(run_learned(xtr, ytr, xva, yva, a.epochs, a.seed,
                               static=True, balanced=a.balanced))
    results.append(run_learned(xtr, ytr, xva, yva, a.epochs, a.seed,
                               balanced=a.balanced))
    pp_epochs = a.stgcnpp_epochs or a.epochs
    if a.stgcnpp in ("scratch", "both"):
        results.append(run_stgcnpp(xtr, ytr, xva, yva, pp_epochs, a.seed,
                                   a.t_out, balanced=a.balanced))
    if a.stgcnpp in ("pretrained", "both"):
        results.append(run_stgcnpp(xtr, ytr, xva, yva, pp_epochs, a.seed,
                                   a.t_out, pretrained=a.pretrained, spec=spec,
                                   balanced=a.balanced))

    print(f"\n{'model':<22}{'acc':>8}{'macroF1':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['model']:<22}{r['accuracy']:>8.3f}{r['macro_f1']:>10.3f}")

    rf = next(r for r in results if r["model"] == "random-forest")
    gcn = next(r for r in results if r["model"] == "st-gcn")
    gap = gcn["macro_f1"] - rf["macro_f1"]
    print(f"\ninterpretability cost (st-gcn - random-forest): {gap:+.3f} macro F1")

    # Two intervals, because the project makes two separate claims and both
    # need one. The interpretability gap is the headline. The margin over the
    # static-pose floor is the one that decides whether the temporal model can
    # be said to use time at all -- and at roughly 0.06 accuracy that claim is
    # small enough that an interval including zero would retire it.
    cis: dict = {}
    if a.bootstrap:
        from boxingvi.stats import bootstrap_gap_ci
        for key, label, high, low in interval_comparisons(results):
            if not (high and low and "_pred" in high and "_pred" in low):
                continue
            # Both metrics, because they can disagree about whether a claim
            # survives: the temporal margin is 0.189 macro F1 but 0.061
            # accuracy, and only the second one ever looked fragile.
            for metric in ("macro_f1", "accuracy"):
                ci = bootstrap_gap_ci(
                    yva, np.array(high["_pred"]), np.array(low["_pred"]),
                    n_boot=a.bootstrap, seed=a.seed, metric=metric)
                cis[f"{key}__{metric}"] = ci
                verdict = ("excludes zero" if ci["excludes_zero"]
                           else "INCLUDES ZERO -- not a claim")
                print(f"  {label} [{metric:>9}]: {ci['point']:+.3f}  "
                      f"95% CI [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  "
                      f"-- {verdict}")

    if a.synthetic and not a.allow_synthetic:
        print("\nnot written: synthetic run without --allow-synthetic")
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"synthetic": bool(a.synthetic), "seed": a.seed, "epochs": a.epochs,
         "split": split_desc, "balanced": bool(a.balanced),
         "n_train_clips": len(train), "n_val_clips": len(val),
         "train_videos": sorted({c.source for c in train}) or None,
         "val_videos": sorted({c.source for c in val}) or None,
         "include_v6": bool(a.include_v6),
         "gap_st_gcn_minus_random_forest":
             cis.get("gap_st_gcn_minus_random_forest__macro_f1"),
         "gap_st_gcn_minus_random_forest_accuracy":
             cis.get("gap_st_gcn_minus_random_forest__accuracy"),
         "gap_st_gcn_minus_static_pose":
             cis.get("gap_st_gcn_minus_static_pose__macro_f1"),
         "gap_st_gcn_minus_static_pose_accuracy":
             cis.get("gap_st_gcn_minus_static_pose__accuracy"),
         "stgcnpp": a.stgcnpp,
         "gap_stgcnpp_pretrained_minus_random_forest":
             cis.get("gap_stgcnpp_pretrained_minus_random_forest__macro_f1"),
         "gap_stgcnpp_pretrained_minus_scratch":
             cis.get("gap_stgcnpp_pretrained_minus_scratch__macro_f1"),
         "gap_stgcnpp_scratch_minus_random_forest":
             cis.get("gap_stgcnpp_scratch_minus_random_forest__macro_f1"),
         "stgcnpp_epochs": pp_epochs if a.stgcnpp != "off" else None,
         # Every interval on both metrics, keyed "<comparison>__<metric>".
         "intervals": cis,
         "results": [{k: v for k, v in r.items()
                      if not k.startswith("_")} for r in results]}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
