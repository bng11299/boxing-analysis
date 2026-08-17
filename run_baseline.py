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


def run_interpretable(xtr_kp, ytr, xva_kp, yva, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    ftr, fva = feature_matrix(xtr_kp), feature_matrix(xva_kp)
    out = []

    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(ftr, ytr)
    r = evaluate("random-forest", yva, rf.predict(fva))
    order = np.argsort(rf.feature_importances_)[::-1][:10]
    r["top_features"] = [
        {"feature": FEATURE_NAMES[i], "importance": float(rf.feature_importances_[i])}
        for i in order
    ]
    out.append(r)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed))
    svm.fit(ftr, ytr)
    out.append(evaluate("svm-rbf", yva, svm.predict(fva)))
    return out


def run_learned(xtr, ytr, xva, yva, epochs=30, seed=0, static=False):
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
    r = evaluate(name, yva, pred)
    r["params"] = model.n_params() if hasattr(model, "n_params") else sum(
        p.numel() for p in model.parameters())
    r["train_seconds"] = round(time.time() - t0, 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="root containing Skeleton_data/")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="permit writing synthetic numbers to the results file")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/baseline.json"))
    a = ap.parse_args()

    if not (a.data or a.synthetic):
        ap.error("pass --data or --synthetic")

    if a.synthetic:
        from boxingvi.synthetic import make_corpus
        print("SYNTHETIC FIXTURES -- not a result about BoxingVI")
        clips = make_corpus(seed=a.seed)
    else:
        from boxingvi.data import discover_layout
        print(json.dumps(discover_layout(a.data), indent=2))
        raise SystemExit(
            "Layout reported above. Real-corpus loading is not wired up until "
            "the on-disk arrangement has been seen -- fill in load_corpus() in "
            "src/boxingvi/data.py against what is printed, then re-run."
        )

    train, val = split_by_subject(clips)
    xtr, ytr = to_arrays(train)
    xva, yva = to_arrays(val)
    print(f"train {len(train)} clips / val {len(val)} clips  "
          f"(subject-disjoint)  x={xtr.shape}")

    results = run_floors(ytr, yva, a.seed)
    results += run_interpretable([c.keypoints for c in train], ytr,
                                 [c.keypoints for c in val], yva, a.seed)
    results.append(run_learned(xtr, ytr, xva, yva, a.epochs, a.seed, static=True))
    results.append(run_learned(xtr, ytr, xva, yva, a.epochs, a.seed))

    print(f"\n{'model':<18}{'acc':>8}{'macroF1':>10}")
    print("-" * 36)
    for r in results:
        print(f"{r['model']:<18}{r['accuracy']:>8.3f}{r['macro_f1']:>10.3f}")

    rf = next(r for r in results if r["model"] == "random-forest")
    gcn = next(r for r in results if r["model"] == "st-gcn")
    gap = gcn["macro_f1"] - rf["macro_f1"]
    print(f"\ninterpretability cost (st-gcn - random-forest): {gap:+.3f} macro F1")

    if a.synthetic and not a.allow_synthetic:
        print("\nnot written: synthetic run without --allow-synthetic")
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"synthetic": bool(a.synthetic), "seed": a.seed, "epochs": a.epochs,
         "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
