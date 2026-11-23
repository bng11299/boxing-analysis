"""Does the random forest beat ST-GCN because of the data, or the model?

M1 found random-forest at 0.510 macro-F1 against ST-GCN's 0.407 on 3,781
training clips (the old V2/V5/V9 split). Two readings fit that: engineered
kinematic features genuinely suit this task, or 217k parameters are starved at
this corpus size. They differ in what they predict as training data grows, so
this sweeps the training set and reports both arms at each size.

It now runs the athlete-disjoint split (data.HELD_OUT_VIDEOS, O55: 2,970
training clips), and the output records which videos were held out.

A rising ST-GCN curve that has not flattened says the gap is a data-size
artifact and the comparison should be stated as such. Two flat curves say the
result is about the models.

    python cli/learning_curve.py --data data
    python cli/learning_curve.py --data data --fracs 0.1,0.5,1.0 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi.data import load_corpus, split_by_video, to_arrays  # noqa: E402
from boxingvi.stats import stratified_subsample  # noqa: E402
from run_baseline import run_interpretable, run_learned  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fracs", default="0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--include-v6", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("results/learning_curve.json"))
    a = ap.parse_args()

    fracs = [float(f) for f in a.fracs.split(",")]
    clips = load_corpus(a.data, include_v6=a.include_v6)
    train, val = split_by_video(clips)
    xtr, ytr = to_arrays(train)
    xva, yva = to_arrays(val)
    kp_tr = [c.keypoints for c in train]
    kp_va = [c.keypoints for c in val]

    print(f"full train {len(train)} clips / val {len(val)} clips")
    print(f"fractions {fracs}  x  {a.seeds} seed(s)\n")

    rows: list[dict] = []
    t_start = time.time()
    for frac in fracs:
        for seed in range(a.seeds):
            # The same subsample feeds both arms, so at every point on the
            # curve the two models see identical training data.
            idx = (np.arange(len(ytr)) if frac >= 1.0
                   else stratified_subsample(ytr, frac, seed))
            sub_y = ytr[idx]
            n = len(idx)

            interp = run_interpretable([kp_tr[i] for i in idx], sub_y,
                                       kp_va, yva, seed)
            rf = next(r for r in interp if r["model"] == "random-forest")
            gcn = run_learned(xtr[idx], sub_y, xva, yva, a.epochs, seed)

            row = {
                "frac": frac, "seed": seed, "n_train": n,
                "random_forest_macro_f1": rf["macro_f1"],
                "st_gcn_macro_f1": gcn["macro_f1"],
                "random_forest_acc": rf["accuracy"],
                "st_gcn_acc": gcn["accuracy"],
                "gap": gcn["macro_f1"] - rf["macro_f1"],
            }
            rows.append(row)
            print(f"  frac {frac:<5} seed {seed}  n={n:<5} "
                  f"RF {row['random_forest_macro_f1']:.3f}  "
                  f"GCN {row['st_gcn_macro_f1']:.3f}  "
                  f"gap {row['gap']:+.3f}")

    print(f"\n{'frac':>6}{'n':>7}{'RF':>9}{'ST-GCN':>9}{'gap':>9}")
    print("-" * 40)
    summary = []
    for frac in fracs:
        sel = [r for r in rows if r["frac"] == frac]
        rf_m = float(np.mean([r["random_forest_macro_f1"] for r in sel]))
        gc_m = float(np.mean([r["st_gcn_macro_f1"] for r in sel]))
        summary.append({"frac": frac, "n_train": sel[0]["n_train"],
                        "random_forest_macro_f1": rf_m,
                        "st_gcn_macro_f1": gc_m, "gap": gc_m - rf_m})
        print(f"{frac:>6}{sel[0]['n_train']:>7}{rf_m:>9.3f}{gc_m:>9.3f}"
              f"{gc_m - rf_m:>+9.3f}")

    first, last = summary[0], summary[-1]
    closing = last["gap"] - first["gap"]
    print(f"\nGCN gain from {first['n_train']} to {last['n_train']} clips: "
          f"{last['st_gcn_macro_f1'] - first['st_gcn_macro_f1']:+.3f} macro F1")
    print(f"RF  gain over the same range: "
          f"{last['random_forest_macro_f1'] - first['random_forest_macro_f1']:+.3f}")
    print(f"gap change: {closing:+.3f} "
          f"({'closing -- consistent with a data-size effect' if closing > 0.02 else 'not closing -- the gap looks structural'})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"fracs": fracs, "seeds": a.seeds, "epochs": a.epochs,
         "n_train_full": len(train), "n_val": len(val),
         "val_videos": sorted({c.source for c in val}),
         "include_v6": bool(a.include_v6),
         "elapsed_seconds": round(time.time() - t_start, 1),
         "runs": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
