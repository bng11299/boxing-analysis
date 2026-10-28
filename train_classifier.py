"""Train the punch-type classifier the pipeline uses, and save it with a manifest.

    python train_classifier.py --data data --pretrained weights/stgcnpp_ntu120_xsub_hrnet_j.pth

This is the reported model, not a new one: ST-GCN++ fine-tuned from the pinned
NTU-120 checkpoint, seed 0, t_out 50, on the video-disjoint training side
(V2/V5/V9 held out), through the SAME function run_baseline.py scores
(`run_stgcnpp`), for the number of epochs the reference run RECORDS (20 for the
seed-0 sweep run, results/sweep-2026-09-22/seed0.json). It refuses to write a
manifest unless the held-out accuracy and macro F1 it just measured equal the
reference's (0.8145 / 0.7207), because a saved model that does not reproduce the
reported number is a different model under the same name.

Why the reference must record its epochs: results/stgcnpp.json (09-16) has
`epochs: 30` for the other arms and no `stgcnpp_epochs`, while its log shows
ST-GCN++ ran 20. Reading the missing field as "same as epochs" trained 30 epochs
and scored 0.7951 / 0.7141 (2026-10-09); this script refused to save it.

The manifest records the saved model's held-out confusion matrix (2026-10-21):
report_punches.py takes its per-type reliability from the reference only when
the two matrices are equal cell for cell. A manifest written before then has no
matrix, and the report refuses it until this script is run again.

The manifest also carries the window geometry (classify.training_geometry),
measured on the training clips only, with the wrist-speed variant the punch
finder's tuned rule uses.

Licence: the backbone started from NTU RGB+D 120 weights, which are
non-commercial. The checkpoint lives under weights/ (gitignored) and is never
released.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi.classify import training_geometry, write_manifest  # noqa: E402
from boxingvi.data import PIPELINE_HELD_OUT_VIDEOS, load_corpus, split_pipeline, to_arrays  # noqa: E402
from boxingvi.stgcnpp import PRETRAINED  # noqa: E402
from run_baseline import run_stgcnpp  # noqa: E402

SPEC = "ntu120_xsub_hrnet_j"
DEFAULT_OUT = Path("weights/classifier/stgcnpp-boxingvi-seed0.pt")
DEFAULT_REFERENCE = Path("results/sweep-2026-09-22/seed0.json")
DEFAULT_RULE = Path("results/punch_finder_v6.json")
TOL = 1e-9


def reference_metrics(path: Path, seed: int, t_out: int) -> dict:
    """The reported run this one must reproduce, checked to be the same recipe.
    Returns its metrics and the ST-GCN++ epochs it recorded."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("seed") != seed or d.get("balanced") or d.get("include_v6"):
        raise ValueError(f"{path} is not a seed-{seed}, unbalanced, V6-excluded run")
    epochs = d.get("stgcnpp_epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 1000:
        raise ValueError(f"{path} does not record its ST-GCN++ epochs (stgcnpp_epochs={epochs!r}); "
                         "a reference that does not say how it was trained cannot be reproduced")
    r = [r for r in d["results"] if r.get("model") == "st-gcn++ pretrained"]
    if len(r) != 1:
        raise ValueError(f"{path} has no single 'st-gcn++ pretrained' result")
    if r[0].get("t_out") != t_out or (r[0].get("pretrained_from") or {}).get("sha256") != PRETRAINED[SPEC]["sha256"]:
        raise ValueError(f"{path}: the reported run used another t_out or another pretrained checkpoint")
    return {"accuracy": r[0]["accuracy"], "macro_f1": r[0]["macro_f1"], "epochs": epochs}


def held_out_block(r: dict) -> dict:
    """What the manifest records about the saved model on the held-out videos: its
    scores AND its confusion matrix (rows true, columns predicted, CLASSES order).
    run_stgcnpp saves the model after scoring it, so this is that model's matrix. The
    report requires the reference's matrix to equal it cell for cell; per-class F1
    alone cannot tell a matrix from its transpose."""
    c = r.get("confusion")
    if not (isinstance(c, list) and c and all(isinstance(row, list) and len(row) == len(c) for row in c)):
        raise ValueError("the run returned no square confusion matrix; nothing to record")
    return {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "per_class_f1": r.get("per_class_f1"),
            "confusion": [[int(v) for v in row] for row in c]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pretrained", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                    help="the reported run the saved model must reproduce")
    ap.add_argument("--rule", type=Path, default=DEFAULT_RULE,
                    help="the punch finder's tuned rule; its variant anchors the window")
    ap.add_argument("--geometry-only", action="store_true",
                    help="print the window geometry and stop (no training)")
    a = ap.parse_args(argv)

    seed, t_out = 0, 50
    variant = json.loads(a.rule.read_text(encoding="utf-8"))["chosen"]["variant"]
    clips = load_corpus(a.data)
    # The checkpoint's own split, kept so it reproduces; NOT athlete-disjoint (O55).
    train, val = split_pipeline(clips)
    geom = training_geometry(train, variant)
    print(f"window geometry from {geom.n_clips} training clips: {geom}")
    print(f"  -> {geom.n_frames} frames per window at {geom.fps:.3f} fps")
    if a.geometry_only:
        return 0

    ref = reference_metrics(a.reference, seed, t_out)
    epochs = ref["epochs"]
    print(f"reproducing {a.reference.as_posix()}: seed {seed}, {epochs} epochs, t_out {t_out}")
    xtr, ytr = to_arrays(train)
    xva, yva = to_arrays(val)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".partial")
    r = run_stgcnpp(xtr, ytr, xva, yva, epochs=epochs, seed=seed, t_out=t_out,
                    pretrained=a.pretrained, spec=PRETRAINED[SPEC], save_to=tmp)
    print(f"held out {'/'.join(PIPELINE_HELD_OUT_VIDEOS)}: accuracy {r['accuracy']:.4f}, macro F1 {r['macro_f1']:.4f} "
          f"(reported {ref['accuracy']:.4f} / {ref['macro_f1']:.4f})")
    if abs(r["accuracy"] - ref["accuracy"]) > TOL or abs(r["macro_f1"] - ref["macro_f1"]) > TOL:
        tmp.unlink(missing_ok=True)
        print("REFUSED: this run does not reproduce the reported model; nothing saved", file=sys.stderr)
        return 1
    tmp.replace(a.out)
    m = write_manifest(a.out.with_suffix(".json"), a.out, geom, t_out, {
        "trained_on": "BoxingVI video-disjoint training side; held out " + "/".join(PIPELINE_HELD_OUT_VIDEOS),
        "seed": seed, "epochs": epochs,
        "held_out": held_out_block(r),
        "reproduces": str(a.reference.as_posix()),
        "pretrained_from": r.get("pretrained_from"),
        "licence": "NTU RGB+D 120-derived weights: non-commercial; never released",
    })
    print(f"saved {a.out} (sha256 {m['sha256'][:12]}...) and its manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
