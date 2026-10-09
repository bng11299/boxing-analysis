"""Train the punch-type classifier the pipeline uses, and save it with a manifest.

    python train_classifier.py --data data --pretrained weights/stgcnpp_ntu120_xsub_hrnet_j.pth

This is the reported model, not a new one: ST-GCN++ fine-tuned from the pinned
NTU-120 checkpoint, seed 0, 30 epochs, t_out 50, on the video-disjoint training
side (V2/V5/V9 held out), through the SAME function run_baseline.py scores
(`run_stgcnpp`). It refuses to write a manifest unless the held-out accuracy and
macro F1 it just measured equal the reported run's (results/stgcnpp.json, 0.8145
/ 0.7207 on seed 0), because a saved model that does not reproduce the reported
number is a different model under the same name.

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
from boxingvi.data import HELD_OUT_VIDEOS, load_corpus, split_by_video, to_arrays  # noqa: E402
from boxingvi.stgcnpp import PRETRAINED  # noqa: E402
from run_baseline import run_stgcnpp  # noqa: E402

SPEC = "ntu120_xsub_hrnet_j"
DEFAULT_OUT = Path("weights/classifier/stgcnpp-boxingvi-seed0.pt")
DEFAULT_REFERENCE = Path("results/stgcnpp.json")
DEFAULT_RULE = Path("results/punch_finder_v6.json")
TOL = 1e-9


def reference_metrics(path: Path, seed: int, epochs: int) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("seed") != seed or d.get("epochs") != epochs or d.get("balanced") or d.get("include_v6"):
        raise ValueError(f"{path} is not a seed-{seed}, {epochs}-epoch, unbalanced, V6-excluded run")
    r = [r for r in d["results"] if r.get("model") == "st-gcn++ pretrained"]
    if len(r) != 1:
        raise ValueError(f"{path} has no single 'st-gcn++ pretrained' result")
    return {"accuracy": r[0]["accuracy"], "macro_f1": r[0]["macro_f1"]}


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

    seed, epochs, t_out = 0, 30, 50
    variant = json.loads(a.rule.read_text(encoding="utf-8"))["chosen"]["variant"]
    clips = load_corpus(a.data)
    train, val = split_by_video(clips)
    geom = training_geometry(train, variant)
    print(f"window geometry from {geom.n_clips} training clips: {geom}")
    print(f"  -> {geom.n_frames} frames per window at {geom.fps:.3f} fps")
    if a.geometry_only:
        return 0

    ref = reference_metrics(a.reference, seed, epochs)
    xtr, ytr = to_arrays(train)
    xva, yva = to_arrays(val)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".partial")
    r = run_stgcnpp(xtr, ytr, xva, yva, epochs=epochs, seed=seed, t_out=t_out,
                    pretrained=a.pretrained, spec=PRETRAINED[SPEC], save_to=tmp)
    print(f"held out {'/'.join(HELD_OUT_VIDEOS)}: accuracy {r['accuracy']:.4f}, macro F1 {r['macro_f1']:.4f} "
          f"(reported {ref['accuracy']:.4f} / {ref['macro_f1']:.4f})")
    if abs(r["accuracy"] - ref["accuracy"]) > TOL or abs(r["macro_f1"] - ref["macro_f1"]) > TOL:
        tmp.unlink(missing_ok=True)
        print("REFUSED: this run does not reproduce the reported model; nothing saved", file=sys.stderr)
        return 1
    tmp.replace(a.out)
    m = write_manifest(a.out.with_suffix(".json"), a.out, geom, t_out, {
        "trained_on": "BoxingVI video-disjoint training side; held out " + "/".join(HELD_OUT_VIDEOS),
        "seed": seed, "epochs": epochs,
        "held_out": {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                     "per_class_f1": r.get("per_class_f1")},
        "reproduces": str(a.reference.as_posix()),
        "pretrained_from": r.get("pretrained_from"),
        "licence": "NTU RGB+D 120-derived weights: non-commercial; never released",
    })
    print(f"saved {a.out} (sha256 {m['sha256'][:12]}...) and its manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
