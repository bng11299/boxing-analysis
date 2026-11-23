"""Score the punch-combination gate (docs/gates/combinations.md) on labelled own footage.

  python cli/score_combinations.py --clip IMG_4406 --clip IMG_4395 --punches-dir results/combinations/own_tol0.2 \\
      --labels-dir data/own --pose-dir results/ingest \\
      --classifier weights/classifier/stgcnpp-boxingvi-seed0.json --out results/combinations_gate.json

For each clip: the classified CSV classify_punches.py wrote (checked against its sidecar: unchanged, this
classifier, this pose's frames and fps), the labels import_labels.py wrote, an `unclear` label taken as framing
(not a punch), and detections matched to labels exactly as RESULTS rows 6a/6b match them
(classify_punches.attribute, find_punches.TOL_S, punchfind.POINT_TOL_S). Then the gate's three bars, pooled and
per clip, and a result file the reports read back (combos.read_gate) only for this classifier, finder rule and gap.

Refuses clips classified with different finder rules, the per-hand finder (the gate scores the pooled rule), a
clip given twice, and any clip whose files do not check out. Nothing here is tuned: the rule is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import classify_punches  # noqa: E402
import find_punches  # noqa: E402
from boxingvi import boxers, combos, punchfind, report  # noqa: E402
from boxingvi.lsimport import UNCLEAR_LABEL  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_CLIP = re.compile(r"[A-Za-z0-9_-]{1,64}")


def score_one(clip: str, punches_dir: Path, labels_dir: Path, pose_dir: Path, classifier: Path):
    """(combos.score_clip's dict, provenance, the sidecar's rule, fps) for one clip."""
    pose = PoseSequence.load(pose_dir / f"{clip}.pose.npz")
    csv_p = punches_dir / f"{clip}.classified.csv"
    side_p = csv_p.with_suffix(".json")
    if not side_p.is_file():
        raise ValueError(f"no {side_p.name} beside {csv_p.name}: re-run classify_punches.py")
    man = report.verify_provenance(csv_p, side_p, classifier, clip)
    side = man["_sidecar"]
    if side.get("finder_per_hand") is not False:   # missing, null or anything but false is not the pooled rule
        raise ValueError(f"{csv_p.name} was not found with the pooled finder (finder_per_hand must be false); "
                         "the gate scores the pooled rule")
    report.check_pose(side, len(pose), pose.fps)
    rows = report.read_classified(csv_p, len(pose) / pose.fps, fps=pose.fps)
    lab_p = labels_dir / f"{clip}.labels.json"
    labelled = find_punches.load_labels(lab_p, len(pose), pose.fps)
    spans = [x for x in labelled if x.label != UNCLEAR_LABEL]     # --unclear not-a-punch, as rows 6a/6b
    tol = int(round(find_punches.TOL_S * pose.fps))
    point_tol = int(round(punchfind.POINT_TOL_S * pose.fps))
    pairs = classify_punches.attribute([r["frame"] for r in rows], spans, tol, point_tol)
    sc = combos.score_clip(rows, spans, punchfind.centers(spans), pairs, pose.fps)
    prov = {"classified_csv_sha256": side["csv_sha256"], "labels_sha256": boxers.sha256_file(lab_p),
            "fps": float(pose.fps), "tolerance_frames": tol, "point_tolerance_frames": point_tol}
    return sc, prov, report._rule_of(side.get("rule"), side_p.name), man["sha256"], float(pose.fps)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", action="append", required=True, help="a labelled clip's name (repeat)")
    ap.add_argument("--punches-dir", type=Path, required=True, help="where classify_punches.py wrote the clips")
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--pose-dir", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--permutations", type=int, default=combos.GATE_PERMUTATIONS, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if len(set(a.clip)) != len(a.clip) or not all(_CLIP.fullmatch(c) for c in a.clip):
        raise ValueError("each --clip once, a plain clip name (letters, digits, _ and -)")
    if not (ROOT / combos.GATE_DOC).is_file():
        raise ValueError(f"no pre-registered gate {combos.GATE_DOC} in this repository")
    if not 1 <= a.permutations <= 100_000:
        raise ValueError("permutations must be 1..100000")
    boxers.check_output_location([a.punches_dir, a.labels_dir, a.pose_dir], a.out.parent)
    clips, rule, sha, fps_of = {}, None, None, {}
    for clip in a.clip:
        sc, prov, r, s, fps = score_one(clip, a.punches_dir, a.labels_dir, a.pose_dir, a.classifier)
        if rule is not None and (r != rule or s != sha):
            raise ValueError(f"{clip} was found or named differently from {a.clip[0]}: one rule and one "
                             "classifier per gate result")
        rule, sha = r, s
        clips[clip], fps_of[clip] = (sc, prov), fps
    res = combos.gate_result(clips, rule, sha, fps_of, n_perm=a.permutations)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    boxers.replace_bytes(a.out, json.dumps(res, indent=1, allow_nan=False).encode("utf-8"))
    p = res["pooled"]
    print(f"exact recall {p['recovered_exactly']} of {p['named_label_combinations']} ({p['exact_recall']:.3f}); "
          f"floor mean {p['floor_mean']:.3f}, {combos.GATE_PERCENTILE}th percentile {p['floor_p975']:.3f}")
    for c, d in res["clips"].items():
        print(f"  {c}: label combinations {d['label_combinations']} (named {d['named_label_combinations']}, "
              f"{d['label_combinations_with_a_gap_under_min_gap']} with a gap under the finder's minimum); "
              f"found {d['found_combinations']}; exact {d['recovered_exactly']}, structure {d['structure_recovered']}"
              f"; favourite {d['favourite_found']} vs {d['favourite_labels']}; "
              f"starter {d['starter_found']} vs {d['starter_labels']}")
    print(f"bars {res['bars']} -> {'PASS' if res['PASS'] else 'FAIL'}")
    print(f"-> {a.out} (sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
