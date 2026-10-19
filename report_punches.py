"""The report a boxer reads: a spreadsheet and a written summary for one clip.

  python report_punches.py --pose results/ingest/IMG_4406.pose.npz \\
      --classified results/punches/IMG_4406.classified.csv \\
      --classifier weights/classifier/stgcnpp-boxingvi-seed0.json
  ... --rounds 0-180,240-420                      # round boundaries in seconds
  ... --score results/punches/IMG_4406.classify_score.json   # after labelling the clip

Writes results/reports/<clip>.report.xlsx (sheets: Punches, Rounds, Reliability,
About) and <clip>.summary.txt. Refuses unless the CSV is unchanged since
classify_punches.py wrote it and was named by the classifier --classifier describes
(the <clip>.classified.json beside it records both hashes). The summary is fixed templates filled with the
counts; its reliability warnings come from measured numbers (see boxingvi.report),
and it says so when nothing has been measured on your own footage. It leads with a
warning when the hand each punch type implies matches the wrist that moved not
significantly more often than chance (one-sided exact binomial test, 0.05).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import report  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402

DEFAULT_REFERENCE = Path("results/sweep-2026-09-22/seed0.json")   # the run train_classifier.py reproduces
DEFAULT_RULE = Path("results/punch_finder_v6.json")
DEFAULT_OUT = Path("results/reports")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", type=Path, required=True, help="the clip's pose file (for its length and name)")
    ap.add_argument("--classified", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True, help="the manifest of the model that classified it")
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                    help="the reported classifier run whose held-out confusion matrix is the reliability")
    ap.add_argument("--rule", type=Path, default=DEFAULT_RULE, help="the punch finder's tuned rule")
    ap.add_argument("--score", type=Path, help="classify_score.json for this clip, if you labelled it")
    ap.add_argument("--rounds", help="round boundaries in seconds, e.g. 0-180,240-420 (default: the whole clip)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    pose = PoseSequence.load(a.pose)
    clip = a.pose.name.split(".")[0]
    if a.classified.name.split(".")[0] != clip:
        raise ValueError(f"{a.classified.name} does not name the clip {clip}")
    sidecar = a.classified.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"no {sidecar.name} beside the CSV: re-run classify_punches.py, which writes it")
    man = report.verify_provenance(a.classified, sidecar, a.classifier, clip)
    if man["_sidecar"].get("pose_frames") != len(pose):
        raise ValueError(f"{a.classified.name} was classified on a {man['_sidecar'].get('pose_frames')}-frame "
                         f"pose; this one has {len(pose)}")
    duration = len(pose) / pose.fps
    rows = report.read_classified(a.classified, duration)
    rounds = report.parse_rounds(a.rounds, duration)
    bench = report.benchmark_reliability(a.reference, man)
    finder = report.finder_reliability(a.rule)
    own = report.own_reliability(a.score, clip, man["sha256"]) if a.score else None

    facts = report.compute_facts(rows, rounds, duration)
    text, _ = report.summary(facts, bench, finder, own)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report.write_workbook(a.out_dir / f"{clip}.report.xlsx", clip, rows, rounds, facts, bench, text, pose.fps,
                          report.cut_line(man["_sidecar"]))
    (a.out_dir / f"{clip}.summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"-> {a.out_dir / f'{clip}.report.xlsx'} and {clip}.summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
