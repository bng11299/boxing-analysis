"""The report a boxer reads: a spreadsheet and a written summary for one clip.

  python report_punches.py --pose results/ingest/IMG_4406.pose.npz \\
      --classified results/punches/IMG_4406.classified.csv \\
      --classifier weights/classifier/stgcnpp-boxingvi-seed0.json
  ... --rounds 0-180,240-420                      # round boundaries in seconds
  ... --score results/punches/IMG_4406.classify_score.json   # after labelling the clip

Writes results/reports/<clip>.report.xlsx (sheets: Punches, Rounds, Reliability,
About) and <clip>.summary.txt. Refuses unless the CSV is unchanged since
classify_punches.py wrote it, was named by the classifier --classifier describes,
from this pose (same frames, same fps) and with the --rule finder rule (the
<clip>.classified.json beside it records all of that); a --score must have scored
this very CSV. The summary is fixed templates filled with the
counts; its reliability warnings come from measured numbers (see boxingvi.report),
and it says so when nothing has been measured on your own footage. It leads with a
warning when the hand each punch type implies matches the wrist that moved less
often than the report requires (report.HAND_FLOOR), or not significantly more
often than chance (one-sided exact binomial and Fisher tests, 0.05), and with a
caution when the check cannot confirm the types at all.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import json  # noqa: E402

from boxingvi import boxers, combos, narrate, report  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402

DEFAULT_REFERENCE = Path("results/sweep-2026-09-22/seed0.json")   # the run train_classifier.py reproduces
DEFAULT_RULE = Path("results/punch_finder_v6.json")
DEFAULT_COMBO_GATE = Path("results/combinations_gate.json")        # score_combinations.py writes it
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
    ap.add_argument("--combo-gate", type=Path, default=DEFAULT_COMBO_GATE,
                    help="score_combinations.py's gate result (used only if it matches this classifier, rule, gap)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prompt-out", action="store_true",
                    help="also write <clip>.narration-prompt.json: a prompt for an optional LLM narrator, checked "
                         "afterwards by narration_check.py (boxingvi.narrate); own footage only")
    a = ap.parse_args(argv)

    pose = PoseSequence.load(a.pose)
    clip = a.pose.name.split(".")[0]
    if a.classified.name.split(".")[0] != clip:
        raise ValueError(f"{a.classified.name} does not name the clip {clip}")
    sidecar = a.classified.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"no {sidecar.name} beside the CSV: re-run classify_punches.py, which writes it")
    man = report.verify_provenance(a.classified, sidecar, a.classifier, clip)
    side = man["_sidecar"]
    report.check_pose(side, len(pose), pose.fps)   # frames AND frame rate
    duration = len(pose) / pose.fps
    rows = report.read_classified(a.classified, duration, fps=pose.fps)
    rounds = report.parse_rounds(a.rounds, duration)
    bench = report.benchmark_reliability(a.reference, man)
    finder = report.finder_reliability(a.rule, side.get("rule"))
    own = report.own_reliability(a.score, clip, man["sha256"], side["csv_sha256"]) if a.score else None

    facts = report.compute_facts(rows, rounds, duration)
    runs = combos.group(combos.solo_punches(rows, rounds), pose.fps)
    # classify_punches.py records the finder mode; a sidecar without it never gets the pooled gate's result
    mode = side.get("finder_per_hand")
    cf = combos.compute(runs, report._rule_of(side.get("rule"), "the sidecar")["min_gap_s"], mode is True)
    gate = (combos.read_gate(a.combo_gate, side.get("rule"), man["sha256"],
                             mode if isinstance(mode, bool) else None) if a.combo_gate.is_file() else None)
    w = report.summary_writer(facts, bench, finder, own, combinations=(cf, gate, False))
    text, numbers = w.text(), w.numbers
    if Counter(report.numbers_in(text)) != Counter(t for n in numbers for t in report.numbers_in(n)):
        raise RuntimeError("a number in the summary did not come from its templates")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sheet = combos.sheet(runs, lambda p: p["unit"] + 1)
    report.write_workbook(a.out_dir / f"{clip}.report.xlsx", clip, rows, rounds, facts, bench, text, pose.fps,
                          report.cut_line(side), extra=(("Combinations", sheet),))
    (a.out_dir / f"{clip}.summary.txt").write_text(text + "\n", encoding="utf-8")
    if a.prompt_out:
        # participant-derived numbers never go to a cloud model (vault O63-O65): the file says so
        local = any(boxers.faceblur._under(Path(x).resolve(), Path(r).resolve())
                    for x in (a.pose, a.classified) for r in boxers.PARTICIPANT_ROOTS)
        boxers.replace_bytes(a.out_dir / f"{clip}.narration-prompt.json", json.dumps(
            narrate.build(w, "your", may_leave_machine=not local), indent=1, ensure_ascii=False).encode("utf-8"))
        print(f"-> {clip}.narration-prompt.json" + (" (participant data: a LOCAL model only)" if local else ""))
    print(text)
    print(f"-> {a.out_dir / f'{clip}.report.xlsx'} and {clip}.summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
