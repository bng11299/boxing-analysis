"""One report per boxer: a spreadsheet and a written summary for each of the two boxers in a clip.

  python report_boxers.py --tracks results/ingest/X.tracks.npz --boxers results/boxers/X.boxers.csv \\
      --identity results/identity/X.identity.csv \\
      --classifier weights/classifier/stgcnpp-boxingvi-seed0.json
  ... --rounds 0-180,240-420          # round boundaries in seconds
  ... --gate-result G.json            # required when the identity records a gate PASS

Writes <clip>.A.report.xlsx, <clip>.A.summary.txt and the same for B to --out-dir. Each is the solo report
(report_punches.py: the same template summary, the same sheets, the same hand-check warning) for that boxer's
punches, plus an Identity sheet, led by a header that says which method told the boxers apart, its gate verdict,
how long this boxer was identified and what share of the punches found could not be assigned to either boxer.
When the method failed its gate or has none, the FIRST sentence warns that the split between the boxers is
unverified.

Refuses unless the identity file checks out against the tracks file (boxingvi.boxers.read_identity), and the
punches are the ones boxer_punches.py wrote (hashes), from this tracks file, with this identity, by this
classifier and finder rule, and are EXACTLY what the finder finds again here. Every number in a summary is
checked to have come from the templates.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import find_punches  # noqa: E402
import report_punches  # noqa: E402
from boxingvi import boxers, classify, report  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402

DEFAULT_OUT = Path("results/reports")


def boxer_text(boxer: str, identity, stats: dict, facts: dict, n_unsure: int, n_all: int, duration_s: float,
               fps: float, bench: dict, finder: dict) -> "tuple[str, list[str]]":
    """The header, then the solo summary of this boxer's punches; every number through the templates."""
    head, n_head = boxers.header(boxer, identity.method, identity.verdict, n_unsure, n_all,
                                 stats["identified_frames"] / fps, duration_s)
    body, n_body = report.summary(facts, bench, finder, None)
    text, numbers = head + " " + body, n_head + n_body
    if Counter(report.numbers_in(text)) != Counter(t for n in numbers for t in report.numbers_in(n)):
        raise RuntimeError("a number in the summary did not come from its templates")
    return text, numbers


def main(argv=None, participant_roots=boxers.PARTICIPANT_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--boxers", type=Path, required=True, help="<clip>.boxers.csv from boxer_punches.py")
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--gate-result", type=Path, help="the identity method's gate result (needed for a PASS)")
    ap.add_argument("--classifier", type=Path, required=True, help="the manifest of the model that named them")
    ap.add_argument("--reference", type=Path, default=report_punches.DEFAULT_REFERENCE)
    ap.add_argument("--rule", type=Path, default=report_punches.DEFAULT_RULE)
    ap.add_argument("--rounds", help="round boundaries in seconds, e.g. 0-180,240-420 (default: the whole clip)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    boxers.check_output_location((a.tracks, a.boxers, a.identity, a.gate_result), a.out_dir, participant_roots)
    table = PoseTable.load(a.tracks)
    clip = boxers.clip_of(a.tracks)
    identity = boxers.read_identity(a.identity, table, a.tracks, a.gate_result)
    man = classify.read_manifest(a.classifier)
    got = boxers.read_boxers(a.boxers, table, a.tracks, identity, man)
    side = got["side"]
    finder = report.finder_reliability(a.rule, side.get("rule"))
    found = boxers.find(table, identity, find_punches.load_rule(a.rule))
    boxers.check_against_finder(got, found, table, man)
    bench = report.benchmark_reliability(a.reference, man)
    duration = table.n_frames / table.fps
    rounds = report.parse_rounds(a.rounds, duration)

    facts = {b: report.compute_facts(got["rows"][b], rounds, duration) for b in boxers.BOXERS}
    n_unsure = sum(report.round_of(u["time_s"], rounds) is not None for u in got["unsure"])
    n_all = facts["A"]["n_found"] + facts["B"]["n_found"] + n_unsure
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for b in boxers.BOXERS:
        stats = found[b]["stats"]
        text, _ = boxer_text(b, identity, stats, facts[b], n_unsure, n_all, duration, table.fps, bench, finder)
        sheet = boxers.identity_sheet(identity, b, stats, n_unsure, n_all)
        boxers.replace_file(a.out_dir / f"{clip}.{b}.report.xlsx", lambda p, b=b, text=text, sheet=sheet: (
            report.write_workbook(p, f"{clip}, boxer {b}", got["rows"][b], rounds, facts[b], bench, text, table.fps,
                                  report.cut_line({"cut": side["cuts"][b]}), extra=(("Identity", sheet),))))
        boxers.replace_bytes(a.out_dir / f"{clip}.{b}.summary.txt", (text + "\n").encode("utf-8"))
        print(f"== boxer {b} ==\n{text}\n")
    print(f"-> {a.out_dir / f'{clip}.A.report.xlsx'}, {clip}.B.report.xlsx and both summaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
