"""Find and name each boxer's punches in two-boxer footage, from a tracks file and an identity file.

  python boxer_punches.py --tracks results/ingest/X.tracks.npz --identity results/identity/X.identity.csv \\
      --stance-a orthodox --stance-b southpaw --classifier weights/classifier/stgcnpp-boxingvi-seed0.json
  ... --gate-result G.json      # required when the identity records a gate PASS

The identity file (boxingvi.boxers, vault ADR-0016) says which tracked person is boxer A, B or unsure, segment by
segment; its sidecar binds it to this tracks file and records the identity method's gate verdict. Both are
checked before anything runs, and refused on any mismatch.

Each boxer's rows are scanned by the punch finder segment by segment, never across a change of track id, and
every punch is named as classify_punches.py names a clip's: one cut pooled from that boxer's own punches, each
punch read on the segment it was found on, lead and rear from THAT boxer's stance. Both stances are required:
the tool never guesses one. Punches on unsure segments are found and counted, never named or assigned.

Writes <clip>.boxers.csv (A's and B's punches: boxer and track id, then classify_punches.py's columns),
<clip>.unsure.csv and the sidecar <clip>.boxers.json to --out-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import classify_punches  # noqa: E402
import find_punches  # noqa: E402
from boxingvi import boxers, classify  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402

DEFAULT_OUT = Path("results/boxers")


def main(argv=None, participant_roots=boxers.PARTICIPANT_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True, help="<clip>.identity.csv, its .json beside it")
    ap.add_argument("--gate-result", type=Path, help="the identity method's gate result (needed for a PASS)")
    ap.add_argument("--stance-a", choices=sorted(classify.STANCES), required=True)
    ap.add_argument("--stance-b", choices=sorted(classify.STANCES), required=True)
    ap.add_argument("--classifier", type=Path, required=True, help="the manifest .json next to the checkpoint")
    ap.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--extent", type=Path, default=classify_punches.DEFAULT_EXTENT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    boxers.check_output_location((a.tracks, a.identity, a.gate_result), a.out_dir, participant_roots)
    table = PoseTable.load(a.tracks)
    clip = boxers.clip_of(a.tracks)
    identity = boxers.read_identity(a.identity, table, a.tracks, a.gate_result)
    model, man = classify.load_classifier(a.classifier)
    geom = man["geometry"]
    rule = find_punches.load_rule(a.rule)
    if rule["variant"] != geom.variant:
        raise ValueError("the punch finder's rule uses another wrist-speed variant than the classifier's window "
                         f"({geom.variant}); re-run one of them to match")
    ex = classify.read_extent(a.extent, geom.variant)
    stances = {"A": a.stance_a, "B": a.stance_b}

    found = boxers.find(table, identity, rule)
    named, cuts = {}, {}
    for b in boxers.BOXERS:
        named[b], cuts[b] = boxers.name_punches(table, found[b]["kept"], model, man, ex, stances[b])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    p, up = boxers.write_boxers(a.out_dir, clip, table, a.tracks, identity, found, named, cuts, stances, man,
                                rule, ex)
    n_unsure = sum(len(k[3]) for k in found[boxers.UNSURE]["kept"])
    print(f"{clip}: identity by {identity.method} (gate: {identity.verdict})")
    for b in boxers.BOXERS:
        n = len(named[b])
        unnamed = sum(r[7] == "unclassified" for r in named[b])   # r: track id, the classified columns
        print(f"  boxer {b} ({stances[b]}): {n} punches, {n - unnamed} named, {unnamed} unnamed; "
              f"{found[b]['stats']['identified_frames']} frames identified")
    print(f"  unsure segments: {n_unsure} punches, counted, never assigned")
    print(f"  -> {p}, {up.name} (+ {clip}.boxers.json provenance)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
