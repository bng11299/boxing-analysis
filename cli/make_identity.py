"""Write the identity file for analyse_clip.py --two-boxer from a person's choice of track ids.

  python cli/run_ingest.py --video X.mp4 --track ...            # X.tracks.npz and a tracked overlay with the ids
  python cli/make_identity.py --tracks X.tracks.npz --a 1 4 --b 2 [--out X.identity.csv]
  python cli/analyse_clip.py --two-boxer --video X.mp4 --tracks X.tracks.npz --identity X.identity.csv ...

Until now this step was a hand-run call to boxingvi.boxers.write_identity. Watch the tracked overlay, note which
track ids are boxer A and which are boxer B, and pass them here. Every row of an id given to --a is boxer A, of an
id given to --b boxer B; any other id (referee, crowd) is nobody. The method is recorded as "by-hand" and the gate
as "ungated": no automatic check stands behind a person's reading of an overlay, and every per-boxer report says
so in its first sentence.

Writes <clip>.identity.csv and its .json sidecar (default: beside the tracks file). Inputs under a participant
folder must be written under it (boxingvi.boxers.check_output_location).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import boxers  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402

METHOD = "by-hand"


def _track_id(text: str) -> int:
    if not text.isdigit() or len(text) > 9:
        raise argparse.ArgumentTypeError(f"{text!r}: a track id is a whole number")
    return int(text)


def main(argv=None, participant_roots=boxers.PARTICIPANT_ROOTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, required=True, help="<clip>.tracks.npz from run_ingest.py --track")
    ap.add_argument("--a", type=_track_id, nargs="+", required=True, help="the track ids that are boxer A")
    ap.add_argument("--b", type=_track_id, nargs="+", required=True, help="the track ids that are boxer B")
    ap.add_argument("--out", type=Path, help="<clip>.identity.csv (default: beside the tracks file)")
    a = ap.parse_args(argv)
    both = set(a.a) & set(a.b)
    if both:
        raise ValueError(f"track ids {sorted(both)} are given to both boxers")
    table = PoseTable.load(a.tracks)
    present = set(int(t) for t in table.track_id.tolist())
    missing = sorted((set(a.a) | set(a.b)) - present)
    if missing:
        raise ValueError(f"track ids {missing} are not in {a.tracks.name}")
    clip = boxers.clip_of(a.tracks)
    out = a.out or a.tracks.with_name(f"{clip}.identity.csv")
    boxers.check_output_location((a.tracks,), out.parent, participant_roots)
    names = ["A" if t in a.a else "B" if t in a.b else None for t in table.track_id.tolist()]
    boxers.write_identity(out, table, a.tracks, boxers.segments_from_row_names(table, names), METHOD,
                          {"verdict": "ungated"})
    print(f"-> {out} (method {METHOD}, ungated; A = ids {sorted(a.a)}, B = ids {sorted(a.b)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
