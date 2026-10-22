"""Credit every labelled Olympic Boxing punch to a tracked person.

  python olympic_attribute.py --data data/olympic --tracks results/ingest/olympic

For each video with a <GoPro id>.tracks.npz (from `run_ingest.py --track`), each
labelled punch is matched to the person whose wrist on the LABELLED hand comes
closest to the glove box. Three things come out of it, all measured, none assumed:

  * who the boxers are: the two ids that collect the most punches, and how often
    both are on screen together (fragmented ids or a referee would show here;
    confirm by eye on the tracks overlay);
  * the left/right convention, tested and not assumed: the thrower is found from
    EITHER wrist, then the nearest wrist is compared with the label's hand. Under
    a mirrored convention this share would collapse towards zero;
  * how many punches no trustworthy wrist reached, and how many were too close to
    call (both boxers' gloves at the box: blocks, clinches).

Writes results/olympic_attribution.json. Nothing derived from this dataset is published.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import olympic  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402
from boxingvi.video import probe  # noqa: E402

DEFAULT_OUT = Path("results/olympic_attribution.json")


def summarise(attrs: "list[olympic.Attribution]", table: "PoseTable | None" = None) -> dict:
    got = [a for a in attrs if a.track_id is not None]
    n = max(len(attrs), 1)
    by_id = collections.Counter(a.track_id for a in got)
    top2 = [t for t, _ in by_id.most_common(2)]
    out = {"events": len(attrs), "credited": len(got), "credited_share": round(len(got) / n, 3),
           "ambiguous_share": round(sum(a.ambiguous for a in attrs) / n, 3),
           "unreached_share": round(sum(a.track_id is None and not a.ambiguous for a in attrs) / n, 3),
           "nearest_wrist_is_labelled_hand": round(
               sum(bool(a.hand_agrees) for a in got if a.hand_agrees is not None)
               / max(sum(a.hand_agrees is not None for a in got), 1), 3),
           "hand_evidence_events": sum(a.hand_agrees is not None for a in got),
           "hand_tied_excluded": sum(a.hand_agrees is None for a in got),
           # If the margin rule is unfair to contact, ambiguity concentrates in blocks and head shots.
           "ambiguous_share_by_target": {
               t: round(sum(a.ambiguous for a in attrs if a.event.target == t)
                        / max(sum(a.event.target == t for a in attrs), 1), 3)
               for t in ("head", "body", "block", "miss")},
           "punches_by_id": {str(k): v for k, v in by_id.most_common(8)},
           "top2_ids": top2, "top2_share_of_credited": round(sum(by_id[t] for t in top2) / max(len(got), 1), 3)}
    if table is not None and len(top2) == 2:
        a_, b_ = (set(table.frame[table.track_id == t].tolist()) for t in top2)
        out["top2_both_on_screen_share"] = round(len(a_ & b_) / max(len(a_ | b_), 1), 3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="folder of <GoPro id>.tracks.npz files")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    report, every = {}, []
    for t in olympic.find_tasks(a.data):
        f = a.tracks / f"{t.video.stem}.tracks.npz"
        if not f.is_file():
            continue
        table = PoseTable.load(f)
        info = probe(t.video)
        if (table.width, table.height) != (info.width, info.height) or abs(table.fps - info.fps) > 1e-6:
            raise ValueError(f"{f.name} was not extracted from {t.video.name}")
        events, _ = olympic.load_events(t.annotations, info.timing.n_frames, info.width, info.height)
        events = [e for e in events if e.first < table.n_frames]   # a --max-frames run covers a prefix
        attrs = olympic.attribute(events, table)
        every += attrs
        report[t.name] = summarise(attrs, table) | {"frames_tracked": table.n_frames, "track_ids": len(table.ids())}
    if not report:
        raise ValueError(f"no <GoPro id>.tracks.npz in {a.tracks} matches a task in {a.data}")
    total = summarise(every)
    out = {"videos": report, "all": {k: v for k, v in total.items() if k not in ("punches_by_id", "top2_ids")}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    for name, r in report.items():
        print(f"{name}: {r['events']} punches in {r['frames_tracked']} frames, {r['track_ids']} ids | credited "
              f"{r['credited_share']:.0%}, too close to call {r['ambiguous_share']:.0%}, no wrist near "
              f"{r['unreached_share']:.0%} | nearest wrist = labelled hand in {r['nearest_wrist_is_labelled_hand']:.0%} "
              f"| top-2 ids {r['top2_ids']} take {r['top2_share_of_credited']:.0%}, both on screen "
              f"{r.get('top2_both_on_screen_share', 0):.0%}")
    print(f"ALL: {total['credited']}/{total['events']} credited; nearest wrist = labelled hand in "
          f"{total['nearest_wrist_is_labelled_hand']:.0%} of {total['hand_evidence_events']} "
          f"({total['hand_tied_excluded']} with both wrists equally near left out; a mirrored convention "
          f"would sit near 0%)")
    print(f"     too close to call, by target: {total['ambiguous_share_by_target']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
