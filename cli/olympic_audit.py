"""Audit the Olympic Boxing release before anything trains on it.

  python cli/olympic_audit.py --data data/olympic

Loads every task through the strict loader (src/boxingvi/olympic.py), checks the
release against the paper (312,774 frames in all), and reports what decides how
it may be used: punch density per video and per camera (an unevenly labelled
video would teach "no punch" on real punches), single-keyframe labelling, and
whether the two cameras cover the same time. Writes results/olympic_audit.json.
Nothing derived from this dataset is ever published.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import olympic  # noqa: E402
from boxingvi.video import probe  # noqa: E402

PAPER_TOTAL_FRAMES = 312_774       # Stefanski et al. 2024, section 3.2
PAPER_PUNCH_FRAMES = 11_345         # ... "considered as punches"; matches no count derived here
DEFAULT_OUT = Path("results/olympic_audit.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    videos, cams = [], collections.defaultdict(lambda: {"frames": 0, "minutes": 0.0, "events": 0, "videos": 0})
    for t in olympic.find_tasks(a.data):
        info = probe(t.video)
        if info.timing is None or info.vfr:
            raise ValueError(f"{t.name}: irregular frame timing; frame-indexed labels cannot be trusted")
        n = info.timing.n_frames
        events, empty = olympic.load_events(t.annotations, n, info.width, info.height)
        minutes = n / info.fps / 60
        single = sum(e.n_keyframes == 1 for e in events)
        by = collections.Counter(f"{e.target}_{e.hand}" for e in events)
        videos.append({"task": t.name, "camera": t.camera, "frames": n, "fps": info.fps,
                       "minutes": round(minutes, 2), "events": len(events), "empty_tracks": empty,
                       "per_minute": round(len(events) / minutes, 1),
                       "single_keyframe_share": round(single / max(len(events), 1), 3),
                       "median_keyframes": int(sorted(e.n_keyframes for e in events)[len(events) // 2]) if events else 0,
                       "open_ended": sum(e.open_ended for e in events), "reappeared": sum(e.reappeared for e in events),
                       "punch_frames_first_to_last": sum(e.last - e.first + 1 for e in events),
                       "by_label": dict(by)})
        c = cams[t.camera]
        c["frames"] += n; c["minutes"] += minutes; c["events"] += len(events); c["videos"] += 1

    total = sum(v["frames"] for v in videos)
    rates = sorted(v["per_minute"] for v in videos)
    med = rates[len(rates) // 2]
    for v in videos:
        v["flags"] = [f for f, bad in (
            ("density under half the median video: under-labelled or a quiet round? view before excluding",
             v["per_minute"] < med / 2),
            ("mostly single-keyframe labels: a different labelling style; view before using its timing",
             v["single_keyframe_share"] > 0.5),
        ) if bad]
    for c in cams.values():
        c["per_minute"] = round(c["events"] / c["minutes"], 1)
        c["minutes"] = round(c["minutes"], 1)
    punch_frames = sum(v["punch_frames_first_to_last"] for v in videos)
    report = {"paper_total_frames": PAPER_TOTAL_FRAMES, "release_total_frames": total,
              "total_frames_match_paper": total == PAPER_TOTAL_FRAMES,
              "paper_punch_frames": PAPER_PUNCH_FRAMES, "release_punch_frames_first_to_last": punch_frames,
              "punch_frames_match_paper": punch_frames == PAPER_PUNCH_FRAMES,
              "open_ended_tracks": sum(v["open_ended"] for v in videos),
              "reappearing_tracks": sum(v["reappeared"] for v in videos),
              "videos": videos, "cameras": dict(cams),
              "events": sum(v["events"] for v in videos), "empty_tracks": sum(v["empty_tracks"] for v in videos)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    print(f"{len(videos)} videos, {total} frames (paper: {PAPER_TOTAL_FRAMES}): "
          f"{'matches' if report['total_frames_match_paper'] else 'DOES NOT MATCH'}")
    print(f"punch frames, first to last box: {punch_frames} (paper: {PAPER_PUNCH_FRAMES}): "
          f"{'matches' if report['punch_frames_match_paper'] else 'DOES NOT MATCH: the paper counts punch frames some other way'}")
    print(f"tracks: {report['open_ended_tracks']} with no closing outside keyframe, "
          f"{report['reappearing_tracks']} hidden briefly and merged")
    print(f"{report['events']} punch events, {report['empty_tracks']} empty tracks skipped")
    for name, c in sorted(cams.items()):
        print(f"  {name}: {c['videos']} videos, {c['minutes']} min, {c['events']} punches, {c['per_minute']}/min")
    print(f"  punches per minute by video: median {med}, range {rates[0]}-{rates[-1]}")
    for v in videos:
        if v["flags"]:
            print(f"  FLAG {v['task']}: {'; '.join(v['flags'])} ({v['per_minute']}/min, "
                  f"single-keyframe {v['single_keyframe_share']:.0%})")
    print(f"-> {a.out}")
    return 0 if report["total_frames_match_paper"] else 1


if __name__ == "__main__":
    sys.exit(main())
