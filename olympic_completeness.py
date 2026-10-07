"""Did the referee label every Olympic Boxing punch? Wrist-speed bursts per linked boxer.

  python olympic_completeness.py --data data/olympic --tracks results/ingest/olympic

For each video: link the ids into boxers A and B (olympic_link.py), run the per-hand
wrist-speed rule on each boxer's own track (the rule in results/punch_finder_v6.json
"chosen"), and keep every burst more than 0.1 s from ANY labelled punch of that video.

Writes results/olympic_completeness.json (bursts and unexplained bursts, per video and in
total, with the rate per identified boxer-minute) and results/olympic_completeness_sample.csv:
a seeded sample of the unexplained bursts, stratified over videos, for YOU to watch and mark
punch / not punch in the empty `user_label` column. Nothing is judged here: a burst without a
label is a wrist moving fast, which a feint, a guard reset or a slip also does.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import find_punches  # noqa: E402
import olympic_link  # noqa: E402
from boxingvi import completeness, linking, olympic  # noqa: E402

DEFAULT_OUT = Path("results/olympic_completeness.json")
DEFAULT_SAMPLE = Path("results/olympic_completeness_sample.csv")
SAMPLE_COLUMNS = ["video", "frame", "time_s", "boxer", "hand", "user_label"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sample-out", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--sample-size", type=int, default=completeness.SAMPLE_N)
    ap.add_argument("--seed", type=int, default=completeness.SAMPLE_SEED)
    ap.add_argument("--reacquire", action="store_true",
                    help="link across gaps of up to 60 s (see olympic_link.py --reacquire); off = as first specified")
    ap.add_argument("--only", nargs="+", metavar="ID")
    a = ap.parse_args(argv)

    rule = find_punches.load_rule(a.rule)
    params = linking.LinkParams(reacquire_s=linking.REACQUIRE_ON_S if a.reacquire else 0.0)
    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    videos, unexplained, failed = {}, {}, []
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if (only is not None and vid not in only) or not (a.tracks / f"{vid}.tracks.npz").is_file():
            continue
        try:
            table, _, events, attrs = olympic_link.load_video(t, a.tracks)
            lk = linking.link_video(table, attrs, params)
            res = completeness.find_bursts(table, lk, events, rule, video=vid)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            videos[t.name] = {"error": str(e)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        bad = [b for b in res["bursts"] if not b.explained]
        unexplained[vid] = bad
        row = {"labelled_punches": len(events), "labelled_with_a_burst": res["events_with_burst"], "bursts": len(res["bursts"]), "unexplained": len(bad),
               "boxers": {}}
        for name, info in res["boxers"].items():
            row["boxers"][name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()} | {
                "unexplained_per_identified_min": round(info["unexplained"] / info["identified_min"], 2)
                if info["identified_min"] > 0 else None,
                "unexplained_per_signal_min": round(info["unexplained"] / info["signal_min"], 2)
                if info["signal_min"] > 0 else None}
        videos[t.name] = row
        print(f"{t.name}: {row['bursts']} bursts, {row['unexplained']} unexplained | " + " | ".join(
            f"{n}: {i['unexplained']}/{i['bursts']} in {i['identified_min']:.1f} min" for n, i in row["boxers"].items()))
    ok = [v for v in videos.values() if "error" not in v]
    if not ok:
        raise ValueError(f"nothing to test in {a.tracks} against {a.data}")
    tot = {"videos": len(ok), "labelled_punches": sum(v["labelled_punches"] for v in ok),
           "labelled_with_a_burst": sum(v["labelled_with_a_burst"] for v in ok),
           "bursts": sum(v["bursts"] for v in ok), "unexplained": sum(v["unexplained"] for v in ok)}
    for name in ("A", "B"):
        im = sum(v["boxers"][name]["identified_min"] for v in ok)
        sm = sum(v["boxers"][name]["signal_min"] for v in ok)
        b = sum(v["boxers"][name]["bursts"] for v in ok)
        u = sum(v["boxers"][name]["unexplained"] for v in ok)
        tot[name] = {"identified_min": round(im, 2), "signal_min": round(sm, 2), "bursts": b, "unexplained": u,
                     "unexplained_per_identified_min": round(u / im, 3) if im else None,
                     "unexplained_per_signal_min": round(u / sm, 3) if sm else None}
    both_min = tot["A"]["identified_min"] + tot["B"]["identified_min"]
    tot["unexplained_per_identified_boxer_min"] = round(tot["unexplained"] / both_min, 3) if both_min else None
    tot["labelled_punches_per_identified_boxer_min"] = round(tot["labelled_punches"] / both_min, 3) if both_min else None
    sample = completeness.stratified_sample(unexplained, a.sample_size, a.seed)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rule": rule, "tolerance_s": completeness.TOL_S, "reacquire": bool(a.reacquire), "videos": videos, "all": tot,
                                 "sample": {"n": len(sample), "seed": a.seed, "file": str(a.sample_out)}},
                                indent=1, allow_nan=False), encoding="utf-8")
    a.sample_out.parent.mkdir(parents=True, exist_ok=True)
    with a.sample_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_COLUMNS)
        for b in sample:
            w.writerow([b.video, b.frame, f"{b.time_s:.2f}", b.boxer, b.hand, ""])
    print(f"ALL: {tot['bursts']} bursts, {tot['unexplained']} more than {completeness.TOL_S} s from any of "
          f"{tot['labelled_punches']} labelled punches; {tot['unexplained_per_identified_boxer_min']} unexplained per "
          f"identified boxer-minute (labelled: {tot['labelled_punches_per_identified_boxer_min']} per boxer-minute; "
          f"{tot['labelled_with_a_burst']} of them have a burst within {completeness.TOL_S} s) "
          f"-> {a.out}; sample of {len(sample)} -> {a.sample_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
