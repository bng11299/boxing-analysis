"""Did the referee label every Olympic Boxing punch? Wrist-speed bursts per linked boxer.

  python olympic_completeness.py --data data/olympic --tracks results/ingest/olympic

For each video: link the ids into boxers A and B (olympic.py/olympic_link.py), run the per-hand
wrist-speed rule on each boxer's own segments (the rule in results/punch_finder_v6.json "chosen";
no signal crosses a join between two track ids), and match every burst ONE-TO-ONE to a labelled
punch within 0.1 s: first to the boxer's own labels (punches credited to that boxer), then, for the
bursts left over, to the labels nobody could be credited with. A burst no label accounts for is
UNEXPLAINED.

Unexplained bursts are CANDIDATES for missed labels. The rule misses punches and fires on
non-punches, so the count is neither an upper nor a lower bound on unlabelled punches until a
human has classified a random sample of them. Nothing is judged here.

Writes results/olympic_completeness.json and results/olympic_completeness_sample.csv: a seeded
simple random sample (every unexplained burst, all videos pooled, has inclusion probability n/N,
printed in the `inclusion_prob` column) for YOU to watch and mark punch / not punch in the empty
`user_label` column.

Rates (all per minute of boxer-time, and each numerator over the denominator it was counted in):
  unexplained_per_identified_boxer_min   unexplained bursts / minutes a boxer was identified
  own_labels_per_identified_boxer_min    labels credited to an identified boxer / the same minutes
                                         (the like-for-like comparison with the line above)
  labels_per_video_boxer_min             ALL labels / (2 x the video's minutes)
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
SAMPLE_COLUMNS = ["video", "frame", "time_s", "boxer", "hand", "inclusion_prob", "user_label"]


def _ratio(num, den):
    return num / den if den else None


def _round(x, nd=3):
    """Round for OUTPUT only; aggregation always runs on the unrounded numbers."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


def video_row(res: dict, n_events: int, video_min: float) -> dict:
    """One video's UNROUNDED numbers from `completeness.find_bursts` output."""
    bursts = res["bursts"]
    row = {"video_min": video_min, "labelled_punches": n_events, "labels": res["labels"],
           "bursts": len(bursts), "unexplained": sum(not b.explained for b in bursts),
           "bursts_by_match": {m: sum(b.match == m for b in bursts) for m in completeness.MATCHES},
           "boxers": {}}
    row["bursts_hand_agrees"] = {"agree": sum(b.hand_agrees is True for b in bursts),
                                 "disagree": sum(b.hand_agrees is False for b in bursts)}
    for name, info in res["boxers"].items():
        row["boxers"][name] = dict(info) | {
            "unexplained_per_identified_min": _ratio(info["unexplained"], info["identified_min"]),
            "unexplained_per_signal_min": _ratio(info["unexplained"], info["signal_min"]),
            "own_labels_per_identified_min": _ratio(info["own_labels"], info["identified_min"])}
    return row


def totals(ok: "list[dict]") -> dict:
    """Pool the per-video rows. Sums of unrounded durations and counts, divided once at the end."""
    tot = {"videos": len(ok), "video_min": sum(v["video_min"] for v in ok),
           "labelled_punches": sum(v["labelled_punches"] for v in ok),
           "bursts": sum(v["bursts"] for v in ok), "unexplained": sum(v["unexplained"] for v in ok),
           "bursts_by_match": {m: sum(v["bursts_by_match"][m] for v in ok) for m in completeness.MATCHES},
           "short_segments": sum(v["boxers"][n]["short_segments"] for v in ok for n in "AB"),
           "bodyless_segments": sum(v["boxers"][n]["bodyless_segments"] for v in ok for n in "AB"),
           "segments": sum(v["boxers"][n]["segments"] for v in ok for n in "AB")}
    lab = {k: sum(v["labels"][k] for v in ok) for k in ok[0]["labels"]}
    tot["labels"] = lab | {"unattributed_share": _ratio(lab["unattributed"], lab["total"]),
                           "matched_to_a_burst": lab["matched_own"] + lab["matched_unattributed"]}
    for name in "AB":
        b = [v["boxers"][name] for v in ok]
        im, sm = sum(x["identified_min"] for x in b), sum(x["signal_min"] for x in b)
        u, nb, ow = sum(x["unexplained"] for x in b), sum(x["bursts"] for x in b), sum(x["own_labels"] for x in b)
        tot[name] = {"identified_min": im, "signal_min": sm,
                     "signal_min_left": sum(x["signal_min_left"] for x in b),
                     "signal_min_right": sum(x["signal_min_right"] for x in b),
                     "bursts": nb, "unexplained": u, "own_labels": ow,
                     "bursts_by_hand": {h: sum(x["bursts_by_hand"][h] for x in b) for h in completeness.HAND_NAMES},
                     "unexplained_by_hand": {h: sum(x["unexplained_by_hand"][h] for x in b)
                                             for h in completeness.HAND_NAMES},
                     "unexplained_per_identified_min": _ratio(u, im), "unexplained_per_signal_min": _ratio(u, sm),
                     "own_labels_per_identified_min": _ratio(ow, im)}
    im = tot["A"]["identified_min"] + tot["B"]["identified_min"]
    own = tot["A"]["own_labels"] + tot["B"]["own_labels"]
    tot["identified_boxer_min"] = im
    tot["unexplained_per_identified_boxer_min"] = _ratio(tot["unexplained"], im)
    tot["own_labels_per_identified_boxer_min"] = _ratio(own, im)
    tot["labels_per_video_boxer_min"] = _ratio(lab["total"], 2 * tot["video_min"])
    return tot


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
            res = completeness.find_bursts(table, lk, events, attrs, rule, video=vid)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            videos[t.name] = {"error": str(e)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        unexplained[vid] = [b for b in res["bursts"] if not b.explained]
        row = video_row(res, len(events), table.n_frames / table.fps / 60)
        row["video_bound"] = bool(table.video_fp)
        videos[t.name] = row
        print(f"{t.name}: {row['bursts']} bursts, {row['unexplained']} unexplained | " + " | ".join(
            f"{n}: {i['unexplained']}/{i['bursts']} in {i['identified_min']:.1f} min" for n, i in row["boxers"].items()))
    ok = [v for v in videos.values() if "error" not in v]
    if not ok:
        raise ValueError(f"nothing to test in {a.tracks} against {a.data}")
    tot = totals(ok)
    sample, population = completeness.random_sample(unexplained, a.sample_size, a.seed)
    prob = len(sample) / population if population else None

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"rule": rule, "tolerance_s": completeness.TOL_S, "reacquire": bool(a.reacquire),
         "matching": "one-to-one, nearest first: own labels, then unattributed labels; hand not required",
         "videos": _round(videos), "all": _round(tot),
         "sample": {"n": len(sample), "population_unexplained": population, "inclusion_prob": prob,
                    "design": "simple random sample without replacement over all unexplained bursts",
                    "seed": a.seed, "file": str(a.sample_out)}},
        indent=1, allow_nan=False), encoding="utf-8")
    a.sample_out.parent.mkdir(parents=True, exist_ok=True)
    with a.sample_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_COLUMNS)
        for b in sample:
            w.writerow([b.video, b.frame, f"{b.time_s:.2f}", b.boxer, b.hand, f"{prob:.6f}", ""])
    lab = tot["labels"]
    print(f"ALL: {tot['bursts']} bursts on {tot['identified_boxer_min']:.1f} identified boxer-minutes, "
          f"{tot['unexplained']} unexplained (no label within {completeness.TOL_S} s that is not already matched; "
          f"candidates for missed labels, not a count of them) = "
          f"{tot['unexplained_per_identified_boxer_min']:.1f} per identified boxer-minute, against "
          f"{tot['own_labels_per_identified_boxer_min']:.1f} OWN labels per identified boxer-minute "
          f"({lab['unattributed']} of {lab['total']} labels, {lab['unattributed_share']:.0%}, are unattributed; "
          f"{tot['labels_per_video_boxer_min']:.1f} labels per video boxer-minute over the whole video); "
          f"{lab['matched_to_a_burst']} labels matched to a burst ({lab['matched_own']} own, "
          f"{lab['matched_unattributed']} unattributed) -> {a.out}; sample of {len(sample)} of {population} -> {a.sample_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
