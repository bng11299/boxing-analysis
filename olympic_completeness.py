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

Rates. Every rate for the bursts divides by ONE denominator, the boxer's EXPOSURE: the frames of
the segments the rule actually ran on, where at least one hand had a valid signal, bridged frames
included (completeness.find_bursts states the policy, and asserts every burst lies inside it):
  unexplained_per_exposure_min       unexplained bursts / exposure minutes
  bursts_per_exposure_min            bursts / exposure minutes
  own_labels_in_exposure_per_exposure_min
                                     labels credited to the boxer whose span touches his exposure /
                                     the same minutes (the same observation as the two lines above)
Two further numbers are DESCRIPTIVE and no rate here divides by them:
  identified_min                     minutes the linking names the boxer (includes segments too
                                     short for the rule, bodyless ones, and frames with no signal)
  labels_per_video_boxer_min         ALL labels / (2 x the video's minutes): a different population
                                     (whole video, both boxers, unattributed labels included), never a
                                     like-for-like comparison with the rates above
Unexplained bursts are candidates; nothing about missed labels follows until the sample is classified.

  python olympic_completeness.py --data data/olympic --tracks results/ingest/olympic --credited [--min-punches N]

CREDITED FRAGMENTS (ADR-0012 Option 2): no linking, no ring, no identity. The test runs on every track id
credited with at least N (default 1) labelled punches, each id being one person for its whole length
(ByteTrack never gives one id to two people by design; an id switch inside a fragment is the tracker's
residual error). Same matching (a fragment's own labels first, then the unattributed ones, each label
once), same exposure, same rates, same seeded sample, which also carries what you need to FIND the
person: track_id, the fragment's box centre and height at the burst frame (pixels) and the time as
m:ss.s. Writes results/olympic_completeness_credited.json and ..._credited_sample.csv by default (the
linking-based files are never overwritten). --reacquire is a linking option and is refused here.

THE STATED LIMITS: this sees only credited fragments. Missed punches outside them (boxer time in
fragments with no credited punch, punches too close to call) are never seen, so coverage can only lose;
but fragment-level detections can DUPLICATE one physical punch when two overlapping ids follow the same
person. So neither a bout-level upper nor a lower bound follows without de-duplication:
`possible_duplicates` (per video and pooled) counts the unexplained bursts that have an unexplained twin
in another credited fragment within 0.1 s whose box overlaps (IoU >= 0.3); it is reported, never
subtracted. A clean result is weak evidence of completeness; a dirty one shows labels are missing inside
the credited fragments, though the count may include duplicates. A credited fragment can be a non-boxer
wrongly credited; the fragments with exactly one credited punch are the least certain and are counted.
Attribution is computed once per video (olympic_link.load_video) and verified against the pose table
inside the test.

  python olympic_completeness.py ... [--credited] --skips            (after olympic_framechange.py)

CAMERA FRAME DROPS (src/boxingvi/skips.py): the GoPro files drop captured frames while the timestamps stay
regular, and the wrist-speed rule reads the on-screen jump as a fast punch. --skips reads each video's
`<tracks>/<id>.framechange.npz` (refusing the whole run when one is missing, malformed, or not for this
video: file name, frame count, content fingerprint), MASKS every frame within reach of a skip (the wrist
signal there is set to 0 and the frame leaves the exposure; completeness.skip_mask_half derives the reach from
the rule's own smoothing window and the frame rate: 3 frames each side at 50 fps; segments, body scale and
smoothing are untouched), and reports per video and pooled
`skips` and, for the unexplained and the explained bursts, how many lie within 2 frames of a skip beside
the BASE RATE (the share of exposure frames within 2 frames of a skip). `--skips-measure` loads the same
files and reports the same numbers WITHOUT masking (the "before" of a before/after). Outputs go to new
default names (`..._skips.json`, `..._skips_sample.csv`; `..._skipsmeasure...` for --skips-measure); the
files of a run without skips are never touched. The rule and its constant come from one observation and one
development video, not from tuning (skips.py). `--skipcheck SAMPLE.csv --skipcheck-out OUT.csv` (--credited)
READS an earlier sample and writes a copy of its video, frame, track_id and hand columns with one more,
`still_unexplained_with_skips`: is a burst of that track and hand, within 2 frames, still unexplained in
this run? (the input file is never written).
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
from boxingvi import completeness, linking, olympic, video  # noqa: E402
from boxingvi import skips as skipmod  # noqa: E402

DEFAULT_OUT = Path("results/olympic_completeness.json")
DEFAULT_SAMPLE = Path("results/olympic_completeness_sample.csv")
SAMPLE_COLUMNS = ["video", "frame", "time_s", "boxer", "hand", "inclusion_prob", "user_label"]
DEFAULT_OUT_CREDITED = Path("results/olympic_completeness_credited.json")
DEFAULT_SAMPLE_CREDITED = Path("results/olympic_completeness_credited_sample.csv")
DEFAULT_OUT_SKIPS = Path("results/olympic_completeness_skips.json")
DEFAULT_SAMPLE_SKIPS = Path("results/olympic_completeness_skips_sample.csv")
DEFAULT_OUT_CREDITED_SKIPS = Path("results/olympic_completeness_credited_skips.json")
DEFAULT_SAMPLE_CREDITED_SKIPS = Path("results/olympic_completeness_credited_skips_sample.csv")
DEFAULT_OUT_SKIPS_MEASURE = Path("results/olympic_completeness_skipsmeasure.json")
DEFAULT_SAMPLE_SKIPS_MEASURE = Path("results/olympic_completeness_skipsmeasure_sample.csv")
DEFAULT_OUT_CREDITED_SKIPS_MEASURE = Path("results/olympic_completeness_credited_skipsmeasure.json")
DEFAULT_SAMPLE_CREDITED_SKIPS_MEASURE = Path("results/olympic_completeness_credited_skipsmeasure_sample.csv")
SKIPCHECK_COLUMNS = ["video", "frame", "track_id", "hand", "still_unexplained_with_skips"]
SAMPLE_COLUMNS_CREDITED = ["video", "frame", "time_s", "time_mss", "track_id", "box_cx", "box_cy", "box_h", "hand",
                           "inclusion_prob", "user_label"]


class SkipsRefused(ValueError):
    """--skips could not be honoured for a video: the whole run stops and nothing is written."""


def load_skips(a, task, table) -> "skipmod.Skips | None":
    """The verified Skips of one video when --skips / --skips-measure is on, else None. Refuses (SkipsRefused)
    a missing, malformed or mismatched file, or one measured with another ratio, window or pixel step than this code's."""
    if not (a.skips or a.skips_measure):
        return None
    f = a.tracks / f"{task.video.stem}.framechange.npz"
    try:
        sk = skipmod.load_for(table, f)
        if sk.video_fp != video.fingerprint(task.video):
            raise ValueError("the frame changes were measured on a different file than this video "
                             "(content fingerprints differ)")
        if (sk.ratio, sk.half_window) != (skipmod.SKIP_RATIO, skipmod.HALF_WINDOW) or sk.step != skipmod.STEP:
            raise ValueError(f"made with ratio {sk.ratio}, half window {sk.half_window} and pixel step {sk.step}, this code uses "
                             f"{skipmod.SKIP_RATIO}, {skipmod.HALF_WINDOW} and {skipmod.STEP} (every one of them changes which frames are skips)")
    except (ValueError, OSError) as e:
        raise SkipsRefused(f"{task.video.stem}: {f.name}: {e}") from e
    return sk


def skip_shares(t: dict) -> dict:
    base = _ratio(t["exposure_near_skip"], t["exposure_frames"])
    un, ex = _ratio(t["unexplained_near_skip"], t["unexplained"]), _ratio(t["explained_near_skip"], t["explained"])
    return {"base_rate": base, "unexplained_near_skip_share": un, "explained_near_skip_share": ex,
            "unexplained_excess_over_base": _ratio(un, base) if un is not None and base else None,
            "explained_excess_over_base": _ratio(ex, base) if ex is not None and base else None}


def with_shares(stats: dict) -> dict:
    return stats | skip_shares(stats)


def pool_skip_stats(rows: "list[dict]") -> dict:
    """Pool the per-video near-skip counts (sum, then divide once) and add the shares and the ratio of
    the unexplained share to the base rate."""
    keys = ("skips", "exposure_frames", "exposure_near_skip", "unexplained", "unexplained_near_skip",
            "explained", "explained_near_skip")
    return with_shares({k: sum(r[k] for r in rows) for k in keys} | {"within_frames": rows[0]["within_frames"]})


def skip_line(t: dict, mode: str) -> str:
    return (f"SKIPS ({mode}; {t['skips']} skips): bursts within {t['within_frames']} frames of a skip: unexplained "
            f"{t['unexplained_near_skip']}/{t['unexplained']} ({_pct(t['unexplained_near_skip_share'])}), explained "
            f"{t['explained_near_skip']}/{t['explained']} ({_pct(t['explained_near_skip_share'])}), against a base rate of "
            f"{_pct(t['base_rate'])} of exposure frames ({t['exposure_near_skip']}/{t['exposure_frames']}); unexplained "
            f"excess over base {_num(t['unexplained_excess_over_base'], 2)}x, explained {_num(t['explained_excess_over_base'], 2)}x. ")


def _skip_report(a, ok: "list[dict]", tot: dict) -> "tuple[dict | None, str]":
    """With --skips / --skips-measure: pool the per-video near-skip counts into tot["skips"], and return
    (the json block that states the mode and the rule, the line for the print). Without: (None, "")."""
    if not (a.skips or a.skips_measure):
        return None, ""
    mode = "mask" if a.skips else "measure only: nothing masked"
    tot["skips"] = pool_skip_stats([v["skips"] for v in ok])
    block = {"mode": "mask" if a.skips else "measure", "ratio": skipmod.SKIP_RATIO, "half_window": skipmod.HALF_WINDOW,
             "step": skipmod.STEP, "near_frames": completeness.NEAR_SKIP,
             "rule": "frame f is a skip when its frame change exceeds ratio x the median over [f-10, f+10] without f",
             "declaration": "rule and ratio come from one observation and one development video (GH069681), not from tuning to any score",
             "mask": "the frames within smooth_window // 2 + 1 of a skip (3 at 50 fps), widened over a bridged hole, get signal 0 "
                     "and leave the exposure; segments, body scale and smoothing are untouched" if a.skips
                     else "nothing was masked: the same bursts as a run without skips"}
    return block, skip_line(tot["skips"], mode)


def read_sample(path: Path) -> "list[dict]":
    """The rows of an earlier sample csv, READ ONLY, with the four columns skipcheck needs."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    for need in ("video", "frame", "track_id", "hand"):
        if need not in rows[0]:
            raise ValueError(f"{path}: no {need!r} column (a --credited sample is expected)")
    for r in rows:
        if not str(r["frame"]).isdigit() or not str(r["track_id"]).isdigit() or r["hand"] not in completeness.HAND_NAMES:
            raise ValueError(f"{path}: malformed row {r}")
    return rows


def skipcheck_rows(rows_in: "list[dict]", bursts_by_video: dict, n_frames_by_video: dict,
                   within: int = completeness.NEAR_SKIP) -> list:
    """For each row of an earlier sample (video, frame, track_id, hand): is a burst of that track, that hand, within
    `within` frames still UNEXPLAINED in this run? A row whose video was not processed in this run, or whose frame is not
    a frame of that video (0 <= frame < its length), is an error, never a quiet False."""
    out = []
    for r in rows_in:
        v = r["video"]
        if v not in bursts_by_video or v not in n_frames_by_video:
            raise ValueError(f"the sample row for {v} frame {r['frame']}: that video is not in this run")
        frame = int(r["frame"])
        if not 0 <= frame < n_frames_by_video[v]:
            raise ValueError(f"the sample row for {v} names frame {frame}, but that video has frames 0..{n_frames_by_video[v] - 1}")
        still = any(not b.explained and str(b.track_id) == str(r["track_id"]) and b.hand == r["hand"]
                    and abs(b.frame - frame) <= within for b in bursts_by_video[v])
        out.append([v, r["frame"], r["track_id"], r["hand"], "true" if still else "false"])
    return out


def _ratio(num, den):
    return num / den if den else None


def _num(x, nd=1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


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
        row["boxers"][name] = dict(info) | _boxer_rates(info["unexplained"], info["bursts"],
                                                        info["own_labels_in_exposure"], info["exposure_min"])
    return row


def _boxer_rates(unexplained, bursts, own_in_exposure, exposure_min) -> dict:
    """The rates, all over the same exposure minutes. Pass raw counts and UNROUNDED minutes."""
    return {"unexplained_per_exposure_min": _ratio(unexplained, exposure_min),
            "bursts_per_exposure_min": _ratio(bursts, exposure_min),
            "own_labels_in_exposure_per_exposure_min": _ratio(own_in_exposure, exposure_min)}


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
        sums = {k: sum(x[k] for x in b) for k in ("identified_min", "identified_frames", "exposure_min", "exposure_frames",
                                                  "signal_min_left", "signal_min_right", "bursts", "unexplained",
                                                  "own_labels", "own_labels_in_exposure", "own_labels_outside_exposure",
                                                  "labels_matched_own")}
        tot[name] = sums | {
            "bursts_by_hand": {h: sum(x["bursts_by_hand"][h] for x in b) for h in completeness.HAND_NAMES},
            "unexplained_by_hand": {h: sum(x["unexplained_by_hand"][h] for x in b) for h in completeness.HAND_NAMES}
        } | _boxer_rates(sums["unexplained"], sums["bursts"], sums["own_labels_in_exposure"], sums["exposure_min"])
    # Pooled over both boxers: raw counts and raw frame counts summed, minutes summed unrounded,
    # every rate divided once at the end.
    for new, old in (("identified_boxer_min", "identified_min"), ("identified_boxer_frames", "identified_frames"),
                     ("exposure_boxer_min", "exposure_min"), ("exposure_boxer_frames", "exposure_frames"),
                     ("own_labels_boxers", "own_labels"), ("own_labels_in_exposure_boxers", "own_labels_in_exposure"),
                     ("own_labels_outside_exposure_boxers", "own_labels_outside_exposure")):
        tot[new] = tot["A"][old] + tot["B"][old]
    tot.update(_boxer_rates(tot["unexplained"], tot["bursts"], tot["own_labels_in_exposure_boxers"],
                            tot["exposure_boxer_min"]))
    tot["labels_per_video_boxer_min"] = _ratio(lab["total"], 2 * tot["video_min"])
    return tot


def video_row_credited(res: dict, n_events: int, video_min: float) -> dict:
    """One video's UNROUNDED numbers from `completeness.find_bursts_credited` output."""
    bursts = res["bursts"]
    frags = [res["fragments"][t] | _boxer_rates(res["fragments"][t]["unexplained"], res["fragments"][t]["bursts"],
                                                res["fragments"][t]["own_labels_in_exposure"],
                                                res["fragments"][t]["exposure_min"])
             for t in sorted(res["fragments"])]
    return {"video_min": video_min, "labelled_punches": n_events, "labels": res["labels"],
            "fragments_n": len(frags), "fragments_one_punch": sum(f["credited_punches"] == 1 for f in frags),
            "bursts": len(bursts), "unexplained": sum(not b.explained for b in bursts),
            "bursts_by_match": {m: sum(b.match == m for b in bursts) for m in completeness.MATCHES},
            "bursts_hand_agrees": {"agree": sum(b.hand_agrees is True for b in bursts),
                                   "disagree": sum(b.hand_agrees is False for b in bursts)},
            "possible_duplicates": sum(b.possible_duplicate for b in bursts),
            "possible_duplicate_pairs": res["possible_duplicate_pairs"],
            "exposure_frames": sum(f["exposure_frames"] for f in frags),
            "exposure_min": sum(f["exposure_min"] for f in frags),
            "own_labels_in_exposure": sum(f["own_labels_in_exposure"] for f in frags),
            "own_labels_outside_exposure": sum(f["own_labels_outside_exposure"] for f in frags),
            "segments": sum(f["segments"] for f in frags), "short_segments": sum(f["short_segments"] for f in frags),
            "bodyless_segments": sum(f["bodyless_segments"] for f in frags), "fragments": frags}


def totals_credited(ok: "list[dict]") -> dict:
    """Pool the per-video rows: counts and unrounded minutes summed, every rate divided once at the end.
    The exposure is summed over fragments (two fragments on screen together are two people)."""
    keys = ("labelled_punches", "fragments_n", "fragments_one_punch", "bursts", "unexplained", "possible_duplicates",
            "possible_duplicate_pairs", "exposure_frames",
            "exposure_min", "own_labels_in_exposure", "own_labels_outside_exposure", "segments", "short_segments",
            "bodyless_segments")
    tot = {"videos": len(ok), "video_min": sum(v["video_min"] for v in ok)} | {k: sum(v[k] for v in ok) for k in keys}
    tot["bursts_by_match"] = {m: sum(v["bursts_by_match"][m] for v in ok) for m in completeness.MATCHES}
    lab = {k: sum(v["labels"][k] for v in ok) for k in ok[0]["labels"]}
    tot["labels"] = lab | {"unattributed_share": _ratio(lab["unattributed"], lab["total"]),
                           "matched_to_a_burst": lab["matched_own"] + lab["matched_unattributed"]}
    tot |= _boxer_rates(tot["unexplained"], tot["bursts"], tot["own_labels_in_exposure"], tot["exposure_min"])
    tot["labels_per_video_boxer_min"] = _ratio(lab["total"], 2 * tot["video_min"])
    return tot


def _main_credited(a, rule, only) -> int:
    videos, unexplained, failed, all_bursts, lengths = {}, {}, [], {}, {}
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if (only is not None and vid not in only) or not (a.tracks / f"{vid}.tracks.npz").is_file():
            continue
        try:
            table, _, events, attrs = olympic_link.load_video(t, a.tracks)
            sk = load_skips(a, t, table)
            res = completeness.find_bursts_credited(table, events, attrs, rule, video=vid, min_punches=a.min_punches,
                                                    skips=sk.frames if a.skips else None)
        except SkipsRefused:
            raise
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            videos[t.name] = {"error": str(e)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        unexplained[vid] = [b for b in res["bursts"] if not b.explained]
        all_bursts[vid] = res["bursts"]
        lengths[vid] = table.n_frames
        row = video_row_credited(res, len(events), table.n_frames / table.fps / 60)
        row["video_bound"] = bool(table.video_fp)
        if sk is not None:
            row["skips"] = with_shares(completeness.near_skip_stats(res["bursts"], res["exposure"], sk.frames, table.n_frames))
        videos[t.name] = row
        print(f"{t.name}: {row['fragments_n']} credited fragments, {row['bursts']} bursts, {row['unexplained']} unexplained "
              f"in {row['exposure_min']:.1f} exposure min" + ("" if sk is None else f", {row['skips']['skips']} skips"))
    ok = [v for v in videos.values() if "error" not in v]
    if not ok:
        raise ValueError(f"nothing to test in {a.tracks} against {a.data}")
    tot = totals_credited(ok)
    skip_json, skip_text = _skip_report(a, ok, tot)
    sample, population = completeness.random_sample(unexplained, a.sample_size, a.seed)
    prob = len(sample) / population if population else None
    lab = tot["labels"]
    summary = (skip_text + f"CREDITED FRAGMENTS (min {a.min_punches} credited punch(es)): {tot['fragments_n']} fragments, "
               f"{tot['fragments_one_punch']} with exactly one credited punch (least certain); {tot['bursts']} bursts in "
               f"{_num(tot['exposure_min'])} exposure minutes (summed over fragments), {tot['unexplained']} unexplained, "
               f"{tot['possible_duplicates']} of them with a possible twin in another fragment ({tot['possible_duplicate_pairs']} "
               f"pairs), i.e. "
               f"{_num(tot['unexplained_per_exposure_min'])} per exposure minute; {_num(tot['bursts_per_exposure_min'])} bursts and "
               f"{_num(tot['own_labels_in_exposure_per_exposure_min'])} own labels (span touching the exposure) per exposure minute; "
               f"labels: {lab['owned']} owned ({lab['owned_outside_exposure']} outside the exposure), {lab['unattributed']} of "
               f"{lab['total']} ({_pct(lab['unattributed_share'])}) unattributed, {lab['credited_to_excluded']} credited to an excluded "
               f"fragment; {lab['matched_to_a_burst']} matched to a burst ({lab['matched_own']} own, {lab['matched_unattributed']} "
               f"unattributed). Different population: {_num(tot['labels_per_video_boxer_min'])} labels per whole-video boxer-minute. "
               f"LIMITS: {completeness.BIAS_CREDITED} Unexplained bursts are candidates: nothing about missed labels follows "
               f"until the sample is classified. -> {a.out}; sample of {len(sample)} of {population} -> {a.sample_out}")
    out_json = json.dumps(
        {"mode": "credited_fragments", "linking": None, "min_punches": a.min_punches, "rule": rule,
         "tolerance_s": completeness.TOL_S, "bias": completeness.BIAS_CREDITED,
         "matching": "one-to-one; optimal within each phase (most matches, then least total gap), a fragment's own labels first, "
                     "then the unattributed labels, so not necessarily the global maximum; ties to the burst that sorts first "
                     "(frame, track id); a label credited to an excluded fragment explains nothing; hand not required",
         "exposure": "frames of processed segments of a credited fragment with a valid signal for at least one hand, bridged "
                     "frames included; summed over fragments",
         "videos": _round(videos), "all": _round(tot),
         "sample": {"n": len(sample), "population_unexplained": population, "inclusion_prob": prob,
                    "design": "simple random sample without replacement over all unexplained bursts",
                    "seed": a.seed, "file": str(a.sample_out)}} | ({"skips": skip_json} if skip_json else {}),
        indent=1, allow_nan=False)
    check_rows = skipcheck_rows(read_sample(a.skipcheck), all_bursts, lengths) if a.skipcheck else None
    sample_rows = [[b.video, b.frame, f"{b.time_s:.2f}", completeness.mss(b.time_s), b.track_id, f"{b.box[0]:.1f}",
                    f"{b.box[1]:.1f}", f"{b.box[2]:.1f}", b.hand, f"{prob:.6f}", ""] for b in sample]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out_json, encoding="utf-8")
    a.sample_out.parent.mkdir(parents=True, exist_ok=True)
    with a.sample_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_COLUMNS_CREDITED)
        w.writerows(sample_rows)
    if check_rows is not None:
        a.skipcheck_out.parent.mkdir(parents=True, exist_ok=True)
        with a.skipcheck_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SKIPCHECK_COLUMNS)
            w.writerows(check_rows)
        summary += (f" SKIPCHECK: {sum(r[4] == 'true' for r in check_rows)} of {len(check_rows)} sampled rows are still "
                    f"unexplained bursts in this run -> {a.skipcheck_out}")
    print(summary)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default {DEFAULT_OUT}, or {DEFAULT_OUT_CREDITED} with --credited")
    ap.add_argument("--sample-out", type=Path, default=None,
                    help=f"default {DEFAULT_SAMPLE}, or {DEFAULT_SAMPLE_CREDITED} with --credited")
    ap.add_argument("--sample-size", type=int, default=completeness.SAMPLE_N)
    ap.add_argument("--seed", type=int, default=completeness.SAMPLE_SEED)
    ap.add_argument("--reacquire", action="store_true",
                    help="link across gaps of up to 60 s (see olympic_link.py --reacquire); off = as first specified")
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--credited", action="store_true",
                    help="ADR-0012 Option 2: test only fragments (track ids) credited with a labelled punch; no linking")
    ap.add_argument("--min-punches", type=int, default=None, metavar="N",
                    help="with --credited: the fewest credited punches a fragment needs (default 1)")
    ap.add_argument("--skips", action="store_true",
                    help="mask the frames within reach of the camera's dropped frames (olympic_framechange.py's files in --tracks) and report "
                         "bursts near skips against the base rate; outputs go to new default names")
    ap.add_argument("--skips-measure", action="store_true",
                    help="load the same files and report the same near-skip numbers WITHOUT masking (the 'before' of a before/after)")
    ap.add_argument("--skipcheck", type=Path, default=None, metavar="SAMPLE.csv",
                    help="with --credited and --skips/--skips-measure: READ an earlier sample csv; --skipcheck-out gets its "
                         "video, frame, track_id, hand columns plus still_unexplained_with_skips")
    ap.add_argument("--skipcheck-out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.skips and a.skips_measure:
        ap.error("--skips masks, --skips-measure only measures: pick one")
    if (a.skipcheck is None) != (a.skipcheck_out is None):
        ap.error("--skipcheck and --skipcheck-out go together")
    if a.skipcheck is not None:
        if not (a.skips or a.skips_measure) or not a.credited:
            ap.error("--skipcheck needs --credited and --skips (or --skips-measure)")
        if a.skipcheck.resolve() == a.skipcheck_out.resolve():
            ap.error("--skipcheck-out would overwrite the sample it reads")
    if a.credited and a.reacquire:
        ap.error("--reacquire is a linking option; --credited does no linking")
    if a.min_punches is not None and not a.credited:
        ap.error("--min-punches needs --credited")
    if a.credited and (a.min_punches is not None and a.min_punches < 1):
        ap.error("--min-punches must be a whole number >= 1")
    a.min_punches = 1 if a.min_punches is None else a.min_punches
    mode = 2 if a.skips_measure else 1 if a.skips else 0         # which family of default names
    if a.out is None:
        a.out = [[DEFAULT_OUT, DEFAULT_OUT_SKIPS, DEFAULT_OUT_SKIPS_MEASURE],
                 [DEFAULT_OUT_CREDITED, DEFAULT_OUT_CREDITED_SKIPS, DEFAULT_OUT_CREDITED_SKIPS_MEASURE]][bool(a.credited)][mode]
    if a.sample_out is None:
        a.sample_out = [[DEFAULT_SAMPLE, DEFAULT_SAMPLE_SKIPS, DEFAULT_SAMPLE_SKIPS_MEASURE],
                        [DEFAULT_SAMPLE_CREDITED, DEFAULT_SAMPLE_CREDITED_SKIPS,
                         DEFAULT_SAMPLE_CREDITED_SKIPS_MEASURE]][bool(a.credited)][mode]
    if a.out == a.sample_out:
        ap.error("--out and --sample-out are the same file")
    if a.skipcheck is not None:
        mine = {a.skipcheck_out.resolve(), a.out.resolve(), a.sample_out.resolve()}
        if len(mine) < 3 or a.skipcheck.resolve() in mine:
            ap.error("--skipcheck-out, --out and --sample-out must be three files, none of them the sample --skipcheck reads")

    rule = find_punches.load_rule(a.rule)
    try:
        return _run(a, rule)
    except SkipsRefused as e:
        print(f"--skips refused, nothing written: {e}", file=sys.stderr)
        return 2


def _run(a, rule) -> int:
    if a.credited:
        return _main_credited(a, rule, {olympic_link._check_video_id(v) for v in a.only} if a.only else None)
    params = linking.LinkParams(reacquire_s=linking.REACQUIRE_ON_S if a.reacquire else 0.0)
    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    videos, unexplained, failed = {}, {}, []
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if (only is not None and vid not in only) or not (a.tracks / f"{vid}.tracks.npz").is_file():
            continue
        try:
            table, _, events, attrs = olympic_link.load_video(t, a.tracks)
            sk = load_skips(a, t, table)
            lk = linking.link_video(table, attrs, params)
            res = completeness.find_bursts(table, lk, events, attrs, rule, video=vid, skips=sk.frames if a.skips else None)
        except SkipsRefused:
            raise
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            videos[t.name] = {"error": str(e)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        unexplained[vid] = [b for b in res["bursts"] if not b.explained]
        row = video_row(res, len(events), table.n_frames / table.fps / 60)
        row["video_bound"] = bool(table.video_fp)
        if sk is not None:
            row["skips"] = with_shares(completeness.near_skip_stats(res["bursts"], res["exposure"], sk.frames, table.n_frames))
        videos[t.name] = row
        print(f"{t.name}: {row['bursts']} bursts, {row['unexplained']} unexplained | " + " | ".join(
            f"{n}: {i['unexplained']}/{i['bursts']} in {i['exposure_min']:.1f} exposure min" for n, i in row["boxers"].items()))
    ok = [v for v in videos.values() if "error" not in v]
    if not ok:
        raise ValueError(f"nothing to test in {a.tracks} against {a.data}")
    tot = totals(ok)
    skip_json, skip_text = _skip_report(a, ok, tot)
    sample, population = completeness.random_sample(unexplained, a.sample_size, a.seed)
    prob = len(sample) / population if population else None

    # Everything is computed BEFORE any file is written or any line formatted: a rate that is absent
    # (no exposure at all in a run) is None, written as null and printed as n/a, never a crash.
    lab = tot["labels"]
    summary = (skip_text + f"ALL: {tot['bursts']} bursts in {_num(tot['exposure_boxer_min'])} exposure boxer-minutes "
               f"(the frames the rule ran on; {_num(tot['identified_boxer_min'])} identified boxer-minutes, descriptive), "
               f"{tot['unexplained']} unexplained = {_num(tot['unexplained_per_exposure_min'])} per exposure minute; "
               f"{_num(tot['bursts_per_exposure_min'])} bursts and "
               f"{_num(tot['own_labels_in_exposure_per_exposure_min'])} own labels (span touching the exposure) per exposure minute; "
               f"{tot['own_labels_outside_exposure_boxers']} own labels fall outside it and "
               f"{lab['unattributed_in_exposure']} unattributed labels inside it. Unexplained bursts are candidates for "
               f"missed labels: nothing about missed labels follows until the sample is classified. Different population: "
               f"{_num(tot['labels_per_video_boxer_min'])} labels per whole-video boxer-minute, {lab['unattributed']} of "
               f"{lab['total']} labels ({_pct(lab['unattributed_share'])}) unattributed; "
               f"{lab['matched_to_a_burst']} labels matched to a burst ({lab['matched_own']} own, "
               f"{lab['matched_unattributed']} unattributed) -> {a.out}; sample of {len(sample)} of {population} -> {a.sample_out}")
    out_json = json.dumps(
        {"rule": rule, "tolerance_s": completeness.TOL_S, "reacquire": bool(a.reacquire),
         "matching": "one-to-one; optimal within each phase (most matches, then least total gap), own labels first, then "
                     "unattributed labels, so not necessarily the global maximum; ties to the burst that sorts first "
                     "(frame, boxer); hand not required",
         "exposure": "frames of processed segments with a valid signal for at least one hand, bridged frames included",
         "videos": _round(videos), "all": _round(tot),
         "sample": {"n": len(sample), "population_unexplained": population, "inclusion_prob": prob,
                    "design": "simple random sample without replacement over all unexplained bursts",
                    "seed": a.seed, "file": str(a.sample_out)}} | ({"skips": skip_json} if skip_json else {}),
        indent=1, allow_nan=False)
    sample_rows = [[b.video, b.frame, f"{b.time_s:.2f}", b.boxer, b.hand, f"{prob:.6f}", ""] for b in sample]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out_json, encoding="utf-8")
    a.sample_out.parent.mkdir(parents=True, exist_ok=True)
    with a.sample_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_COLUMNS)
        w.writerows(sample_rows)
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
