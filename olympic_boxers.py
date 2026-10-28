"""The per-boxer pipeline on Olympic Boxing, with the pose-only linker (#11) as the identity input.

  python olympic_boxers.py --data data/olympic --tracks results/ingest/olympic
  ... --only GH078416 ...           # some videos only

WHAT CAN AND CANNOT BE MEASURED HERE (read first). The Olympic labels say the target and the thrower's own hand
(olympic.py); they do NOT say which boxer or corner threw a punch, and no label pins a boxer's identity over time
(STATE 10-24, option C). So there is no ground-truth identity, a perfect-identity run cannot be built, and how
much of a per-boxer count error is IDENTITY error cannot be measured. Punch types cannot be scored either: the
labels are target x hand, not the six BoxingVI types, and the boxers' stances are unknown, so nothing is named
here (the finder only). What this measures, per video and pooled, through the real per-boxer code path:

  1. The identity file format on real tracks: the linker's chains written as <video>.identity.csv (ungated:
     the linker has no identity gate) and read back through every check, for all 29 videos.
  2. Equivalence: the per-boxer finder's punches for A and B are EXACTLY completeness.find_bursts' bursts
     on the same linking (the tested path behind the completeness results).
  3. Per boxer, the count the report would print against the labels the linker's identity owns
     (completeness.event_owner), total and by hand (the boxer's own wrist against the label's own hand; that
     convention holds for 77.6% of credited punches). This is the finder's count error CONDITIONAL on the
     linker's identity: an identity swap moves punches and their labels together and is invisible here.
  4. What the identity leaves unassigned: punches found on fragments credited with a labelled punch (so a boxer)
     that the linker put in neither chain (unsure segments), as a share of all punches found; labels with no
     owner among the two boxers.
  5. Reference without any identity (credited fragments, ADR-0012 Option 2, #12): per credited fragment, the
     punches found against the labels credited to it, pooled. The gap between 3 and 5 is coverage, not identity.

Olympic outputs stay in the gitignored results/ (default results/olympic_boxers.json and identity files under
results/olympic_boxers/). Never publish anything derived; only pooled aggregates leave this machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

import find_punches  # noqa: E402
import olympic_link  # noqa: E402
from boxingvi import boxers, completeness, linking, olympic  # noqa: E402

DEFAULT_OUT = Path("results/olympic_boxers.json")
DEFAULT_IDENTITY_DIR = Path("results/olympic_boxers")
HANDS = ("left", "right")


def _ratio(a, b):
    return None if not b else a / b


def video_result(task, tracks_dir: Path, id_dir: Path, rule: dict) -> dict:
    table, _, events, attrs = olympic_link.load_video(task, tracks_dir)
    vid = task.video.stem
    tracks = tracks_dir / f"{vid}.tracks.npz"
    lk = linking.link_video(table, attrs)
    chains = set(lk.identities[0]) | set(lk.identities[1])
    credited = {a.track_id for a in attrs if a.track_id is not None}
    names = boxers.row_names_from_linking(table, lk, unsure_tracks=credited - chains)
    idp = id_dir / f"{vid}.identity.csv"
    boxers.write_identity(idp, table, tracks, boxers.segments_from_row_names(table, names), "linker",
                          {"verdict": "ungated"})
    ident = boxers.read_identity(idp, table, tracks)
    found = boxers.find(table, ident, rule)

    def punches(who):
        return [(int(f), HANDS[int(h)]) for _s, _t, _g, fr, hs in found[who]["kept"] for f, h in zip(fr, hs)]

    # 2. equivalence with the tested completeness path on the same linking
    old = completeness.find_bursts(table, lk, events, attrs, rule, video=vid)
    same = all(sorted(punches(b)) == sorted((x.frame, x.hand) for x in old["bursts"] if x.boxer == b)
               for b in boxers.BOXERS)
    owner = [completeness.event_owner(a, lk) for a in attrs]
    out = {"segments": len(ident.segments), "equivalent_to_completeness": same, "boxers": {}}
    for k, b in enumerate(boxers.BOXERS):
        p = punches(b)
        own = [events[i] for i, o in enumerate(owner) if o == k]
        out["boxers"][b] = {"found": len(p), "found_by_hand": {h: sum(x[1] == h for x in p) for h in HANDS},
                            "owned_labels": len(own), "owned_by_hand": {h: sum(e.hand == h for e in own) for h in HANDS},
                            "identified_frames": found[b]["stats"]["identified_frames"],
                            "identified_min": found[b]["stats"]["identified_frames"] / table.fps / 60,
                            "exposure_frames": found[b]["stats"]["exposure_frames"]}
    out["unsure_found"] = len(punches(boxers.UNSURE))
    out["unsure_tracks"] = len(credited - chains)
    out["labels"] = {"total": len(events), "owned": sum(o is not None for o in owner),
                     "credited_to_unsure_tracks": sum(a.track_id in (credited - chains) for a in attrs),
                     "not_credited": sum(a.track_id is None for a in attrs)}
    # 5. credited fragments, no identity: each credited id is one person for its whole length
    bridge, min_rows = boxers.finder_params(table.fps)
    by_track = boxers._rows_by_track(table)
    frag_found = frag_labels = 0
    for t in sorted(credited):
        scan = completeness._scan_segments(table, completeness._split_rows(table, by_track[t], bridge), rule,
                                           f"track {t}", bridge, min_rows)
        frag_found += len(scan["raw"])
        frag_labels += sum(a.track_id == t for a in attrs)
    out["credited_fragments"] = {"fragments": len(credited), "found": frag_found, "credited_labels": frag_labels}
    return out


def pooled(videos: dict) -> dict:
    ok = [v for v in videos.values() if "error" not in v]
    bv = [v["boxers"][b] for v in ok for b in boxers.BOXERS]
    found = sum(x["found"] for x in bv)
    owned = sum(x["owned_labels"] for x in bv)
    rel = [(x["found"] - x["owned_labels"]) / x["owned_labels"] for x in bv if x["owned_labels"]]
    unsure = sum(v["unsure_found"] for v in ok)
    lab = {k: sum(v["labels"][k] for v in ok) for k in ("total", "owned", "credited_to_unsure_tracks", "not_credited")}
    cf = {k: sum(v["credited_fragments"][k] for v in ok) for k in ("fragments", "found", "credited_labels")}
    return {
        "videos": len(ok), "boxer_videos": len(bv), "boxer_videos_with_owned_labels": len(rel),
        "all_equivalent_to_completeness": all(v["equivalent_to_completeness"] for v in ok),
        "found_on_boxers": found, "owned_labels": owned, "count_ratio": _ratio(found, owned),
        "count_ratio_by_hand": {h: _ratio(sum(x["found_by_hand"][h] for x in bv), sum(x["owned_by_hand"][h] for x in bv))
                                for h in HANDS},
        "relative_count_error_per_boxer_video": None if not rel else {
            "median": float(np.median(rel)), "q25": float(np.quantile(rel, 0.25)),
            "q75": float(np.quantile(rel, 0.75)), "median_abs": float(np.median(np.abs(rel)))},
        "unsure_found": unsure, "unassigned_share_of_found": _ratio(unsure, found + unsure),
        # the like-for-like whole-video comparison: every punch found on anyone the identity calls a boxer (A, B
        # or unsure) against every labelled punch; per boxer, two thirds of the labels have no owner
        "found_on_all_boxer_tracks_per_label": _ratio(found + unsure, lab["total"]),
        "labels": lab, "labels_unowned_share": _ratio(lab["total"] - lab["owned"], lab["total"]),
        "credited_fragments": cf | {"count_ratio": _ratio(cf["found"], cf["credited_labels"])},
        "identified_boxer_min": sum(x["identified_min"] for x in bv),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="folder of <GoPro id>.tracks.npz files")
    ap.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--identity-dir", type=Path, default=DEFAULT_IDENTITY_DIR)
    ap.add_argument("--only", nargs="+", metavar="ID")
    a = ap.parse_args(argv)
    rule = find_punches.load_rule(a.rule)
    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    a.identity_dir.mkdir(parents=True, exist_ok=True)
    videos, failed = {}, []
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if (only is not None and vid not in only) or not (a.tracks / f"{vid}.tracks.npz").is_file():
            continue
        try:
            r = video_result(t, a.tracks, a.identity_dir, rule)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            videos[t.name] = {"error": str(e)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        videos[t.name] = r
        print(f"{t.name}: A {r['boxers']['A']['found']} found / {r['boxers']['A']['owned_labels']} owned labels; "
              f"B {r['boxers']['B']['found']} / {r['boxers']['B']['owned_labels']}; unsure {r['unsure_found']}; "
              f"same as completeness: {r['equivalent_to_completeness']}")
    if not any("error" not in v for v in videos.values()):
        raise ValueError(f"nothing measured in {a.tracks} against {a.data}")
    tot = pooled(videos)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rule": rule, "identity": "linker (#11), ungated; credited fragments outside "
                                 "both chains are unsure", "all": tot, "videos": videos}, indent=1, allow_nan=False),
                     encoding="utf-8")
    print(json.dumps(tot, indent=1))
    print(f"-> {a.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
