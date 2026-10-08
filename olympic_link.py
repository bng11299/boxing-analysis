"""Link each Olympic Boxing video's ByteTrack ids into two boxer identities.

  python olympic_link.py --data data/olympic --tracks results/ingest/olympic
  python olympic_link.py --data data/olympic --tracks results/ingest/olympic --overlay          # GH078416
  python olympic_link.py --data data/olympic --tracks results/ingest/olympic --overlay GH099681

ByteTrack gives each boxer hundreds of short ids (olympic_attribute.py measured it).
For each video this chains the fragments into TWO identities (src/boxingvi/linking.py:
forward-only links with a speed limit, scored by credited punches and a ring prior,
solved exactly as an integer program) and reports, per video:

  * the share of frames where both boxers, and each, are identified;
  * the share of CREDITED punches whose id belongs to boxer A or B -- a real fact: punches
    come from the boxers, not the crowd. Punches also drive the linking, so it is also
    given with every second punch held out of the linking (`credited_in_either_share_heldout`);
  * how that splits A vs B, the fragments per identity, the longest unidentified stretch;
  * co-presence: how much of the shorter identity's time the other is also identified. Two
    real boxers are on screen together; one boxer split into two identities never is.

Writes results/olympic_linking.json (gitignored: derived from Olympic Boxing, never published).
--overlay renders ONE video with boxer A green, boxer B magenta and everyone else thin grey to
<tracks>/<GoPro id>.linked.overlay.mp4, for checking identity swaps by eye.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import linking, olympic, overlay  # noqa: E402
from boxingvi.pose_yolo import PoseTable, verify_binding  # noqa: E402
from boxingvi.video import probe  # noqa: E402

DEFAULT_OUT = Path("results/olympic_linking.json")
DEFAULT_OVERLAY = "GH078416"      # the shortest video, 1.1 min
VIDEO_ID = re.compile(r"[A-Za-z0-9]{1,32}")   # a GoPro id; it becomes part of an output file name


def load_video(task: "olympic.Task", tracks: Path):
    """(table, info, events, attributions) for one task, with every cross-check: the table is
    for THIS video (file name; content fingerprint when the table has one; size, rate) and
    covers all of it. Refuses otherwise, never guesses. A table written before fingerprints
    existed has none and is accepted on the name alone: `table.video_fp == ""` says so, and
    main() reports it as `video_bound: false`."""
    f = tracks / f"{task.video.stem}.tracks.npz"
    if not f.is_file():
        raise FileNotFoundError(f"no {f.name}")
    table = PoseTable.load(f)
    try:
        verify_binding(table, task.video)
    except ValueError as e:
        raise ValueError(f"{f.name}: {e}") from e
    info = probe(task.video)
    if (table.width, table.height) != (info.width, info.height) or abs(table.fps - info.fps) > 1e-6:
        raise ValueError(f"{f.name} was not extracted from {task.video.name}")
    n_video = info.timing.n_frames if info.timing else None
    if n_video is None or table.n_frames != n_video:
        raise ValueError(f"{f.name} covers {table.n_frames} of {n_video} frames: linking needs the whole video")
    events, _ = olympic.load_events(task.annotations, n_video, info.width, info.height)
    return table, info, events, olympic.attribute(events, table)


def analyse(table, attrs, params: linking.LinkParams = linking.LinkParams()) -> "tuple[linking.Linking, dict]":
    lk = linking.link_video(table, attrs, params)
    out = linking.summarise(lk, attrs)
    out["ring"] = None if lk.region is None else [round(v, 1) for v in
                                                  (lk.region.x0, lk.region.y0, lk.region.x1, lk.region.y1)]
    out["identity_fragments"] = [list(c) for c in lk.identities]
    out["link_solve_s"] = round(lk.solve_s, 3)
    # Held out: link on every other credited punch, then ask where the OTHER half went.
    mine, theirs = attrs[0::2], attrs[1::2]
    cred = [a for a in theirs if a.track_id is not None]
    try:
        held = linking.link_video(table, mine, params)
    except ValueError:       # too few credited punches in half the video to estimate a ring
        out["credited_in_either_share_heldout"], out["credited_heldout"], out["ring_heldout"] = None, 0, None
        return lk, out
    # The ring the held-out linking used: it must be a function of `mine` alone (tested).
    out["ring_heldout"] = None if held.region is None else [round(v, 1) for v in
                                                            (held.region.x0, held.region.y0, held.region.x1, held.region.y1)]
    inside = sum(a.track_id in held.ids(0) or a.track_id in held.ids(1) for a in cred)
    out["credited_in_either_share_heldout"] = round(inside / max(len(cred), 1), 4)
    out["credited_heldout"] = len(cred)
    return lk, out


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _check_video_id(vid: str) -> str:
    if not isinstance(vid, str) or not VIDEO_ID.fullmatch(vid):
        raise ValueError(f"{vid!r} is not a GoPro id (letters and digits only)")
    return vid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="folder of <GoPro id>.tracks.npz files")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--reacquire", action="store_true",
                    help=f"allow links across gaps of up to {linking.REACQUIRE_ON_S:.0f} s with no speed limit "
                         "(off = the model as first specified; identity labels are then only reliable between re-acquisitions)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="only these GoPro ids (default: every video)")
    ap.add_argument("--overlay", nargs="?", const=DEFAULT_OVERLAY, default=None, metavar="ID",
                    help=f"render one video's linked overlay (default {DEFAULT_OVERLAY})")
    a = ap.parse_args(argv)

    params = linking.LinkParams(reacquire_s=linking.REACQUIRE_ON_S if a.reacquire else 0.0)
    only = {_check_video_id(v) for v in a.only} if a.only else None
    want_overlay = _check_video_id(a.overlay) if a.overlay is not None else None
    tasks = olympic.find_tasks(a.data)
    if want_overlay and want_overlay not in {t.video.stem for t in tasks}:
        raise ValueError(f"--overlay {want_overlay}: no such video in {a.data}")

    report, failed = {}, []
    for t in tasks:
        vid = t.video.stem
        if only is not None and vid not in only:
            continue
        if not (a.tracks / f"{vid}.tracks.npz").is_file():
            continue
        try:
            table, info, events, attrs = load_video(t, a.tracks)
            lk, r = analyse(table, attrs, params)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            report[t.name] = {"error": str(e)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        report[t.name] = r | {"track_ids": len(table.ids()), "events": len(events),
                              "video_bound": bool(table.video_fp)}
        print(f"{t.name}: both {r['both_identified']:.0%} A {r['identified_a']:.0%} B {r['identified_b']:.0%} | "
              f"credited punches in A/B {r['credited_in_a']}/{r['credited_in_b']} of {r['credited_punches']} "
              f"({r['credited_in_either_share']:.0%}; {_pct(r['credited_in_either_share_heldout'])} held out) | "
              f"co-presence {r['co_presence']:.0%} | fragments {r['fragments_a']}+{r['fragments_b']} of "
              f"{len(table.ids())} | solve {r['link_solve_s']}s | longest gap A {r['longest_unidentified_a_s']}s B {r['longest_unidentified_b_s']}s")
        if want_overlay == vid:
            out = (a.tracks / f"{vid}.linked.overlay.mp4").resolve()
            if out.parent != a.tracks.resolve():
                raise ValueError("overlay path left the tracks folder")
            n = overlay.render_linked_overlay(t.video, table, lk, out, info=info)
            print(f"  overlay: {n} frames -> {out}")
    ok = [r for r in report.values() if "error" not in r]
    if not ok:
        raise ValueError(f"no <GoPro id>.tracks.npz in {a.tracks} could be linked against a task in {a.data}")
    pooled = {
        "videos_linked": len(ok), "videos_failed": len(failed),
        "credited_punches": sum(r["credited_punches"] for r in ok),
        "credited_in_a": sum(r["credited_in_a"] for r in ok), "credited_in_b": sum(r["credited_in_b"] for r in ok),
        "credited_heldout": sum(r["credited_heldout"] for r in ok),
        "videos_content_bound": sum(r["video_bound"] for r in ok),
        "link_solve_s_total": round(sum(r["link_solve_s"] for r in ok), 2),
        "link_solve_s_max": round(max(r["link_solve_s"] for r in ok), 2),
        "median_both_identified": round(statistics.median(r["both_identified"] for r in ok), 4),
        "median_co_presence": round(statistics.median(r["co_presence"] for r in ok), 4),
    }
    pooled["credited_in_either_share"] = round((pooled["credited_in_a"] + pooled["credited_in_b"])
                                               / max(pooled["credited_punches"], 1), 4)
    pooled["credited_in_either_share_heldout"] = round(
        sum(r["credited_in_either_share_heldout"] * r["credited_heldout"] for r in ok if r["credited_heldout"])
        / max(pooled["credited_heldout"], 1), 4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"videos": report, "all": pooled}, indent=1, allow_nan=False), encoding="utf-8")
    print(f"ALL ({pooled['videos_linked']} videos): {pooled['credited_in_a']}+{pooled['credited_in_b']} of "
          f"{pooled['credited_punches']} credited punches are on boxer A / B "
          f"({pooled['credited_in_either_share']:.1%}; {pooled['credited_in_either_share_heldout']:.1%} with half "
          f"held out of the linking); median both-identified {pooled['median_both_identified']:.0%}, "
          f"median co-presence {pooled['median_co_presence']:.0%}; {pooled['videos_content_bound']} of "
          f"{pooled['videos_linked']} tables are bound to their video by content (the rest by file name only); "
          f"solver {pooled['link_solve_s_total']} s in total, slowest {pooled['link_solve_s_max']} s -> {a.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
