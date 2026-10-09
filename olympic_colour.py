"""Kit colour (red / blue / none) of every track id in the Olympic Boxing videos, and a self-check.

  python olympic_colour.py --data data/olympic --tracks results/ingest/olympic [--only GH078416 ...]
  python olympic_colour.py --data data/olympic --tracks results/ingest/olympic --check-only

Per video this decodes the footage once, reads a few regions of each tracked person (torso, head,
gloves; src/boxingvi/appearance.py says how and why) and writes <tracks>/<GoPro id>.colours.json: per
track id the decision (red / blue / none / unsure), the median red and blue shares and the number of
rows they come from. ONLY NUMBERS are kept: no crop, thumbnail or pixel array of the footage is
written anywhere (minors, academic licence). The file is gitignored with the rest of results/.

Then the non-circular self-check, the gate before any linking uses the colours: pairs of
fragments that BOTH carry a labelled punch and are on screen together for at least 0.5 s are two
different people, so a working colour test gives them OPPOSITE colours (chance: about half). The
labels never say red or blue, so this cannot be passed by construction. Written to
results/olympic_colour_check.json; the exit code is 2 when the pooled opposite share is below 0.9
(the colours do not work: do not build on them) and 1 when a video failed.

--check-only recomputes the check from existing .colours.json files (no decoding).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import olympic_link  # noqa: E402
from boxingvi import appearance, olympic  # noqa: E402

DEFAULT_CHECK_OUT = Path("results/olympic_colour_check.json")
GATE = 0.9


def colours_path(tracks: Path, vid: str) -> Path:
    olympic_link._check_video_id(vid)
    p = (tracks / f"{vid}.colours.json").resolve()
    if p.parent != tracks.resolve():
        raise ValueError("colours path left the tracks folder")
    return p


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="folder of <GoPro id>.tracks.npz files")
    ap.add_argument("--only", nargs="+", metavar="ID", help="only these GoPro ids (default: every video)")
    ap.add_argument("--check-out", type=Path, default=DEFAULT_CHECK_OUT)
    ap.add_argument("--check-only", action="store_true", help="recompute the self-check from existing colours files")
    a = ap.parse_args(argv)

    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    t_all = time.perf_counter()
    checks, report, failed = {}, {}, []
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if (only is not None and vid not in only) or not (a.tracks / f"{vid}.tracks.npz").is_file():
            continue
        t0 = time.perf_counter()
        try:
            table, info, events, attrs = olympic_link.load_video(t, a.tracks)     # refuses an unbound table
            path = colours_path(a.tracks, vid)
            if a.check_only:
                col = appearance.Colours.load(path)
                col.check_table(table)
            else:
                col = appearance.compute_colours(t.video, table, info)
                col.save(path)
            chk = appearance.colour_check(table, attrs, col)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            report[t.name] = {"error": str(e)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        wall = time.perf_counter() - t0
        n = col.counts()
        checks[t.name] = chk
        report[t.name] = chk | {"fragments": n, "wall_s": round(wall, 1), "video_bound": bool(table.video_fp)}
        print(f"{t.name}: red {n['red']} blue {n['blue']} none {n['none']} unsure {n['unsure']} of {len(col.by_id)} ids | "
              f"co-present credited pairs {chk['copresent_pairs']}, decided {chk['decided_pairs']}, opposite "
              f"{_pct(chk['opposite_share'])} | credited fragments {chk['credited_by_decision']} | {wall:.0f} s")
    if not checks:
        raise ValueError(f"no <GoPro id>.tracks.npz in {a.tracks} could be read against a task in {a.data}")
    pooled = appearance.pool_checks(checks.values())
    passes = pooled["opposite_share"] is not None and pooled["opposite_share"] >= GATE
    out = {"gate": f"pooled opposite share >= {GATE}", "passes": bool(passes), "videos": report, "all": pooled,
           "wall_s_total": round(time.perf_counter() - t_all, 1)}
    a.check_out.parent.mkdir(parents=True, exist_ok=True)
    a.check_out.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    print(f"ALL ({pooled['videos']} videos): {pooled['decided_pairs']} decided co-present credited pairs of "
          f"{pooled['copresent_pairs']}: opposite {pooled['opposite']} ({_pct(pooled['opposite_share'])}), same "
          f"{pooled['same']} ({_pct(pooled['same_share'])}); credited fragments {pooled['credited_by_decision']}; "
          f"gate {GATE:.0%}: {'PASS' if passes else 'FAIL'}; {out['wall_s_total']} s -> {a.check_out}")
    if failed:
        return 1
    return 0 if passes else 2


if __name__ == "__main__":
    sys.exit(main())
