"""Kit colour (red / blue / none) of every track id in the Olympic Boxing videos, and a self-check.

  python cli/olympic_colour.py --data data/olympic --tracks results/ingest/olympic                # dev videos only
  python cli/olympic_colour.py --data data/olympic --tracks results/ingest/olympic --split test   # the 25 test videos
  python cli/olympic_colour.py --data data/olympic --tracks results/ingest/olympic --check-only

Per video this decodes the footage once, reads the torso of each tracked person (src/boxingvi/
appearance.py says how and why) and writes <tracks>/<GoPro id>.colours.json: per track id the
decision (red / blue / none / unsure), the median kit index and kit share, and the row counts they
come from. ONLY NUMBERS are kept: no crop, thumbnail or pixel array of the footage is written
anywhere (minors, academic licence). The file is gitignored with the rest of results/.

DEVELOPMENT / TEST SPLIT (version 2, fixed by video BEFORE the rule was designed; never below the
video level). The rule was designed by looking at any colour statistic of the 4 DEV videos and at
none of the 25 TEST videos; the rule was then frozen and committed, and only afterwards run on the
test videos. --split dev (the default, so a careless run cannot read a test number) / test / all.
The gate is evaluated on the TEST videos only. Dev numbers are reported separately and are not
evidence.

The non-circular self-check, the gate before any linking uses the colours: pairs of fragments that
BOTH carry a labelled punch and are on screen together for at least 0.5 s are two different
people, so a working colour test gives them OPPOSITE colours (chance: about half). The labels
never say red or blue, so this cannot be passed by construction. PASS needs, pooled over the test
videos, an opposite share >= 0.90 AND >= 100 decided pairs. Per camera, fewer than 20 decided test
pairs means "colour not established" for that camera. Written to
results/olympic_colour_check_v2.<split>.json; the exit code is 2 when the test gate fails (the
colours do not work: do not build on them) and 1 when a video failed.

--check-only recomputes the check from existing .colours.json files (no decoding).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import olympic_link  # noqa: E402
from boxingvi import appearance, olympic  # noqa: E402

# Fixed on 2026-10-08, by video, before any version-2 design: the two videos already inspected
# (GH078416 kam2, GH159681 kam4) plus one more per camera drawn by random.Random(20261008) from the
# rest. Everything else is TEST.
DEV_VIDEOS = frozenset({"GH078416", "GH159681", "GH128416", "GH069681"})
GATE = 0.9
GATE_MIN_PAIRS = 100        # decided test pairs needed (pooled) for the gate to mean anything
CAMERA_MIN_PAIRS = 20       # fewer decided test pairs in a camera: colour not established there
SPLITS = ("dev", "test", "all")


def colours_path(tracks: Path, vid: str) -> Path:
    olympic_link._check_video_id(vid)
    p = (tracks / f"{vid}.colours.json").resolve()
    if p.parent != tracks.resolve():
        raise ValueError("colours path left the tracks folder")
    return p


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def split_of(vid: str) -> str:
    return "dev" if vid in DEV_VIDEOS else "test"


def _pooled(rows: "dict[str, dict]", names) -> dict:
    return appearance.pool_checks([rows[n] for n in names])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="folder of <GoPro id>.tracks.npz files")
    ap.add_argument("--only", nargs="+", metavar="ID", help="only these GoPro ids (default: every video of the split)")
    ap.add_argument("--split", choices=SPLITS, default="dev",
                    help="dev (default): the 4 development videos; test: the 25 held-out videos; all: both")
    ap.add_argument("--check-out", type=Path, default=None,
                    help="default results/olympic_colour_check_v2.<split>.json")
    ap.add_argument("--min-pairs", type=int, default=GATE_MIN_PAIRS, help="decided test pairs the gate needs")
    ap.add_argument("--min-camera-pairs", type=int, default=CAMERA_MIN_PAIRS,
                    help="decided test pairs a camera needs for its colour to count as established")
    ap.add_argument("--check-only", action="store_true", help="recompute the self-check from existing colours files")
    a = ap.parse_args(argv)
    if a.check_out is None:
        a.check_out = Path(f"results/olympic_colour_check_v2.{a.split}.json")

    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    t_all = time.perf_counter()
    checks, report, failed, camera = {}, {}, [], {}
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if a.split != "all" and split_of(vid) != a.split:
            continue
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
            report[t.name] = {"error": str(e), "split": split_of(vid)}
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}")
            continue
        wall = time.perf_counter() - t0
        n = col.counts()
        checks[t.name] = chk
        camera[t.name] = t.camera
        report[t.name] = chk | {"split": split_of(vid), "camera": t.camera, "fragments": n, "wall_s": round(wall, 1),
                                "video_bound": bool(table.video_fp)}
        print(f"{t.name} [{split_of(vid)}]: red {n['red']} blue {n['blue']} none {n['none']} unsure {n['unsure']} of "
              f"{len(col.by_id)} ids | co-present credited pairs {chk['copresent_pairs']}, decided {chk['decided_pairs']}, "
              f"opposite {_pct(chk['opposite_share'])} | credited fragments {chk['credited_by_decision']} | {wall:.0f} s")
    if not checks:
        raise ValueError(f"no <GoPro id>.tracks.npz in {a.tracks} could be read against a task in {a.data}")

    out = {"gate": f"TEST videos pooled: opposite share >= {GATE} and >= {a.min_pairs} decided pairs",
           "split_run": a.split, "dev_videos": sorted(DEV_VIDEOS), "videos": report}
    for sp in ("dev", "test"):
        names = [n for n in checks if report[n]["split"] == sp]
        if not names:
            continue
        block = {"all": _pooled(checks, names)}
        for cam in sorted({camera[n] for n in names}):
            block[cam] = _pooled(checks, [n for n in names if camera[n] == cam])
        if sp == "test":
            for cam in sorted({camera[n] for n in names}):
                block[cam]["established"] = block[cam]["decided_pairs"] >= a.min_camera_pairs
        out[sp] = block
        pl = block["all"]
        tag = "NOT EVIDENCE" if sp == "dev" else "GATE DATA"
        print(f"{sp.upper()} ({pl['videos']} videos, {tag}): {pl['decided_pairs']} decided co-present credited pairs of "
              f"{pl['copresent_pairs']}: opposite {pl['opposite']} ({_pct(pl['opposite_share'])}), same {pl['same']} "
              f"({_pct(pl['same_share'])}); credited fragments {pl['credited_by_decision']}")
        for cam in sorted({camera[n] for n in names}):
            c = block[cam]
            est = "" if sp == "dev" else (" established" if c["established"] else f" NOT ESTABLISHED (< {a.min_camera_pairs} decided pairs)")
            print(f"    {cam}: {c['videos']} videos, decided {c['decided_pairs']}, opposite {_pct(c['opposite_share'])}{est}")
    passes = None
    if "test" in out:
        tp = out["test"]["all"]
        passes = bool(tp["opposite_share"] is not None and tp["opposite_share"] >= GATE and tp["decided_pairs"] >= a.min_pairs)
    out["passes"] = passes
    out["wall_s_total"] = round(time.perf_counter() - t_all, 1)
    a.check_out.parent.mkdir(parents=True, exist_ok=True)
    a.check_out.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    if passes is None:
        print(f"gate not evaluated (no test video in this run); {out['wall_s_total']} s -> {a.check_out}")
    else:
        print(f"gate (test, opposite >= {GATE:.0%} and >= {a.min_pairs} decided pairs): {'PASS' if passes else 'FAIL'}; "
              f"{out['wall_s_total']} s -> {a.check_out}")
    if failed:
        return 1
    return 0 if passes in (None, True) else 2


if __name__ == "__main__":
    sys.exit(main())
