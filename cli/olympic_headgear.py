"""Headgear corner colour (kit colour v3) per read row, and the gate G1 self-check.

  python cli/olympic_headgear.py --data data/olympic --tracks results/ingest/olympic                 # dev videos only
  python cli/olympic_headgear.py --data data/olympic --tracks results/ingest/olympic --split heldout  # gate G1, once
  python cli/olympic_headgear.py --data data/olympic --tracks results/ingest/olympic --check-only     # no decoding

The gate is docs/gates/appearance-identity.md (committed before this rule was designed). The rule is
src/boxingvi/headgear.py. Per video this decodes the footage once and writes
<counts-dir>/<GoPro id>.headgear.npz (default: the tracks folder): three counts per read row (red, blue and crown pixels). ONLY NUMBERS are kept,
in the gitignored results/ (minors, academic licence; nothing derived from Olympic Boxing is ever published).

Splits (never below the video):
  dev      ADR-0013's four development videos (the default, so a careless run cannot read a held-out number)
  heldout  the other 25 minus GH089681 and GH118416 (Brendan saw their kit on 2026-10-22) = 23 videos: G1
  all      both, reported separately
The held-out run also reports the ROBUSTNESS row: the held-out set without the videos whose GoPro chapter number
equals a dev video's on the other camera (20 videos).

G1 PASS, pooled over the held-out videos: opposite share >= 0.90 AND >= 100 decided pairs AND the 95%
video-cluster bootstrap lower bound >= 0.80. Written to results/olympic_headgear_check.<split>.json; exit 2 when
the gate fails, 1 when a video failed, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import olympic_link  # noqa: E402
from olympic_colour import DEV_VIDEOS  # noqa: E402
from boxingvi import headgear, olympic  # noqa: E402

EYE_CHECKED = frozenset({"GH089681", "GH118416"})          # their kit was seen by eye: out of the held-out set
# The frozen held-out manifest (docs/gates/appearance-identity.md): the gate needs EXACTLY these, each read once.
HELD_OUT = frozenset({"GH088416", "GH098416", "GH108416", "GH138416", "GH148416", "GH158416", "GH168416", "GH178416",
                      "GH188416", "GH198416", "GH208416", "GH218416", "GH079681", "GH099681", "GH109681", "GH119681",
                      "GH129681", "GH139681", "GH149681", "GH169681", "GH179681", "GH189681", "GH199681"})
CHAPTER_TWINS = frozenset({"GH079681", "GH158416", "GH129681"})   # same chapter number as a dev video
GATE_SHARE = 0.90
GATE_MIN_PAIRS = 100
GATE_BOOT_LO = 0.80
GATE_MAX_DROPPED = 0.01     # bootstrap draws with no decided pair are dropped and counted; over 1% fails the gate
CAMERA_MIN_PAIRS = 20
SPLITS = ("dev", "heldout", "all")


def split_of(vid: str) -> str:
    if vid in DEV_VIDEOS:
        return "dev"
    if vid in EYE_CHECKED:
        return "excluded"
    if vid in HELD_OUT:
        return "heldout"
    return "unknown"


def counts_path(folder: Path, vid: str) -> Path:
    olympic_link._check_video_id(vid)
    p = (folder / f"{vid}.headgear.npz").resolve()
    if p.parent != folder.resolve():
        raise ValueError("headgear counts path left the tracks folder")
    return p


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def block(checks: dict, names, camera: dict) -> dict:
    out = {"all": headgear.pool([checks[n] for n in names])}
    out["all"]["bootstrap95"] = headgear.cluster_bootstrap([(checks[n]["opposite"], checks[n]["decided_pairs"])
                                                            for n in names])
    for cam in sorted({camera[n] for n in names}):
        c = headgear.pool([checks[n] for n in names if camera[n] == cam])
        c["established"] = c["decided_pairs"] >= CAMERA_MIN_PAIRS
        out[cam] = c
    return out


def gate(pooled: dict) -> bool:
    b = pooled["bootstrap95"]
    return bool(pooled["opposite_share"] is not None and pooled["opposite_share"] >= GATE_SHARE
                and pooled["decided_pairs"] >= GATE_MIN_PAIRS and b is not None and b["lo"] >= GATE_BOOT_LO
                and b["dropped"] <= GATE_MAX_DROPPED * b["draws"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True, help="folder of <GoPro id>.tracks.npz files")
    ap.add_argument("--split", choices=SPLITS, default="dev")
    ap.add_argument("--counts-dir", type=Path, default=None, help="where the .headgear.npz files go (default: --tracks)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="only these GoPro ids of the split (no gate verdict)")
    ap.add_argument("--check-only", action="store_true", help="recompute the check from existing counts files")
    ap.add_argument("--check-out", type=Path, default=None, help="default results/olympic_headgear_check.<split>.json")
    a = ap.parse_args(argv)
    if a.check_out is None:
        a.check_out = Path(f"results/olympic_headgear_check.{a.split}.json")
    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    counts_dir = a.tracks if a.counts_dir is None else a.counts_dir
    counts_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    checks, report, camera, failed, unknown, seen = {}, {}, {}, [], [], set()
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if vid in seen:
            raise ValueError(f"video id {vid} appears in two task folders: refusing to guess which one is meant")
        seen.add(vid)
        sp = split_of(vid)
        if sp == "unknown":
            unknown.append(vid)           # not in the frozen manifest: never read, never pooled
            continue
        if sp == "excluded" or (a.split != "all" and sp != a.split):
            continue
        if only is not None and vid not in only:
            continue
        t0 = time.perf_counter()
        try:
            table, info, events, attrs = olympic_link.load_video(t, a.tracks)
            path = counts_path(counts_dir, vid)
            if a.check_only:
                cnt = headgear.HeadgearCounts.load(path)
            else:
                cnt = headgear.compute_headgear(t.video, table, info)
                cnt.save(path)
            chk = headgear.headgear_check(table, attrs, cnt)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            report[vid] = {"error": str(e), "split": sp}
            failed.append(vid)
            print(f"{vid}: FAILED, {e}")
            continue
        checks[vid], camera[vid] = chk, t.camera
        report[vid] = chk | {"split": sp, "camera": t.camera, "chapter_twin": vid in CHAPTER_TWINS,
                             "rows_read": int(len(cnt.rows)), "wall_s": round(time.perf_counter() - t0, 1)}
        print(f"{vid} [{sp} {t.camera}]: co-present credited pairs {chk['copresent_pairs']}, decided "
              f"{chk['decided_pairs']}, opposite {chk['opposite']} ({_pct(chk['opposite_share'])}) | "
              f"{report[vid]['wall_s']} s", flush=True)
    if not checks and not failed:
        raise ValueError(f"no <GoPro id>.tracks.npz in {a.tracks} could be read for split {a.split!r}")

    out = {"gate": f"held-out pooled: opposite >= {GATE_SHARE}, >= {GATE_MIN_PAIRS} decided pairs, "
                   f"video-cluster bootstrap 95% lower bound >= {GATE_BOOT_LO}",
           "rule": "headgear crown, read on the co-present window (src/boxingvi/headgear.py)",
           "split_run": a.split, "only": sorted(only) if only else None, "videos": report, "failed": failed,
           "unknown_videos_not_read": unknown}
    for sp in ("dev", "heldout"):
        names = [n for n in checks if report[n]["split"] == sp]
        if not names:
            continue
        out[sp] = block(checks, names, camera)
        if sp == "heldout":
            robust = [n for n in names if n not in CHAPTER_TWINS]
            if robust:
                out["heldout_without_chapter_twins"] = block(checks, robust, camera)
        pl = out[sp]["all"]
        tag = "NOT EVIDENCE" if sp == "dev" else "GATE DATA"
        b = pl["bootstrap95"]
        print(f"{sp.upper()} ({pl['videos']} videos, {tag}): opposite {pl['opposite']} of {pl['decided_pairs']} decided "
              f"({_pct(pl['opposite_share'])}; Wilson {pl['wilson95']}; video bootstrap "
              f"{None if b is None else [round(b['lo'], 3), round(b['hi'], 3)]}); decided {_pct(pl['decided_share'])} "
              f"of {pl['copresent_pairs']} co-present credited pairs")
        for cam, c in out[sp].items():
            if cam != "all":
                est = "established" if c["established"] else f"NOT ESTABLISHED (< {CAMERA_MIN_PAIRS} decided)"
                print(f"    {cam}: {c['videos']} videos, opposite {c['opposite']} of {c['decided_pairs']} "
                      f"({_pct(c['opposite_share'])}), {est}")
    passes = None
    missing = []
    if a.split in ("heldout", "all") and only is None:
        missing = sorted(HELD_OUT - {n for n in checks if report[n]["split"] == "heldout"})
        out["heldout_missing"] = missing
        if not missing and not failed:
            passes = gate(out["heldout"]["all"])
        else:
            print(f"G1 NOT EVALUATED: the held-out set is incomplete ({len(missing)} missing, {len(failed)} failed)")
    out["passes"] = passes
    out["wall_s_total"] = round(time.perf_counter() - t_all, 1)
    a.check_out.parent.mkdir(parents=True, exist_ok=True)
    a.check_out.write_text(json.dumps(out, indent=1, allow_nan=False), encoding="utf-8")
    print(f"G1: {'not evaluated' if passes is None else ('PASS' if passes else 'FAIL')}; "
          f"{out['wall_s_total']} s -> {a.check_out}")
    if failed or missing:
        return 1
    return 0 if passes in (None, True) else 2


if __name__ == "__main__":
    sys.exit(main())
