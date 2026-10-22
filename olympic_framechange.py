"""Where did the camera drop frames? One `<id>.framechange.npz` per Olympic video.

  python olympic_framechange.py --data data/olympic --tracks results/ingest/olympic [--only GH069681 ...]

The GoPro files have regular packet timestamps but the camera drops captured frames: the picture jumps
forward in one frame and the wrist-speed rule reads that as a very fast punch (src/boxingvi/skips.py has
the rule and the declaration of where its constant comes from). For each video that has a
`<id>.tracks.npz` in --tracks this decodes the video once, writes the per-frame change (one number per
frame: the mean absolute grey difference to the previous frame, on every 4th pixel) with the video's
file name, frame count and content fingerprint to `<tracks>/<id>.framechange.npz`, and prints the
frames, the skips and one skip every N seconds. No image is ever written or shown: only the numbers.
An existing framechange file is replaced only with --overwrite.

Then: python olympic_completeness.py ... --skips
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import olympic_link  # noqa: E402
from boxingvi import olympic, skips, video  # noqa: E402


def framechange_path(tracks: Path, vid: str) -> Path:
    return tracks / f"{vid}.framechange.npz"


def one(task, tracks: Path) -> dict:
    """Measure one video and write its file; returns the numbers printed."""
    t0 = time.time()
    info = video.probe(task.video)
    change = video.frame_change(task.video, info, step=skips.STEP)
    n_probe = info.timing.n_frames if info.timing else None
    if n_probe is not None and len(change) != n_probe:
        raise ValueError(f"{task.video.name}: decoded {len(change)} frames, the packet scan says {n_probe}")
    s = skips.Skips(change, task.video.name, video.fingerprint(task.video))
    frames = s.frames
    s.save(framechange_path(tracks, task.video.stem))
    secs = len(change) / info.fps
    return {"frames": len(change), "skips": len(frames), "every_s": secs / len(frames) if len(frames) else None,
            "per_min": len(frames) / (secs / 60), "wall_s": time.time() - t0, "video_s": secs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing framechange file")
    a = ap.parse_args(argv)
    only = {olympic_link._check_video_id(v) for v in a.only} if a.only else None
    tot_skips, tot_s, tot_wall, failed, done = 0, 0.0, 0.0, [], 0
    for t in olympic.find_tasks(a.data):
        vid = t.video.stem
        if (only is not None and vid not in only) or not (a.tracks / f"{vid}.tracks.npz").is_file():
            continue
        out = framechange_path(a.tracks, vid)
        if out.exists() and not a.overwrite:
            print(f"{t.name}: {out.name} exists, left alone (--overwrite replaces it)")
            continue
        try:
            r = one(t, a.tracks)
        except (ValueError, RuntimeError, OSError) as e:
            failed.append(t.name)
            print(f"{t.name}: FAILED, {e}", flush=True)
            continue
        done += 1
        tot_skips += r["skips"]
        tot_s += r["video_s"]
        tot_wall += r["wall_s"]
        every = "no skip" if r["every_s"] is None else f"one skip every {r['every_s']:.1f} s"
        print(f"{t.name} ({vid}): {r['frames']} frames, {r['skips']} skips, {every} ({r['per_min']:.1f} per minute), "
              f"{r['wall_s']:.0f} s wall", flush=True)
    if done:
        print(f"ALL: {done} videos, {tot_skips} skips in {tot_s / 60:.1f} video minutes = {tot_skips / (tot_s / 60):.1f} per "
              f"minute, {tot_wall:.0f} s wall")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
