"""Desktop side of the two-machine pose step: wait for the laptop's pose, then run the chain.

  python cli/wait_for_pose.py --video data/own/IMG_5835.mov --stance orthodox
  ... --rounds 0-180 --no-overlay --timeout 7200

Drop the clip into data/own (Syncthing carries it to the laptop, whose pose_watcher.py
runs pose on it), then run this. It waits until results/ingest/X.pose.npz has arrived
whole (no Syncthing temporary file, size and modification time unchanged across two
looks), checks it is THIS video's pose (analyse_clip.check_pose: name, content
fingerprint, frame count, rate, size, model), copies it into --out-dir in one rename,
and runs analyse_clip.py --skip-ingest, which checks it again. The exit code is
analyse_clip's.

Stops with exit 1, and runs nothing, when the laptop wrote X.pose.failed (its reason
is printed), when the pose belongs to another recording, or after --timeout seconds.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import analyse_clip  # noqa: E402
import pose_watcher  # noqa: E402
from boxingvi import classify  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from boxingvi.video import probe  # noqa: E402

MAX_FAILED_READ = 4096


def wait(npz: Path, failed: Path, timeout: float, poll: float, sleep=time.sleep, clock=time.monotonic) -> str:
    """'ready', 'failed' or 'timeout'. Ready = present, not being written by Syncthing, and the same size and
    mtime on two looks `poll` seconds apart."""
    end = clock() + timeout
    last = None
    told = False
    while True:
        if failed.exists():
            return "failed"
        if npz.is_file() and not npz.is_symlink() and not pose_watcher.syncing(npz.parent, npz.name):
            try:
                st = pose_watcher._state(npz)
            except OSError:          # replaced between the two calls: look again next time
                st = None
            if st is not None and st == last:
                return "ready"
            last = st
        else:
            last = None
            if not told:
                print(f"waiting for {npz} (the laptop's pose watcher writes it)", flush=True)
                told = True
        if clock() >= end:
            return "timeout"
        sleep(poll)


def main(argv=None, sleep=time.sleep, clock=time.monotonic, analyse=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--stance", choices=sorted(classify.STANCES), required=True, help="never guessed")
    ap.add_argument("--ingest-dir", type=Path, default=pose_watcher.DEFAULT_INGEST)
    ap.add_argument("--out-dir", type=Path, default=analyse_clip.DEFAULT_OUT)
    ap.add_argument("--weights", type=Path, default=analyse_clip.DEFAULT_WEIGHTS,
                    help="only its NAME is used: the pose must come from this model")
    ap.add_argument("--timeout", type=float, default=7200.0, help="seconds to wait (default 2 h)")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between looks")
    ap.add_argument("--rounds", help="passed to analyse_clip")
    ap.add_argument("--no-overlay", action="store_true", help="passed to analyse_clip")
    a = ap.parse_args(argv)
    if not (0 <= a.timeout <= 7 * 86400) or not (1 <= a.poll <= 600):
        ap.error("--timeout must be 0-604800 s and --poll 1-600 s")
    clip = a.video.stem
    if not analyse_clip._CLIP_NAME.fullmatch(clip) or not a.video.is_file():
        ap.error(f"{a.video}: not a video file with a clip name analyse_clip accepts")
    for r in pose_watcher.PARTICIPANT_ROOTS:
        if pose_watcher.under(a.video, r):
            ap.error(f"{a.video} is participant footage: the two-machine pose path is for your own footage only")

    npz, failed = pose_watcher.outputs(a.ingest_dir, clip)
    got = wait(npz, failed, a.timeout, a.poll, sleep, clock)
    if got == "failed":
        try:
            with open(failed, encoding="utf-8", errors="replace") as f:
                reason = f.read(MAX_FAILED_READ).strip()
        except OSError:
            reason = "(unreadable)"
        print(f"wait_for_pose: the laptop could not make {npz.name}: {reason}\n"
              f"(If that is an old failure, its deletion may not have synced here yet.) Fix the cause, delete {failed} on either machine, and the watcher will try again.", file=sys.stderr)
        return 1
    if got == "timeout":
        print(f"wait_for_pose: no {npz} after {a.timeout:g} s. Is the laptop on, logged in, its watcher running "
              "and Syncthing connected?", file=sys.stderr)
        return 1
    try:
        analyse_clip.check_pose(PoseSequence.load(npz), probe(a.video), a.video, a.weights.name)
    except Exception as e:
        print(f"wait_for_pose: {npz.name} is not this video's pose: {e}\nNothing was run.", file=sys.stderr)
        return 1

    a.out_dir.mkdir(parents=True, exist_ok=True)
    dest = a.out_dir / f"{clip}.pose.npz"
    fd, tmp = tempfile.mkstemp(prefix=f".{clip}.pose.", dir=a.out_dir)
    os.close(fd)
    try:
        shutil.copyfile(npz, tmp)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    print(f"{npz.name}: arrived and checked -> {dest}", flush=True)

    args = ["--video", str(a.video), "--stance", a.stance, "--skip-ingest", "--out-dir", str(a.out_dir),
            "--weights", str(a.weights)]
    if a.rounds:
        args += ["--rounds", a.rounds]
    if a.no_overlay:
        args.append("--no-overlay")
    return (analyse or analyse_clip.main)(args)


if __name__ == "__main__":
    sys.exit(main())
