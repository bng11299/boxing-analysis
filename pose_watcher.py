"""Laptop side of the two-machine pose step: watch data/own, run pose on each new clip.

  python pose_watcher.py                       # on the GPU laptop, forever (scheduled task at logon)
  python pose_watcher.py --once --settle 0     # two looks, --interval apart, then exit
  python pose_watcher.py --log ~/boxing-comps-pose-watcher.log

Syncthing brings a clip dropped into data/own on the desktop to this machine. Every
--interval seconds this looks at data/own (top level only) and, for each video X that
has neither results/ingest/X.pose.npz nor results/ingest/X.pose.failed:

1. waits until Syncthing has finished with it: no Syncthing temporary file for it, its
   size and modification time the same on at least two passes (never on the first look), and
   unchanged for --settle seconds by this watcher's clock (not the file's mtime, which
   Syncthing copies from the source);
2. runs run_ingest.py (YOLO11-pose against the PINNED weights hash, the same pin
   analyse_clip.py uses) into a private staging folder OUTSIDE every synced folder;
3. checks the pose is this video's, whole (analyse_clip.check_pose: name, content
   fingerprint, frame count, rate, size, model) and that the video did not change while
   it was read;
4. moves it into results/ingest in one rename, so Syncthing never sees a partial file.

Syncthing then carries X.pose.npz back to the desktop, where wait_for_pose.py picks it
up. A clip whose pose fails gets X.pose.failed (one line: why) and is not retried until
that file is deleted. One watcher at a time: a second one exits.

Only your own footage: a --src under a participant folder is refused, as is a staging
folder inside --ingest-dir. Nothing here reads or writes outside the folders named.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import analyse_clip  # noqa: E402
import run_ingest  # noqa: E402
from boxingvi.boxers import PARTICIPANT_ROOTS  # noqa: E402
from boxingvi.pose_yolo import PoseSequence  # noqa: E402
from boxingvi.video import probe  # noqa: E402

DEFAULT_SRC = Path("data/own")
DEFAULT_INGEST = run_ingest.DEFAULT_OUT                  # results/ingest, the synced folder
DEFAULT_STAGING = Path("results/.pose-staging")        # same drive, never synced
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".mkv", ".avi"}
MAX_REASON = 500
MAX_UNRECORDED = 3   # failures with no .failed written before a clip is left alone until restart


def log(msg: str, logfile: "Path | None" = None) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    if logfile is not None:
        try:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_folders(src: Path, ingest: Path, staging: Path, roots=PARTICIPANT_ROOTS) -> None:
    """Deny by default, paths resolved first: own footage only, and staging never inside a synced folder."""
    for r in roots:
        if under(src, r):
            raise ValueError(f"{src} is under a participant folder: the watcher handles your own footage only")
    for synced in (ingest, src):
        if under(staging, synced):
            raise ValueError(f"the staging folder {staging} must not be inside {synced} (Syncthing would sync "
                             "partial files)")
    if under(ingest, src) or under(src, ingest):
        raise ValueError("--src and --ingest-dir must be separate folders")


def syncing(folder: Path, name: str) -> bool:
    """Syncthing writes into a temporary file and renames it at the end: `.syncthing.NAME.tmp`, and
    `~syncthing~NAME.tmp` on Windows. Either one present means the transfer is not finished."""
    return any((folder / t).exists() for t in (f".syncthing.{name}.tmp", f"~syncthing~{name}.tmp"))


def candidates(src: Path) -> "list[Path]":
    """Top-level regular video files with a clip name analyse_clip accepts. Links are skipped: the watcher
    reads only files that are really in --src. Two videos with one stem (X.mov and X.mp4) are both skipped,
    as their outputs would collide."""
    found: "dict[str, list[Path]]" = {}
    try:
        entries = sorted(src.iterdir())
    except OSError:
        return []
    for p in entries:
        if p.suffix.lower() not in VIDEO_EXTS or not analyse_clip._CLIP_NAME.fullmatch(p.stem):
            continue
        if p.is_symlink() or not p.is_file():
            continue
        found.setdefault(p.stem, []).append(p)
    return [v[0] for v in found.values() if len(v) == 1]


def outputs(ingest: Path, stem: str) -> "tuple[Path, Path]":
    return ingest / f"{stem}.pose.npz", ingest / f"{stem}.pose.failed"


def same_device(a: Path, b: Path, stat=os.stat) -> bool:
    """os.replace only renames within one volume: staging and ingest must share one. (--src is only read, never
    renamed from, so it may live anywhere.)"""
    return stat(a).st_dev == stat(b).st_dev


def write_atomic(path: Path, text: str, staging: Path) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".w.", dir=staging)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _state(p: Path) -> "tuple[int, int]":
    st = p.stat()
    return st.st_size, st.st_mtime_ns


class Watcher:
    def __init__(self, src: Path, ingest: Path, staging: Path, weights: Path, weights_sha256: str, device: str,
                 settle: float, predict=None, logfile: "Path | None" = None, clock=time.time):
        self.src, self.ingest, self.staging = src, ingest, staging
        self.weights, self.sha, self.device = weights, weights_sha256, device
        self.settle, self.predict, self.logfile, self.clock = settle, predict, logfile, clock
        self.seen: "dict[Path, tuple[tuple[int, int], float]]" = {}   # state, when first seen in it
        self.waiting_logged: "set[Path]" = set()
        self.unrecorded: "dict[Path, int]" = {}   # failures whose .failed could not be written

    def log(self, msg: str) -> None:
        log(msg, self.logfile)

    def ready(self, video: Path) -> bool:
        """Finished arriving: no Syncthing temp file, the same size and mtime as the PREVIOUS pass saw (so never
        on the first look), first seen in that state at least --settle seconds ago by THIS watcher's clock. The
        file's own mtime is not used for age: Syncthing gives a copied file the source's mtime, so an old
        recording would look settled the moment it landed."""
        if syncing(video.parent, video.name):
            self.seen.pop(video, None)
            return False
        try:
            now_state = _state(video)
        except OSError:
            self.seen.pop(video, None)
            return False
        before = self.seen.get(video)
        if before is None or before[0] != now_state:
            self.seen[video] = (now_state, self.clock())
            return False
        return self.clock() - before[1] >= self.settle

    def run_one(self, video: Path) -> bool:
        """Pose for one clip; True when a pose was published. A clip that vanishes or changes is left for a later
        pass; any other failure writes X.pose.failed. Nothing raised here stops the watcher."""
        npz, failed = outputs(self.ingest, video.stem)
        work = None
        try:
            try:
                before = _state(video)
            except FileNotFoundError:
                self.seen.pop(video, None)
                self.log(f"{video.name}: gone before pose ran")
                return False
            work = Path(tempfile.mkdtemp(prefix=f".{video.stem}.", dir=self.staging))
            argv = ["--video", str(video.resolve()), "--weights", str(self.weights.resolve()),
                    f"--weights-sha256={self.sha}", f"--device={self.device}", "--out-dir", str(work),
                    "--no-overlay"]
            try:
                rc = run_ingest.main(argv, predict=self.predict)
            except SystemExit as e:
                rc = e.code
            made = work / f"{video.stem}.pose.npz"
            if rc != 0 or not made.is_file():
                raise RuntimeError(f"run_ingest exited with {rc!r} and wrote no pose")
            try:
                after = _state(video)
            except FileNotFoundError:
                after = None
            if after != before:
                self.seen.pop(video, None)
                self.log(f"{video.name}: changed while pose ran; will retry once it settles")
                return False
            analyse_clip.check_pose(PoseSequence.load(made), probe(video), video, self.weights.name)
            os.replace(made, npz)
            self.log(f"{video.name}: pose -> {npz}")
            return True
        except Exception as e:  # one bad clip must not stop the watcher
            reason = f"{type(e).__name__}: {e}".replace("\n", " ")[:MAX_REASON]
            try:
                write_atomic(failed, reason + "\n", self.staging)
                self.log(f"{video.name}: pose FAILED ({reason}); delete {failed.name} to retry")
            except Exception as e2:
                n = self.unrecorded[video] = self.unrecorded.get(video, 0) + 1
                then = (f"left alone until the watcher restarts ({n} failures)" if n >= MAX_UNRECORDED
                        else "will try again next pass")
                self.log(f"{video.name}: pose FAILED ({reason}), and {failed.name} could not be written "
                         f"({type(e2).__name__}: {e2}); {then}")
            return False
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)

    def poll(self) -> int:
        """One pass; returns how many poses were published."""
        done = 0
        current = candidates(self.src)
        for gone in set(self.seen) - set(current):
            del self.seen[gone]
        self.waiting_logged &= set(current)
        self.unrecorded = {v: n for v, n in self.unrecorded.items() if v in current}
        for video in current:
            npz, failed = outputs(self.ingest, video.stem)
            if npz.exists() or failed.exists() or self.unrecorded.get(video, 0) >= MAX_UNRECORDED:
                self.seen.pop(video, None)
                continue
            if not self.ready(video):
                if video not in self.waiting_logged:
                    self.log(f"{video.name}: waiting for it to finish arriving")
                    self.waiting_logged.add(video)
                continue
            self.waiting_logged.discard(video)
            self.log(f"{video.name}: running pose on {self.device}")
            try:
                done += self.run_one(video)
            except Exception as e:   # run_one catches its own; this is the last line of defence
                self.log(f"{video.name}: unexpected {type(e).__name__}: {e}; will try again next pass")
        return done


class SingleInstance:
    """An OS lock on a file in the staging folder: released when the process ends, however it ends, so a
    crash leaves no stale lock. A second watcher fails to take it and exits."""

    def __init__(self, path: Path):
        self.path, self.f = path, None

    def __enter__(self):
        self.f = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise RuntimeError("another pose watcher is already running") from None
        return self

    def __exit__(self, *exc):
        self.f.close()
        return False


def main(argv=None, predict=None, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--ingest-dir", type=Path, default=DEFAULT_INGEST)
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--weights", type=Path, default=analyse_clip.DEFAULT_WEIGHTS)
    ap.add_argument("--weights-sha256", type=analyse_clip._sha256_arg, default=analyse_clip.PINNED_WEIGHTS_SHA256)
    ap.add_argument("--device", type=analyse_clip._device, default="cuda:0")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between passes")
    ap.add_argument("--settle", type=float, default=60.0, help="seconds a clip must be untouched")
    ap.add_argument("--once", action="store_true", help="two looks max(--interval, --settle) apart (a clip needs both), then exit")
    ap.add_argument("--log", type=Path, help="also append every line to this file")
    a = ap.parse_args(argv)
    if not (1 <= a.interval <= 3600) or not (0 <= a.settle <= 3600):
        ap.error("--interval must be 1-3600 s and --settle 0-3600 s")
    try:
        check_folders(a.src, a.ingest_dir, a.staging)
    except ValueError as e:
        ap.error(str(e))
    if not a.src.is_dir():
        ap.error(f"no folder {a.src}")
    if predict is None and not a.weights.is_file():
        ap.error(f"no pose-model weights at {a.weights}")
    a.ingest_dir.mkdir(parents=True, exist_ok=True)
    a.staging.mkdir(parents=True, exist_ok=True)
    if not same_device(a.staging, a.ingest_dir):
        ap.error(f"--staging {a.staging} and --ingest-dir {a.ingest_dir} are on different volumes: the pose could "
                 "not be moved into place in one rename")
    w = Watcher(a.src, a.ingest_dir, a.staging, a.weights, a.weights_sha256, a.device, a.settle, predict, a.log)
    try:
        with SingleInstance(a.staging / "watcher.lock"):
            w.log(f"watching {a.src.resolve()} -> {a.ingest_dir.resolve()} every {a.interval:g} s "
                  f"(settle {a.settle:g} s, {a.device})")
            looks = 0
            while True:
                w.poll()
                looks += 1
                if a.once and looks == 2:   # a clip needs two looks to be ready
                    return 0
                sleep(max(a.interval, a.settle) if a.once else a.interval)
    except RuntimeError as e:
        log(f"pose_watcher: {e}", a.log)
        return 1
    except KeyboardInterrupt:
        w.log("stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
