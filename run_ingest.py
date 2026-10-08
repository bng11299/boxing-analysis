"""Video -> pose (.npz) -> overlay (.mp4), with YOLO11-pose.

  python run_ingest.py --hash-weights weights/yolo11m-pose.pt
  python run_ingest.py --video clip.mp4 --weights weights/yolo11m-pose.pt
         --weights-sha256 <hash> --device cuda:0 --out-dir results/ingest

Outputs land in --out-dir (default results/ingest, gitignored). They contain
your footage or a skeleton of it: never commit them, and never put participant
footage in the public demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import overlay, pose_yolo  # noqa: E402
from boxingvi.video import fingerprint, probe  # noqa: E402

DEFAULT_OUT = Path("results/ingest")


def main(argv=None, predict=None, track=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--weights-sha256")
    ap.add_argument("--hash-weights", type=Path, metavar="FILE",
                    help="print a checkpoint's sha256 and exit")
    ap.add_argument("--device", default="cpu", help="cpu, cuda:0, ...")
    ap.add_argument("--conf", type=float, default=0.25, help="person detection threshold")
    ap.add_argument("--min-conf", type=float, default=0.3, help="joint confidence to draw")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--track", action="store_true",
                    help="two-boxer footage: keep EVERY person with a ByteTrack id (-> <clip>.tracks.npz) "
                         "instead of one boxer per frame")
    ap.add_argument("--track-conf", type=float, default=0.1,
                    help="detection floor with --track; ByteTrack re-uses low-score boxes (config/bytetrack_50fps.yaml)")
    a = ap.parse_args(argv)

    if a.hash_weights:
        print(pose_yolo.sha256_file(a.hash_weights))
        return 0
    if not a.video:
        ap.error("--video is required")
    injected = track if a.track else predict
    if injected is None:  # tests inject one; real runs need pinned weights
        if not a.weights:
            ap.error("--weights is required")
        model = pose_yolo.load_yolo(a.weights, a.weights_sha256, a.device)
        if a.track:
            track = pose_yolo.yolo_tracker(model, a.track_conf, a.device)
        else:
            predict = pose_yolo.yolo_predictor(model, a.conf, a.device)
    name = a.weights.name if a.weights else "injected"

    info = probe(a.video)
    print(f"{a.video.name}: {info.width}x{info.height} @ {info.fps:.3f} fps"
          + ("  [VARIABLE FRAME RATE - timing-based features are suspect]" if info.vfr else "")
          + ("  [BELOW 60 fps - see the camera-chain check]" if info.fps < 59 else ""))
    t = info.timing
    print("timing: UNMEASURABLE (no readable packet timestamps)" if t is None else
          f"timing: {t.n_frames} frames, median interval {t.median_dt * 1000:.2f} ms, "
          f"{t.n_irregular} irregular, intervals {t.min_gap:.2f}-{t.max_gap:.2f}x the median, "
          f"drift {t.max_drift:.2f} intervals")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.track:
        table = pose_yolo.extract_tracks(a.video, track, name, a.max_frames, info)
        table.video_fp = fingerprint(a.video)      # binds the table to THIS recording's content
        npz = a.out_dir / f"{a.video.stem}.tracks.npz"
        table.save(npz)
        ids = table.ids()
        counts = {i: int((table.track_id == i).sum()) for i in ids[:6]}
        people = np.bincount(table.frame, minlength=table.n_frames)
        print(f"{len(ids)} track ids over {table.n_frames} frames; most-seen (id: frames) {counts}; "
              f"people per frame: median {int(np.median(people))}, max {int(people.max(initial=0))}, "
              f"none in {int((people == 0).sum())} frames -> {npz}")
        if not a.no_overlay:
            mp4 = a.out_dir / f"{a.video.stem}.tracks.overlay.mp4"
            print(f"overlay: {overlay.render_tracks_overlay(a.video, table, mp4, a.min_conf, a.max_frames, info)} "
                  f"frames -> {mp4}")
        return 0
    pose = pose_yolo.extract(a.video, predict, name, a.max_frames, info)
    pose.video_fp = fingerprint(a.video)
    npz = a.out_dir / f"{a.video.stem}.pose.npz"
    pose.save(npz)
    n = len(pose)
    joints = pose.usable(a.min_conf, required=())
    good = pose.usable(a.min_conf)
    print(f"person found in {int(pose.detected.sum())}/{n} frames ({pose.detected.mean():.1%}); "
          f">= {pose_yolo.USABLE_MIN_JOINTS} joints at conf >= {a.min_conf}: "
          f"{int(joints.sum())} ({joints.mean():.1%}); USABLE (and both wrists and "
          f"shoulders): {int(good.sum())} ({good.mean():.1%}) -> {npz}")
    if not a.no_overlay:
        mp4 = a.out_dir / f"{a.video.stem}.overlay.mp4"
        print(f"overlay: {overlay.render_overlay(a.video, pose, mp4, a.min_conf, a.max_frames, info)} "
              f"frames -> {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
