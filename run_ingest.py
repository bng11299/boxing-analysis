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

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import overlay, pose_yolo  # noqa: E402
from boxingvi.video import probe  # noqa: E402

DEFAULT_OUT = Path("results/ingest")


def main(argv=None, predict=None) -> int:
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
    a = ap.parse_args(argv)

    if a.hash_weights:
        print(pose_yolo.sha256_file(a.hash_weights))
        return 0
    if not a.video:
        ap.error("--video is required")
    if predict is None:  # tests inject one; real runs need pinned weights
        if not a.weights:
            ap.error("--weights is required")
        model = pose_yolo.load_yolo(a.weights, a.weights_sha256, a.device)
        predict = pose_yolo.yolo_predictor(model, a.conf, a.device)
    name = a.weights.name if a.weights else "injected"

    info = probe(a.video)
    print(f"{a.video.name}: {info.width}x{info.height} @ {info.fps:.3f} fps"
          + ("  [VARIABLE FRAME RATE - timing-based features are suspect]" if info.vfr else "")
          + ("  [BELOW 60 fps - see the camera-chain check]" if info.fps < 59 else ""))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pose = pose_yolo.extract(a.video, predict, name, a.max_frames)
    npz = a.out_dir / f"{a.video.stem}.pose.npz"
    pose.save(npz)
    rate = pose.detected.mean()
    good = pose.usable(a.min_conf)
    print(f"person found in {int(pose.detected.sum())}/{len(pose)} frames ({rate:.1%}); "
          f"USABLE (>= {pose_yolo.USABLE_MIN_JOINTS} joints at conf >= {a.min_conf}): "
          f"{int(good.sum())} ({good.mean():.1%}) -> {npz}")
    if not a.no_overlay:
        mp4 = a.out_dir / f"{a.video.stem}.overlay.mp4"
        print(f"overlay: {overlay.render_overlay(a.video, pose, mp4, a.min_conf, a.max_frames)} "
              f"frames -> {mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
