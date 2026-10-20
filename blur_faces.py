"""Video -> face-blurred H.264 copy (no audio, no metadata) + a JSON report.

  python blur_faces.py --video <clip> --out ~/comps-participant-data/blurred/P01-R1.mp4
         --weights weights/yolo11m-pose.pt --weights-sha256 <hash> [--expect-persons 2]

THIS IS A PRIVACY CONTROL (protocol FA26-22-NG, amendment O57: the second
annotator sees face-blurred clips only, and is never told names). Every person in
every frame is blurred; a frame that cannot be trusted is blurred whole. See
src/boxingvi/faceblur.py.

Output is allowed only under ~/comps-participant-data/ (which must be on an
encrypted disk, docs/MACHINE-SYNC.md), and the folder must already exist.
--allow-outside-participant-root lifts that one rule, for footage that is not a
participant's (your own); the report records it. Nothing lifts the rest: never
inside a git checkout (this repo, the vault), a Syncthing folder, a cloud-synced
folder or a network path. Name outputs by code (P01-R1), never by person: the
report names no file, and identifies the source by its sha256.

Read the report (<name>.blur.json) before handing a clip on:
`frames_fully_blurred` and `full_blur_reasons` say how often the detector was
unsure, `persons_per_frame` how many people it found. For sparring footage pass
--expect-persons 2: a boxer the detector misses for longer than its neighbour
window is otherwise caught only by a human look.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import faceblur, pose_yolo  # noqa: E402
from boxingvi.video import probe  # noqa: E402

DET_CONF = 0.10  # lower than ingest's 0.25: a false person only adds blur, a missed one leaks a face


def main(argv=None, predict=None, location_kwargs=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .mp4")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--weights-sha256")
    ap.add_argument("--device", default="cpu", help="cpu, cuda:0, ...")
    ap.add_argument("--det-conf", type=float, default=DET_CONF, help="person detection threshold")
    ap.add_argument("--expect-persons", type=int, default=0,
                    help="frames with fewer people than this are blurred whole")
    ap.add_argument("--allow-outside-participant-root", action="store_true",
                    help="NOT for participant footage: write outside ~/comps-participant-data/")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not 0 < a.det_conf < 1:
        ap.error("--det-conf must lie in (0, 1)")
    if a.expect_persons < 0:
        ap.error("--expect-persons must be >= 0")

    loc = dict(location_kwargs or {})
    if a.allow_outside_participant_root:
        loc["allowed_root"] = None
    # Where the output goes is decided before a single frame is decoded.
    try:
        out = faceblur.check_output_location(a.out, input_path=a.video, overwrite=a.overwrite,
                                             **loc)
    except faceblur.UnsafeOutput as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    # The report sits in the same (checked) folder; only an existing one needs refusing.
    if faceblur.report_path_for(out).exists() and not a.overwrite:
        print(f"REFUSED: {faceblur.report_path_for(out).name} exists "
              "(pass --overwrite to replace it)", file=sys.stderr)
        return 2

    if predict is None:  # tests inject one; real runs need pinned weights
        if not a.weights:
            ap.error("--weights is required")
        # Ultralytics sends usage events to Google Analytics when online. No pixels go,
        # but a privacy step talks to no third party: offline mode, set before import.
        os.environ["YOLO_OFFLINE"] = "1"
        model = pose_yolo.load_yolo(a.weights, a.weights_sha256, a.device)
        predict = pose_yolo.yolo_predictor(model, a.det_conf, a.device, missing_conf=0.0)
    name = a.weights.name if a.weights else "injected"

    def recheck(path):  # the folder may have changed during a long first pass
        faceblur.check_output_location(path, input_path=a.video, overwrite=a.overwrite, **loc)

    info = probe(a.video)
    print(f"{info.width}x{info.height} @ {info.fps:.3f} fps"
          + ("  [variable frame rate]" if info.vfr else ""))
    try:
        r = faceblur.blur_video(a.video, out, predict, info=info, expect_persons=a.expect_persons,
                                model_name=name, progress=print, recheck=recheck,
                                report_extra={"outside_participant_root":
                                              a.allow_outside_participant_root})
    except faceblur.UnsafeOutput as e:
        print(f"REFUSED at the end, nothing written: {e}", file=sys.stderr)
        return 2
    n, full = r["frames"], r["frames_fully_blurred"]
    print(f"{n} frames: {r['frames_by_mode']}; fully blurred {full} ({full / n:.1%}) "
          f"{r['full_blur_reasons']}; persons per frame {r['persons_per_frame']}")
    if full / n > 0.5:
        print("WARNING: most frames were blurred whole -- the detector was unsure; "
              "the clip is safe but may be useless for labelling", file=sys.stderr)
    print(f"-> {out}\n-> {faceblur.report_path_for(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
