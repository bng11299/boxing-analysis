"""Video -> face-blurred H.264 copy (no audio, no metadata) + a JSON report.

  python blur_faces.py --video <clip> --out ~/comps-participant-data/blurred/P01-R1.mp4

THIS IS A PRIVACY CONTROL (protocol FA26-22-NG, amendment O57: the second
annotator sees face-blurred clips only, and is never told names).

DEFAULT POLICY, deface (Brendan, 2026-10-22): deface's CenterFace face detector
alone decides; every face it finds is mosaicked, and a frame where it finds none
is left AS IT IS. That is an accepted residual, not a guarantee: on own
shadowboxing footage it found no face in 3.8% and 28.5% of frames with a person
in them, and about a third of those sampled showed a visible face (mid-punch,
profile, behind a glove) -- roughly 1% and 9% of person frames with an unblurred
face. The report counts every such frame (`frames_without_face`,
`no_face_ranges`): check those frames of every clip before it goes on. A
detector failure blurs the whole frame. --face-policy agree / trust (with
--weights weights/yolo11m-pose.pt --weights-sha256 <hash>) add YOLO11-pose so a
missed face still gets a body-based blur; see src/boxingvi/faceblur.py and
docs/quality-loop.md (iterations 8-9).

Output is allowed only under ~/comps-participant-data/ (which must be on an
encrypted disk, docs/MACHINE-SYNC.md), and the folder must already exist.
--allow-outside-participant-root lifts that one rule, for footage that is not a
participant's (your own); the report records it. Nothing lifts the rest: never
inside a git checkout (this repo, the vault), a Syncthing folder, a cloud-synced
folder or a network path. Name outputs by code (P01-R1), never by person: the
report names no file, and identifies the source by its sha256.

Read the report (<name>.blur.json) before handing a clip on. Under agree/trust:
`frames_fully_blurred` and `full_blur_reasons` say how often the detector was
unsure, `persons_per_frame` how many people it found; for sparring pass
--expect-persons 2. Those policies cannot catch a person never detected, or a box
that consistently leaves out the head with no confident face points. Feed the
camera's own file: black bars baked in (a re-export or a download) move the
picture edge inside the frame.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import faceblur, pose_yolo  # noqa: E402
from boxingvi.video import probe  # noqa: E402

MAX_DEFACE_THRESH = 0.5  # deface policy: a higher face threshold is refused
NO_FACE_WARN = 0.30      # deface policy: warn when more frames than this had no face found
DET_CONF = 0.10  # lower than ingest's 0.25: a false person only adds blur, a missed one leaks a face


def main(argv=None, predict=None, location_kwargs=None, detect_faces=None) -> int:
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
    ap.add_argument("--face-detector", choices=("centerface", "none"), default="centerface",
                    help="face boxes that, where they agree with the keypoints, replace the "
                         "box floor for that person (deface's CenterFace, pinned by sha256)")
    ap.add_argument("--face-policy", choices=faceblur.FACE_POLICIES, default="deface",
                    help="deface (default, Brendan 2026-10-22): only the face detector's boxes are blurred; a frame where it finds no face is left as it is -- an accepted residual (it missed visible faces mid-punch and in profile on own footage). agree: a face box replaces a person's floor only where it agrees with "
                         "the keypoints; trust: wherever it lands on the person (deface's look, "
                         "but a glove detected while the face is missed leaves that face bare)")
    ap.add_argument("--face-thresh", type=float, default=faceblur.FACE_DET_THRESH,
                    help="face detection threshold")
    ap.add_argument("--allow-no-faces", action="store_true",
                    help="deface policy: accept a clip in which no face is found at all "
                         "(otherwise refused: a detector that never fires is broken)")
    ap.add_argument("--allow-outside-participant-root", action="store_true",
                    help="NOT for participant footage: write outside ~/comps-participant-data/")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    if not 0 < a.det_conf < 1:
        ap.error("--det-conf must lie in (0, 1)")
    if not 0 < a.face_thresh < 1:
        ap.error("--face-thresh must lie in (0, 1)")
    if a.face_policy == "deface" and a.face_thresh > MAX_DEFACE_THRESH:
        ap.error(f"--face-thresh above {MAX_DEFACE_THRESH} under the deface policy: every face "
                 "it misses is left bare (the accepted miss rate was measured at 0.2)")
    if a.face_policy == "deface" and a.face_thresh > faceblur.FACE_DET_THRESH:
        print(f"WARNING: --face-thresh {a.face_thresh} is above the {faceblur.FACE_DET_THRESH} the "
              "accepted miss rate was measured at; more faces will be left bare", file=sys.stderr)
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

    if a.face_policy == "deface" and a.face_detector == "none":
        ap.error("--face-policy deface needs the face detector")
    if predict is None and a.face_policy != "deface":  # tests inject one; real runs pin weights
        if not a.weights:
            ap.error("--weights is required")
        # Ultralytics sends usage events to Google Analytics when online. No pixels go,
        # but a privacy step talks to no third party: offline mode, set before import.
        os.environ["YOLO_OFFLINE"] = "1"
        model = pose_yolo.load_yolo(a.weights, a.weights_sha256, a.device)
        predict = pose_yolo.yolo_predictor(model, a.det_conf, a.device, missing_conf=0.0)
    if a.face_detector == "centerface" and detect_faces is None:
        detect_faces = faceblur.load_centerface(a.face_thresh)
    name = (a.weights.name if a.weights else "injected") if a.face_policy != "deface" else None
    face_model = ("centerface (deface 1.5.0)" if a.face_detector == "centerface"
                  and detect_faces is not None else None)

    def recheck(path):  # the folder may have changed during a long first pass
        faceblur.check_output_location(path, input_path=a.video, overwrite=a.overwrite, **loc)

    info = probe(a.video)
    print(f"{info.width}x{info.height} @ {info.fps:.3f} fps"
          + ("  [variable frame rate]" if info.vfr else ""))
    try:
        r = faceblur.blur_video(a.video, out, predict, info=info, expect_persons=a.expect_persons,
                                model_name=name, progress=print, recheck=recheck,
                                report_extra={"outside_participant_root":
                                              a.allow_outside_participant_root},
                                detect_faces=detect_faces if face_model else None,
                                face_model=face_model or "", face_policy=a.face_policy,
                                face_thresh=a.face_thresh, allow_no_faces=a.allow_no_faces)
    except faceblur.UnsafeOutput as e:
        print(f"REFUSED at the end, nothing written: {e}", file=sys.stderr)
        return 2
    except faceblur.NoFacesFound as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    n, full = r["frames"], r["frames_fully_blurred"]
    counts = r.get("faces_per_frame", r.get("persons_per_frame"))
    print(f"{n} frames: {r['frames_by_mode']}; fully blurred {full} ({full / n:.1%}) "
          f"{r['full_blur_reasons']}; "
          f"{'faces' if a.face_policy == 'deface' else 'persons'} per frame {counts}")
    if a.face_policy == "deface":  # the accepted residual, stated on every run
        nf = r["frames_without_face"]
        print(f"NO FACE FOUND in {nf} of {n} frames ({nf / n:.1%}); those frames are NOT "
              f"blurred -- check them before handing the clip on: {r['no_face_ranges'][:20]}"
              f"{' ...' if len(r['no_face_ranges']) > 20 else ''}", file=sys.stderr)
        if nf / n > NO_FACE_WARN:
            print(f"WARNING: more than {NO_FACE_WARN:.0%} of frames had no face found (the "
                  "measured worst case on own footage was 28.5%)", file=sys.stderr)
    if full / n > 0.5:
        print("WARNING: most frames were blurred whole -- the detector was unsure; "
              "the clip is safe but may be useless for labelling", file=sys.stderr)
    print(f"-> {out}\n-> {faceblur.report_path_for(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
