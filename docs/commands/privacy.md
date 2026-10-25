# Commands: Privacy controls for participant footage

Moved verbatim from the project rules' `## Commands` on 2026-10-22. Add new recipes here, not to the project rules.

Face blur for the second annotator (a privacy control, fails closed): `python blur_faces.py --video <clip> --out ~/comps-participant-data/blurred/P01-R1.mp4 --weights weights/yolo11m-pose.pt --weights-sha256 <hash> [--expect-persons 2]` (face detector: deface's CenterFace, `pip install --no-deps deface==1.5.0` into the pose venv, model pinned by sha256; it narrows the blur only where it agrees with the keypoints; output only under `~/comps-participant-data/`; never in a checkout, the vault, a Syncthing or cloud folder; name outputs by code, never by person; read the `.blur.json` report before handing a clip on) · Blur tests: `python tests/test_blur.py`
