# `chore/deface-env` — 2026-10-22

Task: `scripts/setup-env.ps1` installs and verifies deface for the face blur (#21), on
both machines. Option (a): the `-NoPose` venv gets a pinned `opencv-python` too, so the
default `--face-policy deface` (no YOLO) runs on SLOTMACHINE.

Threat sketch:
- **Assets:** the face blur, a privacy control for participants' faces, and the venv's pinned builds.
- **Entry points:** PyPI wheels (deface, opencv-python), and deface's `centerface.onnx` parsed by OpenCV.
- **Trust boundaries:** PyPI -> venv, and model file -> cv2's ONNX importer.
- **Top threats:**
  - a swapped or tampered model silently changes which faces are blurred; the defence is sha256 before parse;
  - deface's declared dependencies (scikit-image, imageio, an unpinned opencv) replace pinned packages; the defence is `--no-deps` plus the constraints file.

### Iteration 1
- **Built:**
  - `requirements-blur.txt` (`deface==1.5.0`, `opencv-python==5.0.0.93`, the cv2 the blur was measured with), installed last with `--no-deps` under the constraints file, which now also pins that OpenCV for ultralytics;
  - `check_env.py` checks the blur pins on every machine, imports `deface.centerface`, and hashes `centerface.onnx` against `CENTERFACE_SHA256` (read from `faceblur.py` with `ast`); only on a match does it parse the model with the OpenCV backend and require no face on a black frame.
- **Test gate (fresh `-NoPose` venv on SLOTMACHINE):**
  - setup-env installed deface and opencv-python only (no scikit-image, imageio or tqdm) and printed `ENVIRONMENT OK`;
  - `test_env.py` 59/0, `test_machine_scripts.ps1` 48/0 (including end-to-end `-CheckOnly` on a throwaway bare venv: no deface -> exit 1; a fake deface with a wrong model -> exit 1, never parsed), `test_blur.py` 177/0 with no ultralytics.
- **Mutants:** 5/5 killed (parse before hash, blur pins unchecked, deface check dropped, `--no-deps` dropped, cv2 constraint dropped).
- **Review gate (code-reviewer subagent):** SHIP, no blocking findings. Non-blocking:
  - the check read package metadata, not the cv2 that loads; an `opencv-python-headless` beside it would share `cv2/`;
  - the face threshold was a hand copy;
  - the e2e test ignored the venv's exit code and restored `$ErrorActionPreference` without `finally`;
  - the bit-flip test went through a fake CenterFace;
  - the project-rules blur line showed `--weights` as required and described the `agree` policy as the default;
  - MACHINE-SYNC should say to re-run the ingest tests after a laptop rebuild moves cv2;
  - dropping the install line in `Invoke-Main` survives every unit test (only the final check catches it).
- **Security gate:** no blocking findings.
  - pip still gets an argument array;
  - the pin is read with `ast`, nothing executed;
  - hash before parse, no parse after a mismatch;
  - errors name hashes and exception types only.
  - Residuals (low): deface's code and the OpenCV wheel are pinned by version, not wheel hash (as the rest of the repo); and there is a gap between hash and parse, which needs write access to the venv.

### Iteration 2
- **Changed:**
  - `check_opencv`: the loaded `cv2.__version__` must be the pin's `5.0.0`, and no `opencv-python-headless` or contrib build may be installed;
  - the face threshold is read from `faceblur.FACE_DET_THRESH` (generalised `read_constant`);
  - the e2e test checks the venv was created and restores `$ErrorActionPreference` in `finally`, with the expected hash read from `faceblur.py`;
  - the bit-flip test now runs the real `CenterFace`, with an untouched-model control;
  - project rules, MACHINE-SYNC.md and a comment in `Invoke-Main` updated.
- **Test gate:** `test_env.py` 69/0, `test_machine_scripts.ps1` 50/0, `test_blur.py` 177/0, and `setup-env.ps1 -CheckOnly -NoPose` `ENVIRONMENT OK` (exit 0).
- **Review and security:** the iteration-2 diff was re-read; no new findings.
- **Not covered:** the CUDA/pose flavour was not built (SLOTMACHINE has no GPU); run setup-env and `-CheckOnly` on the laptop. Every existing venv fails `-CheckOnly` until setup-env is re-run there.
