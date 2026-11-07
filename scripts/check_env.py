"""Verify this machine's .venv can actually run the pipeline -- by importing, not by
trusting that an install exited 0 (ISSUES-LOG 2026-08-30: pip succeeded and torch
could not load).

  python scripts/check_env.py --torch cpu
  python scripts/check_env.py --torch cuda --pose

Every pinned requirement must be installed at its pinned version. Torch is checked
on its version AND its build (`+cpu` or `+cu130`): requirements.txt pins the CPU
build, and a CUDA machine must not be "satisfied" by it. On CUDA, a tensor is
actually computed on the GPU, because `is_available()` can be True while kernels
fail (a driver too old for the build). Exit 1 on any failure.

The face blur is checked on every machine, pose step or not: deface's CenterFace
must import (it needs cv2), its centerface.onnx must hash to the CENTERFACE_SHA256
that src/boxingvi/faceblur.py pins, and only then is the model parsed by OpenCV and
run on a black frame, which must find no face.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata as md
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = (3, 13)
BUILD_TAGGED = ("torch", "torchvision")  # must carry this machine's build tag (+cpu / +cu130)
FACEBLUR_SRC = ROOT / "src" / "boxingvi" / "faceblur.py"
# These share the cv2/ folder with opencv-python: whichever was installed last is the
# cv2 that loads, and uninstalling one breaks the other.
OTHER_OPENCV = ("opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless")


def parse_pins(path: Path) -> "dict[str, str]":
    """name -> pinned version, from `name==version` lines; comments and blanks skipped.
    Anything else is refused: an unpinned or ranged line would make the check vacuous."""
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, ver = line.partition("==")
        if not sep or not name.strip() or not ver.strip():
            raise ValueError(f"{path.name}: not an exact pin: {raw!r}")
        pins[name.strip().lower()] = ver.strip()
    return pins


def check_pin(name: str, pinned: str, installed: "str | None", torch_build: str) -> "str | None":
    """None if fine, else the reason. For torch and torchvision the build tag must
    be `torch_build` (cpu / cu130 ...), whatever tag the requirements file carries:
    a PyPI torchvision on a CUDA torch has no CUDA kernels (nms fails on the GPU)."""
    if installed is None:
        return f"{name} is not installed (pinned {pinned})"
    if name in BUILD_TAGGED:
        base, _, tag = installed.partition("+")
        want = pinned.partition("+")[0]
        if base != want:
            return f"{name} {installed}: version should be {want}"
        if tag != torch_build:
            return f"{name} {installed}: build should be +{torch_build}"
        return None
    if installed != pinned:
        return f"{name} {installed}: pinned {pinned}"
    return None


def installed_version(name: str) -> "str | None":
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def read_constant(name: str, path: Path = FACEBLUR_SRC):
    """A module-level constant of faceblur.py, read from its source rather than
    imported (one place holds the value, and nothing of the repo runs). ValueError
    unless the module assigns it exactly once, to a literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            values.append(node.value)
    if len(values) != 1:
        raise ValueError(f"{path.name}: expected one {name} assignment, found {len(values)}")
    if not isinstance(values[0], ast.Constant):
        raise ValueError(f"{path.name}: {name} is not a literal")
    return values[0].value


def pinned_centerface_sha256(path: Path = FACEBLUR_SRC) -> str:
    """faceblur.CENTERFACE_SHA256: a 64-hex-digit string, else ValueError."""
    v = read_constant("CENTERFACE_SHA256", path)
    if not (isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)):
        raise ValueError(f"{path.name}: CENTERFACE_SHA256 is not a 64-hex-digit string literal")
    return v


def face_threshold(path: Path = FACEBLUR_SRC) -> float:
    """faceblur.FACE_DET_THRESH, the threshold blur_faces.py runs CenterFace at."""
    v = read_constant("FACE_DET_THRESH", path)
    if not (isinstance(v, float) and 0 < v < 1):
        raise ValueError(f"{path.name}: FACE_DET_THRESH is not a float in (0, 1)")
    return v


def check_opencv(imp, version, pinned: str) -> "list[str]":
    """The cv2 that actually loads is the pinned opencv-python: the dist metadata
    alone does not say which package's files are in cv2/."""
    fails = [f"{other} {version(other)} is installed alongside opencv-python (they share cv2/); "
             f"uninstall it and re-run setup-env.ps1" for other in OTHER_OPENCV if version(other) is not None]
    try:
        cv2 = imp("cv2")
    except Exception as e:
        return fails + [f"import cv2 failed: {type(e).__name__}: {e}"]
    want = ".".join(pinned.split(".")[:3])  # opencv-python 5.0.0.93 is cv2 5.0.0
    got = getattr(cv2, "__version__", None)
    if got != want:
        fails.append(f"cv2 {got} is the one that loads, but opencv-python is pinned {pinned} (cv2 {want})")
    return fails


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_deface(imp, expected_sha256: str, threshold: float) -> "str | None":
    """The face detector exactly as faceblur.load_centerface loads it. The model is
    hashed before OpenCV parses it, and not parsed at all on a mismatch."""
    try:
        cf = imp("deface.centerface")
    except Exception as e:  # missing deface, or deface without cv2 (-NoPose)
        return f"import deface.centerface failed: {type(e).__name__}: {e} (the face blur cannot run)"
    onnx = getattr(cf, "default_onnx_path", None)
    if not isinstance(onnx, str):
        return "deface.centerface has no default_onnx_path (not deface 1.5.0?)"
    try:
        got = sha256_file(Path(onnx))
    except OSError as e:
        return f"deface's centerface.onnx cannot be read: {type(e).__name__}: {e}"
    if got != expected_sha256:
        return (f"deface's centerface.onnx has sha256 {got}, not the pinned {expected_sha256} "
                "(faceblur.CENTERFACE_SHA256); not loading it")
    try:
        import numpy as np
        dets, _ = cf.CenterFace(onnx_path=onnx, backend="opencv")(
            np.zeros((64, 64, 3), np.uint8), threshold=threshold)
    except Exception as e:  # e.g. an OpenCV whose ONNX importer cannot read the model
        return f"CenterFace could not run the pinned model: {type(e).__name__}: {e}"
    shape = tuple(getattr(dets, "shape", ()))
    if shape != (0, 5):
        return f"CenterFace on a black frame gave {shape}, expected no faces, shape (0, 5)"
    return None


def check_cuda(torch) -> "str | None":
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False"
    try:
        got = (torch.ones(4, device="cuda") * 2).sum().item()
    except Exception as e:  # a driver/build mismatch surfaces here, not at import
        return f"a CUDA tensor op failed: {type(e).__name__}: {e}"
    return None if got == 8 else f"CUDA computed {got}, expected 8"


def check_nms(torchvision, torch) -> "str | None":
    """YOLO's last step. Boxes 0 and 1 overlap at IoU 0.68, so 1 is suppressed."""
    try:
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]],
                             device="cuda")
        scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
        keep = torchvision.ops.nms(boxes, scores, 0.5).tolist()
    except Exception as e:  # a CPU-only torchvision raises here, not at import
        return f"torchvision.ops.nms on CUDA failed: {type(e).__name__}: {e}"
    return None if keep == [0, 2] else f"nms kept {keep}, expected [0, 2]"


def run(torch_build: str, pose: bool, version=installed_version, imp=importlib.import_module,
        which=shutil.which, py=sys.version_info[:2],
        centerface_sha=pinned_centerface_sha256, face_thresh=face_threshold) -> "list[str]":
    """Every failure, as a sentence. Empty means the environment is good."""
    fails = []
    if tuple(py) != PYTHON:
        fails.append(f"Python {py[0]}.{py[1]}: expected {PYTHON[0]}.{PYTHON[1]}")
    files = ([ROOT / "requirements.txt", ROOT / "requirements-blur.txt"]
             + ([ROOT / "requirements-pose.txt"] if pose else []))
    for f in files:
        for name, pinned in parse_pins(f).items():
            why = check_pin(name, pinned, version(name), torch_build)
            if why:
                fails.append(why)
    pose_mods = ["torchvision", "ultralytics"] if pose else []
    for mod in ["numpy", "sklearn", "scipy", "pandas", "torch"] + pose_mods:
        try:
            imp(mod)
        except Exception as e:  # e.g. WinError 126: installed but cannot load
            fails.append(f"import {mod} failed: {type(e).__name__}: {e}")
    if not any(f.startswith("import torch failed") for f in fails):  # not "import torchvision"
        torch = imp("torch")
        if torch_build == "cpu":
            if torch.cuda.is_available():
                fails.append("CPU build expected but CUDA is available -- wrong torch?")
        else:
            why = check_cuda(torch)
            if why:
                fails.append(why)
            elif pose and not any(f.startswith("import torchvision failed") for f in fails):
                why = check_nms(imp("torchvision"), torch)
                if why:
                    fails.append(why)
    fails += check_opencv(imp, version, parse_pins(ROOT / "requirements-blur.txt")["opencv-python"])
    try:
        expected, thresh = centerface_sha(), face_thresh()
    except (OSError, ValueError, SyntaxError) as e:
        fails.append(f"the face blur's pinned constants cannot be read: {type(e).__name__}: {e}")
    else:
        why = check_deface(imp, expected, thresh)
        if why:
            fails.append(why)
    for tool in ("ffmpeg", "ffprobe"):
        exe = which(tool)
        if exe is None:
            fails.append(f"{tool} is not on PATH (video ingestion needs it)")
        elif subprocess.run([exe, "-version"], capture_output=True, timeout=30).returncode != 0:
            fails.append(f"{tool} at {exe} does not run")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torch", required=True, help="expected build: cpu, cu130, ...")
    ap.add_argument("--pose", action="store_true", help="also require requirements-pose.txt")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"cpu|cu\d{3}", a.torch):
        ap.error(f"--torch must be cpu or cuNNN (e.g. cu130), got {a.torch!r}")
    fails = run(a.torch, a.pose)
    torch_v = installed_version("torch") or "-"
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print(f"torch {torch_v} (expected +{a.torch}); pose stage: {'yes' if a.pose else 'not checked'}")
    print(f"face blur: deface {installed_version('deface') or '-'}, "
          f"opencv-python {installed_version('opencv-python') or '-'} "
          "(centerface.onnx must hash to faceblur.CENTERFACE_SHA256)")
    for f in fails:
        print(f"  FAIL  {f}")
    print("ENVIRONMENT OK" if not fails else f"ENVIRONMENT BROKEN: {len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
