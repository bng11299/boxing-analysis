"""Tests for scripts/check_env.py -- the check that a machine's .venv really works.

Run:  python tests/test_env.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import check_env  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def raises(name: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc:
        check(name, True)
    except Exception as e:
        check(name, False, f"raised {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


tmp = Path(tempfile.mkdtemp())

print("parse_pins")
pins = check_env.parse_pins(ROOT / "requirements.txt")
check("the real requirements.txt parses, torch pinned to the CPU build",
      pins.get("torch") == "2.13.0+cpu" and "scikit-learn" in pins, str(pins))
f = tmp / "r.txt"
f.write_text("# comment\n\nNumPy==2.2.6  # trailing\n", encoding="utf-8")
check("comments and blanks skipped, names lower-cased", check_env.parse_pins(f) == {"numpy": "2.2.6"})
for bad in ("numpy>=2.0", "numpy", "==1.0", "numpy=="):
    f.write_text(bad + "\n", encoding="utf-8")
    raises(f"a line that is not an exact pin is refused: {bad!r}", lambda: check_env.parse_pins(f))

print("check_pin")
check("torch: right version and right build", check_env.check_pin("torch", "2.13.0+cpu", "2.13.0+cu130", "cu130") is None)
check("torch: the CPU build on a CUDA machine fails",
      "build" in (check_env.check_pin("torch", "2.13.0+cpu", "2.13.0+cpu", "cu130") or ""))
check("torch: the CUDA build where CPU is expected fails",
      "build" in (check_env.check_pin("torch", "2.13.0+cpu", "2.13.0+cu130", "cpu") or ""))
check("torch: wrong version fails even with the right build",
      "version" in (check_env.check_pin("torch", "2.13.0+cpu", "2.12.0+cpu", "cpu") or ""))
check("torch: an untagged build fails", check_env.check_pin("torch", "2.13.0+cpu", "2.13.0", "cpu") is not None)
check("a missing package fails", "not installed" in (check_env.check_pin("numpy", "2.2.6", None, "cpu") or ""))
check("another package off its pin fails", check_env.check_pin("numpy", "2.2.6", "2.3.0", "cpu") is not None)
check("another package on its pin passes", check_env.check_pin("numpy", "2.2.6", "2.2.6", "cpu") is None)


class _T:  # a stand-in for a tensor: (ones * 2).sum().item()
    def __init__(self, v):
        self.v = v

    def __mul__(self, k):
        return _T(self.v * k)

    def sum(self):
        return _T(self.v)

    def item(self):
        return self.v


def fake_torch(available: bool, op_raises: bool = False, value: int = 4):
    class Cuda:
        @staticmethod
        def is_available():
            return available

    class Torch:
        cuda = Cuda

        @staticmethod
        def ones(n, device=None):
            if op_raises:
                raise RuntimeError("no kernel image is available for execution on the device")
            return _T(value)

        @staticmethod
        def tensor(data, device=None):
            return data

    return Torch


class _L:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return self.v


def fake_tv(keep=(0, 2), raises=False):
    class Ops:
        @staticmethod
        def nms(boxes, scores, iou):
            if raises:  # what a CPU-only torchvision does with CUDA tensors
                raise NotImplementedError("Could not run 'torchvision::nms' with arguments from the 'CUDA' backend")
            return _L(list(keep))

    class TV:
        ops = Ops

    return TV


# A stand-in for deface.centerface: a model file of known bytes, and a CenterFace
# that records every model it is asked to parse.
MODEL = tmp / "centerface.onnx"
MODEL.write_bytes(b"stand-in for centerface.onnx")
MODEL_SHA = hashlib.sha256(MODEL.read_bytes()).hexdigest()


class _Dets:
    def __init__(self, shape):
        self.shape = shape


def fake_deface(onnx=MODEL, shape=(0, 5), load_raises=None, parsed=None):
    class CenterFace:
        def __init__(self, onnx_path=None, backend="auto"):
            if parsed is not None:
                parsed.append((onnx_path, backend))
            if load_raises:
                raise load_raises

        def __call__(self, img, threshold=0.5):
            return _Dets(shape), None

    class Mod:
        default_onnx_path = str(onnx)

    Mod.CenterFace = CenterFace
    return Mod


class _Cv2:
    __version__ = "5.0.0"


def env(build="cpu", torch=None, missing=(), broken=(), which_missing=(), py=(3, 13), pose=False,
        tv=None, tv_version=None, deface=None, centerface_sha=lambda: MODEL_SHA, cv2=_Cv2, extra=None):
    every = dict(check_env.parse_pins(ROOT / "requirements.txt"))
    every.update(check_env.parse_pins(ROOT / "requirements-blur.txt"))
    every.update(check_env.parse_pins(ROOT / "requirements-pose.txt"))
    every["torch"] = "2.13.0+" + build
    every["torchvision"] = tv_version or "0.28.0+" + build
    every.update(extra or {})

    def version(name):
        return None if name in missing else every.get(name)

    def imp(mod):
        if mod in broken:
            raise OSError("[WinError 126] The specified module could not be found")
        if mod == "torchvision":
            return tv or fake_tv()
        if mod == "deface.centerface":
            return deface or fake_deface()
        if mod == "cv2":
            return cv2
        return torch if mod == "torch" else object()

    def which(tool):
        return None if tool in which_missing else check_env.shutil.which(tool)

    return check_env.run(build, pose, version=version, imp=imp, which=which, py=py,
                         centerface_sha=centerface_sha)


print("run (injected environment)")
ok = env("cpu", fake_torch(False))
check("control: a complete CPU environment has no failures", ok == [], str(ok))
check("control: a complete CUDA environment has no failures", env("cu130", fake_torch(True)) == [])
check("CUDA expected, not available -> fail",
      any("is_available" in m for m in env("cu130", fake_torch(False))))
check("CUDA 'available' but the tensor op fails -> fail (driver too old for the build)",
      any("tensor op failed" in m for m in env("cu130", fake_torch(True, op_raises=True))))
check("CUDA computes the wrong answer -> fail",
      any("expected 8" in m for m in env("cu130", fake_torch(True, value=3))))
check("CPU expected but CUDA available -> fail (wrong torch)",
      any("CPU build expected" in m for m in env("cpu", fake_torch(True))))
r = env("cpu", fake_torch(False), broken=("torch",))
check("torch installed but cannot load -> one import failure, no crash",
      any(m.startswith("import torch failed") for m in r) and len(r) == 1, str(r))
check("a missing pinned package -> fail", any("scipy is not installed" in m
                                              for m in env("cpu", fake_torch(False), missing=("scipy",))))
check("ffprobe missing from PATH -> fail",
      any("ffprobe is not on PATH" in m for m in env("cpu", fake_torch(False), which_missing=("ffprobe",))))
check("wrong Python -> fail", any("Python 3.12" in m for m in env("cpu", fake_torch(False), py=(3, 12))))
check("pose required and ultralytics missing -> fail",
      any("ultralytics is not installed" in m
          for m in env("cpu", fake_torch(False), missing=("ultralytics",), pose=True)))
check("... but not checked when the pose step is not required",
      env("cpu", fake_torch(False), missing=("ultralytics",)) == [])

print("torchvision (the pose step on CUDA)")
okp = env("cu130", fake_torch(True), pose=True)
check("control: CUDA torch + CUDA torchvision with a working nms has no failures", okp == [], str(okp))
check("PyPI torchvision (no build tag) on a CUDA machine -> fail",
      any("torchvision 0.28.0: build should be +cu130" in m
          for m in env("cu130", fake_torch(True), pose=True, tv_version="0.28.0")))
check("torchvision 0.29.1 (what ultralytics pulls from PyPI) -> fail on version",
      any("version should be 0.28.0" in m
          for m in env("cu130", fake_torch(True), pose=True, tv_version="0.29.1+cu130")))
check("nms has no CUDA kernel -> fail",
      any("nms on CUDA failed" in m for m in env("cu130", fake_torch(True), pose=True, tv=fake_tv(raises=True))))
check("nms keeps the wrong boxes -> fail",
      any("expected [0, 2]" in m for m in env("cu130", fake_torch(True), pose=True, tv=fake_tv(keep=(0, 1, 2)))))
r = env("cu130", fake_torch(True), pose=True, broken=("torchvision",))
check("torchvision cannot load -> one import failure, nms not attempted",
      len(r) == 1 and r[0].startswith("import torchvision failed"), str(r))
r = env("cu130", fake_torch(False), pose=True, broken=("torchvision",))
check("... and a broken torchvision does not hide a CUDA failure in torch",
      any("is_available" in m for m in r), str(r))
check("nms is not run on a CPU machine", env("cpu", fake_torch(False), pose=True, tv=fake_tv(raises=True)) == [])
check("nms is not run without the pose step", env("cu130", fake_torch(True), tv=fake_tv(raises=True)) == [])

print("face blur (deface), on every machine")
okb = env("cpu", fake_torch(False))
check("control: the face blur is checked without the pose step, and passes", okb == [], str(okb))
r = env("cpu", fake_torch(False), missing=("deface",), broken=("deface.centerface",))
check("deface missing -> fail, on the pin and on the import",
      any("deface is not installed (pinned 1.5.0)" in m for m in r)
      and any(m.startswith("import deface.centerface failed") for m in r), str(r))
r = env("cpu", fake_torch(False), missing=("opencv-python",))
check("opencv-python missing (the -NoPose case) -> fail, even with no pose step",
      any("opencv-python is not installed (pinned 5.0.0.93)" in m for m in r), str(r))
blur_pins = check_env.parse_pins(ROOT / "requirements-blur.txt")
check("the blur pins are exactly deface 1.5.0 and opencv-python 5.0.0.93 (installed with --no-deps)",
      blur_pins == {"deface": "1.5.0", "opencv-python": "5.0.0.93"}, str(blur_pins))
r = env("cpu", fake_torch(False), extra={"opencv-python-headless": "4.12.0.88"})
check("opencv-python-headless installed alongside (it shares cv2/) -> fail",
      any("opencv-python-headless 4.12.0.88 is installed alongside" in m for m in r), str(r))


class _Cv2Old:
    __version__ = "4.12.0"


r = env("cpu", fake_torch(False), cv2=_Cv2Old)
check("the cv2 that loads is not the pinned one, whatever the metadata says -> fail",
      any("cv2 4.12.0 is the one that loads" in m for m in r), str(r))
r = env("cpu", fake_torch(False), broken=("cv2",))
check("cv2 cannot load -> fail", any(m.startswith("import cv2 failed") for m in r), str(r))
check("a deface off its pin -> fail",
      check_env.check_pin("deface", "1.5.0", "1.6.0", "cpu") is not None)
parsed = []
r = env("cpu", fake_torch(False), deface=fake_deface(parsed=parsed), centerface_sha=lambda: "0" * 64)
check("wrong model hash -> fail, naming both hashes", any(
    "not the pinned " + "0" * 64 in m and MODEL_SHA in m for m in r), str(r))
check("... and the model is never handed to OpenCV (hash before parse)", parsed == [], str(parsed))
parsed = []
r = env("cpu", fake_torch(False), deface=fake_deface(parsed=parsed))
check("control: the right hash -> the model is parsed once, with the OpenCV backend, from the hashed path",
      r == [] and parsed == [(str(MODEL), "opencv")], f"{r} {parsed}")
r = env("cpu", fake_torch(False), deface=fake_deface(onnx=tmp / "no-such.onnx"))
check("model file missing -> fail", any("cannot be read" in m for m in r), str(r))


class _NoPath:
    pass


r = env("cpu", fake_torch(False), deface=_NoPath)
check("a deface.centerface with no default_onnx_path -> fail, no crash",
      any("no default_onnx_path" in m for m in r), str(r))
r = env("cpu", fake_torch(False), deface=fake_deface(load_raises=RuntimeError("cv2.dnn: cannot parse")))
check("OpenCV cannot parse the model -> fail", any("could not run the pinned model" in m for m in r), str(r))
r = env("cpu", fake_torch(False), deface=fake_deface(shape=(3, 5)))
check("faces found on a black frame -> fail", any("black frame gave (3, 5)" in m for m in r), str(r))
seen = []


def _thresh_spy():
    seen.append("read")
    return 0.37


class _SpyCF:
    def __init__(self, onnx_path=None, backend="auto"):
        pass

    def __call__(self, img, threshold=0.5):
        seen.append(threshold)
        return _Dets((0, 5)), None


spy = fake_deface()
spy.CenterFace = _SpyCF
r = check_env.run("cpu", False, version=lambda n: None, imp=lambda m: {"deface.centerface": spy, "cv2": _Cv2}[m],
                  which=lambda t: None, centerface_sha=lambda: MODEL_SHA, face_thresh=_thresh_spy)
check("CenterFace runs at the threshold read from faceblur.py, not a copy", seen == ["read", 0.37], str(seen))


def _sha_fails():
    raise ValueError("faceblur.py: expected one CENTERFACE_SHA256 assignment, found 0")


parsed = []
r = env("cpu", fake_torch(False), deface=fake_deface(parsed=parsed), centerface_sha=_sha_fails)
check("the pin cannot be read -> fail, and the model is not loaded against nothing",
      len(r) == 1 and "cannot be read" in r[0] and parsed == [], f"{r} {parsed}")

print("pinned_centerface_sha256")
check("the real faceblur.py face threshold is read", check_env.face_threshold() == 0.2)
src0 = tmp / "thresh.py"
for bad_t in ("1.5", "0", '"0.2"', "2"):
    src0.write_text(f"FACE_DET_THRESH = {bad_t}\n", encoding="utf-8")
    raises(f"a face threshold that is not a float in (0, 1) is refused: {bad_t}",
           lambda: check_env.face_threshold(src0))
check("the real faceblur.py pin is read",
      check_env.pinned_centerface_sha256()
      == "09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc")
src = tmp / "faceblur.py"
good = "a" * 64
for name, text in [
    ("no assignment", "X = 1\n"),
    ("two assignments", f'CENTERFACE_SHA256 = "{good}"\nCENTERFACE_SHA256 = "{good}"\n'),
    ("too short", 'CENTERFACE_SHA256 = "abc"\n'),
    ("upper case", f'CENTERFACE_SHA256 = "{good.upper()}"\n'),
    ("not a literal", "CENTERFACE_SHA256 = compute()\n"),
]:
    src.write_text(text, encoding="utf-8")
    raises(f"refused: {name}", lambda: check_env.pinned_centerface_sha256(src))
src.write_text(f'"""doc"""\nCENTERFACE_SHA256: str = "{good}"  # pinned\n', encoding="utf-8")
check("an annotated assignment is read", check_env.pinned_centerface_sha256(src) == good)
src.write_text(f'def f():\n    CENTERFACE_SHA256 = "{"b" * 64}"\nCENTERFACE_SHA256 = "{good}"\n', encoding="utf-8")
check("only the module-level assignment counts", check_env.pinned_centerface_sha256(src) == good)

print("the real deface, where this venv has it")
try:
    import deface.centerface as real_cf
except Exception as e:
    print(f"  SKIP  deface not importable here ({type(e).__name__}); setup-env.ps1 -CheckOnly reports it")
else:
    import importlib
    pinned = check_env.pinned_centerface_sha256()
    why = check_env.check_deface(importlib.import_module, pinned, check_env.face_threshold())
    check("the installed centerface.onnx matches the pin and runs on a black frame", why is None, str(why))
    flipped = bytearray(Path(real_cf.default_onnx_path).read_bytes())
    flipped[len(flipped) // 2] ^= 0x01
    bad = tmp / "tampered" / "centerface.onnx"
    bad.parent.mkdir()
    bad.write_bytes(bytes(flipped))
    parsed = []

    class _RealCF(real_cf.CenterFace):  # the real detector, recording each parse
        def __init__(self, *a, **k):
            parsed.append(k.get("onnx_path"))
            super().__init__(*a, **k)

    class _Tampered:
        default_onnx_path = str(bad)
        CenterFace = _RealCF

    why = check_env.check_deface(lambda m: _Tampered, pinned, check_env.face_threshold())
    check("the real model with ONE bit flipped, through the real CenterFace -> fail, never parsed",
          why is not None and "not the pinned" in why and parsed == [], f"{why} {parsed}")
    _Tampered.default_onnx_path = real_cf.default_onnx_path
    why = check_env.check_deface(lambda m: _Tampered, pinned, check_env.face_threshold())
    check("control: the same wrapper on the untouched model passes and parses it once",
          why is None and parsed == [real_cf.default_onnx_path], f"{why} {parsed}")

print("main")
try:
    check_env.main(["--torch", "cuda"])  # must be a wheel tag, not a word
    check("a --torch value that is not cpu/cuNNN is a usage error", False, "no exit")
except SystemExit as e:
    check("a --torch value that is not cpu/cuNNN is a usage error", e.code == 2)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
