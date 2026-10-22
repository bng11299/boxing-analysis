"""Tests for scripts/check_env.py -- the check that a machine's .venv really works.

Run:  python tests/test_env.py
"""

from __future__ import annotations

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


def env(build="cpu", torch=None, missing=(), broken=(), which_missing=(), py=(3, 13), pose=False,
        tv=None, tv_version=None):
    every = dict(check_env.parse_pins(ROOT / "requirements.txt"))
    every.update(check_env.parse_pins(ROOT / "requirements-pose.txt"))
    every["torch"] = "2.13.0+" + build
    every["torchvision"] = tv_version or "0.28.0+" + build

    def version(name):
        return None if name in missing else every.get(name)

    def imp(mod):
        if mod in broken:
            raise OSError("[WinError 126] The specified module could not be found")
        if mod == "torchvision":
            return tv or fake_tv()
        return torch if mod == "torch" else object()

    def which(tool):
        return None if tool in which_missing else check_env.shutil.which(tool)

    return check_env.run(build, pose, version=version, imp=imp, which=which, py=py)


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

print("main")
try:
    check_env.main(["--torch", "cuda"])  # must be a wheel tag, not a word
    check("a --torch value that is not cpu/cuNNN is a usage error", False, "no exit")
except SystemExit as e:
    check("a --torch value that is not cpu/cuNNN is a usage error", e.code == 2)

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
