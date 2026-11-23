"""The pipeline runs on its own: a fresh clone, no network, no assistant, no hand-run Python.

1. A fresh `git clone` of this repo's HEAD (only committed files: no results/, weights/ or data/) runs
   scripts/offline_smoke.py --allow-synthetic to green: solo and --two-boxer, video in -> every output out,
   every socket call refused, GEMINI_API_KEY unset.
2. The guards that make that true: the smoke run refuses without --allow-synthetic; its network guard does
   refuse a connection; the pose loader sets YOLO_OFFLINE before ultralytics is imported; make_identity.py
   refuses ids given to both boxers, ids not in the tracks, and non-numeric ids.

Run:  python tests/test_offline.py      (needs git and ffmpeg; tests HEAD, so commit first)
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cli"))
sys.path.insert(0, str(ROOT / "scripts"))

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def refused(name, fn, needle):
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            fn()
    except (ValueError, SystemExit) as e:
        check(name, needle in str(e) or isinstance(e, SystemExit), str(e))
    else:
        check(name, False, "not refused")


env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
tmp = Path(tempfile.mkdtemp(prefix="boxingvi-offline-"))

print("== a fresh clone runs the whole chain offline ==")
clone = tmp / "clone"
subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(ROOT), str(clone)], check=True, env=env)
head = subprocess.run(["git", "-C", str(clone), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                      check=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True).stdout.strip()
print(f"  (clone of HEAD {head}" + (", NOTE: the checkout has uncommitted changes the clone does not)" if dirty
                                     else ")"))
check("the clone has no results/, weights/ or data/", not any((clone / d).exists()
                                                               for d in ("results", "weights", "data")))
smoke_out = tmp / "smoke"
r = subprocess.run([sys.executable, "scripts/offline_smoke.py", "--allow-synthetic", "--out-dir", str(smoke_out)],
                   cwd=clone, env=env, capture_output=True, text=True, timeout=1800)
check("offline_smoke.py --allow-synthetic is green in the fresh clone", r.returncode == 0,
      (r.stdout + r.stderr)[-3000:])
check("it ran solo and two-boxer", "solo: exit 0" in r.stdout and "two-boxer: exit 0" in r.stdout)
check("the clone's tree is untouched by the run", subprocess.run(
    ["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True).stdout.strip() == "")

print("== the guards ==")
r2 = subprocess.run([sys.executable, "scripts/offline_smoke.py"], cwd=clone, env=env, capture_output=True,
                    text=True, timeout=120)
check("the smoke run refuses without --allow-synthetic (exit 2)", r2.returncode == 2, r2.stderr)

guard = ("import socket, sys; sys.path.insert(0, 'scripts'); import offline_smoke as s; s.block_network()\n"
         "try:\n    socket.create_connection(('example.com', 443), timeout=2)\nexcept s.NetworkUsed:\n"
         "    sys.exit(0)\nsys.exit(1)")
r3 = subprocess.run([sys.executable, "-c", guard], cwd=clone, env=env, capture_output=True, text=True, timeout=60)
check("the smoke run's network guard refuses a connection", r3.returncode == 0, r3.stderr)

from boxingvi import pose_yolo  # noqa: E402

seen = {}
fake = types.ModuleType("ultralytics")


class _YOLO:
    def __init__(self, p):
        pass

    def to(self, d):
        return self


def _getattr(name):
    if name == "YOLO":
        seen["offline"] = os.environ.get("YOLO_OFFLINE")
        return _YOLO
    raise AttributeError(name)


fake.__getattr__ = _getattr
old_mod, old_env = sys.modules.get("ultralytics"), os.environ.pop("YOLO_OFFLINE", None)
sys.modules["ultralytics"] = fake
w = tmp / "w.pt"
w.write_bytes(b"weights")
try:
    pose_yolo.load_yolo(w, pose_yolo.sha256_file(w))
finally:
    if old_mod is None:
        sys.modules.pop("ultralytics", None)
    else:
        sys.modules["ultralytics"] = old_mod
check("the pose loader sets YOLO_OFFLINE=1 before ultralytics is used", seen.get("offline") == "1", str(seen))
if old_env is None:
    os.environ.pop("YOLO_OFFLINE", None)
else:
    os.environ["YOLO_OFFLINE"] = old_env

import make_identity  # noqa: E402

tracks = smoke_out / "BOUT01.tracks.npz"
if tracks.is_file():
    o = tmp / "ident"
    o.mkdir()
    refused("make_identity: an id given to both boxers", lambda: make_identity.main(
        ["--tracks", str(tracks), "--a", "1", "--b", "1", "2", "--out", str(o / "BOUT01.identity.csv")]), "both")
    refused("make_identity: an id not in the tracks", lambda: make_identity.main(
        ["--tracks", str(tracks), "--a", "1", "--b", "7", "--out", str(o / "BOUT01.identity.csv")]), "not in")
    refused("make_identity: a non-numeric id", lambda: make_identity.main(
        ["--tracks", str(tracks), "--a", "1", "--b", "-2"]), "whole number")
    refused("make_identity: an output named for another clip", lambda: make_identity.main(
        ["--tracks", str(tracks), "--a", "1", "--b", "2", "--out", str(o / "OTHER.identity.csv")]), "clip")
    check("make_identity: nothing written by a refusal", list(o.iterdir()) == [], str(list(o.iterdir())))
else:
    check("make_identity: the smoke run left a tracks file to test on", False)

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
