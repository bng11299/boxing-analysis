"""Fetch the two pinned third-party checkpoints once, and keep each only if its sha256 matches the pin.

  python scripts/fetch_weights.py            # -> weights/yolo11m-pose.pt, weights/stgcnpp_ntu120_xsub_hrnet_j.pth
  python scripts/fetch_weights.py --check    # verify the existing files, no network

Set-up, not the pipeline: these are the only downloads a fresh clone needs, and analyse_clip.py never makes one.
YOLO11m-pose comes from the Ultralytics v8.4.0 GitHub release (the pin equals GitHub's own digest for that asset,
docs/MACHINE-SYNC.md section 4); the ST-GCN++ NTU-120 backbone from the PYSKL model zoo (pinned in
boxingvi.stgcnpp.PRETRAINED). A mismatch deletes the download and exits 1. Both are pickles: the hash says a file
is the published one, not that it is benign, so the loaders check the hash again on every load.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from boxingvi.stgcnpp import PRETRAINED  # noqa: E402

_NTU = PRETRAINED["ntu120_xsub_hrnet_j"]
# (file, url, sha256, size in bytes)
ASSETS = (
    (ROOT / "weights" / "yolo11m-pose.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
     "yolo11m-pose.pt", "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2", 42_459_307),
    (ROOT / "weights" / "stgcnpp_ntu120_xsub_hrnet_j.pth", _NTU["url"], _NTU["sha256"], _NTU["size"]),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(dest: Path, url: str, pin: str, size: int, check_only: bool) -> bool:
    name = dest.relative_to(ROOT)
    if dest.is_file():
        ok = sha256(dest) == pin
        print(f"{name}: {'matches the pin' if ok else 'DOES NOT match the pin'}")
        return ok
    if check_only:
        print(f"{name}: missing (run without --check to fetch it)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:   # noqa: S310 (fixed https URL)
            got = 0
            for block in iter(lambda: r.read(1 << 20), b""):
                got += len(block)
                if got > size:
                    raise ValueError("the download is larger than the pinned file")
                f.write(block)
        if sha256(part) != pin:
            raise ValueError("the download does not match the pinned sha256")
        os.replace(part, dest)
    except (OSError, ValueError) as e:
        part.unlink(missing_ok=True)
        print(f"{name}: {e}; nothing kept", file=sys.stderr)
        return False
    print(f"-> {name} (sha256 matches the pin)")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the existing files only")
    a = ap.parse_args(argv)
    results = [fetch(d, u, h, n, a.check) for d, u, h, n in ASSETS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
