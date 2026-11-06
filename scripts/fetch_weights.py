"""Fetch the pinned YOLO11m-pose weights once, and keep them only if their sha256 matches the pin.

  python scripts/fetch_weights.py            # -> weights/yolo11m-pose.pt
  python scripts/fetch_weights.py --check    # verify an existing file, no network

Set-up, not the pipeline: this is the only download a fresh clone needs, and analyse_clip.py never makes it.
The file comes from the Ultralytics v8.4.0 GitHub release; the pin equals GitHub's own digest for that asset
(docs/MACHINE-SYNC.md section 4). A mismatch deletes the download and exits 1. The checkpoint is a pickle: the
hash says it is the published file, not that it is benign, so the loader checks the hash again on every load.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-pose.pt"
SHA256 = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"
SIZE = 42_459_307
DEST = ROOT / "weights" / "yolo11m-pose.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the existing file only")
    a = ap.parse_args(argv)
    if DEST.is_file():
        ok = sha256(DEST) == SHA256
        print(f"{DEST.relative_to(ROOT)}: {'matches the pin' if ok else 'DOES NOT match the pin'}")
        return 0 if ok else 1
    if a.check:
        print(f"{DEST.relative_to(ROOT)}: missing (run without --check to fetch it)")
        return 1
    DEST.parent.mkdir(parents=True, exist_ok=True)
    part = DEST.with_suffix(".part")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r, part.open("wb") as f:   # noqa: S310 (fixed https URL)
            got = 0
            for block in iter(lambda: r.read(1 << 20), b""):
                got += len(block)
                if got > SIZE:
                    raise ValueError("the download is larger than the pinned file")
                f.write(block)
        if sha256(part) != SHA256:
            raise ValueError("the download does not match the pinned sha256")
        os.replace(part, DEST)
    except (OSError, ValueError) as e:
        part.unlink(missing_ok=True)
        print(f"fetch_weights: {e}; nothing kept", file=sys.stderr)
        return 1
    print(f"-> {DEST.relative_to(ROOT)} (sha256 matches the pin)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
