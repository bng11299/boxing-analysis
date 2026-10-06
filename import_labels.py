"""Label Studio export -> punch labels for one video (labelstudio/punch-labels.xml).

  python import_labels.py --export project-1-at-....json --video results/ingest/IMG_4406.overlay.mp4

Writes data/own/<clip>.labels.json (gitignored): inclusive video frame ids per
punch, after removing Label Studio's frame offset, which the two frame-check
marks measure. Prints the counts and the labelling time Label Studio recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from boxingvi import lsimport  # noqa: E402
from boxingvi.video import probe  # noqa: E402

DEFAULT_OUT = Path("data/own")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True, help='Label Studio export, format "JSON"')
    ap.add_argument("--video", type=Path, required=True, help="the video that was labelled")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="replace existing labels for this clip")
    a = ap.parse_args(argv)

    info = probe(a.video)
    if info.timing is None or info.vfr:
        raise ValueError(f"{a.video.name}: frame timing is irregular or unmeasurable, so Label Studio's "
                         "frame numbers cannot be trusted; label a constant-rate overlay instead")
    stem = a.video.name.split(".")[0]           # IMG_4406.overlay.mp4 -> IMG_4406
    out = a.out_dir / f"{stem}.labels.json"
    if out.exists() and not a.force:
        raise ValueError(f"{out} exists: these are ground-truth labels; pass --force to replace them")
    labels = lsimport.parse_export(lsimport.load_export(a.export), info.timing.n_frames)
    record = labels.to_json(stem, info.fps, info.timing.n_frames)
    record["export_sha256"] = hashlib.sha256(a.export.read_bytes()).hexdigest()
    record["label_studio"] = "1.23 (frames 1-based; offset must be +1)"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
    n = len(labels.spans)
    secs = info.timing.n_frames / info.fps
    print(f"{stem}: {n} punches in {secs:.1f} s of video; Label Studio frame offset {labels.offset:+d}")
    print("  " + ", ".join(f"{k} {v}" for k, v in labels.counts.items()))
    if labels.lead_time_s is not None:  # Label Studio counts idle time in the editor too
        print(f"  labelling time: at most {labels.lead_time_s / 60:.1f} min (Label Studio's clock, "
              "idle time included; your own stopwatch time is the number to use)")
    else:
        print("  labelling time: not in the export; use your stopwatch time")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
