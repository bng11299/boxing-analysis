"""Label Studio export -> punch labels for one video (labelstudio/punch-labels.xml).

  python cli/import_labels.py --export project-1-at-....json --video results/ingest/IMG_4406.overlay.mp4

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxingvi import lsimport  # noqa: E402
from boxingvi.video import probe  # noqa: E402

DEFAULT_OUT = Path("data/own")


def main(argv=None) -> int:
    # allow_abbrev=False: `--spar` must not reach sparring mode without the
    # sanitised error wrapper below, which matches the full flag only.
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True, help='Label Studio export, format "JSON"')
    ap.add_argument("--video", type=Path, required=True, help="the video that was labelled")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--sparring", action="store_true", help="require a boxer on each punch; write next to the video")
    ap.add_argument("--force", action="store_true", help="replace existing labels for this clip")
    a = ap.parse_args(argv)

    if a.sparring:
        return _sparring(a)
    if a.out_dir is None:
        a.out_dir = DEFAULT_OUT

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


def _sparring(args):
    from boxingvi.sparring_io import external_path, staged_output, strict_cfr

    if args.out_dir is None:
        raise ValueError("--out-dir is required in --sparring mode")
    directory = external_path(args.out_dir)
    video = args.video.resolve()
    if directory != video.parent:
        raise ValueError("--out-dir must be next to the video (its containing directory)")
    out = external_path(directory / f"{video.stem}.labels.json")
    if out.exists() and not args.force:
        raise ValueError("output exists; pass --force to replace it")
    if out == args.export.resolve() or (out.exists() and out.samefile(args.export)):
        raise ValueError("output must not replace the source export")
    info = strict_cfr(video)
    if info.fps_frac != 30:
        raise ValueError("sparring config frameRate is 30; video must be constant 30 fps")
    export, digest = lsimport.load_sparring_export(args.export)
    labels = lsimport.parse_sparring_export(export, info.timing.n_frames)
    record = labels.to_json(video.stem, info.fps, info.timing.n_frames)
    record["export_sha256"] = digest
    record["label_studio"] = "1.23.2 (frames 1-based; offset must be +1)"
    with staged_output(out, args.force) as temporary:
        temporary.write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
    print(f"{len(labels.spans)} punches; {info.timing.n_frames} frames; {info.fps:g} fps")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    if "--sparring" in sys.argv[1:]:
        try:
            sys.exit(main())
        except (ValueError, OSError, RuntimeError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else "sparring import failed"
            print(f"error: {message}", file=sys.stderr)
            sys.exit(1)
    else:
        sys.exit(main())
