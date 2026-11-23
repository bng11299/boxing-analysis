"""Write a Label Studio task with finder peaks pre-filled as one-frame `unclear` regions (no boxer, no type).

python scripts/suggest_sparring.py --tracks ~/comps-participant-data/glove-identity/gg1022-v2.tracks.npz \
    --video-url /data/upload/1/xxxx-gg1022-v2.numbered.mp4 \
    --output ~/comps-participant-data/sparring-labels/gg1022-v2.suggested.json

The output must be outside any git checkout. See docs/gates/sparring-per-boxer.md for how kept suggestions are
reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

import find_punches  # noqa: E402
from boxingvi import lssuggest  # noqa: E402
from boxingvi.pose_yolo import PoseTable  # noqa: E402
from boxingvi.sparring_io import staged_output  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--video-url", required=True, help="the task's video path as Label Studio shows it")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rule", type=Path, default=find_punches.DEFAULT_RULE)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    table = PoseTable.load(a.tracks)
    if abs(table.fps - 30) > 1e-6:
        raise ValueError("sparring-punch-labels.xml is set to 30 fps; these tracks are not 30 fps")
    frames = [f for f, _tid in lssuggest.peak_frames(table, find_punches.load_rule(a.rule))]
    doc = lssuggest.task(a.video_url, lssuggest.results(frames, table.n_frames))
    with staged_output(a.output, a.force) as tmp:
        tmp.write_text(json.dumps(doc), encoding="utf-8")
    print(f"{len(frames)} suggested regions ({table.n_frames / table.fps / 60:.1f} min) -> {a.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
