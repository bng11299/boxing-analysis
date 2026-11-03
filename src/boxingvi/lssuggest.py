"""Label Studio pre-filled punch regions for the sparring labelling session (docs/gates/sparring-per-boxer.md).

Suggestions carry TIMING only: each is a one-frame `unclear` region at a finder peak, with NO boxer and no punch
type. Attribution is what the labels are there to measure, so the tool never proposes a thrower; the importer
refuses any punch without one, so every kept suggestion is a decision Brendan made. Who is scanned: the two
largest boxes in each frame (the glove gate's convention for "the two boxers"), all named unsure, through the
same scan the per-boxer report uses (`boxers.find`).
"""

from __future__ import annotations

import re

import numpy as np

from . import boxers
from .lsimport import EXPECTED_OFFSET, FROM_NAME, TO_NAME, UNCLEAR

MODEL_VERSION = "finder-peaks-v1"
# A Label Studio upload path or a plain http(s) URL; nothing that a browser would run.
_VIDEO_URL = re.compile(r"(/data/(upload|local-files)/|https?://)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")


def two_largest_names(table) -> list:
    """Per pose-table row: "unsure" for the two largest boxes of its frame, None for everyone else."""
    area = (table.box[:, 2] - table.box[:, 0]) * (table.box[:, 3] - table.box[:, 1])
    if not np.isfinite(area).all():
        raise ValueError("the tracks file has a non-finite box")
    names = [None] * len(table.frame)
    order = np.lexsort((-area, table.frame))   # by frame, largest first
    seen, last = 0, None
    for r in order.tolist():
        f = int(table.frame[r])
        seen = seen + 1 if f == last else 1
        last = f
        if seen <= 2:
            names[r] = boxers.UNSURE
    return names


def peak_frames(table, rule) -> list:
    """Every finder peak (video frame) on the two largest people, sorted, one per (frame, track)."""
    segs = boxers.segments_from_row_names(table, two_largest_names(table))
    ident = boxers.Identity("suggest", {"verdict": "ungated"}, tuple(segs), "", "")
    kept = boxers.find(table, ident, rule)[boxers.UNSURE]["kept"]
    return sorted({(int(f), int(table.track_id[seg[0]])) for seg, _t, _s, frames, _h in kept for f in frames})


def results(frames, n_frames: int, offset: int = EXPECTED_OFFSET) -> list:
    """One-frame `unclear` regions at video frames `frames`, in Label Studio frames (video frame + offset)."""
    out = []
    for i, f in enumerate(frames):
        if type(f) is not int or not 0 <= f < n_frames:
            raise ValueError("a suggested frame is outside the video")
        ls = f + offset
        out.append({"id": f"sug{i:05d}", "type": "timelinelabels", "from_name": FROM_NAME, "to_name": TO_NAME,
                    "origin": "prediction",
                    "value": {"ranges": [{"start": ls, "end": ls}], "timelinelabels": [UNCLEAR]}})
    return out


def task(video_url: str, regions: list) -> list:
    """A one-task Label Studio import with the regions as a prediction (shown pre-filled, not yet submitted)."""
    if not isinstance(video_url, str) or not _VIDEO_URL.fullmatch(video_url):
        raise ValueError("video URL must be a Label Studio /data/upload/... path or an http(s) URL")
    if any("boxer" in r["value"] or r["type"] != "timelinelabels" for r in regions):
        raise ValueError("suggestions must not carry a boxer")
    return [{"data": {"video": video_url},
             "predictions": [{"model_version": MODEL_VERSION, "result": regions}]}]
