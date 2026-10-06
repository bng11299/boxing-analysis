"""Punch labels out of a Label Studio export (labelstudio/punch-labels.xml).

The export is untrusted input and the labels become the ground truth every
punch-finding number is scored against, so anything ambiguous is refused rather
than guessed: more than one task or annotation, an unknown label, a range outside
the video, overlapping punches by the same hand, or frame checks that disagree.

Frame numbering is MEASURED as well as known. Label Studio 1.23 numbers timeline
frames from 1 (its source: TimelineRegionEditor `min={1}`, VideoCanvas
`ceil(t * fps)`), so its frame f is video frame f - 1: an offset of exactly +1.
The labeller marks single-frame "frame-check" regions on frames whose burned-in
overlay number is known (100 and 2000). Both must give +1: a wrong `frameRate`
drifts them apart, and a check placed one frame off would otherwise shift every
punch by a frame without anything looking wrong.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .data import CLASSES, Span

UNCLEAR = "unclear"
FRAME_CHECK = "frame-check"
UNCLEAR_LABEL = -1                 # Span.label for a punch nobody could name
FRAME_CHECKS = (100, 2000)         # burned-in frame numbers the labeller marks
EXPECTED_OFFSET = 1                # Label Studio 1.23: frames are 1-based
FROM_NAME, TO_NAME = "punches", "video"  # the tag names in labelstudio/punch-labels.xml
MAX_EXPORT_BYTES = 50 * 1024 * 1024
LEAD = {"jab", "lead_hook", "lead_uppercut"}
REAR = {"cross", "rear_hook", "rear_uppercut"}
DUPLICATE_IOU = 0.5                # an "unclear" this much on top of another region is a duplicate


@dataclass
class LabelSet:
    spans: "list[Span]"
    offset: int                    # Label Studio frame - video frame
    lead_time_s: "float | None"    # time Label Studio recorded on the annotation
    task_id: "int | None" = None
    counts: dict = field(default_factory=dict)

    def to_json(self, source: str, fps: float, n_frames: int) -> dict:
        name = lambda k: CLASSES[k] if k >= 0 else UNCLEAR  # noqa: E731
        return {"source": source, "fps": fps, "n_frames": n_frames,
                "label_studio_offset": self.offset, "lead_time_s": self.lead_time_s,
                "task_id": self.task_id, "counts": self.counts,
                "spans": [{"start": s.start, "end": s.end, "label": name(s.label)} for s in self.spans]}


def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
        raise ValueError(f"{what} is not a whole frame number: {v!r}")
    return int(v)


def _no_constant(name):
    raise ValueError(f"the export contains {name}, which is not a number")


def load_export(path) -> object:
    p = Path(path)
    if p.stat().st_size > MAX_EXPORT_BYTES:
        raise ValueError(f"{p.name} is over {MAX_EXPORT_BYTES // 2**20} MB; not a label export")
    try:  # NaN / Infinity are not JSON; Python accepts them unless told not to
        return json.loads(p.read_text(encoding="utf-8"), parse_constant=_no_constant)
    except RecursionError as e:
        raise ValueError(f"{p.name} is nested too deeply to be a label export") from e


def _iou(a, b) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    return max(inter, 0) / (max(a[1], b[1]) - min(a[0], b[0]) + 1)


def parse_export(export, n_frames: int, checks=FRAME_CHECKS,
                 expected_offset: int = EXPECTED_OFFSET) -> LabelSet:
    """Labels for ONE video from a Label Studio JSON export ("JSON", not "JSON-MIN")."""
    if n_frames < 1:
        raise ValueError("the video has no frames")
    if not isinstance(export, list) or len(export) != 1 or not isinstance(export[0], dict):
        n = len(export) if isinstance(export, list) else "not a list"
        raise ValueError(f"expected an export of exactly one task, got {n}; export one video at a time")
    task = export[0]
    if "annotations" not in task:
        raise ValueError('no "annotations" in the task: export as "JSON", not "JSON-MIN"')
    anns = [a for a in task["annotations"] if isinstance(a, dict) and not a.get("was_cancelled")]
    if len(anns) != 1:
        drafts = len(task.get("drafts") or [])
        hint = (f"; there {'is' if drafts == 1 else 'are'} {drafts} unsubmitted draft(s): press Submit "
                "in Label Studio, then export again") if not anns and drafts else ""
        raise ValueError(f"expected exactly one completed annotation, found {len(anns)}{hint}; "
                         "with several, delete the extras so there is no doubt which is ground truth")
    ann = anns[0]

    regions = []  # (label, ls_start, ls_end)
    for r in ann.get("result") or []:
        if not isinstance(r, dict) or r.get("type") != "timelinelabels":
            raise ValueError(f"unexpected result type {r.get('type') if isinstance(r, dict) else r!r}; "
                             "the importer only knows punch-labels.xml")
        if (r.get("from_name"), r.get("to_name")) != (FROM_NAME, TO_NAME):
            raise ValueError(f"a region from {r.get('from_name')!r} on {r.get('to_name')!r}: this export "
                             f"was not made with punch-labels.xml ({FROM_NAME!r} on {TO_NAME!r})")
        v = r.get("value") or {}
        labels, ranges = v.get("timelinelabels"), v.get("ranges")
        if not (isinstance(labels, list) and len(labels) == 1 and isinstance(ranges, list) and ranges):
            raise ValueError(f"a region needs exactly one label and at least one range: {v!r}")
        label = labels[0]
        if label not in CLASSES and label not in (UNCLEAR, FRAME_CHECK):
            raise ValueError(f"unknown label {label!r}")
        for rg in ranges:
            if not isinstance(rg, dict):
                raise ValueError(f"malformed range {rg!r}")
            s, e = _int(rg.get("start"), "range start"), _int(rg.get("end"), "range end")
            if s > e:
                raise ValueError(f"{label} region ends ({e}) before it starts ({s})")
            regions.append((label, s, e))

    marks = sorted((s, e) for lab, s, e in regions if lab == FRAME_CHECK)
    if len(marks) != len(checks):
        raise ValueError(f"expected {len(checks)} frame-check marks (on burned-in frames "
                         f"{', '.join(map(str, checks))}), found {len(marks)}")
    if any(s != e for s, e in marks):
        raise ValueError("each frame-check must be a single frame (click once, do not drag)")
    offsets = [s - c for (s, _), c in zip(marks, sorted(checks))]
    if len(set(offsets)) != 1:
        raise ValueError(f"frame checks disagree (offsets {offsets}): Label Studio's frameRate "
                         "does not match the video; fix it in the config and re-label")
    offset = offsets[0]
    if offset != expected_offset:
        raise ValueError(f"frame-check offset is {offset:+d}, expected {expected_offset:+d} (Label Studio "
                         "numbers frames from 1). The checks are not on burned-in frames "
                         f"{' and '.join(map(str, checks))}, or frameRate is off")

    spans, last_by_hand, placed = [], {}, []
    for label, s, e in sorted((x for x in regions if x[0] != FRAME_CHECK), key=lambda x: (x[1], x[2])):
        s, e = s - offset, e - offset
        if s < 0 or e >= n_frames:
            raise ValueError(f"{label} at frames {s}-{e} is outside the video (0-{n_frames - 1})")
        # "unclear" has no hand, so the hand rule cannot catch it; a duplicate would
        # count as an extra ground-truth punch and quietly lower recall.
        for pl, ps, pe in placed:
            if UNCLEAR not in (label, pl):
                continue
            iou = _iou((s, e), (ps, pe))
            if (label == pl and iou > 0) or iou >= DUPLICATE_IOU:  # unclear on unclear: any overlap
                raise ValueError(f"{label} at {s}-{e} duplicates {pl} at {ps}-{pe}; "
                                 "delete one so each punch is labelled once")
        placed.append((label, s, e))
        hand = "lead" if label in LEAD else "rear" if label in REAR else None
        if hand and hand in last_by_hand and s <= last_by_hand[hand][2]:
            pl, ps, pe = last_by_hand[hand]
            raise ValueError(f"two {hand}-hand punches overlap: {pl} at {ps}-{pe} and {label} at "
                             f"{s}-{e}; one hand throws one punch at a time")
        if hand:
            last_by_hand[hand] = (label, s, e)
        spans.append(Span(s, e, CLASSES.index(label) if label in CLASSES else UNCLEAR_LABEL))

    lead = ann.get("lead_time")  # cumulative wall-clock in the editor, idle time included
    ok = isinstance(lead, (int, float)) and not isinstance(lead, bool) and math.isfinite(lead) and lead >= 0
    lead = float(lead) if ok else None
    tid = task.get("id")
    tid = tid if isinstance(tid, int) and not isinstance(tid, bool) else None
    counts = {c: sum(1 for sp in spans if sp.label == i) for i, c in enumerate(CLASSES)}
    counts[UNCLEAR] = sum(1 for sp in spans if sp.label == UNCLEAR_LABEL)
    return LabelSet(spans, offset, lead, tid, counts)
