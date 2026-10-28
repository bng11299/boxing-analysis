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
import hashlib
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


BOXERS = ("A", "B", "?")


@dataclass
class SparringLabelSet(LabelSet):
    boxers: list[str] = field(default_factory=list)

    def to_json(self, source: str, fps: float, n_frames: int) -> dict:
        record = super().to_json(source, fps, n_frames)
        record["mode"] = "sparring"
        for span, boxer in zip(record["spans"], self.boxers, strict=True):
            span["type"] = span.pop("label")
            span["boxer"] = boxer
        return record


def load_sparring_export(path):
    """Parse and hash the SAME bounded bytes; errors never echo export contents."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_EXPORT_BYTES + 1)
    if len(raw) > MAX_EXPORT_BYTES:
        raise ValueError("export is over the 50 MB size cap")

    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("export contains a non-finite number")
        return value

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("export contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        export = json.loads(raw.decode("utf-8"), parse_constant=_no_constant,
                            parse_float=finite, object_pairs_hook=unique)
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("export is invalid JSON or nested too deeply") from None
    return export, hashlib.sha256(raw).hexdigest()


def parse_sparring_export(export, n_frames: int) -> SparringLabelSet:
    """Join Choices to TimelineLabels by region id, then apply solo rules per boxer.

    '?' is its own conservative overlap bucket: two unidentifiable throwers with
    overlapping same-hand punches require review. A and B never conflict.
    Only one contiguous range per region is allowed (one region = one punch).
    """
    if type(n_frames) is not int or n_frames < 1:
        raise ValueError("the video has no frames or an invalid frame count")
    if not isinstance(export, list) or len(export) != 1 or not isinstance(export[0], dict):
        raise ValueError("expected an export of exactly one task")
    task = export[0]
    if "annotations" not in task:
        raise ValueError('export as "JSON", not "JSON-MIN"')
    if task.get("drafts"):
        raise ValueError("unsubmitted drafts: press Submit and export again")
    annotations = task["annotations"]
    if not isinstance(annotations, list) or any(not isinstance(a, dict) for a in annotations):
        raise ValueError("annotations must be a list of annotation objects")
    if any(type(a.get("was_cancelled", False)) is not bool for a in annotations):
        raise ValueError("invalid annotation cancellation flag")
    completed = [a for a in annotations if not a.get("was_cancelled", False)]
    if len(completed) != 1:
        raise ValueError("expected exactly one completed annotation; press Submit")
    ann = completed[0]
    results = ann.get("result")
    if not isinstance(results, list):
        raise ValueError("annotation result must be a list")
    timelines, choices = {}, {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("malformed result object")
        kind = result.get("type")
        if kind not in ("timelinelabels", "choices"):
            raise ValueError("unexpected result type for sparring-punch-labels.xml")
        expected_name = FROM_NAME if kind == "timelinelabels" else "boxer"
        if (result.get("from_name"), result.get("to_name")) != (expected_name, TO_NAME):
            raise ValueError("unexpected control names for sparring-punch-labels.xml")
        rid = result.get("id")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError("every region result needs a nonempty string id")
        value = result.get("value")
        if not isinstance(value, dict):
            raise ValueError("malformed region value")
        destination = timelines if kind == "timelinelabels" else choices
        if rid in destination:
            reason = "duplicate timeline region id" if kind == "timelinelabels" else "two boxer results for one region"
            raise ValueError(reason)
        destination[rid] = value
    if choices.keys() - timelines.keys():
        raise ValueError("orphan boxer choice has no timeline region")

    checks, by_boxer = [], {boxer: [] for boxer in BOXERS}
    for rid, value in timelines.items():
        labels = value.get("timelinelabels")
        if not isinstance(labels, list) or len(labels) != 1:
            raise ValueError("a region needs exactly one label")
        label = labels[0]
        if not isinstance(label, str) or label not in (*CLASSES, UNCLEAR, FRAME_CHECK):
            raise ValueError("unknown label in timeline region")
        ranges = value.get("ranges")
        if not isinstance(ranges, list) or len(ranges) != 1 or not isinstance(ranges[0], dict):
            raise ValueError("each region needs exactly one range")
        for key in ("start", "end"):
            frame = ranges[0].get(key)
            if (type(frame) not in (int, float) or
                    (isinstance(frame, float) and not math.isfinite(frame)) or frame != int(frame)):
                raise ValueError("range endpoints must be finite whole frame numbers")
        # Reject misplaced attributes rather than silently ignoring a second source.
        if "boxer" in value or "choices" in value:
            raise ValueError("boxer must be a separate per-region Choices result")
        normalized = {"type": "timelinelabels", "from_name": FROM_NAME, "to_name": TO_NAME,
                      "value": {"timelinelabels": [label], "ranges": ranges}}
        if label == FRAME_CHECK:
            if rid in choices:
                raise ValueError("frame-check region must not carry a boxer")
            if any(not 1 <= ranges[0][key] <= n_frames for key in ("start", "end")):
                raise ValueError("frame-check is outside the video")
            checks.append(normalized)
            continue
        if rid not in choices:
            raise ValueError("missing boxer for punch region")
        selection = choices[rid].get("choices")
        if not isinstance(selection, list) or not selection:
            raise ValueError("missing boxer value for punch region")
        if len(selection) != 1:
            raise ValueError("two or more boxer values; exactly one boxer is required")
        boxer = selection[0]
        if not isinstance(boxer, str) or boxer not in BOXERS:
            raise ValueError("unknown boxer value; expected A, B or ?")
        # LS 1.23.2 serializes the same ranges on both results. Compare types too:
        # Python's equality would otherwise accept true in place of frame 1.
        if json.dumps(choices[rid].get("ranges"), sort_keys=True) != json.dumps(ranges, sort_keys=True):
            raise ValueError("boxer ranges do not match the timeline region")
        by_boxer[boxer].append(normalized)

    # Across boxers, A and B may punch at the same time, so their regions may overlap.
    # A "?" region on top of a named one is far more likely the same punch marked
    # twice (once unsure, once decided): a phantom punch that lowers recall. Refuse.
    def _span(region):
        r = region["value"]["ranges"][0]
        return (int(r["start"]), int(r["end"]))
    for unsure in by_boxer["?"]:
        for named in by_boxer["A"] + by_boxer["B"]:
            if _iou(_span(unsure), _span(named)) >= DUPLICATE_IOU:
                s, e = _span(unsure)
                raise ValueError(f"a '?' region at Label Studio frames {s}-{e} duplicates a punch "
                                 "already given boxer A or B; delete one, or name the boxer")

    lead = ann.get("lead_time")
    if type(lead) not in (int, float) or not 0 <= lead <= 1e12:
        lead = None
    punches = []
    metadata = None
    for boxer, regions in by_boxer.items():
        projected = [{"id": task.get("id"), "annotations": [
            {"result": checks + regions, "lead_time": lead}]}]
        parsed = parse_export(projected, n_frames)
        if metadata is not None and parsed.offset != metadata.offset:
            raise ValueError("frame offset differs between boxers")  # same checks: cannot happen
        metadata = parsed
        punches.extend((span, boxer) for span in parsed.spans)
    punches.sort(key=lambda item: (item[0].start, item[0].end, item[1]))
    spans = [span for span, _ in punches]
    counts = {c: sum(span.label == i for span in spans) for i, c in enumerate(CLASSES)}
    counts[UNCLEAR] = sum(span.label == UNCLEAR_LABEL for span in spans)
    return SparringLabelSet(spans, metadata.offset, metadata.lead_time_s,
                            metadata.task_id, counts, [boxer for _, boxer in punches])
