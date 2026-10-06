"""Olympic Boxing Punch Classification Video Dataset (Stefanski, Kozak & Jach,
Entropy 26(8) 617, 2024): CVAT "video" JSON annotations -> punch events.

What the data is, as measured on the release (2026-10-06):
  * 29 videos from two GoPro cameras (task_kam2_* x15, task_kam4_* x14), 1920x1080,
    constant 50 fps, 312,774 frames in all -- exactly the paper's "312,774 frames".
  * One CVAT track per punch: a few bounding boxes on the punching glove (median
    4 frames, 0.08 s), labelled TARGET x HAND in Polish. The paper: a licensed
    referee "looked at 312,774 frames, where 11,345 ... were considered as punches";
    misses and blocks are punches too, so a label marks every punch THROWN.
  * HAND is the boxer's own left or right ("lewa/prawa reka"), not lead/rear, and
    the label does not say which boxer threw it: the box position does.

Splits: the paper says the fight was cut "into 14 videos for each camera", so kam2
and kam4 may show the SAME punches from two angles. Hold out whole bouts or time
windows, never single videos, or the same punch sits in training and test.

Licence: non-commercial academic use, no redistribution. Nothing derived from it
(events, keypoints, clips) is ever released (project rules, ADR-0011).

The annotation file is third-party input: anything outside the shape this loader
knows is refused, not guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGETS = {"Głowa": "head", "Korpus": "body", "Blok": "block", "Chybienie": "miss"}
HANDS = {"lewą ręką": "left", "prawą ręką": "right"}
MAX_ANNOTATION_BYTES = 20 * 1024 * 1024
DATASET_DIR = "Olympic Boxing Punch Classification Video Dataset"


MAX_HIDDEN = 2      # a track hidden this briefly and then visible again is still one punch


@dataclass(frozen=True)
class Event:
    """One labelled punch: the glove box is visible on frames first..last.

    CVAT semantics: a track is visible from a visible keyframe until its next
    `outside` keyframe, so `last` is that outside frame minus one, not the last
    visible keyframe (965 of 4,605 events in the release run 2-12 frames past it).
    """

    first: int
    last: int
    target: str                 # head / body / block / miss
    hand: str                   # the boxer's left / right
    box: tuple                  # (x1, y1, x2, y2) at the first visible frame, pixels
    n_keyframes: int
    open_ended: bool = False    # no closing outside keyframe: `last` is the last visible keyframe
    reappeared: bool = False    # hidden for <= MAX_HIDDEN frames mid-track, then visible again

    @property
    def frame(self) -> int:
        """The event's frame for spotting: the first frame the glove box appears."""
        return self.first


def _no_constant(name):
    raise ValueError(f"the annotation file contains {name}, which is not a number")


def parse_label(label: str) -> "tuple[str, str]":
    if not isinstance(label, str):
        raise ValueError(f"label is not text: {label!r}")
    for pl_t, t in TARGETS.items():
        for pl_h, h in HANDS.items():
            if label == f"{pl_t} {pl_h}":
                return t, h
    raise ValueError(f"unknown label {label!r}")


def _box(points, width: int, height: int) -> tuple:
    if (not isinstance(points, list) or len(points) != 4
            or any(isinstance(p, bool) or not isinstance(p, (int, float)) for p in points)):
        raise ValueError(f"a rectangle needs four numbers, got {points!r}")
    x1, y1, x2, y2 = (float(p) for p in points)
    if not (np.isfinite([x1, y1, x2, y2]).all() and x1 < x2 and y1 < y2):
        raise ValueError(f"degenerate box {points!r}")
    # Strict: none of the release's 17,649 visible boxes crosses the frame edge.
    if not (0 <= x1 and x2 <= width and 0 <= y1 and y2 <= height):
        raise ValueError(f"box {points!r} is not inside the {width}x{height} frame")
    return (x1, y1, x2, y2)


def load_events(annotations, n_frames: int, width: int = 1920, height: int = 1080) -> "tuple[list[Event], int]":
    """Events from one task's annotations.json (path or parsed list), and the number
    of tracks with no visible shape (CVAT leaves a few; they carry no punch)."""
    if isinstance(annotations, (str, Path)):
        p = Path(annotations)
        if p.stat().st_size > MAX_ANNOTATION_BYTES:
            raise ValueError(f"{p} is too large to be an annotation file")
        annotations = json.loads(p.read_text(encoding="utf-8"), parse_constant=_no_constant)
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("expected a non-empty list of CVAT job documents")
    events, empty = [], 0
    for doc in annotations:
        if not isinstance(doc, dict):
            raise ValueError("a job document is not an object")
        if doc.get("shapes") or doc.get("tags"):
            raise ValueError("free-standing shapes or tags: this loader only knows punch tracks")
        for tr in doc.get("tracks") or []:
            target, hand = parse_label(tr.get("label"))
            shapes = tr.get("shapes")
            if not isinstance(shapes, list):
                raise ValueError("a track has no shape list")
            keys = []  # (frame, outside, box)
            for s in shapes:
                if not isinstance(s, dict) or s.get("type") != "rectangle":
                    raise ValueError(f"unexpected shape {s.get('type') if isinstance(s, dict) else s!r}")
                f = s.get("frame")
                if isinstance(f, bool) or not isinstance(f, int) or not 0 <= f < n_frames:
                    raise ValueError(f"shape frame {f!r} is outside the video (0-{n_frames - 1})")
                out = bool(s.get("outside"))
                keys.append((f, out, None if out else _box(s.get("points"), width, height)))
            keys.sort(key=lambda k: k[0])
            if len({k[0] for k in keys}) != len(keys):
                raise ValueError(f"a {target}/{hand} track has two keyframes on one frame")
            # Walk the keyframes: visible from a visible keyframe up to the next outside one.
            intervals, start = [], None
            for f, out, _ in keys:
                if not out and start is None:
                    start = f
                elif out and start is not None:
                    intervals.append([start, f - 1])
                    start = None
            open_ended = start is not None
            if open_ended:  # CVAT would show it to the end of the video; a punch is not that long
                intervals.append([start, max(f for f, out, _ in keys if not out)])
            if not intervals:
                empty += 1
                continue
            for prev, nxt in zip(intervals, intervals[1:]):
                if nxt[0] - prev[1] - 1 > MAX_HIDDEN:
                    raise ValueError(f"a {target}/{hand} track disappears at frame {prev[1] + 1} and comes back "
                                     f"at {nxt[0]}: two punches in one track? Refusing to merge them")
            first_box = next(b for f, out, b in keys if not out)
            events.append(Event(intervals[0][0], intervals[-1][1], target, hand, first_box,
                                sum(1 for k in keys if not k[1]), open_ended, len(intervals) > 1))
    events.sort(key=lambda e: (e.first, e.last))
    return events, empty


@dataclass(frozen=True)
class Task:
    name: str           # task_kam2_gh078416
    camera: str         # kam2
    video: Path
    annotations: Path


def find_tasks(root) -> "list[Task]":
    """Every task folder under `root` (the dataset folder, or its parent)."""
    root = Path(root)
    if (root / DATASET_DIR).is_dir():
        root = root / DATASET_DIR
    tasks = []
    for t in sorted(p for p in root.iterdir() if p.is_dir()):
        parts = t.name.split("_")
        if len(parts) != 3 or parts[0] != "task" or not parts[1].startswith("kam"):
            raise ValueError(f"unexpected folder {t.name!r} in the dataset")
        vids = list((t / "data").glob("*.mp4"))
        ann = t / "annotations.json"
        if len(vids) != 1 or not ann.is_file():
            raise ValueError(f"{t.name}: expected one video in data/ and an annotations.json")
        tasks.append(Task(t.name, parts[1], vids[0], ann))
    if not tasks:
        raise ValueError(f"no tasks under {root}")
    return tasks
