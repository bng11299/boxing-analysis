"""Face blur for the second annotator: video in, face-blurred H.264 copy out.

THIS IS A PRIVACY CONTROL. Protocol FA26-22-NG as amended (vault ISSUES-LOG O57,
O58, O63): the second annotator sees face-blurred clips only, and is never told
names. Every decision below fails toward MORE blur, never less:

- Every person YOLO11-pose finds in a frame is blurred, not only the tracked boxer
  (`pose_yolo.extract` keeps one person; this does not).
- The BOX is the floor: every person's box patch is blurred, always -- the top
  fifth of the box when it is body-shaped (at least twice as tall as wide; 1.5 box
  widths if it touches the frame edge, where close-up crops come from), else the
  whole box (a crop, or a deep crouch, can be mostly head).
  Keypoints only ADD blur, never replace it: four review rounds each found a new
  way a confidently wrong skeleton moved a face patch off a head, and no list of
  guards closes that space (docs/quality-loop.md, blocker report, option 1).
- A person with two or more confident face keypoints (nose, eyes, ears) also gets
  a HEAD-SIZED patch (~1.8 head widths), the head measured from the face points
  themselves (`head_width`). With one face point the patch is sized generously from
  the body instead (the LARGEST of shoulder width, torso length, box height, and the
  box itself unless the skeleton reaches the legs inside it).
- The whole frame is blurred when the frame cannot be trusted: the predictor
  raised or returned malformed output; nobody was found; a person seen in a
  neighbouring frame has no counterpart here (a detector miss in a fast-motion
  frame is the realistic way an unblurred face gets through -- matched by box
  position, one to one, so a false detection elsewhere cannot stand in for the
  missed boxer); two boxes overlap and either face is missing or ambiguous (in a
  clinch both skeletons can claim the same head, leaving the other bare); or a face
  sits outside its own person's box.

The blur is a mosaic (block MEANS), not a Gaussian: averaging discards the
information, while a known Gaussian kernel can be partly deconvolved. A face patch
is averaged into 3x3 cells, so a head spans about two cells of flat colour.

Nothing unblurred reaches the disk: frames are decoded into memory, blurred, and
piped to the encoder; the first pass keeps only predictions. The encoder reads raw
pixels from a pipe, so no audio track or source metadata can reach the output by
construction, and the writer strips what the muxer would add on its own.

Where output may go is checked BEFORE anything is decoded, and again before the
finished file is moved into place (`check_output_location`): deny by default --
only under `~/comps-participant-data/` (docs/MACHINE-SYNC.md; it must be on an
encrypted disk) unless the operator overrides, which the report records -- and
never inside a git checkout (this repo, any worktree, the vault), a Syncthing
folder, a cloud-synced folder or a network path, override or not.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import N_JOINTS
from .punchfind import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST
from .video import VideoInfo, VideoWriter, iter_frames, probe

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_HIP, R_HIP = 11, 12
FACE_JOINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
FACE_MIN_CONF = 0.5   # a face keypoint below this does not place a patch
BODY_MIN_CONF = 0.3   # shoulders/hips used for SIZE only (USABLE_MIN_CONF in pose_yolo)
# Patch half-size in body-scale units. The patch is centred on the CONFIDENT face
# points, and when that is a single ear the centre sits on the side of the head, so
# the patch must reach a full head width past it. A head is ~0.67 body-scale units
# wide at the smallest scale (0.2 x box height), so 0.6 left a two-pixel sliver of
# face uncovered (found by the random-degradation test); 0.8 reaches past it with
# hair to spare. The cost is that a high guard is mosaicked too.
PAD = 0.8
MIN_SCALE_BOX_H = 0.2  # body scale is never below 0.2 x box height (~1.5 head heights)
FACE_SPREAD_SCALE = 3.0  # ... nor below 3 x the spread of the confident face points
TORSO_SCALE = 0.8      # ... nor below 0.8 x shoulder-to-hip length (survives a side view)
# ... and, unless the skeleton reaches the legs (`_whole_body`), nor below 1.25 x
# the box's shorter side, so the half-size reaches the box's full width: a
# head-and-shoulders box can be little wider than the head, and the face points may
# sit on its edge.
BOX_SCALE = 1.25
LEG_JOINTS = (13, 14, 15, 16)  # knees, ankles
LEG_SPAN = 0.4  # a knee or ankle this far (x box height) below the shoulders: a whole body
# Head-sized patch, used when two or more face points are confident: the head's
# width is estimated from the face points themselves, each pair scaled by its
# frontal-view ratio to head width (interpupillary ~0.42 head widths, ear to ear
# ~0.9, ...), taking the LARGEST estimate -- turning the head foreshortens some
# pairs but lengthens others, so the max over visible pairs does not collapse. The
# body-scaled patch above (PAD x 3-4 head widths) hid the guard and chest, which is
# where feints show (Brendan, 2026-10-20); the box floor still sits underneath.
HEAD_PAIR_FACTORS = {
    frozenset((L_EAR, R_EAR)): 1.3, frozenset((L_EYE, R_EYE)): 2.6,
    frozenset((NOSE, L_EYE)): 3.8, frozenset((NOSE, R_EYE)): 3.8,
    frozenset((L_EYE, L_EAR)): 4.5, frozenset((R_EYE, R_EAR)): 4.5,
    frozenset((NOSE, L_EAR)): 2.6, frozenset((NOSE, R_EAR)): 2.6,
    frozenset((L_EYE, R_EAR)): 1.7, frozenset((R_EYE, L_EAR)): 1.7,
}
HEAD_PAD = 0.9         # patch half-size, in head widths: a patch ~1.8 head widths across
HEAD_MARGIN = 0.5      # ... and never less than the farthest face point + half a head width
HEAD_MIN_BOX_H = 0.05  # an estimate under 5% of the box height is not trusted (fallback)
FACE_IN_BOX_SLACK = 0.25  # a face this far (x box size) outside its own box is not trusted
FLOOR_PAD = 0.10   # the box floor reaches 10% of the box above it and to each side ...
FLOOR_FRAC = 0.2       # ... and down to the top fifth of a whole-body box (a head is ~13-15%
FLOOR_BELOW = 0.05     # of a standing body's height) or the shoulder line if lower, plus
FLOOR_NO_SHOULDER = 0.05  # 5% of the box below that; with no confident shoulder, a third + 10% ...
FULL_BODY_ASPECT = 2.0  # ... when the box is at least twice as tall as wide; else all of it
ARMS_UP_ASPECT = 4.0   # a box this tall and narrow may be arms up: the deep floor, always
# A box this close to the frame edge may be a crop (see box_floor_region): 1% of the
# long side, ~6 model pixels at imgsz 640. A fixed 2 px was finer than the model can
# place an edge (one model pixel is ~3 px at 1080p), and a crop 3 px short lost the
# extension: 70% of the head covered (review of option c).
EDGE_FRAC = 0.01
EDGE_PX = 2            # ... and never less than this
FACE_CELLS = 3         # mosaic cells per side of a face patch
FULL_CELLS = 16        # cells along the longer side of a fully blurred frame
NEIGHBOUR_S = 0.25     # a person seen within this many seconds must be found here too
MATCH_IOU = 0.1        # ... as a box overlapping theirs by at least this much,
MATCH_TOP = 0.25       # ... whose top edge is within this much (x their height) of theirs
MAX_COORD = 1e5        # a coordinate beyond this is not a pixel position
# Face detector (deface's CenterFace, MIT): its boxes are tight, so where it and the
# keypoints AGREE on a face only that box is blurred and the guard stays visible. On
# its own it missed real faces mid-punch and in profile (own footage, 2026-10-21), so
# it never removes blur by itself.
FACE_DET_THRESH = 0.2  # deface's default
FACE_DET_PAD = 1.3     # deface's default mask scale
DET_MIN_HW = 0.5       # a detection narrower than half the measured head is not the face
FRONTAL_HALF = 0.6     # agreed frontal face: the box plus a 1.2-head-width square on the face
CENTERFACE_SHA256 = "09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc"  # deface 1.5.0

MODES = ("detected", "keypoints", "box", "full")
# Folder names of sync clients: distinctive ones match as a prefix ("OneDrive -
# Contoso"), generic ones only exactly (so "boxing" is not "Box").
CLOUD_PREFIXES = ("onedrive", "google drive", "dropbox", "icloud drive", "iclouddrive",
                  "nextcloud", "owncloud", "pcloud", "seafile", "megasync")
CLOUD_EXACT = ("my drive", "shared drives", "box", "box sync", "mega", "sync", "tresorit")
PARTICIPANT_ROOT = Path.home() / "comps-participant-data"


class UnsafeOutput(ValueError):
    """The output location is one blurred participant footage must never be written to.
    `reason` is a stable code (tests assert on it; messages may change)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class FramePlan:
    """What happens to one frame. `regions` are clipped pixel boxes (x0, y0, x1, y1),
    half-open; empty for mode "full", which blurs everything. Mode "detected": every
    person's face was agreed on by the face detector and the keypoints, so only face
    boxes are blurred; "keypoints": someone fell back to the box floor + face patch;
    "box": someone had only the box floor. `agreed` counts persons in the first case."""
    mode: str
    regions: "tuple[tuple[int, int, int, int], ...]"
    persons: int
    reason: str = ""
    agreed: int = 0


class PredictorError:
    """Stands in for a frame whose predictor call raised."""


class InvalidPrediction:
    """Stands in for a frame whose predictor output failed validation."""


# --- per-frame decision ------------------------------------------------------

def validate_prediction(pred, width: int, height: int):
    """Raise ValueError unless `pred` is (boxes (N,4), kxy (N,17,2), kconf (N,17)),
    finite and of plausible magnitude, with every box non-degenerate and overlapping
    the frame, and every confidence in [0, 1]. Returns float64 copies with boxes
    clipped to the frame. Idempotent."""
    try:
        boxes, kxy, kc = pred
        boxes = np.asarray(boxes, dtype=np.float64)
        kxy = np.asarray(kxy, dtype=np.float64)
        kc = np.asarray(kc, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("prediction is not (boxes, keypoints, confidences)") from e
    if boxes.size == 0 and kxy.size == 0 and kc.size == 0:
        return np.zeros((0, 4)), np.zeros((0, N_JOINTS, 2)), np.zeros((0, N_JOINTS))
    n = len(boxes)
    if boxes.shape != (n, 4) or kxy.shape != (n, N_JOINTS, 2) or kc.shape != (n, N_JOINTS):
        raise ValueError(f"shape mismatch: boxes {boxes.shape}, kxy {kxy.shape}, kconf {kc.shape}")
    if not (np.all(np.isfinite(boxes)) and np.all(np.isfinite(kxy)) and np.all(np.isfinite(kc))):
        raise ValueError("non-finite prediction")
    if np.abs(boxes).max() > MAX_COORD or np.abs(kxy).max() > MAX_COORD:
        raise ValueError("coordinate out of range")
    if not np.all((kc >= 0) & (kc <= 1)):
        raise ValueError("keypoint confidence outside [0, 1]")
    if not np.all((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])):
        raise ValueError("degenerate or inverted box")
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    if not np.all((clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])):
        raise ValueError("a box lies wholly outside the frame")
    return clipped, kxy, kc


def _clip_region(x0, y0, x1, y1, width, height):
    r = (max(0, math.floor(x0)), max(0, math.floor(y0)),
         min(width, math.ceil(x1)), min(height, math.ceil(y1)))
    return r if r[2] > r[0] and r[3] > r[1] else None


@dataclass(frozen=True)
class _Face:
    centre: "tuple[float, float]"
    half: float
    region: "tuple[int, int, int, int] | None"
    reason: str  # why region is None


def _whole_body(kxy, kc, sh, hp, box) -> bool:
    """Does the skeleton reach the legs, INSIDE its own box? Shoulders and hips
    confident, and a confident knee or ankle below the hips, at least LEG_SPAN x box
    height below the shoulders and not below the box (the model places joints it
    cannot see anywhere in the frame -- review P2). Only then do body measures, not
    the box, bound the head's size."""
    if not (sh and hp):
        return False
    legs = [j for j in LEG_JOINTS if kc[j] >= BODY_MIN_CONF]
    if not legs:
        return False
    bh = box[3] - box[1]
    shoulder_y = kxy[sh, 1].mean()
    hip_y = kxy[hp, 1].mean()
    low = kxy[legs, 1].max()
    return bool(shoulder_y < hip_y < low <= box[3] + 0.05 * bh
                and low - shoulder_y >= LEG_SPAN * bh)


def head_width(face, kxy, bh) -> "float | None":
    """Head width estimated from the confident face points `face`, or None when
    fewer than two are confident, or the estimate is implausibly small for the box
    (points nearly on top of each other say nothing about size)."""
    est = [float(np.linalg.norm(kxy[a] - kxy[b])) * f
           for pair, f in HEAD_PAIR_FACTORS.items() if pair <= set(face)
           for a, b in [tuple(pair)]]
    if not est or max(est) < HEAD_MIN_BOX_H * bh:
        return None
    return max(est)


def face_patch(box, kxy, kc, width, height) -> "_Face | None":
    """The face patch for one person; None when no face keypoint is confident. A
    `_Face` with region None means the keypoints exist but cannot be trusted."""
    face = [j for j in FACE_JOINTS if kc[j] >= FACE_MIN_CONF]
    if not face:
        return None
    pts = kxy[face]
    cx, cy = (float(v) for v in pts.mean(axis=0))
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if not (x0 - FACE_IN_BOX_SLACK * bw <= cx <= x1 + FACE_IN_BOX_SLACK * bw
            and y0 - FACE_IN_BOX_SLACK * bh <= cy <= y1 + FACE_IN_BOX_SLACK * bh):
        return _Face((cx, cy), 0.0, None, "face_outside_box")
    hw = head_width(face, kxy, bh)
    if hw is not None:  # head-sized: the face points measure the head directly
        half = max(HEAD_PAD * hw, float(np.abs(pts - (cx, cy)).max()) + HEAD_MARGIN * hw)
        r = _clip_region(cx - half, cy - half, cx + half, cy + half, width, height)
        return _Face((cx, cy), half, r, "" if r else "empty_region")
    # One face point (or an implausibly small head): size from the body, generously.
    scales = [MIN_SCALE_BOX_H * bh]
    if len(pts) > 1:
        spread = max(float(np.linalg.norm(a - b)) for a in pts for b in pts)
        scales.append(FACE_SPREAD_SCALE * spread)
    sh = [j for j in (L_SHOULDER, R_SHOULDER) if kc[j] >= BODY_MIN_CONF]
    hp = [j for j in (L_HIP, R_HIP) if kc[j] >= BODY_MIN_CONF]
    if len(sh) == 2:
        scales.append(float(np.linalg.norm(kxy[L_SHOULDER] - kxy[R_SHOULDER])))
    if sh and hp:
        scales.append(TORSO_SCALE * float(np.linalg.norm(kxy[sh].mean(0) - kxy[hp].mean(0))))
    if not _whole_body(kxy, kc, sh, hp, box):
        # The body is cut off or hidden, so the box may be little more than a head.
        # Body measures cannot be trusted to say otherwise: in a head-and-shoulders
        # crop the model can place hips just under the shoulders (review N2).
        scales.append(BOX_SCALE * min(bw, bh))
    scale = max(scales)
    half = PAD * scale
    # ... and never smaller than what holds every confident face point with a margin
    half = max(half, float(np.abs(pts - (cx, cy)).max()) + 0.25 * scale)
    r = _clip_region(cx - half, cy - half, cx + half, cy + half, width, height)
    return _Face((cx, cy), half, r, "" if r else "empty_region")


def _touches_edge(box, width, height) -> bool:
    """Is the (frame-clipped) box within the edge band (EDGE_FRAC of the frame's long
    side, at least EDGE_PX) of any frame edge?"""
    x0, y0, x1, y1 = box
    e = max(EDGE_PX, EDGE_FRAC * max(width, height))
    return x0 <= e or y0 <= e or x1 >= width - e or y1 >= height - e


def box_floor_region(box, width, height, kxy=None, kc=None):
    """The box patch, every person's floor: the top fifth of a whole body, or down to
    the confident-shoulder line if lower, or a third + 10% with no confident
    shoulder (+5% of the box below it, 10% above and to each side). Shoulders can
    only deepen it past the top fifth. A box that is not body-shaped (a deep
    crouch, or a crop) may be mostly head, and its top fifth misses the chin, so the
    WHOLE box is blurred instead. A body-shaped box touching the frame edge may be a
    close-up crop: 1.5 box widths from the top."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    top = FLOOR_FRAC * bh
    # The top fifth holds the head only while the box top is the head. With arms
    # raised the box top is the gloves and the head sits at 0.3-0.4 of the box: a
    # wrong skeleton then left it bare (review R1). So the floor also reaches the
    # shoulder line, and with no confident shoulder, the old third + 10%.
    # The LOWER shoulder, not the mean: one shoulder wrongly placed high (on an
    # elbow) pulled the mean line halfway up (47% of the head covered).
    sh = [] if kc is None else [j for j in (L_SHOULDER, R_SHOULDER) if kc[j] >= BODY_MIN_CONF]
    if sh:
        top = max(top, float(kxy[sh, 1].max()) - y0)
    else:
        top = max(top, bh / 3 + FLOOR_NO_SHOULDER * bh)
    # ... but a skeleton fitted up a raised arm puts its shoulders up there too: one
    # failure, not two (0%). Raised arms make the box tall and narrow (arms up 5.6:1,
    # a referee raising one 4.7:1; guard and punches 3.4:1 and below), so from 4:1 the
    # deep floor applies whatever the keypoints say.
    if bh >= ARMS_UP_ASPECT * bw:
        top = max(top, bh / 3 + FLOOR_NO_SHOULDER * bh)
    if _touches_edge(box, width, height):
        # A head is no wider than its box and ~1.25x as tall as wide, so a
        # head-to-chest crop that passes the 2:1 test still has its chin inside
        # 1.5 widths (review P3 96%, H1 96% with a third alone). Only at the frame
        # edge, where real close-up crops come from: on a whole boxer in guard
        # (~0.4x as wide as tall) it would mosaic head to hips (Brendan, option c).
        top = max(top, 1.5 * bw)
    bottom = y0 + top + FLOOR_BELOW * bh if bh >= FULL_BODY_ASPECT * bw else y1
    return _clip_region(x0 - FLOOR_PAD * bw, y0 - FLOOR_PAD * bh,
                        x1 + FLOOR_PAD * bw, bottom, width, height)


def _intersect(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _inside(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _slack(box):
    """`box` grown by FACE_IN_BOX_SLACK a side: where its owner's face may sit."""
    x0, y0, x1, y1 = box
    dx, dy = FACE_IN_BOX_SLACK * (x1 - x0), FACE_IN_BOX_SLACK * (y1 - y0)
    return (x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def _full(persons: int, reason: str) -> FramePlan:
    return FramePlan("full", (), persons, reason)


def plan_frame(pred, width: int, height: int, expected_persons: int = 0,
               neighbour_missing: bool = False, face_dets=None) -> FramePlan:
    """Decide how one frame is blurred. `pred` is the predictor's output, or a
    `PredictorError` instance. `neighbour_missing`: a person in a nearby frame has no
    counterpart here (`missing_vs_neighbours`). `expected_persons`: the operator's
    --expect-persons; fewer here is a miss. `face_dets`: a face detector's boxes for
    this frame (`validate_face_dets`), or None for none."""
    if face_dets is not None:
        try:
            face_dets = validate_face_dets(face_dets, width, height)
        except ValueError:
            face_dets = None  # no detections: less trust, never less blur
    if isinstance(pred, PredictorError):
        return _full(0, "predictor_error")
    if isinstance(pred, InvalidPrediction):
        return _full(0, "invalid_prediction")
    try:
        boxes, kxy, kc = validate_prediction(pred, width, height)
    except ValueError:
        return _full(0, "invalid_prediction")
    n = len(boxes)
    try:
        return _plan_people(boxes, kxy, kc, width, height, expected_persons, neighbour_missing,
                            face_dets)
    except ArithmeticError:  # overflow in a size computation: no partial plan, blur it all
        return _full(n, "invalid_prediction")


def _plan_people(boxes, kxy, kc, width, height, expected_persons, neighbour_missing,
                 face_dets=None):
    n = len(boxes)
    if n == 0:
        return _full(0, "no_person")
    if n < expected_persons:
        return _full(n, "fewer_persons_than_expected")
    if neighbour_missing:
        return _full(n, "person_missing_vs_neighbours")
    faces = [face_patch(boxes[i], kxy[i], kc[i], width, height) for i in range(n)]
    for f in faces:
        if f is not None and f.region is None:
            return _full(n, f.reason)
    for i in range(n):
        for j in range(i + 1, n):
            fi, fj = faces[i], faces[j]
            if fi is None or fj is None:
                if _intersect(boxes[i], boxes[j]):
                    return _full(n, "overlap_without_face_keypoints")
                # One face, and it sits on or near the OTHER person's box: it may be
                # their head, claimed by the wrong skeleton, leaving its owner's head
                # bare (review P1). With no second face there is nothing to tell them apart.
                if ((fi is not None and _inside(fi.centre, _slack(boxes[j])))
                        or (fj is not None and _inside(fj.centre, _slack(boxes[i])))):
                    return _full(n, "ambiguous_faces")
                continue
            # Two skeletons may claim one head, whether or not their boxes touch: a
            # face may sit up to FACE_IN_BOX_SLACK outside its own box (review N1).
            # Trust the pair only if the faces are distinct heads, each outside the
            # other person's box.
            if (math.dist(fi.centre, fj.centre) <= max(fi.half, fj.half)
                    or _inside(fi.centre, boxes[j]) or _inside(fj.centre, boxes[i])):
                return _full(n, "ambiguous_faces")
    # A person whose face TWO independent models agree on (a face-detector box and
    # YOLO's confident face keypoints) gets just that face box, padded. Everyone else
    # gets the box FLOOR with a face patch on top: no keypoint can shrink what the box
    # alone would blur (blocker report, option 1), and no lone detector can either.
    dets = np.zeros((0, 5)) if face_dets is None else face_dets
    used, regions, any_box, agreed = set(), [], False, 0
    for i in range(n):
        k = _agreeing_face(faces[i], boxes[i], kxy[i], kc[i], dets, used)
        if k is not None:
            used.add(k)
            r = det_region(dets[k], width, height)
            if r is None:
                return _full(n, "empty_region")
            # Never the detector's box alone: with only one side of the face
            # confident, a box shifted that way "agrees" and misses the other side
            # (79 of 1000 random scenes). So add the keypoints' own estimate: a tight
            # square on a frontal face (nose and both eyes), else the head patch.
            regions += [r, _frontal_square(boxes[i], kxy[i], kc[i], width, height)
                        or faces[i].region]
            bw_i, bh_i = boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]
            if bh_i >= ARMS_UP_ASPECT * bw_i:
                # Arms up: the deep floor holds whatever the keypoints AND the face
                # detector say -- a glove can fool both (review D1, 0% otherwise).
                floor = box_floor_region(boxes[i], width, height, kxy[i], kc[i])
                if floor is None:
                    return _full(n, "empty_region")
                regions.append(floor)
            agreed += 1
            continue
        r = box_floor_region(boxes[i], width, height, kxy[i], kc[i])
        if r is None:  # cannot happen for a validated box; fail closed regardless
            return _full(n, "empty_region")
        regions.append(r)
        if faces[i] is not None:
            regions.append(faces[i].region)
        else:
            any_box = True
    for k in range(len(dets)):  # a face no person claimed is blurred all the same
        if k not in used:
            r = det_region(dets[k], width, height)
            if r is not None:
                regions.append(r)
    mode = "detected" if agreed == n else ("box" if any_box else "keypoints")
    return FramePlan(mode, tuple(regions), n, agreed=agreed)


def validate_face_dets(dets, width: int, height: int) -> np.ndarray:
    """Face-detector output as (K, 5) float64 [x0, y0, x1, y1, score], clipped to the
    frame; raise ValueError on anything else (the caller then uses no detections,
    which only means less trust, never less blur)."""
    try:
        d = np.asarray(dets, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("face detections are not an array") from e
    if d.size == 0:
        return np.zeros((0, 5))
    if d.ndim != 2 or d.shape[1] < 5 or not np.all(np.isfinite(d[:, :5])):
        raise ValueError(f"bad face detections {d.shape}")
    d = d[:, :5].copy()
    if np.abs(d[:, :4]).max() > MAX_COORD or not np.all((d[:, 4] >= 0) & (d[:, 4] <= 1)):
        raise ValueError("face detection out of range")
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, width)
    d[:, [1, 3]] = d[:, [1, 3]].clip(0, height)
    return d[(d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])]


def det_region(det, width, height):
    """A detected face box grown FACE_DET_PAD times about its centre (deface's mask
    scale, as a rectangle, which covers its ellipse)."""
    x0, y0, x1, y1 = det[:4]
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    return _clip_region(cx - FACE_DET_PAD * hw, cy - FACE_DET_PAD * hh,
                        cx + FACE_DET_PAD * hw, cy + FACE_DET_PAD * hh, width, height)


def _frontal_square(box, kxy, kc, width, height):
    """With nose AND both eyes confident (a frontal face, so their mean is the face's
    centre), a square FRONTAL_HALF head widths either side of it; else None."""
    pts = [NOSE, L_EYE, R_EYE]
    if not all(kc[j] >= FACE_MIN_CONF for j in pts):
        return None
    hw = head_width(pts, kxy, box[3] - box[1])
    if hw is None:
        return None
    cx, cy = kxy[pts].mean(axis=0)
    h = FRONTAL_HALF * hw
    return _clip_region(cx - h, cy - h, cx + h, cy + h, width, height)


def _agreeing_face(face, box, kxy, kc, dets, used):
    """Index of the detection that agrees with this person's face, or None. Agreement:
    two or more confident face keypoints, so the head can be measured (`head_width`);
    every confident nose and eye point INSIDE the detection box itself (a box beside
    the face, or on a glove, does not contain them); the detection at least
    DET_MIN_HW of the measured head wide (a box on an eye is not the face); and its
    centre within the person's box (with the usual slack)."""
    if face is None or face.region is None or len(dets) == 0:
        return None
    pts = [j for j in FACE_JOINTS if kc[j] >= FACE_MIN_CONF]
    inner = [NOSE, L_EYE, R_EYE]
    # Frontal faces only (nose AND both eyes): a three-quarter or profile face can be
    # under-measured, and a tight box then left the chin bare -- the floor catches it.
    if not all(kc[j] >= FACE_MIN_CONF for j in inner):
        return None
    hw = head_width(pts, kxy, box[3] - box[1])
    if hw is None:
        return None
    wrists = [kxy[j] for j in (L_WRIST, R_WRIST) if kc[j] >= BODY_MIN_CONF]
    best, best_score = None, -1.0
    for k, (x0, y0, x1, y1, s) in enumerate(dets):
        if k in used:
            continue
        if not all(_inside(kxy[j], (x0, y0, x1, y1)) for j in inner):
            continue
        if not _inside(((x0 + x1) / 2, (y0 + y1) / 2), _slack(box)):
            continue
        if (x1 - x0) < DET_MIN_HW * hw:
            continue
        # A glove fools BOTH models the same way (it looks like a head), so a box
        # with a confident wrist inside it is not trusted to be the face (review D2).
        if any(_inside(wr, (x0, y0, x1, y1)) for wr in wrists):
            continue
        if s > best_score:
            best, best_score = k, s
    return best


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _all_matched(theirs: np.ndarray, ours: np.ndarray) -> bool:
    """Can every box in `theirs` be paired with a DISTINCT box in `ours` at IoU >=
    MATCH_IOU? Greedy by IoU, so one box cannot vouch for two people."""
    if len(theirs) == 0:
        return True
    if len(ours) < len(theirs):
        return False
    iou = _iou_matrix(theirs, ours)
    # A box that lost the head (legs only) can overlap the whole person well; the
    # head is what matters, so the top edges must agree too.
    th = theirs[:, 3] - theirs[:, 1]
    iou[np.abs(ours[None, :, 1] - theirs[:, None, 1]) > MATCH_TOP * th[:, None]] = 0.0
    for _ in range(len(theirs)):
        k = int(np.argmax(iou))
        r, c = divmod(k, iou.shape[1])
        if iou[r, c] < MATCH_IOU:
            return False
        iou[r, :] = -1
        iou[:, c] = -1
    return True


def missing_vs_neighbours(boxes_per_frame, window: int) -> np.ndarray:
    """Per frame: does some person boxed within `window` frames either side lack a
    counterpart box here? (A frame with unusable predictions contributes no boxes;
    it is blurred whole on its own account.)"""
    f = len(boxes_per_frame)
    out = np.zeros(f, bool)
    for t in range(f):
        for u in range(max(0, t - window), min(f, t + window + 1)):
            if u != t and not _all_matched(boxes_per_frame[u], boxes_per_frame[t]):
                out[t] = True
                break
    return out


# --- the blur ------------------------------------------------------------------

def _mosaic(a: np.ndarray, ny: int, nx: int) -> None:
    """In place: replace each of ny x nx cells of `a` by its mean colour."""
    h, w = a.shape[:2]
    ys = np.linspace(0, h, max(1, min(ny, h)) + 1).round().astype(int)
    xs = np.linspace(0, w, max(1, min(nx, w)) + 1).round().astype(int)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            cell = a[y0:y1, x0:x1]
            cell[...] = cell.reshape(-1, a.shape[2]).mean(axis=0).round().astype(a.dtype)


def apply_plan(frame: np.ndarray, plan: FramePlan) -> np.ndarray:
    """A blurred copy of `frame`; the input is not modified."""
    out = frame.copy()
    h, w = out.shape[:2]
    if plan.mode == "full":
        cell = math.ceil(max(h, w) / FULL_CELLS)
        _mosaic(out, math.ceil(h / cell), math.ceil(w / cell))
    elif plan.mode in ("detected", "keypoints", "box"):
        for x0, y0, x1, y1 in plan.regions:
            _mosaic(out[y0:y1, x0:x1], FACE_CELLS, FACE_CELLS)
    else:
        raise ValueError(f"unknown blur mode {plan.mode!r}")
    return out


def blurred_mask(plan: FramePlan, height: int, width: int) -> np.ndarray:
    """Pixels a plan blurs (for tests and reporting)."""
    m = np.zeros((height, width), bool)
    if plan.mode == "full":
        m[:] = True
    for x0, y0, x1, y1 in plan.regions:
        m[y0:y1, x0:x1] = True
    return m


# --- where output may go -----------------------------------------------------------

def _norm(p: Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _under(p: Path, root: Path) -> bool:
    a, b = _norm(p), _norm(root)
    return a == b or a.startswith(b.rstrip("\\/") + os.sep)


def _strip_verbatim(p: Path) -> Path:
    r"""`resolve()` returns \\?\C:\... for some names (a trailing space); compare the
    plain form. \\?\UNC\... stays a network path."""
    s = str(p)
    if s.startswith("\\\\?\\") and not s.upper().startswith("\\\\?\\UNC\\"):
        return Path(s[4:])
    return p


def _is_network(p: Path) -> bool:
    s = str(p)
    if s.startswith("\\\\") and not s.startswith("\\\\?\\"):
        return True
    if s.upper().startswith("\\\\?\\UNC\\"):
        return True
    if os.name == "nt" and p.drive:
        import ctypes
        DRIVE_REMOTE = 4
        return ctypes.windll.kernel32.GetDriveTypeW(p.drive + "\\") == DRIVE_REMOTE
    return False


def syncthing_folders(configs) -> "list[Path]":
    """Folder paths from Syncthing config files. A config that exists but cannot be
    read is an error (fail closed): we cannot tell what it shares."""
    out = []
    for cfg in configs:
        cfg = Path(cfg)
        if not cfg.exists():
            continue
        try:
            root = ET.parse(cfg).getroot()
        except (ET.ParseError, OSError) as e:
            raise UnsafeOutput("syncthing_config_unreadable",
                               f"cannot read the Syncthing config {cfg.name}; "
                               "refusing to guess what it syncs") from e
        for f in root.iter("folder"):
            path = f.get("path")
            if path:
                p = Path(os.path.expanduser(path))
                out.append((p if p.is_absolute() else cfg.parent / p).resolve())
    return out


def default_forbidden_roots() -> "list[Path]":
    here = Path(__file__).resolve().parents[2]  # this repo (or worktree) root
    roots = [here, Path.home() / "ai-os-vault"]
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    return [r.resolve() for r in roots]


def default_syncthing_configs() -> "list[Path]":
    local = os.environ.get("LOCALAPPDATA")
    cands = [Path(local) / "Syncthing" / "config.xml"] if local else []
    cands += [Path.home() / ".local" / "state" / "syncthing" / "config.xml",
              Path.home() / ".config" / "syncthing" / "config.xml"]
    return cands


_DEFAULT = object()


def check_output_location(out, *, input_path=None, overwrite: bool = False,
                          allowed_root=_DEFAULT, forbidden_roots=None,
                          syncthing_configs=None) -> Path:
    """Return the canonical output path, or raise UnsafeOutput (with a `reason` code).

    Deny by default: the output must be under `allowed_root` (PARTICIPANT_ROOT unless
    given; None lifts this rule, and only this rule -- the caller records that). Then
    the denylist, which no override lifts. Checked on both the path as given and its
    resolved form (links and junctions followed), so a link out of, or into, a
    forbidden folder is refused either way."""
    raw = Path(os.path.abspath(Path(out).expanduser()))
    resolved = raw.resolve()
    if resolved.suffix.lower() != ".mp4":
        raise UnsafeOutput("not_mp4", "output must be an .mp4 file")
    if _is_network(raw) or _is_network(resolved):
        raise UnsafeOutput("network_path", "output on a network path")
    resolved = _strip_verbatim(resolved)
    if not resolved.parent.is_dir():
        raise UnsafeOutput("no_folder", "output folder does not exist (create it deliberately, "
                           "on an encrypted disk, under ~/comps-participant-data/)")
    if allowed_root is _DEFAULT:
        allowed_root = PARTICIPANT_ROOT
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        if not (_under(raw, root) and _under(resolved, root)):
            raise UnsafeOutput("outside_participant_root",
                               f"output must be under {Path(allowed_root).name}/ "
                               "(override only for footage that is not a participant's)")
    if forbidden_roots is None:
        forbidden_roots = default_forbidden_roots()
    synced = syncthing_folders(default_syncthing_configs() if syncthing_configs is None
                               else syncthing_configs)
    for p in (raw, resolved):
        for root in forbidden_roots:
            if _under(p, Path(root)):
                raise UnsafeOutput("forbidden_root",
                                   f"output inside a forbidden folder ({Path(root).name})")
        for root in synced:
            if _under(p, root):
                raise UnsafeOutput("syncthing_folder", f"output inside a Syncthing folder ({root.name})")
        for part in p.parent.parts:
            name = part.casefold().strip()
            if name.startswith(CLOUD_PREFIXES) or name in CLOUD_EXACT:
                raise UnsafeOutput("cloud_folder", f"output inside a cloud-synced folder ({part})")
        for anc in (p.parent, *p.parent.parents):
            if (anc / ".git").exists():
                raise UnsafeOutput("git_checkout", f"output inside a git checkout ({anc.name})")
            if (anc / ".obsidian").exists():
                raise UnsafeOutput("obsidian_vault", f"output inside an Obsidian vault ({anc.name})")
            if any(anc.glob(".stfolder*")):
                raise UnsafeOutput("syncthing_marker", f"output inside a Syncthing folder ({anc.name})")
    if input_path is not None and _strip_verbatim(Path(input_path).resolve()) == resolved:
        raise UnsafeOutput("is_input", "output would overwrite the input")
    if resolved.exists() and not overwrite:
        raise UnsafeOutput("exists", f"{resolved.name} exists (pass --overwrite to replace it)")
    return resolved


# --- the whole video ---------------------------------------------------------------

def _ranges(idx) -> "list[list[int]]":
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report_path_for(out: Path) -> Path:
    return out.with_name(out.stem + ".blur.json")


def load_centerface(threshold: float = FACE_DET_THRESH, sha256: str = CENTERFACE_SHA256):
    """deface's CenterFace face detector as detect(frame_rgb) -> (K, 5) boxes+scores,
    refused unless the model file's sha256 is the pinned one (it is parsed by OpenCV's
    ONNX importer; same check-then-load caveat as the YOLO weights)."""
    try:
        from deface.centerface import CenterFace, default_onnx_path
    except ImportError as e:
        raise RuntimeError("deface is not installed (pip install --no-deps deface==1.5.0)") from e
    if sha256_file(Path(default_onnx_path)) != sha256:
        raise ValueError("centerface.onnx does not match the pinned sha256; not loading it")
    cf = CenterFace(onnx_path=default_onnx_path, backend="opencv")

    def detect(frame_rgb):
        return cf(frame_rgb, threshold=threshold)[0]
    return detect


def blur_video(video, out: Path, predict, *, info: "VideoInfo | None" = None,
               expect_persons: int = 0, model_name: str = "", progress=None,
               writer_factory=VideoWriter, recheck=None, report_extra=None,
               detect_faces=None, face_model: str = "") -> dict:
    """Two passes over `video`: predict on every frame (predictions only are kept),
    then blur and encode. `detect_faces(frame) -> (K, 5)`, if given, is a face
    detector whose boxes replace the box floor for a person only where they agree
    with the keypoints. `out` must already have passed `check_output_location`;
    `recheck(out)` re-runs that check just before the finished file is moved into
    place. Writes `out` and its `.blur.json` only on success; on any failure neither
    exists and nothing partial is left behind. Returns the report.

    The report names no file: the source is identified by its sha256 (a clip may be
    named after the person in it, and the annotator is never told names)."""
    video = Path(video).resolve()
    out = Path(out)
    info = info or probe(video)
    w, h = info.width, info.height
    source_sha = sha256_file(video)

    preds, boxes, fdets = [], [], []
    face_errors = 0
    for i, frame in enumerate(iter_frames(video, info)):
        try:
            p = validate_prediction(predict(frame), w, h)
            boxes.append(p[0])
        except ValueError:
            p = InvalidPrediction()
            boxes.append(np.zeros((0, 4)))
        except Exception:  # fail closed: a frame the predictor choked on is fully blurred
            p = PredictorError()
            boxes.append(np.zeros((0, 4)))
        preds.append(p)
        d = None
        if detect_faces is not None:
            try:
                d = validate_face_dets(detect_faces(frame), w, h)
            except Exception:  # no detections: less trust, never less blur
                face_errors += 1
        fdets.append(d)
        if progress and (i + 1) % 250 == 0:
            progress(f"pass 1: {i + 1} frames")
    if not preds:
        raise ValueError("no frames could be decoded")
    window = max(1, round(info.fps * NEIGHBOUR_S))
    missing = missing_vs_neighbours(boxes, window)
    plans = [plan_frame(p, w, h, expect_persons, bool(m), fd)
             for p, m, fd in zip(preds, missing, fdets)]
    del preds, boxes, fdets

    report_path = report_path_for(out)
    tmp = out.with_name(f".{out.stem}.{secrets.token_hex(6)}.partial.mp4")
    tmp_report = report_path.with_name(f".{report_path.name}.{secrets.token_hex(6)}.partial")
    placed = False
    try:
        n = 0
        with writer_factory(tmp, w, h, info.fps_frac, private=True) as wr:
            for frame in iter_frames(video, info):
                if n >= len(plans):
                    raise RuntimeError("second decode produced more frames than the first")
                wr.write(apply_plan(frame, plans[n]))
                n += 1
                if progress and n % 250 == 0:
                    progress(f"pass 2: {n} frames")
            if n != len(plans):
                raise RuntimeError(f"second decode produced {n} frames, the first {len(plans)}")
        if sha256_file(video) != source_sha:  # the plans were made for other pixels
            raise RuntimeError("the input changed while it was being blurred")
        modes = Counter(p.mode for p in plans)
        full_idx = [i for i, p in enumerate(plans) if p.mode == "full"]
        report = {
            **(report_extra or {}),  # first, so it can never overwrite a core field
            "tool": "blur_faces", "report_version": 4,
            "source_sha256": source_sha, "output_sha256": sha256_file(tmp),
            "model": model_name, "width": w, "height": h, "fps": info.fps,
            "variable_frame_rate": info.vfr,
            "frames": len(plans),
            "frames_by_mode": {m: modes.get(m, 0) for m in MODES},
            "frames_fully_blurred": len(full_idx),
            "full_blur_reasons": dict(sorted(Counter(p.reason for p in plans
                                                     if p.mode == "full").items())),
            "fully_blurred_ranges": _ranges(full_idx),
            "persons_per_frame": {str(k): v for k, v in
                                  sorted(Counter(p.persons for p in plans).items())},
            "per_frame": {"persons": [p.persons for p in plans],
                          "mode": "".join(p.mode[0] for p in plans)},
            "face_detector": face_model or None,
            "face_detector_errors": face_errors,
            "persons_face_agreed": sum(p.agreed for p in plans if p.mode != "full"),
            "persons_fallback": sum(p.persons - p.agreed for p in plans if p.mode != "full"),
            "params": {"face_min_conf": FACE_MIN_CONF, "pad": PAD, "face_cells": FACE_CELLS,
                       "full_cells": FULL_CELLS, "neighbour_frames": window,
                       "match_iou": MATCH_IOU, "match_top": MATCH_TOP,
                       "box_floor": True, "edge_frac": EDGE_FRAC,
                       "face_det_pad": FACE_DET_PAD, "det_min_hw": DET_MIN_HW,
                       "expect_persons": expect_persons},
            "audio": "none", "metadata": "stripped",
        }
        tmp_report.write_text(json.dumps(report, indent=1), encoding="utf-8")
        if recheck is not None:
            recheck(out)
        os.replace(tmp, out)
        placed = True
        os.replace(tmp_report, report_path)
        return report
    except BaseException:
        if placed:  # never a video without its report, nor a stale report left by --overwrite
            out.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)
        tmp_report.unlink(missing_ok=True)
