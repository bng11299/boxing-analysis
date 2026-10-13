"""Classify punches found in continuous pose: a window of skeleton around each
detected punch, in, ST-GCN++ punch type and confidence, out.

The classifier was trained on BoxingVI's pre-cut clips, so what it is shown has to
look like one of those clips, in time and in space:

* time: a BoxingVI clip is a labelled punch, start to end, at the source's own
  24-30 fps. The pipeline reads how long a punch takes IN THIS CLIP off the
  wrist-speed curve: each punch's extent from the width of its speed peak
  (`punchfind.extent`, ratios from `training_extent_ratios`), then the median over
  the clip (`clip_cut`). Every punch is read over that one length, placed around
  its own peak (`clip_window`), and `cut_span` resamples those frames to the
  training frame rate -- footage at 60 fps is converted explicitly, not passed
  through at twice the frames per punch (O54).
  Why per clip and not per punch (vault ADR-0014, amended 2026-10-13): sizing each
  punch from its own peak width is worse on normal-tempo footage and no better on
  slow footage. V5 + V9, 742 punches: per clip 0.826, the old fixed 0.375 s window
  0.806, per punch 0.774 (per punch minus per clip -0.053 [-0.075, -0.030]). V6
  second half: per clip 0.631, per punch 0.608, old window 0.500 (per punch minus
  per clip -0.022 [-0.054, +0.010], a tie; at the finder's own detections 0.678
  against 0.684, also a tie). A boxer's tempo is steady within a clip; one punch's
  peak width is a noisy reading of it. `cut_window`, the fixed-length window,
  stays as the baseline these are measured against.
  THE CHOICE WAS MADE LOOKING AT THOSE NUMBERS: V5, V9 and V6's second half are no
  longer clean tests of the per-clip cut. Its first clean test is labelled footage
  it has not seen (own footage, once labelled).
* space: BoxingVI coordinates are pixels of a 1920x1080 frame divided by the
  frame size. Phone footage is portrait, and the boxer is nearer or farther. The
  window is scaled so the torso is the training median length in 1080p pixels,
  centred so the hips sit at the training median position, then divided by the
  same 1920x1080. Position and scale are thereby normalised; posture and motion
  are not touched. (Measured on V6: harmless, 0.708 against 0.712.)

The training statistics (`training_geometry`, `training_extent_ratios`) come from
the TRAINING clips only.

Trust boundaries: the classifier checkpoint is a pickle, so it is opened only
with ``weights_only=True``, after its size and sha256 match the manifest. The
manifest is JSON written by train_classifier.py; it is validated field by field
as untrusted. Like the YOLO weights, an attacker who can write the weights
directory can rewrite both files -- keep that directory user-private.

Known approximations, stated rather than hidden:
* the lead-in is the wrist-speed peak inside a short training clip, where the
  smoothing and gradient are one-sided at the clip's edges; in continuous pose
  `punchfind.detect` sees both sides. Peaks near a clip's edge can differ by a frame.
* the torso length is one median per video, not per window: right for a boxer
  working on one spot, wrong for one who walks toward or away from the phone.
  (A per-window torso would also shrink as the body turns into a hook.)

What this module does not do: it does not decide WHEN a punch happened (that is
`punchfind`), and it does not make the punch type agree with the finder's hand.
The two are independent readings of the same skeleton, and their agreement is
reported as a check rather than enforced.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .data import CLASSES, MAX_FRAMES, N_CLASSES, N_JOINTS, NATIVE_FPS, SOURCE_SIZE, Track, pad_or_trim
from .punchfind import (L_HIP, L_SHOULDER, MIN_CONF, R_HIP, R_SHOULDER, VARIANTS, Extent,
                        body_scale, extent, peak_width, wrist_signal)

MANIFEST_VERSION = 1
MAX_CLASSIFIER_BYTES = 64 * 1024 * 1024   # the real file is ~5.5 MB
MAX_MANIFEST_BYTES = 64 * 1024
LEAD_CLASSES = frozenset({"jab", "lead_hook", "lead_uppercut"})
REAR_CLASSES = frozenset({"cross", "rear_hook", "rear_uppercut"})
# Which COCO side leads. Orthodox boxers lead with the left hand.
STANCES = {"orthodox": "left", "southpaw": "right"}
# A training clip has a pose in every one of its real frames (only its padding
# is empty), so a window must show the boxer in at least half of its frames to
# look like one; below that the resampler would be inventing the punch from a
# frame or two. Never fewer than 2, the corpus loader's own floor.
MIN_OBSERVED_FRAMES = 2
MIN_OBSERVED_SHARE = 0.5
# prepare_input refuses coordinates beyond these; a window outside them is
# reported as unclassifiable instead of failing the whole batch.
COORD_LO, COORD_HI = -0.5, 1.5
# The classifier is shown EVERY joint the pose model detected, however unsure. It was trained
# on AlphaPose skeletons, which carry all 17 joints, and a skeleton with joints blanked out is
# something it has never seen. Measured on the held-out punches of V5 and V9, the same frames
# read by YOLO11-pose (results/pose_switch.json, 742 punches): AlphaPose 0.896; YOLO with every
# detected joint 0.889 (-0.007 [-0.009, +0.024]); YOLO with joints under confidence 0.3 blanked,
# as this module did until 2026-10-12, 0.348. That rule blanked about 5% of the joints there,
# mostly a hidden ear or eye. The punch FINDER keeps its 0.3 rule: a guessed wrist that jumps
# reads as a punch (punchfind.MIN_CONF). Only what the classifier is shown changed.
CLASSIFY_MIN_CONF = 0.0

assert LEAD_CLASSES | REAR_CLASSES == set(CLASSES) and not LEAD_CLASSES & REAR_CLASSES


@dataclass(frozen=True)
class WindowGeometry:
    """What a training clip looks like, in seconds and in 1080p pixels."""

    duration_s: float     # median real length of a training clip
    lead_in_s: float      # median time from clip start to its wrist-speed peak
    fps: float            # median native frame rate over training clips
    torso_px: float       # median shoulder-to-hip length, pixels of SOURCE_SIZE
    hip_x: float          # median hip-centre x, as a fraction of SOURCE_SIZE width
    hip_y: float          # ... and y, as a fraction of height
    variant: str          # the wrist-speed variant the peak was measured with
    n_clips: int          # training clips the medians came from

    def __post_init__(self) -> None:
        for k in ("duration_s", "lead_in_s", "fps", "torso_px", "hip_x", "hip_y"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"geometry {k} must be a finite number, got {v!r}")
        if not (self.duration_s > 0 and self.fps > 0 and self.torso_px > 1):
            raise ValueError("geometry duration, fps and torso must be positive")
        if not 0 <= self.lead_in_s <= self.duration_s:
            raise ValueError("geometry lead-in must lie inside the clip")
        if not (0 < self.hip_x < 1 and 0 < self.hip_y < 1):
            raise ValueError("geometry hip centre must lie inside the frame")
        if self.variant not in VARIANTS:
            raise ValueError(f"geometry variant must be one of {VARIANTS}")
        if isinstance(self.n_clips, bool) or not isinstance(self.n_clips, int) or self.n_clips < 1:
            raise ValueError("geometry n_clips must be a positive integer")
        if not MIN_OBSERVED_FRAMES <= self.n_frames <= MAX_FRAMES:
            # No training clip is longer than the corpus's 25 frames, and a manifest
            # asking for more would also size every window from untrusted input.
            raise ValueError(f"geometry window must be 2..{MAX_FRAMES} frames, got {self.n_frames}")

    @property
    def n_frames(self) -> int:
        return max(1, int(round(self.duration_s * self.fps)))


# ------------------------------------------------------- training geometry ---

def _clip_track(kp: np.ndarray, fps: float, source: str) -> "Track | None":
    """A training clip's observed frames as a pixel-space Track, validity as
    confidence (so a dropped joint at (0, 0) is never differenced)."""
    mask = kp[..., 2] > 0
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size < 3:
        return None
    w, h = SOURCE_SIZE
    xy = kp[rows, :, :2].astype(np.float64) * np.array([w, h])
    return Track(frame_ids=np.arange(rows.size), xy=xy, fps=fps, source=source,
                 conf=mask[rows].astype(np.float64))


def training_geometry(clips, variant: str) -> WindowGeometry:
    """Medians over the TRAINING clips only. Clips whose body or wrist signal
    cannot be measured are skipped and do not move the medians."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    dur, lead, fps, torso, hx, hy = [], [], [], [], [], []
    w, h = SOURCE_SIZE
    for c in clips:
        if c.source not in NATIVE_FPS:
            raise ValueError(f"clip from {c.source!r} has no recorded native frame rate")
        f = NATIVE_FPS[c.source]
        tr = _clip_track(c.keypoints, f, c.source)
        if tr is None:
            continue
        try:
            scale = body_scale(tr)
            sig = wrist_signal(tr, variant)
        except ValueError:   # no usable shoulders and hips in this clip
            continue
        best = sig.max(axis=1)
        if not best.max() > 0:
            continue
        n = len(tr.frame_ids)
        dur.append(n / f)
        # np.gradient is centred, so best[i] is the speed AT frame i: the peak is
        # at frame i, exactly where punchfind.detect reports a peak in continuous pose.
        lead.append(int(np.argmax(best)) / f)
        fps.append(f)
        torso.append(scale)
        ok = (tr.conf[:, [L_HIP, R_HIP]] > 0).all(axis=1)
        if ok.any():
            mid = (tr.xy[ok, L_HIP] + tr.xy[ok, R_HIP]) / 2
            hx.append(float(np.median(mid[:, 0])) / w)
            hy.append(float(np.median(mid[:, 1])) / h)
    if len(dur) < 10 or len(hx) < 10:
        raise ValueError(f"only {len(dur)} measurable training clips: too few for a geometry")
    duration = float(np.median(dur))
    return WindowGeometry(
        duration_s=duration, lead_in_s=min(float(np.median(lead)), duration),
        fps=float(np.median(fps)), torso_px=float(np.median(torso)),
        hip_x=float(np.median(hx)), hip_y=float(np.median(hy)),
        variant=variant, n_clips=len(dur))


def training_extent_ratios(clips, variant: str, frac: float) -> dict:
    """How many peak-widths a labelled punch runs before and after its wrist-speed
    peak: medians (taken separately) over the TRAINING clips whose peak is whole. A
    clip whose speed is still above the level at its first or last frame has an
    unknown width and is left out, and counted: the clip starts or ends mid-motion
    (the annotator's cut, or the previous punch's retraction).

    Leaving those out is a selection: a clip is kept only when its lead and tail
    are each longer than their half-width, which removes the smallest ratios, so
    the medians lean long. How far the kept clips differ from all of them is
    returned (`duration_s_median_*`, `censored_by_class`) so it can be seen."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    lead, tail, dur_kept, dur_all, skipped = [], [], [], [], 0
    by_class = {c: [0, 0] for c in CLASSES}   # [censored, measurable]
    for c in clips:
        if c.source not in NATIVE_FPS:
            raise ValueError(f"clip from {c.source!r} has no recorded native frame rate")
        f = NATIVE_FPS[c.source]
        tr = _clip_track(c.keypoints, f, c.source)
        sig = None
        if tr is not None:
            try:
                sig = wrist_signal(tr, variant)
            except ValueError:
                pass
        if sig is None or not sig.max() > 0:
            skipped += 1   # too short, no usable body, or no wrist speed at all
            continue
        p = int(np.argmax(sig.max(axis=1)))
        w, cut = peak_width(tr, sig, p, int(np.argmax(sig[p])), frac)
        n = len(tr.frame_ids)
        dur_all.append(n / f)
        by_class[CLASSES[c.label]][1] += 1
        if cut:
            by_class[CLASSES[c.label]][0] += 1
            continue
        dur_kept.append(n / f)
        lead.append((p / f) / w)
        tail.append(((n - 1 - p) / f) / w)
    if len(lead) < 10:
        raise ValueError(f"only {len(lead)} training clips with a whole speed peak: too few")
    return {"frac": float(frac), "lead_widths": float(np.median(lead)), "tail_widths": float(np.median(tail)),
            "n_clips": len(lead), "n_censored": len(dur_all) - len(lead), "n_skipped": skipped, "variant": variant,
            "duration_s_median_kept": float(np.median(dur_kept)), "duration_s_median_all": float(np.median(dur_all)),
            "censored_by_class": {c: {"censored": a, "of": b} for c, (a, b) in by_class.items()}}


# ----------------------------------------------------------------- windows ---

@dataclass(frozen=True)
class Window:
    """One punch's skeleton in BoxingVI space, or why there is none."""

    keypoints: "np.ndarray | None"   # (n_frames, 17, 3), validity in channel 2
    observed: int                    # frames of the window with any observed joint
    reason: str = ""                 # empty when keypoints is set


def _window(track: Track, want: np.ndarray, geom: WindowGeometry, torso_px: float,
            min_conf: float) -> Window:
    """Video frames `want`, in order, as one clip in BoxingVI's normalised 1920x1080
    space. A frame with no pose, and a joint the pose model did not detect at all
    (confidence 0) or detected below `min_conf`, become zeros with validity 0 --
    exactly how the corpus marks padding and dropped joints, so the classifier's
    input preparation drops them the same way. `min_conf` is 0 in the pipeline:
    see CLASSIFY_MIN_CONF for why blanking unsure joints is the wrong call."""
    if not (math.isfinite(torso_px) and torso_px > 1):
        raise ValueError(f"torso length must be a positive pixel length, got {torso_px!r}")
    n = len(want)
    pos = np.searchsorted(track.frame_ids, want)
    pos_c = np.clip(pos, 0, len(track.frame_ids) - 1)
    present = track.frame_ids[pos_c] == want

    xy = np.zeros((n, N_JOINTS, 2))
    ok = np.zeros((n, N_JOINTS), dtype=bool)
    xy[present] = track.xy[pos_c[present]]
    if track.conf is None:
        ok[present] = True
    else:
        c = track.conf[pos_c[present]]
        ok[present] = (c > 0) & (c >= min_conf)
    observed = int(ok.any(axis=1).sum())
    if observed < max(MIN_OBSERVED_FRAMES, math.ceil(MIN_OBSERVED_SHARE * n)):
        return Window(None, observed, f"pose in only {observed} of the window's {n} frames")

    hips = ok[:, L_HIP] & ok[:, R_HIP]
    if not hips.any():
        return Window(None, observed, "no frame in the window shows both hips")
    hip = np.median((xy[hips, L_HIP] + xy[hips, R_HIP]) / 2, axis=0)
    w, h = SOURCE_SIZE
    s = geom.torso_px / torso_px
    out = np.zeros((n, N_JOINTS, 3), dtype=np.float32)
    out[..., 0] = ((xy[..., 0] - hip[0]) * s + geom.hip_x * w) / w
    out[..., 1] = ((xy[..., 1] - hip[1]) * s + geom.hip_y * h) / h
    out[..., 2] = ok
    out[~ok] = 0.0
    seen = out[ok][:, :2]
    if seen.min() < COORD_LO or seen.max() > COORD_HI:
        return Window(None, observed, "joints fall far outside the frame after scaling")
    return Window(out, observed)


def _frame(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"{what} must be an integer frame, got {v!r}")
    return int(v)


def _nearest_frames(track: Track, t0: float, n: int, geom: WindowGeometry) -> np.ndarray:
    # Nearest frame, ties upward: np.round would send ties to the even frame, so
    # which frames a window holds would depend on the first frame's parity.
    return np.floor((t0 + np.arange(n) / geom.fps) * track.fps + 0.5).astype(np.int64)


def cut_window(track: Track, peak_frame: int, geom: WindowGeometry, torso_px: float,
               min_conf: float = CLASSIFY_MIN_CONF) -> Window:
    """A FIXED-length window (the training clips' median length and peak position)
    around the punch whose wrist-speed peak is at video frame `peak_frame`. The first
    design, kept as the baseline the per-clip window is measured against: on V6 it
    cost 21 points of accuracy, because that footage's punches run about twice the
    median training clip."""
    t0 = _frame(peak_frame, "peak frame") / track.fps - geom.lead_in_s
    return _window(track, _nearest_frames(track, t0, geom.n_frames, geom), geom, torso_px, min_conf)


def cut_span(track: Track, start: int, end: int, geom: WindowGeometry, torso_px: float,
             min_conf: float = CLASSIFY_MIN_CONF) -> Window:
    """The punch from video frame `start` to `end` (inclusive), as the corpus cuts
    a labelled span: every frame of it, resampled to the training frame rate, and
    trimmed to the corpus's 25 frames if longer (the corpus trims the tail too)."""
    start, end = _frame(start, "start"), _frame(end, "end")
    if end < start:
        raise ValueError(f"span ends ({end}) before it starts ({start})")
    # Steps of one training frame that fit in the span. The 0.05 lets a span at 24.000 fps be read whole
    # against a training rate of 23.976 (11 steps come to 10.989), without ever stepping past its last frame.
    n = min(MAX_FRAMES, int(math.floor((end - start) / track.fps * geom.fps + 0.05)) + 1)
    if n < MIN_OBSERVED_FRAMES:
        return Window(None, 0, "the punch is shorter than two frames at the training frame rate")
    return _window(track, _nearest_frames(track, start / track.fps, n, geom), geom, torso_px, min_conf)


# -------------------------------------------------------------- classifier ---

_HEX64 = re.compile(r"[0-9a-f]{64}")
_FILENAME = re.compile(r"[A-Za-z0-9._-]{1,128}")


def _no_constant(name):  # json: refuse NaN / Infinity rather than load them
    raise ValueError(f"manifest contains {name}, which is not valid JSON")


def write_manifest(path: Path, checkpoint: Path, geom: WindowGeometry, t_out: int,
                   extra: dict) -> dict:
    """The manifest train_classifier.py writes next to the checkpoint."""
    blob = Path(checkpoint).read_bytes()
    pinned = {"manifest_version", "classes", "t_out", "checkpoint", "size", "sha256", "geometry"}
    if pinned & set(extra):
        raise ValueError(f"extra fields may not override pinned ones: {sorted(pinned & set(extra))}")
    m = {"manifest_version": MANIFEST_VERSION, "classes": list(CLASSES), "t_out": int(t_out),
         "checkpoint": Path(checkpoint).name, "size": len(blob),
         "sha256": hashlib.sha256(blob).hexdigest(), "geometry": asdict(geom), **extra}
    Path(path).write_text(json.dumps(m, indent=1, allow_nan=False), encoding="utf-8")
    return m


def read_manifest(path: Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as f:
        raw = f.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError(f"{path.name}: over {MAX_MANIFEST_BYTES} bytes; not a classifier manifest")
    m = json.loads(raw.decode("utf-8"), parse_constant=_no_constant)
    if not isinstance(m, dict) or m.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"{path.name}: not a version-{MANIFEST_VERSION} classifier manifest")
    if m.get("classes") != list(CLASSES):
        raise ValueError(f"{path.name}: its classes {m.get('classes')!r} are not this code's {CLASSES}")
    t_out = m.get("t_out")
    if isinstance(t_out, bool) or not isinstance(t_out, int) or not 1 <= t_out <= 1000:
        raise ValueError(f"{path.name}: t_out must be an integer in 1..1000")
    name = m.get("checkpoint")
    # A bare file name in the manifest's own directory: no separators, no "..".
    if not isinstance(name, str) or not _FILENAME.fullmatch(name) or name in (".", ".."):
        raise ValueError(f"{path.name}: checkpoint must be a plain file name next to the manifest")
    size = m.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_CLASSIFIER_BYTES:
        raise ValueError(f"{path.name}: size must be a positive integer up to {MAX_CLASSIFIER_BYTES}")
    if not isinstance(m.get("sha256"), str) or not _HEX64.fullmatch(m["sha256"]):
        raise ValueError(f"{path.name}: sha256 must be 64 lowercase hex digits")
    g = m.get("geometry")
    if not isinstance(g, dict) or set(g) != set(WindowGeometry.__dataclass_fields__):
        raise ValueError(f"{path.name}: geometry fields are missing or unexpected")
    m["geometry"] = WindowGeometry(**g)
    m["checkpoint_path"] = path.parent / name
    return m


def load_classifier(manifest_path: Path):
    """(model in eval mode, manifest). Deny by default: every check below must pass."""
    import torch

    from .stgcnpp import STGCNPP

    m = read_manifest(manifest_path)
    p: Path = m["checkpoint_path"]
    if not p.is_file():
        raise FileNotFoundError(f"classifier checkpoint not found: {p.name}")
    if p.stat().st_size != m["size"]:
        raise ValueError(f"{p.name} is not the size its manifest pins")
    with p.open("rb") as f:   # hash and load the same bytes, and never more than pinned
        blob = f.read(m["size"] + 1)
    if len(blob) != m["size"] or hashlib.sha256(blob).hexdigest() != m["sha256"]:
        raise ValueError(f"{p.name} does not match the sha256 its manifest pins; not loading it")
    try:
        sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except Exception as e:  # never fall back to an unsafe load
        raise ValueError(f"classifier refused by the safe loader: {type(e).__name__}") from None
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise ValueError("classifier checkpoint is not a state_dict of tensors")
    model = STGCNPP(n_classes=N_CLASSES)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, m


def predict_proba(model, windows: "list[np.ndarray]", t_out: int) -> np.ndarray:
    """(N, n_classes) softmax probabilities for windows of shape (T, 17, 3). The
    windows may differ in length (a punch's extent does): each is padded to the
    corpus's 25 frames with empty frames, which the input preparation drops, so a
    window's own frames are all the classifier sees."""
    import torch

    from .stgcnpp import prepare_input

    if not windows:
        return np.zeros((0, N_CLASSES))
    if any(len(w) > MAX_FRAMES for w in windows):
        raise ValueError(f"a window is longer than the corpus's {MAX_FRAMES} frames")
    x = prepare_input(np.stack([pad_or_trim(np.asarray(w, dtype=np.float32)) for w in windows]), t_out)
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])
    return torch.softmax(logits, dim=1).numpy().astype(np.float64)


# -------------------------------------------------------------- hand check ---

def class_side(cls: str) -> str:
    """'lead' or 'rear': the hand a punch type is thrown with."""
    if cls in LEAD_CLASSES:
        return "lead"
    if cls in REAR_CLASSES:
        return "rear"
    raise ValueError(f"unknown punch type {cls!r}")


def hand_side(hand: str, stance: str) -> str:
    """The finder's COCO hand ('left' / 'right') as 'lead' / 'rear' for a stance."""
    if stance not in STANCES:
        raise ValueError(f"stance must be one of {sorted(STANCES)}")
    if hand not in ("left", "right"):
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    return "lead" if hand == STANCES[stance] else "rear"


def hand_agreement(type_sides: "list[str]", finder_sides: "list[str]") -> dict:
    """How often the classifier's punch type and the finder's hand name the same
    side, against the agreement two independent readings with the same marginals
    would reach by chance. Neither reading is ground truth."""
    if len(type_sides) != len(finder_sides):
        raise ValueError("two readings of different lengths")
    n = len(type_sides)
    if n == 0:
        return {"n": 0, "agree": None, "chance": None}
    a = np.array([t == "lead" for t in type_sides])
    b = np.array([f == "lead" for f in finder_sides])
    agree = float((a == b).mean())
    p, q = float(a.mean()), float(b.mean())
    return {"n": n, "agree": agree, "chance": p * q + (1 - p) * (1 - q)}


# ------------------------------------------------------------ extent rule ---

# What run_punch_extent.py's `beats_fixed_window` certifies, written into the rule file and
# required by read_extent. Change it whenever the pipeline's cut, or how the gate measures it, changes.
GATED_CUT = "per clip, median over the finder's detections, windows at its matched detections"

def max_span_s(geom: WindowGeometry) -> float:
    """The longest punch the classifier can be shown: the corpus's 25 frames at the training rate."""
    return (MAX_FRAMES - 1) / geom.fps


def punch_extent(track: Track, signal: np.ndarray, peak_frame: int, hand: int, rule: dict,
                 geom: WindowGeometry) -> Extent:
    """One punch's own extent: the rule's ratios times ITS peak width, capped at what
    the classifier can be shown. Not the pipeline's cut since 2026-10-13 (see
    `clip_cut`); kept because the evaluations still measure it against that cut."""
    return extent(track, signal, peak_frame, hand, rule["lead_widths"], rule["tail_widths"], rule["frac"],
                  max_s=max_span_s(geom))


# A clip's tempo is read from its punches. With fewer than this many, the median of their own
# extents is the per-punch design under another name (one punch: exactly that), so the window
# falls back to the training clips' median length and peak position: the old fixed window.
# Not measured: every clip scored so far has 41 punches or more.
MIN_CLIP_PUNCHES = 5


class ClipCut(NamedTuple):
    """How every punch in one clip is cut."""

    length_s: float     # first frame to last frame of the window, in seconds
    lead_share: float   # the share of it before the punch's wrist-speed peak
    n_used: int         # punches whose own extent went into the median (0 when it came from training)
    n_left_out: int     # punches left out of it: their speed peak ran into the edge of the hand's signal
    capped: bool        # the median was longer than the classifier can be shown, and was shortened
    source: str         # "clip", or "training" when the clip had too few punches to read a tempo from


def clip_cut(track: Track, signal: np.ndarray, peaks: "list[tuple[int, int]]", rule: dict,
             geom: WindowGeometry) -> "ClipCut | None":
    """The clip's punch length: the median of its punches' own extents, over `peaks`
    = [(peak frame, hand), ...]. Each extent is the rule's ratios times that punch's
    peak width, in exact seconds (not rounded to frames, and not capped).

    A punch whose speed peak was cut off by the edge of its hand's signal has an
    unknown extent and is left out, as long as the whole ones are a strict majority;
    otherwise all are used (a median that leans short is closer than a median of a
    minority). That rule is UNMEASURED: it left out 0, 0 and 1 punches on V5, V9 and
    V6. A peak whose hand has no speed there raises, as in `punchfind.peak_width`.

    With fewer than MIN_CLIP_PUNCHES punches the clip's tempo cannot be read, and the
    cut is the training clips' median window. None when the clip has no punch.
    """
    if not peaks:
        return None
    per_width = rule["lead_widths"] + rule["tail_widths"]
    whole, cut_off = [], []
    for peak, hand in peaks:
        w, censored = peak_width(track, signal, peak, hand, rule["frac"])
        (cut_off if censored else whole).append(w * per_width)
    if len(peaks) < MIN_CLIP_PUNCHES:
        return ClipCut(geom.duration_s, geom.lead_in_s / geom.duration_s, 0, 0, False, "training")
    use = whole if len(whole) > len(cut_off) else whole + cut_off
    length = float(np.median(use))
    cap = max_span_s(geom)
    return ClipCut(min(length, cap), rule["lead_widths"] / per_width, len(use),
                   len(peaks) - len(use), bool(length > cap), "clip")


def clip_window(peak_frame: int, cut: ClipCut, fps: float) -> "tuple[int, int]":
    """(start, end) inclusive video frames of the clip's window around one punch's
    peak, at the frame rate `fps` of the track the peak is a frame of. The start can
    fall before frame 0 near the beginning of a clip; the window cutter marks frames
    with no pose empty."""
    peak_frame = _frame(peak_frame, "peak frame")
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be positive, got {fps!r}")
    start = int(math.floor(peak_frame - cut.lead_share * cut.length_s * fps + 0.5))
    end = int(math.floor(peak_frame + (1 - cut.lead_share) * cut.length_s * fps + 0.5))
    while (end - start) / fps > cut.length_s + 1e-9 and end > peak_frame:   # rounding both ends outward
        end -= 1
    return start, max(end, start + 1)


def read_extent(path: Path, variant: str) -> dict:
    """The punch-extent rule run_punch_extent.py wrote: {frac, lead_widths,
    tail_widths}. Refused unless it is well formed, was measured with the same
    wrist-speed `variant` as the classifier's window, and its file records that
    THE PER-CLIP CUT, as the pipeline runs it, beat the old fixed window where it
    was tested. A file from before the per-clip cut carries the same
    `beats_fixed_window` flag for another cut (each punch's own extent), so the
    flag alone is not accepted: the file must also say which cut it gated."""
    path = Path(path)
    with path.open("rb") as f:
        raw = f.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError(f"{path.name}: too large to be a punch-extent rule")
    d = json.loads(raw.decode("utf-8"), parse_constant=_no_constant)
    if not isinstance(d, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    out = {}
    for k, lo, hi in (("frac", 0.0, 1.0), ("lead_widths", 0.0, 20.0), ("tail_widths", 0.0, 20.0)):
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ValueError(f"{path.name}: {k} must be a number in [{lo:g}, {hi:g}], got {v!r}")
        out[k] = float(v)
    if not (0 < out["frac"] < 1 and out["lead_widths"] + out["tail_widths"] > 0):
        raise ValueError(f"{path.name}: frac must be strictly inside (0, 1) and the extent non-empty")
    if d.get("variant") != variant:
        raise ValueError(f"{path.name}: measured with wrist-speed variant {d.get('variant')!r}, "
                         f"not the classifier window's {variant!r}")
    if d.get("gated_cut") != GATED_CUT:
        raise ValueError(f"{path.name}: gated another cut than the pipeline's ({d.get('gated_cut')!r}); "
                         "re-run run_punch_extent.py")
    if d.get("beats_fixed_window") is not True:
        raise ValueError(f"{path.name}: this extent rule did not beat the fixed window where it was tested")
    return out
