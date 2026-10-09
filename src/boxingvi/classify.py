"""Classify punches found in continuous pose: a window of skeleton around each
detected punch, in, ST-GCN++ punch type and confidence, out.

The classifier was trained on BoxingVI's pre-cut clips, so the window has to look
like one of those clips, in time and in space. Both are measured on the TRAINING
clips (`training_geometry`) and carried in the classifier's manifest, never tuned
on anything that is scored:

* time: a BoxingVI clip is the labelled span, a median of about 9 real frames
  (0.375 s) at the source's own 24-30 fps. A window is cut to the same duration,
  placed so the wrist-speed peak sits where it sits in a training clip, and
  sampled at the training frame rate -- footage at 60 fps is converted
  explicitly, not passed through at twice the frames per punch (O54).
* space: BoxingVI coordinates are pixels of a 1920x1080 frame divided by the
  frame size. Phone footage is portrait, and the boxer is nearer or farther. The
  window is scaled so the torso is the training median length in 1080p pixels,
  centred so the hips sit at the training median position, then divided by the
  same 1920x1080. Position and scale are thereby normalised; posture and motion
  are not touched.

Trust boundaries: the classifier checkpoint is a pickle, so it is opened only
with ``weights_only=True``, after its size and sha256 match the manifest. The
manifest is JSON written by train_classifier.py; it is validated field by field
as untrusted. Like the YOLO weights, an attacker who can write the weights
directory can rewrite both files -- keep that directory user-private.

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

import numpy as np

from .data import CLASSES, N_CLASSES, N_JOINTS, NATIVE_FPS, SOURCE_SIZE, Track
from .punchfind import (L_HIP, L_SHOULDER, MIN_CONF, R_HIP, R_SHOULDER, VARIANTS,
                        body_scale, wrist_signal)

MANIFEST_VERSION = 1
MAX_CLASSIFIER_BYTES = 64 * 1024 * 1024   # the real file is ~5.5 MB
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
        if self.n_frames < MIN_OBSERVED_FRAMES:
            raise ValueError("geometry window is shorter than two frames")

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
        lead.append((int(np.argmax(best)) + 0.5) / f)
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


# ----------------------------------------------------------------- windows ---

@dataclass(frozen=True)
class Window:
    """One punch's skeleton in BoxingVI space, or why there is none."""

    keypoints: "np.ndarray | None"   # (n_frames, 17, 3), validity in channel 2
    observed: int                    # frames of the window with any observed joint
    reason: str = ""                 # empty when keypoints is set


def cut_window(track: Track, peak_frame: int, geom: WindowGeometry, torso_px: float,
               min_conf: float = MIN_CONF) -> Window:
    """The window around a punch whose wrist-speed peak is at video frame
    `peak_frame`, resampled to the training frame rate and mapped into BoxingVI's
    normalised 1920x1080 space.

    A sample time with no pose frame, and a joint below `min_conf`, become zeros
    with validity 0 -- exactly how the corpus marks padding and dropped joints,
    so the classifier's input preparation drops them the same way.
    """
    if isinstance(peak_frame, bool) or not isinstance(peak_frame, (int, np.integer)):
        raise ValueError(f"peak frame must be an integer, got {peak_frame!r}")
    if not (math.isfinite(torso_px) and torso_px > 1):
        raise ValueError(f"torso length must be a positive pixel length, got {torso_px!r}")
    n = geom.n_frames
    t0 = int(peak_frame) / track.fps - geom.lead_in_s
    want = np.round((t0 + np.arange(n) / geom.fps) * track.fps).astype(np.int64)
    pos = np.searchsorted(track.frame_ids, want)
    pos_c = np.clip(pos, 0, len(track.frame_ids) - 1)
    present = track.frame_ids[pos_c] == want

    xy = np.zeros((n, N_JOINTS, 2))
    ok = np.zeros((n, N_JOINTS), dtype=bool)
    xy[present] = track.xy[pos_c[present]]
    if track.conf is None:
        ok[present] = True
    else:
        ok[present] = track.conf[pos_c[present]] >= min_conf
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


# -------------------------------------------------------------- classifier ---

_HEX64 = re.compile(r"[0-9a-f]{64}")
_FILENAME = re.compile(r"[A-Za-z0-9._-]{1,128}")


def _no_constant(name):  # json: refuse NaN / Infinity rather than load them
    raise ValueError(f"manifest contains {name}, which is not valid JSON")


def write_manifest(path: Path, checkpoint: Path, geom: WindowGeometry, t_out: int,
                   extra: dict) -> dict:
    """The manifest train_classifier.py writes next to the checkpoint."""
    blob = Path(checkpoint).read_bytes()
    m = {"manifest_version": MANIFEST_VERSION, "classes": list(CLASSES), "t_out": int(t_out),
         "checkpoint": Path(checkpoint).name, "size": len(blob),
         "sha256": hashlib.sha256(blob).hexdigest(), "geometry": asdict(geom), **extra}
    Path(path).write_text(json.dumps(m, indent=1, allow_nan=False), encoding="utf-8")
    return m


def read_manifest(path: Path) -> dict:
    path = Path(path).resolve()
    m = json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_constant)
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
    blob = p.read_bytes()   # hash and load the same bytes
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
    """(N, n_classes) softmax probabilities for windows of shape (T, 17, 3)."""
    import torch

    from .stgcnpp import prepare_input

    if not windows:
        return np.zeros((0, N_CLASSES))
    x = prepare_input(np.stack(windows).astype(np.float32), t_out)
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
